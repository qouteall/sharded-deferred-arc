//! Integrated stress test — a simulated cache server where everything is held
//! in `Sdarc` (not `std::sync::Arc`).  Tokio tasks recursively spawn child tasks
//! that clone the shared context, exercising Sdarc clone/drop under realistic
//! backend workloads.
//!
//! At the end we verify no memory leaks: a custom `TrackedDrop` type ensures
//! every allocation is freed, and the sharded allocator should have exactly
//! one slot still in use (the reader-critical-section counter).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use rusty_fork::rusty_fork_test;
use crate::collector::collector_update_now_and_wait;
// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn set_env(shard_count: usize, collector_interval_ms: u64) {
    unsafe {
        std::env::set_var("RUST_SDARC_SHARD_COUNT", shard_count.to_string());
        std::env::set_var("RUST_SDARC_COLLECTOR_INTERVAL_MS", collector_interval_ms.to_string());
    }
}

struct CheapRng(u64);
impl CheapRng {
    fn new(seed: u64) -> Self { Self(seed | 1) }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn usize(&mut self, max: usize) -> usize { (self.next() as usize) % max }
    fn bool(&mut self, pct: u8) -> bool { self.usize(100) < pct as usize }
}

const SCENARIO_DURATION: Duration = Duration::from_secs(10);

// ===========================================================================
// TrackedDrop — global counter for leak detection
// ===========================================================================

static TRACKED_ALLOC_COUNT: AtomicI64 = AtomicI64::new(0);

#[derive(Debug)]
struct TrackedDrop {
    id: u64,
    _pad: [u64; 4],
}

impl TrackedDrop {
    fn new(id: u64) -> Self {
        TRACKED_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        Self { id, _pad: [0; 4] }
    }
}

impl Clone for TrackedDrop {
    fn clone(&self) -> Self {
        TRACKED_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        Self { id: self.id, _pad: self._pad }
    }
}

impl Drop for TrackedDrop {
    fn drop(&mut self) {
        TRACKED_ALLOC_COUNT.fetch_sub(1, Ordering::Relaxed);
    }
}

fn tracked_alloc_count() -> i64 {
    TRACKED_ALLOC_COUNT.load(Ordering::Relaxed)
}

// ===========================================================================
// Cache entry
// ===========================================================================

#[derive(Clone, Debug)]
struct CacheEntry {
    #[allow(dead_code)]
    key: u64,
    #[allow(dead_code)]
    value: u64,
    generation: u64,
    payload: TrackedDrop,
}

impl CacheEntry {
    fn new(key: u64, value: u64, generation: u64) -> Self {
        Self { key, value, generation, payload: TrackedDrop::new(key) }
    }
}

// ===========================================================================
// SharedContext — the single Sdarc-held object everything shares
// ===========================================================================

use crate::sdarc::{AtomicSdarc, Sdarc, WeakSdarc};

struct SharedContext {
    /// The "cache database" — periodically swapped by updater tasks.
    atomic_cache: AtomicSdarc<HashMap<u64, CacheEntry>>,

    /// Pool of hot items for clone/drop contention.
    hot_pool: Vec<Sdarc<TrackedDrop>>,

    /// Metrics.
    ops: AtomicU64,
    upgrades_ok: AtomicU64,
    upgrades_fail: AtomicU64,
    swaps: AtomicU64,
    forwards: AtomicU64,
    spawns: AtomicU64,
    invariant_ok: AtomicBool,

    /// Stop signal.
    stop: AtomicBool,

    /// Max recursion depth for tokio task spawning.
    max_depth: u32,

    /// Number of std workers (for channel forwarding).
    worker_count: usize,

    /// Per-worker weak sets (for weak upgrade/downgrade cycles).
    weak_sets: Vec<Mutex<Vec<WeakSdarc<CacheEntry>>>>,
}

impl SharedContext {
    fn new(
        atomic_cache: AtomicSdarc<HashMap<u64, CacheEntry>>,
        hot_pool: Vec<Sdarc<TrackedDrop>>,
        max_depth: u32,
        worker_count: usize,
    ) -> Self {
        Self {
            atomic_cache,
            hot_pool,
            ops: AtomicU64::new(0),
            upgrades_ok: AtomicU64::new(0),
            upgrades_fail: AtomicU64::new(0),
            swaps: AtomicU64::new(0),
            forwards: AtomicU64::new(0),
            spawns: AtomicU64::new(0),
            invariant_ok: AtomicBool::new(true),
            stop: AtomicBool::new(false),
            max_depth,
            worker_count,
            weak_sets: (0..worker_count).map(|_| Mutex::new(Vec::new())).collect(),
        }
    }
}

// ===========================================================================
// Scenario
// ===========================================================================

fn scenario_integrated() {
    use crate::collector::collector_update_now;

    const STD_WORKERS: usize = 8;
    const MAX_DEPTH: u32 = 4;
    const HOT_POOL_SIZE: usize = 200;

    // ---- Build the hot pool ----
    let hot_pool: Vec<Sdarc<TrackedDrop>> = (0..HOT_POOL_SIZE)
        .map(|i| Sdarc::new(TrackedDrop::new(i as u64)))
        .collect();

    let initial_cache = AtomicSdarc::new(HashMap::new());

    // ---- Build shared context (in Sdarc, not Arc) ----
    let context = Sdarc::new(SharedContext::new(
        initial_cache,
        hot_pool,
        MAX_DEPTH,
        STD_WORKERS,
    ));

    // ---- Per-worker mpsc channels (outside Sdarc — Receiver is !Sync) ----
    let mut worker_receivers: Vec<mpsc::Receiver<Sdarc<CacheEntry>>> = vec![];
    let worker_senders: Vec<mpsc::Sender<Sdarc<CacheEntry>>> = (0..STD_WORKERS)
        .map(|_| {
            let (tx, rx) = mpsc::channel();
            worker_receivers.push(rx);
            tx
        })
        .collect();

    // =====================================================================
    // 1. Background cache updater (std thread)
    // =====================================================================
    let updater_ctx = context.clone(); // clone Sdarc
    let updater_handle = thread::spawn(move || {
        let mut rng = CheapRng::new(0xc0ffee);
        let mut generation: u64 = 0;
        while !updater_ctx.stop.load(Ordering::Relaxed) {
            generation += 1;
            let entry_count = 30 + rng.usize(40);
            let mut map = HashMap::with_capacity(entry_count);
            for _ in 0..entry_count {
                let k = rng.next();
                map.insert(k, CacheEntry::new(k, rng.next(), generation));
            }
            let _old = updater_ctx.atomic_cache.swap(Sdarc::new(map));
            updater_ctx.swaps.fetch_add(1, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(15 + rng.usize(60) as u64));
        }
    });

    // =====================================================================
    // 2. Request producers → push CacheEntry clones into worker inboxes
    // =====================================================================
    let producer_senders = worker_senders.clone();
    let producer_ctx = context.clone();
    let producers: Vec<thread::JoinHandle<()>> = (0..3).map(|p| {
        let senders = producer_senders.clone();
        let ctx = producer_ctx.clone();
        thread::spawn(move || {
            let mut rng = CheapRng::new((p + 100) as u64 * 0x9e3779b9);
            while !ctx.stop.load(Ordering::Relaxed) {
                let cache = ctx.atomic_cache.load();
                if !cache.is_empty() {
                    let keys: Vec<u64> = cache.keys().copied().collect();
                    let k = keys[rng.usize(keys.len())];
                    if let Some(entry) = cache.get(&k) {
                        let clone = Sdarc::new(entry.clone());
                        let dst = rng.usize(STD_WORKERS);
                        let _ = senders[dst].send(clone);
                        ctx.ops.fetch_add(1, Ordering::Relaxed);
                    }
                }
                drop(cache);
                thread::sleep(Duration::from_micros(50 + rng.usize(300) as u64));
            }
        })
    }).collect();

    // =====================================================================
    // 3. Std worker threads
    // =====================================================================
    let worker_barrier = Arc::new(Barrier::new(STD_WORKERS));
    // We use Arc for the Barrier only — it's std sync infrastructure, not user data.
    let worker_senders = worker_senders.clone();
    let workers: Vec<thread::JoinHandle<()>> = (0..STD_WORKERS).map(|w| {
        let rx = std::mem::replace(&mut worker_receivers[w], {
            let (_, rx) = mpsc::channel::<Sdarc<CacheEntry>>(); rx
        });
        let senders = worker_senders.clone();
        let ctx = context.clone();
        let barrier = Arc::clone(&worker_barrier);
        thread::spawn(move || {
            let mut rng = CheapRng::new((w + 1) as u64 * 0x7f4a7c15);
            let mut hand: Vec<Sdarc<CacheEntry>> = vec![];
            let mut hot_hand: Vec<Sdarc<TrackedDrop>> = vec![];
            let mut spawned: Vec<thread::JoinHandle<()>> = vec![];

            barrier.wait();
            while !ctx.stop.load(Ordering::Relaxed) {
                // Drain inbox.
                while let Ok(msg) = rx.try_recv() {
                    hand.push(msg);
                    ctx.ops.fetch_add(1, Ordering::Relaxed);
                }

                match rng.usize(8) {
                    0 => {
                        // Load cache, clone entry.
                        let cache = ctx.atomic_cache.load();
                        if !cache.is_empty() {
                            let keys: Vec<u64> = cache.keys().copied().collect();
                            let k = keys[rng.usize(keys.len())];
                            if let Some(entry) = cache.get(&k) {
                                hand.push(Sdarc::new(entry.clone()));
                            }
                        }
                        drop(cache);
                        ctx.ops.fetch_add(1, Ordering::Relaxed);
                    }
                    1 => {
                        let idx = rng.usize(ctx.hot_pool.len());
                        hot_hand.push(ctx.hot_pool[idx].clone());
                        ctx.ops.fetch_add(1, Ordering::Relaxed);
                    }
                    2 => { hand.clear(); hot_hand.clear(); }
                    3 => {
                        if let Some(entry) = hand.pop() {
                            let weak_ref = entry.downgrade();
                            ctx.weak_sets[w].lock().unwrap().push(weak_ref);
                        }
                    }
                    4 => {
                        let mut ws = ctx.weak_sets[w].lock().unwrap();
                        if let Some(idx) = ws.iter().position(|_| rng.bool(50)) {
                            let weak_ref = ws.swap_remove(idx);
                            drop(ws);
                            match weak_ref.upgrade() {
                                Some(u) => {
                                    assert!(u.generation > 0);
                                    ctx.upgrades_ok.fetch_add(1, Ordering::Relaxed);
                                    hand.push(u);
                                }
                                None => {
                                    ctx.upgrades_fail.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                    5 => {
                        if let Some(entry) = hand.last() {
                            let clone = entry.clone();
                            let dst = rng.usize(STD_WORKERS);
                            let _ = senders[dst].send(clone);
                            ctx.forwards.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    6 => {
                        // Spawn short-lived helper thread.
                        if spawned.len() < 12 {
                            let child_ctx = ctx.clone();
                            let child_senders = senders.clone();
                            let mut rng2 = CheapRng::new(rng.next());
                            let child = thread::spawn(move || {
                                let idx = rng2.usize(child_ctx.hot_pool.len());
                                let clone = child_ctx.hot_pool[idx].clone();
                                for _ in 0..rng2.usize(4) {
                                    let _c2 = clone.clone();
                                    child_ctx.ops.fetch_add(1, Ordering::Relaxed);
                                }
                                let _ = child_senders[rng2.usize(child_senders.len())].send(
                                    Sdarc::new(CacheEntry::new(
                                        rng2.next(), rng2.next(), 999,
                                    ))
                                );
                                child_ctx.forwards.fetch_add(1, Ordering::Relaxed);
                                drop(clone);
                            });
                            spawned.push(child);
                            ctx.spawns.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    7 => {
                        for entry in &hand {
                            if entry.generation == 0 {
                                ctx.invariant_ok.store(false, Ordering::Relaxed);
                            }
                        }
                    }
                    _ => unreachable!(),
                }

                spawned.retain(|h| !h.is_finished());
                if hand.len() > 64 { hand.truncate(32); }
                if hot_hand.len() > 48 { hot_hand.clear(); }

                if rng.bool(25) {
                    thread::sleep(Duration::from_micros(rng.usize(150) as u64));
                }
            }

            for h in spawned { let _ = h.join(); }
            drop(hand);
            drop(hot_hand);
        })
    }).collect();

    // =====================================================================
    // 4. Tokio async worker pool — with recursive spawn
    // =====================================================================
    let tokio_ctx = context.clone();
    let tokio_senders = worker_senders.clone();
    let leak_check_ready = Arc::new(AtomicBool::new(false));
    let leak_check_ready_tokio = Arc::clone(&leak_check_ready);

    let tokio_handle = thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_time()
            .build()
            .unwrap();

        rt.block_on(async {
            // Spawn root tokio tasks — each is a recursive worker.
            const ROOT_TASKS: usize = 10;
            let mut task_handles = vec![];

            for t in 0..ROOT_TASKS {
                let ctx = tokio_ctx.clone();
                let senders = tokio_senders.clone();
                task_handles.push(tokio::spawn(tokio_recursive_worker(
                    ctx, senders, t as u64, 0,
                )));
            }

            // Background tokio cache updater.
            let bg_ctx = tokio_ctx.clone();
            let bg_handle = tokio::spawn(async move {
                let mut rng = CheapRng::new(0xbeef);
                let mut generation: u64 = 0;
                while !bg_ctx.stop.load(Ordering::Relaxed) {
                    generation += 1;
                    let mut map = HashMap::new();
                    for _ in 0..20 + rng.usize(25) {
                        let k = rng.next();
                        map.insert(k, CacheEntry::new(k, rng.next(), generation));
                    }
                    let _old = bg_ctx.atomic_cache.swap(Sdarc::new(map));
                    bg_ctx.swaps.fetch_add(1, Ordering::Relaxed);
                    tokio::time::sleep(
                        Duration::from_millis(25 + rng.usize(55) as u64)
                    ).await;
                }
            });

            // Channel drainer.
            let drain_ctx = tokio_ctx.clone();
            let (drain_tx, mut drain_rx) = tokio::sync::mpsc::channel::<Sdarc<CacheEntry>>(128);
            let drain_handle = tokio::spawn(async move {
                while !drain_ctx.stop.load(Ordering::Relaxed) {
                    while let Ok(item) = drain_rx.try_recv() {
                        drain_ctx.ops.fetch_add(1, Ordering::Relaxed);
                        drop(item);
                    }
                    tokio::time::sleep(Duration::from_micros(100)).await;
                }
                while let Ok(item) = drain_rx.try_recv() { drop(item); }
            });

            // Let the storm run.
            tokio::time::sleep(SCENARIO_DURATION).await;
            tokio_ctx.stop.store(true, Ordering::Relaxed);

            // Wait for tasks.
            for h in task_handles { let _ = h.await; }
            let _ = bg_handle.await;
            drop(drain_tx);
            let _ = drain_handle.await;
        });

        // Drop the runtime — this drops all tokio thread-local state.
        drop(rt);
        leak_check_ready_tokio.store(true, Ordering::Relaxed);
    });

    // =====================================================================
    // 5. Watchdog
    // =====================================================================
    let wd_ctx = context.clone();
    let wd_handle = thread::spawn(move || {
        let start = Instant::now();
        while !wd_ctx.stop.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(2));
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!(
                "  [{elapsed:4.1}s] ops={} up(ok={} fail={}) swaps={} fwd={} spawns={} tracked={}",
                wd_ctx.ops.load(Ordering::Relaxed),
                wd_ctx.upgrades_ok.load(Ordering::Relaxed),
                wd_ctx.upgrades_fail.load(Ordering::Relaxed),
                wd_ctx.swaps.load(Ordering::Relaxed),
                wd_ctx.forwards.load(Ordering::Relaxed),
                wd_ctx.spawns.load(Ordering::Relaxed),
                tracked_alloc_count(),
            );
        }
    });

    // =====================================================================
    // Wait for everything to stop.
    // =====================================================================
    thread::sleep(SCENARIO_DURATION);
    context.stop.store(true, Ordering::Relaxed);

    // Join all std threads.
    for p in producers { p.join().unwrap(); }
    for w in workers { w.join().unwrap(); }
    updater_handle.join().unwrap();
    wd_handle.join().unwrap();
    tokio_handle.join().unwrap();

    eprintln!(
        "=== integrated done ===\n  ops={}  up(ok={} fail={})  swaps={}  fwd={}  spawns={}",
        context.ops.load(Ordering::Relaxed),
        context.upgrades_ok.load(Ordering::Relaxed),
        context.upgrades_fail.load(Ordering::Relaxed),
        context.swaps.load(Ordering::Relaxed),
        context.forwards.load(Ordering::Relaxed),
        context.spawns.load(Ordering::Relaxed),
    );

    assert!(context.invariant_ok.load(Ordering::Relaxed), "invariant violated");

    // =====================================================================
    // Memory leak check — drain everything, then let the collector catch up.
    // =====================================================================

    // 1. Drain worker channels BEFORE dropping context (they hold Sdarc<CacheEntry>).
    for rx in &mut worker_receivers {
        while let Ok(item) = rx.try_recv() { drop(item); }
    }
    drop(worker_senders);
    for rx in &mut worker_receivers {
        while let Ok(_) = rx.try_recv() {}
    }
    drop(worker_receivers);

    eprintln!("  after channel drain: tracked={}, slots={}",
        tracked_alloc_count(), crate::sharded_alloc::total_sharded_alloc_used_slots());

    // 2. Drop the shared context — this triggers the entire cleanup chain.
    drop(context);

    eprintln!("  after context drop: tracked={}, slots={}",
        tracked_alloc_count(), crate::sharded_alloc::total_sharded_alloc_used_slots());

    // 3. Let the collector run multiple iterations to free everything.
    //    Each Sdarc needs at least 2 collector passes (tag + confirm).
    //    The SharedContext itself needs 2 passes, then its contents need
    //    another 2+ passes.
    for iteration in 0..8 {
        collector_update_now_and_wait();
        let remaining = tracked_alloc_count();
        let used = crate::sharded_alloc::total_sharded_alloc_used_slots();
        eprintln!("  cleanup iter {iteration}: tracked={remaining}, shard_slots={used}");
        if remaining == 0 && used <= 2 {
            break;
        }
    }

    let remaining = tracked_alloc_count();
    let used_slots = crate::sharded_alloc::total_sharded_alloc_used_slots();

    // The critical-section counter always uses 1 slot.
    // There may also be 1 slot for the reader-critical-section's ShardedBox
    // (which wraps the AtomicU64 counters) — actually the ShardedBox itself
    // takes one slot per shard... no, ShardedBox is one slot total (the usage
    // flag + N shard data slots). The usage flag counts as 1 slot.
    assert_eq!(used_slots, 1,
        "sharded alloc leak: {used_slots} slots still used (expected 1 for critical-section counter)");
    assert_eq!(remaining, 0,
        "TrackedDrop leak: {remaining} instances still alive");
}

// ===========================================================================
// Recursive tokio worker
// ===========================================================================
//
/// A task that randomly: clones from the hot pool, loads the atomic cache,
/// upgrades weak refs, sends through channels, or **spawns a child task**
/// with a cloned `Sdarc<SharedContext>`.  Recursion depth is bounded.
///
/// Returns `impl Future + Send` so it can be passed to `tokio::spawn`.
fn tokio_recursive_worker(
    ctx: Sdarc<SharedContext>,
    senders: Vec<mpsc::Sender<Sdarc<CacheEntry>>>,
    id: u64,
    depth: u32,
) -> impl std::future::Future<Output = ()> + Send + 'static {
    async move {
    let mut rng = CheapRng::new(id.wrapping_mul(0x517cc1b727220a95));
    let mut hand: Vec<Sdarc<CacheEntry>> = vec![];
    let mut hot: Vec<Sdarc<TrackedDrop>> = vec![];
    let mut weak: Option<WeakSdarc<CacheEntry>> = None;
    let mut child_handles = vec![];

    while !ctx.stop.load(Ordering::Relaxed) {
        match rng.usize(8) {
            0 => {
                // Load from atomic cache.
                let cache = ctx.atomic_cache.load();
                if !cache.is_empty() {
                    let keys: Vec<u64> = cache.keys().copied().collect();
                    let k = keys[rng.usize(keys.len())];
                    if let Some(entry) = cache.get(&k) {
                        hand.push(Sdarc::new(entry.clone()));
                    }
                }
                drop(cache);
                ctx.ops.fetch_add(1, Ordering::Relaxed);
            }
            1 => {
                // Clone from hot pool.
                let idx = rng.usize(ctx.hot_pool.len());
                hot.push(ctx.hot_pool[idx].clone());
                ctx.ops.fetch_add(1, Ordering::Relaxed);
            }
            2 => {
                // Simulate async I/O.
                tokio::time::sleep(
                    Duration::from_micros(rng.usize(250) as u64)
                ).await;
            }
            3 => {
                // Downgrade → store weak ref.
                if let Some(entry) = hand.pop() {
                    weak = Some(entry.downgrade());
                }
            }
            4 => {
                // Try upgrade.
                if let Some(ref w) = weak {
                    match w.upgrade() {
                        Some(u) => {
                            if u.generation == 0 {
                                ctx.invariant_ok.store(false, Ordering::Relaxed);
                            }
                            ctx.upgrades_ok.fetch_add(1, Ordering::Relaxed);
                            hand.push(u);
                        }
                        None => {
                            ctx.upgrades_fail.fetch_add(1, Ordering::Relaxed);
                            weak = None;
                        }
                    }
                }
            }
            5 => {
                // Forward to a std worker via mpsc.
                if let Some(entry) = hand.pop() {
                    let dst = rng.usize(senders.len());
                    let _ = senders[dst].send(entry);
                    ctx.forwards.fetch_add(1, Ordering::Relaxed);
                }
            }
            6 => {
                // ** Recursive spawn ** — the key pattern.
                // Clone the Sdarc<SharedContext> and spawn a child task.
                if depth < ctx.max_depth && child_handles.len() < 8 && rng.bool(30) {
                    let child_ctx = ctx.clone();
                    let child_senders = senders.clone();
                    let child_id = rng.next();
                    let handle = tokio::spawn(tokio_recursive_worker(
                        child_ctx, child_senders, child_id, depth + 1,
                    ));
                    child_handles.push(handle);
                    ctx.spawns.fetch_add(1, Ordering::Relaxed);
                }
            }
            7 => {
                hand.clear();
                hot.clear();
                tokio::task::yield_now().await;
            }
            _ => unreachable!(),
        }

        if hand.len() > 32 { hand.clear(); }
        if hot.len() > 32 { hot.clear(); }

        // Prune finished children.
        child_handles.retain(|h| !h.is_finished());
    }

    // Wait for remaining children — they'll stop when they see stop flag.
    for child in child_handles {
        let _ = child.await;
    }
    drop(hand);
    drop(hot);
    }} // close async move block and fn body

// ===========================================================================
// Test definitions — 6 configs, half with collector interval = 0
// ===========================================================================
//
//   (1,   0)   — 1 shard,  fastest collector (tightest race window)
//   (8,   0)   — 8 shards, fastest collector
//   (128, 0)   — many shards, fastest collector
//   (1,   200) — 1 shard,  normal collector
//   (16,  200) — medium,   normal collector
//   (256, 500) — max shards, slow collector

rusty_fork_test! {
    #[test] fn integrated_shard1_int0()    { set_env(1,   0);   scenario_integrated(); }
    #[test] fn integrated_shard8_int0()    { set_env(8,   0);   scenario_integrated(); }
    #[test] fn integrated_shard128_int0()  { set_env(128, 0);   scenario_integrated(); }
    #[test] fn integrated_shard1_int200()  { set_env(1,   200); scenario_integrated(); }
    #[test] fn integrated_shard16_int200() { set_env(16,  200); scenario_integrated(); }
    #[test] fn integrated_shard256_int500(){ set_env(256, 500); scenario_integrated(); }
}
