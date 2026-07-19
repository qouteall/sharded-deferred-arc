use crate::env_params::CollectorParams;
use crate::reader_critical_section::READER_CRITICAL_SECTION;
use crate::sdarc::{ClearWeakBackRefResult, SdarcInnerFatPtr};
use crate::shard_index::{ShardsArr, shard_indexes};
use crate::sharded_alloc::{FULL_SHARD_ALLOC, ShardedDataPtr};
use crate::tagged_counter::AtomicTaggedCounter;
use crossbeam::utils::CachePadded;
use log::{debug, error, warn};
use parking_lot::Mutex;
use std::cell::{OnceCell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, atomic};
use std::thread::JoinHandle;
use std::time::Instant;
use std::{mem, panic, thread};

pub(crate) struct CollectorShared {
    params: CollectorParams,
    thread_handle: JoinHandle<()>,
    /// Every time a new `Sdarc` is allocated, it's put into here.
    /// It's also sharded.
    ///
    /// Why not use [`sharded_alloc::ShardedBox`]: it can only hold 8 bytes per shard,
    /// but Vec is larger than that.
    pending_to_track: ShardsArr<CachePadded<Mutex<CollectorPendingDataShard>>>,

    /// The outer iteration runs in interval specified by [`CollectorParams::interval`]
    /// Within each outer iteration, there are inner iterations.
    /// These counters are just for logging, so use Relaxed ordering.
    outer_iteration_counter: AtomicU64,
    inner_iteration_counter: AtomicU64,
}

pub(crate) struct CollectorPendingDataShard {
    new_counters_to_track: Vec<(SdarcInnerFatPtr, ShardedDataPtr<AtomicTaggedCounter>)>,
}

impl CollectorPendingDataShard {
    pub fn new() -> CollectorPendingDataShard {
        Self {
            new_counters_to_track: Vec::new(),
        }
    }
}

impl CollectorShared {
    fn new(params: CollectorParams) -> Self {
        Self {
            params,
            thread_handle: thread::spawn(move || {
                let r = panic::catch_unwind(|| collector_thread_main());
                match r {
                    Ok(()) => {
                        error!("Collector main should not finish.")
                    }
                    Err(any_err) => {
                        let msg = if let Some(s) = any_err.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = any_err.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "non-string panic payload".to_string()
                        };
                        error!("Collector panicked {msg}");
                        eprintln!("Collector panicked {msg}");
                    }
                }
            }),
            // The CachePadded ensure the rwlock and vec's outer 3 fields (ptr, length and capacity) are in unique cache lines.
            // The 8 ensures initial inner spaces are in unique cache lines.
            pending_to_track: ShardsArr::new(|_| {
                CachePadded::new(Mutex::new(CollectorPendingDataShard::new()))
            }),
            outer_iteration_counter: AtomicU64::new(0),
            inner_iteration_counter: AtomicU64::new(0),
        }
    }

    fn on_new_sdarc_allocated(
        &self,
        fat_ptr: SdarcInnerFatPtr,
        counters_ptr: ShardedDataPtr<AtomicTaggedCounter>,
    ) {
        self.pending_to_track
            .at_curr_thread_shard()
            .lock()
            .new_counters_to_track
            .push((fat_ptr, counters_ptr));
    }
}

pub(crate) fn on_new_sdarc_allocated(
    fat_ptr: SdarcInnerFatPtr,
    counters_ptr: ShardedDataPtr<AtomicTaggedCounter>,
) {
    get_collector().on_new_sdarc_allocated(fat_ptr, counters_ptr);
}

static COLLECTOR: OnceLock<CollectorShared> = OnceLock::new();

fn get_collector() -> &'static CollectorShared {
    COLLECTOR.get_or_init(|| CollectorShared::new(CollectorParams::new_from_env_var()))
}

/// Interrupt the collector thread from parking.
///
/// Make collector quickly collect the objects whose reference count sum become zero.
pub fn collector_update_now() {
    /// Synchronizes-with the Acquire fence in collector,
    /// ensure curr thread's ref count decrement is visible to collector after unparking
    atomic::fence(Ordering::Release);
    get_collector().thread_handle.thread().unpark();
}

struct CollectorThreadState {
    collector: &'static CollectorShared,

    /// Use BTreeMap and the key is pointer to counters, not pointer to SdarcInner:
    /// The main thing that collector does is to read the counters.
    /// By using counter ptr as key, and using sorted BTreeMap,
    /// the cache locality of reading counters improves.
    tracked_counters: BTreeMap<ShardedDataPtr<AtomicTaggedCounter>, TrackedCounter>,
}

struct TrackedCounter {
    fat_ptr: SdarcInnerFatPtr,
    state: TrackedCounterState,
}

pub(crate) enum TrackedCounterState {
    DefaultState,
    RequiresReChecking,
    ReadyToFree,
}

impl TrackedCounter {
    fn new(fat_ptr: SdarcInnerFatPtr) -> Self {
        Self {
            fat_ptr,
            state: TrackedCounterState::DefaultState,
        }
    }

    fn update_state(&mut self, counters: ShardedDataPtr<AtomicTaggedCounter>) {
        let fat_ptr = self.fat_ptr;

        match self.state {
            TrackedCounterState::DefaultState => {
                let relaxed_sum = read_ref_count_sum_relaxed(counters);
                if relaxed_sum == 0 {
                    let sum = clear_tags_and_read_ref_count_sum_relaxed(counters);
                    if sum == 0 {
                        // counters tagged and observed sum is zero, going to re-check
                        self.state = TrackedCounterState::RequiresReChecking;
                    } else {
                        // The observed counter sum changed from 0 to nonzero.
                        // It's normal because there are race conditions.
                        // It stays in default state.
                        // There is side effect that counter tags are cleared. No need to re-set tags.
                        // Because after observing counter sum being 0 again tags will be re-cleared.
                        self.state = TrackedCounterState::DefaultState;
                    }
                } else {
                    // pre-check sum not 0, stay in default state
                    self.state = TrackedCounterState::DefaultState;
                }
            }
            TrackedCounterState::RequiresReChecking => {
                let opt_sum = read_ref_count_sum_if_all_tags_unset_acquire(counters);
                match opt_sum {
                    None => {
                        // Observed that some tag is set. There is counter decrement in between.
                        // Go back to default state.
                        self.state = TrackedCounterState::DefaultState;
                    }
                    Some(sum) => {
                        if sum == 0 {
                            // At here we are confident that strong count sum reaches zero.
                            // No tagged counter is observed.
                            // However, weak reference upgrade may happen in parallel.
                            // so clear the weak backref so upgrade can no longer happen.
                            match fat_ptr.clear_weak_back_ref() {
                                ClearWeakBackRefResult::WeakRefNotInvolved
                                | ClearWeakBackRefResult::WeakBackRefWasAlreadyNull => {
                                    // weak backref doesn't exist or was already cleared, ready to free
                                    self.state = TrackedCounterState::ReadyToFree;
                                }
                                ClearWeakBackRefResult::WeakBackRefCleared => {
                                    // Weak backref is cleared, upgrade can no longer happen,
                                    // but an upgrade may happen in parallel with clearing of weak backref
                                    // under reader critical section,
                                    // so re-check in next iteration of collection, which syncs
                                    // with reader critical section.
                                    self.state = TrackedCounterState::RequiresReChecking;
                                }
                            }
                        } else {
                            assert!(
                                sum > 0,
                                "In RequiresReChecking state, no tag is set, then counter sum should not be negative"
                            );

                            // No tag is set but counter sum is not zero
                            // Go back to default state.
                            self.state = TrackedCounterState::DefaultState;
                        }
                    }
                }
            }
            TrackedCounterState::ReadyToFree => {
                panic!("In state ReadyToFree, update_state should not be called")
            }
        }
    }
}

/// It uses Relaxed ordering to read counter sum.
/// Only when its result is 0 does counter sum be re-read using Acquire ordering.
///
/// In most cases, collector will see non-zero sum so this can improve collector performance.
fn read_ref_count_sum_relaxed(counters: ShardedDataPtr<AtomicTaggedCounter>) -> i64 {
    let mut sum: i64 = 0;

    for shard_index in shard_indexes() {
        let atomic_counter = unsafe { counters.ptr_at_shard(shard_index).as_ref() };

        // Why use Relaxed ordering: it's just a pre-check
        let tagged_counter = atomic_counter.load_relaxed();
        sum += tagged_counter.ref_count();
    }

    sum
}

fn clear_tags_and_read_ref_count_sum_relaxed(counters: ShardedDataPtr<AtomicTaggedCounter>) -> i64 {
    let mut sum: i64 = 0;

    for shard_index in shard_indexes() {
        let atomic_counter = unsafe { counters.ptr_at_shard(shard_index).as_ref() };

        /// Why use Relaxed ordering: the [`read_ref_count_sum_if_all_tags_unset_acquire`]
        /// during re-check ensures correctness.
        let tagged_counter = atomic_counter.fetch_and_clear_tag_relaxed();
        sum += tagged_counter.ref_count();
    }

    sum
}

/// If all tags are unset, returns Some containing counter sum
/// If one tag is set, return None
///
/// It uses Acquire ordering to read.
///
/// If there is decrement in parallel:
/// - If it observes decrement, then it observes tag being set, then collection will be delayed.
/// - If it doesn't observe decrement, it will observe counter sum higher than zero, so collection will still be delayed.
///
/// If there is increment in parallel:
/// - If the increment comes from existing strong reference, as increment happens-before decrement,
///   it cannot observe zero sum.
/// - If the increment comes from loading atomic pointer or weak ref upgrade,
///   reader critical section will ensure collector observes incremented counter.
fn read_ref_count_sum_if_all_tags_unset_acquire(
    counters: ShardedDataPtr<AtomicTaggedCounter>,
) -> Option<i64> {
    let mut sum: i64 = 0;

    for shard_index in shard_indexes() {
        let atomic_counter = unsafe { counters.ptr_at_shard(shard_index).as_ref() };

        // Why use Acquire ordering: see function doc
        let tagged_counter = atomic_counter.load_acquire();
        if tagged_counter.tag() {
            return None;
        }
        sum += tagged_counter.ref_count();
    }

    Some(sum)
}

impl CollectorThreadState {
    fn update(&mut self) {
        self.take_new_counters_to_track();

        self.do_collection_inner_iterations();

        FULL_SHARD_ALLOC.do_maintenance_by_collector();

        if log::log_enabled!(log::Level::Trace) {
            FULL_SHARD_ALLOC.log_status_in_trace_level();
        }
    }

    fn do_collection_inner_iterations(&mut self) {
        let inner_iter_counter = &self.collector.inner_iteration_counter;
        let inner_iter_count = inner_iter_counter.fetch_add(1, Ordering::Relaxed);

        debug!("Inner iteration {inner_iter_count} started");

        let mut to_re_check =
            self.update_all_tracked_counters_and_collect_and_get_ptrs_to_re_check();

        let mut loop_counter: u64 = 0;
        loop {
            if to_re_check.is_empty() {
                return;
            }

            let inner_iter_count = inner_iter_counter.fetch_add(1, Ordering::Relaxed);
            debug!("Inner iteration {inner_iter_count} started for re-checking");

            to_re_check =
                self.update_specific_counters_and_collect_and_get_ptrs_to_re_check(to_re_check);

            // Is it possible that the inner iteration loop deadloops?
            // The new to_re_check have elements in two cases:
            // 1. A `Sdarc` is dropped when dropping a `SdarcInner` in collector thread
            // 2. For a `SdarcInner` tracked by previous `to_re_check`, its counter sum is observed to be zero (then counter tags get cleared)
            // For 1: normally the children Sdarc will be dropped in parent drop. One object dropping can cause
            // other objects dropping. Normally there is finite objects to be dropped so it won't deadloop.
            // What if user thread keeps allocating new Sdarc in parallel? Then it will be buffered into pending
            // track list and won't be tracked in an inner iteration.
            // What if user type's `drop` creates a new Sdarc and drops? It will still be not tracked immediately.
            // For 2, it cannot have new element other than previous `to_re_check`.
            // What if the user type drop clones an existing living Sdarc then drop?
            // It will not cause deadloop because that Sdarc is living and the pre-check should be non-zero.
            // What about the race condition where [0, 1] becomes [1, 1] then [1, 0]?
            // It won't cause memory safety issue because tagged counter will be observed then it delays freeing.
            // Then its state will become DefaultState ad won't go into to_re_check.
            // TODO

            loop_counter += 1;
            if loop_counter == 100000 {
                warn!(
                    "Two many inner iterations in one outer iteration. It's either caused by user frees a very deep structure made of Sdarc, or there is a bug. Exiting the inner iteration loop. {to_re_check:?}"
                );
                break;
            }
        }
    }

    fn update_all_tracked_counters_and_collect_and_get_ptrs_to_re_check(
        &mut self,
    ) -> BTreeSet<ShardedDataPtr<AtomicTaggedCounter>> {
        // This is important
        READER_CRITICAL_SECTION.spin_until_observing_non_critical_section_once_in_each_shard();

        let mut to_free: Vec<ShardedDataPtr<AtomicTaggedCounter>> = Vec::new();
        let mut new_to_re_check: BTreeSet<ShardedDataPtr<AtomicTaggedCounter>> = BTreeSet::new();

        for (counters_ptr, tracked_counter) in &mut self.tracked_counters {
            tracked_counter.update_state(*counters_ptr);
            match &tracked_counter.state {
                TrackedCounterState::DefaultState => {}
                TrackedCounterState::RequiresReChecking => {
                    new_to_re_check.insert(*counters_ptr);
                }
                TrackedCounterState::ReadyToFree => {
                    to_free.push(*counters_ptr);
                }
            }
        }

        self.free_pointers(&mut to_free);

        let to_recheck_from_thread_local =
            COLLECTOR_THREAD_LOCAL.with(|cell| cell.get().unwrap().take_to_recheck());
        new_to_re_check.extend(to_recheck_from_thread_local);

        new_to_re_check
    }

    fn update_specific_counters_and_collect_and_get_ptrs_to_re_check(
        &mut self,
        old_to_recheck: BTreeSet<ShardedDataPtr<AtomicTaggedCounter>>,
    ) -> BTreeSet<ShardedDataPtr<AtomicTaggedCounter>> {
        // This is important
        READER_CRITICAL_SECTION.spin_until_observing_non_critical_section_once_in_each_shard();

        let mut to_free: Vec<ShardedDataPtr<AtomicTaggedCounter>> = Vec::new();
        let mut new_to_recheck: BTreeSet<ShardedDataPtr<AtomicTaggedCounter>> = BTreeSet::new();

        for counters_ptr in old_to_recheck {
            match self.tracked_counters.get_mut(&counters_ptr) {
                None => {
                    /// It's possible that user type's drop creates new Sdarc then drop,
                    /// in current outer iteration [`Self::take_new_counters_to_track`] won't be called,
                    /// so the counter stays untracked.
                    /// Just ignore it.
                    continue;
                }
                Some(tracked_counter) => {
                    tracked_counter.update_state(counters_ptr);
                    match &tracked_counter.state {
                        TrackedCounterState::DefaultState => {}
                        TrackedCounterState::RequiresReChecking => {
                            new_to_recheck.insert(counters_ptr);
                        }
                        TrackedCounterState::ReadyToFree => {
                            to_free.push(counters_ptr);
                        }
                    }
                }
            }
        }

        self.free_pointers(&mut to_free);

        let to_recheck_from_thread_local =
            COLLECTOR_THREAD_LOCAL.with(|cell| cell.get().unwrap().take_to_recheck());
        new_to_recheck.extend(to_recheck_from_thread_local);

        new_to_recheck
    }

    fn free_pointers(&mut self, to_free: &mut Vec<ShardedDataPtr<AtomicTaggedCounter>>) {
        for counters_ptr in to_free {
            let tracked_counter = self.tracked_counters.remove(&counters_ptr).unwrap();
            assert!(matches!(
                tracked_counter.state,
                TrackedCounterState::ReadyToFree
            ));

            let fat_ptr = tracked_counter.fat_ptr;

            assert_eq!(fat_ptr.get_counters_ptr(), *counters_ptr);

            let res = panic::catch_unwind(move || {
                fat_ptr.free();
            });

            if let Err(err) = res {
                error!("Error dropping Sdarc content {:?} {:?}", fat_ptr, err);
            }
        }
    }

    /// It uses locking. The locking ensures that when it starts tracking a `SdarcInner`,
    /// so the collector won't observe uninitialized counter or uninitialized pointee.
    fn take_new_counters_to_track(&mut self) {
        for shard_index in shard_indexes() {
            // Use empty container to replace it, minimize time of taking lock
            let taken = {
                let mut guard = self.collector.pending_to_track[shard_index].deref().lock();
                mem::replace(guard.deref_mut(), CollectorPendingDataShard::new())
            };

            for (fat_ptr, counters_ptr) in taken.new_counters_to_track {
                let replaced = self
                    .tracked_counters
                    .insert(counters_ptr, TrackedCounter::new(fat_ptr));
                assert!(replaced.is_none());
            }
        }
    }
}

fn collector_thread_main() {
    debug!("Collector thread started");

    let collector = get_collector();

    COLLECTOR_THREAD_LOCAL.with(|cell| {
        cell.set(CollectorThreadLocal::new()).unwrap();
    });

    let mut state: CollectorThreadState = CollectorThreadState {
        collector,
        tracked_counters: BTreeMap::new(),
    };

    loop {
        // This counter is just for logging, Relaxed ordering is fine
        let outer_iteration_counter = collector
            .outer_iteration_counter
            .fetch_add(1, Ordering::Relaxed);

        let iteration_start_time = Instant::now();

        state.update();

        let elapsed_time = iteration_start_time.elapsed();

        debug!("Collection outer iteration {outer_iteration_counter} took {elapsed_time:?}");

        let to_wait = collector.params.interval.saturating_sub(elapsed_time);

        debug!("Collector thread is going to wait {to_wait:?}");

        thread::park_timeout(to_wait);

        /// Synchronizes-with Release fence in [`collector_update_now`],
        /// ensure that reference count decrements before calling [`collector_update_now`] in caller thread
        /// is visible to collector.
        atomic::fence(Ordering::Acquire);
    }
}

#[derive(Debug)]
struct CollectorThreadLocal {
    /// Its purpose is to make collector collect deep structures faster.
    ///
    /// Without this, the collector observes that root node of deep structure reference count sum reach 0,
    /// then it takes two iteration to drop the root node, but the child nodes' ref count sum is not 0,
    /// because root node is not yet dropped, so it just drops root node. Then it takes 2 iterations to
    /// drop the second layer of nodes, then 2 iterations for third layer of nodes, etc.
    ///
    /// To solve that layer-by-layer dropping issue, we do special treatments for dropping in collector thread.
    /// In [`Sdarc::drop`] it uses thread local to see whether it's the collector thread. If is, then
    /// the ptr is added to this set. The collector then re-check this set and do immediate updates without waiting.
    /// In the between the collector goes through reader critical section to ensure safety.
    counters_to_recheck: RefCell<BTreeSet<ShardedDataPtr<AtomicTaggedCounter>>>,
}

/// It's only initialized in collector thread. Initialized in [`collector_thread_main`]
thread_local! {
    static COLLECTOR_THREAD_LOCAL: OnceCell<CollectorThreadLocal> = OnceCell::new();
}

impl CollectorThreadLocal {
    fn new() -> CollectorThreadLocal {
        CollectorThreadLocal {
            counters_to_recheck: RefCell::new(BTreeSet::new()),
        }
    }

    fn add_pending_to_check(&self, ptr: ShardedDataPtr<AtomicTaggedCounter>) {
        self.counters_to_recheck.borrow_mut().insert(ptr);
    }

    fn take_to_recheck(&self) -> BTreeSet<ShardedDataPtr<AtomicTaggedCounter>> {
        mem::take(self.counters_to_recheck.borrow_mut().deref_mut())
    }
}

pub(crate) fn on_sdarc_drop(counters_ptr: ShardedDataPtr<AtomicTaggedCounter>) {
    COLLECTOR_THREAD_LOCAL.with(|cell| {
        match cell.get() {
            None => {
                // This is not collector thread
            }
            Some(collector_thread_local) => {
                collector_thread_local.add_pending_to_check(counters_ptr)
            }
        }
    })
}
