use crate::reader_critical_section::READER_CRITICAL_SECTION;
use crate::sdark::{ClearWeakBackRefResult, SdarkInnerFatPtr};
use crate::shard_index::{ShardsArr, shard_indexes};
use crate::sharded_alloc;
use crate::sharded_alloc::FULL_SHARD_ALLOC;
use crossbeam::utils::CachePadded;
use log::{debug, error};
use parking_lot::Mutex;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Deref;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use std::{panic, thread};

#[derive(Debug, Clone)]
pub struct CollectorParams {
    pub cycle_duration: Duration,
}

pub(crate) struct CollectorShared {
    params: CollectorParams,
    thread_handle: JoinHandle<()>,
    /// Every time a new `Sdark` is allocated, it's put into here.
    /// It's also sharded.
    ///
    /// Why not use [`sharded_alloc::ShardedBox`]: it can only hold 8 bytes per shard,
    /// but Vec is larger than that.
    pending_to_track: ShardsArr<CachePadded<Mutex<Vec<SdarkInnerFatPtr>>>>,
    collection_iteration_counter: AtomicU64,
}

impl CollectorShared {
    fn new(params: CollectorParams) -> Self {
        Self {
            params,
            thread_handle: thread::spawn(move || collector_thread_main()),
            // The CachePadded ensure the rwlock and vec's outer 3 fields (ptr, length and capacity) are in unique cache lines.
            // The 8 ensures initial inner spaces are in unique cache lines.
            pending_to_track: ShardsArr::new(|_| {
                CachePadded::new(Mutex::new(Vec::with_capacity(8)))
            }),
            collection_iteration_counter: AtomicU64::new(0),
        }
    }

    fn on_new_sdark_allocated(&self, tc: SdarkInnerFatPtr) {
        self.pending_to_track.at_curr_thread_shard().lock().push(tc);
    }
}

pub(crate) fn on_new_sdark_allocated(fat_ptr: SdarkInnerFatPtr) {
    get_collector().on_new_sdark_allocated(fat_ptr);
}

static COLLECTOR: OnceLock<CollectorShared> = OnceLock::new();

static DEFAULT_PARAM: CollectorParams = CollectorParams {
    cycle_duration: Duration::from_millis(500),
};

/// Return false if the collector have already been initialized. The collector can only be initialized once.
pub fn try_init_collector(params: CollectorParams) -> bool {
    let mut is_new_init = false;
    COLLECTOR.get_or_init(|| {
        is_new_init = true;
        CollectorShared::new(params)
    });
    is_new_init
}

fn get_collector() -> &'static CollectorShared {
    COLLECTOR.get_or_init(|| CollectorShared::new(DEFAULT_PARAM.clone()))
}

/// Interrupt the collector thread from parking.
///
/// Note that this function doesn't ensure early dropping of data when reference count sum goes 0.
pub fn collector_update_now() {
    get_collector().thread_handle.thread().unpark();
}

struct CollectorThreadState {
    collector: &'static CollectorShared,
    tracked_counters: Vec<TrackedCounter>,
}

struct TrackedCounter {
    sdark_fat_ptr: SdarkInnerFatPtr,
    state: TrackedCounterState,
}

enum TrackedCounterState {
    CounterSumMayBeNotZero,
    /// To compare whether previous counters equal current counters, we use hash comparison to avoid allocation.
    /// It uses [`DefaultHasher`] which is `SipHasher13`, which is collision-resistant enough.
    ObservedCounterSumBeingZeroInOneIteration {
        counter_hash: u64,
    },
    ReadyToFree,
}

impl TrackedCounter {
    fn new(sdark_erased_info: SdarkInnerFatPtr) -> Self {
        Self {
            sdark_fat_ptr: sdark_erased_info,
            state: TrackedCounterState::CounterSumMayBeNotZero,
        }
    }

    fn update_state(&mut self) {
        match self.state {
            TrackedCounterState::CounterSumMayBeNotZero => {
                let sum = read_counter_sum(self.sdark_fat_ptr);
                if sum == 0 {
                    let (new_sum, curr_hash) =
                        read_counter_sum_and_compute_hash(self.sdark_fat_ptr);
                    if new_sum == 0 {
                        self.state =
                            TrackedCounterState::ObservedCounterSumBeingZeroInOneIteration {
                                counter_hash: curr_hash,
                            };
                    }
                }
            }
            TrackedCounterState::ObservedCounterSumBeingZeroInOneIteration { counter_hash } => {
                let (new_sum, curr_hash) = read_counter_sum_and_compute_hash(self.sdark_fat_ptr);
                if new_sum == 0 && curr_hash == counter_hash {
                    match self.sdark_fat_ptr.clear_weak_back_ref() {
                        ClearWeakBackRefResult::WeakRefNotInvolved
                        | ClearWeakBackRefResult::WeakBackRefWasAlreadyNull => {
                            self.state = TrackedCounterState::ReadyToFree;
                        }
                        ClearWeakBackRefResult::WeakBackRefCleared => {
                            // re-check counter because upgrade may happen in parallel.
                            // but after the weak back ref has been cleared, upgrade can no longer happen.
                            self.state = TrackedCounterState::CounterSumMayBeNotZero;
                        }
                    }
                } else {
                    self.state = TrackedCounterState::CounterSumMayBeNotZero;
                }
            }
            TrackedCounterState::ReadyToFree => {
                // No need to update
            }
        }
    }
}

fn read_counter_sum(fat_ptr: SdarkInnerFatPtr) -> i64 {
    let mut sum: i64 = 0;

    let counters = unsafe { fat_ptr.get_counters().as_ref() };

    for shard_index in shard_indexes() {
        // Why use Acquire ordering: the decrementing of counter use Release.
        // It synchronizes-with decrementing of counter which use Release.
        let counter = counters[shard_index].load(Ordering::Acquire);
        sum += counter;
    }

    sum
}

/// Computing hash is semi-expensive. Only compute hash when counter sum is likely zero.
fn read_counter_sum_and_compute_hash(fat_ptr: SdarkInnerFatPtr) -> (i64, u64) {
    let mut sum: i64 = 0;
    let mut hasher = DefaultHasher::new();

    let counters = unsafe { fat_ptr.get_counters().as_ref() };

    for shard_index in shard_indexes() {
        // Why use Acquire ordering: see `read_counter_sum`
        let counter = counters[shard_index].load(Ordering::Acquire);
        sum += counter;
        counter.hash(&mut hasher);
    }

    (sum, hasher.finish())
}

impl CollectorThreadState {
    fn update(&mut self) {
        self.take_new_counters_to_track();

        // This is important
        READER_CRITICAL_SECTION.spin_until_observing_non_critical_section_once_in_each_shard();

        self.update_tracked_counters();

        FULL_SHARD_ALLOC.do_maintenance();
    }

    fn update_tracked_counters(&mut self) {
        for tracked_counter in &mut self.tracked_counters {
            tracked_counter.update_state();
        }

        let mut to_free: Vec<SdarkInnerFatPtr> = Vec::new();

        self.tracked_counters
            .retain(|tracked_counter| match &tracked_counter.state {
                TrackedCounterState::CounterSumMayBeNotZero => true,
                TrackedCounterState::ObservedCounterSumBeingZeroInOneIteration { .. } => true,
                TrackedCounterState::ReadyToFree => {
                    to_free.push(tracked_counter.sdark_fat_ptr);
                    false
                }
            });

        for fat_ptr in to_free {
            let res = panic::catch_unwind(move || {
                fat_ptr.free();
            });

            if let Err(err) = res {
                error!("Error dropping Sdarc content {:?} {:?}", fat_ptr, err);
            }
        }
    }

    fn take_new_counters_to_track(&mut self) {
        for shard_index in shard_indexes() {
            let mut guard = self.collector.pending_to_track[shard_index].deref().lock();
            self.tracked_counters
                .extend(guard.drain(0..).map(|info| TrackedCounter::new(info)));
        }
    }
}

fn collector_thread_main() {
    debug!("Collector thread started");

    let collector = get_collector();

    let mut state: CollectorThreadState = CollectorThreadState {
        collector,
        tracked_counters: Vec::new(),
    };

    loop {
        // This counter is just for logging, Relaxed ordering is fine
        let iteration_counter = collector
            .collection_iteration_counter
            .fetch_add(1, Ordering::Relaxed);

        let iteration_start_time = Instant::now();

        state.update();

        let elapsed_time = iteration_start_time.elapsed();

        debug!("Collection iteration {iteration_counter} took {elapsed_time:?}");

        let to_wait = collector.params.cycle_duration.saturating_sub(elapsed_time);

        debug!("Collector thread is going to wait {to_wait:?}");

        thread::park_timeout(to_wait);
    }
}
