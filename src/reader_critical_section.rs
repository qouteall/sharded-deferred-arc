//! Critical section used by `AtomicSdark`
//!
//! when reading from `AtomicSdark` it firstly loads pointer then increment the reference count.
//!
//! But between loading pointer and incrementing ref count, the thread may be preempted, another thread could replace the pointer, then decrement original object's refcount so that sum becomes zero. If the first thread keeps not running for long time, the background collector could free the object. Then the first thread's incrementing of reference count will be use-after-free.
//!
//! So there is reader critical section. Before loading pointer, it increments critical section counter. After incrementing object reference count, it decrements critical section counter. The background thread will spin until all shards' critical section being 0 is observed once.
//!
//! It has some similarity to read-write lock, but with differences. The writer(background collector) can only spin until reader(critical section) count goes 0, but writer cannot acquire the lock. The reader never blocks.
//! 
//! Mutating `AtomicSdark` doesn't need to care about reader critical section. The collector cares about reader critical section.

use crate::shard_index::{ShardIndex, shard_indexes};
use crate::sharded_alloc::ShardedBox;
use log::warn;
use std::hint::spin_loop;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) struct ReaderCriticalSection {
    counters: ShardedBox<AtomicU64>,
}

pub(crate) static READER_CRITICAL_SECTION: LazyLock<ReaderCriticalSection> =
    LazyLock::new(|| ReaderCriticalSection::new());

impl ReaderCriticalSection {
    pub fn new() -> Self {
        Self {
            counters: ShardedBox::<AtomicU64>::allocate_data_in_each_shard(|_| AtomicU64::new(0)),
        }
    }

    /// See module-level doc for details.
    ///
    /// The `func` should finish quickly and should not block.
    pub fn reader_critical_section<R>(&self, func: impl FnOnce() -> R) -> R {
        let counter: &AtomicU64 = self.counters.at_curr_thread_shard();

        /// Why use Release ordering: 
        /// it should synchronize-with the Acquire ordering load in
        /// [`Self::spin_until_observing_non_critical_section_once_in_each_shard`]
        /// TODO check whether ordering is strong enough.
        counter.fetch_add(1, Ordering::Release);

        let _guard = scopeguard::guard((), |()| {
            /// The Release ordering synchronizes-with the Acquire ordering load in
            /// [`Self::spin_until_observing_non_critical_section_once_in_each_shard`]
            counter.fetch_sub(1, Ordering::Release);
        });

        func()
    }

    /// Spin until observing that each shard is not in critical section once.
    ///
    /// It's just for ensuring that no thread stuck in critical section to continue collection.
    /// 
    /// After this finishes, a reader thread could enter critical section in parallel with collection. 
    /// But it's ok, because the collector will wait until counter sum goes 0 and keeps being same across one iteration.
    pub fn spin_until_observing_non_critical_section_once_in_each_shard(&self) {
        let mut shards_to_spin: Vec<ShardIndex> = Vec::new();

        for shard_index in shard_indexes() {
            let counter: &AtomicU64 = &self.counters[shard_index];

            /// Why use Acquire ordering:
            /// Synchronize-with incrementing/decrementing in [`Self::reader_critical_section`]
            let counter_num = counter.load(Ordering::Acquire);
            if counter_num != 0 {
                shards_to_spin.push(shard_index);
            }
        }

        for shard_index in shards_to_spin {
            let counter: &AtomicU64 = &self.counters[shard_index];
            let mut spin_count: u64 = 0;

            // spin until it becomes zero
            'spin_loop: loop {
                // Why use Acquire ordering: same as the above
                let counter_num = counter.load(Ordering::Acquire);
                if counter_num == 0 {
                    break 'spin_loop;
                } else {
                    spin_loop();
                    spin_count += 1;

                    if spin_count == 100000 {
                        self.warn_about_too_long_spin(shard_index);
                    }
                }
            }
        }
    }

    fn warn_about_too_long_spin(&self, shard_index: ShardIndex) {
        let counters_for_logging: Vec<u64> = shard_indexes()
            .map(|shard_index| self.counters[shard_index].load(Ordering::Relaxed))
            .collect();

        warn!(
            "Critical section spins too much times on shard {:?}. Some possible causes: 1. a reader thread stuck too long time in critical section, 2. a reader thread was force-killed and didn't decrement counter, 3. other bugs. Current counters {:?}",
            shard_index, counters_for_logging
        );
    }
}
