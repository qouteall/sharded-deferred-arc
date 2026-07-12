use crate::shard_index::shard_indexes;
use crate::sharded_alloc::ShardedBox;
use parking_lot::lock_api::RwLockWriteGuard;
use parking_lot::{RawRwLock, RwLock};
use std::cell::UnsafeCell;
use std::mem;
use std::ops::{Deref, DerefMut};

/// Acquiring read lock only acquires lock of one shard. But acquiring write lock acquires all shards.
///
/// It makes read locking faster (lower contention) and write locking slower.
///
/// It's similar to crossbeam [`crossbeam_utils::sync::sharded_lock::ShardedLock`]. Re-implement because
/// 1. ensure shard index matches,
/// 2. allocate the lock shards in more efficient manner (less padding),
/// 3. use parking lot mutex instead of std mutex, which is usually faster and has no poison mechanism.
pub struct ShardedRwLock<T> {
    data: UnsafeCell<T>,
    locks: ShardedBox<RwLock<()>>,
}

#[allow(clippy::needless_lifetimes)]
impl<T: Send + Sync> ShardedRwLock<T> {
    pub fn new(value: T) -> Self {
        Self {
            data: UnsafeCell::new(value),
            locks: ShardedBox::allocate_data_in_each_shard(|_| RwLock::new(())),
        }
    }

    /// It only locks the shard corresponding to current thread. It's likely low-contention.
    pub fn read<'a>(&'a self) -> ReadGuardOfShardedRwLock<'a, T> {
        let lock_shard: &RwLock<()> = self.locks.at_curr_thread_shard();

        ReadGuardOfShardedRwLock {
            _raw_guard: lock_shard.read(),
            cell_ref: &self.data,
        }
    }

    pub fn write<'a>(&'a self) -> WriteGuardOfShardedRwLock<'a, T> {
        // this loop cannot panic as far as I know, so no need to consider half-locked state
        for shard_index in shard_indexes() {
            let lock_ref: &RwLock<()> = &self.locks[shard_index];
            let guard = lock_ref.write();

            // It's a hack to avoid allocating a Vec to hold the guards.
            // The parking_lot write guard only contains lock reference and phantom data.
            // The guard can be recovered from lock reference on drop.
            mem::forget(guard);
        }

        // Why not firstly try-lock all shards then wait for remaining shards:
        // Different locking order may cause deadlock.

        WriteGuardOfShardedRwLock { parent_lock: self }
    }
}

unsafe impl<T: Send + Sync> Send for ShardedRwLock<T> {}
unsafe impl<T: Send + Sync> Sync for ShardedRwLock<T> {}

pub struct ReadGuardOfShardedRwLock<'a, T> {
    _raw_guard: parking_lot::RwLockReadGuard<'a, ()>,
    cell_ref: &'a UnsafeCell<T>,
}

impl<'a, T> Deref for ReadGuardOfShardedRwLock<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.cell_ref.get().as_ref_unchecked() }
    }
}

pub struct WriteGuardOfShardedRwLock<'a, T> {
    parent_lock: &'a ShardedRwLock<T>,
}

impl<'a, T> Drop for WriteGuardOfShardedRwLock<'a, T> {
    fn drop(&mut self) {
        for shard_index in shard_indexes() {
            let lock_ref: &RwLock<()> = &self.parent_lock.locks[shard_index];

            // recover the guard and drop it (avoid allocating a Vec to hold the guards)
            let _guard: RwLockWriteGuard<RawRwLock, ()> =
                unsafe { lock_ref.make_write_guard_unchecked() };
        }
    }
}

impl<'a, T> Deref for WriteGuardOfShardedRwLock<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.parent_lock.data.get().as_ref_unchecked() }
    }
}

impl<'a, T> DerefMut for WriteGuardOfShardedRwLock<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.parent_lock.data.get().as_mut_unchecked() }
    }
}
