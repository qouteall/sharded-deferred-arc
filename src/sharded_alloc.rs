//! Structure of sharded alloc
#![doc= include_str!("../docs/shard_alloc.drawio.svg")]

use crate::shard_index::{
    ShardIndex, ShardsArr, curr_thread_shard_index, get_shard_count, shard_indexes,
};
use crossbeam::utils::CachePadded;
use parking_lot::RwLock;
use scopeguard::guard_on_unwind;
use std::marker::PhantomData;
use std::ops::{Deref, Index, IndexMut, Not};
use std::ptr::{NonNull, drop_in_place};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// In mainstream platforms (X86-64 and ARM64), CachePadded use 128 alignment, which is 16 `usize`s.
const U64_COUNT_PER_SHARD: usize = 16;

pub(crate) struct AllocUnit {
    data_ptr: NonNull<u8>,
}

// Safety: the allocation uses usage map atomically. Allocating requires `Send + Sync`.
unsafe impl Send for AllocUnit {}
unsafe impl Sync for AllocUnit {}

impl AllocUnit {
    fn new() -> AllocUnit {
        let len_bytes = Self::data_len_in_bytes();

        let the_box: Box<[u8]> = vec![0u8; len_bytes].into_boxed_slice();
        let slice_ptr: *mut [u8] = Box::into_raw(the_box);
        let thin_ptr: *const u8 = unsafe { (*slice_ptr).as_mut_ptr() };

        // the usage flags will also be initialized as 0.

        AllocUnit {
            data_ptr: NonNull::new(thin_ptr as *mut u8).unwrap(),
        }
    }

    fn data_len_in_bytes() -> usize {
        // the added 1 is for usage flags. see the svg for structure
        U64_COUNT_PER_SHARD * (1 + get_shard_count().0 as usize) * 8
    }

    /// The `index_of_unit` will be used for deallocating.
    ///
    /// If it will never be deallocated, `index_of_unit` can be `usize::MAX`
    fn allocate_and_initialize<T: Send + Sync>(
        &self,
        init_func: impl Fn(ShardIndex) -> T,
    ) -> Option<ShardedDataPtr<T>> {
        if let Some(sharded_data_ptr) = self.allocate_without_initializing() {
            // write data slots
            for shard_index in shard_indexes() {
                let ele_ptr: NonNull<T> = sharded_data_ptr.ptr_at_shard(shard_index);

                let init_value: T = {
                    let _unwind_guard = guard_on_unwind((), |()| {
                        // the `init_func` can panic.
                        // when it panics, drop the already-written values and de-allocate
                        for shard_index_to_drop in 0..shard_index.as_8() {
                            let shard_index_to_drop =
                                ShardIndex::from_bounded_u8(shard_index_to_drop);
                            let ele_ptr_to_drop: NonNull<T> =
                                sharded_data_ptr.ptr_at_shard(shard_index_to_drop);
                            unsafe {
                                drop_in_place(ele_ptr_to_drop.as_ptr());
                            }
                        }

                        unsafe {
                            sharded_data_ptr.deallocate_without_dropping();
                        }
                    });

                    init_func(shard_index)
                };

                // Safety: ele_ptr is not dangling.
                unsafe { ele_ptr.write(init_value) };
            }
            Some(sharded_data_ptr)
        } else {
            None
        }
    }

    fn allocate_without_initializing<T: Send + Sync>(&self) -> Option<ShardedDataPtr<T>> {
        // offset_in_shard is in unit of u64

        let u64_ptr = self.data_ptr.cast::<u64>();

        for slot_index in 0..U64_COUNT_PER_SHARD {
            let offseted_ptr = unsafe { u64_ptr.offset(slot_index as isize) };
            let usage_atomic: &AtomicU64 = unsafe { offseted_ptr.cast::<AtomicU64>().as_ref() };

            /// The Acquire ordering synchronizes-with Release ordering in
            /// [`ShardedDataPtr::deallocate_without_dropping`]
            if usage_atomic.swap(1, Ordering::Acquire) == 0 {
                let sharded_data_ptr = ShardedDataPtr::new(offseted_ptr);
                return Some(sharded_data_ptr);
            }
        }

        None
    }

    /// Intentionally take mut self
    #[allow(clippy::wrong_self_convention)]
    fn is_any_slot_used(&mut self) -> bool {
        let u64_ptr = self.data_ptr.cast::<u64>();

        for slot_index in 0..U64_COUNT_PER_SHARD {
            let offseted_ptr = unsafe { u64_ptr.offset(slot_index as isize) };
            let usage_atomic: &AtomicU64 = unsafe { offseted_ptr.cast::<AtomicU64>().as_ref() };

            // Why use Relaxed ordering: this is called within write lock. The lock already establish ordering.
            if usage_atomic.load(Ordering::Relaxed) == 1 {
                return true;
            }
        }

        return false;
    }

    fn has_any_free_slot(&self) -> bool {
        let u64_ptr = self.data_ptr.cast::<u64>();

        for offset_in_shard in 0..U64_COUNT_PER_SHARD {
            let offseted_ptr = unsafe { u64_ptr.offset(offset_in_shard as isize) };
            let usage_atomic: &AtomicU64 = unsafe { offseted_ptr.cast::<AtomicU64>().as_ref() };

            // Why use Relaxed ordering: this is called within locking. The lock already establish ordering.
            if usage_atomic.load(Ordering::Relaxed) == 0 {
                return true;
            }
        }

        return false;
    }
}

impl Drop for AllocUnit {
    fn drop(&mut self) {
        assert!(self.is_any_slot_used().not());

        let len = Self::data_len_in_bytes();
        let slice_ptr: *mut [u8] = std::ptr::slice_from_raw_parts_mut(self.data_ptr.as_ptr(), len);

        // Safety: ownership ensures no dangling and no double free
        let the_box: Box<[u8]> = unsafe { Box::from_raw(slice_ptr) };
        drop(the_box);
    }
}

pub(crate) struct ShardOfShardAlloc {
    all_units: Vec<AllocUnit>,
    index_of_units_to_check_for_allocation: Vec<usize>,
}

impl ShardOfShardAlloc {
    fn new() -> ShardOfShardAlloc {
        ShardOfShardAlloc {
            all_units: Vec::new(),
            index_of_units_to_check_for_allocation: Vec::new(),
        }
    }

    /// It only requires read lock. It can set atomic usage flag to true, but cannot change the memory layout.
    fn allocate_from_existing_units<T: Send + Sync>(
        &self,
        init_func: impl Fn(ShardIndex) -> T,
    ) -> Option<ShardedDataPtr<T>> {
        for i in &self.index_of_units_to_check_for_allocation {
            let i = *i;
            let unit = &self.all_units[i];
            if let Some(p) = unit.allocate_and_initialize::<T>(&init_func) {
                return Some(p);
            }
        }

        None
    }

    fn allocate_using_new_unit<T: Send + Sync>(
        &mut self,
        init_func: impl Fn(ShardIndex) -> T,
    ) -> ShardedDataPtr<T> {
        let new_unit = AllocUnit::new();

        let ptr: ShardedDataPtr<T> = new_unit
            .allocate_and_initialize(init_func)
            .expect("New unit should not fail allocation");

        self.all_units.push(new_unit);
        self.index_of_units_to_check_for_allocation
            .push(self.all_units.len() - 1);
        ptr
    }

    fn do_maintenance(&mut self) {
        self.all_units.retain_mut(|unit| unit.is_any_slot_used());

        self.index_of_units_to_check_for_allocation.clear();

        for (i, unit) in self.all_units.iter().enumerate() {
            if unit.has_any_free_slot() {
                self.index_of_units_to_check_for_allocation.push(i);
            }
        }
    }
}

pub(crate) struct FullShardAlloc {
    shards: ShardsArr<CachePadded<RwLock<ShardOfShardAlloc>>>,
}

impl FullShardAlloc {
    fn initialize() -> FullShardAlloc {
        let shards =
            ShardsArr::new(|_shard_index| CachePadded::new(RwLock::new(ShardOfShardAlloc::new())));
        FullShardAlloc { shards }
    }

    fn allocate_and_init<T: Send + Sync>(
        &self,
        init_func: impl Fn(ShardIndex) -> T,
    ) -> ShardedDataPtr<T> {
        let shard_index = curr_thread_shard_index();
        let shard = &self.shards[shard_index];
        let lock: &RwLock<ShardOfShardAlloc> = shard.deref();

        // Firstly try to allocate under read lock. If failed, then allocate under write lock.
        {
            let g = lock.read();
            if let Some(p) = g.allocate_from_existing_units::<T>(&init_func) {
                return p;
            }
        }

        let mut g = lock.write();
        g.allocate_using_new_unit(&init_func)
    }

    pub(crate) fn do_maintenance(&self) {
        for shard_index in shard_indexes() {
            let shard = &self.shards[shard_index];
            let lock: &RwLock<ShardOfShardAlloc> = shard.deref();
            let mut guard = lock.write();
            guard.do_maintenance();
        }
    }
}

pub(crate) static FULL_SHARD_ALLOC: LazyLock<FullShardAlloc> =
    LazyLock::new(|| FullShardAlloc::initialize());

/// It represents pointer to a piece of data in same offset in every shard.
///
/// The data's size should be same as `u64`.
pub(crate) struct ShardedDataPtr<T> {
    base_ptr: NonNull<u8>,
    _phantom: PhantomData<*mut T>,
}

unsafe impl<T: Send> Send for ShardedDataPtr<T> {}
unsafe impl<T: Sync> Sync for ShardedDataPtr<T> {}

impl<T> Copy for ShardedDataPtr<T> {}

impl<T> Clone for ShardedDataPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> ShardedDataPtr<T> {
    fn new(base_ptr: NonNull<u64>) -> ShardedDataPtr<T> {
        const { assert!(size_of::<T>() <= size_of::<u64>()) }
        const { assert!(align_of::<T>() <= align_of::<u64>()) }

        ShardedDataPtr {
            base_ptr: base_ptr.cast::<u8>(),
            _phantom: PhantomData,
        }
    }

    /// Creating pointer is not unsafe. But using pointer is unsafe.
    pub(crate) fn ptr_at_shard(self, shard_index: ShardIndex) -> NonNull<T> {
        let offset: usize = U64_COUNT_PER_SHARD * (shard_index.as_usize() + 1);

        let u64_ptr: NonNull<u64> = self.base_ptr.cast::<u64>();
        // Safety: offset is within allocation
        let offseted: NonNull<u64> = unsafe { u64_ptr.offset(offset as isize) };

        offseted.cast::<T>()
    }

    /// Creating pointer is not unsafe. But using pointer is unsafe.
    pub(crate) fn ptr_at_curr_thread_shard(self) -> NonNull<T> {
        self.ptr_at_shard(curr_thread_shard_index())
    }

    fn usage_flag_ptr(self) -> NonNull<AtomicU64> {
        self.base_ptr.cast::<AtomicU64>()
    }

    /// Safety: Ensure pointer is not dangling before deallocating. And don't use it after deallocating it.
    pub(crate) unsafe fn deallocate_without_dropping(self) {
        // Safety: caller ensures not dangling. and usage flag is never converted to mutable reference until dropping.
        let usage_flag: &AtomicU64 = unsafe { self.usage_flag_ptr().as_ref() };

        // why use Release ordering: cannot use Relaxed because Relaxed allows delaying writes before deallocating to apply after deallocating.
        // the allocation sets flag using Acquire which synchronizes-with it. no need to use SeqCst
        let original_usage = usage_flag.swap(0, Ordering::Release);
        assert_eq!(
            original_usage, 1,
            "deallocated a slot whose usage flag was not set. free of dangling pointer"
        );

        // It will only set usage flag. Other allocator maintenance work is done by background thread.
    }
}

/// It owns the shard-allocated data (similar to Box).
///
/// The size of T is at most 8 bytes, due to how the allocator work.
///
/// The data of different shards will be in different cache lines.
pub struct ShardedBox<T>(ShardedDataPtr<T>);

impl<T: Send + Sync> ShardedBox<T> {
    pub fn allocate_data_in_each_shard(init_func: impl Fn(ShardIndex) -> T) -> ShardedBox<T> {
        const { assert!(size_of::<T>() <= size_of::<u64>()) }
        const { assert!(align_of::<T>() <= align_of::<u64>()) }

        let ptr = FULL_SHARD_ALLOC.allocate_and_init(init_func);
        Self(ptr)
    }

    /// Note: the current thread's shard index can be mutated by [`shard_index::set_current_thread_shard_index`]
    pub fn at_curr_thread_shard(&self) -> &T {
        unsafe { self.0.ptr_at_curr_thread_shard().as_ref() }
    }
}

impl<T> Drop for ShardedBox<T> {
    fn drop(&mut self) {
        for shard_index in shard_indexes() {
            let ptr = self.0.ptr_at_shard(shard_index);
            unsafe { drop_in_place(ptr.as_ptr()) };
        }

        unsafe { self.0.deallocate_without_dropping() };
    }
}

impl<T> Index<ShardIndex> for ShardedBox<T> {
    type Output = T;

    fn index(&self, index: ShardIndex) -> &Self::Output {
        unsafe { self.0.ptr_at_shard(index).as_ref() }
    }
}

impl<T> IndexMut<ShardIndex> for ShardedBox<T> {
    fn index_mut(&mut self, index: ShardIndex) -> &mut Self::Output {
        unsafe { self.0.ptr_at_shard(index).as_mut() }
    }
}
