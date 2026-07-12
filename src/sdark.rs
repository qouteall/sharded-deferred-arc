use crate::collector::on_new_sdark_allocated;
use crate::reader_critical_section::READER_CRITICAL_SECTION;
use crate::sharded_alloc::ShardedBox;
use std::any::type_name;
use std::fmt::{Debug, Formatter};
use std::mem;
use std::mem::offset_of;
use std::ops::Deref;
use std::ptr::{NonNull, null_mut};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, AtomicPtr, Ordering};

/// Sharded deferred atomic reference counting.
///
/// Its counters are sharded. Each clone or drop will only change the counter shard corresponding to current thread.
/// So it will have much fewer cache contention than std `Arc`.
///
/// When the counter sum goes 0, it's not immediately freed. It's freed by the background collector deferred.
pub struct Sdark<T> {
    inner_ptr: NonNull<SdarkInner<T>>,
}

impl<T: Send + Sync> Sdark<T> {
    pub fn new(value: T) -> Sdark<T> {
        let ptr: NonNull<SdarkInner<T>> = Box::leak(Box::new(SdarkInner::new(value))).into();
        on_new_sdark_allocated(SdarkInnerFatPtr {
            sdark_inner: SdarkInnerPtrErased::from_typed(ptr),
            vtable_ref: get_sdark_vtable_ref::<T>(),
        });
        Sdark { inner_ptr: ptr }
    }
}

impl<T> Sdark<T> {
    /// Creating a `Sdark` from raw pointer without incrementing reference count
    pub(crate) unsafe fn from_raw_ptr(ptr: NonNull<SdarkInner<T>>) -> Sdark<T> {
        Self { inner_ptr: ptr }
    }

    /// Consuming `Sdark` into raw pointer without decrementing reference count
    fn into_raw_ptr(self: Sdark<T>) -> NonNull<SdarkInner<T>> {
        let result = self.inner_ptr;
        // don't decrement reference count
        mem::forget(self);
        result
    }
}

impl<T> Deref for Sdark<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner_ref().data
    }
}

impl<T> Sdark<T> {
    fn inner_ref(&self) -> &SdarkInner<T> {
        // Safety: reference counting ensures it's not dangling.
        // And it's never mutably borrowed before dropping.
        // For non-Send-Sync types, the SdarkInner cannot be created.
        unsafe { self.inner_ptr.as_ref() }
    }
}

unsafe impl<T: Send> Send for Sdark<T> {}
unsafe impl<T: Sync> Sync for Sdark<T> {}

pub(crate) struct SdarkInner<T> {
    /// One counter shard can go negative. The sum of them matters.
    pub(crate) counters: ShardedBox<AtomicI64>,
    /// It will never be initialized if [`Sdark::downgrade`] is never called.
    pub(crate) weak_sdark_inner_ref: OnceLock<Sdark<WeakSdarkInner<T>>>,
    pub(crate) data: T,
}

impl<T: Send + Sync> SdarkInner<T> {
    fn new(value: T) -> SdarkInner<T> {
        let counters = ShardedBox::allocate_data_in_each_shard(|_| AtomicI64::new(0));

        /// initially current shard's counter is 1, other shards' counters are 0
        /// Why use Release ordering: synchronize-with background collector's
        /// reading of counter using Acquire ordering.
        counters
            .at_curr_thread_shard()
            .fetch_add(1, Ordering::Release);

        SdarkInner {
            counters,
            weak_sdark_inner_ref: OnceLock::new(),
            data: value,
        }
    }
}

impl<T> Clone for Sdark<T> {
    fn clone(&self) -> Self {
        // Why use Relaxed ordering: Similar to std `Arc`, it can only clone from an existing Sdark.
        // Incrementing late or early is fine.
        // Sending to another thread will be synchronized,
        // so that incrementing will be before it's observable by other threads.
        self.inner_ref()
            .counters
            .at_curr_thread_shard()
            .fetch_add(1, Ordering::Relaxed);

        Self {
            inner_ptr: self.inner_ptr,
        }
    }
}

impl<T> Drop for Sdark<T> {
    fn drop(&mut self) {
        // Why use Release ordering: the background collector will use Acquire which synchronizes-with it.
        self.inner_ref()
            .counters
            .at_curr_thread_shard()
            .fetch_sub(1, Ordering::Release);
        // Maybe it's ok to use Relaxed ordering. Because the background collector checks in deferred way.
        // If a memory region hasn't been mutated for long time, all writes are visible.
        // But Miri doesn't care about time length so using Relaxed will probably cause Miri error.
    }
}

// Erase type in vtable function signature
#[derive(Copy, Clone, Debug)]
pub(crate) struct SdarkInnerPtrErased(pub NonNull<u8>);

impl SdarkInnerPtrErased {
    pub fn from_typed<T>(r: NonNull<SdarkInner<T>>) -> Self {
        Self(r.cast())
    }

    /// Safety: must use the correct type. Only use within vtable function impl.
    pub fn into_typed<T>(self) -> NonNull<SdarkInner<T>> {
        self.0.cast()
    }
}

/// The vtable is needed because the collector need to handle dropping of different types.
pub(crate) struct SdarkVTable {
    /// Offset of [`SdarkInner::counters`] field.
    ///
    /// Rust compiler can reorder fields so it's not necessarily in beginning.
    pub(crate) offset_of_counter: usize,

    /// See [`clear_weak_backref_impl`]
    pub(crate) clear_weak_backref: fn(SdarkInnerPtrErased) -> ClearWeakBackRefResult,

    /// See [`drop_sdark_inner_impl`]
    pub(crate) drop_sdark_inner: fn(SdarkInnerPtrErased) -> (),

    pub(crate) get_type_name_for_debugging: fn() -> &'static str,
}

pub(crate) fn get_sdark_vtable_ref<T>() -> &'static SdarkVTable {
    &SdarkVTable {
        offset_of_counter: offset_of!(SdarkInner<T>, counters),
        clear_weak_backref: clear_weak_backref_impl::<T>,
        drop_sdark_inner: drop_sdark_inner_impl::<T>,
        get_type_name_for_debugging: get_type_name_for_debugging_impl::<T>,
    }
}

fn drop_sdark_inner_impl<T>(ptr: SdarkInnerPtrErased) {
    let p: NonNull<SdarkInner<T>> = ptr.into_typed::<T>();

    let _box = unsafe { Box::from_raw(p.as_ptr()) };
}

fn get_type_name_for_debugging_impl<T>() -> &'static str {
    type_name::<T>()
}

impl Debug for SdarkVTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SdarkVTable({})", (self.get_type_name_for_debugging)())
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct SdarkInnerFatPtr {
    pub sdark_inner: SdarkInnerPtrErased,
    pub vtable_ref: &'static SdarkVTable,
}

impl SdarkInnerFatPtr {
    pub fn get_counters(self) -> NonNull<ShardedBox<AtomicI64>> {
        unsafe {
            self.sdark_inner
                .0
                .offset(self.vtable_ref.offset_of_counter as isize)
                .cast::<ShardedBox<AtomicI64>>()
        }
    }

    /// See [`clear_weak_backref_impl`]
    pub fn clear_weak_back_ref(self) -> ClearWeakBackRefResult {
        (self.vtable_ref.clear_weak_backref)(self.sdark_inner)
    }

    /// See [`drop_sdark_inner_impl`]
    pub fn free(self) {
        (self.vtable_ref.drop_sdark_inner)(self.sdark_inner);
    }
}

unsafe impl Send for SdarkInnerFatPtr {}
unsafe impl Sync for SdarkInnerFatPtr {}

pub struct AtomicNullableSdark<T> {
    inner_ptr: AtomicPtr<SdarkInner<T>>,
}

unsafe impl<T: Send> Send for AtomicNullableSdark<T> {}
unsafe impl<T: Sync> Sync for AtomicNullableSdark<T> {}

impl<T: Send + Sync> AtomicNullableSdark<T> {
    pub fn new() -> Self {
        Self {
            inner_ptr: AtomicPtr::new(null_mut()),
        }
    }

    pub fn new_with_value(value: T) -> Self {
        let r = Self::new();
        r.set(Some(Sdark::new(value)));
        r
    }
}

impl<T> AtomicNullableSdark<T> {
    /// Load the atomic pointer. If not null, it will increment counter and give owned `Sdark<T>`.
    pub fn load(&self) -> Option<Sdark<T>> {
        // There is a chance thread A get stuck right after loading pointer but right before incrementing counter,
        // the thread B mutates atomic pointer and drop the original Sdark, then inner data freed by background collector,
        // then thread A resumes and then use-after-free.
        // The reader cirtical section avoids it. Background collector will only free if no thread is stuck in reader side critical section.
        READER_CRITICAL_SECTION.reader_critical_section(|| {
            /// Why use Acquire ordering: synchronize-with mutating of pointer in
            /// [`Self::set`]
            let ptr = self.inner_ptr.load(Ordering::Acquire);
            match NonNull::new(ptr) {
                None => None,
                Some(ptr) => {
                    let sdark = unsafe { Sdark::from_raw_ptr(ptr) };

                    // Increment counter
                    // Why use Release ordering but normal `clone` use Relaxed ordering:
                    // It's not determined that a Sdark of it exits at that time (writing is not blocked),
                    // so it should synchronize-with collector's reading of counter in Acquire ordering.
                    sdark
                        .inner_ref()
                        .counters
                        .at_curr_thread_shard()
                        .fetch_add(1, Ordering::Release);

                    Some(sdark)
                }
            }
        })
    }

    /// Set the atomic pointer and get the replaced one.
    pub fn set(&self, sdark: Option<Sdark<T>>) -> Option<Sdark<T>> {
        let new_ptr: *mut SdarkInner<T> = match sdark {
            None => null_mut(),
            Some(sdark) => sdark.into_raw_ptr().as_ptr(),
        };

        /// Why use Release ordering: synchronize-with [`Self::load`]'s loading pointer in Acquire ordering.
        /// The pointer reader should observe all mutations to the content pointed by `new_ptr`.
        let old_ptr = self.inner_ptr.swap(new_ptr, Ordering::Release);

        match NonNull::new(old_ptr) {
            None => None,
            Some(old_ptr) => Some(unsafe { Sdark::from_raw_ptr(old_ptr) }),
        }
    }
}

impl<T> Drop for AtomicNullableSdark<T> {
    fn drop(&mut self) {
        let _old = self.set(None);
    }
}

pub struct AtomicSdark<T>(AtomicNullableSdark<T>);

impl<T: Send + Sync> AtomicSdark<T> {
    pub fn new(value: T) -> Self {
        Self(AtomicNullableSdark::new_with_value(value))
    }

    /// Load the atomic pointer and give owned `Sdark<T>`.
    pub fn load(&self) -> Sdark<T> {
        self.0.load().unwrap()
    }

    /// Set the atomic pointer and get the replaced one.
    pub fn set(&self, new_sdark: Sdark<T>) -> Sdark<T> {
        self.0.set(Some(new_sdark)).unwrap()
    }
}

pub(crate) struct WeakSdarkInner<T> {
    /// There is a circular reference. `SdarkInner` has `Sdark<WeakSdarkInner>`, this references back.
    /// When initialized, it's not null.
    /// When the SdarkInner's strong counter sum reach zero and stay unchanged, this ptr will be set to null.
    /// Upgrade can only succeed if it's not null, and upgrade is under reader side critical section.
    ///
    /// Note: it's possible that a concurrent upgrade resurrects the Sdark. After resurrection, `Sdark` can still upgrade, but `WeakSdark` won't ever be able to upgrade.
    back_ref: AtomicPtr<SdarkInner<T>>,
}

unsafe impl<T: Send> Send for WeakSdarkInner<T> {}
unsafe impl<T: Sync> Sync for WeakSdarkInner<T> {}

impl<T> Drop for WeakSdarkInner<T> {
    fn drop(&mut self) {
        // use Relaxed ordering because it's just an assertion
        assert!(
            self.back_ref.load(Ordering::Relaxed).is_null(),
            "WeakSdarkInner's backref is not cleared"
        );
    }
}

/// The weak reference version of [`Sdark`].
///
/// The weak reference behavior is very different to std `Arc` and `Weak`.
/// When there is no strong reference of `Sdark`, the [`WeakSdark::upgrade`] may still succeed.
/// Then the dead `Sdark` will be resurrected.
///
/// Why have the weird resurrection mechanism, instead of ensuring that resurrection is not possible:
/// Avoiding resurrection requires [`WeakSdark::upgrade`] to ensure whether strong count sum is 0 instantly.
/// Without locking, it's not possible. We avoid locking of counters to improve scalability.
pub struct WeakSdark<T> {
    sdark_weak_inner: Sdark<WeakSdarkInner<T>>,
}

pub(crate) enum ClearWeakBackRefResult {
    WeakRefNotInvolved,
    WeakBackRefCleared,
    WeakBackRefWasAlreadyNull,
}

/// When this function is called, the strong count sum reaches 0.
/// But there may be weak references, and the weak references can still upgrade at the same time.
/// But the [`SdarkInner::weak_sdark_inner_ref`] will never be initialized if it was not initialized,
/// because it can only be initialized from strong reference, and strong reference doesn't exist
/// if no weak reference to it exits.
///
/// If [`SdarkInner::weak_sdark_inner_ref`] has been initialized, it will clear the backref.
/// After clearing, weak ref's upgrade will fail. And the backref will never become non-null again.
///
/// If the `Sdark` has never been downgraded, it will return [`ClearWeakBackRefResult::WeakRefNotInvolved`],
/// and the collector will free it once strong count sum reaches 0 and counters keeps being same across one iteration.
///
/// If the `Sdark` has been downgraded, and it's the first time that `clear_weak_backref_impl` get called for it,
/// then it will return [`ClearWeakBackRefResult::WeakBackRefCleared`],
/// and the collector will assume that it may resurrect, and will not free despite strong counter sum being 0 and not changing.
///
/// If the `Sdark` has been downgraded, then resurrected, then `clear_weak_backref_impl` may be called for it again.
/// In that case, the backref has already been cleared. No more upgrade is possible. The collector will free it
/// once strong count sum reaches 0 and counters keep being same across one iteration.
///
/// Note that if it dies then resurrects quickly, without the dead state being observed by collector, then this function won't be called at that time.
fn clear_weak_backref_impl<T>(ptr: SdarkInnerPtrErased) -> ClearWeakBackRefResult {
    let p: NonNull<SdarkInner<T>> = ptr.into_typed::<T>();

    let r: &SdarkInner<T> = unsafe { p.as_ref() };

    if let Some(inner) = r.weak_sdark_inner_ref.get() {
        /// reset the backref to null. the weak ref will no longer be able to upgrade.
        /// the clearing is one-way. after clearing, it cannot become non-null.
        /// Why use Release ordering: synchronize-with loading of pointer in [`WeakSdark::upgrade`],
        /// wait is it really useful TODO
        let swapped_ptr = inner.back_ref.swap(null_mut(), Ordering::Release);
        if swapped_ptr.is_null() {
            ClearWeakBackRefResult::WeakBackRefWasAlreadyNull
        } else {
            ClearWeakBackRefResult::WeakBackRefCleared
        }
    } else {
        // When `clear_weak_backref_impl` is called, the strong count reaches 0.
        // If at this time the `weak_sdark_inner_ref` is not initialized, it means there is no weak ref,
        // so upgrade cannot happen.
        ClearWeakBackRefResult::WeakRefNotInvolved
    }
}

impl<T: Send + Sync> Sdark<T> {
    pub fn downgrade(&self) -> WeakSdark<T> {
        let inner_ptr = self.inner_ptr;
        let inner = self.inner_ref();
        let r: &Sdark<WeakSdarkInner<T>> = inner.weak_sdark_inner_ref.get_or_init(|| {
            Sdark::new(WeakSdarkInner {
                back_ref: AtomicPtr::new(inner_ptr.as_ptr()),
            })
        });
        WeakSdark {
            sdark_weak_inner: r.clone(),
        }
    }
}

impl<T: Send + Sync> WeakSdark<T> {
    /// If the strong count sum never reaches 0, upgrade will succeed.
    ///
    /// If the strong count has reached 0, then it's not deterministic whether upgrade will succeed.
    ///
    /// Unlike std `Arc`, `Sdark` has resurrection mechanism.
    /// Even after strong count sum reach zero, upgrade may still succeed, then it will be resurrected.
    ///
    /// Even if there is strong reference, if it has undergone resurrection, its weak ref may not be able to upgrade.
    pub fn upgrade(&self) -> Option<Sdark<T>> {
        let weak_inner: &WeakSdarkInner<T> = self.sdark_weak_inner.deref();
        // Similar to loading from atomic Sdark, it may be stuck between loading pointer and incrementing counter,
        // so use reader side critical section.
        READER_CRITICAL_SECTION.reader_critical_section(|| {
            /// Why use Acquire ordering: TODO
            let back_ref_loaded = weak_inner.back_ref.load(Ordering::Acquire);

            match NonNull::new(back_ref_loaded) {
                None => {
                    // backref has been cleared, won't be able to upgrade
                    None
                }
                Some(sdark_inner) => {
                    let upgraded = unsafe { Sdark::from_raw_ptr(sdark_inner) };

                    // Unlike `clone`, use Release ordering instead of Relaxed ordering,
                    // because `Sdark` has resurrection mechanism.
                    // After strong count sum reach 0, weak ref upgrade may still succeed.
                    upgraded
                        .inner_ref()
                        .counters
                        .at_curr_thread_shard()
                        .fetch_add(1, Ordering::Release);

                    Some(upgraded)
                }
            }
        })
    }
}
