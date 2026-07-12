use std::cell::Cell;
use std::cmp::min;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::{Index, IndexMut};
use std::sync::LazyLock;
use std::thread;

/// Shard count can be at most 256
#[derive(Copy, Clone, Debug)]
pub struct ShardCount(pub u16);

impl ShardCount {
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

static SHARD_COUNT: LazyLock<ShardCount> = LazyLock::new(init_shard_count);
const MAX_SHARD_COUNT: usize = 256;

fn init_shard_count() -> ShardCount {
    let available_parallelism: usize = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let num = min(available_parallelism, MAX_SHARD_COUNT);

    ShardCount(num as u16)
}

/// The shard count won't change after initialization
pub(crate) fn get_shard_count() -> ShardCount {
    *SHARD_COUNT
}

/// It's u8 because shard count can be at most 256.
///
/// It's ensured that the number is smaller than shard size.
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct ShardIndex(u8);

impl ShardIndex {
    pub fn from_u64(value: u64) -> ShardIndex {
        let modulus: u64 = value % (get_shard_count().0 as u64);
        ShardIndex(modulus as u8)
    }

    pub fn as_8(self) -> u8 {
        self.0
    }

    pub fn from_bounded_u8(value: u8) -> Self {
        assert!((value as u64) < (get_shard_count().0 as u64));
        Self(value)
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

thread_local! {
    static CURR_THREAD_SHARD_INDEX: Cell<ShardIndex> = {
        Cell::new(shard_index_from_thread_id_hash())
    };
}

fn shard_index_from_thread_id_hash() -> ShardIndex {
    let thread_id = thread::current().id();
    let mut hasher = DefaultHasher::new();
    thread_id.hash(&mut hasher);
    let value: u64 = hasher.finish();

    ShardIndex::from_u64(value)
}

/// Note: current thread's shard index may be changed by [`set_current_thread_shard_index`]
pub fn curr_thread_shard_index() -> ShardIndex {
    CURR_THREAD_SHARD_INDEX.get()
}

/// The thread's shard index will by default be initialized using thread id hash.
///
/// But there may be collisions. Two different threads get same shard index.
///
/// For the threads that you manage, you can set the shard index in each thread to manually avoid collision.
pub fn set_current_thread_shard_index(shard_index: ShardIndex) {
    CURR_THREAD_SHARD_INDEX.replace(shard_index);
}

pub fn shard_indexes() -> impl Iterator<Item = ShardIndex> {
    (0..get_shard_count().0)
        .into_iter()
        .map(|i| ShardIndex(i as u8))
}

/// A helper type that wraps heap-allocated slice so that you can use ShardIndex as index. The user no longer need to convert ShardIndex to usize.
/// The elements will be contiguous in memory, unlike [`crate::sharded_alloc::ShardedBox`]
pub struct ShardsArr<T>(pub Box<[T]>);

impl<T> ShardsArr<T> {
    pub fn new(init_fn: impl Fn(ShardIndex) -> T) -> ShardsArr<T> {
        let shard_count = get_shard_count().as_usize();

        let mut vec: Vec<T> = Vec::with_capacity(shard_count);

        for shard_index in shard_indexes() {
            vec.push(init_fn(shard_index));
        }

        assert_eq!(vec.len(), shard_count);

        ShardsArr(vec.into_boxed_slice())
    }

    /// Note: the current thread's shard index can be mutated by [`set_current_thread_shard_index`]
    pub fn at_curr_thread_shard(&self) -> &T {
        &self.0[curr_thread_shard_index().as_usize()]
    }
}

impl<T> Index<ShardIndex> for ShardsArr<T> {
    type Output = T;

    fn index(&self, index: ShardIndex) -> &Self::Output {
        &self.0[index.as_usize()]
    }
}

impl<T> IndexMut<ShardIndex> for ShardsArr<T> {
    fn index_mut(&mut self, index: ShardIndex) -> &mut Self::Output {
        &mut self.0[index.as_usize()]
    }
}
