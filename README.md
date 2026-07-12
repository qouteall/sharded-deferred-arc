# Sharded-Deferred-Arc

`Arc` is commonly used in Rust. But when many threads increment/decrement same atomic counter, cache contention may hurt performance.

Examples:

- [The Concurrency Trap: How An Atomic Counter Stalled A Pipeline](https://www.conviva.ai/resource/the-concurrency-trap-how-an-atomic-counter-stalled-a-pipeline/)
- [How a Single Line of Code Made a 24-core Server Slower Than a Laptop](https://pkolaczk.github.io/server-slower-than-a-laptop/)

This library provides sharded-deferred-atomic-reference-counting (sdark). A thread increment/decrement the counter shard according to thread id hash. One counter shard can become negative. There is a background thread periodically observing the counters and do freeing. This reduces contention of incrementing/decrementing counter.

The background thread will free when counters sum to 0 and counters don't change for one period. (Memory order issue may cause background thread to see wrong counter sum. But when there is no reference to it, the counters will stay same for some time then it's safe to free.)

It doesn't make each counter shard exclusively occupy cache line. The different counters in same shard can be put together. This library provides sharded allocation functionality. This library also supports sharded RwLock (reader acquire one sharded lock, writer acquire all locks, readers have low contention with each other).

Its weak reference behavior is different to std `Arc`. Because that reclamation is deferred, upgrade from weak ref to strong ref can happen when strong counter sum is 0. The `Sdark` can be "resurrected". After resurrection, the upgrading from weak ref to strong ref may fail or not fail.

Because the dropping is deferred, a deep structure may require long time to be fully dropped.

Don't use this library if:

- There won't be many parallel threads incrementing/decrementing the atomic counter of `Arc`.
- You want it to drop content immediately when strong reference count goes 0.
- In no_std environments.

