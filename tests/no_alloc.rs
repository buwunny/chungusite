//! The lifter's hot path must not touch the heap once its buffers are warm.
mod common;
use chungusite::{ir::Function, lift::Lifter};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { ALLOCS.fetch_add(1, Relaxed); unsafe { System.alloc(l) } }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { unsafe { System.dealloc(p, l) } }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn second_lift_allocates_nothing() {
    let code = common::sum_loop();
    let mut lifter = Lifter::new();
    let mut f = Function::with_capacity(64, 8);
    lifter.lift(&code, common::BASE, &mut f).unwrap(); // warm-up

    let before = ALLOCS.load(Relaxed);
    for _ in 0..1000 {
        lifter.lift(&code, common::BASE, &mut f).unwrap();
    }
    assert_eq!(ALLOCS.load(Relaxed) - before, 0);
}
