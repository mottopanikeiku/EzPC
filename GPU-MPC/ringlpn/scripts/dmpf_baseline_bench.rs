// Benchmark adapter copied into the pinned CC0 MatanHamilis/dmpf tree by
// run_dmpf_baseline_comparison.sh. It measures the repository's public
// centralized DMPF implementations; it is not a distributed key generator.
use dmpf::big_state::BigStateDmpf;
use dmpf::okvs::OkvsDmpf;
use dmpf::rb_okvs::EpsilonPercent;
use dmpf::{Dmpf, DmpfKey, DpfDmpf, DpfOutput, PrimeField64x2};
use rand::{CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::env;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::Instant;

struct CountingAlloc;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

fn add_live(delta: isize) {
    let current = LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
    PEAK.fetch_max(current, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() { add_live(layout.size() as isize); }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() { add_live(layout.size() as isize); }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        add_live(-(layout.size() as isize));
    }
    unsafe fn realloc(&self, ptr: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, old, new_size);
        if !new_ptr.is_null() { add_live(new_size as isize - old.size() as isize); }
        new_ptr
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

trait RandomOutput {
    fn random_output<R: CryptoRng + RngCore>(rng: R) -> Self;
}

impl RandomOutput for PrimeField64x2 {
    fn random_output<R: CryptoRng + RngCore>(rng: R) -> Self {
        Self::random(rng)
    }
}

fn make_inputs<F: DpfOutput + RandomOutput>(log_domain: usize, points: usize, seed: u64) -> Vec<(u128, F)> {
    let domain = 1usize << log_domain;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut map = HashMap::with_capacity(points);
    while map.len() < points {
        let x = (rng.next_u64() as usize) % domain;
        let encoded = (x as u128) << (128 - log_domain);
        map.entry(encoded).or_insert_with(|| F::random_output(&mut rng));
    }
    let mut inputs: Vec<_> = map.into_iter().collect();
    inputs.sort_by_key(|(x, _)| *x);
    inputs
}

fn run<F, D>(name: &str, scheme: D, log_domain: usize, points: usize, seed: u64)
where
    F: DpfOutput + RandomOutput,
    D: Dmpf<F>,
{
    let inputs = make_inputs::<F>(log_domain, points, seed);
    let expected: HashMap<u128, F> = inputs.iter().copied().collect();
    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0x9e3779b97f4a7c15);
    let live_before = LIVE.load(Ordering::Relaxed);
    PEAK.store(live_before, Ordering::Relaxed);
    let start = Instant::now();
    let mut attempts = 0usize;
    let (key0, key1) = loop {
        attempts += 1;
        if let Some(keys) = scheme.try_gen(log_domain, &inputs, &mut rng) { break keys; }
    };
    let keygen_us = start.elapsed().as_secs_f64() * 1e6;
    let live_with_keys = LIVE.load(Ordering::Relaxed);
    let key_bytes_both = live_with_keys - live_before;
    let keygen_peak_bytes = PEAK.load(Ordering::Relaxed) - live_before;

    PEAK.store(live_with_keys, Ordering::Relaxed);
    let eval0_start = Instant::now();
    let output0 = key0.eval_all();
    let eval0_us = eval0_start.elapsed().as_secs_f64() * 1e6;
    let eval1_start = Instant::now();
    let output1 = key1.eval_all();
    let eval1_us = eval1_start.elapsed().as_secs_f64() * 1e6;
    let eval_us = eval0_us + eval1_us;
    let eval_peak_bytes = PEAK.load(Ordering::Relaxed) - live_with_keys;

    let mut correct = output0.len() == (1usize << log_domain) && output1.len() == output0.len();
    if correct {
        for i in 0..output0.len() {
            let encoded = (i as u128) << (128 - log_domain);
            let got = output0[i] + output1[i];
            match expected.get(&encoded) {
                Some(want) if got == *want => {},
                None if got.is_zero() => {},
                _ => { correct = false; break; }
            }
        }
    }
    println!("{},goldilocks_x2,{},{},{},{},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{},{}",
        name, std::mem::size_of::<F>() * 8, log_domain, points, seed,
        keygen_us, eval0_us, eval1_us, eval_us, attempts, key_bytes_both,
        keygen_peak_bytes, eval_peak_bytes,
        output0.len() * std::mem::size_of::<F>(),
        if correct { "pass" } else { "FAIL" });
    if !correct { std::process::exit(1); }
}

fn dispatch<F: DpfOutput + RandomOutput>(scheme: &str, log_domain: usize, points: usize, seed: u64) {
    match scheme {
        "dpf" => run::<F, _>("dpf", DpfDmpf, log_domain, points, seed),
        "big_state" => run::<F, _>("big_state", BigStateDmpf::new(8), log_domain, points, seed),
        "okvs" => run::<F, _>("okvs", OkvsDmpf::<1, 49, F>::new(EpsilonPercent::Hundred, 8), log_domain, points, seed),
        _ => panic!("unknown scheme: {scheme}"),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        eprintln!("usage: ringlpn_compare SCHEME LOG_DOMAIN POINTS FIELD_BITS SEED");
        std::process::exit(2);
    }
    let log_domain = args[2].parse().unwrap();
    let points = args[3].parse().unwrap();
    let field_bits: usize = args[4].parse().unwrap();
    let seed = args[5].parse().unwrap();
    println!("scheme,field,field_storage_bits,log_domain,points,seed,keygen_us,party0_full_eval_us,party1_full_eval_us,full_eval_serial_us,attempts,key_bytes_both,keygen_peak_extra_bytes,eval_peak_extra_bytes,output_bytes_per_party,validation");
    if field_bits != 128 {
        panic!("this public implementation exposes only its 128-bit two-field output");
    }
    dispatch::<PrimeField64x2>(&args[1], log_domain, points, seed);
}
