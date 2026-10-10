//! SSE lane operations in the emitted Rust.
//!
//! The lifter keeps an xmm register as two 64-bit halves, so a vector
//! instruction becomes one `BinOp::Lane` / `UnOp::Lane` per half. The emitter
//! prints each as a call into a `simd` module (`name`), which `PRELUDE` defines
//! once per output file when some function uses it (`uses`): `simd::b8::cmpeq(x, y)`
//! compares the eight bytes of `x` and `y`, `simd::f64::add` adds two `f64` bit
//! patterns, `simd::cvt::f64_from_i32` converts.
use crate::ir::{BinOp, Function, InstKind, LaneOp, LaneUn, UnOp};

/// The function the emitter calls for a two-operand lane op on `w`-byte lanes.
pub fn name(op: LaneOp, w: u8) -> String {
    use LaneOp::*;
    if let F80ToF64 | Fist | FRem = op {
        return match op {
            F80ToF64 => "simd::x87::f64_from_f80".into(),
            FRem => "simd::x87::fprem".into(),
            _ => format!("simd::x87::i{}_from_f64", 8 * w),
        };
    }
    if let Cpuid | Xgetbv = op {
        let reg = ["eax", "ebx", "ecx", "edx"][w as usize & 3];
        return format!("simd::cpu::{}_{reg}", if op == Cpuid { "cpuid" } else { "xgetbv" });
    }
    if op == ClMul {
        return format!("simd::clmul::{}", if w == 0 { "lo" } else { "hi" });
    }
    let float = matches!(
        op,
        FAdd | FSub | FMul | FDiv | FMin | FMax | FCmpEq | FCmpLt | FCmpLe | FCmpUnord | FCmpNeq | FCmpNlt | FCmpNle
            | FCmpOrd | FCmpGt | FCmpGe | FCmpLtGt
    );
    let f = match op {
        Add | FAdd => "add",
        Sub | FSub => "sub",
        MulLo | FMul => "mul",
        FDiv => "div",
        AddSatU => "adds_u",
        SubSatU => "subs_u",
        AddSatS => "adds_s",
        SubSatS => "subs_s",
        MinU => "min_u",
        MaxU => "max_u",
        MinS => "min_s",
        MaxS => "max_s",
        FMin => "min",
        FMax => "max",
        AvgU => "avg_u",
        CmpEq | FCmpEq => "cmpeq",
        CmpGtS => "cmpgt_s",
        Shl => "shl",
        LShr => "shr",
        AShr => "sar",
        UnpackLo => "unpacklo",
        UnpackHi => "unpackhi",
        SumAbsDiff => "sad",
        PackS => "pack_s",
        PackU => "pack_u",
        FCmpLt => "cmplt",
        FCmpLe => "cmple",
        FCmpUnord => "cmpunord",
        FCmpNeq => "cmpneq",
        FCmpNlt => "cmpnlt",
        FCmpNle => "cmpnle",
        FCmpOrd => "cmpord",
        FCmpGt => "cmpgt",
        FCmpGe => "cmpge",
        FCmpLtGt => "cmplg",
        Cpuid | Xgetbv | ClMul | F80ToF64 | Fist | FRem => unreachable!("handled above"),
    };
    match float {
        true => format!("simd::f{}::{f}", 8 * w),
        false => format!("simd::b{}::{f}", 8 * w),
    }
}

/// The function the emitter calls for a one-operand lane op.
pub fn un_name(op: LaneUn, w: u8) -> String {
    let bits = 8 * w as u32;
    match op {
        LaneUn::MoveMask => format!("simd::b{bits}::movemask"),
        LaneUn::FSqrt => format!("simd::f{bits}::sqrt"),
        LaneUn::IntToF32 => format!("simd::cvt::f32_from_i{bits}"),
        LaneUn::IntToF64 => format!("simd::cvt::f64_from_i{bits}"),
        LaneUn::F32ToIntTrunc => format!("simd::cvt::i{bits}_from_f32_trunc"),
        LaneUn::F64ToIntTrunc => format!("simd::cvt::i{bits}_from_f64_trunc"),
        LaneUn::F32ToInt => format!("simd::cvt::i{bits}_from_f32"),
        LaneUn::F64ToInt => format!("simd::cvt::i{bits}_from_f64"),
        LaneUn::F32ToF64 => "simd::cvt::f64_from_f32".into(),
        LaneUn::F64ToF32 => "simd::cvt::f32_from_f64".into(),
        LaneUn::F64ToF80Lo => "simd::x87::f80_lo".into(),
        LaneUn::F64ToF80Hi => "simd::x87::f80_hi".into(),
    }
}

/// Does `f` use the `simd` module?
pub fn uses(f: &Function) -> bool {
    f.blocks.iter().any(|(_, b)| {
        b.insts.get(&f.value_pool).iter().any(|&v| {
            matches!(f.insts[v].kind, InstKind::Bin { op: BinOp::Lane(..), .. } | InstKind::Un { op: UnOp::Lane(..), .. })
        })
    })
}

/// The `simd` module. Each function takes and returns 64-bit halves of xmm
/// registers; lanes are little-endian, as in memory.
pub const PRELUDE: &str = r#"
/// SSE operations on one 64-bit half of an xmm register, lane by lane.
#[allow(dead_code)]
pub mod simd {
    macro_rules! int_lanes {
        ($m:ident, $u:ty, $i:ty) => {
            pub mod $m {
                const W: usize = core::mem::size_of::<$u>();
                const N: usize = 8 / W;
                fn split(x: u64) -> [$u; N] {
                    let b = x.to_le_bytes();
                    core::array::from_fn(|k| <$u>::from_le_bytes(b[k * W..k * W + W].try_into().unwrap()))
                }
                fn join(l: [$u; N]) -> u64 {
                    let mut b = [0u8; 8];
                    for k in 0..N {
                        b[k * W..k * W + W].copy_from_slice(&l[k].to_le_bytes());
                    }
                    u64::from_le_bytes(b)
                }
                fn each(x: u64, y: u64, op: impl Fn($u, $u) -> $u) -> u64 {
                    let (a, b) = (split(x), split(y));
                    join(core::array::from_fn(|k| op(a[k], b[k])))
                }
                const ONES: $u = <$u>::MAX;
                pub fn add(x: u64, y: u64) -> u64 { each(x, y, |a, b| a.wrapping_add(b)) }
                pub fn sub(x: u64, y: u64) -> u64 { each(x, y, |a, b| a.wrapping_sub(b)) }
                pub fn mul(x: u64, y: u64) -> u64 { each(x, y, |a, b| a.wrapping_mul(b)) }
                pub fn adds_u(x: u64, y: u64) -> u64 { each(x, y, |a, b| a.saturating_add(b)) }
                pub fn subs_u(x: u64, y: u64) -> u64 { each(x, y, |a, b| a.saturating_sub(b)) }
                pub fn adds_s(x: u64, y: u64) -> u64 { each(x, y, |a, b| (a as $i).saturating_add(b as $i) as $u) }
                pub fn subs_s(x: u64, y: u64) -> u64 { each(x, y, |a, b| (a as $i).saturating_sub(b as $i) as $u) }
                pub fn min_u(x: u64, y: u64) -> u64 { each(x, y, |a, b| a.min(b)) }
                pub fn max_u(x: u64, y: u64) -> u64 { each(x, y, |a, b| a.max(b)) }
                pub fn min_s(x: u64, y: u64) -> u64 { each(x, y, |a, b| (a as $i).min(b as $i) as $u) }
                pub fn max_s(x: u64, y: u64) -> u64 { each(x, y, |a, b| (a as $i).max(b as $i) as $u) }
                pub fn avg_u(x: u64, y: u64) -> u64 { each(x, y, |a, b| ((a as u128 + b as u128 + 1) >> 1) as $u) }
                pub fn cmpeq(x: u64, y: u64) -> u64 { each(x, y, |a, b| if a == b { ONES } else { 0 }) }
                pub fn cmpgt_s(x: u64, y: u64) -> u64 { each(x, y, |a, b| if a as $i > b as $i { ONES } else { 0 }) }
                /// Every lane shifted by `n`; a count of the lane width or more
                /// clears it (fills it with its sign, for `sar`).
                pub fn shl(x: u64, n: u64) -> u64 { each(x, 0, |a, _| if n < 8 * W as u64 { a << n } else { 0 }) }
                pub fn shr(x: u64, n: u64) -> u64 { each(x, 0, |a, _| if n < 8 * W as u64 { a >> n } else { 0 }) }
                pub fn sar(x: u64, n: u64) -> u64 { each(x, 0, |a, _| ((a as $i) >> n.min(8 * W as u64 - 1)) as $u) }
                /// The low (high) half of the lanes of `x` and `y`, interleaved.
                pub fn unpacklo(x: u64, y: u64) -> u64 {
                    let (a, b) = (split(x), split(y));
                    join(core::array::from_fn(|k| if k % 2 == 0 { a[k / 2] } else { b[k / 2] }))
                }
                pub fn unpackhi(x: u64, y: u64) -> u64 {
                    let (a, b) = (split(x), split(y));
                    join(core::array::from_fn(|k| if k % 2 == 0 { a[N / 2 + k / 2] } else { b[N / 2 + k / 2] }))
                }
                /// The top bit of each lane, packed into the low bits.
                pub fn movemask(x: u64) -> u64 {
                    split(x).iter().enumerate().fold(0, |m, (k, &a)| m | ((a >> (8 * W - 1)) as u64) << k)
                }
                /// The (signed) lanes of `x`, then of `y`, each narrowed to half
                /// its width, saturating to the signed (unsigned) range.
                pub fn pack_s(x: u64, y: u64) -> u64 { pack(x, y, -(1i64 << (4 * W - 1)), (1i64 << (4 * W - 1)) - 1) }
                pub fn pack_u(x: u64, y: u64) -> u64 { pack(x, y, 0, (1i64 << (4 * W)) - 1) }
                fn pack(x: u64, y: u64, lo: i64, hi: i64) -> u64 {
                    let (a, b) = (split(x), split(y));
                    let mask = u64::MAX >> (64 - 4 * W);
                    a.iter().chain(b.iter()).enumerate().fold(0, |out, (k, &v)| {
                        out | ((v as $i as i64).clamp(lo, hi) as u64 & mask) << (k * 4 * W)
                    })
                }
                /// The sum of the absolute differences of the lanes.
                pub fn sad(x: u64, y: u64) -> u64 {
                    let (a, b) = (split(x), split(y));
                    (0..N).map(|k| a[k].abs_diff(b[k]) as u64).sum()
                }
            }
        };
    }
    int_lanes!(b8, u8, i8);
    int_lanes!(b16, u16, i16);
    int_lanes!(b32, u32, i32);
    int_lanes!(b64, u64, i64);

    macro_rules! float_lanes {
        ($m:ident, $f:ty, $u:ty) => {
            pub mod $m {
                const W: usize = core::mem::size_of::<$f>();
                const N: usize = 8 / W;
                fn split(x: u64) -> [$f; N] {
                    core::array::from_fn(|k| <$f>::from_bits((x >> (8 * W * k)) as $u))
                }
                fn join(l: [$u; N]) -> u64 {
                    (0..N).fold(0, |x, k| x | (l[k] as u64) << (8 * W * k))
                }
                fn each(x: u64, y: u64, op: impl Fn($f, $f) -> $f) -> u64 {
                    let (a, b) = (split(x), split(y));
                    join(core::array::from_fn(|k| op(a[k], b[k]).to_bits()))
                }
                fn test(x: u64, y: u64, op: impl Fn($f, $f) -> bool) -> u64 {
                    let (a, b) = (split(x), split(y));
                    join(core::array::from_fn(|k| if op(a[k], b[k]) { <$u>::MAX } else { 0 }))
                }
                pub fn add(x: u64, y: u64) -> u64 { each(x, y, |a, b| a + b) }
                pub fn sub(x: u64, y: u64) -> u64 { each(x, y, |a, b| a - b) }
                pub fn mul(x: u64, y: u64) -> u64 { each(x, y, |a, b| a * b) }
                pub fn div(x: u64, y: u64) -> u64 { each(x, y, |a, b| a / b) }
                /// x86's min and max: the second operand unless the first is
                /// strictly smaller (larger), so a NaN or equal zeros give `b`.
                pub fn min(x: u64, y: u64) -> u64 { each(x, y, |a, b| if a < b { a } else { b }) }
                pub fn max(x: u64, y: u64) -> u64 { each(x, y, |a, b| if a > b { a } else { b }) }
                pub fn sqrt(x: u64) -> u64 { each(x, 0, |a, _| a.sqrt()) }
                pub fn cmpeq(x: u64, y: u64) -> u64 { test(x, y, |a, b| a == b) }
                pub fn cmplt(x: u64, y: u64) -> u64 { test(x, y, |a, b| a < b) }
                pub fn cmple(x: u64, y: u64) -> u64 { test(x, y, |a, b| a <= b) }
                pub fn cmpunord(x: u64, y: u64) -> u64 { test(x, y, |a, b| a.is_nan() || b.is_nan()) }
                pub fn cmpneq(x: u64, y: u64) -> u64 { test(x, y, |a, b| a != b) }
                pub fn cmpnlt(x: u64, y: u64) -> u64 { test(x, y, |a, b| !(a < b)) }
                pub fn cmpnle(x: u64, y: u64) -> u64 { test(x, y, |a, b| !(a <= b)) }
                pub fn cmpord(x: u64, y: u64) -> u64 { test(x, y, |a, b| !a.is_nan() && !b.is_nan()) }
                pub fn cmpgt(x: u64, y: u64) -> u64 { test(x, y, |a, b| a > b) }
                pub fn cmpge(x: u64, y: u64) -> u64 { test(x, y, |a, b| a >= b) }
                pub fn cmplg(x: u64, y: u64) -> u64 { test(x, y, |a, b| a < b || a > b) }
            }
        };
    }
    float_lanes!(f32, f32, u32);
    float_lanes!(f64, f64, u64);

    /// Scalar conversions on the low lane. Integers are sign-extended from their
    /// width; a float that doesn't fit the integer (or a NaN) gives its minimum,
    /// the "integer indefinite" x86 returns.
    pub mod cvt {
        fn to_int(v: f64, bits: u32) -> u64 {
            let lim = (1u64 << (bits - 1)) as f64;
            let n = if v >= -lim && v < lim { v as i64 } else { i64::MIN >> (64 - bits) };
            n as u64 & (u64::MAX >> (64 - bits))
        }
        pub fn f64_from_i16(x: u64) -> u64 { (x as i16 as f64).to_bits() }
        pub fn f32_from_i32(x: u64) -> u64 { (x as i32 as f32).to_bits() as u64 }
        pub fn f32_from_i64(x: u64) -> u64 { (x as i64 as f32).to_bits() as u64 }
        pub fn f64_from_i32(x: u64) -> u64 { (x as i32 as f64).to_bits() }
        pub fn f64_from_i64(x: u64) -> u64 { (x as i64 as f64).to_bits() }
        pub fn f64_from_f32(x: u64) -> u64 { (f32::from_bits(x as u32) as f64).to_bits() }
        pub fn f32_from_f64(x: u64) -> u64 { (f64::from_bits(x) as f32).to_bits() as u64 }
        pub fn i32_from_f32_trunc(x: u64) -> u64 { to_int((f32::from_bits(x as u32) as f64).trunc(), 32) }
        pub fn i64_from_f32_trunc(x: u64) -> u64 { to_int((f32::from_bits(x as u32) as f64).trunc(), 64) }
        pub fn i32_from_f64_trunc(x: u64) -> u64 { to_int(f64::from_bits(x).trunc(), 32) }
        pub fn i64_from_f64_trunc(x: u64) -> u64 { to_int(f64::from_bits(x).trunc(), 64) }
        pub fn i32_from_f32(x: u64) -> u64 { to_int((f32::from_bits(x as u32) as f64).round_ties_even(), 32) }
        pub fn i64_from_f32(x: u64) -> u64 { to_int((f32::from_bits(x as u32) as f64).round_ties_even(), 64) }
        pub fn i32_from_f64(x: u64) -> u64 { to_int(f64::from_bits(x).round_ties_even(), 32) }
        pub fn i64_from_f64(x: u64) -> u64 { to_int(f64::from_bits(x).round_ties_even(), 64) }
    }

    /// x87 values, which the lifted code keeps as f64: conversions from and to
    /// the 80-bit format in memory, and `fist`'s rounding.
    pub mod x87 {
        /// `fprem` run to completion: C's `fmod`.
        pub fn fprem(x: u64, y: u64) -> u64 { (f64::from_bits(x) % f64::from_bits(y)).to_bits() }
        pub fn f64_from_f80(lo: u64, hi: u64) -> u64 {
            let sign = (hi >> 15 & 1) << 63;
            let e = hi & 0x7fff;
            if e == 0x7fff {
                // infinity, or a NaN keeping its top payload bits
                let frac = lo << 1 >> 12;
                return sign | 0x7ff << 52 | if frac == 0 && lo << 1 != 0 { 1 << 51 } else { frac };
            }
            if lo == 0 {
                return sign;
            }
            // lo * 2^(e - 16383 - 63), rounded to nearest by the conversion
            let m = lo as f64;
            // a denormal has the smallest exponent, without the integer bit
            let mut k = e.max(1) as i64 - 16383 - 63;
            let mut v = m;
            while k > 0 {
                let s = k.min(1000);
                v *= 2f64.powi(s as i32);
                k -= s;
            }
            while k < 0 {
                let s = (-k).min(1000);
                v /= 2f64.powi(s as i32);
                k += s;
            }
            sign | v.to_bits()
        }
        fn split(x: u64) -> (u64, u64) {
            let sign = x >> 63 << 15;
            let e = x >> 52 & 0x7ff;
            let frac = x & ((1 << 52) - 1);
            match e {
                0 if frac == 0 => (0, sign),
                // subnormal: normalize
                0 => {
                    let z = frac.leading_zeros() as u64;
                    (frac << z, sign | (16383 - 1023 + 1 + 11 - z))
                }
                0x7ff => (1 << 63 | frac << 11, sign | 0x7fff),
                _ => (1 << 63 | frac << 11, sign | (e + 16383 - 1023)),
            }
        }
        pub fn f80_lo(x: u64) -> u64 { split(x).0 }
        pub fn f80_hi(x: u64) -> u64 { split(x).1 }
        fn fist(x: u64, cw: u64, bits: u32) -> u64 {
            let v = f64::from_bits(x);
            let r = match cw >> 10 & 3 {
                0 => v.round_ties_even(),
                1 => v.floor(),
                2 => v.ceil(),
                _ => v.trunc(),
            };
            let lim = (1u64 << (bits - 1)) as f64;
            let n = if r >= -lim && r < lim { r as i64 } else { i64::MIN >> (64 - bits) };
            n as u64 & (u64::MAX >> (64 - bits))
        }
        pub fn i16_from_f64(x: u64, cw: u64) -> u64 { fist(x, cw, 16) }
        pub fn i32_from_f64(x: u64, cw: u64) -> u64 { fist(x, cw, 32) }
        pub fn i64_from_f64(x: u64, cw: u64) -> u64 { fist(x, cw, 64) }
    }

    /// `cpuid` and `xgetbv`, one output register each.
    #[allow(unused_unsafe)]
    /// `pclmulqdq`: the carry-less (polynomial) product of two 64-bit values.
    pub mod clmul {
        fn wide(a: u64, b: u64) -> u128 {
            (0..64).filter(|k| b >> k & 1 != 0).fold(0u128, |p, k| p ^ ((a as u128) << k))
        }
        pub fn lo(a: u64, b: u64) -> u64 { wide(a, b) as u64 }
        pub fn hi(a: u64, b: u64) -> u64 { (wide(a, b) >> 64) as u64 }
    }
    pub mod cpu {
        fn cpuid(a: u64, c: u64) -> core::arch::x86_64::CpuidResult {
            unsafe { core::arch::x86_64::__cpuid_count(a as u32, c as u32) }
        }
        pub fn cpuid_eax(a: u64, c: u64) -> u64 { cpuid(a, c).eax as u64 }
        pub fn cpuid_ebx(a: u64, c: u64) -> u64 { cpuid(a, c).ebx as u64 }
        pub fn cpuid_ecx(a: u64, c: u64) -> u64 { cpuid(a, c).ecx as u64 }
        pub fn cpuid_edx(a: u64, c: u64) -> u64 { cpuid(a, c).edx as u64 }
        fn xgetbv(c: u64) -> (u32, u32) {
            let (a, d): (u32, u32);
            unsafe { core::arch::asm!("xgetbv", in("ecx") c as u32, out("eax") a, out("edx") d, options(nomem, nostack, preserves_flags)) };
            (a, d)
        }
        pub fn xgetbv_eax(c: u64, _: u64) -> u64 { xgetbv(c).0 as u64 }
        pub fn xgetbv_edx(c: u64, _: u64) -> u64 { xgetbv(c).1 as u64 }
    }
}
"#;
