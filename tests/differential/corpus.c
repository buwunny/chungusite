/* Differential-test corpus (tests/differential.rs).
 *
 * Each function is compiled with every available C compiler at several
 * optimization levels, decompiled in both modes, and called side by side with
 * the original on random inputs. The `@diff` line above a function tells the
 * harness how to call it:
 *
 *     // @diff NAME: RET(ARG, ARG, ...)
 *
 * Types: i8 u8 i16 u16 i32 u32 i64 u64 (C integers of that width).
 *   ARG  i32:-5..100   an integer drawn from lo..hi (exclusive), e.g. a length
 *   ARG  buf:N         pointer to N random bytes (writable; compared afterwards)
 *   ARG  str:N         pointer to an N-byte buffer holding a NUL-terminated string
 *   RET  void | bool | an integer type
 *   RET  ptr:K         pointer into argument K's buffer, or NULL
 *
 * Results are compared at the width of the return type, since the ABI leaves the
 * upper bits of rax undefined for narrower returns. Integer arguments narrower than
 * 64 bits reach the decompiled code with random upper register bits, as they may
 * from a real caller. Keep functions free of libc names: the originals are linked
 * into a Rust program. Calling libc is fine.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <emmintrin.h>
#include <cpuid.h>
#include <immintrin.h>

#define NOINLINE __attribute__((noinline))

/* ---- straight-line integer code ---- */

// @diff add_u64: u64(u64, u64)
uint64_t add_u64(uint64_t a, uint64_t b) { return a + b; }

// @diff sub_u64: u64(u64, u64)
uint64_t sub_u64(uint64_t a, uint64_t b) { return a - b; }

// @diff xor3: u64(u64, u64, u64)
uint64_t xor3(uint64_t a, uint64_t b, uint64_t c) { return a ^ b ^ c; }

// @diff and_or: u64(u64, u64, u64)
uint64_t and_or(uint64_t a, uint64_t b, uint64_t c) { return (a & b) | c; }

// @diff add_i32: i32(i32, i32)
int32_t add_i32(int32_t a, int32_t b) { return (int32_t)((uint32_t)a + (uint32_t)b); }

// @diff lin_u32: u32(u32)
uint32_t lin_u32(uint32_t x) { return x * 5 + 3; }

// @diff lin_u64: u64(u64, u64)
uint64_t lin_u64(uint64_t x, uint64_t y) { return x + y * 8 + 0x40; }

// @diff low_byte: u8(u64)
uint8_t low_byte(uint64_t x) { return (uint8_t)x; }

// @diff mask_lo: u64(u64)
uint64_t mask_lo(uint64_t x) { return x & 0xfff0; }

// @diff six_args: u64(u64, u64, u64, u64, u64, u64)
uint64_t six_args(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f) {
    return a - b + (c ^ d) - (e | f);
}

// @diff shl_u64: u64(u64, u8)
uint64_t shl_u64(uint64_t x, uint8_t s) { return x << (s & 63); }

// @diff shr_u64: u64(u64, u8)
uint64_t shr_u64(uint64_t x, uint8_t s) { return x >> (s & 63); }

// @diff sar_i64: i64(i64, u8)
int64_t sar_i64(int64_t x, uint8_t s) { return x >> (s & 63); }

// @diff rotl_u64: u64(u64, u8)
uint64_t rotl_u64(uint64_t x, uint8_t s) { s &= 63; return (x << s) | (x >> ((64 - s) & 63)); }

// @diff mul_u64: u64(u64, u64)
uint64_t mul_u64(uint64_t a, uint64_t b) { return a * b; }

// @diff mul_i32: i32(i32, i32)
int32_t mul_i32(int32_t a, int32_t b) { return (int32_t)((uint32_t)a * (uint32_t)b); }

// @diff div_i64: i64(i64, i64:1..1000)
int64_t div_i64(int64_t a, int64_t b) { return a / b; }

// @diff mod_u64: u64(u64, u64:1..100000)
uint64_t mod_u64(uint64_t a, uint64_t b) { return a % b; }

// @diff neg_i32: i32(i32)
int32_t neg_i32(int32_t x) { return (int32_t)(0u - (uint32_t)x); }

// @diff not_u32: u32(u32)
uint32_t not_u32(uint32_t x) { return ~x; }

// @diff sext8: i32(i8)
int32_t sext8(int8_t x) { return x; }

// @diff zext16: u64(u16)
uint64_t zext16(uint16_t x) { return x; }

// @diff sext32: i64(i32)
int64_t sext32(int32_t x) { return x; }

/* ---- comparisons and selects ---- */

// @diff is_even: bool(u64)
_Bool is_even(uint64_t x) { return (x & 1) == 0; }

// @diff lt_i64: bool(i64, i64)
_Bool lt_i64(int64_t a, int64_t b) { return a < b; }

// @diff max_i32: i32(i32, i32)
int32_t max_i32(int32_t a, int32_t b) { return a > b ? a : b; }

// @diff min_u64: u64(u64, u64)
uint64_t min_u64(uint64_t a, uint64_t b) { return a < b ? a : b; }

// @diff abs_i64: i64(i64)
int64_t abs_i64(int64_t x) { return x < 0 ? (int64_t)(0ull - (uint64_t)x) : x; }

// @diff sign_i64: i32(i64)
int32_t sign_i64(int64_t x) { return (x > 0) - (x < 0); }

// @diff clamp_i32: i32(i32, i32:-100..0, i32:0..100)
int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// @diff pick_u64: u64(u64, u64, u64)
uint64_t pick_u64(uint64_t c, uint64_t a, uint64_t b) {
    if (c) return a;
    return b;
}

// @diff classify: u64(u64)
uint64_t classify(uint64_t x) {
    if (x == 0) return 10;
    if (x < 100) return 20;
    if (x >= 0x8000000000000000ull) return 30;
    return 40;
}

// @diff switch8: u64(u64:0..10)
uint64_t switch8(uint64_t x) {
    switch (x) {
    case 0: return 11;
    case 1: return 23;
    case 2: return 5;
    case 3: return 79;
    case 4: return 1000;
    case 5: return 7;
    case 6: return 42;
    case 7: return 3;
    default: return 0;
    }
}

/* ---- loops over integers ---- */

// @diff popcount_loop: u32(u32)
uint32_t popcount_loop(uint32_t x) {
    uint32_t c = 0;
    while (x) { c += x & 1; x >>= 1; }
    return c;
}

// @diff sum_to: u64(u64:0..1000)
uint64_t sum_to(uint64_t n) {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n; i++) s += i;
    return s;
}

// @diff fib: u64(u32:0..94)
uint64_t fib(uint32_t n) {
    uint64_t a = 0, b = 1;
    while (n--) { uint64_t t = a + b; a = b; b = t; }
    return a;
}

// @diff fact: u64(u64:0..21)
uint64_t fact(uint64_t n) {
    uint64_t r = 1;
    while (n > 1) r *= n--;
    return r;
}

// @diff gcd_u64: u64(u64, u64)
uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b) { uint64_t t = a % b; a = b; b = t; }
    return a;
}

// @diff collatz: u32(u32:1..10000)
uint32_t collatz(uint32_t n) {
    uint32_t steps = 0;
    while (n != 1) { n = (n & 1) ? 3 * n + 1 : n / 2; steps++; }
    return steps;
}

/* ---- memory ---- */

// @diff get8: u64(buf:64, u64:0..8)
uint64_t get8(const uint64_t *p, uint64_t i) { return p[i]; }

// @diff set8: void(buf:64, u64:0..8, u64)
void set8(uint64_t *p, uint64_t i, uint64_t v) { p[i] = v; }

// @diff field_sum: u64(buf:16)
struct rec { uint64_t a; uint32_t b; uint16_t c; uint8_t d; };
uint64_t field_sum(const struct rec *r) { return r->a + r->b + r->c + r->d; }

// @diff field_set: void(buf:16, u32)
void field_set(struct rec *r, uint32_t v) { r->b = v; r->d = (uint8_t)v; }

// @diff swap_pair: void(buf:16)
void swap_pair(uint64_t *p) { uint64_t t = p[0]; p[0] = p[1]; p[1] = t; }

// @diff sum_u64: u64(buf:128, u64:0..17)
uint64_t sum_u64(const uint64_t *p, uint64_t n) {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n; i++) s += p[i];
    return s;
}

// @diff sum_i32: i32(buf:64, i32:0..17)
int32_t sum_i32(const int32_t *p, int32_t n) {
    uint32_t s = 0;
    for (int32_t i = 0; i < n; i++) s += (uint32_t)p[i];
    return (int32_t)s;
}

// @diff max_u64: u64(buf:64, u64:1..9)
uint64_t max_u64(const uint64_t *p, uint64_t n) {
    uint64_t m = p[0];
    for (uint64_t i = 1; i < n; i++) if (p[i] > m) m = p[i];
    return m;
}

// @diff dot_u32: u64(buf:64, buf:64, u64:0..17)
uint64_t dot_u32(const uint32_t *a, const uint32_t *b, uint64_t n) {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n; i++) s += (uint64_t)a[i] * b[i];
    return s;
}

// @diff inc_all: void(buf:64, u64:0..17)
void inc_all(uint32_t *p, uint64_t n) { for (uint64_t i = 0; i < n; i++) p[i]++; }

// @diff fill_u8: void(buf:64, u64:0..65, u8)
void fill_u8(uint8_t *p, uint64_t n, uint8_t v) {
    for (uint64_t i = 0; i < n; i++) ((volatile uint8_t *)p)[i] = v;
}

// @diff reverse_u32: void(buf:64, u64:0..17)
void reverse_u32(uint32_t *p, uint64_t n) {
    for (uint64_t i = 0, j = n; i + 1 < j; i++, j--) { uint32_t t = p[i]; p[i] = p[j - 1]; p[j - 1] = t; }
}

// @diff count_zero: u64(buf:64, u64:0..65)
uint64_t count_zero(const uint8_t *p, uint64_t n) {
    uint64_t c = 0;
    for (uint64_t i = 0; i < n; i++) c += p[i] == 0;
    return c;
}

// @diff index_of: u64(buf:64, u64:0..65, u8)
uint64_t index_of(const uint8_t *p, uint64_t n, uint8_t v) {
    for (uint64_t i = 0; i < n; i++) if (p[i] == v) return i;
    return n;
}

// @diff find_byte: ptr:0(buf:64, u64:0..65, u8)
uint8_t *find_byte(uint8_t *p, uint64_t n, uint8_t v) {
    for (uint64_t i = 0; i < n; i++) if (p[i] == v) return p + i;
    return NULL;
}

// @diff copy_u64: void(buf:64, buf:64, u64:0..9)
void copy_u64(uint64_t *dst, const uint64_t *src, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) ((volatile uint64_t *)dst)[i] = src[i];
}

// @diff list_len: u64(buf:128)
/* A linked list laid out in one buffer: each 16-byte node is {next offset, value}. */
uint64_t list_len(const uint64_t *base) {
    uint64_t n = 0, i = 0;
    while (n < 8 && (base[i * 2] & 7) != 0) { i = base[i * 2] & 7; n++; }
    return n;
}

/* ---- strings ---- */

// @diff str_len: u64(str:32)
uint64_t str_len(const char *s) { const char *p = s; while (*p) p++; return (uint64_t)(p - s); }

// @diff str_hash: u64(str:32)
uint64_t str_hash(const char *s) {
    uint64_t h = 5381;
    while (*s) h = h * 33 + (uint8_t)*s++;
    return h;
}

// @diff str_eq: bool(str:8, str:8)
_Bool str_eq(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return *a == *b;
}

// @diff parse_int: i64(str:16)
int64_t parse_int(const char *s) {
    int neg = *s == '-';
    if (neg) s++;
    uint64_t v = 0;
    while (*s >= '0' && *s <= '9') v = v * 10 + (uint64_t)(*s++ - '0');
    return neg ? (int64_t)(0 - v) : (int64_t)v;
}

// @diff str_chr: ptr:0(str:32, u8)
char *str_chr(char *s, uint8_t c) {
    for (; *s; s++) if ((uint8_t)*s == c) return s;
    return NULL;
}

/* ---- calls, signatures and stack frames ---- */

// @diff mix: u64(u64, u64)
NOINLINE uint64_t mix(uint64_t a, uint64_t b) { return a * 31 + (b ^ (b >> 7)); }

// @diff call_mix: u64(u64, u64)
uint64_t call_mix(uint64_t a, uint64_t b) { return mix(a, b) + mix(b, 3); }

// Values live across calls sit in callee-saved registers, pushed and popped.
// @diff across_calls: u64(u64, u64, u64)
uint64_t across_calls(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t x = mix(a, b);
    uint64_t y = mix(c, x);
    return x ^ y ^ a ^ c;
}

// A tail call with the arguments swapped: `jmp mix`.
// @diff tail_swap: u64(u64, u64)
uint64_t tail_swap(uint64_t a, uint64_t b) { return mix(b, a); }

// Not in the corpus itself: called through its symbol, as an extern.
NOINLINE uint64_t hidden_square(uint64_t x) { return x * x + 1; }

// @diff call_hidden: u64(u64)
uint64_t call_hidden(uint64_t x) { return hidden_square(x) + hidden_square(x + 1); }

// @diff bump: void(buf:8)
NOINLINE void bump(uint64_t *p) { *p += 3; }

// A void function calling a void function.
// @diff bump_twice: void(buf:8)
void bump_twice(uint64_t *p) { bump(p); bump(p); }

// @diff store_sum: void(buf:8, u64, u64)
NOINLINE void store_sum(uint64_t *out, uint64_t a, uint64_t b) { *out = a + b; }

// An address-taken local: it stays in the frame and its address is passed on.
// @diff via_local: u64(u64, u64)
uint64_t via_local(uint64_t a, uint64_t b) {
    uint64_t r;
    store_sum(&r, a, b);
    return r * 2;
}

// @diff eight_args: u64(u64, u64, u64, u64, u64, u64, u64, u64)
NOINLINE uint64_t eight_args(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f, uint64_t g, uint64_t h) {
    return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f + 7 * g + 8 * h;
}

// Two arguments go on the stack.
// @diff call_eight: u64(u64, u64)
uint64_t call_eight(uint64_t x, uint64_t y) { return eight_args(x, y, 1, 2, 3, 4, x ^ y, 5) + 1; }

// @diff fib_rec: u64(u64:0..18)
NOINLINE uint64_t fib_rec(uint64_t n) { return n < 2 ? n : fib_rec(n - 1) + fib_rec(n - 2); }

// @diff len_plus: u64(str:32)
uint64_t len_plus(const char *s) { return strlen(s) + 1; }

// @diff copy_n: void(buf:64, buf:64, u64:0..65)
void copy_n(uint8_t *d, const uint8_t *s, uint64_t n) { memcpy(d, s, n); }

// An array on the stack, indexed.
// @diff stack_table: u64(u64:0..16, u64)
uint64_t stack_table(uint64_t i, uint64_t k) {
    volatile uint64_t t[16];
    for (int j = 0; j < 16; j++) t[j] = k * (uint64_t)j + (uint64_t)j;
    return t[i];
}

// A buffer on the stack, passed to a callee and read back.
// @diff stack_buf: u64(u64, u64)
uint64_t stack_buf(uint64_t a, uint64_t b) {
    uint64_t buf[4];
    store_sum(&buf[0], a, b);
    store_sum(&buf[1], b, 7);
    bump(&buf[1]);
    return buf[0] ^ buf[1];
}

// @diff max_of3: i64(i64, i64, i64)
NOINLINE int64_t max2(int64_t a, int64_t b) { return a > b ? a : b; }
int64_t max_of3(int64_t a, int64_t b, int64_t c) { return max2(max2(a, b), c); }

/* ---- ownership (docs/ownership.md): borrows of locals, heap allocations ---- */

// @diff add_into: void(buf:8, buf:8)
NOINLINE void add_into(uint64_t *d, const uint64_t *s) { *d += *s; }

// Two locals lent to one callee, one of them mutably: disjoint parts of the frame.
// @diff two_locals: u64(u64, u64)
uint64_t two_locals(uint64_t a, uint64_t b) {
    uint64_t x = a, y = b;
    add_into(&x, &y);
    return x ^ (y << 1);
}

// @diff add_self: void(buf:8, buf:8)
NOINLINE void add_self(uint64_t *d, const uint64_t *s) { *d += *s * 2; }

// The same local lent twice, once mutably: safe Rust can't, so it stays raw.
// @diff same_local_twice: u64(u64)
uint64_t same_local_twice(uint64_t a) {
    uint64_t x = a;
    add_self(&x, &x);
    return x;
}

// A heap allocation that is written, lent to a callee and freed: a Box.
// @diff heap_pair: u64(u64, u64)
uint64_t heap_pair(uint64_t a, uint64_t b) {
    uint64_t *p = malloc(2 * sizeof *p);
    if (!p) return 0;
    p[0] = a;
    p[1] = b;
    add_into(&p[0], &p[1]);
    uint64_t r = p[0] * 3 + p[1];
    free(p);
    return r;
}

// A local array filled and copied with memset and memcpy, then summed. The
// lengths are arguments, so these stay calls.
// @diff local_copy: u64(buf:32, u8, u64:0..49, u64:0..33)
uint64_t local_copy(const uint8_t *src, uint8_t c, uint64_t n, uint64_t m) {
    uint8_t t[48];
    memset(t, c, n);
    memset(t + n, 0, 48 - n);
    if (m > 40 - (n > 40 ? 40 : n)) m = 0;
    memcpy(t + 8, src, m);
    uint64_t s = 0;
    for (int i = 0; i < 48; i++) s = s * 31 + t[i];
    return s;
}

/* ---- wide multiplies and 16-byte copies ---- */

// @diff mulhi_u64: u64(u64, u64)
uint64_t mulhi_u64(uint64_t a, uint64_t b) { return (uint64_t)(((unsigned __int128)a * b) >> 64); }

// @diff mulhi_i64: i64(i64, i64)
int64_t mulhi_i64(int64_t a, int64_t b) { return (int64_t)(((__int128)a * b) >> 64); }

// @diff mul_overflows: bool(u64, u64)
_Bool mul_overflows(uint64_t a, uint64_t b) { uint64_t r; return __builtin_mul_overflow(a, b, &r); }

// @diff smul_overflows: bool(i64, i64)
_Bool smul_overflows(int64_t a, int64_t b) { int64_t r; return __builtin_mul_overflow(a, b, &r); }

// @diff mul32_wide: u64(u32, u32)
uint64_t mul32_wide(uint32_t a, uint32_t b) { return (uint64_t)a * b; }

// @diff copy16: void(buf:16, buf:16)
void copy16(uint8_t *d, const uint8_t *s) { memcpy(d, s, 16); }

// @diff zero32: void(buf:32)
void zero32(uint8_t *d) { memset(d, 0, 32); }

struct pair { uint64_t a, b; };

// @diff swap_pairs: void(buf:32)
void swap_pairs(uint8_t *p) {
    struct pair x, y;
    memcpy(&x, p, 16);
    memcpy(&y, p + 16, 16);
    memcpy(p, &y, 16);
    memcpy(p + 16, &x, 16);
}

/* ---- wide arithmetic and bit instructions ---- */

// The high half of a 64x64 product: one-operand mul.
// @diff mul_hi: u64(u64, u64)
uint64_t mul_hi(uint64_t a, uint64_t b) { return (uint64_t)(((unsigned __int128)a * b) >> 64); }

// @diff mul_hi32: u32(u32, u32)
uint32_t mul_hi32(uint32_t a, uint32_t b) { return (uint32_t)(((uint64_t)a * b) >> 32); }

// @diff smul_hi: i64(i64, i64)
int64_t smul_hi(int64_t a, int64_t b) { return (int64_t)(((__int128)a * b) >> 64); }

// Overflow checks read CF/OF after mul and imul.
// @diff mul_ovf: u64(u64, u64)
uint64_t mul_ovf(uint64_t a, uint64_t b) { uint64_t r; return __builtin_mul_overflow(a, b, &r) ? 0 : r + 1; }

// @diff smul_ovf: i64(i64, i64)
int64_t smul_ovf(int64_t a, int64_t b) { int64_t r; return __builtin_mul_overflow(a, b, &r) ? -1 : r; }

// @diff smul_ovf32: i32(i32, i32)
int32_t smul_ovf32(int32_t a, int32_t b) { int32_t r; return __builtin_mul_overflow(a, b, &r) ? -1 : r; }

// 128-bit add and subtract: add/adc, sub/sbb.
// @diff add128_hi: u64(u64, u64, u64, u64)
uint64_t add128_hi(uint64_t alo, uint64_t ahi, uint64_t blo, uint64_t bhi) {
    unsigned __int128 a = ((unsigned __int128)ahi << 64) | alo, b = ((unsigned __int128)bhi << 64) | blo;
    return (uint64_t)((a + b) >> 64);
}

// @diff sub128_hi: u64(u64, u64, u64, u64)
uint64_t sub128_hi(uint64_t alo, uint64_t ahi, uint64_t blo, uint64_t bhi) {
    unsigned __int128 a = ((unsigned __int128)ahi << 64) | alo, b = ((unsigned __int128)bhi << 64) | blo;
    return (uint64_t)((a - b) >> 64);
}

// A mask from the carry: cmp; sbb.
// @diff below_mask: u64(u64, u64)
uint64_t below_mask(uint64_t a, uint64_t b) { return a < b ? ~(uint64_t)0 : 0; }

// 128-bit shifts: shld / shrd.
// @diff shl128_hi: u64(u64, u64, u32:0..128)
uint64_t shl128_hi(uint64_t lo, uint64_t hi, uint32_t n) {
    unsigned __int128 x = ((unsigned __int128)hi << 64) | lo;
    return (uint64_t)((x << (n & 127)) >> 64);
}

// @diff shr128_lo: u64(u64, u64, u32:0..128)
uint64_t shr128_lo(uint64_t lo, uint64_t hi, uint32_t n) {
    unsigned __int128 x = ((unsigned __int128)hi << 64) | lo;
    return (uint64_t)(x >> (n & 127));
}

// @diff ctz64: u64(u64)
uint64_t ctz64(uint64_t x) { return x ? (uint64_t)__builtin_ctzll(x) : 64; }

// @diff clz64: u64(u64)
uint64_t clz64(uint64_t x) { return x ? (uint64_t)__builtin_clzll(x) : 64; }

// @diff bswap64: u64(u64)
uint64_t bswap64(uint64_t x) { return __builtin_bswap64(x); }

// @diff bswap32: u32(u32)
uint32_t bswap32(uint32_t x) { return __builtin_bswap32(x); }

// @diff rotl64: u64(u64, u32)
uint64_t rotl64(uint64_t x, uint32_t n) { return (x << (n & 63)) | (x >> (-n & 63)); }

// @diff rotr16: u16(u16, u32)
uint16_t rotr16(uint16_t x, uint32_t n) { return (uint16_t)((x >> (n & 15)) | (x << (-n & 15))); }

// Shifting a byte by up to 31: x86 doesn't wrap the count at 8.
// @diff shr8_cl: u8(buf:4, u32)
uint8_t shr8_cl(uint8_t *p, uint32_t n) { uint8_t v = p[0]; __asm__("shrb %%cl, %0" : "+r"(v) : "c"(n)); return v; }

// @diff bit_set: u64(u64, u32:0..64)
uint64_t bit_set(uint64_t x, uint32_t n) { return (x >> (n & 63)) & 1 ? 7 : 9; }

/* ---- atomics (single-threaded here) ---- */

// @diff fetch_add: u64(buf:16, u64)
uint64_t fetch_add(uint8_t *p, uint64_t v) { return __atomic_fetch_add((uint64_t *)p, v, __ATOMIC_SEQ_CST); }

// @diff swap_in: u64(buf:16, u64)
uint64_t swap_in(uint8_t *p, uint64_t v) { return __atomic_exchange_n((uint64_t *)p, v, __ATOMIC_SEQ_CST); }

// @diff cas: u64(buf:16, u64, u64)
uint64_t cas(uint8_t *p, uint64_t expect, uint64_t v) {
    __atomic_compare_exchange_n((uint64_t *)p, &expect, v, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return expect;
}

// @diff cas_hit: u64(buf:16, u64)
uint64_t cas_hit(uint8_t *p, uint64_t v) {
    uint64_t expect = *(uint64_t *)p;
    return __atomic_compare_exchange_n((uint64_t *)p, &expect, v, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) ? 1 : 2;
}

/* ---- SSE moves ---- */

// Zeroing: xorps / pxor, then 16-byte stores.
// @diff zero48: void(buf:64)
void zero48(uint8_t *p) { memset(p, 0, 48); }

// Two 8-byte values into one 16-byte store: movq / punpcklqdq.
// @diff store_pair: void(buf:32, u64, u64)
void store_pair(uint8_t *p, uint64_t a, uint64_t b) { struct pair x = { a, b }; *(struct pair *)p = x; }

/* ---- jump tables ---- */

// Dense cases with code in each: an indirect jump through a table.
// @diff jump_table: u64(u64:0..16, u64)
uint64_t jump_table(uint64_t i, uint64_t x) {
    switch (i) {
    case 0: return x * 3 + 1;
    case 1: return x ^ 0x5555;
    case 2: return x << 3;
    case 3: return x - 77;
    case 4: return ~x;
    case 5: return x * x;
    case 6: return x >> 5;
    case 7: return x + i * 1000;
    case 8: return x | 0xf0f0;
    case 9: return x & 0x0ff0;
    case 10: return x * 41;
    default: return 5;
    }
}

// A switch in a loop: the table's address is loaded once, before the loop.
// @diff switch_loop: u64(str:24)
uint64_t switch_loop(const char *s) {
    uint64_t h = 0, n = 0;
    for (; *s; s++, n++) {
        uint64_t c = (uint8_t)*s;
        c = (c * 0x9e37) ^ (h >> 7);
        c += (c >> 3) * 5;
        c ^= (c << 9) + (h >> 13);
        c += (c >> 11) ^ n;
        switch (c % 11) {
        case 0: h += 3; break;
        case 1: h ^= 0x55; break;
        case 2: h *= 7; break;
        case 3: h -= 11; break;
        case 4: h <<= 1; break;
        case 5: h += 0x100; break;
        case 6: h |= 9; break;
        case 7: h = ~h; break;
        case 8: h += n * 17; break;
        default: h += 1;
        }
    }
    return h;
}

/* ---- flags read in another block ---- */

// gcc tests `==` and then `<` of the same cmp in the next block.
// @diff cmp3: i32(i64, i64)
int cmp3(int64_t a, int64_t b) { if (a == b) return 0; if (a < b) return -1; return 1; }

// @diff count_down: u64(i32:-5..100, u64)
uint64_t count_down(int32_t n, uint64_t k) {
    uint64_t s = 0;
    while (--n > 0) s += n * k + (s >> 3);
    return s;
}

/* ---- thread-locals ---- */

static __thread uint64_t tls_counter = 7;
static __thread uint32_t tls_zero;

// Initialized (.tdata) and zeroed (.tbss) thread-locals, read and written at
// fixed offsets below the thread pointer.
// @diff tls_bump: u64(u64)
uint64_t tls_bump(uint64_t x) {
    tls_counter += x;
    tls_zero ^= (uint32_t)x;
    return tls_counter * 3 + tls_zero;
}

/* ---- SSE2 vectors ---- */

// @diff simd_count_eq: u32(buf:64, u8)
uint32_t simd_count_eq(const uint8_t *p, uint8_t c) {
    __m128i k = _mm_set1_epi8((char)c);
    uint32_t n = 0;
    for (int i = 0; i < 64; i += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(p + i));
        n += (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(v, k)) * (i + 1);
    }
    return n;
}

// @diff simd_add32: void(buf:64)
void simd_add32(uint8_t *p) {
    __m128i a = _mm_loadu_si128((const __m128i *)p);
    __m128i b = _mm_loadu_si128((const __m128i *)(p + 16));
    _mm_storeu_si128((__m128i *)(p + 32), _mm_add_epi32(a, b));
    _mm_storeu_si128((__m128i *)(p + 48), _mm_sub_epi64(_mm_add_epi16(a, b), _mm_xor_si128(a, b)));
}

// @diff simd_saturate: void(buf:64)
void simd_saturate(uint8_t *p) {
    __m128i a = _mm_loadu_si128((const __m128i *)p);
    __m128i b = _mm_loadu_si128((const __m128i *)(p + 16));
    __m128i c = _mm_loadu_si128((const __m128i *)(p + 32));
    __m128i r = _mm_subs_epu8(_mm_max_epu8(a, b), _mm_min_epu8(b, c));
    r = _mm_xor_si128(r, _mm_adds_epi16(a, c));
    r = _mm_or_si128(r, _mm_and_si128(_mm_avg_epu8(a, c), _mm_max_epi16(b, c)));
    _mm_storeu_si128((__m128i *)(p + 48), _mm_andnot_si128(_mm_cmpgt_epi8(a, b), r));
}

// AVX2: 32-byte registers, each instruction working on both 16-byte halves.
// @diff avx2_lanes: u32(buf:128)
__attribute__((target("avx2"))) uint32_t avx2_lanes(uint8_t *p) {
    __m256i a = _mm256_loadu_si256((const __m256i *)p);
    __m256i b = _mm256_loadu_si256((const __m256i *)(p + 32));
    __m256i c = _mm256_loadu_si256((const __m256i *)(p + 64));
    __m256i r = _mm256_xor_si256(_mm256_add_epi8(a, b), _mm256_sub_epi32(c, a));
    r = _mm256_or_si256(r, _mm256_andnot_si256(b, _mm256_cmpgt_epi8(c, b)));
    _mm256_storeu_si256((__m256i *)(p + 96), _mm256_and_si256(r, _mm256_max_epu8(a, c)));
    return (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, c));
}

// The first byte equal to `c` in 64 bytes, as memchr finds it; 64 if none.
// @diff avx2_find: u64(buf:64, u8:0..4)
__attribute__((target("avx2"))) uint64_t avx2_find(const uint8_t *p, uint8_t c) {
    __m256i n = _mm256_set1_epi8((char)c);
    for (uint64_t k = 0; k < 64; k += 32) {
        uint32_t m = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)(p + k)), n));
        if (m) return k + __builtin_ctz(m);
    }
    return 64;
}

// Feature checks ask the CPU: cpuid's leaf 0 and 1 (the vendor, family and
// feature bits, the same on every core) and xgetbv's enabled state.
// @diff cpu_info: u64(u64:0..2)
uint64_t cpu_info(uint64_t leaf) {
    unsigned a, b, c, d;
    __cpuid_count((unsigned)leaf, 0, a, b, c, d);
    if (leaf == 1) b &= 0xffff;  // the APIC id differs between cores
    uint64_t r = ((uint64_t)b << 32 | d) ^ ((uint64_t)a << 17) ^ c;
    if (leaf == 1 && (c >> 27 & 1)) {  // OSXSAVE
        unsigned lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        r += (uint64_t)hi << 32 | lo;
    }
    return r;
}

// Narrowing with saturation: packsswb, packuswb, packssdw.
// @diff simd_pack: void(buf:96)
void simd_pack(uint8_t *p) {
    __m128i a = _mm_loadu_si128((const __m128i *)p);
    __m128i b = _mm_loadu_si128((const __m128i *)(p + 16));
    _mm_storeu_si128((__m128i *)(p + 32), _mm_packus_epi16(a, b));
    _mm_storeu_si128((__m128i *)(p + 48), _mm_packs_epi16(b, a));
    _mm_storeu_si128((__m128i *)(p + 64), _mm_packs_epi32(a, b));
}

// @diff simd_shuffle: void(buf:48)
void simd_shuffle(uint8_t *p) {
    __m128i a = _mm_loadu_si128((const __m128i *)p);
    __m128i b = _mm_loadu_si128((const __m128i *)(p + 16));
    __m128i r = _mm_shuffle_epi32(a, 0x1b);
    r = _mm_add_epi8(r, _mm_shufflelo_epi16(b, 0x4e));
    r = _mm_xor_si128(r, _mm_unpacklo_epi8(a, b));
    r = _mm_add_epi16(r, _mm_srli_epi16(_mm_unpackhi_epi16(a, b), 3));
    r = _mm_sub_epi32(r, _mm_slli_si128(_mm_srai_epi32(b, 5), 4));
    _mm_storeu_si128((__m128i *)(p + 32), r);
}

// @diff simd_sad: u64(buf:32)
uint64_t simd_sad(const uint8_t *p) {
    __m128i a = _mm_loadu_si128((const __m128i *)p);
    __m128i b = _mm_loadu_si128((const __m128i *)(p + 16));
    __m128i s = _mm_sad_epu8(a, b);
    __m128i m = _mm_mul_epu32(a, b);
    __m128i w = _mm_mullo_epi16(a, b);
    s = _mm_add_epi64(_mm_add_epi64(s, m), _mm_srli_si128(w, 8));
    return (uint64_t)_mm_cvtsi128_si64(s) ^ ((uint64_t)_mm_extract_epi16(s, 5) << 7);
}

/* ---- floating point ---- */

// @diff f_mix: i64(i32, i32)
int64_t f_mix(int32_t a, int32_t b) {
    double x = a, y = b;
    double m = x > y ? x : y;
    return (int64_t)(x * 0.75 + y / 3.0 - m);
}

// @diff f_mix32: i32(i32:-1000..1000, i32:1..1000)
int32_t f_mix32(int32_t a, int32_t b) {
    float x = (float)a, y = (float)b;
    return (int32_t)((x / y) * 100.0f + (double)x * 0.5);
}

// Random bytes: NaNs and infinities come up among the f32 bit patterns.
// @diff f_order32: u32(buf:8)
uint32_t f_order32(const uint8_t *p) {
    float a, b;
    memcpy(&a, p, 4);
    memcpy(&b, p + 4, 4);
    return (a < b) | (a <= b) << 1 | (a == b) << 2 | (a != b) << 3 | (a > b) << 4 | (a >= b) << 5
        | __builtin_isunordered(a, b) << 6;
}

// @diff f_order64: u32(buf:16)
uint32_t f_order64(uint8_t *p) {
    double a, b;
    if (p[0] & 1) { // b: NaN half the time (random f64 bits almost never are)
        p[15] |= 0x7f;
        p[14] |= 0xf8;
    }
    memcpy(&a, p, 8);
    memcpy(&b, p + 8, 8);
    return (a < b) | (a <= b) << 1 | (a == b) << 2 | (a != b) << 3 | (a > b) << 4 | (a >= b) << 5;
}

// @diff f_trunc: i64(buf:8)
int64_t f_trunc(const uint8_t *p) {
    float a;
    memcpy(&a, p, 4);
    __m128 v = _mm_set_ss(a);
    return (int64_t)_mm_cvttss_si32(v) + _mm_cvttsd_si64(_mm_cvtss_sd(_mm_setzero_pd(), v));
}

// @diff f_vec: void(buf:48)
void f_vec(uint8_t *p) {
    __m128d a = _mm_loadu_pd((const double *)p);
    __m128d b = _mm_loadu_pd((const double *)(p + 16));
    __m128d r = _mm_add_pd(_mm_mul_pd(a, b), _mm_min_pd(a, b));
    r = _mm_and_pd(r, _mm_cmplt_pd(a, b));
    __m128 c = _mm_max_ps(_mm_castpd_ps(a), _mm_castpd_ps(b));
    r = _mm_xor_pd(r, _mm_castps_pd(_mm_sqrt_ps(c)));
    r = _mm_unpackhi_pd(r, _mm_div_pd(_mm_sqrt_pd(a), b));
    _mm_storeu_pd((double *)(p + 32), r);
}

/* ---- float arguments and results ---- */

// @diff f_add: f64(f64, f64)
double f_add(double a, double b) { return a + b * 2.0; }

// Integer and float arguments interleaved: each kind fills its own registers.
// @diff f_mixed: f64(i32, f64, i64, f32)
double f_mixed(int32_t a, double x, int64_t b, float y) { return x * a - (double)y + (double)(b >> 3); }

// @diff f_poly: f64(f64)
double f_poly(double x) { return ((0.25 * x - 1.5) * x + 3.0) * x - 7.0; }

// @diff f_cmp: i32(f64, f64)
int32_t f_cmp(double a, double b) { return (a < b) | (a <= b) << 1 | (a == b) << 2 | (a > b) << 3; }

NOINLINE double f_scale(double x, int32_t k) { return x * k + 0.5; }

// A float argument and result across a call.
// @diff f_call: f64(f64, i32:-50..50)
double f_call(double x, int32_t k) { return f_scale(x + 1.0, k) - f_scale(x, k + 1); }

// A tail call that returns the callee's float.
// @diff f_tail: f64(f64)
double f_tail(double x) { return f_scale(x * 3.0, 7); }

// @diff f_sum: f64(buf:64, u64:0..9)
double f_sum(const uint8_t *p, uint64_t n) {
    double s = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        double d;
        memcpy(&d, p + 8 * i, 8);
        s += d;
    }
    return s;
}

// @diff f32_ops: f32(f32, f32)
float f32_ops(float a, float b) { return a * b - a / 4.0f; }

// @diff f_trunc64: i64(f64)
int64_t f_trunc64(double x) { return (int64_t)x; }

// @diff f_select: f64(f64, f64, i32:0..2)
double f_select(double a, double b, int32_t c) { return c ? a - b : b * 0.5; }

// @diff f_abs: f64(f64)
double f_abs(double x) { return -__builtin_fabs(x) + 1.0; }

// @diff f_to_f32: f32(f64, i32)
float f_to_f32(double x, int32_t k) { return (float)x + (float)k; }

/* ---- arrays, function pointers, variadic calls ---- */

// A loop over one array stops at `keys + 16`, the address of whatever follows.
int64_t keys[16] = { 5, -3, 8, 1, 9, -7, 2, 6, 4, -1, 3, 7, -9, 11, 0, 13 };
int64_t vals[16] = { 50, 30, 80, 10, 90, 70, 20, 60, 40, 15, 33, 77, 99, 11, 1, 13 };

// @diff key_is: i32(i64, i64)
NOINLINE int key_is(int64_t a, int64_t b) { return a == b; }

// @diff lookup_key: i64(i64:-10..15)
int64_t lookup_key(int64_t k) {
    for (int i = 0; i < 16; i++)
        if (key_is(vals[i] - keys[i] * 10, k)) return vals[i];
    return -1;
}

// A loop walking down an array stops at `steps - 1`, inside the array before it.
int64_t steps[6] = { 1, 4, 13, 40, 121, 364 };

// @diff steps_down: u64(u64:0..400)
uint64_t steps_down(uint64_t n) {
    const int64_t *p = &steps[5];
    while (p != steps && (uint64_t)*p >= n) p--;
    uint64_t s = 0;
    for (; p != steps - 1; p--) s = mix(*p, s);
    return s;
}

struct pool { uint64_t (*release)(uint64_t opaque, uint64_t p); uint64_t opaque; };
// @diff pool_release: u64(u64, u64)
NOINLINE uint64_t pool_release(uint64_t opaque, uint64_t p) { return opaque * 31 + p; }
struct pool the_pool = { pool_release, 7 };

// Each pointer is loaded and tested before the call that passes it.
// @diff release_all: u64(buf:24)
uint64_t release_all(const uint64_t *v) {
    uint64_t s = 0;
    if (v[0]) s += the_pool.release(the_pool.opaque, v[0]);
    if (v[1]) s += the_pool.release(the_pool.opaque, v[1]);
    if (v[2]) s += the_pool.release(the_pool.opaque, v[2]);
    return s;
}

// @diff twice: u64(u64)
NOINLINE uint64_t twice(uint64_t x) { return x * 2 + 1; }
// @diff thrice: u64(u64)
NOINLINE uint64_t thrice(uint64_t x) { return x * 3 + 2; }
NOINLINE uint64_t call_with(uint64_t (*f)(uint64_t), uint64_t x) { return f(x) ^ x; }

// A function's address as a value.
// @diff pick_fn: u64(u64)
uint64_t pick_fn(uint64_t x) { return call_with((x & 1) ? twice : thrice, x); }

// An argument moved from another register into a variadic one (r9 for
// gcc's fortified `__snprintf_chk`, rcx for plain `snprintf`).
// @diff fmt_moved: void(buf:64, u64, u64)
void fmt_moved(char *b, uint64_t x, uint64_t y) { snprintf(b, 64, "%lu", (unsigned long)y); (void)x; }

// Both paths set the variadic arguments up for one call.
// @diff fmt_either: void(buf:64, u64, u64)
void fmt_either(char *b, uint64_t x, uint64_t y) {
    const char *f;
    uint64_t v;
    if (x > y) { f = "%lu >"; v = x / (y | 1); }
    else { f = "%lu <="; v = y % (x | 1); }
    snprintf(b, 64, f, (unsigned long)v);
}

// Variadic arguments past the sixth go on the stack.
// @diff fmt_many: void(buf:96, u64, u64, u64)
void fmt_many(char *b, uint64_t x, uint64_t y, uint64_t z) {
    snprintf(b, 96, "%lu %lu %lu %lu %lu %lu", (unsigned long)x, (unsigned long)y, (unsigned long)z,
             (unsigned long)(x ^ y), (unsigned long)(y + z), (unsigned long)(z * 3));
}

/* ---- shapes found in zlib, Lua and SQLite ---- */

// gcc divides with `xor edx, edx; idiv` when it knows the dividend is
// non-negative, instead of `cdq; idiv`.
// @diff div_nonneg: i32(u32, i32:1..1000)
int32_t div_nonneg(uint32_t a, int32_t b) { return (int32_t)(a & 0x7fffffff) / b + (int32_t)(a >> 1 & 0xffff) % b; }

// More than a page of locals: gcc's -fstack-clash-protection (Ubuntu's
// default) probes the frame a page at a time in a loop.
// @diff big_frame: u64(buf:64, u64:0..70000)
uint64_t big_frame(const uint8_t *b, uint64_t n) {
    volatile uint8_t tmp[70000];
    for (uint64_t i = 0; i < 70000; i += 997) tmp[i] = b[i % 64];
    tmp[n] = b[n % 64] ^ 0x5a;
    uint64_t s = 0;
    for (uint64_t i = 0; i < 70000; i += 997) s = s * 31 + tmp[i];
    return s + tmp[n];
}

// Cases that call a function that doesn't return go to the function's
// `.cold` part, outside the function: the jump table points there too.
// @diff switch_cold: u64(u64:0..9)
uint64_t switch_cold(uint64_t x) {
    switch (x) {
    case 0: return 11;
    case 1: return x * 7 + 3;
    case 2: return 42;
    case 3: return x << 4;
    case 4: return 99;
    case 5: return x ^ 0x55;
    case 6: return 1234;
    case 7: return x + 1000;
    case 8: return 8888;
    case 9: abort();
    case 10: abort();
    default: abort();
    }
}

// Computed goto through a table of labels, the index bounded by a mask
// rather than a compare (`jmp [r13 + rax*8]`, as in interpreters).
// @diff goto_table: u64(u64, u64)
uint64_t goto_table(uint64_t op, uint64_t x) {
    static void *const labels[] = { &&l_add, &&l_mul, &&l_xor, &&l_shl };
    uint64_t n = 4;
next:
    if (n-- == 0) return x;
    goto *labels[op & 3];
l_add: x += 0x1234; op = op / 4 + x; goto next;
l_mul: x *= 3; op = op / 4 + 1; goto next;
l_xor: x ^= op; op >>= 2; goto next;
l_shl: x = (x << 3) | (x >> 61); op = op / 4 + 2; goto next;
}

// fmod of doubles: gcc inlines it as an x87 `fprem` loop.
// @diff fmod_ints: i64(i64:-100000..100000, i64:1..1000)
int64_t fmod_ints(int64_t a, int64_t b) { return (int64_t)(fmod((double)a + 0.25, (double)b) * 1000.0); }

// A float argument to a variadic call, and a float result from a library
// call passed straight on to another.
// @diff fmt_float: void(buf:64, i64, i64:1..1000)
void fmt_float(char *b, int64_t x, int64_t y) { snprintf(b, 64, "%.3f %g", (double)x / (double)y, sqrt(fabs((double)x))); }

// strtod returns a double in xmm0.
// @diff parse_float: i64(i64:-100000..100000)
int64_t parse_float(int64_t x) {
    char b[32];
    snprintf(b, sizeof b, "%ld.5", (long)x);
    return (int64_t)(strtod(b, NULL) * 4.0);
}

// A call through a pointer passing the caller's own arguments on unchanged
// (`malloc(n) { return hooks.malloc(n); }`).
// @diff hook_impl: u64(u64, u64, u64)
NOINLINE uint64_t hook_impl(uint64_t a, uint64_t b, uint64_t c) { return a * 3 + (b ^ c); }
uint64_t (*volatile hook)(uint64_t, uint64_t, uint64_t) = hook_impl;
// @diff call_hook: u64(u64, u64, u64)
uint64_t call_hook(uint64_t a, uint64_t b, uint64_t c) { return hook(a, b, c); }

// A variadic function of the program: it saves its argument registers for
// `va_arg`, xmm ones only if the caller says (al) it passed floats.
NOINLINE int64_t vsum(int n, ...) {
    va_list ap;
    va_start(ap, n);
    int64_t s = 0;
    for (int i = 0; i < n; i++) s = s * 7 + va_arg(ap, int64_t);
    va_end(ap);
    return s;
}
NOINLINE double vfsum(int n, ...) {
    va_list ap;
    va_start(ap, n);
    double s = 0;
    for (int i = 0; i < n; i++) s = s * 2 + va_arg(ap, double);
    va_end(ap);
    return s;
}
// @diff call_vsum: i64(i64, i64, i64)
int64_t call_vsum(int64_t a, int64_t b, int64_t c) { return vsum(3, a, b, c) + (int64_t)vfsum(2, (double)(a & 0xffff), (double)(b & 0xff)); }

// A recursive function returning what its recursive call returns.
NOINLINE const char *skip_run(const char *s, char c) {
    if (*s != c) return s;
    return skip_run(s + 1, c);
}
// @diff run_len: u64(str:32)
uint64_t run_len(const char *s) { return (uint64_t)(skip_run(s, s[0]) - s); }

// A float result checked on the way out: the error path spills it to the
// frame around a call, and the caller's x87 `fmod` (`fnstsw ax`) writes ax
// after the call without reading the rest of rax.
volatile int64_t num_errors;
NOINLINE double num_of(int64_t x, int *ok) { *ok = x % 5 != 0; return (double)x * 0.5; }
NOINLINE void num_error(int64_t x) { num_errors += x; }
NOINLINE double check_num(int64_t x) {
    int ok;
    double d = num_of(x, &ok);
    if (!ok) num_error(x);
    return d;
}
// @diff fmod_checked: i64(i64:-100000..100000, i64:1..1000)
int64_t fmod_checked(int64_t a, int64_t b) { return (int64_t)(fmod(check_num(a), check_num(b)) * 1000.0); }

// An interpreter loop: the label table stays in a callee-saved register
// across calls, and one path returns from the middle of the function.
NOINLINE uint64_t op_step(uint64_t x) { return x * 31 + 7; }
// @diff goto_calls: u64(u64, u64)
uint64_t goto_calls(uint64_t code, uint64_t x) {
    static void *const ops[] = { &&o_call, &&o_add, &&o_ret, &&o_rot };
    for (int i = 0; i < 64; i++) {
        uint64_t op = code & 3;
        code = code >> 2 | code << 62;
        goto *ops[op];
    o_call: x = op_step(x); continue;
    o_add: x += code; continue;
    o_ret: if (x & 1) return x ^ code; continue;
    o_rot: x = op_step(x >> 3); continue;
    }
    return x;
}

// Two functions that tail-call each other: each returns what the other does.
NOINLINE int64_t walk_b(uint64_t x, int64_t n);
NOINLINE int64_t walk_a(uint64_t x, int64_t n) {
    if (n <= 0) return 101;
    if (x & 1) return walk_b(x >> 1, n - 1);
    return n * 3;
}
NOINLINE int64_t walk_b(uint64_t x, int64_t n) {
    if (n <= 0) return 7;
    if (x & 2) return walk_a(x >> 2, n - 1);
    return -n;
}
// @diff walk_ab: i64(u64, i64:0..40)
int64_t walk_ab(uint64_t x, int64_t n) { return walk_a(x, n) * 2 + 1; }

// A variadic function of the program called with more arguments than fit
// in registers: `va_arg` reads the rest from the caller's stack.
// @diff call_vsum_many: i64(i64, i64, i64)
int64_t call_vsum_many(int64_t a, int64_t b, int64_t c) { return vsum(8, a, b, c, a ^ b, b - c, c * 3, a + 7, b | 1); }
