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
