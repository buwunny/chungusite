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
