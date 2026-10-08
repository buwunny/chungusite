//! Rust expression text: just enough precedence to know when an expression the
//! emitter inlines into another one needs parentheses, and to negate a condition
//! without wrapping it in `!( )`.
//!
//! The emitter only builds a handful of shapes (names, literals, method calls,
//! `as` casts, prefix `!`, binary operators with spaces around them, `if`/`unsafe`
//! blocks), so a scan for top-level spaces is enough to tell them apart.

/// How tightly an expression binds, loosest last.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Prec {
    /// A name, literal, call, method call or parenthesized expression.
    Atom,
    /// `!x`, `-x`.
    Unary,
    /// `x as T as U`.
    Cast,
    /// Anything else: binary operators, `if`, `unsafe { }` blocks.
    Other,
}

/// Split `s` at spaces outside brackets and string literals.
pub fn words(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start, mut quoted) = (0i32, 0, false);
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'\\' if quoted => i += 1,
            b'"' => quoted = !quoted,
            _ if quoted => {}
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b' ' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    out.push(&s[start..]);
    out
}

pub fn prec(s: &str) -> Prec {
    let w = words(s);
    if w.len() == 1 {
        return if s.starts_with('!') || s.starts_with('-') { Prec::Unary } else { Prec::Atom };
    }
    // `x as T as U`: odd positions are all `as`, and the types are single words.
    if w.len() % 2 == 1 && w.iter().skip(1).step_by(2).all(|&k| k == "as") {
        return Prec::Cast;
    }
    Prec::Other
}

fn paren_unless(s: String, ok: bool) -> String {
    if ok { s } else { format!("({s})") }
}

/// `s` as the receiver of a method call: `s.f()`.
pub fn recv(s: String) -> String {
    let ok = prec(&s) == Prec::Atom;
    paren_unless(s, ok)
}

/// `s` as the operand of a prefix operator: `!s`.
pub fn unary(s: String) -> String {
    let ok = prec(&s) <= Prec::Unary;
    paren_unless(s, ok)
}

/// `s` as the operand of `as`: `s as T`.
pub fn cast(s: String) -> String {
    let ok = prec(&s) <= Prec::Cast;
    paren_unless(s, ok)
}

/// `s` as the left operand of binary operator `op`. A cast reads fine before
/// `==` or `&`, but `x as u8 < y` doesn't parse (`<` starts generic arguments).
pub fn lhs(s: String, op: &str) -> String {
    let p = prec(&s);
    let ok = p <= Prec::Unary || (p == Prec::Cast && matches!(op, "==" | "!=" | "&" | "|" | "^"));
    paren_unless(s, ok)
}

/// `s` as the right operand of a binary operator.
pub fn rhs(s: String) -> String {
    let ok = prec(&s) <= Prec::Cast;
    paren_unless(s, ok)
}

/// `s` as an operand of `&&` (`and`) or `||`. Comparisons and bit operators bind
/// tighter, so only the other one of the two (needed for `||` inside `&&`, for
/// clarity the other way round) and block expressions get parentheses.
pub fn logic(s: String, and: bool) -> String {
    let w = words(&s);
    let other = if and { "||" } else { "&&" };
    let ok = !matches!(w[0], "if" | "unsafe" | "match" | "loop") && !w.contains(&other);
    paren_unless(s, ok)
}

/// Remove one pair of parentheses around the whole of `s`.
fn unparen(s: &str) -> &str {
    if s.starts_with('(') && s.ends_with(')') && words(s).len() == 1 {
        let inner = &s[1..s.len() - 1];
        // `(a).f(b)` also starts and ends with parentheses: the opening one must
        // close at the very end.
        let mut depth = 0;
        for (i, c) in s.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return if i == s.len() - 1 { inner } else { s };
                    }
                }
                _ => {}
            }
        }
    }
    s
}

/// The negation of condition `c`: flip a comparison, push `!` through `&&` and
/// `||`, drop a leading `!`, and only otherwise write `!c` or `!(c)`.
pub fn not(c: &str) -> String {
    let c = unparen(c);
    let w = words(c);
    for (op, flip) in [("||", " && "), ("&&", " || ")] {
        if w.contains(&op) {
            let parts = split(&w, op);
            let and = flip == " && ";
            return parts.iter().map(|p| logic(not(p), and)).collect::<Vec<_>>().join(flip);
        }
    }
    let flip = |op: &str| match op {
        "==" => Some("!="),
        "!=" => Some("=="),
        "<" => Some(">="),
        ">=" => Some("<"),
        ">" => Some("<="),
        "<=" => Some(">"),
        _ => None,
    };
    let ops: Vec<usize> = (0..w.len()).filter(|&i| flip(w[i]).is_some()).collect();
    if let [i] = ops[..] {
        // `a >= b` for `!(a < b)`, but a cast before `<` needs parentheses.
        let f = flip(w[i]).unwrap();
        return format!("{} {f} {}", lhs(w[..i].join(" "), f), w[i + 1..].join(" "));
    }
    if w.len() == 1 {
        if let Some(inner) = c.strip_prefix('!') {
            return unparen(inner).to_string();
        }
    }
    format!("!{}", unary(c.to_string()))
}

/// The operands of `op` in `w` (a list of top-level words), rejoined.
fn split(w: &[&str], op: &str) -> Vec<String> {
    w.split(|&x| x == op).map(|p| p.join(" ")).collect()
}

/// An integer literal: small values in decimal, the rest in hex.
pub fn lit(v: u64) -> String {
    if v < 10 { v.to_string() } else { format!("{v:#x}") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedence() {
        assert_eq!(prec("v1"), Prec::Atom);
        assert_eq!(prec("v1.wrapping_add(v2 as u64)"), Prec::Atom);
        assert_eq!(prec("!v1"), Prec::Unary);
        assert_eq!(prec("v1 as u32 as u64"), Prec::Cast);
        assert_eq!(prec("v1 == 3"), Prec::Other);
        assert_eq!(prec("unsafe { (v1 as *const u8).read_unaligned() }"), Prec::Other);
        assert_eq!(prec("todo!(\"a ( b\")"), Prec::Atom);
    }

    #[test]
    fn negation() {
        assert_eq!(not("a == b"), "a != b");
        assert_eq!(not("(a as i32) < (b as i32)"), "(a as i32) >= (b as i32)");
        assert_eq!(not("(a as u8) > 9"), "(a as u8) <= 9");
        assert_eq!(not("!v3"), "v3");
        assert_eq!(not("!(a & b)"), "a & b");
        assert_eq!(not("v3"), "!v3");
        assert_eq!(not("a & b"), "!(a & b)");
        assert_eq!(not("a == 0 || b < c"), "a != 0 && b >= c");
        assert_eq!(not("a && b || c"), "(!a || !b) && !c");
        assert_eq!(not("(a == 0)"), "a != 0");
        assert_eq!(not("a as u32 != 1"), "a as u32 == 1");
        assert_eq!(not("a as u32 >= b as u32"), "(a as u32) < b as u32");
        assert_eq!(not("(a).f() == (b)"), "(a).f() != (b)");
    }
}
