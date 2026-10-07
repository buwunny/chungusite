//! Rust identifiers for symbol names.
use std::collections::HashSet;

/// A Rust identifier for `name` that isn't in `used` yet (adding it): every
/// character outside `[A-Za-z0-9_]` becomes `_`, so `core::fmt::write` is
/// `core__fmt__write`, and a clash gets the address appended.
pub fn rust_ident(name: &str, addr: u64, used: &mut HashSet<String>) -> String {
    let mut s = sanitize(name);
    if !used.insert(s.clone()) {
        s = format!("{s}_{addr:x}");
        while !used.insert(s.clone()) {
            s.push('_');
        }
    }
    s
}

pub fn sanitize(name: &str) -> String {
    let mut s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
    // `<T as Trait>::f` would start and end with underscores that say nothing.
    if s != name {
        s = s.trim_matches('_').to_string();
    }
    if s.is_empty() || s.starts_with(|c: char| c.is_ascii_digit()) || is_keyword(&s) {
        s.insert_str(0, "f_");
    }
    s
}

fn is_keyword(s: &str) -> bool {
    matches!(
        s,
        "as" | "break" | "const" | "continue" | "crate" | "else" | "enum" | "extern" | "false" | "fn" | "for"
            | "if" | "impl" | "in" | "let" | "loop" | "match" | "mod" | "move" | "mut" | "pub" | "ref"
            | "return" | "self" | "Self" | "static" | "struct" | "super" | "trait" | "true" | "type"
            | "unsafe" | "use" | "where" | "while" | "async" | "await" | "dyn" | "abstract" | "become"
            | "box" | "do" | "final" | "macro" | "override" | "priv" | "typeof" | "unsized" | "virtual"
            | "yield" | "try" | "gen" | "_"
    )
}
