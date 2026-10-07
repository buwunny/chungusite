//! Text dump of a `Function`, for tests and debugging. Allocates; never on the hot path.
use crate::ir::*;
use std::fmt::{self, Write};

pub fn dump(f: &Function) -> String {
    let mut s = String::new();
    write_fn(&mut s, f).unwrap();
    s
}

fn v(id: ValueId) -> String { format!("v{}", id.index()) }

fn list(f: &Function, l: ListRef) -> String {
    l.get(&f.value_pool).iter().map(|&x| v(x)).collect::<Vec<_>>().join(", ")
}

fn write_fn(s: &mut String, f: &Function) -> fmt::Result {
    for (b, blk) in f.blocks.iter() {
        writeln!(s, "bb{}({}):", b.index(), list(f, blk.params))?;
        for &id in blk.insts.get(&f.value_pool) {
            let k = f.insts[id].kind;
            let body = match k {
                InstKind::Const(c) => format!("const {:#x}", f.consts[c.index()]),
                InstKind::Param(i) => format!("param r{i}"),
                InstKind::Bin { op, lhs, rhs } => format!("{op:?} {}, {}", v(lhs), v(rhs)),
                InstKind::Cmp { cc, lhs, rhs } => format!("cmp.{cc:?} {}, {}", v(lhs), v(rhs)),
                InstKind::Cast { kind, v: x } => format!("{kind:?} {}", v(x)),
                InstKind::PtrOffset { base, index, scale, disp } => match index {
                    Some(i) => format!("ptr {} + {}*{scale} + {disp}", v(base), v(i)),
                    None => format!("ptr {} + {disp}", v(base)),
                },
                InstKind::IntToPtr(x) => format!("inttoptr {}", v(x)),
                InstKind::Un { op, v: x } => format!("{op:?} {}", v(x)),
                InstKind::Select { c, t, f: e } => format!("select {}, {}, {}", v(c), v(t), v(e)),
                InstKind::Call { callee, args } => format!("call {}({})", v(callee), list(f, args)),
                InstKind::CallHi(x) => format!("callhi {}", v(x)),
                InstKind::Load { ptr, .. } => format!("load {}", v(ptr)),
                InstKind::Store { ptr, val, .. } => format!("store {} <- {}", v(ptr), v(val)),
                other => format!("{other:?}"),
            };
            match k {
                InstKind::Store { .. } => writeln!(s, "  {body}")?,
                _ => writeln!(s, "  {} = {body}", v(id))?,
            }
        }
        let t = match blk.term {
            Terminator::Jump { to, args } => format!("jump bb{}({})", to.index(), list(f, args)),
            Terminator::Branch { c, t, f: e, args } => {
                let a = args.get(&f.value_pool);
                let nt = f.blocks[t].params.len as usize;
                let show = |xs: &[ValueId]| xs.iter().map(|&x| v(x)).collect::<Vec<_>>().join(", ");
                format!("br {} bb{}({}) bb{}({})", v(c), t.index(), show(&a[..nt]), e.index(), show(&a[nt..]))
            }
            Terminator::Return(r) => format!("ret {}", r.map(v).unwrap_or_default()),
            other => format!("{other:?}"),
        };
        writeln!(s, "  {t}")?;
    }
    Ok(())
}
