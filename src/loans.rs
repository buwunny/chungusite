//! Polonius-style loan and move checking (docs/ownership.md, stage 6), on the
//! `datafrog` engine that Polonius itself runs on.
//!
//! The borrow analysis (`borrow.rs`) generates the input facts from the SSA and
//! CFG; this module only runs the rules. Points are positions in the CFG: one per
//! instruction, plus one per terminator, numbered by the caller.
//!
//! Loans:
//!
//! ```text
//! contains(O, L, P)  :- loan_issued_at(O, L, P).
//! contains(O2, L, P) :- contains(O1, L, P), subset(O1, O2, P).
//! contains(O, L, Q)  :- contains(O, L, P), cfg_edge(P, Q), !loan_killed_at(L, P),
//!                       origin_live_at(O, Q).
//! loan_live_at(L, P) :- contains(O, L, P), origin_live_at(O, P).
//! error(L, P)        :- invalidates(P, L), loan_live_at(L, P).
//! ```
//!
//! Moves (an owned path that may have been moved out, then used):
//!
//! ```text
//! maybe_moved(X, Q) :- moved_at(X, P), cfg_edge(P, Q).
//! maybe_moved(X, Q) :- maybe_moved(X, P), cfg_edge(P, Q), !assigned_at(X, P).
//! move_error(X, P)  :- maybe_moved(X, P), accessed_at(X, P).
//! ```
//!
//! An assignment at `P` (re-initialising `X`) stops `maybe_moved` from flowing
//! past `P`.
use datafrog::{Iteration, Relation, RelationLeaper};
use std::collections::HashSet;

pub type Point = u32;
pub type Loan = u32;
pub type Origin = u32;
pub type Path = u32;

#[derive(Default, Debug, Clone)]
pub struct Facts {
    pub cfg_edge: Vec<(Point, Point)>,
    pub loan_issued_at: Vec<(Origin, Loan, Point)>,
    pub loan_killed_at: Vec<(Loan, Point)>,
    pub origin_live_at: Vec<(Origin, Point)>,
    pub subset: Vec<(Origin, Origin, Point)>,
    pub invalidates: Vec<(Point, Loan)>,
    pub moved_at: Vec<(Path, Point)>,
    pub assigned_at: Vec<(Path, Point)>,
    pub accessed_at: Vec<(Path, Point)>,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Output {
    /// A loan is live where something invalidates it, sorted.
    pub errors: Vec<(Loan, Point)>,
    /// A path that may have been moved out is used, sorted.
    pub move_errors: Vec<(Path, Point)>,
}

pub fn solve(facts: &Facts) -> Output {
    Output { errors: loan_errors(facts), move_errors: move_errors(facts) }
}

fn loan_errors(facts: &Facts) -> Vec<(Loan, Point)> {
    if facts.loan_issued_at.is_empty() || facts.invalidates.is_empty() {
        return Vec::new();
    }
    let mut iteration = Iteration::new();
    // ((origin, point), loan)
    let contains = iteration.variable::<((Origin, Point), Loan)>("contains");
    // ((origin, point), loan): `contains` moved along a CFG edge, before the
    // liveness filter
    let stepped = iteration.variable::<((Origin, Point), Loan)>("stepped");

    let edge: Relation<(Point, Point)> = facts.cfg_edge.iter().copied().collect();
    let killed: Relation<(Loan, Point)> = facts.loan_killed_at.iter().copied().collect();
    let live: Relation<((Origin, Point), ())> = facts.origin_live_at.iter().map(|&(o, p)| ((o, p), ())).collect();
    let subset: Relation<((Origin, Point), Origin)> = facts.subset.iter().map(|&(a, b, p)| ((a, p), b)).collect();

    contains.extend(facts.loan_issued_at.iter().map(|&(o, l, p)| ((o, p), l)));
    while iteration.changed() {
        contains.from_join(&contains, &subset, |&(_, p), &l, &o2| ((o2, p), l));
        stepped.from_leapjoin(
            &contains,
            (
                edge.extend_with(|&((_, p), _)| p),
                killed.filter_anti(|&((_, p), l)| (l, p)),
            ),
            |&((o, _), l), &q| ((o, q), l),
        );
        contains.from_join(&stepped, &live, |&(o, q), &l, &()| ((o, q), l));
    }
    let contains = contains.complete();
    let live: HashSet<(Origin, Point)> = facts.origin_live_at.iter().copied().collect();
    let loan_live: HashSet<(Loan, Point)> =
        contains.iter().filter(|((o, p), _)| live.contains(&(*o, *p))).map(|&((_, p), l)| (l, p)).collect();
    let mut errors: Vec<(Loan, Point)> =
        facts.invalidates.iter().map(|&(p, l)| (l, p)).filter(|x| loan_live.contains(x)).collect();
    errors.sort_unstable();
    errors.dedup();
    errors
}

fn move_errors(facts: &Facts) -> Vec<(Path, Point)> {
    if facts.moved_at.is_empty() {
        return Vec::new();
    }
    let mut iteration = Iteration::new();
    // (point, path)
    let maybe_moved = iteration.variable::<(Point, Path)>("maybe_moved");
    let edge: Relation<(Point, Point)> = facts.cfg_edge.iter().copied().collect();
    let assigned: Relation<(Path, Point)> = facts.assigned_at.iter().copied().collect();

    let moved: HashSet<(Path, Point)> = facts.moved_at.iter().copied().collect();
    let mut seed = Vec::new();
    for &(p, q) in &facts.cfg_edge {
        for &(x, at) in &moved {
            if at == p {
                seed.push((q, x));
            }
        }
    }
    maybe_moved.extend(seed);
    while iteration.changed() {
        maybe_moved.from_leapjoin(
            &maybe_moved,
            (edge.extend_with(|&(p, _)| p), assigned.filter_anti(|&(p, x)| (x, p))),
            |&(_, x), &q| (q, x),
        );
    }
    let maybe_moved = maybe_moved.complete();
    let mm: HashSet<(Path, Point)> = maybe_moved.iter().map(|&(p, x)| (x, p)).collect();
    let mut errors: Vec<(Path, Point)> = facts.accessed_at.iter().copied().filter(|x| mm.contains(x)).collect();
    errors.sort_unstable();
    errors.dedup();
    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 0 -> 1 -> 2 -> 3, one loan issued at 0 whose origin is live at 0..=2.
    fn line() -> Facts {
        Facts {
            cfg_edge: vec![(0, 1), (1, 2), (2, 3)],
            loan_issued_at: vec![(0, 0, 0)],
            origin_live_at: vec![(0, 0), (0, 1), (0, 2)],
            ..Facts::default()
        }
    }

    #[test]
    fn a_write_while_the_loan_is_live_is_an_error() {
        let f = Facts { invalidates: vec![(2, 0), (3, 0)], ..line() };
        assert_eq!(solve(&f).errors, vec![(0, 2)]);
    }

    #[test]
    fn a_killed_loan_stops_flowing() {
        let f = Facts { invalidates: vec![(2, 0)], loan_killed_at: vec![(0, 1)], ..line() };
        assert!(solve(&f).errors.is_empty());
    }

    #[test]
    fn loans_flow_through_subsets() {
        // origin 0 flows into origin 1 at point 0; only origin 1 is live later
        let f = Facts {
            cfg_edge: vec![(0, 1), (1, 2)],
            loan_issued_at: vec![(0, 7, 0)],
            subset: vec![(0, 1, 0)],
            origin_live_at: vec![(0, 0), (1, 0), (1, 1), (1, 2)],
            invalidates: vec![(2, 7)],
            ..Facts::default()
        };
        assert_eq!(solve(&f).errors, vec![(7, 2)]);
    }

    #[test]
    fn use_after_move_on_one_path() {
        // 0 -> 1 (moves x) -> 3, 0 -> 2 -> 3; x used at 3, re-assigned at 4 then used at 5
        let f = Facts {
            cfg_edge: vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)],
            moved_at: vec![(9, 1)],
            assigned_at: vec![(9, 4)],
            accessed_at: vec![(9, 0), (9, 3), (9, 5)],
            ..Facts::default()
        };
        assert_eq!(solve(&f).move_errors, vec![(9, 3)]);
    }
}
