"""
filter-map: loops over the patches of a filtered column take them from the patch filter.

    do p = col_pp%pfti(c), col_pp%pftf(c)       ->   do fp = soilc_soilp_beg(fc), soilc_soilp_end(fc)
       ...                                              p = filter_soilp(fp)
                                                        ...
    do pi = 1, maxpatch_pft                     ->   do fp = soilc_soilp_beg(fc), soilc_soilp_end(fc)
       if (pi <= col_pp%npfts(c)) then                  p = filter_soilp(fp)
          p = col_pp%pfti(c) + pi - 1                   ...
          ...
       end if

The loops and the verdict whether the mapped loop is the same come from core/filter_maps.py
(only exact ones are rewritten: always with `filter_inactive_and_active`, with `filter` when the
body runs only for active patches). The patches are visited in the same ascending order, so sums
over them keep their order.

The map (clumpfilter%soilc_soilp_beg/end, built with the filters) reaches the routine as two
dummies, `soilc_soilp_beg(:)` and `soilc_soilp_end(:)`, declared after its column filter dummy;
a routine without a dummy for the soil patch filter of the same clumpfilter also gets
`filter_soilp(:)` (and so does every caller that has to pass it on).
Every analyzed call of the routine passes them as keyword actuals: `<filter>%soilc_soilp_beg`
where the caller passes a clumpfilter member (`filter(nc)%soilc` gives
`filter(nc)%soilc_soilp_beg`), or the caller's own map dummies where it passes its own column
filter dummy, which then get added to the caller in the same way, up to the driver.

Refused: a routine some visit of which binds its column filter to anything but a clumpfilter
soil filter, or whose loop is not exact in every visit; a call whose column filter actual is neither a clumpfilter member
nor a dummy; slot loops that use the slot index other than in the guard and the patch assignment;
headers or assignments over several lines; compute constructs other than a `parallel/kernels loop`
around the loop. Callers the analysis does not contain are not seen: the build finds them. The new private variables (fp, and p where it is no longer a loop index) are
added to the enclosing compute loop's private list.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from spel.scripts.fortran_parser import spel_ast as A
from spel.scripts.fortran_parser.scope_walk import Access
from spel.transforms.core.edits import insert, replace, replace_cols, span_text
from spel.transforms.core.fortran import fresh_name
from spel.transforms.core.registry import transform
from spel.transforms.core.source import code, statement_end
from spel.transforms.core.transform import Change, Plan, Refusal, Transform

if TYPE_CHECKING:
    from spel.scripts.analyze_subroutines import Subroutine
    from spel.scripts.fortran_parser.scope_walk import LoopInfo, SubroutineRecord

BEG, END = "soilc_soilp_beg", "soilc_soilp_end"
TAG = "! spel filter-map"


@dataclass
class Options:
    root: str = field(default="elm_driver::elm_drv", metadata={"help": "routine whose call tree is visited"})
    routines: list[str] = field(default_factory=list, metadata={"help": "only loops of these routine ids"})


class Skip(Exception):
    pass


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


@transform
class FilterMapLoops(Transform):
    """
    Matches: loops over the patches of a soil column (`do p = pfti(c), pftf(c)`, or slots
             `p = pfti(c) + pi - 1`) whose column comes from a clumpfilter soil column filter and
             that core/filter_maps.py finds exact in every visit (and active in the configuration).
    Changes: the loop runs over the column's range of the soil patch filter
             (`do fp = soilc_soilp_beg(fc), soilc_soilp_end(fc); p = filter_soilp(fp)`); the routine
             and its callers up to the driver get the map as two dummies / keyword actuals.
    Refuses: see the module docstring.
    Example: CNLitterToColumn's per-column patch sums over the soil filter's patches.
    """

    name = "filter-map"
    summary = "loops over a column's patches take them from the patch filter (soilc -> soilp map)"
    Options = Options

    def build(self, ctx, opts) -> Plan:
        from spel.transforms.core import filter_maps as fm

        plan = Plan(self.name)
        subs = ctx.fut.subroutine_dict
        routines = None
        if ctx.config is not None:
            routines = set(ctx.config.call_tree(ctx.fut, (opts.root,)))
        # every visit of every routine: its env (dummy -> paths) and its child loops
        visits: dict[str, list] = defaultdict(list)
        loops: dict[tuple[str, int], list] = defaultdict(list)
        for ictx in fm.contexts(ctx.fut, opts.root):
            visits[ictx.sub_id].append(ictx)
            for cl in fm.child_loops(ictx):
                loops[(cl.sub_id, cl.loop.start_ln)].append((ictx, cl))

        # targets: per routine, the loops exact in every visit
        targets: dict[str, list] = defaultdict(list)
        for (sid, ln), found in sorted(loops.items()):
            where = f"{sid}:{ln + 1}"
            if opts.routines and sid not in opts.routines:
                continue
            if routines is not None and sid not in routines:
                continue
            cl0 = found[0][1]
            if not ctx.active(subs[sid], cl0.loop.start_ln, cl0.loop.end_ln):
                continue
            if len(found) != len(visits[sid]):
                plan.add(Refusal(where, "the column filter is not a clumpfilter soil filter in every visit"))
                continue
            bad = [cl for _, cl in found if not cl.exact]
            if bad:
                plan.add(Refusal(where, f"not exact: {bad[0].reason}"))
                continue
            if any(cl.parent_pos is None or cl.parent_array is None for _, cl in found):
                plan.add(Refusal(where, "the column's filter position or filter array is unknown"))
                continue
            targets[sid].append(found)

        # the routines that get the map dummies: name of their column filter dummy, and the
        # call sites that pass it (caller id, call line, actual of the column filter)
        need: dict[str, str] = {}
        for sid, found_loops in sorted(targets.items()):
            sub = subs[sid]
            arrays = {cl.parent_array for found in found_loops for _, cl in found}
            if len(arrays) != 1:
                plan.add(Refusal(sid, f"columns from several filters ({', '.join(sorted(arrays))})"))
                continue
            (arr,) = arrays
            sym = sub.record.lookup(arr)
            if sym is None or sym.origin.name != "DUMMY":
                plan.add(Refusal(sid, f"{arr} is not a dummy"))
                continue
            need[sid] = arr

        calls, failed = self.close_over_callers(ctx, need, plan)
        for sid in failed:
            need.pop(sid, None)
            targets.pop(sid, None)

        # the soil patch filter: the routine's dummy bound to it, or a new dummy `filter_soilp`
        # (the targets need it, and so does a caller that passes a dummy to a routine getting one)
        pf = {sid: self.patch_filter_dummy(visits.get(sid, []), need[sid]) for sid in need}
        new_pf: set[str] = set()
        want = set(targets)
        while True:
            new_pf = {sid for sid in want if pf[sid] is None}
            more = {caller for (caller, _), (callee, text) in calls.items()
                    if callee in new_pf and text is None and caller in need} - want
            if not more:
                break
            want |= more
        pf_name = {sid: pf[sid] or fresh_name("filter_soilp", self.declared(subs[sid].record)) for sid in need}

        for sid in sorted(need):
            sub = subs[sid]
            try:
                edits = self.routine_edits(ctx, sub, need[sid], targets.get(sid, []), pf_name[sid],
                                           sid in new_pf)
            except Skip as err:
                plan.add(Refusal(sid, str(err)))
                continue
            why = (f"{len(targets.get(sid, []))} loops over soilc patches use the soilc -> soilp map"
                   if sid in targets else "passes the soilc -> soilp map on")
            plan.add(Change(ctx.path(sub), edits, where=sid, why=why))
        for (caller, ln), (callee, text) in sorted(calls.items()):
            if callee not in need:
                continue
            csub = subs[caller]
            extra = None
            if callee in new_pf:
                extra = (pf_name[callee], pf_name.get(caller) if text is None else None)
            try:
                edit = self.call_edit(ctx, csub, ln, callee, text, extra)
            except Skip as err:
                plan.add(Refusal(f"{caller}:{ln + 1}", str(err)))
                continue
            plan.add(Change(ctx.path(csub), [edit], where=f"{caller}:{ln + 1}",
                            why=f"passes the soilc -> soilp map to {callee.split('::')[-1]}"))
        if need:
            plan.add("callers outside the analysis (e.g. elm_interface_funcsMod::elm_bgc_run, PFLOTRAN) are not "
                     "updated: build the case; a missing actual is error #6631 (Intel) / S-0186 (nvfortran)")
        return plan

    # --- analysis ---------------------------------------------------------------------------
    @staticmethod
    def patch_filter_dummy(visits, col_dummy: str) -> Optional[str]:
        """The dummy bound in every visit to the soilp filter of the clumpfilter of `col_dummy`."""
        cands = None
        for ictx in visits:  # no visits: unknown (a new dummy is added)
            insts = {p.rsplit("%", 1)[0] for p in ictx.env.get(col_dummy, ())}
            want = {f"{i}%soilp" for i in insts}
            here = {d for d, paths in ictx.env.items() if paths and set(paths) == want}
            cands = here if cands is None else cands & here
        return sorted(cands)[0] if cands else None

    def close_over_callers(self, ctx, need: dict[str, str], plan: Plan):
        """
        Call sites to edit, (caller, line) -> (callee, actual text or None for a dummy), adding the
        callers that pass a dummy of their own to `need`. Returns them and the routines whose map
        cannot be supplied (with everything that would depend on them).
        """
        subs = ctx.fut.subroutine_dict
        callers: dict[str, list] = defaultdict(list)  # callee -> [(caller id, ArgBinding)]
        for s in subs.values():
            maps = s.record_access
            for b in maps.bindings if maps is not None else []:
                callers[b.callee].append((s.id, b))
        calls: dict[tuple[str, int], tuple[str, Optional[str]]] = {}
        depends: dict[str, set] = defaultdict(set)  # caller -> callees it passes the map to
        failed: set[str] = set()
        todo = sorted(need)
        while todo:
            sid = todo.pop()
            dummy = need[sid]
            sites = defaultdict(list)
            for caller, b in callers.get(sid, []):
                sites[(caller, b.ln)].append(b)
            for (caller, ln), binds in sorted(sites.items()):
                b = next((x for x in binds if x.dummy.lower() == dummy.lower()), None)
                where = f"{caller}:{ln + 1}"
                if b is None:
                    plan.add(Refusal(where, f"call of {sid} without its {dummy} argument"))
                    failed.add(sid)
                    continue
                if b.origin.name in ("GLOBAL", "EXTERNAL") and b.actual.lower().endswith("%soilc"):
                    calls[(caller, ln)] = (sid, b.actual)
                elif b.origin.name == "DUMMY" and "%" not in b.actual:
                    calls[(caller, ln)] = (sid, None)
                    depends[caller].add(sid)
                    if caller not in need:
                        need[caller] = b.actual
                        todo.append(caller)
                    elif need[caller].lower() != b.actual.lower():
                        plan.add(Refusal(where, f"{caller} passes two column filters on"))
                        failed.add(sid)
                else:
                    plan.add(Refusal(where, f"{sid}'s {dummy} is bound to {b.actual} (not a clumpfilter soilc)"))
                    failed.add(sid)
        # a routine fails with every callee that needs it, and every caller only passing it on
        changed = True
        while changed:
            changed = False
            for caller, callees in depends.items():
                if caller in failed and not callees <= failed:
                    failed |= callees
                    changed = True
        return calls, failed

    # --- edits ------------------------------------------------------------------------------
    def tree(self, ctx, sub: Subroutine) -> A.SubroutineDefinitionConstruct:
        from spel.scripts.registrations import parse_routine

        end = sub.end_stmt_ln or sub.endline
        try:
            return parse_routine(sub.filepath, sub.startline, end, keep_spans=True)
        except BaseException as err:  # the parser exits on fatal errors
            raise Skip(f"does not parse: {err}")

    def routine_edits(self, ctx, sub: Subroutine, col_dummy: str, found_loops, pf: str, pf_new: bool) -> list:
        lines = ctx.lines(sub)
        rec = sub.record
        tree = self.tree(ctx, sub)
        edits = []
        # dummies: appended to the dummy list
        close = tree.args_close
        if close is None or lines[close[0]][close[1]:close[3]] != ")":
            raise Skip("no closing parenthesis of the dummy list")
        if any(rec.declaration(n) is not None for n in (BEG, END)):
            raise Skip(f"already declares {BEG}/{END}")
        added = f"{BEG}, {END}" + (f", {pf}" if pf_new else "")
        edits.append(self.append_text(lines, close[0], close[1], added))
        # declarations: after the column filter's
        decl = rec.declaration(col_dummy)
        if decl is None:
            raise Skip(f"no declaration of {col_dummy}")
        dln = statement_end(lines, decl.ln)
        ind = _indent(lines[decl.ln])
        new = [f"{ind}integer, intent(in) :: {BEG}(:)  ! the soil patches of soil column fc are",
               f"{ind}integer, intent(in) :: {END}(:)  !   {pf}({BEG}(fc):{END}(fc))"]
        if pf_new:
            new.append(f"{ind}integer, intent(in) :: {pf}(:)  ! filter for soil patches")
        if not found_loops:
            edits.append(insert(dln + 1, new, why="the soilc -> soilp map"))
            return edits
        # the patch position index
        taken = {d.lower() for d in self.declared(rec)}
        fp = "fp"
        loop_lines = [ln for found in found_loops for ln in range(found[0][1].loop.start_ln, found[0][1].loop.end_ln + 1)]
        if fp not in taken or any(isinstance(a, Access) and a.ln in set(loop_lines) and a.path == fp
                                  for a in rec.events):
            fp = fresh_name("fp_map", taken)
            new.append(f"{ind}integer :: {fp}  {TAG}: position in the soil patch filter")
        elif fp in taken and rec.declaration(fp).dim != 0:
            raise Skip("fp is not a scalar")
        edits.append(insert(dln + 1, new, why="the soilc -> soilp map"))
        privates: dict[int, set] = defaultdict(set)
        from spel.transforms.acc_directives import find_directives
        from spel.transforms.core.query import compute_ranges

        directives = find_directives(lines, sub.startline, sub.endline)
        ranges = list(compute_ranges(directives, rec))
        for found in found_loops:
            cl = found[0][1]
            edits += self.loop_edits(lines, tree, rec, cl, fp, pf)
            enclosing = [(d, a, b) for d, a, b in ranges if a <= cl.loop.start_ln <= b]
            if enclosing:
                d, _, _ = max(enclosing, key=lambda r: r[1])
                if not d.is_compute_loop:
                    raise Skip(f"line {cl.loop.start_ln + 1}: in a compute region without a compute loop directive")
                have = {v.lower() for v in d.var_list("private")}
                want = {fp} | ({cl.child} if cl.style == "range" else set())
                privates[d.end_ln] |= {v for v in want if v.lower() not in have}
        for ln, names in privates.items():
            if names:
                line = lines[ln]
                c = len(line.rstrip())  # a directive line is a comment to code()
                edits.append(replace(ln, [line[:c] + " &" + line[c:],
                                          f"{_indent(line)}!$acc&   private({', '.join(sorted(names))})"],
                                     expect=[line], why="private filter-map indices"))
        return edits

    @staticmethod
    def declared(rec: SubroutineRecord) -> set:
        return {n.lower() for n in rec.declarations}

    def loop_edits(self, lines, tree, rec, cl, fp: str, pf: str) -> list:
        loop = cl.loop
        h = loop.start_ln
        if statement_end(lines, h) != h:
            raise Skip(f"line {h + 1}: the loop header continues")
        ind = _indent(lines[h])
        header = f"{ind}do {fp} = {BEG}({cl.parent_pos}), {END}({cl.parent_pos})  {TAG}"
        if cl.style == "range":
            return [replace(h, [header, f"{ind}   {cl.child} = {pf}({fp})"], expect=[lines[h]],
                            why="patches from the soil patch filter")]
        # slots: the patch assignment becomes the gather, the `pi <= npfts(c)` guard goes
        a = cl.assign_ln
        if statement_end(lines, a) != a:
            raise Skip(f"line {a + 1}: the patch assignment continues")
        out = [replace(h, [header], expect=[lines[h]], why="patches from the soil patch filter"),
               replace(a, [f"{_indent(lines[a])}{cl.child} = {pf}({fp})"], expect=[lines[a]],
                       why="patches from the soil patch filter")]
        guard = self.slot_guard(tree, loop)
        used = {e.ln for e in rec.events if isinstance(e, Access) and e.path == loop.index
                and loop.start_ln < e.ln <= loop.end_ln}
        allowed = {a} | ({guard[0]} if guard else set())
        if used - allowed:
            raise Skip(f"line {min(used - allowed) + 1}: the slot index {loop.index} is used in the body")
        if guard is not None:
            g0, g1 = guard
            if statement_end(lines, g0) != g0:
                raise Skip(f"line {g0 + 1}: the slot guard continues")
            out += [replace(g0, [f"{_indent(lines[g0])}! (all patches of the range belong to the column)"],
                            expect=[lines[g0]], why="slot guard"),
                    replace(g1, [], expect=[lines[g1]], why="slot guard")]
        return out

    @staticmethod
    def slot_guard(tree: A.SubroutineDefinitionConstruct, loop: LoopInfo) -> Optional[tuple[int, int]]:
        """(if line, end if line) of `if (pi <= ...) then` without else branches in the slot loop."""
        from spel.scripts.registrations import iter_statements

        def find(block):
            for st, _ in iter_statements(block):
                if st.lineno == loop.start_ln and isinstance(st, A.DoLoop):
                    return st
            return None

        do = find(tree.body)
        if do is None:
            return None
        for st in do.body.statements:
            if not isinstance(st, A.IfConstruct) or st.else_ifs or st.else_ is not None:
                continue
            cond = st.condition
            if isinstance(cond, A.InfixExpression) and cond.operator in ("<=", ".le.") \
                    and isinstance(cond.left_expr, A.Identifier) and cond.left_expr.value.lower() == loop.index:
                end = st.construct_end_ln or st.end_ln
                if end > st.lineno:
                    return st.lineno, end
        return None

    def call_edit(self, ctx, csub, ln: int, callee: str, actual: Optional[str], extra=None):
        lines = ctx.lines(csub)
        tree = self.tree(ctx, csub)
        from spel.scripts.registrations import iter_statements

        name = callee.split("::")[-1]
        stmts = [s for s, _ in iter_statements(tree.body)
                 if isinstance(s, A.SubCallStatement) and s.function.span and s.lineno == ln]
        if len(stmts) != 1:
            raise Skip(f"expected one call statement, found {len(stmts)}")
        fn = stmts[0].function
        l1, c1 = fn.span[2], fn.span[3]
        if lines[l1][c1 - 1:c1] != ")":
            raise Skip("no closing parenthesis of the call")
        if actual is None:
            beg, end, stem = BEG, END, None
        else:
            text = self.actual_text(lines, fn, actual)
            stem = text[: len(text) - len("soilc")]
            beg, end = stem + BEG, stem + END
        items = f"{BEG}={beg}, {END}={end}"
        if extra is not None:  # the callee gets a new soil patch filter dummy
            kw, mine = extra
            if stem is not None:
                items += f", {kw}={stem}soilp"
            elif mine is not None:
                items += f", {kw}={mine}"
            else:
                raise Skip(f"no soil patch filter to pass to {name}")
        return self.append_text(lines, l1, c1 - 1, items)

    @staticmethod
    def actual_text(lines: list[str], fn: A.FuncExpression, path: str) -> str:
        """Source text of the actual whose path is `path` (`filter(nc)%soilc` for filter%soilc)."""
        want = path.lower()
        for arg in fn.args:
            e = arg.value if isinstance(arg, A.KeywordArgument) else arg
            if e.span is None:
                continue
            text = span_text(lines, e.span).replace(" ", "")
            bare = "".join(_strip_subscripts(text)).lower()
            if bare == want and text.lower().endswith("soilc"):
                return text
        raise Skip(f"no actual {path} in the call")

    @staticmethod
    def append_text(lines, ln: int, col: int, items: str):
        """`, items` before the `)` at (ln, col), on a continuation line if the line gets long."""
        line = lines[ln]
        if len(line) + len(items) + 2 <= 120:
            return replace_cols(ln, col, col, f", {items}", why="filter-map arguments")
        return replace_cols(ln, col, col, f", &\n{_indent(line)}     {items}", why="filter-map arguments")


def _strip_subscripts(text: str):
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0:
            yield ch
