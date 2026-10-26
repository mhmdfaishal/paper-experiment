"""Split a customer permutation into PTR/PVR/MVR routes (Prins-style split DP).

MVRs get one sub-tour with the TCs, parked at the cheapest VC of the route.
"""
import math

import numpy as np
from numba import njit

from .solution import Route

EXTRA = 3
KIND_PTR, KIND_PVR, KIND_MVR = 0, 1, 2
EPS = 1e-9


@njit(cache=True)
def _ceil(x):
    return math.ceil(x - EPS)


@njit(cache=True)
def _split(perm, dist, demand, is_tc, h_t, nh_t, h_l, nh_l, max_a, max_b, lim_a, lim_b, pen):
    n = perm.shape[0]
    nf = demand.shape[1]
    INF = 1e18
    # F[pos, a, b]: min cost to serve perm[:pos] with a complete vehicles and b PTRs
    F = np.full((n + 1, max_a + 1, max_b + 1), INF)
    pred_i = np.full((n + 1, max_a + 1, max_b + 1), -1, np.int64)
    pred_k = np.zeros((n + 1, max_a + 1, max_b + 1), np.int64)
    pred_r = np.zeros((n + 1, max_a + 1, max_b + 1), np.int64)
    F[0, 0, 0] = 0.0
    dp_l = np.empty(nh_l + 1, np.int64)
    nxt = np.empty(nh_l + 1, np.int64)
    BIG = 1 << 40

    for i in range(n):
        if F[i].min() >= INF:
            continue
        dp_l[:] = BIG          # min truck hoppers for VC items, by trailer hoppers used
        dp_l[0] = 0
        all_h = 0
        tc_h = 0
        all_path = 0.0; all_first = -1; all_last = -1
        vc_path = 0.0; vc_first = -1; vc_last = -1
        tc_path = 0.0; tc_first = -1; tc_last = -1
        for j in range(i, n):
            v = perm[j]
            for f in range(nf):
                d = demand[v, f]
                if d <= EPS:
                    continue
                all_h += _ceil(d / h_t)
                if is_tc[v]:
                    tc_h += _ceil(d / h_t)
                else:
                    nxt[:] = BIG
                    max_x = min(nh_l, _ceil(d / h_l))
                    for used in range(nh_l + 1):
                        t = dp_l[used]
                        if t >= BIG:
                            continue
                        for x in range(max_x + 1):
                            if used + x > nh_l:
                                break
                            rest = d - x * h_l
                            y = _ceil(rest / h_t) if rest > EPS else 0
                            if t + y < nxt[used + x]:
                                nxt[used + x] = t + y
                    dp_l[:] = nxt
            if all_first < 0:
                all_first = v
            else:
                all_path += dist[all_last, v]
            all_last = v
            if is_tc[v]:
                if tc_first < 0:
                    tc_first = v
                else:
                    tc_path += dist[tc_last, v]
                tc_last = v
            else:
                if vc_first < 0:
                    vc_first = v
                else:
                    vc_path += dist[vc_last, v]
                vc_last = v

            ptr_ok = all_h <= nh_t
            ptr_cost = dist[0, all_first] + all_path + dist[all_last, 0]
            comp_ok = False
            comp_cost = INF
            comp_kind = KIND_PVR
            root = 0
            if vc_first >= 0:
                main_cost = dist[0, vc_first] + vc_path + dist[vc_last, 0]
                if tc_first < 0:
                    comp_ok = dp_l.min() <= nh_t
                    comp_cost = main_cost
                elif tc_h <= nh_t and dp_l.min() <= nh_t - tc_h:
                    comp_ok = True
                    comp_kind = KIND_MVR
                    best = INF
                    for q in range(i, j + 1):
                        u = perm[q]
                        if not is_tc[u]:
                            c = dist[u, tc_first] + dist[tc_last, u]
                            if c < best:
                                best = c
                                root = u
                    comp_cost = main_cost + best + tc_path
            if not ptr_ok and not comp_ok:
                break
            for a in range(max_a + 1):
                for b in range(max_b + 1):
                    base = F[i, a, b]
                    if base >= INF:
                        continue
                    if ptr_ok and b + 1 <= max_b:
                        c = base + ptr_cost
                        if c < F[j + 1, a, b + 1]:
                            F[j + 1, a, b + 1] = c
                            pred_i[j + 1, a, b + 1] = i
                            pred_k[j + 1, a, b + 1] = KIND_PTR
                            pred_r[j + 1, a, b + 1] = 0
                    if comp_ok and a + 1 <= max_a:
                        c = base + comp_cost
                        if c < F[j + 1, a + 1, b]:
                            F[j + 1, a + 1, b] = c
                            pred_i[j + 1, a + 1, b] = i
                            pred_k[j + 1, a + 1, b] = comp_kind
                            pred_r[j + 1, a + 1, b] = root

    best_cost = INF
    best_a = -1
    best_b = -1
    for a in range(max_a + 1):
        for b in range(max_b + 1):
            if F[n, a, b] >= INF:
                continue
            excess = max(0, a - lim_a) + max(0, a + b - lim_b)
            c = F[n, a, b] + pen * excess
            if c < best_cost:
                best_cost = c
                best_a = a
                best_b = b
    segs = np.zeros((n, 4), np.int64)
    ns = 0
    if best_a >= 0:
        pos, a, b = n, best_a, best_b
        while pos > 0:
            i = pred_i[pos, a, b]
            k = pred_k[pos, a, b]
            segs[ns, 0] = i
            segs[ns, 1] = pos
            segs[ns, 2] = k
            segs[ns, 3] = pred_r[pos, a, b]
            ns += 1
            if k == KIND_PTR:
                b -= 1
            else:
                a -= 1
            pos = i
    feasible = best_a >= 0 and best_a <= lim_a and best_a + best_b <= lim_b
    return best_cost, feasible, segs[:ns][::-1]


class Decoder:
    """Evaluates permutations of customers 1..n for one instance."""

    def __init__(self, inst):
        self.inst = inst
        self.lim_a = inst.n_trailers
        self.lim_b = inst.n_trucks
        # extra vehicles are allowed but cost more than any feasible solution
        self.pen = 2.0 * inst.n * float(inst.dist.max()) + 1.0
        self.n_evals = 0
        self._args = (inst.dist, inst.demand, inst.is_tc, float(inst.h_truck),
                      int(inst.nh_truck), float(inst.h_trailer), int(inst.nh_trailer),
                      self.lim_a + EXTRA, self.lim_b + EXTRA, self.lim_a, self.lim_b,
                      self.pen)

    def _run(self, perm):
        self.n_evals += 1
        return _split(np.asarray(perm, dtype=np.int64), *self._args)

    def evaluate(self, perm):
        """Return (cost, feasible) of a permutation."""
        cost, feasible, _ = self._run(perm)
        return cost, bool(feasible)

    def routes(self, perm):
        """Return the routes of a permutation."""
        perm = np.asarray(perm, dtype=np.int64)
        _, _, segs = _split(perm, *self._args)
        out = []
        for s, e, k, root in segs:
            nodes = [int(v) for v in perm[s:e]]
            if k == KIND_PTR:
                out.append(Route("PTR", main=nodes))
            elif k == KIND_PVR:
                out.append(Route("PVR", main=nodes))
            else:
                vcs = [v for v in nodes if not self.inst.is_tc[v]]
                tcs = [v for v in nodes if self.inst.is_tc[v]]
                out.append(Route("MVR", main=vcs, subtours=[(int(root), tcs)]))
        return out
