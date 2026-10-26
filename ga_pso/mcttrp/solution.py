"""Route representation, cost and feasibility check shared by all algorithms."""
import math
from dataclasses import dataclass, field

from .instance import Instance

EPS = 1e-9


@dataclass
class Route:
    kind: str                                       # "PTR", "PVR" or "MVR"
    main: list
    subtours: list = field(default_factory=list)   # [(root, [customers])]


def _tour_cost(dist, start, nodes):
    cost, prev = 0.0, start
    for v in nodes:
        cost += dist[prev, v]
        prev = v
    return cost + dist[prev, start]


def route_cost(inst: Instance, r: Route) -> float:
    c = _tour_cost(inst.dist, 0, r.main)
    for root, nodes in r.subtours:
        c += _tour_cost(inst.dist, root, nodes)
    return c


def solution_cost(inst: Instance, routes) -> float:
    return sum(route_cost(inst, r) for r in routes)


def _items(inst, customers):
    return [d for i in customers for d in inst.demand[i] if d > EPS]


def truck_hoppers_needed(inst, items):
    return sum(math.ceil(d / inst.h_truck - EPS) for d in items)


def mixed_fits(inst, items, free_truck):
    """Check if items fit in free truck hoppers plus all trailer hoppers (exact DP)."""
    if free_truck < 0:
        return False
    H = inst.nh_trailer
    INF = 10 ** 9
    best = [0] + [INF] * H
    for d in items:
        nxt = [INF] * (H + 1)
        max_x = min(H, math.ceil(d / inst.h_trailer - EPS))
        for used, t in enumerate(best):
            if t >= INF:
                continue
            for x in range(0, max_x + 1):
                if used + x > H:
                    break
                rest = max(0.0, d - x * inst.h_trailer)
                y = math.ceil(rest / inst.h_truck - EPS)
                nxt[used + x] = min(nxt[used + x], t + y)
        best = nxt
    return min(best) <= free_truck


def validate(inst: Instance, routes):
    """Return (ok, message) for a full solution."""
    seen = []
    n_complete = n_ptr = 0
    for k, r in enumerate(routes):
        sub_nodes = [v for _, nodes in r.subtours for v in nodes]
        if r.kind == "PTR":
            if r.subtours:
                return False, f"route {k}: PTR cannot have sub-tours"
            n_ptr += 1
            if truck_hoppers_needed(inst, _items(inst, r.main)) > inst.nh_truck:
                return False, f"route {k}: truck hopper capacity exceeded"
        elif r.kind in ("PVR", "MVR"):
            if r.kind == "PVR" and r.subtours:
                return False, f"route {k}: PVR cannot have sub-tours"
            n_complete += 1
            if any(inst.is_tc[v] for v in r.main):
                return False, f"route {k}: TC on a main tour"
            for root, nodes in r.subtours:
                if root not in r.main:      # no parking at the depot (MILP constraint 10)
                    return False, f"route {k}: sub-tour root {root} is not a VC on the main tour"
                if not nodes:
                    return False, f"route {k}: empty sub-tour"
            used_truck = truck_hoppers_needed(inst, _items(inst, sub_nodes))
            if not mixed_fits(inst, _items(inst, r.main), inst.nh_truck - used_truck):
                return False, f"route {k}: hopper capacity exceeded"
        else:
            return False, f"route {k}: unknown kind {r.kind}"
        seen += r.main + sub_nodes
    if sorted(seen) != list(range(1, inst.n + 1)):
        return False, "each customer must be visited exactly once"
    # fleet rule used by the paper's heuristic and Table 4 (see README)
    if n_complete > inst.n_trailers or n_complete + n_ptr > inst.n_trucks:
        return False, (f"fleet exceeded: {n_complete} complete vehicles, "
                       f"{n_ptr} pure trucks")
    return True, "ok"
