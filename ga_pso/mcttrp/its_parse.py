"""Read ITS results written by run_its.R."""
import json

from .solution import Route


def parse_r_route(kind, seq):
    """Split an R route, e.g. 0 4 7 6 4 2 1 3 0, into main tour and sub-tours."""
    seq = [int(v) for v in seq]
    if kind == "PTR":
        return Route("PTR", main=[v for v in seq if v != 0])
    inner = seq[1:-1]
    if kind == "PVR" or len(set(inner)) == len(inner):
        return Route("PVR" if kind == "PVR" else "MVR", main=[v for v in inner if v != 0])
    main, subtours = [], []
    root, sub = None, []
    for k, v in enumerate(inner):
        if root is not None:
            if v == root:
                subtours.append((root, sub))
                root = root if root in inner[k + 1:] else None   # another sub-tour from the same node
                sub = []
            else:
                sub.append(v)
            continue
        if v == 0:
            root, sub = 0, []
            continue
        main.append(v)
        if inner.count(v) >= 2:
            root, sub = v, []
    if root is not None:
        raise ValueError(f"unclosed sub-tour in {seq}")
    return Route("MVR", main=main, subtours=subtours)


def load_its_result(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    d["routes"] = [parse_r_route(r["type"], r["route"]) for r in d["routes"]]
    return d
