"""Load MC-TTRP instances with the same node order as the R code (depot, VCs, TCs)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Instance:
    name: str
    dist: np.ndarray
    demand: np.ndarray
    is_tc: np.ndarray
    h_truck: float
    nh_truck: int
    h_trailer: float
    nh_trailer: int
    n_trucks: int
    n_trailers: int

    @property
    def n(self) -> int:
        return self.dist.shape[0] - 1

    @property
    def n_vc(self) -> int:
        return int((~self.is_tc[1:]).sum())


def _vc_first_order(types):
    vcs = [i for i in range(1, len(types)) if types[i] == 0]
    tcs = [i for i in range(1, len(types)) if types[i] == 1]
    return [0] + vcs + tcs


def _load_csv(path):
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
    head = [float(x) for x in lines[1].split(",")[:8]]
    cap_t, cap_l, n, _, _, nf, h_t, h_l = head
    n, nf = int(n), int(nf)
    rows = [[float(x) for x in l.split(",")] for l in lines[3:3 + n + 1]]
    rows = np.array(rows)
    dist = rows[:, 1:n + 2]
    types = rows[:, n + 2].astype(int)
    demand = rows[:, n + 3:n + 3 + nf]
    order = _vc_first_order(types)
    # R reorders CSV demands but not distances; safe only if the file is already VC-first
    assert order == list(range(n + 1)), "CSV instance is not VC-first"
    return dist, demand, types, cap_t, cap_l, h_t, h_l


def _load_txt(path):
    data = [l.split() for l in open(path, encoding="utf-8")
            if l.strip() and not l.lstrip().startswith("#")]
    cap_t, cap_l, n = float(data[0][0]), float(data[0][1]), int(data[0][2])
    nf, h_t, h_l = int(data[1][0]), float(data[1][1]), float(data[1][2])
    xy = [(float(data[2][1]), float(data[2][2]))]
    types, demand = [0], [[0.0] * nf]
    for row in data[3:3 + n]:
        xy.append((float(row[1]), float(row[2])))
        types.append(int(row[3]))
        demand.append([float(v) for v in row[4:4 + nf]])
    order = _vc_first_order(types)
    xy = np.array(xy)[order]
    types = np.array(types)[order]
    demand = np.array(demand)[order]
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return dist, demand, types, cap_t, cap_l, h_t, h_l


def load_instance(path, n_trucks, n_trailers) -> Instance:
    path = Path(path)
    loader = _load_csv if path.suffix == ".csv" else _load_txt
    dist, demand, types, cap_t, cap_l, h_t, h_l = loader(path)
    return Instance(
        name=path.stem,
        dist=np.ascontiguousarray(dist, dtype=np.float64),
        demand=np.ascontiguousarray(demand, dtype=np.float64),
        is_tc=np.asarray(types) == 1,
        h_truck=h_t, nh_truck=int(round(cap_t / h_t)),
        h_trailer=h_l, nh_trailer=int(round(cap_l / h_l)),
        n_trucks=n_trucks, n_trailers=n_trailers,
    )
