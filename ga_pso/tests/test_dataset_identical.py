import hashlib
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def test_data_matches_checksums_of_original_instances():
    # SHA256SUMS was computed from the instances folder of the original ITS repo
    expected = {}
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        expected[name] = digest
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in DATA.iterdir() if p.name != "SHA256SUMS"}
    assert actual == expected
