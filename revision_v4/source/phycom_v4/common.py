import hashlib
import json
import os
import time
from pathlib import Path


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def source_hash():
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'run_revision.py']
    paths += sorted((root / 'phycom_v4').glob('*.py'))
    paths += sorted((root / 'legacy_v32').glob('*.py'))
    return digest({p.relative_to(root).as_posix(): file_hash(p) for p in paths})


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


class PauseRun(Exception):
    pass


def check_deadline():
    t = float(os.environ.get('PHYCOM_V4_STOP_AT', '0'))
    if t and time.time() >= t:
        raise PauseRun('Session budget reached; restart with the same configuration.')


def group_id(group):
    return f"{group['family']}_{group['n']}x{group['n']}_seed{group['seed']}"
