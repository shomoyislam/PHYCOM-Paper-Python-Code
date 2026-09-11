"""Versioned experiment configuration and deterministic identities."""
import hashlib
import json
import os
import time
from pathlib import Path


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def code_hash():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    files = [root/n for n in ['bootstrap.py', 'run_phycom.py', 'diagnostics.py', 'wheel_manifest.json']]
    files += sorted((root/'phycom').glob('*.py'))
    for p in files:
        h.update(p.relative_to(root).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


class SessionPaused(RuntimeError):
    """An epoch/evaluation boundary was reached after the session deadline."""


def check_deadline():
    limit = float(os.environ.get('PHYCOM_STOP_AT', '0'))
    if limit and time.time() >= limit:
        raise SessionPaused('Session budget reached; committed work can resume.')


def base_config(smoke=False):
    cfg = dict(schema='phycom-v3.2', mode='smoke' if smoke else 'publication',
        arrays=[2,4], seeds=[42,123,2024], main_T=5, alpha=.8, M=16,
        train_samples=100000, val_samples=20000, test_samples=20000,
        train_snr=[0,30], snrs=[0,5,10,15,20,25,30], batch_size=256,
        epochs=240, reference_epochs=60, patience=10, lr=.001,
        lr_patience=5, lr_factor=.5, loss='linear_mse', gradient_clip=None,
        seq_lengths=[3,5,10,15,20], seq_seeds=[42,123,2024],
        seq_train=40000, seq_val=8000, seq_models=['lstm_gate','mamba_gate','lsres_gate'],
        temporal_grid=[.3,.5,.65,.8,.9,.95], spatial_grid=[0,.3,.5,.7,.9],
        snr_bias_grid=[-3.,-1.,0.,1.,3.], snr_error_std_grid=[0.,1.,2.,3.],
        stress_snr=15., stress_samples=10000,
        ber_blocks=2048, bits_per_block=1000, bootstrap_resamples=2000, rate_blocks=4096,
        latency_T=[5,8,16,32,64,128,256], latency_batch=32, latency_trials=100,
        models=['cnn_gate','raw_lstm_gate','lstm_none','lstm_concat','lstm_film',
                'lstm_gate','lstm_resgate','gru_none','gru_gate','mamba_none','mamba_gate',
                'lsres_none','lsres_gate'],
        wide_models=['lstm_wide_none','lstm_wide_gate'], wide_arrays=[4],
        stress_models=['lstm_none','lstm_gate','mamba_none','mamba_gate',
                       'lsres_none','lsres_gate','lstm_wide_none','lstm_wide_gate'],
        test_namespace='phycom-v3.2-confirmation-20260909',
        convergence_window=20, convergence_relative_gain=.01,
        mamba_version='2.3.2.post1', causal_conv_version='1.6.2.post1',
        include_mamba=True, torch_threads=2)
    if smoke:
        cfg.update(seeds=[42], seq_seeds=[42], train_samples=96, val_samples=48,
            test_samples=96, epochs=3, reference_epochs=1, patience=3, batch_size=32,
            seq_lengths=[3,5], seq_train=96, seq_val=48, stress_samples=64,
            ber_blocks=32, bits_per_block=128, rate_blocks=48, bootstrap_resamples=100,
            latency_T=[5,8], latency_trials=3, temporal_grid=[.3,.8], spatial_grid=[0,.7],
            snr_bias_grid=[-3.,0.,3.], snr_error_std_grid=[0.,2.], snrs=[0,15,30],
            convergence_window=2)
    return cfg


def jobs(cfg):
    out = [dict(kind='main', n=n, T=cfg['main_T'], seed=s) for n in cfg['arrays'] for s in cfg['seeds']]
    out += [dict(kind='seq', n=2, T=t, seed=s) for t in cfg['seq_lengths'] for s in cfg['seq_seeds']]
    return out


def model_names(cfg, job):
    if job['kind'] == 'seq': return list(cfg['seq_models'])
    return list(cfg['models']) + (list(cfg['wide_models']) if job['n'] in cfg['wide_arrays'] else [])


def job_id(job):
    return '{kind}_{n}x{n}_T{T}_seed{seed}'.format(**job)


def stages(job):
    return ['budget60','extended'] if job['kind']=='main' else ['extended']
