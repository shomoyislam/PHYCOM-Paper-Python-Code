"""Install verified official CUDA wheels without rebuilding CUDA or replacing torch.

V3.1 targets Kaggle Python 3.12 / torch 2.10 / CUDA 12.8 / T4. Other runtimes
must match an exact entry in wheel_manifest.json. No source build or reference
Mamba is substituted for the official CUDA implementation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parent
PIN = '2.3.2.post1'
CONV_PIN = '1.6.2.post1'


def public_version(value):
    """Keep post releases; remove only the wheel's local CUDA/torch build tag."""
    return value.split('+', 1)[0]


def probe(source, timeout=240):
    return subprocess.run([sys.executable, '-c', source], text=True,
                          capture_output=True, timeout=timeout)


GPU_PROBE = r'''
import torch, json, sys, platform, importlib.metadata as md
packages = {}
for name in ['torch', 'numpy', 'scipy', 'matplotlib', 'triton']:
    try: packages[name] = md.version(name)
    except md.PackageNotFoundError: pass
print(json.dumps(dict(torch=str(torch.__version__), cuda=torch.version.cuda,
    available=torch.cuda.is_available(), gpus=torch.cuda.device_count(),
    devices=[dict(name=torch.cuda.get_device_name(i), cc=list(torch.cuda.get_device_capability(i)))
             for i in range(torch.cuda.device_count())],
    python_tag=f'cp{sys.version_info.major}{sys.version_info.minor}',
    machine=platform.machine(), system=platform.system(),
    cxx11abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    executable=sys.executable, packages=packages)))
'''


MAMBA_PROBE = r'''
import copy, json, torch, importlib.metadata as md
import torch.nn.functional as F
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert md.version('mamba-ssm').split('+')[0] == '2.3.2.post1'
assert md.version('causal-conv1d').split('+')[0] == '1.6.2.post1'
import selective_scan_cuda, causal_conv1d_cuda
from causal_conv1d import causal_conv1d_fn
from mamba_ssm import Mamba
verified = []
for dev in range(torch.cuda.device_count()):
    device = f'cuda:{dev}'
    torch.cuda.set_device(dev)
    x = torch.randn(2,64,20,device=device)
    w = torch.randn(64,4,device=device)
    b = torch.randn(64,device=device)
    got = causal_conv1d_fn(x,w,b,activation='silu')
    ref = F.silu(F.conv1d(x,w[:,None],b,padding=3,groups=64)[:,:,:20])
    torch.testing.assert_close(got,ref,rtol=5e-4,atol=5e-5)
    for length in (3,5,20,256):
        m = Mamba(d_model=32,d_state=16,d_conv=4,expand=2).to(device)
        x = torch.randn(4,length,32,device=device,requires_grad=True)
        y = m(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        y.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        if length == 5:
            other = copy.deepcopy(m)
            other.use_fast_path = False
            xx = x.detach().clone().requires_grad_(True)
            yy = other(xx)
            yy.square().mean().backward()
            torch.testing.assert_close(y,yy,rtol=1e-3,atol=1e-5)
            torch.testing.assert_close(x.grad,xx.grad,rtol=2e-3,atol=1e-6)
    torch.cuda.synchronize(dev)
    verified.append(dict(index=dev,name=torch.cuda.get_device_name(dev)))
print('Official Mamba CUDA forward/backward and convolution checks passed: ' +
      json.dumps(verified))
'''


def choose_wheel(meta, package, version, assets):
    """Select by Python tag, torch major/minor, CUDA major, C++ ABI, and CPU arch."""
    match = re.match(r'^(\d+)\.(\d+)', meta['torch'])
    if not match or not meta.get('cuda') or meta['system'] != 'Linux':
        raise RuntimeError('This installer requires CUDA-enabled Linux PyTorch.')
    torch_tag = '.'.join(match.groups())
    cuda_major = meta['cuda'].split('.')[0]
    abi = 'TRUE' if meta['cxx11abi'] else 'FALSE'
    machine = {'AMD64':'x86_64', 'arm64':'aarch64'}.get(meta['machine'],meta['machine'])
    py = meta['python_tag']
    wanted = (f'{package}-{version}+cu{cuda_major}torch{torch_tag}'
              f'cxx11abi{abi}-{py}-{py}-linux_{machine}.whl')
    found = [asset for asset in assets if asset['name'] == wanted]
    if len(found) != 1:
        raise RuntimeError(f'No verified official wheel for this runtime: {wanted}. '
                           'This revision targets Kaggle Python 3.12, torch 2.10 + CUDA 12.8 on T4. '
                           'No numerical libraries have been changed.')
    return found[0]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''): h.update(block)
    return h.hexdigest()


def download_wheel(asset, cache, attempts=3):
    """Retry a download; accept cached bytes only after SHA-256 verification."""
    cache.mkdir(parents=True,exist_ok=True)
    target = cache / asset['name']
    if target.exists() and sha256(target) == asset['sha256']:
        print('Verified cached wheel:',target.name,flush=True)
        return target
    if not asset['url'].startswith(('https://github.com/Dao-AILab/causal-conv1d/releases/download/',
                                     'https://github.com/state-spaces/mamba/releases/download/')):
        raise ValueError('Unexpected wheel origin in manifest')
    last_error = None
    for attempt in range(1,attempts+1):
        tmp = target.with_suffix('.part')
        try:
            print(f'Downloading {target.name} (attempt {attempt}/{attempts})',flush=True)
            req = urllib.request.Request(asset['url'],headers={'User-Agent':'PHYCOM-Kaggle-3.1'})
            size = 0; last = time.monotonic()
            with urllib.request.urlopen(req,timeout=40) as response, tmp.open('wb') as f:
                while block := response.read(1<<20):
                    f.write(block); size += len(block)
                    if time.monotonic()-last >= 20:
                        print(f'  {size/1e6:.0f}/{asset["bytes"]/1e6:.0f} MB',flush=True)
                        last = time.monotonic()
            if size != asset['bytes'] or sha256(tmp) != asset['sha256']:
                raise RuntimeError('Downloaded wheel failed size/SHA-256 verification')
            os.replace(tmp,target)
            return target
        except Exception as exc:
            last_error = exc
            print(f'Download failed: {type(exc).__name__}: {exc}',flush=True)
        finally:
            if tmp.exists(): tmp.unlink()
    raise RuntimeError(f'Could not download {target.name}. Enable Kaggle Internet; '
                       f'last error: {last_error}')


def error_excerpt(path):
    lines = path.read_text(errors='replace').splitlines()
    keys = ('error:', 'runtimeerror', 'undefined symbol', 'not found', 'no matching distribution')
    selected = set(range(max(0,len(lines)-35),len(lines)))
    for i,line in enumerate(lines):
        if any(key in line.lower() for key in keys):
            selected.update(range(max(0,i-3),min(len(lines),i+8)))
    return '\n'.join(lines[i] for i in sorted(selected))[-12000:]


def stop_process(proc):
    if proc.poll() is not None: return
    try:
        if os.name == 'posix': os.killpg(proc.pid,signal.SIGTERM)
        else: proc.terminate()
    except ProcessLookupError: return
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == 'posix': os.killpg(proc.pid,signal.SIGKILL)
        else: proc.kill()
        proc.wait()


def pip_run(args, logs, label, timeout=1500):
    path = logs / (label+'.log')
    command = [sys.executable,'-m','pip','--disable-pip-version-check',*map(str,args)]
    with path.open('w') as f:
        f.write('Command: '+repr(command)+'\n');f.flush()
        proc = subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,
                                start_new_session=(os.name=='posix'))
        start = last = time.monotonic()
        try:
            while proc.poll() is None:
                now = time.monotonic()
                if now-start > timeout:
                    raise RuntimeError(f'{label} timed out. Full log: {path}')
                if now-last >= 20:
                    lines = path.read_text(errors='replace').splitlines()
                    print(f'[{label} {int(now-start)}s] '+(lines[-1][-200:] if lines else 'working'),flush=True)
                    last = now
                time.sleep(.5)
        finally: stop_process(proc)
    if proc.returncode:
        print(error_excerpt(path),flush=True)
        raise RuntimeError(f'{label} failed (exit {proc.returncode}). Full log: {path}')


def save_probe(logs, name, source):
    try: result = probe(source)
    except subprocess.TimeoutExpired as exc:
        (logs/(name+'.log')).write_text('CUDA/import test timed out.\n'+str(exc))
        raise RuntimeError(f'{name} timed out; see {logs}') from exc
    (logs/(name+'.log')).write_text(result.stdout+'\n'+result.stderr)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--no-mamba',action='store_true')
    args = p.parse_args()
    logs = ROOT/'setup_logs'/time.strftime('%Y%m%d_%H%M%S')
    logs.mkdir(parents=True,exist_ok=True)
    imports = save_probe(logs,'base_imports','import torch,numpy,scipy,matplotlib')
    if imports.returncode:
        raise RuntimeError('Base scientific-library imports failed:\n'+imports.stderr[-4000:])
    if args.no_mamba:
        if not args.smoke: p.error('--no-mamba is allowed only with --smoke')
        print('CPU/non-Mamba smoke setup ready. No packages installed.',flush=True)
        return
    gpu = save_probe(logs,'runtime',GPU_PROBE)
    if gpu.returncode: raise RuntimeError(gpu.stderr[-4000:])
    meta = json.loads(gpu.stdout.strip().splitlines()[-1])
    (logs/'runtime.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('GPU runtime:',json.dumps(meta,indent=2),flush=True)
    if not meta['available'] or not meta['gpus']:
        raise RuntimeError('CUDA GPU unavailable. In Kaggle choose Settings > Accelerator > GPU T4 x2, '
                           'allow the session to restart, then Run All.')
    if any(tuple(device['cc']) < (7,5) for device in meta['devices']):
        raise RuntimeError('This notebook targets T4 (compute capability 7.5) or newer GPUs.')
    manifest = json.loads((ROOT/'wheel_manifest.json').read_text())
    assets = [choose_wheel(meta,'causal_conv1d',CONV_PIN,manifest['assets']),
              choose_wheel(meta,'mamba_ssm',PIN,manifest['assets'])]
    (logs/'selected_wheels.json').write_text(json.dumps(assets,indent=2)+'\n')
    check = save_probe(logs,'initial_cuda_test',MAMBA_PROBE)
    if check.returncode == 0:
        print(check.stdout,flush=True);return
    print('Installing exact official binary wheels. CUDA compilation is not required.',flush=True)
    cache = ROOT/'wheel_cache'
    wheels = [download_wheel(asset,cache) for asset in assets]
    constraints = logs/'runtime_constraints.txt'
    constraints.write_text(''.join(f'{name}=={version}\n' for name,version in meta['packages'].items()))
    # Replace only the two extensions, including any broken prior installation.
    pip_run(['install','--no-deps','--force-reinstall',*wheels],logs,'official_wheels')
    # Resolve declared Python requirements without building any package from source.
    pip_run(['install','--only-binary=:all:','--constraint',constraints,
             f'mamba-ssm=={PIN}',f'causal-conv1d=={CONV_PIN}'],logs,'python_dependencies')
    after = save_probe(logs,'runtime_after',GPU_PROBE)
    if after.returncode: raise RuntimeError(after.stderr[-4000:])
    new = json.loads(after.stdout.strip().splitlines()[-1])
    if any(new['packages'].get(k) != v for k,v in meta['packages'].items()):
        raise RuntimeError('The protected scientific runtime unexpectedly changed; training stopped.')
    check = save_probe(logs,'final_cuda_test',MAMBA_PROBE)
    if check.returncode:
        print(check.stdout+'\n'+check.stderr[-8000:],flush=True)
        raise RuntimeError('Official CUDA verification failed. Training has not started. '
                           f'The complete error is saved in {logs}/final_cuda_test.log')
    print(check.stdout,flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        from diagnostics import create_diagnostics
        path = ROOT/'setup_logs'/'failure_traceback.txt'
        path.parent.mkdir(exist_ok=True)
        path.write_text(traceback.format_exc())
        archive = create_diagnostics(ROOT,ROOT.parent)
        print(f'\nSETUP STOPPED: {exc}\nDiagnostic ZIP: {archive}',flush=True)
        sys.exit(1)
