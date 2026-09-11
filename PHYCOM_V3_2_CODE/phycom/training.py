"""Validation-only selection, atomic epoch transactions, and restartable training."""
import gc
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import atomic_json, stable_seed, check_deadline


def seed_torch(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_torch(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(obj, tmp); os.replace(tmp, path)


def load_checkpoint(path):
    # Tensor/primitive state dictionaries only; no pickle class allow-list or fallback.
    return torch.load(path, map_location='cpu', weights_only=True)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}


def publish_checkpoints(folder, ck):
    common = {k: ck[k] for k in ['run_id','model_name','job','origin']}
    save_torch(folder/'best.pt', dict(common, state_dict=ck['best_state_dict'],
        best_val=ck['best_val'], epoch=ck['best_epoch']))
    if ck['budget_state_dict'] is not None:
        save_torch(folder/'budget60.pt', dict(common, state_dict=ck['budget_state_dict'],
            best_val=ck['budget_val'], epoch=ck['budget_epoch']))


def import_legacy_state(last, best, run_id, job, model_name, origin):
    """Explicit migration to a new experiment identity; no in-place V3.1 edits."""
    for ck in (last, best):
        if ck['job'] != job or ck['model_name'] != model_name:
            raise ValueError('Parent checkpoint job/model mismatch')
    if last['epoch']+1 > 60 or best['epoch'] > last['epoch']+1:
        raise ValueError('Invalid 60-epoch parent checkpoint')
    if len(last['history']) != last['epoch']+1 or best['best_val'] != last['best_val']:
        raise ValueError('Parent history and selected checkpoint disagree')
    return dict(last, run_id=run_id, origin=origin,
        best_state_dict=best['state_dict'], best_epoch=best['epoch'],
        budget_state_dict=best['state_dict'], budget_val=best['best_val'], budget_epoch=best['epoch'])


def train(model, train_arrays, val_arrays, cfg, job, folder, device, run_id):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    last_path = folder/'last.pt'
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
        patience=cfg['lr_patience'], factor=cfg['lr_factor'])
    start = wait = 0; best = float('inf'); elapsed = 0.; history = []
    best_state = budget_state = None; best_epoch = budget_epoch = 0; budget_val = None
    origin = dict(kind='fresh', initialization_seed=stable_seed('model',job['seed']))
    seed_torch(stable_seed('optimizer', job['seed']))
    if last_path.exists():
        ck = load_checkpoint(last_path)
        if ck['run_id'] != run_id or ck['model_name'] != model.name or ck['job'] != job:
            raise RuntimeError(f'Checkpoint/config mismatch at {last_path}')
        model.load_state_dict(ck['state_dict'])
        scheduler.load_state_dict(ck['scheduler']); opt.load_state_dict(ck['optimizer'])
        # Optimizer.load_state_dict places moments on each parameter's device.
        # For non-capturable/non-fused Adam the step counter remains on CPU.
        # Do not blanket-move every optimizer tensor to CUDA.
        start, wait, best = ck['epoch']+1, ck['wait'], ck['best_val']
        elapsed, history, origin = ck['elapsed'], ck['history'], ck['origin']
        best_state, best_epoch = ck['best_state_dict'], ck['best_epoch']
        budget_state, budget_val, budget_epoch = ck['budget_state_dict'], ck['budget_val'], ck['budget_epoch']
        torch.set_rng_state(ck['torch_rng'])
        if device.type=='cuda' and ck['cuda_rng'] is not None:
            torch.cuda.set_rng_state(ck['cuda_rng'], device)
        publish_checkpoints(folder, ck)
        print(f'Resume {model.name}: {start} epochs committed; best validation {best:.6g}', flush=True)
    tr = va = None
    try:
        tr = tuple(torch.from_numpy(a).to(device) for a in train_arrays)
        va = tuple(torch.from_numpy(a).to(device) for a in val_arrays)
        resident = True
    except torch.OutOfMemoryError:
        tr = va = None; gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
        tr = tuple(torch.from_numpy(a) for a in train_arrays)
        va = tuple(torch.from_numpy(a) for a in val_arrays)
        resident = False
    bs = cfg['batch_size']; criterion = nn.MSELoss()
    try:
        for epoch in range(start, cfg['epochs']):
            if wait >= cfg['patience']: break
            check_deadline()
            t0 = time.perf_counter(); model.train()
            g = torch.Generator().manual_seed(stable_seed('shuffle',job['seed'],epoch))
            order = torch.randperm(len(tr[0]), generator=g).to(tr[0].device)
            total = torch.zeros((), device=device)
            for lo in range(0, len(order), bs):
                ix = order[lo:lo+bs]; y,s,h = (a[ix].to(device) for a in tr)
                opt.zero_grad(set_to_none=True); loss = criterion(model(y,s),h)
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss: {model.name}')
                loss.backward()
                if cfg['gradient_clip'] is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
                opt.step(); total += loss.detach()*len(ix)
            model.eval(); vtotal = torch.zeros((),device=device)
            with torch.inference_mode():
                for lo in range(0,len(va[0]),bs):
                    y,s,h = (a[lo:lo+bs].to(device) for a in va)
                    vtotal += criterion(model(y,s),h)*len(y)
            tl = float((total/len(tr[0])).cpu()); vl = float((vtotal/len(va[0])).cpu())
            if not np.isfinite(vl): raise FloatingPointError('Nonfinite validation loss')
            scheduler.step(vl); elapsed += time.perf_counter()-t0
            history.append(dict(epoch=epoch+1, train_mse=tl, val_mse=vl, lr=opt.param_groups[0]['lr']))
            state = cpu_state(model)
            if vl < best: best, wait, best_state, best_epoch = vl, 0, state, epoch+1
            else: wait += 1
            if epoch+1 <= cfg['reference_epochs']:
                budget_state, budget_val, budget_epoch = best_state, best, best_epoch
            ck = dict(run_id=run_id, model_name=model.name, job=job, state_dict=state,
                optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch, wait=wait,
                best_val=best, best_state_dict=best_state, best_epoch=best_epoch,
                budget_state_dict=budget_state, budget_val=budget_val, budget_epoch=budget_epoch,
                elapsed=elapsed, history=history, origin=origin, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
            # The last.pt transaction includes both selected states. A crash between
            # writing it and its convenient exports is repaired at the next resume.
            save_torch(last_path, ck)
            publish_checkpoints(folder, ck)
            atomic_json(folder/'training.json', dict(history=history, training_seconds=elapsed,
                best_val=best, best_epoch=best_epoch, gpu_resident_data=resident, origin=origin))
            if epoch==0 or (epoch+1)%5==0 or wait>=cfg['patience']:
                print(f'{model.name}: {epoch+1}/{cfg["epochs"]} val={vl:.6g} best={best:.6g}', flush=True)
        if best_state is None: raise RuntimeError('No valid checkpoint produced')
        publish_checkpoints(folder, load_checkpoint(last_path))
        stop = 'validation_patience' if wait>=cfg['patience'] else 'epoch_cap'
        atomic_json(folder/'training.json', dict(history=history, training_seconds=elapsed,
            best_val=best, best_epoch=best_epoch, reference_best_val=budget_val,
            reference_best_epoch=budget_epoch, stop_reason=stop, epoch_cap=cfg['epochs'],
            gpu_resident_data=resident, origin=origin))
        model.load_state_dict(best_state); model.eval()
        return model
    finally:
        del tr, va
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()


def predict(model, y, snr, device, batch=2048):
    from .simulation import real_input
    snr = np.broadcast_to(np.asarray(snr, dtype=np.float32), (len(y),))
    estimates = []; model.eval()
    with torch.inference_mode():
        for lo in range(0,len(y),batch):
            x = torch.from_numpy(real_input(y[lo:lo+batch])).to(device)
            s = torch.from_numpy(((snr[lo:lo+batch]-15)/15).reshape(-1,1).copy()).to(device)
            p = model(x,s).cpu().numpy(); estimates.append(p[:,0]+1j*p[:,1])
    return np.concatenate(estimates)
