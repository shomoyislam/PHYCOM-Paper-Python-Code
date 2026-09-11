"""Exercise an actual estimator optimizer checkpoint on each selected CUDA GPU."""
import io
import torch
from .models import Estimator
from .training import seed_torch


def optimizer_resume_probe(device, name='lstm_gate'):
    seed_torch(781)
    model=Estimator(name,4,123).to(device).train()
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5)
    y=torch.randn(8,5,2,4,4,device=device); s=torch.zeros(8,1,device=device)
    target=torch.randn(8,2,4,4,device=device)
    def step(m,o):
        o.zero_grad(set_to_none=True); loss=(m(y,s)-target).square().mean(); loss.backward(); o.step()
        if not torch.isfinite(loss): raise RuntimeError('Optimizer preflight loss is nonfinite')
        return loss.detach()
    first=step(model,opt); scheduler.step(float(first))
    buffer=io.BytesIO(); torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict()),buffer)
    buffer.seek(0); ck=torch.load(buffer,map_location='cpu',weights_only=True)
    reference=step(model,opt)
    resumed=Estimator(name,4,456).to(device).train(); resumed.load_state_dict(ck['model'])
    ropt=torch.optim.Adam(resumed.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(ropt,patience=5)
    sched.load_state_dict(ck['scheduler']); ropt.load_state_dict(ck['optimizer'])
    for state in ropt.state.values():
        if state['step'].device.type!='cpu': raise RuntimeError('Non-capturable Adam step was moved off CPU')
    got=step(resumed,ropt)
    torch.testing.assert_close(got,reference,rtol=1e-3,atol=1e-5)
    for p,q in zip(model.parameters(),resumed.parameters()):
        torch.testing.assert_close(p,q,rtol=2e-3,atol=2e-5)
    return dict(model=name,device=str(device),passed=True)


def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA preflight requires an accelerator')
    for i in range(min(2,torch.cuda.device_count())):
        torch.cuda.set_device(i)
        for name in ['lstm_gate','lstm_wide_gate','mamba_gate']:
            print(optimizer_resume_probe(torch.device(f'cuda:{i}'),name),flush=True)
    print('Estimator training and Adam checkpoint resume checks passed.',flush=True)


if __name__=='__main__': main()
