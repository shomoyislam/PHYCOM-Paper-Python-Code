"""V3.2-compatible LSTMs with explicitly separated capacity and initialization controls."""
from contextlib import contextmanager
import torch
from torch import nn
from .protocol import spec_id


@contextmanager
def init_seed(s):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(s)
        yield


class Estimator(nn.Module):
    def __init__(self,spec,n,seed):
        super().__init__()
        self.spec,self.n,self.name=spec,n,spec_id(spec)
        p,h,q=spec['projection'],spec['hidden'],spec['head']
        self.condition=spec['condition']
        with init_seed(seed+101):
            if spec['encoder']=='cnn':
                self.encoder=nn.Sequential(nn.Conv2d(2,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),
                    nn.Conv2d(16,8,3,padding=1),nn.BatchNorm2d(8),nn.ReLU())
                self.proj=nn.Linear(8*n*n,p)
        with init_seed(seed+201):
            self.core=nn.LSTM(2*n*n if spec['encoder']=='raw' else p,h,batch_first=True)
        c=self.condition
        with init_seed(seed+301):
            if c in ('gate','gate_identity','resgate','resgate_identity'):
                self.gate=nn.Linear(1,h)
                if c.endswith('_identity'):
                    nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
            if c in ('film','film_identity'):
                self.scale=nn.Linear(1,h);self.shift=nn.Linear(1,h)
                if c=='film_identity':
                    for l in (self.scale,self.shift):
                        nn.init.zeros_(l.weight);nn.init.zeros_(l.bias)
        concat=c.startswith('concat')
        with init_seed(seed+401):
            self.fc1=nn.Linear(h+int(concat),q)
        if concat:
            with init_seed(seed+401):
                common=nn.Linear(h,q)
            with torch.no_grad():
                self.fc1.weight[:,:h].copy_(common.weight);self.fc1.bias.copy_(common.bias)
                if c=='concat_identity': self.fc1.weight[:,h:].zero_()
        with init_seed(seed+402):
            self.fc2=nn.Linear(q,2*n*n)
        if spec['residual']:
            nn.init.zeros_(self.fc2.weight);nn.init.zeros_(self.fc2.bias)

    def forward(self,y,s,return_features=False):
        B,T=y.shape[:2]
        raw=y.flatten(2)
        if self.spec['encoder']=='cnn':
            conv=self.encoder(y.reshape(B*T,2,self.n,self.n)).reshape(B,T,-1)
            projected=self.proj(conv)
        else:
            conv=raw;projected=raw
        _,(h,_)=self.core(projected)
        z=h[-1];c=self.condition
        if c=='gate': zc=z*torch.sigmoid(self.gate(s))
        elif c=='gate_identity': zc=z*(2*torch.sigmoid(self.gate(s)))
        elif c=='resgate': zc=z*(1+torch.sigmoid(self.gate(s)))
        elif c=='resgate_identity': zc=z*(1+torch.tanh(self.gate(s)))
        elif c=='film': zc=self.scale(s)*z+self.shift(s)
        elif c=='film_identity': zc=(1+self.scale(s))*z+self.shift(s)
        elif c.startswith('concat'): zc=torch.cat([z,s],dim=-1)
        else: zc=z
        head=torch.relu(self.fc1(zc))
        correction=self.fc2(head).reshape(B,2,self.n,self.n)
        out=y[:,-1]+correction if self.spec['residual'] else correction
        if return_features:
            return out,{'conv_last':conv[:,-1], 'projection_last':projected[:,-1],
                'temporal_last':z,'conditioned':zc,'head_relu':head,
                'prediction':out.flatten(1),'correction':correction.flatten(1)}
        return out


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
