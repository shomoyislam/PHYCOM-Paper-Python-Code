"""Original CNN/LSTM dimensions retained; LS-residual variants explicitly named.

The component-specific initialization seeds keep shared backbone/head weights
identical across conditioning variants where the tensor shapes match.
"""
import math
from contextlib import contextmanager
import torch
from torch import nn

SPECS = {
    'lstm_wide_none': ('lstm', 'none', False),
    'lstm_wide_gate': ('lstm', 'gate', False),
    '_transformer_gate': ('transformer', 'gate', False),
    'cnn_gate': ('cnn', 'gate', False),
    'raw_lstm_gate': ('raw_lstm', 'gate', False),
    'lstm_none': ('lstm', 'none', False),
    'lstm_concat': ('lstm', 'concat', False),
    'lstm_film': ('lstm', 'film', False),
    'lstm_gate': ('lstm', 'gate', False),
    'lstm_resgate': ('lstm', 'resgate', False),
    'gru_none': ('gru', 'none', False),
    'gru_gate': ('gru', 'gate', False),
    'mamba_none': ('mamba', 'none', False),
    'mamba_gate': ('mamba', 'gate', False),
    'lsres_none': ('lstm', 'none', True),
    'lsres_gate': ('lstm', 'gate', True),
}


@contextmanager
def init_seed(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


class Estimator(nn.Module):
    def __init__(self, name, n=2, seed=42):
        super().__init__()
        self.name, self.n = name, n
        self.core_kind, self.condition, self.ls_residual = SPECS[name]
        k = self.core_kind
        with init_seed(seed+101):
            if k != 'raw_lstm':
                self.encoder = nn.Sequential(nn.Conv2d(2,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                                             nn.Conv2d(16,8,3,padding=1), nn.BatchNorm2d(8), nn.ReLU())
            if k not in ('cnn','raw_lstm'):
                self.proj = nn.Linear(8*n*n,32)
        with init_seed(seed+201):
            if k in ('lstm','raw_lstm'):
                self.core = nn.LSTM(2*n*n if k=='raw_lstm' else 32,32,batch_first=True)
            elif k == 'gru':
                self.core = nn.GRU(32,32,batch_first=True)
            elif k == 'mamba':
                # Publication runs never substitute a different Mamba implementation.
                from mamba_ssm import Mamba
                self.core = Mamba(d_model=32,d_state=16,d_conv=4,expand=2)
            elif k == 'transformer':
                self.core = nn.TransformerEncoderLayer(32,4,64,dropout=0.,batch_first=True)
        width = 8 if k=='cnn' else 32
        with init_seed(seed+301):
            if self.condition in ('gate','resgate'):
                self.gate = nn.Linear(1,width)
            elif self.condition=='film':
                self.scale = nn.Linear(1,width)
                self.shift = nn.Linear(1,width)
        hidden = 64 if name.startswith('lstm_wide_') else max(16,2*n*n)
        base_input = 8*n*n if k=='cnn' else 32
        with init_seed(seed+401):
            self.fc1 = nn.Linear(base_input+(self.condition=='concat'),hidden)
        # Reproduce the same base head initialization when concatenation adds a column.
        if self.condition=='concat':
            with init_seed(seed+401):
                common = nn.Linear(base_input,hidden)
            with torch.no_grad():
                self.fc1.weight[:,:base_input].copy_(common.weight)
                self.fc1.bias.copy_(common.bias)
        with init_seed(seed+402):
            self.fc2 = nn.Linear(hidden,2*n*n)
        if self.ls_residual:
            # Starts exactly at LS; this is an initialization, not a dominance guarantee.
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, y, s):
        B,T = y.shape[:2]
        k = self.core_kind
        if k=='raw_lstm':
            x = y.reshape(B,T,-1)
        elif k=='cnn':
            x = self.encoder(y[:,-1])
            x = (x*torch.sigmoid(self.gate(s))[:,:,None,None]).flatten(1)
        else:
            x = self.encoder(y.reshape(B*T,2,self.n,self.n)).reshape(B,T,-1)
            x = self.proj(x)
        if k in ('lstm','raw_lstm'):
            _,(h,_) = self.core(x)
            x = h[-1]
        elif k=='gru':
            _,h = self.core(x)
            x = h[-1]
        elif k in ('mamba','transformer'):
            x = self.core(x)[:,-1]
        if k!='cnn':
            if self.condition=='gate':
                x = x*torch.sigmoid(self.gate(s))
            elif self.condition=='resgate':
                x = x*(1+torch.sigmoid(self.gate(s)))
            elif self.condition=='film':
                x = self.scale(s)*x+self.shift(s)
            elif self.condition=='concat':
                x = torch.cat([x,s],dim=-1)
        pred = self.fc2(torch.relu(self.fc1(x))).reshape(B,2,self.n,self.n)
        return y[:,-1]+pred if self.ls_residual else pred


def analytic_ops(model,T):
    """Approximate per-sample real arithmetic count; MAC=2 operations.

    Excludes biases, BatchNorm, nonlinearities, indexing, and memory movement.
    Mamba scan counted with the explicit 9*d_inner*d_state convention.
    These are operation estimates, not hardware instruction counts.
    """
    n,k,c = model.n,model.core_kind,model.condition
    cnn = 2*n*n*(2*16*9+16*8*9)
    core = 0
    if k=='cnn':
        shared = cnn
        inp = 8*n*n
    else:
        shared = 0 if k=='raw_lstm' else T*(cnn+2*8*n*n*32)
        inp = 32
        if k in ('lstm','raw_lstm','gru'):
            cin = 2*n*n if k=='raw_lstm' else 32
            core = T*2*(3 if k=='gru' else 4)*32*(cin+32)
        elif k=='transformer':
            core = T*2*32*32*4+4*T*T*32+T*2*32*64*2
        else:
            d,di,N,dc,r = 32,64,16,4,2
            core = T*(2*d*2*di+2*di*dc+2*di*(r+2*N)+2*r*di+9*di*N+di+2*di*d)
    width = 8 if k=='cnn' else 32
    cond = 2*width if c in ('gate','resgate') else 4*width if c=='film' else 0
    hidden = model.fc1.out_features
    head = 2*(inp+(c=='concat'))*hidden+2*hidden*2*n*n
    add = 2*n*n if model.ls_residual else 0
    return int(shared+core+cond+head+add)
