"""Computed paper tables, paired uncertainty, convergence diagnostics and honest claims."""
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import time
import numpy as np
from scipy.stats import t as student_t
from .common import read_json,atomic_json,job_id,jobs,model_names,stages,file_hash

LABELS = dict(LS='LS', LMMSE_instant_iid='Instantaneous LMMSE', Kalman_AR1_fixed='AR(1) Kalman',
    Jakes_LMMSE_fixed='Jakes LMMSE (fixed covariance)', Jakes_LMMSE_oracle='Jakes LMMSE (matched covariance)',
    Jakes_LMMSE_reported_snr='Jakes LMMSE (reported SNR)', Perfect_CSI='Perfect CSI',
    cnn_gate='CNN + gate', raw_lstm_gate='Raw LSTM + gate', lstm_none='CNN–LSTM',
    lstm_gate='CNN–LSTM + gate', lstm_concat='CNN–LSTM + SNR concat', lstm_film='CNN–LSTM + FiLM',
    lstm_resgate='CNN–LSTM + residual gate', gru_none='CNN–GRU', gru_gate='CNN–GRU + gate',
    mamba_none='CNN–Mamba', mamba_gate='CNN–Mamba + gate', lsres_none='LS + LSTM correction',
    lsres_gate='LS + gated LSTM correction', lstm_wide_none='CNN–LSTM (head 64)',
    lstm_wide_gate='CNN–LSTM + gate (head 64)', _transformer_gate='Transformer (timing only)')


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: raise ValueError(f'No rows for {path}')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)


def collect(root,cfg):
    rows=[]; histories=[]; convergence=[]; heads=[]
    for job in jobs(cfg):
        folder=root/'jobs'/job_id(job)
        for name in model_names(cfg,job):
            for stage in stages(job):
                rows+=read_json(folder/name/stage/'metrics.json')
                if job['kind']=='main': heads.append(read_json(folder/name/stage/'head_diagnostics.json'))
            rec=read_json(folder/name/'training.json'); hist=rec['history']
            histories += [dict(**job,model=name,**h) for h in hist]
            epoch=hist[-1]['epoch']; W=cfg['convergence_window']
            previous=min(h['val_mse'] for h in hist[:-W]) if len(hist)>W else hist[0]['val_mse']
            gain=max(0.,(previous-rec['best_val'])/previous)
            cap=rec['stop_reason']=='epoch_cap'
            convergence.append(dict(**job,model=name,epochs_completed=epoch,best_epoch=rec['best_epoch'],
                stop_reason=rec['stop_reason'],training_seconds=rec['training_seconds'],
                best_val=rec['best_val'],reference_best_val=rec['reference_best_val'],
                validation_relative_gain_vs_reference=(rec['reference_best_val']-rec['best_val'])/rec['reference_best_val'],
                recent_best_validation_relative_gain=gain,recent_window=W,
                best_within_last_window=rec['best_epoch']>max(0,epoch-W),
                needs_convergence_review=cap and (gain>=cfg['convergence_relative_gain'] or rec['best_epoch']>epoch-W),
                initialization_origin=rec['origin']['kind']))
        first=cfg['seeds'][0] if job['kind']=='main' else cfg['seq_seeds'][0]
        if job['seed']==first: rows+=read_json(folder/'baselines'/'metrics.json')
    return rows,histories,convergence,heads


def summarize(rows):
    fields=['suite','n','T','model','stage','scenario','parameter','snr_db']
    group=defaultdict(list)
    for row in rows: group[tuple(row[k] for k in fields)].append(row)
    out=[]
    for key,rs in sorted(group.items()):
        row=dict(zip(fields,key)); classical=rs[0]['stage']=='classical'
        row['n_training_seeds']=0 if classical else len(rs)
        row['uncertainty_unit']='independent channel blocks' if classical else 'training seeds; test blocks shared'
        for metric in ['nmse','ber','zf_sum_rate']:
            if metric not in rs[0]: continue
            vals=np.array([r[metric] for r in rs])
            row[metric+'_mean']=float(vals.mean())
            row[metric+'_seed_std']=None if classical else float(vals.std(ddof=1)) if len(vals)>1 else 0.
        if classical:
            row['nmse_mc_ci95_halfwidth']=rs[0]['nmse_ci95_halfwidth']
            for k in ['ber_ci95_low','ber_ci95_high','rate_ci95_low','rate_ci95_high']:
                if k in rs[0]: row[k]=rs[0][k]
        out.append(row)
    return out


def paired_comparisons(root,cfg,rows):
    from .simulation import bootstrap_mean
    fields=['suite','n','T','stage','model','scenario','parameter','snr_db','seed']
    index={tuple(r[k] for k in fields):r for r in rows}
    @lru_cache(maxsize=64)
    def arrays(path):
        with np.load(root/path,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
    out=[]
    pairs=[('lstm_gate','lstm_none'),('mamba_gate','mamba_none'),('gru_gate','gru_none'),
        ('lsres_gate','lsres_none'),('lstm_gate','lstm_concat'),('lstm_gate','lstm_film'),
        ('lstm_wide_gate','lstm_wide_none'),('lstm_wide_gate','lstm_gate'),('lstm_wide_none','lstm_none'),
        ('raw_lstm_gate','lstm_gate'),('lsres_gate','lstm_gate'),('mamba_gate','lstm_gate'),
        ('lsres_gate','LS'),('lsres_gate','Jakes_LMMSE_fixed'),('lstm_gate','Jakes_LMMSE_fixed'),
        ('mamba_gate','Jakes_LMMSE_fixed')]
    experiments=sorted({(r['suite'],r['n'],r['T'],r['scenario'],r['parameter'],r['snr_db']) for r in rows})
    for suite,n,T,scenario,parameter,snr in experiments:
        seeds=cfg['seeds'] if suite=='main' else cfg['seq_seeds']
        contrasts=[]
        for stage in (['budget60','extended'] if suite=='main' else ['extended']):
            contrasts += [(a,stage,b,'classical' if b in ('LS','Jakes_LMMSE_fixed') else stage,'method') for a,b in pairs]
        if suite=='main':
            names=cfg['models']+(cfg['wide_models'] if n in cfg['wide_arrays'] else [])
            contrasts += [(name,'extended',name,'budget60','training_extension') for name in names]
        for a,sa,b,sb,kind in contrasts:
            refs=[]
            for seed in seeds:
                ka=(suite,n,T,sa,a,scenario,parameter,snr,seed)
                kb=(suite,n,T,sb,b,scenario,parameter,snr,None if sb=='classical' else seed)
                if ka not in index or kb not in index: break
                refs.append((index[ka],index[kb]))
            if len(refs)!=len(seeds): continue
            e_delta=[]; nmse_diffs=[]; ber_delta=[]; rate_delta=[]
            for ar,br in refs:
                aa,bb=arrays(ar['raw_file']),arrays(br['raw_file'])
                power=aa['channel_energy']
                if not np.array_equal(power,bb['channel_energy']): raise ValueError('Unpaired raw channels')
                delta=aa['error_energy']-bb['error_energy']; e_delta.append(delta)
                nmse_diffs.append(float(delta.sum()/power.sum()))
                if 'ber_errors' in aa:
                    if aa['bits_per_block']!=bb['bits_per_block']: raise ValueError('Unpaired BER payloads')
                    ber_delta.append((aa['ber_errors']-bb['ber_errors'])/float(aa['bits_per_block']))
                    rate_delta.append(aa['rate_blocks']-bb['rate_blocks'])
            delta=np.mean(e_delta,axis=0); mean=float(delta.sum()/power.sum())
            se=float(np.std(delta-mean*power,ddof=1)/np.sqrt(len(power))/power.mean())
            sd=float(np.std(nmse_diffs,ddof=1)) if len(seeds)>1 else 0.
            seed_hw=float(student_t.ppf(.975,len(seeds)-1)*sd/np.sqrt(len(seeds))) if len(seeds)>1 else None
            row=dict(suite=suite,n=n,T=T,scenario=scenario,parameter=parameter,snr_db=snr,
                candidate=a,candidate_stage=sa,reference=b,reference_stage=sb,contrast=kind,
                nmse_diff_mean=mean,nmse_diff_seed_std=sd,
                nmse_mc_conditional_ci95_low=mean-1.96*se,nmse_mc_conditional_ci95_high=mean+1.96*se,
                nmse_seed_t_ci95_low=None if seed_hw is None else mean-seed_hw,
                nmse_seed_t_ci95_high=None if seed_hw is None else mean+seed_hw,
                n_seeds=len(seeds),seed_wins=int(np.sum(np.array(nmse_diffs)<0)),
                interpretation='candidate minus reference; pointwise intervals; MC conditional on fitted models; seed-t conditional on test set')
            if ber_delta:
                ci=bootstrap_mean(np.mean(ber_delta,axis=0),resamples=cfg['bootstrap_resamples'])
                rate=bootstrap_mean(np.mean(rate_delta,axis=0),resamples=cfg['bootstrap_resamples'])
                row.update(ber_diff=ci['mean'],ber_mc_conditional_ci95_low=ci['ci95_low'],ber_mc_conditional_ci95_high=ci['ci95_high'],
                    rate_diff=rate['mean'],rate_mc_conditional_ci95_low=rate['ci95_low'],rate_mc_conditional_ci95_high=rate['ci95_high'])
            out.append(row)
    arrays.cache_clear()
    return out


def figures(root,cfg,summary,histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.labelsize':10,'legend.fontsize':7,'pdf.fonttype':42,
        'savefig.dpi':250,'axes.grid':True,'grid.alpha':.25})
    dest=root/'Figures'; dest.mkdir(exist_ok=True)
    colors=['#16697a','#d97706','#7044a3','#15803d','#b6254b','#475569','#816323','#2563eb']
    def draw(name,data,names,x='snr_db',metric='nmse',xlabel='Nominal SNR (dB)'):
        fig,ax=plt.subplots(figsize=(7.4,4.7),layout='constrained')
        for i,m in enumerate(names):
            rs=sorted([r for r in data if r['model']==m],key=lambda r:r[x])
            if not rs: continue
            err=[r.get(metric+'_seed_std') or 0. for r in rs]
            if rs[0]['stage']=='classical' and metric=='nmse': err=[r['nmse_mc_ci95_halfwidth'] for r in rs]
            ax.errorbar([r[x] for r in rs],[r[metric+'_mean'] for r in rs],yerr=err,
                marker=['o','s','^','D','v','P','X','*'][i%8],color=colors[i%8],lw=1.4,ms=3.5,capsize=2,
                label=LABELS.get(m,m))
        if metric in ('nmse','ber'): ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(dict(nmse='NMSE',ber='QPSK block BER',zf_sum_rate='Effective ZF sum rate (bit/s/Hz)')[metric])
        ax.legend(loc='best'); ax.set_title(name.replace('_',' ')+(' [SMOKE]' if cfg['mode']=='smoke' else ''))
        fig.savefig(dest/(name+'.pdf')); fig.savefig(dest/(name+'.png')); plt.close(fig)
    for n in cfg['arrays']:
        data=[r for r in summary if r['suite']=='main' and r['n']==n and r['stage'] in ('extended','classical')]
        main=[r for r in data if r['scenario']=='main']
        primary=['LS','Jakes_LMMSE_fixed','raw_lstm_gate','lstm_none','lstm_gate','mamba_gate','lsres_gate']
        for metric in ('nmse','ber','zf_sum_rate'): draw(f'{n}x{n}_{metric}',main,primary,metric=metric)
        draw(f'{n}x{n}_conditioning',main,['lstm_none','lstm_concat','lstm_film','lstm_gate','lstm_resgate'])
        draw(f'{n}x{n}_backbones',main,['lstm_none','lstm_gate','gru_none','gru_gate','mamba_none','mamba_gate'])
        draw(f'{n}x{n}_ls_residual',main,['LS','Jakes_LMMSE_fixed','lstm_gate','lsres_none','lsres_gate'])
        if n in cfg['wide_arrays']:
            draw(f'{n}x{n}_head_width_control',main,['LS','Jakes_LMMSE_fixed','lstm_none','lstm_gate','lstm_wide_none','lstm_wide_gate'])
        for scenario in ['temporal','spatial','snr_bias','snr_error_std']:
            for snr in sorted({r['snr_db'] for r in data if r['scenario']==scenario}):
                sub=[r for r in data if r['scenario']==scenario and r['snr_db']==snr]
                for stem in ['lstm','mamba','lsres']:
                    draw(f'{n}x{n}_{scenario}_{snr:g}dB_{stem}',sub,
                        ['LS','Jakes_LMMSE_reported_snr' if scenario.startswith('snr_') else 'Jakes_LMMSE_fixed',
                         'Jakes_LMMSE_oracle',stem+'_none',stem+'_gate'],x='parameter',
                        xlabel={'temporal':'Test lag-one correlation','spatial':'Spatial correlation',
                            'snr_bias':'Reported SNR bias (dB)','snr_error_std':'Reported SNR error std (dB)'}[scenario])
        fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
        for ax,name in zip(axes,['lstm_none','lstm_gate']):
            for seed in cfg['seeds']:
                rs=[r for r in histories if r['kind']=='main' and r['n']==n and r['model']==name and r['seed']==seed]
                ax.plot([r['epoch'] for r in rs],[r['val_mse'] for r in rs],label=f'Seed {seed}')
            ax.axvline(cfg['reference_epochs'],color='0.4',ls='--',lw=1)
            ax.set(xlabel='Epoch',ylabel='Validation MSE',title=LABELS[name]); ax.set_yscale('log'); ax.legend()
        fig.savefig(dest/f'{n}x{n}_validation_curves.pdf'); plt.close(fig)
    draw('sequence_equal_budget',[r for r in summary if r['suite']=='seq'],
        ['LS','Jakes_LMMSE_fixed','lstm_gate','mamba_gate','lsres_gate'],x='T',xlabel='Observation window T')
    v=read_json(root/'channel_validation.json')
    fig,ax=plt.subplots(figsize=(6.8,4),layout='constrained')
    ax.plot(v['target'],label='J0 target'); ax.plot(v['empirical'],'o',ms=3,label='Empirical SoS')
    ax.set(xlabel='Lag in pilot-block intervals',ylabel='Normalized autocorrelation'); ax.legend()
    fig.savefig(dest/'jakes_validation.pdf'); plt.close(fig)
    (dest/'CAPTIONS.md').write_text(
        '# Figure interpretation\n\nNeural error bars show one sample standard deviation across training seeds. '
        'Classical NMSE bars show 95% Monte Carlo half-widths across independent channel sequences. '
        'These describe different sources of uncertainty. For paired differences use Paper_Data/paired_comparisons.csv.\n\n'
        'Main and robustness figures use extended checkpoints selected only on validation MSE. '
        'Sequence experiments use identical train/validation counts and three training seeds per window in publication mode; '
        'accuracy is measured at 15 dB. Budget60 comparisons are in the exported tables and CSV files.\n\n'
        'Reported-SNR robustness applies the same erroneous side information to neural gates and the reported-SNR Jakes estimator. '
        'The fixed-covariance Jakes estimator in raw CSV side-information rows uses true nominal SNR and is a reference only.\n\n'
        'QPSK payload blocks hold the channel fixed. The Gaussian-input effective ZF rate is a diagnostic proxy without pilot overhead.\n')


def timing(root,cfg,device):
    import torch
    from .models import Estimator,analytic_ops
    from .training import load_checkpoint,seed_torch
    torch.set_num_threads(cfg['torch_threads'])
    if device.type=='cuda': torch.cuda.set_device(device)
    def measure(model,T,B):
        x=torch.zeros(B,T,2,model.n,model.n,device=device); s=torch.zeros(B,1,device=device)
        model.eval(); times=[]
        with torch.inference_mode():
            for _ in range(10): model(x,s)
            for _ in range(cfg['latency_trials']):
                if device.type=='cuda': torch.cuda.synchronize(device)
                start=time.perf_counter(); model(x,s)
                if device.type=='cuda': torch.cuda.synchronize(device)
                times.append((time.perf_counter()-start)*1000)
        return dict(batch_latency_ms_median=float(np.median(times)),batch_latency_ms_p95=float(np.quantile(times,.95)),
            amortized_ms_per_sample=float(np.median(times)/B),batch_size=B)
    complexity=[]
    for n in cfg['arrays']:
        job=dict(kind='main',n=n,T=cfg['main_T'],seed=cfg['seeds'][0]); folder=root/'jobs'/job_id(job)
        for name in model_names(cfg,job):
            model=Estimator(name,n).to(device); model.load_state_dict(load_checkpoint(folder/name/'best.pt')['state_dict'])
            complexity.append(dict(n=n,T=cfg['main_T'],model=name,parameters=sum(p.numel() for p in model.parameters()),
                approximate_operations=analytic_ops(model,cfg['main_T']),device=str(device),
                **measure(model,cfg['main_T'],1)))
            del model
    sweep=[]
    for T in cfg['latency_T']:
        for name in ['lstm_gate','mamba_gate','_transformer_gate']:
            if name.startswith('mamba') and not cfg['include_mamba']: continue
            seed_torch(42); model=Estimator(name,2).to(device)
            sweep.append(dict(n=2,T=T,model=name,weights='fresh initialization; timing only',device=str(device),
                parameters=sum(p.numel() for p in model.parameters()),approximate_operations=analytic_ops(model,T),
                **measure(model,T,cfg['latency_batch'])))
            del model
    write_csv(root/'Paper_Data'/'complexity.csv',complexity)
    write_csv(root/'Paper_Data'/'latency_vs_T.csv',sweep)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    for name in sorted({r['model'] for r in sweep}):
        rs=[r for r in sweep if r['model']==name]
        axes[0].plot([r['T'] for r in rs],[r['amortized_ms_per_sample'] for r in rs],'o-',label=LABELS[name])
        axes[1].plot([r['T'] for r in rs],[r['approximate_operations'] for r in rs],'o-',label=LABELS[name])
    axes[0].set(xlabel='T',ylabel='Amortized time per sample (ms)')
    axes[1].set(xlabel='T',ylabel='Estimated arithmetic operations (MAC = 2)')
    for ax in axes: ax.set_xscale('log',base=2); ax.set_yscale('log'); ax.legend(fontsize=7)
    fig.savefig(root/'Figures'/'latency_and_operations.pdf'); plt.close(fig)
    return sweep


def tables(root,cfg,summary):
    for n in cfg['arrays']:
        for stage in ['budget60','extended']:
            rs=[r for r in summary if r['suite']=='main' and r['n']==n and r['scenario']=='main' and r['stage'] in (stage,'classical')]
            names=list(dict.fromkeys(r['model'] for r in rs if r['model']!='Perfect_CSI'))
            table=['% Neural: mean +/- training-seed std; classical: mean +/- 95% MC half-width.',
                r'\begin{tabular}{lccc}',r'\hline',r'Method & 0 dB & 15 dB & 30 dB \\',r'\hline']
            for name in names:
                cells=[]
                for snr in [0,15,30]:
                    row=next(r for r in rs if r['model']==name and r['snr_db']==snr)
                    err=row['nmse_mc_ci95_halfwidth'] if row['stage']=='classical' else row['nmse_seed_std']
                    cells.append('$'+f'{row["nmse_mean"]:.4g}'+r' \pm '+f'{err:.2g}'+'$')
                label=LABELS.get(name,name).replace('–','--').replace('_',r'\_')
                table.append(label+' & '+' & '.join(cells)+r' \\')
            table += [r'\hline',r'\end{tabular}']
            (root/'Paper_Data'/f'nmse_{n}x{n}_{stage}.tex').write_text('\n'.join(table)+'\n')


def manuscript_report(root,cfg,summary,convergence,pairs,sweep):
    capped=[r for r in convergence if r['stop_reason']=='epoch_cap']
    review=[r for r in convergence if r['needs_convergence_review']]
    primary=[r for r in summary if r['suite']=='main' and r['scenario']=='main' and r['stage'] in ('extended','classical')]
    decisions=[]
    for n in cfg['arrays']:
        for snr in cfg['snrs']:
            rs={r['model']:r for r in primary if r['n']==n and r['snr_db']==snr}
            nn=[r for r in rs.values() if r['stage']=='extended']
            best=min(nn,key=lambda r:r['nmse_mean'])
            baseline=rs['Jakes_LMMSE_fixed']['nmse_mean']
            decisions.append(dict(n=n,snr_db=snr,lowest_mean_neural_nmse_method=best['model'],
                neural_nmse=best['nmse_mean'],fixed_jakes_nmse=baseline,
                lowest_neural_mean_beats_fixed_jakes=best['nmse_mean']<baseline,
                label='descriptive comparison only; not test-selected deployment model'))
    claims=dict(epoch_capped_fits=len(capped),convergence_review_fits=len(review),total_fits=len(convergence),
        baseline_comparisons=decisions,interpretation='No acceptance decision. Model improvements and mechanisms require the actual paired evidence.')
    atomic_json(root/'Paper_Data'/'CLAIM_CHECKS.json',claims)
    lines=['# Results guide for the manuscript','',
        f'Mode: **{cfg["mode"]}**. Completed {len(convergence)} fits. The maximum is {cfg["epochs"]} epochs; '
        f'{len(capped)} fits reached that limit and {len(review)} require a convergence review. '
        'A patience stop does not prove a global optimum. A capped fit can only support a stated training-budget result.','',
        'All designs and validation-selected checkpoints were frozen before fresh test generation. '
        'The 60-epoch snapshot is the best validation checkpoint within that budget; it is not necessarily epoch 60 itself. '
        'Extended and budget60 predictions use the same new test channels.','',
        '## Main NMSE at 30 dB','',
        '| Method | Array | Mean NMSE | Uncertainty |','|---|---|---:|---|']
    for row in primary:
        if row['snr_db']!=30 or row['model']=='Perfect_CSI': continue
        err=(f'95% MC half-width {row["nmse_mc_ci95_halfwidth"]:.3g}' if row['stage']=='classical'
             else f'Seed std {row["nmse_seed_std"]:.3g}')
        lines.append(f'| {LABELS[row["model"]]} | {row["n"]}×{row["n"]} | {row["nmse_mean"]:.6g} | {err} |')
    lines += ['', '## How to write the conclusions','',
        '- Use `paired_comparisons.csv` for gating, head width, and training-extension differences. Negative NMSE/BER differences favor the candidate. Each pair uses identical test channels and data-link random draws.',
        '- Report all three seeds, including reversals. Monte Carlo intervals condition on fitted models; the separate seed-t interval conditions on the finite test set and is imprecise with three seeds. These are pointwise intervals without multiplicity correction.',
        '- Keep the fixed Jakes LMMSE baseline in every main accuracy comparison. Under the matched scenario its covariance is already matched; it is not disadvantaged by an unknown test covariance.',
        '- Compare wide gated against wide ungated, and each wide model against its original-width counterpart. Changing width tests one architectural factor; it does not by itself identify an attenuation mechanism.',
        '- Head diagnostics show units inactive on one observed test set and their output span. They do not establish that a unit was always inactive during training.',
        '- Robustness claims require gated/ungated pairs at the same mismatch. SNR-error plots include a classical estimator using the same reported SNR.',
        '- Sequence experiments use three seeds per window and the same train/validation counts. They evaluate accuracy at 15 dB. Timing is measured after concurrent training has stopped, at a fixed batch size.',
        '- Describe the payload BER as a frozen-channel diagnostic. The effective ZF Gaussian-input rate excludes pilot overhead and is not measured QPSK throughput.',
        '- Read `convergence.csv` before using the word converged. Keep the 240-epoch cap visible if any central fit still improves. Do not tune again on this confirmation set to obtain a preferred ranking.',
        '', '## Files to use','',
        '`summary_mean_std.csv`, `paired_comparisons.csv`, `convergence.csv`, `head_diagnostics.csv`, '
        '`complexity.csv`, `latency_vs_T.csv`, the compact `.tex` tables, and `Figures/CAPTIONS.md`. '
        'All scalar metrics are traceable to saved raw per-channel statistics. `FINAL_AUDIT.json` checks completeness and arithmetic only.','']
    (root/'Paper_Data'/'RESULTS_SUMMARY.md').write_text('\n'.join(lines))


def audit(root,cfg,rows,convergence):
    checks=[]
    def check(name,value): checks.append(dict(check=name,passed=bool(value)))
    check('Jakes moments',read_json(root/'channel_validation.json')['passed'])
    check('all jobs completed',all((root/'jobs'/job_id(j)/'COMPLETE.json').exists() for j in jobs(cfg)))
    check('expected number of fitted models',len(convergence)==sum(len(model_names(cfg,j)) for j in jobs(cfg)))
    check('finite metrics',all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v,(float,int))))
    selection=read_json(root/'frozen_selection.json')
    check('selected checkpoint hashes unchanged',all(file_hash(root/r['checkpoint'])==r['sha256'] for r in selection['checkpoints']))
    seen=set(); unique=True; agrees=True
    for r in rows:
        key=tuple(r[k] for k in ['suite','n','T','model','stage','seed','scenario','parameter','snr_db'])
        if key in seen: unique=False
        seen.add(key)
        with np.load(root/r['raw_file'],allow_pickle=False) as a:
            agrees &= bool(np.isclose(a['error_energy'].sum()/a['channel_energy'].sum(),r['nmse'],rtol=1e-10,atol=1e-12))
            if 'ber' in r:
                agrees &= bool(np.isclose(a['ber_errors'].mean()/float(a['bits_per_block']),r['ber'],rtol=1e-10,atol=1e-12))
                agrees &= bool(np.isclose(a['rate_blocks'].mean(),r['zf_sum_rate'],rtol=1e-10,atol=1e-12))
    check('unique metric identities',unique); check('all scalar metrics reproduce from raw arrays',agrees)
    check('valid NMSE and BER range',all(r['nmse']>=0 and 0<=r.get('ber',0)<=1 for r in rows))
    check('equal sequence budgets and seed counts',all(read_json(root/'jobs'/job_id(j)/'protocol.json')['train_samples']==cfg['seq_train']
        and read_json(root/'jobs'/job_id(j)/'protocol.json')['val_samples']==cfg['seq_val'] for j in jobs(cfg) if j['kind']=='seq'))
    result=dict(checks_passed=all(c['passed'] for c in checks),mode=cfg['mode'],publication_run=cfg['mode']=='publication',
        total_fits=len(convergence),metric_rows=len(rows),checks=checks,
        convergence_review_fits=sum(r['needs_convergence_review'] for r in convergence),
        meaning='Completeness and numerical consistency only; not proof of convergence, novelty or publication acceptance.')
    atomic_json(root/'FINAL_AUDIT.json',result)
    if not result['checks_passed']: raise RuntimeError('Final audit failed; inspect FINAL_AUDIT.json')
    return result


def export(root,cfg,device):
    dest=root/'Paper_Data'; dest.mkdir(exist_ok=True)
    rows,histories,convergence,heads=collect(root,cfg); summary=summarize(rows)
    pairs=paired_comparisons(root,cfg,rows)
    for filename,data in [('metrics_by_seed.csv',rows),('summary_mean_std.csv',summary),('paired_comparisons.csv',pairs),
        ('training_history.csv',histories),('convergence.csv',convergence),('head_diagnostics.csv',heads)]:
        write_csv(dest/filename,data)
    atomic_json(dest/'all_metrics.json',rows)
    figures(root,cfg,summary,histories); tables(root,cfg,summary)
    sweep=timing(root,cfg,device)
    manuscript_report(root,cfg,summary,convergence,pairs,sweep)
    return audit(root,cfg,rows,convergence)
