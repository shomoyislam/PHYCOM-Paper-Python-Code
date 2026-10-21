"""Reconstruct metrics from raw blocks; distinguish training variation from MC error."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest,t
from .common import atomic_json,read_json,group_id,file_hash,seed,check_deadline
from .protocol import groups,models,spec_id,comparisons


def ratio_se(e,p):
    ratio=float(e.sum()/p.sum())
    se=float(np.std(e-ratio*p,ddof=1)/np.sqrt(len(e))/p.mean()) if len(e)>1 else 0.
    return ratio,se


def percentile_interval(values,rng,resamples):
    if len(values)<2: return None,None
    v=np.asarray(values,np.float64)
    samples=v[rng.integers(0,len(v),(resamples,len(v)))].mean(1)
    return tuple(map(float,np.quantile(samples,[.025,.975])))


def block_interval(values,rng,resamples):
    values=np.asarray(values,np.float64);means=[]
    for lo in range(0,resamples,32):
        ix=rng.integers(0,len(values),(min(32,resamples-lo),len(values)))
        means.extend(values[ix].mean(1))
    return tuple(map(float,np.quantile(means,[.025,.975])))


def holm(pvalues):
    p=np.asarray(pvalues,float);ix=np.argsort(p);out=np.empty(len(p))
    out[ix]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[ix]))
    return out


def generate_report(root,c,run_id):
    root=Path(root);paper=root/'Paper_Data';paper.mkdir(exist_ok=True)
    rows=[];training=[];coverage=[];spectra=[];singular=[];rawlookup={};cost=[]
    for g in groups(c):
        gid=group_id(g);gd=root/'jobs'/gid
        for s in models(c,g):
            md=gd/spec_id(s);p=md/'training.json'
            if p.exists() and read_json(p).get('complete'):
                d=read_json(p)
                training.append(dict(group=gid,family=g['family'],n=g['n'],seed=g['seed'],model=spec_id(s),
                    **{k:d[k] for k in ('best_epoch','best_val','epochs_run','training_seconds','parameters','stop_reason',
                        'recent_relative_validation_gain','validation_residual_scale')}))
            diag=md/'diagnostics.json'
            if diag.exists():
                d=read_json(diag)
                for kind in ('feature_spectra','weight_spectra'):
                    for part,v in d[kind].items():
                        spectra.append(dict(group=gid,model=spec_id(s),kind=kind,part=part,
                            **{k:v[k] for k in ('numerical_rank','domain_nullity','condition_number_nonzero_spectrum',
                                'effective_rank_variance_entropy','stable_rank')}))
                        for j,value in enumerate(v['singular_values']):
                            singular.append(dict(group=gid,family=g['family'],n=g['n'],seed=g['seed'],model=spec_id(s),
                                kind=kind,part=part,index=j+1,singular_value=value,
                                relative_value=value/max(v['singular_values'][0],1e-30),rank_tolerance=v['rank_tolerance']))
        cp=gd/'covariances.json'
        if cp.exists():
            for name,record in read_json(cp).items():
                cost.append(dict(group=gid,method=name,**record))
        for p in sorted((gd/'evaluation').glob('*/COMPLETE.json')):
            record=read_json(p);case=record['case'];coverage.append(dict(group=gid,case=case['name'],**{k:record[k] for k in
                ('samples','outside_training_fraction','reported_min','reported_max','overhead_noise_samples')}))
            if record['run_id']!=run_id: raise RuntimeError('Wrong evaluation run')
            for method,sha in record['files'].items():
                file=p.parent/(method+'.npz')
                if file_hash(file)!=sha: raise RuntimeError(f'Raw-data hash mismatch: {file}')
                with np.load(file) as a:
                    if str(a['run_id'])!=run_id: raise RuntimeError('Raw identity mismatch')
                    e,energy=a['errors'],a['power'];nmse,se=ratio_se(e,energy)
                    row=dict(group=gid,family=g['family'],n=g['n'],seed=g['seed'],case=case['name'],
                        category=case['category'],snr=case['snr'],method=method,nmse=nmse,
                        nmse_mc_se_conditional=se,samples=len(e),
                        correction_energy=float(a['correction_energy'].sum()/energy.sum()),
                        correction_cross=float(a['correction_cross'].sum()/energy.sum()))
                    if 'ber_errors' in a:
                        row.update(ber=float(a['ber_errors'].sum()/(len(a['ber_errors'])*int(a['bits_per_block']))),
                            zf_rate_proxy=float(a['rate'].mean()),ber_independent_blocks=len(a['ber_errors']))
                    for label,mask in [('in_range',a['outside_training_range']==0),('out_of_range',a['outside_training_range']==1)]:
                        row[label+'_count']=int(mask.sum())
                        row[label+'_nmse']=float(e[mask].sum()/energy[mask].sum()) if mask.any() else None
                    rows.append(row)
                rawlookup[(g['family'],g['n'],g['seed'],case['name'],method)]=file
    expected=sum(len(models(c,g)) for g in groups(c))
    complete_train=len(training)==expected
    pd.DataFrame(training).to_csv(paper/'convergence.csv',index=False)
    pd.DataFrame(coverage).to_csv(paper/'evaluation_coverage.csv',index=False)
    pd.DataFrame(spectra).to_csv(paper/'representation_diagnostics.csv',index=False)
    pd.DataFrame(singular).to_csv(paper/'singular_value_spectra.csv',index=False)
    pd.DataFrame(cost).to_csv(paper/'covariance_fitting_cost.csv',index=False)
    metrics=pd.DataFrame(rows);metrics.to_csv(paper/'metrics_per_seed.csv',index=False)
    if metrics.empty:
        atomic_json(root/'FINAL_AUDIT.json',dict(run_id=run_id,complete=False,training_complete=complete_train,
            expected_fits=expected,completed_fits=len(training),reason='Training/evaluation pending'))
        (paper/'RESULTS_SUMMARY.md').write_text(f'# Revision status\n\n{len(training)}/{expected} fits completed. '
            'No confirmation results released yet. Resume the same configuration.\n')
        return
    keys=['family','n','case','category','snr','method']
    summary=metrics.groupby(keys).agg(nmse_mean=('nmse','mean'),nmse_sd=('nmse','std'),seed_rows=('seed','nunique'),
        ber_mean=('ber','mean'),ber_sd=('ber','std'),zf_rate_mean=('zf_rate_proxy','mean')).reset_index()
    summary['independent_training_runs']=summary.apply(lambda r: int(r.seed_rows) if r.method.startswith(('cnn_','raw_')) else
        (int(r.seed_rows) if r.method.startswith('Empirical_') else 0),axis=1)
    summary.to_csv(paper/'summary_mean_std.csv',index=False)
    paired=[]
    for family,n,case in metrics[['family','n','case']].drop_duplicates().itertuples(index=False,name=None):
        block=metrics[(metrics.family==family)&(metrics.n==n)&(metrics.case==case)]
        declared=comparisons(c,int(n),family)
        # Every trained neural method is also compared against LS and the best-informed linear reference.
        neural=[x for x in block.method.unique() if x.startswith(('cnn_','raw_'))]
        empirical=f"Empirical_LMMSE_K{c['train_samples']}"
        declared += [(x,ref,'classical_comparison') for x in neural for ref in ('LS','Oracle_component_trueSNR',empirical)]
        unique={}
        for a,b,kind in declared: unique.setdefault((a,b),kind)
        for (a,b),kind in unique.items():
            aa=block[block.method==a].set_index('seed');bb=block[block.method==b].set_index('seed')
            shared=sorted(set(aa.index)&set(bb.index))
            if not shared: continue
            diff=(aa.loc[shared,'nmse']-bb.loc[shared,'nmse']).to_numpy()
            rng=np.random.default_rng(seed('paired',family,n,case,a,b))
            low,high=percentile_interval(diff,rng,c['bootstrap_resamples'])
            nonzero=diff[diff!=0];wins=int(np.sum(diff<0));m=len(diff)
            p=float(binomtest(np.sum(nonzero<0),len(nonzero),.5).pvalue) if len(nonzero) else 1.
            primary=(family=='iid' and kind=='gating' and case in [f'main_snr{s}' for s in c['primary_snrs']]
                     and a in (f'cnn_p32_h32_head{max(16,2*n*n)}_gate',f'raw_p0_h32_head{max(16,2*n*n)}_gate'))
            rec=dict(family=family,n=n,case=case,candidate=a,reference=b,comparison=kind,
                paired_seeds=m,delta_nmse_mean=float(diff.mean()),candidate_wins=wins,
                relative_reduction_percent=float(-diff.mean()/bb.loc[shared,'nmse'].mean()*100),
                seed_bootstrap_low=low,seed_bootstrap_high=high,sign_test_p=p,
                primary_family=bool(primary),sign_test_p_holm_primary=None)
            if m>1:
                half=float(t.ppf(.975,m-1)*diff.std(ddof=1)/np.sqrt(m))
                rec.update(seed_t_sensitivity_low=float(diff.mean()-half),seed_t_sensitivity_high=float(diff.mean()+half))
            # Conditional channel uncertainty is separate from training-seed uncertainty.
            if primary or (kind in ('residual_vs_LS','scaled_residual_vs_LS') and case=='main_snr30'):
                es=[];ber=[];rates=[];power=None
                for ss in shared:
                    with np.load(rawlookup[(family,n,ss,case,a)]) as ar, np.load(rawlookup[(family,n,ss,case,b)]) as br:
                        if not np.array_equal(ar['power'],br['power']): raise RuntimeError('Unpaired evaluation channels')
                        if power is not None and not np.array_equal(power,ar['power']): raise RuntimeError('Seeds used different confirmation sets')
                        power=ar['power'].copy();es.append(ar['errors']-br['errors'])
                        if 'ber_errors' in ar and 'ber_errors' in br:
                            ber.append((ar['ber_errors']-br['ber_errors'])/int(ar['bits_per_block']))
                            rates.append(ar['rate']-br['rate'])
                ed=np.mean(es,axis=0);ratio,se=ratio_se(ed,power)
                rec.update(conditional_channel_delta=ratio,conditional_channel_ci_low=ratio-1.96*se,
                    conditional_channel_ci_high=ratio+1.96*se)
                if ber:
                    blo,bhi=block_interval(np.mean(ber,axis=0),rng,c['bootstrap_resamples'])
                    rlo,rhi=block_interval(np.mean(rates,axis=0),rng,c['bootstrap_resamples'])
                    rec.update(paired_ber_delta=float(np.mean(ber)),conditional_ber_block_low=blo,conditional_ber_block_high=bhi,
                        paired_rate_delta=float(np.mean(rates)),conditional_rate_block_low=rlo,conditional_rate_block_high=rhi)
            paired.append(rec)
    # Twelve primary hypotheses: 3 arrays x 2 SNRs x {CNN,raw}. Missing tests stay missing.
    indices=[i for i,x in enumerate(paired) if x['primary_family']]
    if len(indices)==len(c['primary_arrays'])*len(c['primary_snrs'])*2 and all(paired[i]['paired_seeds']==len(c['seeds']) for i in indices):
        for i,p in zip(indices,holm([paired[i]['sign_test_p'] for i in indices])):
            paired[i]['sign_test_p_holm_primary']=float(p)
    pd.DataFrame(paired).to_csv(paper/'paired_comparisons.csv',index=False)
    report_error_comparisons(paper,metrics,rawlookup,c)
    init=metrics[metrics.category=='main'].pivot_table(index=['family','n','seed','case'],columns='method',values='nmse')
    init.to_csv(paper/'all_model_nmse_wide.csv')
    # Exact residual-risk identity independently checked against saved LS arrays.
    max_identity_error=0.
    for row in rows:
        if not row['method'].endswith(('_lsres','__val_scaled')): continue
        ls=rawlookup[(row['family'],row['n'],row['seed'],row['case'],'LS')]
        with np.load(ls) as a: ls_nmse=float(a['errors'].sum()/a['power'].sum())
        discrepancy=abs(row['nmse']-ls_nmse-row['correction_energy']-row['correction_cross'])
        max_identity_error=max(max_identity_error,discrepancy)
    if max_identity_error>1e-8: raise RuntimeError('Residual decomposition failed')
    complete_eval=all((root/'jobs'/group_id(g)/'EVAL_COMPLETE.json').exists() for g in groups(c))
    capped=[x for x in training if x['stop_reason']=='epoch_cap']
    unresolved=[x for x in capped if x['recent_relative_validation_gain']>c['convergence_recent_gain_threshold']]
    timing_complete=(paper/'TIMING_COMPLETE.json').exists()
    audit=dict(run_id=run_id,complete=complete_train and complete_eval and timing_complete,
        confirmation_complete=complete_eval,timing_complete=timing_complete,mode=c['mode'],expected_fits=expected,
        completed_fits=len(training),metric_rows=len(rows),paired_rows=len(paired),primary_tests=len(indices),
        epoch_cap_fits=len(capped),capped_with_recent_gain_above_threshold=len(unresolved),
        max_residual_identity_error=max_identity_error,raw_hashes_verified=True,
        convergence_not_proven=True,scientific_superiority_not_assumed=True)
    atomic_json(root/'FINAL_AUDIT.json',audit)
    text=['# PHYCOM V4 revision results','',f"Mode: **{c['mode']}**. Complete: **{audit['complete']}**.",
        f"Fits: {len(training)}/{expected}; central independent seeds: {len(c['seeds'])}; supplemental seeds: {c['supplemental_seed_count']}.",
        f"Epoch-cap fits: {len(capped)}; capped fits with >{100*c['convergence_recent_gain_threshold']:g}% recent validation gain: {len(unresolved)}.",
        '', 'All confirmation results are generated after all model selections are frozen. '
        'The experiment is pre-specified in config.json; it is not an externally registered protocol. '
        'Do not tune on these confirmation results and then call the same set fresh.', '',
        'Primary inference: exact two-sided sign tests on paired training-run differences with Holm across '
        '12 predeclared tests (three arrays, CNN/raw LSTM, 15/30 dB). This tests directional consistency/median effects, '
        'not the population mean. Bootstrap intervals across seeds are descriptive; paired-t intervals are sensitivity checks. '
        'Conditional channel/BER intervals hold the fitted ensemble fixed and do not include training uncertainty. '
        'Five-seed supplementary studies remain exploratory; more fits do not replace replication.', '',
        '## Main 30 dB comparisons','']
    for n in c['arrays']:
        v=summary[(summary.family=='iid')&(summary.n==n)&(summary['case']=='main_snr30')]
        text.extend([f'### {n}x{n}','', '| Method | Mean NMSE | Seed SD |','|---|---:|---:|'])
        for row in v.itertuples(): text.append(f'| {row.method} | {row.nmse_mean:.7g} | {row.nmse_sd:.3g} |')
        text.append('')
    text.extend(['## Limits','',
        '- Wider projection, recurrent width, and output head are distinct interventions. Spectra diagnose association, not causation.',
        '- Exact-Gaussian test cases are separate from finite-16-sinusoid Jakes cases. LMMSE is only guaranteed MMSE for the Gaussian known-statistics case.',
        '- Spatial controls assume half-wavelength ULAs and a uniform angular power spectrum; they are synthetic, not measured channels.',
        '- Empirical LMMSE uses subsets of the same clean labeled training windows available to the neural estimator. It sees no confirmation targets.',
        '- The unknown-statistics suite withholds per-window covariance parameters from neural and empirical methods. Component oracles have extra knowledge and are labeled.',
        '- Calibration-based SNR uses 64 extra noise-only complex samples per window; no free-SNR-acquisition or end-to-end throughput claim is supported.',
        '- QPSK blocks hold each channel fixed. Rate is a ZF Gaussian-input proxy without pilot/training/calibration overhead.',
        '- V3.2 Mamba/GRU/sequence results are preserved in the old study, not silently retrained or replaced by this focused LSTM revision.',
        '- Report runtime from latency.csv only when measured with the training workers stopped. Include covariance setup and fitting costs separately.',
        '- No result establishes publication acceptance. Preserve adverse and null results.'])
    (paper/'RESULTS_SUMMARY.md').write_text('\n'.join(text)+'\n')
    plots(root,metrics,summary,pd.DataFrame(training),pd.DataFrame(singular),c)


def report_error_comparisons(paper,metrics,lookup,c):
    out=[]
    for n,case,method in metrics[metrics.category.str.startswith(('report_','estimated_snr'))][['n','case','method']].drop_duplicates().itertuples(index=False,name=None):
        if case.startswith('report_reference'): continue
        rows=metrics[(metrics.family=='iid')&(metrics.n==n)&(metrics['case']==case)&(metrics.method==method)]
        snr=float(rows.iloc[0].snr);reference=f'report_reference_snr{snr:g}'
        refs=[reference]
        if case.endswith('clip1'): refs.append(case[:-1]+'0')
        for ref in refs:
            other=metrics[(metrics.family=='iid')&(metrics.n==n)&(metrics['case']==ref)&(metrics.method==method)]
            aa=rows.set_index('seed');bb=other.set_index('seed');ss=sorted(set(aa.index)&set(bb.index))
            if not ss: continue
            d=(aa.loc[ss,'nmse']-bb.loc[ss,'nmse']).values
            lo,hi=percentile_interval(d,np.random.default_rng(seed('report_error',n,case,ref,method)),c['bootstrap_resamples'])
            result=dict(n=n,snr=snr,method=method,case=case,reference_case=ref,seeds=len(ss),
                delta_nmse_mean=float(d.mean()),relative_increase_percent=float(d.mean()/bb.loc[ss,'nmse'].mean()*100),
                seed_bootstrap_low=lo,seed_bootstrap_high=hi,exploratory=True)
            # Use the *unclipped* report mask to isolate identical in/out-range windows.
            for label in ('in_range','out_of_range'):
                effects=[];count=0
                for s in ss:
                    with np.load(lookup[('iid',n,s,case,method)]) as ar,np.load(lookup[('iid',n,s,ref,method)]) as br:
                        if not np.array_equal(ar['power'],br['power']): raise RuntimeError('Reporting cases are unpaired')
                        mask=ar['outside_training_range']==int(label=='out_of_range');count=int(mask.sum())
                        if count: effects.append(float((ar['errors'][mask]-br['errors'][mask]).sum()/ar['power'][mask].sum()))
                result[label+'_count']=count
                result[label+'_paired_delta']=float(np.mean(effects)) if effects else None
            out.append(result)
    pd.DataFrame(out).to_csv(paper/'reporting_errors_paired.csv',index=False)


def plots(root,metrics,summary,training,singular,c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=Path(root)/'Figures';figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.labelsize':12,'legend.fontsize':8,'savefig.dpi':200})
    for n in sorted(metrics.n.unique()):
        q=max(16,2*n*n)
        methods=['LS','Oracle_component_trueSNR',f'Empirical_LMMSE_K{c["train_samples"]}',
                 f'cnn_p32_h32_head{q}_none',f'cnn_p32_h32_head{q}_gate',
                 f'raw_p0_h32_head{q}_none',f'raw_p0_h32_head{q}_gate']
        f,ax=plt.subplots(figsize=(8,5))
        for method in methods:
            d=summary[(summary.family=='iid')&(summary.n==n)&(summary.category=='main')&(summary.method==method)].sort_values('snr')
            if not d.empty: ax.semilogy(d.snr,d.nmse_mean,'o-',label=method)
        ax.set(xlabel='Nominal pilot SNR (dB)',ylabel='NMSE',title=f'{n} x {n}: fresh confirmation set')
        ax.grid(True,which='both',alpha=.2);ax.legend();f.tight_layout();f.savefig(figs/f'nmse_{n}x{n}.pdf');plt.close(f)
    d=metrics[(metrics.family=='iid')&(metrics.n==4)&(metrics['case']=='main_snr30')]
    names=[x for x in d.method.unique() if x.startswith('cnn_') and x.endswith(('_none','_gate'))]
    if names:
        f,ax=plt.subplots(figsize=(11,6))
        for i,name in enumerate(names):
            v=d[d.method==name].nmse.values;ax.scatter(np.full(len(v),i),v,s=18)
            ax.plot([i-.25,i+.25],[v.mean(),v.mean()],color='black',lw=2)
        ax.set_xticks(range(len(names)),names,rotation=60,ha='right');ax.set_yscale('log')
        ax.set(ylabel='30 dB NMSE',title='4 x 4 architecture controls: every independent training run')
        f.tight_layout();f.savefig(figs/'4x4_every_seed.pdf');plt.close(f)
    if not singular.empty:
        f,axes=plt.subplots(1,2,figsize=(11,4))
        names=[f'cnn_p{p}_h{h}_head32_{cond}' for p,h in [(32,32),(128,128)] for cond in ('none','gate')]
        for ax,part in zip(axes,['projection','head_out']):
            for name in names:
                d=singular[(singular.family=='iid')&(singular.n==4)&(singular.model==name)&
                    (singular.kind=='weight_spectra')&(singular.part==part)]
                if len(d):
                    v=d.groupby('index').relative_value.mean()
                    ax.semilogy(v.index,np.maximum(v.values,1e-12),label=name)
            ax.set(xlabel='Singular-value index',ylabel='Mean normalized singular value',title=part)
            ax.grid(alpha=.2);ax.legend()
        f.tight_layout();f.savefig(figs/'4x4_weight_spectra.pdf');plt.close(f)
