"""Fixed, test-independent experiment design. Edits create a different run ID."""
from copy import deepcopy


def config(smoke=False):
    c = dict(schema='phycom-review-v4.0', mode='publication', T=5, M=16,
        seeds=[31001,31013,31019,31033,31039,31051,31063,31069,31079,31081],
        supplemental_seed_count=5, arrays=[2,3,4], alpha=.8,
        train_samples=100000, val_samples=20000, test_samples=20000,
        stress_samples=5000, train_snr=[0.,30.], snrs=[0,5,10,15,20,25,30],
        batch_size=256, eval_batch_size=2048, epochs=240, patience=10,
        lr=.001, lr_patience=5, lr_factor=.5, gradient_clip=None,
        precision='float32', deterministic=True, torch_threads=2,
        confirmation_namespace='phycom-R1-fresh-confirmation-20261019-v4',
        train_namespace='phycom-R1-fresh-training-20261019-v4',
        cov_samples=[256,2048,100000], covariance_shrinkage=0.,
        spatial_mean_deg=20., spatial_spread_deg=15., spacing_wavelengths=.5,
        mixture_alphas=[.3,.6,.9], mixture_spreads=[10.,30.,60.],
        snr_biases=[-3.,-1.,0.,1.,3.], snr_error_stds=[1.,3.],
        stress_true_snrs=[0.,15.,30.], noise_calibration_samples=64,
        diagnostics_samples=2048, bootstrap_resamples=2000,
        ber_blocks=1024, bits_per_stream=512, rate_blocks=2048,
        timing_batches=[1,32,256], timing_trials=100, timing_warmup=20,
        convergence_window=20, convergence_recent_gain_threshold=.01,
        primary_snrs=[15,30], primary_arrays=[2,3,4])
    if smoke:
        c.update(mode='smoke', seeds=[31001,31013], supplemental_seed_count=2,
            train_samples=96, val_samples=48, test_samples=96, stress_samples=64,
            batch_size=32, eval_batch_size=64, epochs=2, patience=2,
            snrs=[0,15,30], cov_samples=[32,96], diagnostics_samples=48,
            bootstrap_resamples=100, ber_blocks=16, bits_per_stream=16,
            rate_blocks=32, timing_trials=3, timing_warmup=2,
            timing_batches=[1,8], convergence_window=1)
    return c


def spec(encoder='cnn', p=32, h=32, head=0, cond='none', residual=False):
    return dict(encoder=encoder, projection=p, hidden=h, head=head,
                condition=cond, residual=residual)


def spec_id(s):
    return (f"{s['encoder']}_p{s['projection']}_h{s['hidden']}_head{s['head']}_"
            f"{s['condition']}" + ('_lsres' if s['residual'] else ''))


def resolved(s, n):
    s = deepcopy(s)
    if not s['head']:
        s['head'] = max(16, 2*n*n)
    return s


def param_count_formula(s, n):
    d = 2*n*n
    h, p, q = s['hidden'], s['projection'], s['head'] or max(16,d)
    e = 0 if s['encoder']=='raw' else (16*2*9+16+32+8*16*9+8+16)+(8*n*n*p+p)
    inp = d if s['encoder']=='raw' else p
    core = 4*h*(inp+h+2)
    c = s['condition']
    cond = 2*h if c in ('gate','gate_identity','resgate','resgate_identity') else 4*h if c in ('film','film_identity') else 0
    head = (h+int(c.startswith('concat')))*q+q+q*d+d
    return int(e+core+cond+head)


def matched_raw_width(n, p=32, h=32):
    target = param_count_formula(spec(p=p,h=h),n)
    return min(range(8,257), key=lambda x: abs(param_count_formula(spec('raw',0,x),n)-target))


def pairs(**kw):
    return [spec(cond=c,**kw) for c in ('none','gate')]


def models(c, group):
    n = group['n']
    if group['family'] != 'iid':
        # Physical-correlation/practical suite: same total-parameter target, both conditions.
        r = pairs(p=128,h=128) + pairs(encoder='raw',p=0,h=matched_raw_width(n,128,128))
    else:
        r = pairs() + pairs(encoder='raw',p=0)
        if n in (2,4):
            r += pairs(residual=True)
        if n==4:
            # Separate projection width, recurrent width, and their interaction.
            for p,h in [(64,32),(128,32),(32,64),(32,128),(64,64),(128,128)]:
                r += pairs(p=p,h=h)
            r += pairs(head=64)
            r += pairs(encoder='raw',p=0,h=matched_raw_width(n))
        if n in (2,4) and group['seed'] in c['seeds'][:c['supplemental_seed_count']]:
            r += [spec(cond=x) for x in ('gate_identity','resgate_identity','film_identity',
                'concat_identity','film','resgate','concat')]
    return [resolved(s,n) for s in r]


def groups(c):
    # Central comparisons first, then targeted physical/practical controls.
    r = [dict(family='iid',n=n,seed=s) for s in c['seeds'] for n in c['arrays']]
    r += [dict(family=f,n=4,seed=s) for s in c['seeds'][:c['supplemental_seed_count']]
          for f in ('spatial','unknown_stats')]
    return r


def plan(c):
    return [dict(**g, models=models(c,g)) for g in groups(c)]


def comparisons(c, n, family):
    """Pairs are fixed before results; tuple is candidate minus reference."""
    g = dict(family=family,n=n,seed=c['seeds'][0])
    out = []
    available = models(c,g)
    for s in available:
        if s['condition']=='gate':
            ref = dict(s,condition='none')
            out.append((spec_id(s),spec_id(ref),'gating'))
        if s['condition'].endswith('_identity') or s['condition'] in ('film','resgate','concat'):
            out.append((spec_id(s),spec_id(dict(s,condition='none')),'initialization'))
        if s['residual']:
            out.append((spec_id(s),'LS','residual_vs_LS'))
            out.append((spec_id(s)+'__val_scaled','LS','scaled_residual_vs_LS'))
            out.append((spec_id(s)+'__val_scaled',spec_id(s),'validation_scaling'))
    if family=='iid':
        base = resolved(spec(),n)
        for cond in ('none','gate'):
            out.append((spec_id(resolved(spec('raw',0,cond=cond),n)),
                        spec_id(dict(base,condition=cond)),'raw_vs_cnn'))
            if n==4:
                for p,h in [(64,32),(128,32),(32,64),(32,128),(64,64),(128,128)]:
                    out.append((spec_id(resolved(spec(p=p,h=h,cond=cond),n)),
                                spec_id(dict(base,condition=cond)),'width_vs_original'))
                out.append((spec_id(resolved(spec(head=64,cond=cond),n)),
                            spec_id(dict(base,condition=cond)),'output_head'))
                out.append((spec_id(resolved(spec('raw',0,matched_raw_width(n),cond=cond),n)),
                            spec_id(dict(base,condition=cond)),'parameter_matched_encoder'))
    else:
        for cond in ('none','gate'):
            out.append((spec_id(resolved(spec('raw',0,matched_raw_width(n,128,128),cond=cond),n)),
                        spec_id(resolved(spec(p=128,h=128,cond=cond),n)),'parameter_matched_encoder'))
    return out
