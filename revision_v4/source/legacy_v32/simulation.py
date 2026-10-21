"""Channel simulation, matched linear estimators, and independent-block metrics.

SNR denotes nominal expected-power pilot SNR, constant within each window.
The data link uses y=sqrt(rho/Nt)Hx+n with unit-energy QPSK symbols.
No function here uses a neural prediction to synthesize a target or observation.
"""
import numpy as np
from scipy.special import j0
from scipy.optimize import brentq


def fd_from_alpha(alpha):
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0,1]; inversion uses the first J0 lobe")
    return float(brentq(lambda x: j0(2*np.pi*x)-alpha, 0, 2.4048255577/(2*np.pi)))


def spatial_cov(n, rho):
    if not 0 <= rho < 1:
        raise ValueError("spatial correlation must be in [0,1)")
    return rho ** np.abs(np.arange(n)[:, None]-np.arange(n)[None, :])


def temporal_cov(T, alpha):
    return j0(2*np.pi*fd_from_alpha(alpha)*np.abs(np.arange(T)[:, None]-np.arange(T)))


def jakes(count, n, T, alpha, rng, rho=0.0, M=16):
    """Unit-complex-power Zheng-Xiao SoS; independent phases/angle per tap.

    a_m=(2*pi*m-pi+theta)/(4M). Each quadrature has variance 1/2.
    M finite approximates Gaussian Rayleigh fading; second moments follow J0.
    """
    taps = count*n*n
    out = np.empty((taps, T), np.complex64)
    m = np.arange(1, M+1)[None, :]
    t = np.arange(T)[None, None, :]
    omega = 2*np.pi*fd_from_alpha(alpha)
    chunk = max(1, min(taps, 4_000_000//max(1, M*T)))
    for lo in range(0, taps, chunk):
        hi = min(taps, lo+chunk)
        theta = rng.uniform(-np.pi, np.pi, (hi-lo, 1))
        pc = rng.uniform(-np.pi, np.pi, (hi-lo, M))
        ps = rng.uniform(-np.pi, np.pi, (hi-lo, M))
        a = (2*np.pi*m-np.pi+theta)/(4*M)
        re = np.cos(omega*t*np.cos(a)[:, :, None]+pc[:, :, None]).sum(1)/np.sqrt(M)
        im = np.cos(omega*t*np.sin(a)[:, :, None]+ps[:, :, None]).sum(1)/np.sqrt(M)
        out[lo:hi] = re+1j*im
    h = out.reshape(count, n, n, T).transpose(0, 3, 1, 2).copy()
    if rho:
        d, v = np.linalg.eigh(spatial_cov(n, rho))
        root = (v*np.sqrt(np.maximum(d, 0)))@v.T
        h = np.einsum('ij,btjk,kl->btil', root, h, root).astype(np.complex64)
    return h


def dft(n):
    return (np.exp(-2j*np.pi*np.outer(np.arange(n), np.arange(n))/n)/np.sqrt(n)).astype(np.complex64)


def observe(h, snr_db, rng):
    snr = np.broadcast_to(np.asarray(snr_db), (len(h),))
    scale = np.sqrt(10.0**(-snr/10)/2).reshape((-1,)+(1,)*(h.ndim-1))
    noise = (rng.standard_normal(h.shape)+1j*rng.standard_normal(h.shape))*scale
    x = dft(h.shape[-1])
    return ((h@x+noise)@x.conj().T).astype(np.complex64)


def real_input(y):
    return np.ascontiguousarray(np.stack([y.real, y.imag], axis=2), dtype=np.float32)


def real_target(h):
    return np.ascontiguousarray(np.stack([h.real, h.imag], axis=1), dtype=np.float32)


def training_data(count, n, T, alpha, seed, M=16, snr_bounds=(0,30)):
    rng = np.random.default_rng(seed)
    h = jakes(count, n, T, alpha, rng, M=M)
    snr = rng.uniform(*snr_bounds, count).astype(np.float32)
    y = observe(h, snr, rng)
    return real_input(y), ((snr-15)/15).reshape(-1, 1), real_target(h[:, -1])


def test_channels(count, n, T, alpha, seed, rho=0., M=16):
    return jakes(count, n, T, alpha, np.random.default_rng(seed), rho=rho, M=M)


def ar1_filter(y, snr_db, assumed_alpha=0.8):
    q = 1-assumed_alpha**2
    r = 10.**(-snr_db/10)
    estimate = np.zeros_like(y[:, 0])
    P = 1.
    for i in range(y.shape[1]):
        if i:
            estimate = assumed_alpha*estimate
            P = assumed_alpha**2*P+q
        K = P/(P+r)
        estimate = estimate+K*(y[:, i]-estimate)
        P = (1-K)*P
    return estimate


def temporal_lmmse(y, snr_db, assumed_alpha=0.8):
    R = temporal_cov(y.shape[1], assumed_alpha)
    w = np.linalg.solve(R+10.**(-snr_db/10)*np.eye(len(R)), R[:, -1])
    return np.einsum('t,btij->bij', w, y).astype(np.complex64)


def temporal_lmmse_theory(T, snr_db, alpha=.8):
    R = temporal_cov(T, alpha)
    return float(1-R[-1]@np.linalg.solve(R+10.**(-snr_db/10)*np.eye(T), R[:, -1]))


def joint_lmmse(y, snr_db, alpha, rho):
    """Oracle separable temporal/spatial LMMSE. No covariance estimated from test targets."""
    n = y.shape[-1]
    # Row-major vec(H): covariance R_rx kron R_tx.T.
    C = np.kron(spatial_cov(n, rho), spatial_cov(n, rho).T)
    d, v = np.linalg.eigh(C)
    transformed = y.reshape(len(y), y.shape[1], -1)@v
    R = temporal_cov(y.shape[1], alpha)
    out = np.empty((len(y), n*n), np.complex128)
    for i, lam in enumerate(d):
        w = np.linalg.solve(lam*R+10.**(-snr_db/10)*np.eye(len(R)), lam*R[:, -1])
        out[:, i] = transformed[:, :, i]@w
    return (out@v.T).reshape(-1, n, n).astype(np.complex64)


def baselines(y, snr_db, alpha=.8, rho=0.):
    return {
        'LS': y[:, -1],
        'LMMSE_instant_iid': y[:, -1]/(1+10.**(-snr_db/10)),
        'Kalman_AR1_fixed': ar1_filter(y, snr_db, .8),
        'Jakes_LMMSE_fixed': temporal_lmmse(y, snr_db, .8),
        'Jakes_LMMSE_oracle': joint_lmmse(y, snr_db, alpha, rho),
    }


def nmse(h, estimate):
    e = np.sum(np.abs(h-estimate).astype(np.float64)**2, axis=(1, 2))
    p = np.sum(np.abs(h).astype(np.float64)**2, axis=(1, 2))
    ratio = float(e.sum()/p.sum())
    # Delta method applied to independent channel sequences, NOT individual taps.
    se = float(np.std(e-ratio*p, ddof=1)/np.sqrt(len(e))/p.mean()) if len(e)>1 else 0.
    return {'nmse': ratio, 'nmse_ci95_halfwidth': 1.96*se}, e, p


def zf_rate_blocks(h, estimate, snr_db):
    n = h.shape[-1]
    he = estimate.astype(np.complex128)
    W = np.linalg.solve(he.conj().transpose(0, 2, 1)@he+1e-8*np.eye(n), he.conj().transpose(0, 2, 1))
    G = W@h
    signal = (10.**(snr_db/10)/n)*np.abs(np.diagonal(G, axis1=1, axis2=2))**2
    total = (10.**(snr_db/10)/n)*np.sum(np.abs(G)**2, axis=2)
    noise = np.sum(np.abs(W)**2, axis=2)
    return np.log2(1+signal/np.maximum(total-signal+noise, 1e-15)).sum(1)


def ber_blocks(h, estimate, snr_db, bits_per_block, seed):
    """Return error counts by independent channel block. Same seed pairs methods.

    The channel is held fixed inside a diagnostic QPSK block; this is not a
    Doppler-evolving payload transmission or a pilot-overhead-adjusted rate.
    """
    n = h.shape[-1]
    if bits_per_block % (2*n):
        raise ValueError('bits_per_block must be divisible by 2*Nt')
    rng = np.random.default_rng(seed)
    result = []
    for lo in range(0, len(h), 32):
        hc = h[lo:lo+32]
        ec = estimate[lo:lo+32].astype(np.complex128)
        bits = rng.integers(0, 2, (len(hc), bits_per_block), dtype=np.int8)
        symbols = ((2*bits[:, 0::2]-1)+1j*(2*bits[:, 1::2]-1))/np.sqrt(2)
        X = symbols.reshape(len(hc), -1, n).transpose(0, 2, 1)
        Y = np.sqrt(10.**(snr_db/10)/n)*(hc@X)
        Y += (rng.standard_normal(Y.shape)+1j*rng.standard_normal(Y.shape))/np.sqrt(2)
        eh = ec.conj().transpose(0, 2, 1)
        Xe = np.linalg.solve(eh@ec+1e-8*np.eye(n), eh@Y).transpose(0, 2, 1).reshape(len(hc), -1)
        bh = np.empty_like(bits)
        bh[:, 0::2] = Xe.real > 0
        bh[:, 1::2] = Xe.imag > 0
        result.extend(np.sum(bh != bits, axis=1).tolist())
    return np.array(result, dtype=np.int64)


def bootstrap_mean(values, seed=887, resamples=2000):
    """Percentile interval across independent blocks. Use same seed to pair methods."""
    values = np.asarray(values, np.float64)
    if not len(values):
        raise ValueError('empty blocks')
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for lo in range(0, resamples, 100):
        count = min(100, resamples-lo)
        means[lo:lo+count] = values[rng.integers(0, len(values), (count, len(values)))].mean(1)
    low, high = np.quantile(means, [.025, .975])
    return {'mean': float(values.mean()), 'ci95_low': float(low), 'ci95_high': float(high)}


def validate_channel(seed=876, count=20000, T=32):
    h = jakes(count, 1, T, .8, np.random.default_rng(seed))[:, :, 0, 0]
    power = float(np.mean(np.abs(h)**2))
    empirical = [float(np.mean(h[:, k:]*h[:, :T-k].conj()).real/power) for k in range(T)]
    target = temporal_cov(T, .8)[0]
    error = float(np.max(np.abs(target-empirical)))
    passed = abs(power-1)<.05 and error<.05
    return {'passed': passed, 'power': power, 'max_acf_error': error,
            'target': target.tolist(), 'empirical': empirical, 'count': count, 'T': T,
            'fd_ts': fd_from_alpha(.8), 'M': 16}
