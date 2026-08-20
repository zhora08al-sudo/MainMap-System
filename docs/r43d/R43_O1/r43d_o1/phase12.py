"""
R43 O1 — Ф12: атака двух «фундаментальных» пределов (F1 и F2) по требованию автора.

F1: найти/создать режим, где derivative-free ДИСКРЕТНЫЙ шаг бьёт ОДУ-решатель.
    Кандидат — система с ЧАСТЫМИ разрывами (телеграф-форсинг, много переключений):
    адаптивный solve_ivp вынужден локализовать каждый разрыв → стоимость растёт с
    числом событий; фиксированный шаг просто идёт. Сравнение точность-vs-стоимость.

F2: чисто-топологические сообщества (без ритма). Идея автора «сначала корреляция,
    потом когеренция» применяется ПРАВИЛЬНО: убрать общий синхронный режим →
    корреляция/когеренция ОСТАТКОВ (co-fluctuation сообществ) → топ-k → кластеризация.

Зависимости: numpy, scipy, scikit-learn.
"""
from __future__ import annotations
import warnings, time; warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from harden import plv


# ====================================================================
# F1 — система с ЧАСТЫМИ разрывами: дискретный шаг vs solve_ivp
# ====================================================================
def f1_experiment():
    print("="*68); print("F1 · частые разрывы: derivative-free шаг vs solve_ivp"); print("="*68)
    rng = np.random.default_rng(0)
    N = 30
    omega = rng.normal(0, 1.0, N)
    A = (rng.random((N, N)) < 0.2).astype(float); A = np.triu(A, 1); A = A + A.T
    deg = A.sum(1); deg[deg == 0] = 1
    T = 20.0
    nsw = 400                                       # МНОГО переключений форсинга
    sw_t = np.sort(rng.uniform(0, T, nsw))
    sw_val = rng.choice([-1.0, 1.0], nsw+1) * 1.5   # телеграф-амплитуда
    def Kf(t):
        return sw_val[np.searchsorted(sw_t, t)]
    phi0 = rng.uniform(-np.pi, np.pi, N)
    def rhs(t, phi):
        s, c = np.sin(phi), np.cos(phi)
        return omega + Kf(t) * (c*(A@s) - s*(A@c)) / deg

    # эталон: разбиение на КАЖДОМ разрыве, высокоточно
    grid = np.linspace(0, T, 401)
    def reference():
        phi = phi0.copy(); ts = np.r_[0.0, sw_t, T]; segs = []
        cur = phi
        outs = []
        bounds = np.r_[0.0, sw_t, T]
        sol_t = []; sol_y = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a: continue
            s = solve_ivp(rhs, (a, b), cur, rtol=1e-10, atol=1e-12, dense_output=True)
            cur = s.y[:, -1]; sol_t.append((a, b, s))
        def evalg(t):
            for a, b, s in sol_t:
                if a <= t <= b: return s.sol(t)
            return sol_t[-1][2].sol(t)
        Y = np.array([evalg(t) for t in grid])
        nfev = sum(s.nfev for _,_,s in sol_t)
        return np.abs(np.exp(1j*Y).mean(1)), nfev
    r_ref, nfev_ref = reference()

    def order(Y): return np.abs(np.exp(1j*Y).mean(1))
    def err(Y): return np.abs(order(Y) - r_ref).max()

    # solve_ivp наивный (без объявления разрывов), разные tol
    rows = []
    for tol in (1e-3, 1e-6):
        s = solve_ivp(rhs, (0, T), phi0, rtol=tol, atol=tol*1e-2, dense_output=True)
        Y = np.array([s.sol(t) for t in grid])
        rows.append((f"solve_ivp tol={tol:.0e}", err(Y), s.nfev))
    # фиксированный derivative-free шаг (R43 O1), разные dt
    for dt in (0.02, 0.01, 0.005):
        steps = int(T/dt); phi = phi0.copy(); t = 0.0
        rec = [phi.copy()]; tt = [0.0]
        for k in range(steps):
            s, c = np.sin(phi), np.cos(phi)
            phi = phi + dt*(omega + Kf(t)*(c*(A@s) - s*(A@c))/deg); t += dt
            rec.append(phi.copy()); tt.append(t)
        Z = np.exp(1j*np.array(rec))
        zr = np.empty((len(grid), N), complex)
        for j in range(N):
            zr[:, j] = np.interp(grid, tt, Z[:, j].real) + 1j*np.interp(grid, tt, Z[:, j].imag)
        Y_order = np.abs(zr.mean(1))
        rows.append((f"R43O1 fixed dt={dt}", np.abs(Y_order - r_ref).max(), steps))

    print(f"  эталон (разбиение на {nsw} разрывах): nfev={nfev_ref}")
    print(f"  {'метод':24} {'max|Δr|':>10} {'стоимость':>10}")
    for n, e, c in rows:
        print(f"  {n:24} {e:>10.4f} {c:>10d}")
    # честный вердикт: есть ли точка R43O1 дешевле solve_ivp при не худшей точности
    ode = [r for r in rows if r[0].startswith("solve_ivp")]
    r43 = [r for r in rows if r[0].startswith("R43")]
    win = any(rr[1] <= oo[1] and rr[2] < oo[2] for rr in r43 for oo in ode)
    best_ode_cost = min(o[2] for o in ode)
    print(f"  -> F1: {'ЕСТЬ выигрыш по стоимости при не худшей точности' if win else 'выигрыша нет'}"
          f" (мин. стоимость solve_ivp={best_ode_cost})")
    return win


# ====================================================================
# F2 — топология без ритма: остатки после удаления общей моды
# ====================================================================
def gen_topo(N=150, M=3, K0=1.6, sw=0.6, seed=0, steps=1500, dt=0.05, tail=600,
             intra=0.85, inter=0.02, obs_noise=0.2):
    rng = np.random.default_rng(seed)
    comm = np.repeat(np.arange(M), N // M)
    same = comm[:, None] == comm[None, :]
    pr = np.where(same, intra, inter); np.fill_diagonal(pr, 0)
    iu = np.triu_indices(N, 1); e = rng.random(len(iu[0])) < pr[iu]
    A = np.zeros((N, N)); A[iu[0][e], iu[1][e]] = 1; A = A + A.T
    deg = A.sum(1); deg[deg == 0] = 1
    omega = sw*rng.standard_normal(N)               # один ритм для всех
    phi = rng.uniform(-np.pi, np.pi, N)
    Z = np.empty((tail, N), complex); t0 = steps - tail
    for k in range(steps):
        s, c = np.sin(phi), np.cos(phi)
        phi = phi + dt*(omega + K0*(c*(A@s) - s*(A@c))/deg)
        if k >= t0: Z[k-t0] = np.exp(1j*phi)
    pho = np.angle(Z) + obs_noise*rng.standard_normal((tail, N))
    return pho, comm


def knn_sym(W, k):
    N = W.shape[0]; Awk = W.copy(); np.fill_diagonal(Awk, -np.inf)
    idx = np.argsort(-Awk, axis=1)[:, :k]
    M = np.zeros_like(W, bool); rows = np.repeat(np.arange(N), k)
    M[rows, idx.ravel()] = True; M = M | M.T
    out = np.where(M, np.clip(W, 0, None), 0.0); np.fill_diagonal(out, 1.0); return out

def spec_pre(aff, K):
    return SpectralClustering(n_clusters=K, affinity="precomputed",
                             assign_labels="kmeans", random_state=0).fit_predict(aff)

def aff_residual(phi):
    """Убрать общую моду → когеренция ОСТАТКОВ (co-fluctuation сообществ)."""
    Z = np.exp(1j*phi); g = Z.mean(1, keepdims=True)
    a = (np.conj(g)*Z).sum(0, keepdims=True)/((np.abs(g)**2).sum()+1e-12)
    R = Z - g*a; Rn = R/(np.linalg.norm(R, axis=0, keepdims=True)+1e-12)
    return np.abs(Rn.conj().T @ Rn)

def f2_experiment():
    print("\n"+"="*68)
    print("F2 · топология без ритма: остатки после удаления общей моды (идея corr→coh)")
    print("="*68)
    print(f"  {'K0':>5} {'PLV+kNN':>10} {'остатки+kNN':>13}")
    best = 0.0
    for K0 in (0.6, 1.0, 1.6, 2.5, 4.0):
        phi, comm = gen_topo(K0=K0, seed=1)
        a_plv = adjusted_rand_score(comm, spec_pre(knn_sym(plv(phi), 12), 3))
        a_res = adjusted_rand_score(comm, spec_pre(knn_sym(aff_residual(phi), 12), 3))
        best = max(best, a_res)
        print(f"  {K0:>5} {a_plv:>10.3f} {a_res:>13.3f}")
    ok = best >= 0.6
    print(f"  -> F2: лучший ARI остатками = {best:.3f}  "
          f"{'ЕСТЬ прогресс' if ok else 'по-прежнему вне области'}")
    return ok


if __name__ == "__main__":
    w1 = f1_experiment()
    w2 = f2_experiment()
    print(f"\n[ИТОГ Ф12] F1 (разрывы): {'выигрыш' if w1 else 'нет выигрыша'} | "
          f"F2 (топология): {'прогресс' if w2 else 'вне области'}")
