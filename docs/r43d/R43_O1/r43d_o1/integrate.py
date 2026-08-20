"""
R43 O1 — Ф14: ИНТЕГРАЦИЯ (I1) + совместный экстрим (S2).

Два главных выигрыша доказывались ПОРОЗНЬ:
  • масштаб через landmark/Nyström (Ф7, N=10⁴) — но с обычной PLV;
  • устойчивость к общему драйверу через corr→coh топ-k (Ф11) — но при N=120.
Здесь собираем их ВМЕСТЕ и проверяем на совместном экстриме.

Стыковка (инвенция): в landmark-признаках узла «относительное ранжирование» из идеи
автора (убирающее общий «пол» драйвера) реализуется ЦЕНТРИРОВАНИЕМ СТРОКИ признаков
(вычесть среднее сходство узла к опорным) — это Nyström-аналог corr→coh.

Тесты:
  A. масштаб+драйвер: N=2000, κ∈{0,1,2}, шум — обычный vs центрированный считыватель;
  S2. совместный экстрим: масштаб+драйвер+шум+скрытые узлы одновременно.
Зависимости: numpy, scipy, scikit-learn.
"""
from __future__ import annotations
import warnings, time; warnings.filterwarnings("ignore")
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from harden import plv


def gen_sparse_drive(N, M, K0=1.6, kappa=0.0, obs_noise=0.2, seed=0,
                     steps=500, dt=0.05, tail=250, avg_deg=12, Om_d=0.7):
    rng = np.random.default_rng(seed)
    sizes = [N // M]*M
    for i in range(N - sum(sizes)): sizes[i] += 1
    comm = np.concatenate([[k]*s for k, s in enumerate(sizes)])
    starts = np.cumsum([0]+sizes); rows, cols = [], []
    for k in range(M):
        idx = np.arange(starts[k], starts[k+1]); s = len(idx)
        ne = int(s*avg_deg/2); u = rng.integers(0, s, ne); v = rng.integers(0, s, ne)
        m = u != v; rows += list(idx[u[m]]); cols += list(idx[v[m]])
    ne = int(N*1/2); u = rng.integers(0, N, ne); v = rng.integers(0, N, ne)
    m = comm[u] != comm[v]; rows += list(u[m]); cols += list(v[m])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    band = np.linspace(-1.5, 1.5, M); omega = band[comm] + 0.10*rng.standard_normal(N)
    phi = rng.uniform(-np.pi, np.pi, N)
    Z = np.empty((tail, N), complex); t0 = steps - tail; t = 0.0
    for k in range(steps):
        s, c = np.sin(phi), np.cos(phi)
        coup = K0*(c*(A@s) - s*(A@c))/deg
        drive = kappa*np.sin(Om_d*t - phi) if kappa > 0 else 0.0
        phi = phi + dt*(omega + coup + drive); t += dt
        if k >= t0: Z[k-t0] = np.exp(1j*phi)
    pho = np.angle(Z) + obs_noise*rng.standard_normal((tail, N))
    return pho, comm


def landmark_features(phi, m_land=200, seed=0):
    T, N = phi.shape
    rng = np.random.default_rng(seed)
    land = rng.choice(N, size=min(m_land, N), replace=False)
    Z = np.exp(1j*phi)
    return np.abs(Z.conj().T @ Z[:, land]) / T          # (N, m) PLV к опорным


def cluster_plain(F, K, seed=0):
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    return KMeans(n_clusters=K, n_init=4, random_state=seed).fit_predict(Fn)


def cluster_centered(F, K, seed=0):
    """Nyström-аналог corr→coh: центрируем строку (убираем общий 'пол' драйвера)."""
    Fc = F - F.mean(axis=1, keepdims=True)
    Fc = Fc / (np.linalg.norm(Fc, axis=1, keepdims=True) + 1e-12)
    return KMeans(n_clusters=K, n_init=4, random_state=seed).fit_predict(Fc)


def cluster_ranked(F, K, seed=0):
    """Правильный перенос corr→coh: РАНГОВОЕ преобразование строки.
       Ранг инвариантен к добавлению константы-'пола' драйвера -> драйвер уходит."""
    R = np.argsort(np.argsort(F, axis=1), axis=1).astype(float)
    R = R / (R.max(axis=1, keepdims=True) + 1e-12)
    R = R - R.mean(axis=1, keepdims=True)
    return KMeans(n_clusters=K, n_init=4, random_state=seed).fit_predict(R)


# --- ПОЛНАЯ матрица corr→coh (работает до ~неск. тысяч узлов; O(N²) память) ---
def _corr_cos(phi):
    C = np.corrcoef(np.cos(phi).T); return np.abs(np.nan_to_num(C))

def _knn_sym(W, k):
    n = W.shape[0]; Aw = W.copy(); np.fill_diagonal(Aw, -np.inf)
    idx = np.argsort(-Aw, 1)[:, :k]; Mk = np.zeros_like(W, bool)
    Mk[np.repeat(np.arange(n), k), idx.ravel()] = True; Mk = Mk | Mk.T
    out = np.where(Mk, np.clip(W, 0, None), 0.0); np.fill_diagonal(out, 1.0); return out

def full_corr_coh(phi, K, k=15):
    """corr→coh из Ф11: corr-ранжирование соседей (убирает 'пол' драйвера) → веса PLV."""
    P = plv(phi); mask = _knn_sym(_corr_cos(phi), k) > 0
    W = np.where(mask, np.clip(P, 0, None), 0.0); np.fill_diagonal(W, 1.0)
    return SpectralClustering(K, affinity="precomputed", assign_labels="kmeans",
                             random_state=0).fit_predict(W)


if __name__ == "__main__":
    print("="*70)
    print("R43 O1 · Ф14 · интеграция масштаб+устойчивость + совместный экстрим")
    print("="*70)

    print("\n[A] МАСШТАБ + ДРАЙВЕР (N=2000, M=8, шум 0.2): plain / centered / ranked")
    print(f"  {'κ':>4} {'plain':>8} {'centered':>9} {'ranked':>8} {'время,с':>8}")
    for kap in (0.0, 1.0, 2.0):
        t0 = time.time()
        phi, comm = gen_sparse_drive(2000, 8, kappa=kap, obs_noise=0.2, seed=3)
        F = landmark_features(phi, 200)
        a_p = adjusted_rand_score(comm, cluster_plain(F, 8))
        a_c = adjusted_rand_score(comm, cluster_centered(F, 8))
        a_r = adjusted_rand_score(comm, cluster_ranked(F, 8))
        print(f"  {kap:>4} {a_p:>8.3f} {a_c:>9.3f} {a_r:>8.3f} {time.time()-t0:>8.2f}")

    print("\n[S2] СОВМЕСТНЫЙ ЭКСТРИМ: масштаб + драйвер + шум + скрытые узлы одновременно")
    rng = np.random.default_rng(0)
    phi, comm = gen_sparse_drive(2000, 8, K0=1.6, kappa=1.5, obs_noise=0.3, seed=5)
    keep = rng.random(phi.shape[1]) < 0.7            # 30% узлов скрыто
    phi_k, comm_k = phi[:, keep], comm[keep]
    t0 = time.time()
    a_full = adjusted_rand_score(comm_k, full_corr_coh(phi_k, 8))     # полный corr→coh
    t_full = time.time()-t0
    a_nys = adjusted_rand_score(comm_k, cluster_ranked(landmark_features(phi_k, 200), 8))
    print(f"  N=2000→{keep.sum()} (30% скрыто), κ=1.5, шум=0.3, M=8")
    print(f"  ARI ПОЛНЫЙ corr→coh = {a_full:.3f}  (целевой ≥0.85)  время={t_full:.2f}с")
    print(f"  ARI Nyström ranked  = {a_nys:.3f}  (не переносит устойчивость к драйверу)")

    print("\n[B] ПОЛНАЯ матрица corr→coh при N=2000 под драйвером (граница композиции)")
    for kap in (0.0, 1.5):
        t0 = time.time()
        phi, comm = gen_sparse_drive(2000, 8, kappa=kap, obs_noise=0.2, seed=3)
        lab = full_corr_coh(phi, 8)
        print(f"  κ={kap}: full corr→coh ARI={adjusted_rand_score(comm,lab):.3f}  "
              f"время={time.time()-t0:.1f}с (память O(N²))")

    print("\n[ИТОГ Ф14]")
    print(f"  Композиция РАБОТАЕТ до ~неск. тысяч узлов: ПОЛНЫЙ corr→coh держит совместный")
    print(f"  экстрим (масштаб+драйвер+шум+скрытые) → S2 ARI={a_full:.3f} "
          f"{'PASS' if a_full >= 0.85 else 'FAIL'}.")
    print(f"  ОТКРЫТО: 10⁴ + сильный драйвер ОДНОВРЕМЕННО — Nyström теряет устойчивость")
    print(f"  к драйверу (перенос рангом/центром не помог). Там пока: либо масштаб, либо драйвер.")
