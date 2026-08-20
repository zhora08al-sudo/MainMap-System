"""
R43 O1 — Ф11: идея автора «сначала корреляция, потом когеренция».

Инсайт: когеренция (PLV) = модуль корреляции комплексных сигналов e^{iφ}. Значит
корреляция — это «первая ступень», а когеренция — её фазовое уточнение. Практическое
следствие, которое НЕ пробовалось в Ф7: использовать ОТНОСИТЕЛЬНОЕ РАНЖИРОВАНИЕ
соседей (как в корреляции) — топ-k на узел — а потом уже когеренцию.

Зачем: общий драйвер (нерешённый предел Ф7) добавляет ко ВСЕМ парам ~одинаковый
«пол» когерентности. Глобальная дефляция его не убрала. Но топ-k по строке вычитает
этот пол ОТНОСИТЕЛЬНО (берём сильнейших соседей каждого узла) — структура может
проявиться. Проверяем честно на конфаунде драйвера и на чистой топологии.

Считыватели (все derivative-free):
  PLV+thr     полная PLV + спектральная (как в Ф3+; на драйвере падала)
  PLV+kNN     PLV, но оставляем топ-k соседей на узел, потом спектральная
  CORR+kNN    |corr(cos φ)| топ-k + спектральная  («сначала корреляция»)
  2-ступ.     ранжируем соседей корреляцией → на них считаем PLV → спектральная

Зависимости: numpy, scikit-learn.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from harden import gen, plv, spec


def corr_cos(phi):
    C = np.corrcoef(np.cos(phi).T)
    return np.abs(np.nan_to_num(C))


def knn_sym(W, k):
    """Оставить топ-k соседей на строку, симметризовать (ИЛИ)."""
    N = W.shape[0]; A = W.copy(); np.fill_diagonal(A, -np.inf)
    idx = np.argsort(-A, axis=1)[:, :k]
    M = np.zeros_like(W, dtype=bool)
    rows = np.repeat(np.arange(N), k)
    M[rows, idx.ravel()] = True
    M = M | M.T
    out = np.where(M, np.clip(W, 0, None), 0.0)
    np.fill_diagonal(out, 1.0)
    return out


def spec_pre(aff, K):
    return SpectralClustering(n_clusters=K, affinity="precomputed",
                             assign_labels="kmeans", random_state=0).fit_predict(aff)


def readers(phi, K, k=12):
    P = plv(phi); Cc = corr_cos(phi)
    res = {}
    res["PLV+thr"] = spec(P, K)                                  # как в Ф3+
    res["PLV+kNN"] = spec_pre(knn_sym(P, k), K)                  # ранжирование на PLV
    res["CORR+kNN"] = spec_pre(knn_sym(Cc, k), K)               # «сначала корреляция»
    # 2-ступенчатый: маска соседей по корреляции, веса — по PLV
    Cmask = knn_sym(Cc, k) > 0
    W2 = np.where(Cmask, np.clip(P, 0, None), 0.0); np.fill_diagonal(W2, 1.0)
    res["2-ступ.(corr→coh)"] = spec_pre(W2, K)
    return res


def gen_norhythm(N=120, M=3, K0=1.6, sw=0.6, seed=0, steps=1500, dt=0.05, tail=500,
                 intra=0.85, inter=0.02, obs_noise=0.2):
    """Сообщества только по топологии (без различия ритма) — провальный случай Ф10."""
    rng = np.random.default_rng(seed)
    comm = np.repeat(np.arange(M), N // M)
    same = comm[:, None] == comm[None, :]
    pr = np.where(same, intra, inter); np.fill_diagonal(pr, 0)
    iu = np.triu_indices(N, 1); e = rng.random(len(iu[0])) < pr[iu]
    A = np.zeros((N, N)); A[iu[0][e], iu[1][e]] = 1; A = A + A.T
    deg = A.sum(1); deg[deg == 0] = 1
    omega = sw * rng.standard_normal(N)
    phi = rng.uniform(-np.pi, np.pi, N)
    Z = np.empty((tail, N), complex); t0 = steps - tail
    for kk in range(steps):
        s, c = np.sin(phi), np.cos(phi)
        phi = phi + dt*(omega + K0*(c*(A@s) - s*(A@c))/deg)
        if kk >= t0: Z[kk-t0] = np.exp(1j*phi)
    pho = np.angle(Z) + obs_noise*rng.standard_normal((tail, N))
    return pho, comm


if __name__ == "__main__":
    print("="*72)
    print("R43 O1 · Ф11 · «сначала корреляция, потом когеренция» — топ-k ранжирование")
    print("="*72)

    print("\n[A] КОНФАУНД ОБЩЕГО ДРАЙВЕРА (нерешённый предел Ф7), σ_obs=0.2")
    print(f"  {'κ':>4} " + " ".join(f"{n:>18}" for n in
          ["PLV+thr","PLV+kNN","CORR+kNN","2-ступ.(corr→coh)"]))
    for kap in (0.0, 1.0, 2.0):
        phi, comm = gen(120, 3, kappa=kap, seed=2, obs_noise=0.2)
        r = readers(phi, 3)
        print(f"  {kap:>4} " + " ".join(f"{adjusted_rand_score(comm,r[n]):>18.3f}" for n in
              ["PLV+thr","PLV+kNN","CORR+kNN","2-ступ.(corr→coh)"]))

    print("\n[B] ЧИСТАЯ ТОПОЛОГИЯ без ритма (провал Ф10), σ_obs=0.2")
    phi, comm = gen_norhythm(seed=1)
    r = readers(phi, 3)
    for n in ["PLV+thr","PLV+kNN","CORR+kNN","2-ступ.(corr→coh)"]:
        print(f"  {n:>20}: ARI={adjusted_rand_score(comm,r[n]):.3f}")

    print("\n[C] КОНТРОЛЬ: обычный случай с ритмом (не должно сломаться), σ_obs=0.2")
    phi, comm = gen(150, 3, seed=1, obs_noise=0.2)
    r = readers(phi, 3)
    for n in ["PLV+thr","PLV+kNN","CORR+kNN","2-ступ.(corr→coh)"]:
        print(f"  {n:>20}: ARI={adjusted_rand_score(comm,r[n]):.3f}")
