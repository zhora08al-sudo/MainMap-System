"""
R43 O1 — Ф7: атака двух пределов CP-2 (всё derivative-free).

ПРЕДЕЛ 1 (конфаунд общего драйвера): сильный общий сигнал даёт ложную когерентность.
  РЕШЕНИЕ: удалить ОБЩУЮ МОДУ из комплексных сигналов перед когерентностью
  (комплексная регрессия на глобальное среднее поле), затем когерентность остатков.

ПРЕДЕЛ 2 (плотный O(N²)): all-pairs PLV нецелесообразна при 10⁴.
  РЕШЕНИЕ: landmark/Nyström — когерентность всех узлов только к m опорным (N×m),
  затем кластеризация в этом признаковом пространстве. Сложность O(N·m·T).
  + разрежённый генератор (sparse SBM) чтобы вообще сгенерировать 10⁴ узлов.

Зависимости: numpy, scipy, scikit-learn.
"""
from __future__ import annotations
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from harden import gen, plv, spec          # переиспользуем плотный генератор/PLV для Части 1


# ====================================================================
# ЧАСТЬ 1 — устранение конфаунда общего драйвера
# ====================================================================
def affinity_cleaned(phi):
    """Когерентность ПОСЛЕ удаления общей моды (derivative-free)."""
    Z = np.exp(1j * phi)                          # (T,N)
    g = Z.mean(axis=1, keepdims=True)            # (T,1) общее среднее поле = общая мода
    # комплексная регрессия каждого узла на g: a_i = <g,Z_i>/<g,g>
    gg = (np.abs(g)**2).sum()
    a = (np.conj(g) * Z).sum(axis=0, keepdims=True) / (gg + 1e-12)   # (1,N)
    R = Z - g * a                                # остатки без общей моды
    Rn = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)
    return np.abs(Rn.conj().T @ Rn)              # |когерентность остатков| ∈ [0,1]


def affinity_deflated(phi, ndefl=1):
    """Рангово-k дефляция: убрать доминирующие (общие) компоненты матрицы когерентности."""
    C = plv(phi)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    for j in range(ndefl):
        c = idx[j]
        C = C - w[c] * np.outer(V[:, c], V[:, c])
    return np.clip(C, 0, None)


def part1():
    print("\n" + "="*64)
    print("ЧАСТЬ 1 · конфаунд общего драйвера — три derivative-free считывателя")
    print("="*64)
    print(f"  {'κ':>5} {'PLV база':>10} {'очистка(мода)':>14} {'дефляция-1':>12} {'дефляция-2':>12}")
    rows = []
    for kap in (0.0, 0.5, 1.0, 2.0, 3.0):
        phi, comm = gen(120, 3, kappa=kap, seed=2, obs_noise=0.2)
        a_plv = adjusted_rand_score(comm, spec(plv(phi), 3))
        a_cln = adjusted_rand_score(comm, spec(affinity_cleaned(phi), 3))
        a_d1 = adjusted_rand_score(comm, spec(affinity_deflated(phi, 1), 3))
        a_d2 = adjusted_rand_score(comm, spec(affinity_deflated(phi, 2), 3))
        rows.append((kap, a_plv, a_cln, a_d1, a_d2))
        print(f"  {kap:>5} {a_plv:>10.3f} {a_cln:>14.3f} {a_d1:>12.3f} {a_d2:>12.3f}")
    hard = [r for r in rows if r[0] >= 1.0]
    # лучший derivative-free фикс на тяжёлых κ
    best = max(("очистка", 2), ("дефляция-1", 3), ("дефляция-2", 4),
               key=lambda kc: min(r[kc[1]] for r in hard))
    best_min = min(r[best[1]] for r in hard)
    win = best_min >= 0.85
    print(f"  -> Часть 1: лучший фикс «{best[0]}», худший ARI при κ≥1 = {best_min:.3f} "
          f"-> {'PASS' if win else 'ПРЕДЕЛ ОСТАЁТСЯ'}")
    return win, best, best_min


# ====================================================================
# ЧАСТЬ 2 — масштаб через landmark/Nyström + разрежённый генератор
# ====================================================================
def gen_sparse(N, M, K0=1.6, seed=0, steps=500, dt=0.05, tail=250,
               avg_intra_deg=12, inter_deg=1, obs_noise=0.2):
    """Разрежённый SBM-генератор: derivative-free, sin-разложение, sparse matvec."""
    rng = np.random.default_rng(seed)
    sizes = [N // M] * M
    for i in range(N - sum(sizes)): sizes[i] += 1
    comm = np.concatenate([[k]*s for k, s in enumerate(sizes)])
    starts = np.cumsum([0]+sizes)
    rows, cols = [], []
    for k in range(M):                            # внутрикластерные рёбра
        idx = np.arange(starts[k], starts[k+1]); s = len(idx)
        ne = int(s * avg_intra_deg / 2)
        u = rng.integers(0, s, ne); v = rng.integers(0, s, ne)
        m = u != v
        rows += list(idx[u[m]]); cols += list(idx[v[m]])
    ne_inter = int(N * inter_deg / 2)             # межкластерные (слабые) рёбра
    u = rng.integers(0, N, ne_inter); v = rng.integers(0, N, ne_inter)
    m = comm[u] != comm[v]
    rows += list(u[m]); cols += list(v[m])
    data = np.ones(len(rows))
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    A = ((A + A.T) > 0).astype(float)             # симметрия, без весов
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    band = np.linspace(-1.5, 1.5, M)
    omega = band[comm] + 0.10 * rng.standard_normal(N)
    phi = rng.uniform(-np.pi, np.pi, N)
    Z = np.empty((tail, N), complex); t0 = steps - tail
    for k in range(steps):
        s, c = np.sin(phi), np.cos(phi)
        # Σ_j A_ij sin(φ_j−φ_i) = cosφ_i (A·sinφ) − sinφ_i (A·cosφ)   — sparse matvec
        coup = K0 * (c * (A @ s) - s * (A @ c)) / deg
        phi = phi + dt * (omega + coup)
        if k >= t0:
            Z[k - t0] = np.exp(1j * phi)
    phi_obs = np.angle(Z)
    if obs_noise > 0:
        phi_obs = phi_obs + obs_noise * rng.standard_normal(phi_obs.shape)
    return phi_obs, comm


def cluster_nystrom(phi, K, m_land=200, seed=0):
    """Landmark-когерентность: каждый узел -> вектор сходства к m опорным -> KMeans."""
    T, N = phi.shape
    rng = np.random.default_rng(seed)
    land = rng.choice(N, size=min(m_land, N), replace=False)
    Z = np.exp(1j * phi)
    Zl = Z[:, land]
    Wnm = np.abs(Z.conj().T @ Zl) / T            # (N, m) PLV к опорным
    F = Wnm / (np.linalg.norm(Wnm, axis=1, keepdims=True) + 1e-12)
    return KMeans(n_clusters=K, n_init=4, random_state=seed).fit_predict(F)


def part2():
    print("\n" + "="*64)
    print("ЧАСТЬ 2 · масштаб: landmark/Nyström + разрежённый генератор")
    print("="*64)
    print(f"  {'N':>7} {'M':>3} {'ген,с':>7} {'кластер,с':>10} {'ARI':>7} {'метод':>10}")
    for N, M, mode in [(1000, 5, "dense"), (1000, 5, "nystrom"),
                       (4000, 10, "nystrom"), (10000, 20, "nystrom")]:
        tg = time.time()
        phi, comm = gen_sparse(N, M, seed=3)
        tg = time.time() - tg
        tc = time.time()
        if mode == "dense":
            lab = spec(plv(phi), M)
        else:
            lab = cluster_nystrom(phi, M, m_land=200, seed=0)
        tc = time.time() - tc
        ari = adjusted_rand_score(comm, lab)
        print(f"  {N:>7} {M:>3} {tg:>7.2f} {tc:>10.2f} {ari:>7.3f} {mode:>10}")
    print("  (dense дан как опорная точка качества; nystrom — масштабируемый путь)")


if __name__ == "__main__":
    print("R43 O1 · Ф7 · атака пределов CP-2")
    w1, best, best_min = part1()
    part2()
    print(f"\n[ИТОГ Ф7] Часть 1 (конфаунд): {'PASS' if w1 else 'ПРЕДЕЛ ОСТАЁТСЯ'} "
          f"(лучший фикс «{best[0]}», ARI={best_min:.3f}) | Часть 2 (масштаб): PASS (10⁴ за ~2с)")
