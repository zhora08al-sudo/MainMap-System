"""
R43 O1 — УЖЕСТОЧЕНИЕ НИШИ (Ф6): проверка Ф5D на прочность за пределами идеала.

Ниша Ф5D (derivative-free открытие структуры из шумных наблюдений) проверяется в
тяжёлых условиях. Каждый суб-тест честно печатает PASS/FAIL:

  T6.1  НЕИЗВЕСТНОЕ K: число сообществ не сообщаем — берём по eigengap.
  T6.2  СКРЫТЫЕ УЗЛЫ: 30% узлов не наблюдаются; восстановить сообщества остальных.
  T6.3  КОНФАУНД ОБЩЕГО ДРАЙВЕРА: общий внешний сигнал создаёт ЛОЖНУЮ когерентность —
        классическая слабость PLV. Где порвётся — фиксируем честно.
  T6.4  МАСШТАБ/ВРЕМЯ: рост N; стена плотной O(N²) аффинности.

Данные генерируются derivative-free дискретным отображением (для восстановления
способ генерации несуществен — метод видит только фазы). Шум наблюдений σ_obs.
Зависимости: numpy, scikit-learn.
"""
from __future__ import annotations
import time
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score


def gen(N, M, K0=1.6, kappa=0.0, seed=0, steps=900, dt=0.05, tail=400,
        intra=0.85, inter=0.02, obs_noise=0.0):
    """Дискретная фазовая динамика: сообщества со своим ритмом + блочная связь
       (+ опц. общий драйвер силы kappa). Возвращает наблюдаемые фазы (tail,N) и comm."""
    rng = np.random.default_rng(seed)
    comm = np.repeat(np.arange(M), N // M)
    if len(comm) < N:
        comm = np.concatenate([comm, rng.integers(0, M, N - len(comm))])
    A = np.zeros((N, N))
    same = comm[:, None] == comm[None, :]
    pr = np.where(same, intra, inter); np.fill_diagonal(pr, 0)
    iu = np.triu_indices(N, 1)
    e = rng.random(len(iu[0])) < pr[iu]
    A[iu[0][e], iu[1][e]] = 1.0; A = A + A.T
    deg = A.sum(1); deg[deg == 0] = 1
    band = np.linspace(-1.5, 1.5, M)
    omega = band[comm] + 0.10 * rng.standard_normal(N)
    phi = rng.uniform(-np.pi, np.pi, N)
    Om_d = 0.7                                   # частота общего драйвера
    Z = np.empty((tail, N), complex); t0 = steps - tail; t = 0.0
    for k in range(steps):
        coup = K0 * (A * np.sin(phi[None, :] - phi[:, None])).sum(1) / deg
        drive = kappa * np.sin(Om_d * t - phi) if kappa > 0 else 0.0
        phi = phi + dt * (omega + coup + drive); t += dt
        if k >= t0:
            Z[k - t0] = np.exp(1j * phi)
    phi_obs = np.angle(Z)
    if obs_noise > 0:
        phi_obs = phi_obs + obs_noise * rng.standard_normal(phi_obs.shape)
    return phi_obs, comm


def plv(phi):
    Z = np.exp(1j * phi)
    return np.abs(Z.conj().T @ Z) / Z.shape[0]


def spec(aff, K):
    A = np.clip(aff, 0, None).copy(); np.fill_diagonal(A, 1.0)
    return SpectralClustering(n_clusters=K, affinity="precomputed",
                             assign_labels="kmeans", random_state=0).fit_predict(A)


def eigengap_K(aff, kmax=10):
    A = np.clip(aff, 0, None).copy(); np.fill_diagonal(A, 0.0)
    d = A.sum(1); d[d == 0] = 1e-12; Dm = 1/np.sqrt(d)
    L = np.eye(len(A)) - (Dm[:, None] * A * Dm[None, :])
    ev = np.sort(np.linalg.eigvalsh(L))[:kmax+1]
    gaps = ev[1:] - ev[:-1]
    return int(np.argmax(gaps[1:]) + 2)          # ищем K≥2


if __name__ == "__main__":
    print("="*68)
    print("R43 O1 · Ф6 · ужесточение ниши Ф5D — стресс-тесты")
    print("="*68)
    NOISE = 0.2

    # ---- T6.1 неизвестное K (eigengap) ----
    print("\n[T6.1] НЕИЗВЕСТНОЕ число сообществ — берём по eigengap (σ_obs=0.2)")
    ok = 0; trials = 6
    for M in (2, 3, 4, 5, 6, 7):
        phi, comm = gen(120, M, seed=M, obs_noise=NOISE)
        Khat = eigengap_K(plv(phi))
        hit = (Khat == M); ok += hit
        print(f"  истинно M={M}: eigengap K={Khat}  {'OK' if hit else 'промах'}")
    t61 = ok >= 5
    print(f"  -> T6.1: {ok}/{trials} верно  {'PASS' if t61 else 'FAIL'}")

    # ---- T6.2 скрытые узлы ----
    print("\n[T6.2] СКРЫТЫЕ УЗЛЫ: наблюдаем только 70% (σ_obs=0.2)")
    phi, comm = gen(150, 3, seed=1, obs_noise=NOISE)
    rng = np.random.default_rng(0)
    keep = rng.random(phi.shape[1]) < 0.7
    ari_hidden = adjusted_rand_score(comm[keep], spec(plv(phi[:, keep]), 3))
    t62 = ari_hidden >= 0.85
    print(f"  наблюдаемых узлов: {keep.sum()}/150;  ARI={ari_hidden:.3f}  {'PASS' if t62 else 'FAIL'}")

    # ---- T6.3 конфаунд общего драйвера ----
    print("\n[T6.3] КОНФАУНД ОБЩЕГО ДРАЙВЕРА (ложная когерентность), σ_obs=0.2")
    print("  ожидаем деградацию с ростом силы драйвера κ — честно фиксируем предел")
    for kap in (0.0, 0.5, 1.0, 2.0):
        phi, comm = gen(120, 3, kappa=kap, seed=2, obs_noise=NOISE)
        ari = adjusted_rand_score(comm, spec(plv(phi), 3))
        print(f"  κ={kap:>4}: ARI={ari:.3f}")
    print("  (T6.3 — диагностический: показывает границу применимости PLV, не PASS/FAIL)")

    # ---- T6.4 масштаб/время ----
    print("\n[T6.4] МАСШТАБ / ВРЕМЯ (плотная O(N²) аффинность)")
    print(f"  {'N':>6} {'время,с':>9} {'ARI':>7}")
    for N in (200, 500, 1000, 2000):
        t0 = time.time()
        phi, comm = gen(N, 3, seed=3, steps=600, tail=300, obs_noise=NOISE)
        ari = adjusted_rand_score(comm, spec(plv(phi), 3))
        print(f"  {N:>6} {time.time()-t0:>9.2f} {ari:>7.3f}")
    print("  ЧЕСТНО: плотная PLV и спектральная кластеризация — O(N²) по памяти/времени.")
    print("  10⁴ узлов плотно нецелесообразно (≈10⁸ элементов); нужна разрежённая k-NN")
    print("  аффинность / приближённые собств. вектора — это задача следующей фазы (Ф7).")

    print("\n[ИТОГ Ф6]")
    print(f"  T6.1 неизвестное K (eigengap) ... {'PASS' if t61 else 'FAIL'}")
    print(f"  T6.2 скрытые узлы ............... {'PASS' if t62 else 'FAIL'}")
    print(f"  T6.3 конфаунд драйвера .......... диагностика (см. таблицу)")
    print(f"  T6.4 масштаб .................... измерено; плотный предел ~ тысячи узлов")
