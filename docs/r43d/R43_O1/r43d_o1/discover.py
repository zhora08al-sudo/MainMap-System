"""
R43 O1 — DERIVATIVE-FREE ВОССТАНОВЛЕНИЕ СТРУКТУРЫ ИЗ НАБЛЮДЕНИЙ (Ф5, новая ниша)

Задача, где дифференциальный подход неприменим/ломается:
  дана НЕ система уравнений, а зашумлённые НАБЛЮДАЕМЫЕ фазовые сигналы связанной
  системы; восстановить её СТРУКТУРУ СООБЩЕСТВ (кто с кем работает в гармонии).

Честное сравнение «одинаковая кластеризация на двух аффинностях»:
  R43 O1:   аффинность = PLV (фазовая когерентность, DERIVATIVE-FREE)
  BASELINE: аффинность = |C| из RIDGE-регрессии Курамото на dφ/dt (ДИФФЕРЕНЦИАЛЬНЫЙ;
            ridge — честно, не strawman). Требует производную сигнала.
  Обе → одна и та же спектральная кластеризация → ARI против истинных сообществ.
  Сетка наблюдательного шума σ_obs.

Опора на факт Ф3: производная зашумлённого сигнала взрывается → дифференциальная
аффинность должна рушиться, derivative-free — держаться.

Данные генерируются НАСТОЯЩЕЙ ОДУ (solve_ivp), режим частичной синхронизации.
Зависимости: numpy, scipy, scikit-learn.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

rng = np.random.default_rng(11)

# ---------- истинная сеть: M сообществ (блочная) ----------
N, M = 45, 3
comm = np.repeat(np.arange(M), N // M)
A = np.zeros((N, N))
for i in range(N):
    for j in range(i+1, N):
        p = 0.85 if comm[i] == comm[j] else 0.02
        if rng.random() < p:
            A[i, j] = A[j, i] = 1.0
deg = A.sum(1); deg[deg == 0] = 1
# у каждого сообщества — свой собственный ритм (реалистично: группа со схожей частотой)
freq_band = np.linspace(-1.5, 1.5, M)
omega = freq_band[comm] + 0.10 * rng.standard_normal(N)
K0 = 1.6                                    # частичная синхронизация: блоки видны, нет глоб. слияния

def rhs(t, phi):
    d = phi[None, :] - phi[:, None]
    return omega + K0 * (A * np.sin(d)).sum(1) / deg

T = 80.0
ts = np.linspace(0, T, 1600)
sol = solve_ivp(rhs, (0, T), rng.uniform(-np.pi, np.pi, N), t_eval=ts,
                rtol=1e-7, atol=1e-9, method="RK45")
phi_true = sol.y.T
dt_obs = ts[1] - ts[0]

def cluster(affinity):
    Aff = affinity.copy()
    Aff = np.clip(Aff, 0, None); np.fill_diagonal(Aff, Aff.max() if Aff.max() > 0 else 1.0)
    sc = SpectralClustering(n_clusters=M, affinity="precomputed",
                            assign_labels="kmeans", random_state=0)
    return sc.fit_predict(Aff)

def affinity_r43o1(phi_noisy):
    """DERIVATIVE-FREE: PLV когерентность."""
    Z = np.exp(1j * phi_noisy)
    return np.abs(Z.conj().T @ Z) / Z.shape[0]

def affinity_derivative(phi_noisy, alpha=1.0):
    """ДИФФЕРЕНЦИАЛЬНЫЙ: ridge-регрессия Курамто на dφ/dt (требует производную)."""
    D = np.gradient(phi_noisy, dt_obs, axis=0)          # ПРОИЗВОДНАЯ сигнала
    C = np.zeros((N, N))
    for i in range(N):
        F = np.sin(phi_noisy - phi_noisy[:, [i]])       # (samples, N)
        X = np.hstack([np.ones((F.shape[0], 1)), F])
        # ridge: (XᵀX+αI)⁻¹ Xᵀy  — честная регуляризация
        XtX = X.T @ X + alpha * np.eye(X.shape[1])
        coef = np.linalg.solve(XtX, X.T @ D[:, i])
        C[i] = coef[1:]
    return np.abs(C) + np.abs(C.T)

if __name__ == "__main__":
    print("="*70)
    print("R43 O1 · Ф5 · восстановление СООБЩЕСТВ из зашумлённых наблюдений")
    print("="*70)
    print(f"Сеть N={N}, {M} сообщества, рёбер={int(A.sum()//2)}; данные из НАСТОЯЩЕЙ ОДУ;")
    print(f"режим частичной синхронизации (K0={K0}). Обе аффинности → одна кластеризация.\n")

    noises = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    print(f"{'σ_obs':>7} {'ARI R43O1 (PLV)':>17} {'ARI дифф (ridge dφ/dt)':>24}")
    rows = []
    for s in noises:
        pn = phi_true + s * rng.standard_normal(phi_true.shape)
        a_r = adjusted_rand_score(comm, cluster(affinity_r43o1(pn)))
        a_d = adjusted_rand_score(comm, cluster(affinity_derivative(pn)))
        rows.append((s, a_r, a_d)); print(f"{s:>7.2f} {a_r:>17.3f} {a_d:>24.3f}")

    r_lo = rows[0]; r_hi = rows[-1]
    r43_robust = min(r[1] for r in rows)            # худший ARI R43O1 по шуму
    deriv_worst = min(r[2] for r in rows)
    # победа: R43O1 устойчиво высок и обыгрывает дифф под шумом
    win = all(r[1] >= 0.85 for r in rows) and any((r[1] - r[2]) >= 0.3 for r in rows[2:])
    print("\n[ЧЕСТНЫЙ ВЫВОД]")
    print(f"  R43 O1 (derivative-free): худший ARI по шуму = {r43_robust:.3f} (устойчивость)")
    print(f"  дифференциальный (ridge): худший ARI = {deriv_worst:.3f} (рушится под шумом)")
    print(f"  при σ_obs=0.8: R43O1={r_hi[1]:.3f} vs дифф={r_hi[2]:.3f}")
    print(f"\n  ИТОГ Ф5: {'PASS' if win else 'нужна доводка'}")
    if win:
        print("  R43 O1 устойчиво восстанавливает структуру из зашумлённых наблюдений,")
        print("  дифференциальный метод деградирует (производная усиливает шум). Это ниша,")
        print("  где derivative-free подход реально замещает дифференциальный — по концепции и цели.")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ns = [r[0] for r in rows]
        plt.figure(figsize=(7.5,4.5))
        plt.plot(ns, [r[1] for r in rows], "o-", lw=2, label="R43 O1 (PLV, derivative-free)")
        plt.plot(ns, [r[2] for r in rows], "s--", lw=2, label="дифференциальный (ridge dφ/dt)")
        plt.axhline(0.85, color="g", ls=":", alpha=.6, label="порог ARI=0.85")
        plt.xlabel("наблюдательный шум σ_obs"); plt.ylabel("ARI восстановления сообществ")
        plt.title("Структура из шумных наблюдений: derivative-free vs дифференциальный")
        plt.legend(); plt.grid(alpha=.3); plt.ylim(-0.05, 1.05); plt.tight_layout()
        plt.savefig("discover.png", dpi=90); print("  график сохранён: discover.png")
    except Exception as e:
        print(f"  (график пропущён: {e})")
