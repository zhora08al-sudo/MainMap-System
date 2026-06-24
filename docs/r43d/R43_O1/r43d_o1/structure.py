"""
R43 O1 — ИЗВЛЕЧЕНИЕ СТРУКТУРЫ (Ф2, столп P3)

Реализует сценарий автора: «хаотическое поле фазовых узлов → выделить когерентные
ансамбли». Полностью derivative-free.

Конвейер (= D.I.D.I. F2–F7, подтверждённый в GLOBAL_CONTEXT §3.2):
  1. эволюция фаз дискретным ядром со СВЯЗЯМИ по сообществам (community-Kuramoto)
  2. PLV_ij = |<e^{i(φ_i−φ_j)}>_t|  — фазовая когерентность по хвосту траектории
  3. порог θ_coh → граф связности → компоненты связности = ансамбли (F4/F7)
  4. сверка восстановленных ансамблей с ground-truth: ARI и NMI (критерий SC2)

Зависимости: numpy, scipy, scikit-learn (метрики). matplotlib опционально.
"""
from __future__ import annotations
import inspect
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ----------------------------------------------------------------------
# Генератор «хаоса со скрытой структурой»: M сообществ
# ----------------------------------------------------------------------
def make_communities(N=600, M=6, sigma_jitter=0.05, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.sort(rng.integers(0, M, N))            # принадлежность узлов
    base = np.linspace(-2.0, 2.0, M)                   # своя частота у каждого сообщества
    omega = base[labels] + sigma_jitter * rng.standard_normal(N)
    phi0 = rng.uniform(-np.pi, np.pi, N)               # старт из хаоса
    return labels, omega, phi0


def coupling_matrix(labels, K_in=3.0, K_out=0.2):
    """W_ij: сильная связь внутри сообщества, слабая — между."""
    same = labels[:, None] == labels[None, :]
    return np.where(same, K_in, K_out)


# ----------------------------------------------------------------------
# Эволюция (derivative-free дискретный шаг со связями W) + сбор PLV
# ----------------------------------------------------------------------
def evolve_and_plv(omega, phi, W, steps=1500, dt=0.05, tail=400):
    N = len(phi)
    Z_tail = np.empty((tail, N), dtype=complex)
    t0 = steps - tail
    for k in range(steps):
        # (1/N) Σ_j W_ij sin(φ_j − φ_i)   — только sin/суммы, без производных
        coupling = (W * np.sin(phi[None, :] - phi[:, None])).mean(axis=1)
        phi = phi + dt * (omega + coupling)
        if k >= t0:
            Z_tail[k - t0] = np.exp(1j * phi)
    plv = np.abs(Z_tail.conj().T @ Z_tail) / tail       # PLV_ij ∈ [0,1]
    return plv, phi


# ----------------------------------------------------------------------
# Выделение ансамблей: порог когерентности → компоненты связности
# ----------------------------------------------------------------------
def extract_ensembles(plv, theta_coh=0.8):
    adj = (plv >= theta_coh)
    np.fill_diagonal(adj, False)
    n_comp, comp_labels = connected_components(adj, directed=False)
    return n_comp, comp_labels


# ----------------------------------------------------------------------
# SC1 — derivative-free самопроверка рантайма
# ----------------------------------------------------------------------
def check_derivative_free() -> bool:
    forbidden = ("np.gradient", "gradient(", "np.diff", ".diff(",
                 "derivative", "d2", "d/dt", "/dt", "finite_difference")
    runtime = (evolve_and_plv, extract_ensembles, coupling_matrix)
    hits = [(fn.__name__, t) for fn in runtime for t in forbidden if t in inspect.getsource(fn)]
    if hits:
        print("  SC1 НАРУШЕН:", hits); return False
    print("  SC1: рантайм без операторов производной — derivative-free")
    return True


# ----------------------------------------------------------------------
# Приёмка Ф2
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("="*64)
    print("R43 O1 · Ф2 · извлечение структуры — приёмка")
    print("="*64)

    print("\n[SC1] derivative-free:")
    sc1 = check_derivative_free()

    N, M = 600, 6
    labels, omega, phi0 = make_communities(N=N, M=M, seed=1)
    W = coupling_matrix(labels, K_in=3.0, K_out=0.2)
    print(f"\n[Сцена] N={N} узлов, M={M} зашитых сообществ, старт из хаоса")
    plv, phi = evolve_and_plv(omega, phi0, W, steps=1500, dt=0.05, tail=400)

    # средняя когерентность внутри/между (диагностика блочности)
    same = labels[:, None] == labels[None, :]
    off = ~same
    print(f"  средн. PLV внутри сообществ : {plv[same].mean():.3f}  (ожид. →1)")
    print(f"  средн. PLV между сообществами: {plv[off].mean():.3f}  (ожид. →0)")

    n_comp, rec = extract_ensembles(plv, theta_coh=0.8)
    ari = adjusted_rand_score(labels, rec)
    nmi = normalized_mutual_info_score(labels, rec)
    print(f"\n[Восстановление] найдено компонент: {n_comp} (истинно {M})")
    print(f"  ARI = {ari:.3f}   (критерий SC2 ≥ 0.90)")
    print(f"  NMI = {nmi:.3f}   (критерий SC2 ≥ 0.90)")

    ok_blocks = plv[same].mean() > 0.8 and plv[off].mean() < 0.3
    ok_sc2 = ari >= 0.9 and nmi >= 0.9
    print("\n[ИТОГ Ф2]")
    print(f"  SC1 derivative-free .......... {'PASS' if sc1 else 'FAIL'}")
    print(f"  блочная когерентность ........ {'PASS' if ok_blocks else 'FAIL'}")
    print(f"  SC2 восстановление (ARI/NMI) . {'PASS' if ok_sc2 else 'FAIL'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = np.argsort(labels)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        im = ax[0].imshow(plv[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
        ax[0].set_title("PLV (упорядочено по истинным сообществам)")
        plt.colorbar(im, ax=ax[0], fraction=0.046)
        ax[1].scatter(range(N), rec[order], s=6, label="восстановлено")
        ax[1].plot(range(N), labels[order], "r-", lw=1, label="истина")
        ax[1].set_title(f"Ансамбли: ARI={ari:.2f}, NMI={nmi:.2f}")
        ax[1].set_xlabel("узлы (упорядочены)"); ax[1].legend()
        plt.tight_layout(); plt.savefig("structure.png", dpi=90)
        print("  график сохранён: structure.png")
    except Exception as e:
        print(f"  (график пропущен: {e})")
