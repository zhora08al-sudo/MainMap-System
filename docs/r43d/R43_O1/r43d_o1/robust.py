"""
R43 O1 — УСТОЙЧИВЫЕ МЕХАНИЗМЫ ИЗВЛЕЧЕНИЯ (доработка слабого места Ф3)

Концепция сохранена (фазовая когерентность из хаоса), правила соблюдены
(derivative-free, без d/dt). Меняется только СЧИТЫВАТЕЛЬ структуры, который в Ф3
оказался хрупким под комбинированным возмущением (порог + компоненты связности
склеиваются от одного ложного ребра).

Тестируем три derivative-free считывателя на ОДНОЙ и той же провальной сцене
(«ВСЁ сразу»: шум+скачки+Piecewise, ARI базового = 0.895):

  R0  базовый:        PLV (плоское среднее) + порог + компоненты связности  [Ф3]
  R1  оператор C̄:     МНОГООКОННАЯ МЕДИАННАЯ когерентность + спектральное извлечение
  R2  спектральный:   PLV (плоское среднее) + СПЕКТРАЛЬНОЕ извлечение

Новое понятие (честно, с родословной):
  ОПЕРАТОР КОГЕРЕНТНОСТИ  C̄_ij = median_w | <e^{i(φ_i−φ_j)}>_{окно w} |
  — робастная (медианная по окнам) оценка фазового сцепления; математически это
  устойчивый估 оценщик PLV (lineage: phase-locking value + оконное усреднение +
  робастная статистика). Извлечение структуры — спектральная кластеризация
  (собственные вектора нормированного лапласиана), derivative-free.

Зависимости: numpy, scipy, scikit-learn.
"""
from __future__ import annotations
import inspect
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from structure import make_communities, coupling_matrix, extract_ensembles
from events import evolve_with_events


# --- нужен доступ к комплексному хвосту Z, поэтому лёгкая копия эволюции, ---
# --- возвращающая Z_tail (та же derivative-free динамика, что и в events) ---
def evolve_collect_Z(omega, phi, W, labels, M, steps=1500, dt=0.05, tail=400,
                     sigma_noise=0.0, lam_jump=0.0, sigma_jump=1.5,
                     piecewise_switch=False, seed=0):
    rng = np.random.default_rng(seed)
    N = len(phi); omega = omega.copy()
    base_vals = np.linspace(-2.0, 2.0, M); switch_at = steps // 2 if piecewise_switch else -1
    Z_tail = np.empty((tail, N), dtype=complex); t0 = steps - tail
    for k in range(steps):
        if k == switch_at:
            omega = base_vals[rng.permutation(M)][labels] + 0.05 * rng.standard_normal(N)
        coupling = (W * np.sin(phi[None, :] - phi[:, None])).mean(axis=1)
        phi = phi + dt * (omega + coupling)
        if sigma_noise > 0:
            phi = phi + sigma_noise * np.sqrt(dt) * rng.standard_normal(N)
        if lam_jump > 0:
            hit = rng.random(N) < lam_jump * dt
            if hit.any():
                phi = phi + hit * rng.normal(0.0, sigma_jump, N)
        if k >= t0:
            Z_tail[k - t0] = np.exp(1j * phi)
    return Z_tail


def plv_flat(Z):
    T = Z.shape[0]
    return np.abs(Z.conj().T @ Z) / T


def coherence_operator(Z, nwin=8):
    """C̄: медианная по окнам когерентность — робастный оценщик PLV (новое понятие)."""
    T, N = Z.shape
    w = T // nwin
    mats = []
    for i in range(nwin):
        Zi = Z[i*w:(i+1)*w]
        mats.append(np.abs(Zi.conj().T @ Zi) / Zi.shape[0])
    return np.median(np.stack(mats), axis=0)


def extract_spectral(affinity, n_clusters):
    """Спектральная кластеризация на операторе когерентности (derivative-free)."""
    A = affinity.copy(); np.fill_diagonal(A, 1.0)
    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed",
                            assign_labels="kmeans", random_state=0)
    return sc.fit_predict(A)


def eigengap_estimate(affinity, kmax=12):
    """Оценка числа кластеров по собственному зазору нормированного лапласиана."""
    A = affinity.copy(); np.fill_diagonal(A, 0.0)
    d = A.sum(1); d[d == 0] = 1e-12
    Dm = 1.0 / np.sqrt(d)
    L = np.eye(len(A)) - (Dm[:, None] * A * Dm[None, :])
    ev = np.sort(np.linalg.eigvalsh(L))[:kmax+1]
    gaps = ev[1:] - ev[:-1]                      # зазор спектра (модель-селект), не d/dt
    return int(np.argmax(gaps) + 1), ev


def _code_only(src: str) -> str:
    """Убрать комментарии и строковые литералы — чтобы аудит не ловил слова в тексте."""
    import io, tokenize
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except tokenize.TokenError:
        pass
    return " ".join(out)


def check_derivative_free() -> bool:
    """SC1: в ВЫЧИСЛИТЕЛЬНОМ коде нет операторов производной по времени (d/dt сигнала)."""
    forbidden = ("np.gradient", "gradient", "np.diff", "derivative", "finite_difference")
    runtime = (evolve_collect_Z, plv_flat, coherence_operator, extract_spectral, eigengap_estimate)
    hits = [(fn.__name__, t) for fn in runtime
            for t in forbidden if t in _code_only(inspect.getsource(fn))]
    if hits:
        print("  SC1 НАРУШЕН:", hits); return False
    print("  SC1: вычислительный код без операторов производной — derivative-free")
    print("       (спектральная кластеризация = линейная алгебра на матрице сходства; нет d/dt)")
    return True


if __name__ == "__main__":
    print("="*66)
    print("R43 O1 · Доработка считывателя структуры — head-to-head на провальной сцене")
    print("="*66)
    print("\n[SC1] derivative-free:")
    sc1 = check_derivative_free()

    N, M = 600, 6
    labels, omega, phi0 = make_communities(N=N, M=M, seed=1)
    W = coupling_matrix(labels, K_in=3.0, K_out=0.2)
    hard = dict(sigma_noise=0.4, lam_jump=0.05, sigma_jump=1.5, piecewise_switch=True)

    Z = evolve_collect_Z(omega, phi0.copy(), W, labels, M, steps=1500, dt=0.05,
                         tail=400, seed=2, **hard)
    plv = plv_flat(Z)
    Cbar = coherence_operator(Z, nwin=8)

    def score(rec): return adjusted_rand_score(labels, rec), normalized_mutual_info_score(labels, rec)

    # R0 базовый (как в Ф3)
    _, rec0 = extract_ensembles(plv, theta_coh=0.8); a0, n0 = score(rec0)
    # R2 спектральный на плоском PLV
    rec2 = extract_spectral(plv, M); a2, n2 = score(rec2)
    # R1 оператор C̄ (медианный) + спектральный
    rec1 = extract_spectral(Cbar, M); a1, n1 = score(rec1)

    kgap_plv, _ = eigengap_estimate(plv)
    kgap_cbar, _ = eigengap_estimate(Cbar)

    print("\nСцена: «ВСЁ сразу» (шум σ=0.4 + скачки λ=0.05 + Piecewise-разрыв)")
    print(f"  {'механизм':52} {'ARI':>6} {'NMI':>6}")
    print(f"  {'R0  PLV + порог + компоненты связности (Ф3, база)':52} {a0:>6.3f} {n0:>6.3f}")
    print(f"  {'R2  PLV + спектральное извлечение':52} {a2:>6.3f} {n2:>6.3f}")
    print(f"  {'R1  оператор C̄ (медиан. по окнам) + спектральное':52} {a1:>6.3f} {n1:>6.3f}")
    print(f"\n  оценка числа кластеров по eigengap: PLV→{kgap_plv}, C̄→{kgap_cbar} (истинно {M})")

    best = max([("R0", a0), ("R1", a1), ("R2", a2)], key=lambda x: x[1])
    print(f"\n[ИТОГ] лучший считыватель: {best[0]} с ARI={best[1]:.3f}")
    print(f"  порог SC3 (ARI≥0.9) на комбо-экстриме: "
          f"{'ПРЕОДОЛЁН' if best[1] >= 0.9 else 'НЕ преодолён'}")
    print(f"  derivative-free сохранён: {'да' if sc1 else 'НЕТ'}")
    if best[1] >= 0.9 and a0 < 0.9:
        print(f"  => улучшение over Ф3-базы: {a0:.3f} → {best[1]:.3f} (+{best[1]-a0:.3f}), "
              f"концепция сохранена, правила соблюдены")
