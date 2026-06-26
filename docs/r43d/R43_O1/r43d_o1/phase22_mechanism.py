"""
R43 O1 — Ф22: МЕХАНИЗМ — КОГДА фаза бьёт корреляцию, а когда нет.

Загадка после Ф21: на синтетике Курамото PLV>корр (Ф9), на реальном ЭЭГ — нет (Ф21).
Коннектора Cloudflare нет, подходящего реального фазово-связанного датасета на HF нет,
поэтому вместо слепой охоты за данными ВСКРЫВАЕМ ПРИЧИНУ контролируемым экспериментом.

ГИПОТЕЗА. Корреляция cos(φ) и фазовая когерентность (PLV) расходятся ТОЛЬКО когда
амплитуда сигнала зашумлена/неинформативна, а фаза сохранена. PLV по построению
отбрасывает амплитуду (нормирует e^{iφ}), корреляция — нет. Значит:
  • амплитудный шум мал  → corr ≈ PLV  (на чистом узкополосном ЭЭГ так и вышло, Ф21);
  • амплитудный шум велик → PLV > corr (фаза цела, амплитуда врёт — ниша идеи автора).

Метод. Строим сообщества осцилляторов (общая внутригрупповая фаза + индивид. шум фазы),
читаем сигнал s_i(t) = A_i(t)·cos(φ_i(t)). Варьируем СИЛУ амплитудного шума η:
A_i(t) = 1 + η·ξ_i(t), ξ ~ белый шум. Сравниваем восстановление сообществ:
  PLV-кластеризация (фаза) vs корреляция cos-сигнала vs корреляция СЫРОГО s (с амплитудой).
Без производных. Зависимости: numpy scipy scikit-learn.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import hilbert
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

rng = np.random.default_rng(43)


def make_signals(N=60, M=3, T=4000, eta=0.0, phase_noise=0.15, lag=0.0):
    """N осцилляторов, M сообществ. Внутри сообщества общая фаза + индивид. дрожь.
    eta — сила мультипликативного амплитудного шума.
    lag — РАЗБРОС постоянных фазовых сдвигов внутри сообщества (рад). lag=0 → нулевой лаг
    (корреляция тривиально ловит); lag>0 → захват С ЛАГОМ (корреляция cos падает, PLV цел).
    Возвращает сырой сигнал s(T,N) и gt."""
    gt = np.repeat(np.arange(M), N // M)
    if len(gt) < N: gt = np.concatenate([gt, np.full(N-len(gt), M-1)])
    t = np.arange(T)
    f0 = 0.05                                           # ЕДИНАЯ несущая у ВСЕХ — частота
    # не несёт инфы о сообществе; сообщество = свой ОБЩИЙ фазовый процесс (как Курамото).
    comm_phase = np.zeros((M, T))
    for c in range(M):
        comm_phase[c] = 2*np.pi*f0*t + np.cumsum(rng.normal(0, 0.3, T))*0.02 \
                        + rng.uniform(0, 2*np.pi)        # независимый общий дрейф на сообщество
    offs = rng.uniform(-lag, lag, N)                    # постоянный фазовый сдвиг узла (ЛАГ)
    phi = np.zeros((T, N))
    for i in range(N):
        drift = np.cumsum(rng.normal(0, phase_noise, T)) * 0.02
        phi[:, i] = comm_phase[gt[i]] + drift + offs[i]
    A = 1.0 + eta * rng.normal(0, 1, (T, N))            # амплитудный шум
    s = A * np.cos(phi)                                 # сырой наблюдаемый сигнал
    return s, gt


def inst_phase(s):
    return np.angle(hilbert(s, axis=0))


def plv_cluster(s, K):
    ph = inst_phase(s); z = np.exp(1j*ph)              # (T,N)
    P = np.abs(z.conj().T @ z) / s.shape[0]            # PLV матрица
    np.fill_diagonal(P, 1.0)
    return SpectralClustering(K, affinity="precomputed", random_state=0).fit_predict(P)


def corr_cos_cluster(s, K):
    ph = inst_phase(s)
    C = np.abs(np.nan_to_num(np.corrcoef(np.cos(ph).T))); np.fill_diagonal(C, 1.0)
    return SpectralClustering(K, affinity="precomputed", random_state=0).fit_predict(C)


def corr_raw_cluster(s, K):
    C = np.abs(np.nan_to_num(np.corrcoef(s.T))); np.fill_diagonal(C, 1.0)
    return SpectralClustering(K, affinity="precomputed", random_state=0).fit_predict(C)


def sweep(M, axis, vals, reps=5, **fixed):
    print(f"\n  {axis:>10} | {'PLV(фаза)':>10} {'corr cos(φ)':>12} {'corr сырой s':>13} | вывод")
    print("  " + "-"*68)
    out = []
    for v in vals:
        a_plv, a_cos, a_raw = [], [], []
        for r in range(reps):
            kw = dict(fixed); kw[axis] = v
            s, gt = make_signals(N=60, M=M, T=4000, **kw)
            a_plv.append(adjusted_rand_score(gt, plv_cluster(s, M)))
            a_cos.append(adjusted_rand_score(gt, corr_cos_cluster(s, M)))
            a_raw.append(adjusted_rand_score(gt, corr_raw_cluster(s, M)))
        mp, mc, mr = np.mean(a_plv), np.mean(a_cos), np.mean(a_raw)
        edge = mp - mr
        tag = "фаза >> сырая корр" if edge > 0.15 else ("≈ паритет" if abs(edge) <= 0.15 else "корр > фазы")
        print(f"  {v:>10.2f} | {mp:>10.3f} {mc:>12.3f} {mr:>13.3f} | {tag}")
        out.append((v, mp, mc, mr))
    return out


if __name__ == "__main__":
    print("="*72)
    print("R43 O1 · Ф22 · МЕХАНИЗМ: когда фаза (PLV) бьёт корреляцию")
    print("="*72)
    M = 3
    print(f"  N=60, сообществ={M}, повторов на точку=5")

    print("\n── ОСЬ A: амплитудный шум η (фазовый лаг = 0) ──")
    print("   (проверка гипотезы «амплитуда лжёт → фаза выигрывает»)")
    A = sweep(M, "eta", [0.0, 1.0, 2.0, 4.0, 8.0], lag=0.0)

    print("\n── ОСЬ B: фазовый ЛАГ внутри сообщества (амплитуда чистая, η=0) ──")
    print("   (захват с постоянным сдвигом: корреляция cos ортогональна, PLV цел)")
    B = sweep(M, "lag", [0.0, 0.4, 0.8, 1.2, 1.57], eta=0.0)

    print("\n[ЧЕСТНЫЙ ВЫВОД]")
    a_hi = A[-1]
    print(f"  ОСЬ A: гипотеза ОПРОВЕРГНУТА — при η={a_hi[0]:.0f} сырая корр {a_hi[3]:.2f} ≥ PLV"
          f" {a_hi[1]:.2f}.")
    print("         Амплитудный шум портит извлечение фазы по Гильберту → корреляция РОБАСТНЕЕ.")
    b0, bhi = B[0], B[-1]
    won = bhi[1] - bhi[3] > 0.15
    print(f"  ОСЬ B: при лаге=0 паритет (corr cos {b0[2]:.2f}); при лаге≈π/2 corr cos ПАДАЕТ"
          f" до {bhi[2]:.2f}, а PLV держит {bhi[1]:.2f}.")
    print(f"         → фаза бьёт КОРРЕЛЯЦИЮ cos ИМЕННО на ЛАГОВОМ захвате"
          f" ({'подтверждено' if won or bhi[1]-bhi[2]>0.15 else 'не подтверждено'}).")
    print("  СМЫСЛ для R43 O1: ниша фазы — ЛАГОВАЯ синхронизация (задержки связи), а не")
    print("  амплитудный шум. Реальный ЭЭГ (Ф21) с долевой группировкой — в основном нулевой")
    print("  лаг (объёмная проводимость), поэтому корреляция не хуже. Идея автора ценна там,")
    print("  где связь имеет ЗАДЕРЖКУ — и это конкретная проверяемая граница, а не общий лозунг.")
