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
from integrate import full_corr_coh, _knn_sym, plv   # метод автора corr→coh + утилиты

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


# ── метод автора: СНАЧАЛА корреляция (отбор), ПОТОМ когеренция (вес) ──
def corr_coh_asis(s, K, k=8):
    """corr→coh КАК ЕСТЬ (Ф11/Ф16): отбор соседей по нуль-лаговой корр cos(φ), вес PLV."""
    return full_corr_coh(inst_phase(s), K, k)


def lag_corr_mat(s):
    """ЛАГ-устойчивая корреляция = |комплексная корреляция аналитического сигнала|.
    Инвариантна к постоянному фазовому сдвигу (нуль-лаговая corr cos этим свойством НЕ
    обладает). Это и есть «когеренция, проявленная как корреляция» — сохраняем замысел
    автора (корреляция-первой), но делаем её лаг-инвариантной."""
    z = hilbert(s, axis=0); z = z - z.mean(0)
    num = np.abs(z.conj().T @ z)
    nrm = np.sqrt(np.sum(np.abs(z)**2, 0)); nrm[nrm == 0] = 1.0
    C = num / np.outer(nrm, nrm); np.fill_diagonal(C, 1.0)
    return C


def corr_coh_lag(s, K, k=8):
    """corr→coh ЛАГ-УСТОЙЧИВЫЙ: отбор соседей по лаг-инвариантной корреляции, вес PLV.
    Тот же двухступенчатый замысел автора, но первая ступень переживает фазовый лаг."""
    ph = inst_phase(s)
    mask = _knn_sym(lag_corr_mat(s), k) > 0
    P = plv(ph); W = np.where(mask, np.clip(P, 0, None), 0.0); np.fill_diagonal(W, 1.0)
    return SpectralClustering(K, affinity="precomputed", random_state=0).fit_predict(W)


def sweep(M, axis, vals, reps=5, **fixed):
    # 5 методов: чистая фаза | чистая корр cos | corr→coh КАК ЕСТЬ | corr→coh ЛАГ-устойч
    print(f"\n  {axis:>8} | {'PLV':>7} {'corr cos':>9} {'corr→coh(как есть)':>19} {'corr→coh(лаг)':>14}")
    print("  " + "-"*72)
    out = []
    for v in vals:
        a_plv, a_cos, a_a, a_l = [], [], [], []
        for r in range(reps):
            kw = dict(fixed); kw[axis] = v
            s, gt = make_signals(N=60, M=M, T=4000, **kw)
            a_plv.append(adjusted_rand_score(gt, plv_cluster(s, M)))
            a_cos.append(adjusted_rand_score(gt, corr_cos_cluster(s, M)))
            a_a.append(adjusted_rand_score(gt, corr_coh_asis(s, M)))
            a_l.append(adjusted_rand_score(gt, corr_coh_lag(s, M)))
        row = (v, np.mean(a_plv), np.mean(a_cos), np.mean(a_a), np.mean(a_l))
        print(f"  {v:>8.2f} | {row[1]:>7.3f} {row[2]:>9.3f} {row[3]:>19.3f} {row[4]:>14.3f}")
        out.append(row)
    return out


if __name__ == "__main__":
    print("="*72)
    print("R43 O1 · Ф22 · МЕХАНИЗМ: когда фаза (PLV) бьёт корреляцию")
    print("="*72)
    M = 3
    print(f"  N=60, сообществ={M}, повторов на точку=5")

    print("\n── ОСЬ A: амплитудный шум η (фазовый лаг = 0) ──")
    A = sweep(M, "eta", [0.0, 2.0, 4.0, 8.0], lag=0.0)

    print("\n── ОСЬ B: фазовый ЛАГ внутри сообщества (амплитуда чистая, η=0) ──")
    print("   ключевой тест замысла автора «сначала корреляция, потом когеренция»")
    B = sweep(M, "lag", [0.0, 0.4, 0.8, 1.2, 1.57], eta=0.0)

    print("\n[ЧЕСТНЫЙ ВЫВОД — про двухступенчатый метод автора corr→coh]")
    b0, bhi = B[0], B[-1]
    print(f"  ОСЬ B (лаг π/2): чистая корр cos рушится {b0[2]:.2f}→{bhi[2]:.2f}.")
    print(f"    • corr→coh КАК ЕСТЬ (отбор нуль-лаговой корр) ТОЖЕ падает: {b0[3]:.2f}→{bhi[3]:.2f}")
    print(f"      — подтверждает твою поправку: первая ступень (корреляция) под лагом ломает трубу.")
    print(f"    • corr→coh ЛАГ-устойчивый (отбор лаг-инвариантной корр → когеренция) держит:"
          f" {b0[4]:.2f}→{bhi[4]:.2f}.")
    print("  → Замысел «сначала корреляция, потом когеренция» ВЕРЕН, но корреляция-отбор обязана")
    print("    быть ЛАГ-ИНВАРИАНТНОЙ (|комплексная корр| аналитического сигнала). Тогда двухступка")
    print("    переживает задержанную связь, где простая корреляция и наивный corr→coh слепнут.")
    print("  ОСЬ A: амплитудный шум — не ниша фазы (Гильберт-фаза портится); это про лаг, не амплитуду.")
    print("  СВЯЗЬ с Ф21: ЭЭГ-доли ≈ нулевой лаг (объёмн. проводимость) → даже наивная корр работает;")
    print("    выигрыш ждать на ЗАДЕРЖАННОЙ связи, и там нужен ЛАГ-устойчивый отбор первой ступени.")
