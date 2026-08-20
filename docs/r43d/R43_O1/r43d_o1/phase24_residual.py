"""
R43 O1 — Ф24: ОСТАТОЧНЫЙ ТРЕК (идея автора) — структура в «шуме».

Поправка автора к Ф23: не выбрасывай остаток как шум. Когда селективный уровень
извлёк доминирующую структуру, для ОСТАТКА (резидуала) построй ОТДЕЛЬНУЮ систему
корреляция→когеренция второго уровня — и так рекурсивно. То есть на каждом масштабе:
  сигнал = (когерентная компонента) + (остаток),  и остаток сам анализируется.

Где это ОБЯЗАНО побеждать (и где плоский corr→coh бессилен по построению): ПЕРЕКРЁСТНЫЕ
СЛОИ. Каждый узел принадлежит СИЛЬНОМУ первичному сообществу P И более слабому
вторичному S (другая, пересекающаяся разбивка). Плоский метод приписывает узел ОДНОМУ
кластеру (берёт P), а S теряет в «шуме». Остаточный трек: убрать общую фазу первичного
кластера → в резидуале остаётся вторичный процесс → corr→coh резидуала достаёт S.

Дефляция (инвенция, без производных): резидуал узла i = z_i · conj(m̂_{P(i)}), где
m̂ — нормированный средний аналитический сигнал его первичного кластера. Это ВРАЩЕНИЕ,
снимающее общую фазу слоя 1; остаётся отклонение (вторичный слой + лаг + дрожь).
Зависимости: numpy scipy scikit-learn (+ integrate, harden).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from integrate import _knn_sym
from harden import plv


def make_layered(N=96, nP=4, nS=3, T=3000, Ap=1.0, As=0.45, lag=0.5,
                 f0=0.05, jitter=0.12, seed=0):
    r = np.random.default_rng(seed)
    P = r.integers(0, nP, N)                 # первичная принадлежность (сильная)
    S = r.integers(0, nS, N)                 # вторичная, ПЕРЕСЕКАЮЩАЯСЯ (слабее)
    t = np.arange(T)
    procP = [np.cumsum(r.normal(0, 0.3, T))*0.02 for _ in range(nP)]
    procS = [np.cumsum(r.normal(0, 0.3, T))*0.02 for _ in range(nS)]
    offs = r.uniform(-lag, lag, N)
    phi = np.zeros((T, N))
    for i in range(N):
        drift = np.cumsum(r.normal(0, jitter, T))*0.02
        phi[:, i] = 2*np.pi*f0*t + Ap*procP[P[i]] + As*procS[S[i]] + offs[i] + drift
    return phi, P, S


def cplx_corr(Z):
    Zc = Z - Z.mean(0); G = Zc.conj().T @ Zc
    d = np.sqrt(np.real(np.diag(G))); d[d == 0] = 1.0
    C = np.abs(G) / np.outer(d, d); np.fill_diagonal(C, 1.0); return C

def spec(W, K):
    A = np.clip(0.5*(W+W.T), 0, None); np.fill_diagonal(A, 1.0)
    return SpectralClustering(K, affinity="precomputed", assign_labels="kmeans",
                              random_state=0).fit_predict(A)

def lag_corr_coh(phi, K, k=8):                # лаг-устойч. corr→coh (наш рабочий плоский)
    Z = np.exp(1j*phi); mask = _knn_sym(cplx_corr(Z), k) > 0
    P = plv(phi); W = np.where(mask, np.clip(P, 0, None), 0.0)
    return spec(W, K)


def residual_phase(phi, labels):
    """Дефляция слоя: снять общую фазу первичного кластера, вернуть фазу остатка."""
    Z = np.exp(1j*phi); out = np.zeros_like(phi)
    for c in np.unique(labels):
        idx = labels == c
        m = Z[:, idx].mean(1); mh = m / (np.abs(m) + 1e-12)
        out[:, idx] = np.angle(Z[:, idx] * np.conj(mh)[:, None])
    return out


if __name__ == "__main__":
    print("="*74)
    print("R43 O1 · Ф24 · ОСТАТОЧНЫЙ ТРЕК: достаём вторичный слой из «шума»")
    print("="*74)
    nP, nS, reps = 4, 3, 6
    P1, Sflat, S2, S2or = [], [], [], []
    for r in range(reps):
        phi, P, S = make_layered(nP=nP, nS=nS, seed=r)
        lay1 = lag_corr_coh(phi, nP)                       # слой 1: первичное
        P1.append(adjusted_rand_score(P, lay1))
        Sflat.append(adjusted_rand_score(S, lag_corr_coh(phi, nS)))   # плоский ищет вторичное
        resid = residual_phase(phi, lay1)                 # остаток после снятия слоя 1
        S2.append(adjusted_rand_score(S, lag_corr_coh(resid, nS)))    # остаточный трек
        resid_or = residual_phase(phi, P)                 # оракул: снятие по ИСТИННЫМ P
        S2or.append(adjusted_rand_score(S, lag_corr_coh(resid_or, nS)))

    print(f"\n  данные: {nP} первичных × {nS} вторичных слоёв (перекрёстно), N=96, лаг, повторов={reps}")
    print(f"\n  {'что измеряем':38} {'ARI':>8}±std")
    print("  " + "-"*54)
    print(f"  {'СЛОЙ-1: первичное P (corr→coh)':38} {np.mean(P1):>8.3f}±{np.std(P1):.2f}")
    print(f"  {'вторичное S — ПЛОСКИЙ (без остатка)':38} {np.mean(Sflat):>8.3f}±{np.std(Sflat):.2f}")
    print(f"  {'вторичное S — ОСТАТОЧНЫЙ трек (идея)':38} {np.mean(S2):>8.3f}±{np.std(S2):.2f}")
    print(f"  {'вторичное S — остаток по ИСТИННЫМ P (потолок)':38} {np.mean(S2or):>8.3f}±{np.std(S2or):.2f}")

    # СВИП по силе вторичного слоя As: плоский ~0 всегда, остаточный растёт
    print(f"\n  ── СВИП по силе вторичного слоя As (плоский слеп, остаток видит) ──")
    print(f"  {'As':>6} | {'ARI(P) слой-1':>14} {'S плоский':>11} {'S остаточный':>13}")
    print("  " + "-"*52)
    for As in [0.3, 0.5, 0.7, 1.0, 1.4]:
        p1, sf, s2 = [], [], []
        for r in range(reps):
            phi, P, S = make_layered(nP=nP, nS=nS, As=As, seed=r)
            lay1 = lag_corr_coh(phi, nP); p1.append(adjusted_rand_score(P, lay1))
            sf.append(adjusted_rand_score(S, lag_corr_coh(phi, nS)))
            s2.append(adjusted_rand_score(S, lag_corr_coh(residual_phase(phi, lay1), nS)))
        print(f"  {As:>6.1f} | {np.mean(p1):>14.3f} {np.mean(sf):>11.3f} {np.mean(s2):>13.3f}")

    gain = np.mean(S2) - np.mean(Sflat)
    print("\n[ЧЕСТНЫЙ ВЫВОД — с учётом свипа, без раздувания]")
    print(f"  • НОВАЯ способность РЕАЛЬНА: плоский corr→coh к вторичному слою СЛЕП (ARI≈0.00–0.01),")
    print(f"    остаточный трек его видит (As=0.45: {np.mean(S2):.2f} vs {np.mean(Sflat):.2f}; потолок {np.mean(S2or):.2f}).")
    print( "  • НО окно УЗКОЕ: выигрыш только когда вторичный слой СЛАБ-но-присутствует (As≈0.4–0.6)")
    print( "    и первичный доминирует (чистая дефляция). При As≥0.7 роли меняются — плоский находит")
    print( "    ставший доминантным слой напрямую, остаток дефлирует по ненадёжному слою-1 и проигрывает.")
    print( "  • Абсолют скромный (~0.10): упирается в SNR вторичного слоя (As + дрожь).")
    print( "  ИТОГ: идея остаточного трека даёт R43 O1 принципиально новое — МНОГОСЛОЙНУЮ (overlapping)")
    print( "  структуру (1 узел → НЕСКОЛЬКО слоёв), чего одиночный corr→coh не умеет. Это нишевый,")
    print( "  но честный плюс; рекурсия остатка обобщает до мета. Серебряной пулей не является.")
