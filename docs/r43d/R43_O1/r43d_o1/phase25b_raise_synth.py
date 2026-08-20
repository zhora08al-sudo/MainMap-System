"""
R43 O1 — Ф25b: попытка ПОДНЯТЬ остаточный трек (Ф24) — ЧЕСТНЫЙ отрицательный результат.

Две идеи «подъёма» проверены на синтетике, где вторичный слой ЕСТЬ по построению:
  • СИЛЬНАЯ дефляция: комплексная проекция (убрать общий временной компонент целиком),
    вместо снятия только средней фазы;
  • ИТЕРАТИВНАЯ (EM-подобная) дефляция: чередовать оценку слоёв 1 и 2.
Вывод: НИ ОДНА не подняла абсолют в рабочем режиме (слабый вторичный слой). Простой
однопроходный метод по средней фазе — лучший; узкое место = SNR вторичного слоя.
Зависимости: numpy scikit-learn (+ phase24_residual, phase25_eeg_layers).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import adjusted_rand_score
from phase24_residual import make_layered, lag_corr_coh, residual_phase
from phase25_eeg_layers import residual_proj, residual_iter

if __name__ == "__main__":
    print("="*70)
    print("R43 O1 · Ф25b · попытка ПОДНЯТЬ остаточный трек (синтетика, слой ЕСТЬ)")
    print("="*70)
    print(f"\n  {'As':>5} | {'1проход(ср.фаза)':>17} {'1проход(проекция)':>17} {'итеративный':>12}")
    print("  " + "-"*56)
    rows = []
    for As in [0.45, 0.60, 0.75, 0.90]:
        sm, sp, si = [], [], []
        for r in range(6):
            phi, P, S = make_layered(As=As, seed=r)
            lay1 = lag_corr_coh(phi, 4)
            sm.append(adjusted_rand_score(S, lag_corr_coh(residual_phase(phi, lay1), 3)))
            sp.append(adjusted_rand_score(S, lag_corr_coh(residual_proj(phi, lay1), 3)))
            _, lab2 = residual_iter(phi, 4, 3, iters=3)
            si.append(adjusted_rand_score(S, lab2))
        rows.append((As, np.mean(sm), np.mean(sp), np.mean(si)))
        print(f"  {As:>5.2f} | {np.mean(sm):>17.3f} {np.mean(sp):>17.3f} {np.mean(si):>12.3f}")

    print("\n[ЧЕСТНЫЙ ВЫВОД]")
    print("  В рабочем режиме (As≤0.6, где остаточный трек и нужен) простой 1-проход по средней")
    print("  фазе ЛУЧШЕ и проекции, и итерации. Итерация деградирует: дефляция по шумному слою-2")
    print("  портит слой-1. «Подъём» НЕ достигнут — узкое место это SNR вторичного слоя, не дефляция.")
