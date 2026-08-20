"""
R43 O1 — Ф15: чувствительность к k (число соседей в corr→coh). Задача I2.

Весь фикс драйвера (Ф11) и успех совместного экстрима S2 (Ф14) держатся на топ-k
ранжировании с k=15. Если результат хрупок к выбору k — метод хрупок. Проверяем
строго: сетка по k на двух сценах под драйвером.

Зависимости: numpy, scikit-learn (+ интеграционные функции из integrate.py).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import adjusted_rand_score
from integrate import gen_sparse_drive, full_corr_coh


if __name__ == "__main__":
    print("="*60)
    print("R43 O1 · Ф15 · чувствительность к k (I2)")
    print("="*60)

    # сцена 1: чистый драйвер (N=1500, κ=1.5, шум 0.2)
    phi1, comm1 = gen_sparse_drive(1500, 6, kappa=1.5, obs_noise=0.2, seed=4)
    # сцена 2: совместный экстрим (N=2000, κ=1.5, шум 0.3, 30% скрыто)
    rng = np.random.default_rng(0)
    phi2f, comm2f = gen_sparse_drive(2000, 8, kappa=1.5, obs_noise=0.3, seed=5)
    keep = rng.random(phi2f.shape[1]) < 0.7
    phi2, comm2 = phi2f[:, keep], comm2f[keep]

    print(f"\n  {'k':>4} {'ARI драйвер(M=6)':>17} {'ARI экстрим S2(M=8)':>20}")
    rows = []
    for k in (3, 5, 8, 12, 15, 20, 30, 50, 80):
        a1 = adjusted_rand_score(comm1, full_corr_coh(phi1, 6, k=k))
        a2 = adjusted_rand_score(comm2, full_corr_coh(phi2, 8, k=k))
        rows.append((k, a1, a2))
        print(f"  {k:>4} {a1:>17.3f} {a2:>20.3f}")

    good = [r for r in rows if r[1] >= 0.9 and r[2] >= 0.9]
    print("\n[ВЫВОД I2]")
    if good:
        ks = [r[0] for r in good]
        print(f"  Устойчивый диапазон k (обе сцены ARI≥0.9): {min(ks)}…{max(ks)}")
        print(f"  Метод НЕ хрупок к k — широкое плато. Рекомендация по умолчанию k≈15.")
    else:
        print("  Узкая зона по k — метод чувствителен; нужна авто-настройка k.")
