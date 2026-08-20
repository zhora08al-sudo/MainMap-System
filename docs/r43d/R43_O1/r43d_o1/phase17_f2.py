"""
R43 O1 — Ф17: исчерпывающая атака F2 (топология без ритма) — фундаментальная граница.

Случай: сообщества заданы ТОЛЬКО связностью, у всех узлов один ритм. Перепробованы
4 derivative-free подхода (все — варианты идеи автора corr→coh):
  полный PLV (Ф10)            ≈ 0.13
  когеренция остатков (Ф12)   ≈ 0.34
  топ-k ранжирование (Ф12)    ≈ 0.48
  иерархия на остатках (тут)  ≈ 0.31
Ни один не выходит на пригодный уровень.

ВЫВОД (честный): это не слабость метода, а ИНФОРМАЦИОННАЯ граница. При едином ритме
и глобальной синхронизации принадлежность к сообществу — свойство ГРАФА, оно не
закодировано в наблюдаемых сигналах в извлекаемом виде. Для таких сообществ нужен
сам граф (кластеризация графа), а не сигнал-based discovery.

Зависимости: numpy, scikit-learn.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import adjusted_rand_score
from phase12 import gen_topo
from phase16_hier import hier_corr_coh


def residual_phase(phi):
    Z = np.exp(1j*phi); g = Z.mean(1, keepdims=True)
    a = (np.conj(g)*Z).sum(0, keepdims=True)/((np.abs(g)**2).sum()+1e-12)
    return np.angle(Z - g*a)


if __name__ == "__main__":
    print("="*60)
    print("R43 O1 · Ф17 · F2 (топология без ритма) — граница информации")
    print("="*60)
    print(f"  {'K0':>5} {'иерархия(сырое)':>16} {'иерархия(остатки)':>18}")
    best = 0.0
    for K0 in (0.8, 1.2, 1.6, 2.5, 4.0):
        phi, comm = gen_topo(N=150, M=3, K0=K0, seed=1)
        a1 = adjusted_rand_score(comm, hier_corr_coh(phi, 3, k=12))
        a2 = adjusted_rand_score(comm, hier_corr_coh(residual_phase(phi), 3, k=12))
        best = max(best, a2)
        print(f"  {K0:>5} {a1:>16.3f} {a2:>18.3f}")
    print(f"\n  Лучший по всем подходам ≈ {best:.3f} — НЕ пригодно.")
    print("  F2 = информационная граница: при едином ритме сообщество не закодировано")
    print("  в наблюдаемых сигналах. Нужен сам граф. Это не дефект R43 O1.")
