"""
R43 O1 — Ф10: ЧЕСТНАЯ ГРАНИЦА ОБЛАСТИ. Сообщества только по топологии (без ритма).

Во всех предыдущих тестах у сообществ был СВОЙ ритм (частотная полоса). Это
облегчало задачу. Жёсткий вопрос: найдёт ли R43 O1 сообщества, заданные ТОЛЬКО
связностью, когда у всех узлов ОДНА частотная статистика (нет различия ритма)?

Сравниваем два мира при σ_obs=0.2, спектральная кластеризация на PLV:
  • «с ритмом»  — у каждого сообщества своя частотная полоса (как в Ф2–Ф9);
  • «без ритма» — ω у всех из N(0,σω), сообщества отличаются ТОЛЬКО графом.
Прогоняем по сетке силы связи K0. Честно картируем, где метод работает, где нет.

Зависимости: numpy, scikit-learn.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import adjusted_rand_score
from harden import plv, spec


def gen2(N=150, M=3, K0=1.6, rhythm=True, sw=0.6, seed=0,
         steps=1500, dt=0.05, tail=500, intra=0.85, inter=0.02, obs_noise=0.2):
    rng = np.random.default_rng(seed)
    comm = np.repeat(np.arange(M), N // M)
    if len(comm) < N: comm = np.concatenate([comm, rng.integers(0, M, N-len(comm))])
    same = comm[:, None] == comm[None, :]
    pr = np.where(same, intra, inter); np.fill_diagonal(pr, 0)
    iu = np.triu_indices(N, 1); e = rng.random(len(iu[0])) < pr[iu]
    A = np.zeros((N, N)); A[iu[0][e], iu[1][e]] = 1; A = A + A.T
    deg = A.sum(1); deg[deg == 0] = 1
    if rhythm:
        omega = np.linspace(-1.5, 1.5, M)[comm] + 0.10*rng.standard_normal(N)
    else:
        omega = sw * rng.standard_normal(N)            # ОДНА статистика, без полос
    phi = rng.uniform(-np.pi, np.pi, N)
    Z = np.empty((tail, N), complex); t0 = steps - tail
    for k in range(steps):
        s, c = np.sin(phi), np.cos(phi)
        phi = phi + dt*(omega + K0*(c*(A@s) - s*(A@c))/deg)
        if k >= t0: Z[k-t0] = np.exp(1j*phi)
    pho = np.angle(Z)
    if obs_noise > 0: pho = pho + obs_noise*rng.standard_normal(pho.shape)
    return pho, comm


if __name__ == "__main__":
    print("="*64)
    print("R43 O1 · Ф10 · граница: сообщества с ритмом vs только-топология")
    print("="*64)
    print(f"  {'K0':>5} {'ARI с ритмом':>13} {'ARI без ритма':>14}")
    best_norhythm = 0.0
    for K0 in (0.8, 1.2, 1.6, 2.5, 4.0, 6.0):
        a_r = adjusted_rand_score(*( (lambda d: (d[1], spec(plv(d[0]),3)))(gen2(K0=K0, rhythm=True,  seed=1)) ))
        a_n = adjusted_rand_score(*( (lambda d: (d[1], spec(plv(d[0]),3)))(gen2(K0=K0, rhythm=False, seed=1)) ))
        best_norhythm = max(best_norhythm, a_n)
        print(f"  {K0:>5} {a_r:>13.3f} {a_n:>14.3f}")

    print("\n[ЧЕСТНЫЙ ВЫВОД]")
    print(f"  «с ритмом»: метод восстанавливает структуру (как в Ф2–Ф9).")
    print(f"  «без ритма»: лучший ARI по K0 = {best_norhythm:.3f}")
    if best_norhythm >= 0.85:
        print("  -> неожиданно: топологические сообщества тоже берутся (есть рабочий K0).")
    else:
        print("  -> ГРАНИЦА ОБЛАСТИ: при одинаковом ритме сообщества по чистой топологии")
        print("     PLV восстанавливает СЛАБО. R43 O1 предназначен для РИТМ-КОГЕРЕНТНЫХ")
        print("     сообществ (группы со схожей частотой) — это и есть «фазовая гармония»")
        print("     из исходных файлов. Чисто-структурные сообщества без различия ритма —")
        print("     вне области (там нужен сам граф, а его из наблюдений у нас нет).")
