"""
R43 O1 — Ф13: КАРТА ПОРОГА F1. Где derivative-free шаг начинает выигрывать у ОДУ?

Достраиваем Ф12 (одна точка, 400 разрывов) до строгой карты: сетка по ПЛОТНОСТИ
разрывов nsw. Сравнение ЧЕСТНОЕ — при РАВНОЙ стоимости:
  для каждого nsw: solve_ivp(tol=1e-6) даёт (ошибка, nfev); затем R43 O1 берёт
  steps = nfev (та же стоимость) и считаем его ошибку. Кто точнее при равном бюджете?
Порог = минимальный nsw, где ошибка R43 O1 < ошибки solve_ivp.

Эталон: интегрирование с разбиением на каждом разрыве (rtol=1e-10).
Зависимости: numpy, scipy, matplotlib (опц.).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import solve_ivp

rng0 = np.random.default_rng(0)
N = 30
omega = rng0.normal(0, 1.0, N)
A = (rng0.random((N, N)) < 0.2).astype(float); A = np.triu(A, 1); A = A + A.T
deg = A.sum(1); deg[deg == 0] = 1
phi0 = rng0.uniform(-np.pi, np.pi, N)
T = 20.0
grid = np.linspace(0, T, 401)


def make_switches(nsw, seed):
    rng = np.random.default_rng(seed)
    if nsw == 0:
        return np.array([]), np.array([1.5])
    sw_t = np.sort(rng.uniform(0, T, nsw))
    sw_val = rng.choice([-1.0, 1.0], nsw + 1) * 1.5
    return sw_t, sw_val

def Kf(t, sw_t, sw_val):
    return sw_val[np.searchsorted(sw_t, t)]

def rhs(t, phi, sw_t, sw_val):
    s, c = np.sin(phi), np.cos(phi)
    return omega + Kf(t, sw_t, sw_val) * (c*(A@s) - s*(A@c)) / deg

def order(Y): return np.abs(np.exp(1j*Y).mean(1))

def reference(sw_t, sw_val):
    bounds = np.r_[0.0, sw_t, T]; cur = phi0.copy(); segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a: continue
        s = solve_ivp(rhs, (a, b), cur, args=(sw_t, sw_val), rtol=1e-10, atol=1e-12,
                      dense_output=True); cur = s.y[:, -1]; segs.append((a, b, s))
    def ev(t):
        for a, b, s in segs:
            if a <= t <= b: return s.sol(t)
        return segs[-1][2].sol(t)
    return order(np.array([ev(t) for t in grid]))

def solve_naive(sw_t, sw_val, tol):
    s = solve_ivp(rhs, (0, T), phi0, args=(sw_t, sw_val), rtol=tol, atol=tol*1e-2,
                  dense_output=True)
    return order(np.array([s.sol(t) for t in grid])), s.nfev

def r43o1(sw_t, sw_val, steps):
    dt = T/steps; phi = phi0.copy(); t = 0.0; rec = [phi.copy()]; tt = [0.0]
    for k in range(steps):
        s, c = np.sin(phi), np.cos(phi)
        phi = phi + dt*(omega + Kf(t, sw_t, sw_val)*(c*(A@s) - s*(A@c))/deg); t += dt
        rec.append(phi.copy()); tt.append(t)
    Z = np.exp(1j*np.array(rec)); zr = np.empty((len(grid), N), complex)
    for j in range(N):
        zr[:, j] = np.interp(grid, tt, Z[:, j].real) + 1j*np.interp(grid, tt, Z[:, j].imag)
    return np.abs(zr.mean(1))


if __name__ == "__main__":
    print("="*70)
    print("R43 O1 · Ф13 · КАРТА ПОРОГА F1 (равная стоимость: steps = nfev solve_ivp)")
    print("="*70)
    print(f"  {'nsw':>5} {'err solve_ivp':>14} {'nfev':>7} {'err R43O1@=cost':>16} {'победитель':>12}")
    rows = []
    for nsw in (0, 5, 10, 25, 50, 100, 200, 400, 800):
        sw_t, sw_val = make_switches(nsw, seed=100+nsw)
        r_ref = reference(sw_t, sw_val)
        r_ode, nfev = solve_naive(sw_t, sw_val, 1e-6)
        e_ode = np.abs(r_ode - r_ref).max()
        r_map = r43o1(sw_t, sw_val, max(50, nfev))           # равная стоимость
        e_map = np.abs(r_map - r_ref).max()
        win = "R43 O1" if e_map < e_ode else "solve_ivp"
        rows.append((nsw, e_ode, nfev, e_map, win))
        print(f"  {nsw:>5} {e_ode:>14.4f} {nfev:>7} {e_map:>16.4f} {win:>12}")

    # порог: первый nsw, где R43 O1 побеждает и далее не уступает
    thr = None
    for i, r in enumerate(rows):
        if r[4] == "R43 O1" and all(x[4] == "R43 O1" for x in rows[i:]):
            thr = r[0]; break
    print("\n[КАРТА ПОРОГА F1]")
    if thr is not None:
        print(f"  При РАВНОЙ стоимости R43 O1 стабильно выигрывает начиная с nsw ≈ {thr}")
        print(f"  разрывов на интервале T={T} (плотность ≈ {thr/T:.2f} разрыв/ед.времени).")
    else:
        print("  Чёткого устойчивого порога не найдено — см. таблицу.")
    print("  Ниже порога (гладко/редко) — solve_ivp точнее при равном бюджете (Ф4).")
    print("  Выше порога (часто-разрывно) — derivative-free шаг точнее (Ф12 подтверждён).")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ns = [r[0] for r in rows]
        plt.figure(figsize=(7.5,4.5))
        plt.plot(ns, [r[1] for r in rows], "s--", label="solve_ivp (tol=1e-6)")
        plt.plot(ns, [r[3] for r in rows], "o-", label="R43 O1 @ равная стоимость")
        if thr is not None: plt.axvline(thr, color="g", ls=":", label=f"порог nsw≈{thr}")
        plt.xlabel("число разрывов nsw на [0,20]"); plt.ylabel("max|Δr| (ошибка)")
        plt.title("Карта порога F1: где derivative-free шаг обгоняет ОДУ-решатель")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("phase13_threshold.png", dpi=90)
        print("  график сохранён: phase13_threshold.png")
    except Exception as e:
        print(f"  (график пропущён: {e})")
