"""
R43 O1 — СТЕНД ПРОТИВ КЛАССИКИ (Ф4, критерий SC5)

Операциональный смысл «замены ОДУ» из файлов автора: «вместо dy/dt», «устраняет
ошибки Piecewise». Проверяем это ЧЕСТНО, голова-в-голову, на одной системе.

Система: Курамото из N осцилляторов с PIECEWISE-разрывом связи:
    dφ_i/dt = ω_i + (K(t)/N)·Σ_j sin(φ_j − φ_i),   K(t)= K1 (t<t*) | K2 (t≥t*)
Разрыв в правой части на t=t* — ровно то, на чём спотыкается наивное интегрирование.

Сравниваем по r(t) (параметр порядка) против ЭТАЛОНА:
  ЭТАЛОН   solve_ivp, RTOL=1e-11, с КОРРЕКТНЫМ разбиением интервала на разрыве
  M1       наивный solve_ivp по всему интервалу (разрыв не объявлен), RTOL=1e-3
  M2       наивный solve_ivp по всему интервалу, RTOL=1e-6
  R43 O1   дискретное derivative-free отображение (forward step), фикс. dt

Метрики: max|Δr|, RMS|Δr|, стоимость = число вычислений правой части (nfev / шаги).
Это не «ОДУ устарели». Это честный замер ниши: где у адаптивного решателя растёт
цена/падает точность на разрыве, а дискретный шаг проходит ровно и предсказуемо.

Зависимости: numpy, scipy.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(7)
N = 60
omega = rng.normal(0.0, 1.0, N)
phi0 = rng.uniform(-np.pi, np.pi, N)
T, t_star = 20.0, 10.0
K1, K2 = 0.4, 3.0                      # разрыв: рассинхрон -> синхрон
grid = np.linspace(0, T, 401)         # общая сетка оценки r(t)


def Kt(t): return K1 if t < t_star else K2

def rhs(t, phi, K):
    coupling = np.sin(phi[None, :] - phi[:, None]).mean(axis=1)
    return omega + K * coupling

def order_r(phi_mat):
    # phi_mat: (steps, N) -> r(t)
    return np.abs(np.exp(1j * phi_mat).mean(axis=1))


# ---------- ЭТАЛОН: корректное разбиение на разрыве, сверх-точно ----------
def reference():
    s1 = solve_ivp(rhs, (0, t_star), phi0, args=(K1,), rtol=1e-11, atol=1e-12,
                   dense_output=True, method="RK45")
    s2 = solve_ivp(rhs, (t_star, T), s1.y[:, -1], args=(K2,), rtol=1e-11, atol=1e-12,
                   dense_output=True, method="RK45")
    phi = np.empty((len(grid), N))
    for i, t in enumerate(grid):
        phi[i] = s1.sol(t) if t < t_star else s2.sol(t)
    return order_r(phi), s1.nfev + s2.nfev


# ---------- M1/M2: наивный solve_ivp по всему интервалу (разрыв НЕ объявлен) ----------
def naive_ode(rtol):
    def f(t, phi): return rhs(t, phi, Kt(t))
    s = solve_ivp(f, (0, T), phi0, rtol=rtol, atol=rtol*1e-2,
                  dense_output=True, method="RK45")
    phi = np.array([s.sol(t) for t in grid])
    return order_r(phi), s.nfev


# ---------- R43 O1: дискретное derivative-free отображение ----------
def r43o1(dt):
    steps = int(round(T / dt))
    phi = phi0.copy()
    rec_t, rec_phi = [0.0], [phi.copy()]
    t = 0.0
    for k in range(steps):
        K = Kt(t)
        coupling = np.sin(phi[None, :] - phi[:, None]).mean(axis=1)   # без d/dt
        phi = phi + dt * (omega + K * coupling)
        t += dt
        rec_t.append(t); rec_phi.append(phi.copy())
    rec_t = np.array(rec_t); rec_phi = np.array(rec_phi)
    # интерполяция фаз на общую сетку (по комплексной экспоненте, чтобы не рвать на 2π)
    z = np.exp(1j * rec_phi)
    zr = np.empty((len(grid), N), complex)
    for j in range(N):
        zr[:, j] = np.interp(grid, rec_t, z[:, j].real) + 1j*np.interp(grid, rec_t, z[:, j].imag)
    return np.abs(zr.mean(axis=1)), steps   # стоимость = число шагов (1 rhs-экв./шаг)


if __name__ == "__main__":
    print("="*70)
    print("R43 O1 · Ф4 · голова-в-голову с ОДУ-решателем (Piecewise-разрыв)")
    print("="*70)
    print(f"Система: Курамото N={N}, разрыв связи K {K1}->{K2} на t={t_star}")

    r_ref, nfev_ref = reference()

    results = []
    for name, (r, cost) in {
        "M1 наивный solve_ivp rtol=1e-3": naive_ode(1e-3),
        "M2 наивный solve_ivp rtol=1e-6": naive_ode(1e-6),
        "R43 O1 dt=0.05":                 r43o1(0.05),
        "R43 O1 dt=0.02":                 r43o1(0.02),
        "R43 O1 dt=0.01":                 r43o1(0.01),
    }.items():
        err = np.abs(r - r_ref)
        results.append((name, err.max(), np.sqrt((err**2).mean()), cost))

    print(f"\nЭталон: solve_ivp с разбиением на разрыве, nfev={nfev_ref}")
    print(f"\n{'метод':34} {'max|Δr|':>9} {'RMS|Δr|':>9} {'стоимость(вызовы/шаги)':>22}")
    for name, mx, rms, cost in results:
        print(f"{name:34} {mx:>9.4f} {rms:>9.4f} {cost:>22d}")

    # честный разбор — управляется числами, без приукрашивания
    m1 = next(r for r in results if r[0].startswith("M1"))
    m2 = next(r for r in results if r[0].startswith("M2"))
    o05 = next(r for r in results if r[0]=="R43 O1 dt=0.05")
    o01 = next(r for r in results if r[0]=="R43 O1 dt=0.01")
    # «победа» = быть не хуже по точности при не большей цене
    ode_better_acc = m1[1] <= o05[1]
    ode_better_cost = m1[3] <= o01[3]
    sc5_pass = not (ode_better_acc and ode_better_cost)
    print("\n[ЧЕСТНЫЙ ВЫВОД — без приукрашивания]")
    print(f"  solve_ivp rtol=1e-3: max|Δr|={m1[1]:.4f}, цена={m1[3]} вызовов")
    print(f"  solve_ivp rtol=1e-6: max|Δr|={m2[1]:.4f}, цена={m2[3]} вызовов (практически точно)")
    print(f"  R43 O1 dt=0.05:      max|Δr|={o05[1]:.4f}, цена={o05[3]} шагов")
    print(f"  R43 O1 dt=0.01:      max|Δr|={o01[1]:.4f}, цена={o01[3]} шагов")
    print(f"\n  SC5 (R43 O1 не хуже ОДУ-решателя на этом бенчмарке): {'PASS' if sc5_pass else 'FAIL'}")
    if not sc5_pass:
        print("  РЕЗУЛЬТАТ ОТРИЦАТЕЛЬНЫЙ: на простом Piecewise-разрыве коэффициента адаптивный")
        print("  solve_ivp ТОЧНЕЕ и ДЕШЕВЛЕ, чем R43 O1. Гипотеза «ОДУ ломается на Piecewise»")
        print("  здесь НЕ подтвердилась: скачок коэффициента — мягкий разрыв, RK45 проходит его")
        print("  дёшево. Тезис «замены» этим тестом НЕ доказан.")
        print("  Настоящие тяжёлые для ОДУ режимы — жёсткость (stiff), событийная локализация")
        print("  состояния, Zeno/чаттеринг. Их и надо проверять, либо честно сузить миссию")
        print("  R43 O1 до подтверждённой сильной стороны — извлечение структуры из хаоса (P3).")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4.5))
        plt.plot(grid, r_ref, "k-", lw=2.5, label="эталон (разбиение)")
        r_m1,_ = naive_ode(1e-3); plt.plot(grid, r_m1, "--", label="наивный solve_ivp 1e-3")
        r_o,_ = r43o1(0.05);        plt.plot(grid, r_o, ":", lw=2, label="R43 O1 dt=0.05")
        plt.axvline(t_star, color="r", ls="--", alpha=.5, label="Piecewise-разрыв")
        plt.xlabel("t"); plt.ylabel("r(t)"); plt.title("R43 O1 vs ОДУ на разрыве связи")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("benchmark.png", dpi=90)
        print("  график сохранён: benchmark.png")
    except Exception as e:
        print(f"  (график пропущён: {e})")
