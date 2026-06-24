"""
R43 O1 — ЯДРО ДИНАМИКИ (Ф1, столп P1)

Дискретное, derivative-free, фазово-связанное отображение. Никакой оператор d/dt
НЕ вычисляется: эволюция задана рекуррентой (разностным шагом), а не интегрированием
производных. Это и есть требование автора «без производных» (категорически).

Закон шага (дискретный аналог фазовой связи):
    φ_i(k+1) = φ_i(k) + Δ·[ ω_i + (K/N)·Σ_j sin(φ_j − φ_i) ]

Параметр порядка (мера синхронности / «гармонии из хаоса»):
    r(k)·e^{iψ(k)} = (1/N)·Σ_j e^{iφ_j(k)},   r∈[0,1]

Критерии этой фазы:
    SC1 — derivative-free (статическая самопроверка исходника рантайма)
    переход синхронизации: r→0 при малом K, r→1 при K≫K_c
    K_c (теория, для ω~N(0,σ²)) = 2σ·sqrt(2/π)

Зависимости: numpy (matplotlib опционально для графика).
"""
from __future__ import annotations
import inspect
import numpy as np


# ----------------------------------------------------------------------
# ЯДРО (рантайм) — только алгебра, sin и суммы. Никаких производных.
# ----------------------------------------------------------------------
def kuramoto_step(phi: np.ndarray, omega: np.ndarray, K: float, dt: float) -> np.ndarray:
    """Один дискретный шаг фазовой связи. Возвращает φ(k+1)."""
    # матрица разностей фаз φ_j − φ_i, связь через sin
    coupling = np.sin(phi[None, :] - phi[:, None]).mean(axis=1)   # (1/N)Σ_j sin(φ_j−φ_i)
    return phi + dt * (omega + K * coupling)


def order_parameter(phi: np.ndarray) -> tuple[float, float]:
    """Параметр порядка Курамото: r (синхронность) и ψ (средняя фаза)."""
    z = np.exp(1j * phi).mean()
    return float(np.abs(z)), float(np.angle(z))


def simulate(N=500, K=2.0, steps=2000, dt=0.05, sigma=1.0, seed=0, burn=0.5):
    """Прогон ядра. Возвращает dict с финальным r (усреднён по хвосту) и историей r."""
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, sigma, N)            # собственные частоты ~ N(0,σ²)
    phi = rng.uniform(-np.pi, np.pi, N)          # старт из «хаоса»
    r_hist = np.empty(steps)
    for k in range(steps):
        phi = kuramoto_step(phi, omega, K, dt)
        r_hist[k], _ = order_parameter(phi)
    tail = slice(int(burn * steps), None)
    return {"r_final": float(r_hist[tail].mean()),
            "r_std": float(r_hist[tail].std()),
            "r_hist": r_hist, "phi": phi, "omega": omega}


def critical_coupling(sigma=1.0) -> float:
    """K_c для ω~N(0,σ²):  K_c = 2/(π·g(0)),  g(0)=1/(σ√(2π))  =>  K_c = 2σ·sqrt(2/π)."""
    return 2.0 * sigma * np.sqrt(2.0 / np.pi)


def sweep_K(K_values, N=500, steps=2000, dt=0.05, sigma=1.0, seed=0):
    """Кривая r(K): для каждого K возвращает установившийся r."""
    return np.array([simulate(N=N, K=K, steps=steps, dt=dt, sigma=sigma, seed=seed)["r_final"]
                     for K in K_values])


# ----------------------------------------------------------------------
# SC1 — статическая самопроверка: ноль производных в рантайме
# ----------------------------------------------------------------------
def check_derivative_free() -> bool:
    """Сканирует исходники рантайм-функций на запрещённые операции дифференцирования."""
    forbidden = ("np.gradient", "gradient(", "np.diff", ".diff(",
                 "derivative", "d2", "d/dt", "/dt", "finite_difference")
    runtime = (kuramoto_step, order_parameter, simulate, sweep_K)
    hits = []
    for fn in runtime:
        src = inspect.getsource(fn)
        for tok in forbidden:
            if tok in src:
                hits.append((fn.__name__, tok))
    if hits:
        print("  SC1 НАРУШЕН:", hits); return False
    print("  SC1: в рантайме нет операторов производной (gradient/diff/d/dt) — derivative-free")
    return True


# ----------------------------------------------------------------------
# Демонстрация / приёмка Ф1
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("="*64)
    print("R43 O1 · Ф1 · ядро динамики — приёмка")
    print("="*64)

    print("\n[SC1] derivative-free самопроверка:")
    sc1 = check_derivative_free()

    sigma = 1.0
    Kc = critical_coupling(sigma)
    print(f"\n[Переход синхронизации] теоретический K_c (σ={sigma}) = {Kc:.3f}")
    Ks = np.linspace(0.0, 4.0, 13)
    r = sweep_K(Ks, N=300, steps=1200, dt=0.05, sigma=sigma, seed=1)
    print(f"  {'K':>5} {'r_final':>9}  переход")
    for Ki, ri in zip(Ks, r):
        mark = "■"*int(round(ri*20))
        print(f"  {Ki:>5.2f} {ri:>9.3f}  {mark}")

    # проверки перехода
    r_low  = r[Ks <= 0.5].mean()                      # ниже порога — рассинхрон
    r_high = r[Ks >= 3.5].mean()                      # выше порога — синхрон
    # эмпирический порог: где r впервые устойчиво > 0.5
    idx = np.argmax(r > 0.5)
    K_emp = Ks[idx] if r[idx] > 0.5 else float("nan")
    print(f"\n  r при K≤0.5  = {r_low:.3f}  (ожид. ~0, рассинхрон)")
    print(f"  r при K≥3.5  = {r_high:.3f}  (ожид. →1, синхрон)")
    print(f"  эмпир. порог перехода K ≈ {K_emp:.2f}  (теория K_c={Kc:.2f})")

    ok_transition = (r_low < 0.3) and (r_high > 0.8)
    ok_kc = abs(K_emp - Kc) <= 1.0 if np.isfinite(K_emp) else False

    print("\n[ИТОГ Ф1]")
    print(f"  SC1 derivative-free .......... {'PASS' if sc1 else 'FAIL'}")
    print(f"  переход r:0→1 ................ {'PASS' if ok_transition else 'FAIL'}")
    print(f"  порог согласован с K_c ....... {'PASS' if ok_kc else 'FAIL'}")

    # опциональный график
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,4))
        plt.plot(Ks, r, "o-", label="r(K) эмпир.")
        plt.axvline(Kc, color="r", ls="--", label=f"K_c теор.={Kc:.2f}")
        plt.xlabel("K (сила связи)"); plt.ylabel("r (синхронность)")
        plt.title("R43 O1 · переход синхронизации (дискретное ядро)")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig("r_vs_K.png", dpi=90)
        print("  график сохранён: r_vs_K.png")
    except Exception as e:
        print(f"  (график пропущен: {e})")
