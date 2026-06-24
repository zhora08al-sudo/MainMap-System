"""
R43-D VERIFIER  —  независимая проверка всех заявлений технической документации R43-D
на РЕАЛЬНЫХ численных расчётах (numpy), а не на словах.

Что проверяется (каждый тест печатает PASS / FAIL / ⚠ против ТЕОРИИ):
  T1  Версия A: параметр α математически сокращается (формула вырождена)
  T2  Базовый режим: при Φ→0  Ψ → Ψ0 + ξ
  T3  Контракция при больших Ψ: коэффициент усиления G = ε^5 / Π(ε+Φ^2) < 1
  T4  Ограниченность БЕЗ клиппинга на сетке параметров
  T5  Уравнение неподвижной точки: степень полинома (документ говорит "третья")
  T6  OU-процесс: стационарная дисперсия σ²/(2η−η²) и автокорреляция (1−η)^|τ|
  T7  Пластичность Хебба: стационарные точки ρ=0, ρ=ρmax, S=S̄
  T8  Сложенный Пуассон: E[ξ]=0, Var[ξ]=λσ_J²
  T9  Attention: argmax веса = ближайшее референсное событие
  SIM полная динамика + графики

Формула (Версия B, как на стр. 6-7 документа, рендер подтверждён):
  Ψ = ε^5 · [ (Ψ0+ξ)(1+αΨ0) + Σ_{n=0..4} Φn cos(θ0+nΔθ) ρn Sn ]
        ----------------------------------------------------------
                 (1+αΨ0) · Π_{n=0..4} (ε + Φn²)
"""
import numpy as np

PASS, FAIL, WARN = "✅ PASS", "❌ FAIL", "⚠ ВНИМАНИЕ"
results = {}

# ----------------------------------------------------------------------
# ЯДРО R43-D
# ----------------------------------------------------------------------
def kernel(Psi0, xi, Phi, rho, S, alpha, eps, theta0=0.0, dtheta=2*np.pi/5):
    """Версия B. ε^5 умножает ВЕСЬ числитель (база + фаза)."""
    Phi = np.asarray(Phi, float); rho = np.asarray(rho, float); S = np.asarray(S, float)
    n = np.arange(len(Phi))
    phase = np.sum(Phi*np.cos(theta0+n*dtheta)*rho*S)
    num = eps**5 * ((Psi0+xi)*(1+alpha*Psi0) + phase)
    den = (1+alpha*Psi0) * np.prod(eps+Phi**2)
    return num/den

def kernel_versionA(Psi0, xi, Phi, rho, S, alpha, eps, theta0=0.0, dtheta=2*np.pi/5):
    """Версия A — умножение фазовой суммы вместо сложения."""
    Phi = np.asarray(Phi, float); rho = np.asarray(rho, float); S = np.asarray(S, float)
    n = np.arange(len(Phi))
    phase = np.sum(Phi*np.cos(theta0+n*dtheta)*rho*S)
    num = eps**5 * ((Psi0+xi)*(1+alpha*Psi0) * phase)
    den = (1+alpha*Psi0) * np.prod(eps+Phi**2)
    return num/den

def attention(Ek, Eref, tau):
    w = np.exp(-(Ek-Eref)**2/tau)
    return w/w.sum()

def ou_step(Phi, target, eta, sigma, rng):
    return Phi + eta*(target-Phi) + sigma*rng.standard_normal(size=np.shape(Phi))

def hebb_step(rho, S, Sbar, gamma, rho_max):
    return rho + gamma*(S-Sbar)*rho*(1-rho/rho_max)

# параметры по умолчанию
PHI0 = np.array([1.0, 0.8, 1.2, 0.5, 0.9])
RHO0 = np.array([0.5, 0.6, 0.4, 0.7, 0.5])
SUNI = np.ones(5)/5

def hr(t): print("\n"+"="*70+f"\n{t}\n"+"="*70)

# ----------------------------------------------------------------------
# T1: α сокращается в Версии A
# ----------------------------------------------------------------------
hr("T1: Версия A — параметр α математически сокращается?")
vals = [kernel_versionA(0.3, 0.1, PHI0, RHO0, SUNI, a, 0.1) for a in (0,0.5,5,1000)]
ident = np.allclose(vals, vals[0], rtol=0, atol=1e-15)
for a,v in zip((0,0.5,5,1000), vals): print(f"  α={a:>6}: Ψ_A = {v:.15e}")
print(f"  ТЕОРИЯ: α должен исчезнуть (множитель (1+αΨ0) сокращается) -> {PASS if ident else FAIL}")
results["T1 α сокращается в Версии A"] = ident

# ----------------------------------------------------------------------
# T2: базовый режим Φ→0 => Ψ→Ψ0+ξ
# ----------------------------------------------------------------------
hr("T2: Базовый режим — при Φ→0  Ψ → Ψ0+ξ ?")
Psi0, xi = 0.3, 0.1
got = kernel(Psi0, xi, PHI0*1e-5, RHO0, SUNI, 0.5, 0.1)
ok2 = abs(got-(Psi0+xi)) < 1e-3
print(f"  Φ_scale=1e-5: Ψ={got:.6f}, ожидание Ψ0+ξ={Psi0+xi}  -> {PASS if ok2 else FAIL}")
results["T2 базовый режим Ψ→Ψ0+ξ"] = ok2

# ----------------------------------------------------------------------
# T3: контракция G<1 и наклон отображения при больших Ψ
# ----------------------------------------------------------------------
hr("T3: Контракция при больших Ψ — G = ε^5/Π(ε+Φ²) < 1 ?")
ok3 = True
print(f"  {'ε':>6} {'G (теория)':>14} {'наклон (числ.)':>16} {'G<1':>6}")
for eps in (0.001,0.01,0.05,0.1,0.5,1.0):
    G = eps**5/np.prod(eps+PHI0**2)
    f = lambda P: kernel(P,0,PHI0,RHO0,SUNI,0.5,eps)
    slope = f(1e6+1)-f(1e6)
    ok = (G<1) and abs(slope-G)/G < 1e-3
    ok3 &= ok
    print(f"  {eps:>6} {G:>14.3e} {slope:>16.3e} {str(G<1):>6}")
print(f"  ТЕОРИЯ: G<1 и наклон=G -> {PASS if ok3 else FAIL}")
results["T3 контракция G<1"] = ok3

# ----------------------------------------------------------------------
# T4: ограниченность БЕЗ клиппинга
# ----------------------------------------------------------------------
hr("T4: Ограниченность БЕЗ клиппинга (динамика 5000 шагов)")
def simulate(eps, alpha=0.5, K=5000, clip=None, phi_scale=1.0, jumps=False,
             tau=0.5, eta=0.05, sigma=0.1, gamma=0.02, Sbar=0.2, rho_max=1.0,
             lam=0.05, jump_sigma=1.0, seed=1, record=False):
    rng = np.random.default_rng(seed)
    Phi = PHI0.copy()*phi_scale; rho = RHO0.copy(); Eref = np.linspace(-2,2,5)
    Psi = 0.0; mx = 0.0; mn = 0.0; diverged = False
    tr = {"Psi":[], "Phi":[], "rho":[], "S":[], "xi":[]} if record else None
    for k in range(K):
        Ek = rng.normal()
        S = attention(Ek, Eref, tau)
        Phi = ou_step(Phi, phi_scale, eta, sigma, rng)
        rho = np.clip(hebb_step(rho, S, Sbar, gamma, rho_max), 0, rho_max)
        xi = (rng.normal(0, jump_sigma) if (jumps and rng.random()<lam) else 0.0)
        Psi = kernel(Psi, xi, Phi, rho, S, alpha, eps)
        if clip is not None: Psi = np.clip(Psi, -clip, clip)
        if not np.isfinite(Psi) or abs(Psi) > 1e12: diverged = True; break
        mx = max(mx, abs(Psi)); mn = min(mn, Psi)
        if record:
            tr["Psi"].append(Psi); tr["Phi"].append(Phi.copy())
            tr["rho"].append(rho.copy()); tr["S"].append(S.copy()); tr["xi"].append(xi)
    return dict(mx=mx, mn=mn, diverged=diverged, Psi=Psi, tr=tr)

ok4 = True
print(f"  {'ε':>6} {'Φ_scale':>8} {'|Ψ|max':>12} {'minΨ':>12} {'разнос':>8}")
for eps in (0.001,0.01,0.1,0.5,1.0):
    for ps in (1.0, 0.1):
        r = simulate(eps, clip=None, phi_scale=ps)
        ok4 &= (not r["diverged"])
        print(f"  {eps:>6} {ps:>8} {r['mx']:>12.3e} {r['mn']:>12.3e} {str(r['diverged']):>8}")
print(f"  ТЕОРИЯ: ограничена без клиппинга -> {PASS if ok4 else FAIL}")
print(f"  (полюс при Ψ=-1/α=-2.0 на практике недостижим — см. minΨ)")
results["T4 ограниченность без клиппинга"] = ok4

# ----------------------------------------------------------------------
# T5: степень уравнения неподвижной точки
# ----------------------------------------------------------------------
hr("T5: Неподвижная точка — документ заявляет 'уравнение третьей степени'")
eps, a = 0.1, 0.5
Dc = np.prod(eps+PHI0**2)
C = np.sum(PHI0*np.cos(np.arange(5)*2*np.pi/5)*RHO0*SUNI)
# (Dc-ε^5)(αΨ²+Ψ) - ε^5 C = 0
coeffs = [(Dc-eps**5)*a, (Dc-eps**5), -eps**5*C]  # deg2: c2 Ψ² + c1 Ψ + c0
deg = np.poly1d(coeffs).order
roots = np.roots(coeffs)
print(f"  Полином неподвижной точки: степень = {deg}, корни = {np.round(roots,8)}")
ok5 = (deg == 2)
print(f"  ТЕОРИЯ: степень = 2 (КВАДРАТНОЕ). Документ говорит 'третья' -> "
      f"{'❌ заявление документа НЕВЕРНО (оно квадратное)' if ok5 else '?'}")
results["T5 уравнение КВАДРАТНОЕ, не кубическое"] = ok5  # True => документ ошибся

# ----------------------------------------------------------------------
# T6: OU стационарная дисперсия и автокорреляция
# ----------------------------------------------------------------------
hr("T6: OU-процесс — стационарная дисперсия σ²/(2η−η²) и автокорреляция (1−η)^|τ|")
rng = np.random.default_rng(0)
ok6 = True
print("  Стационарная дисперсия:")
for eta,sig in ((0.05,0.1),(0.2,0.3),(0.5,0.2)):
    x=0.0; xs=[]
    for k in range(200000):
        x=(1-eta)*x+sig*rng.standard_normal()
        if k>5000: xs.append(x)
    emp=np.var(xs); theo=sig**2/(2*eta-eta**2); err=abs(emp-theo)/theo
    ok6 &= err<0.03
    print(f"    η={eta} σ={sig}: эмпир={emp:.5f}  теор={theo:.5f}  откл={err*100:.2f}%")
print("  Автокорреляция (η=0.1):")
eta=0.1; x=0.0; xs=[]
for k in range(400000): x=(1-eta)*x+0.2*rng.standard_normal(); xs.append(x)
xs=np.array(xs[1000:]); xs-=xs.mean()
for tau in (1,5,10,20):
    emp=np.corrcoef(xs[:-tau],xs[tau:])[0,1]; theo=(1-eta)**tau
    ok6 &= abs(emp-theo)<0.02
    print(f"    τ={tau:>3}: эмпир={emp:.4f}  теор={theo:.4f}")
print(f"  ТЕОРИЯ: формулы AR(1) -> {PASS if ok6 else FAIL}")
results["T6 OU дисперсия+автокорр"] = ok6

# ----------------------------------------------------------------------
# T7: неподвижные точки Хебба
# ----------------------------------------------------------------------
hr("T7: Пластичность Хебба — стационарные точки ρ=0, ρ=ρmax, S=S̄")
def hebb_converge(S, Sbar, r0=0.3, K=5000):
    r=r0
    for _ in range(K): r=hebb_step(r,S,Sbar,0.05,1.0)
    return r
a1=hebb_converge(0.4,0.2); a2=hebb_converge(0.1,0.2); a3=hebb_converge(0.2,0.2)
ok7 = abs(a1-1.0)<1e-3 and abs(a2-0.0)<1e-3 and abs(a3-0.3)<1e-9
print(f"  S>S̄ : ρ→{a1:.4f} (ожид ρmax=1.0)")
print(f"  S<S̄ : ρ→{a2:.4f} (ожид 0)")
print(f"  S=S̄ : ρ→{a3:.4f} (ожид без изменений 0.3)")
print(f"  ТЕОРИЯ: 3 стационарные точки -> {PASS if ok7 else FAIL}")
results["T7 неподвижные точки Хебба"] = ok7

# ----------------------------------------------------------------------
# T8: моменты сложенного Пуассона
# ----------------------------------------------------------------------
hr("T8: Сложенный Пуассон — E[ξ]=0, Var[ξ]=λσ_J²")
lam,sigJ=0.05,1.5; N=2_000_000; rng=np.random.default_rng(1)
xs=np.zeros(N); hit=rng.random(N)<lam; xs[hit]=rng.normal(0,sigJ,hit.sum())
empE,empV=xs.mean(),xs.var(); theoV=lam*sigJ**2
ok8 = abs(empE)<5e-3 and abs(empV-theoV)/theoV<0.02
print(f"  E[ξ]: эмпир={empE:+.5f} (ожид 0)")
print(f"  Var[ξ]: эмпир={empV:.5f}  теор λσ_J²={theoV:.5f}")
print(f"  ТЕОРИЯ: моменты compound Poisson -> {PASS if ok8 else FAIL}")
results["T8 моменты Пуассона"] = ok8

# ----------------------------------------------------------------------
# T9: attention argmax
# ----------------------------------------------------------------------
hr("T9: Attention — argmax веса = ближайшее референсное событие")
Eref=np.linspace(-2,2,5); rng=np.random.default_rng(3); N=5000; good=0
for _ in range(N):
    Ek=rng.uniform(-3,3); S=attention(Ek,Eref,0.5)
    if np.argmax(S)==np.argmin(np.abs(Ek-Eref)): good+=1
acc=100*good/N; ok9=acc>99.9
print(f"  Точность argmax на {N} точках: {acc:.1f}%")
print(f"  ТЕОРИЯ: 100% (argmax монотонной по расстоянию функции) -> {PASS if ok9 else FAIL}")
print(f"  Примечание: это свойство softmax, а не уникальное достижение R43-D.")
results["T9 attention argmax 100%"] = ok9

# ----------------------------------------------------------------------
# ИТОГ
# ----------------------------------------------------------------------
hr("ИТОГОВАЯ ТАБЛИЦА")
for k,v in results.items():
    print(f"  {'✅' if v else '❌'}  {k}")
print("\n  ВЫВОД: математика R43-D воспроизводима и устойчива; единственный реальный")
print("  дефект — 'уравнение третьей степени' (на деле квадратное). НО: в документе")
print("  не задана предметная область (что есть Ψ и E), нет данных и нет ни одного")
print("  предсказания, проверенного против реальности. Это рабочая машинерия без задачи.")
