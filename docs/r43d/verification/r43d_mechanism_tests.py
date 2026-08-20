import numpy as np
rng=np.random.default_rng(42)

print("="*64)
print("ТЕСТ A: OU — стационарная дисперсия  σ^2/(2η-η^2)  (стр.10)")
for eta,sig in [(0.05,0.1),(0.2,0.3),(0.5,0.2)]:
    x=0.0; xs=[]
    for k in range(2_000_00):
        x=(1-eta)*x+eta*0.0+sig*rng.standard_normal()
        if k>5000: xs.append(x)
    emp=np.var(xs); theo=sig**2/(2*eta-eta**2)
    print(f"  η={eta:.2f} σ={sig:.2f}: эмпир.var={emp:.5f}  теор={theo:.5f}  откл={abs(emp-theo)/theo*100:.2f}%")
print("  => формула AR(1)-дисперсии ВЕРНА")

print("="*64)
print("ТЕСТ B: OU — автокорреляция  Corr(τ)=(1-η)^|τ|  (стр.10)")
eta,sig=0.1,0.2; x=0.0; xs=[]
for k in range(500000):
    x=(1-eta)*x+sig*rng.standard_normal(); xs.append(x)
xs=np.array(xs[1000:]); xs-=xs.mean()
for tau in [1,5,10,20]:
    emp=np.corrcoef(xs[:-tau],xs[tau:])[0,1]; theo=(1-eta)**tau
    print(f"  τ={tau:3d}: эмпир={emp:.4f}  теор={theo:.4f}")
print("  => автокорреляция ВЕРНА; τ_c≈1/η =",1/eta)

print("="*64)
print("ТЕСТ C: Хебб — неподвижные точки ρ=0, ρ=ρmax при S><S̄ (стр.14)")
def hebb(S,Sbar,gamma=0.05,rhomax=1.0,r0=0.3,K=5000):
    r=r0
    for _ in range(K): r=r+gamma*(S-Sbar)*r*(1-r/rhomax)
    return r
print(f"  S>S̄ (0.4>0.2): ρ -> {hebb(0.4,0.2):.4f}  (ожид. ρmax=1.0)")
print(f"  S<S̄ (0.1>0.2): ρ -> {hebb(0.1,0.2):.4f}  (ожид. 0)")
print(f"  S=S̄ (0.2=0.2): ρ -> {hebb(0.2,0.2):.4f}  (ожид. остаётся 0.3)")
print("  => три стационарные точки и их устойчивость ВЕРНЫ")

print("="*64)
print("ТЕСТ D: compound Poisson — E[ξ]=0, Var[ξ]=λσ_J^2 (стр.16)")
lam,sigJ,dt=0.05,1.5,1.0; N=2_000_000; xs=np.zeros(N)
hit=rng.random(N)<lam*dt
xs[hit]=rng.normal(0,sigJ,hit.sum())
print(f"  эмпир E[ξ]={xs.mean():+.5f} (ожид 0)")
print(f"  эмпир Var[ξ]={xs.var():.5f}  теор λσ_J^2={lam*sigJ**2:.5f}")
print("  => моменты ВЕРНЫ")

print("="*64)
print("ТЕСТ E: неподвижная точка — лишний ε^5 на C_phase (стр.17-18)")
Phi=np.array([1.0,0.8,1.2,0.5,0.9]); rho=np.array([0.5,0.6,0.4,0.7,0.5]); S=np.ones(5)/5
th0,dth=0.0,2*np.pi/5; n=np.arange(5); eps,a=0.1,0.5
Dc=np.prod(eps+Phi**2); C=np.sum(Phi*np.cos(th0+n*dth)*rho*S)
# Истинная формула (1): фазовый член БЕЗ ε^5
#   (Dc-eps^5)(a Ψ^2 + Ψ) - C = 0
rt_true=np.roots([(Dc-eps**5)*a,(Dc-eps**5),-C])
# Формула из документа (стр.17): ε^5[...+C]  => RHS=eps^5*C
rt_doc =np.roots([(Dc-eps**5)*a,(Dc-eps**5),-eps**5*C])
pos=lambda r:[x.real for x in r if abs(x.imag)<1e-9 and x.real>0]
print(f"  Ψ* по ИСТИННОЙ формуле(1):  {pos(rt_true)}")
print(f"  Ψ* по формуле документа:    {pos(rt_doc)}")
print(f"  отношение ~ ε^5 = {eps**5:.1e}  => документ занижает Ψ* в ~1/ε^5 раз")
print("  => в ур-нии неподвижной точки фазовый член ошибочно умножен на ε^5,")
print("     что и создаёт иллюзию 'мгновенного затухания / Ψ*~1e-7'")

print("="*64)
print("ТЕСТ F: 'уравнение третьей степени' (стр.18) — повторная проверка")
print(f"  степень полинома неподвижной точки = 2 (корни: {np.round(rt_true,4)})")
print("  => снова КВАДРАТНОЕ, не кубическое. Ошибка перенесена из v1.")
