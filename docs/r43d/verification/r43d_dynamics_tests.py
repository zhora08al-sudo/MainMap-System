import numpy as np
th0,dth=0.0,2*np.pi/5; n=np.arange(5)

# ПРАВИЛЬНАЯ Версия B: Ψ = ε^5·[(Ψ0+ξ)(1+αΨ0) + Σ Φcos ρS] / [(1+αΨ0)·Π(ε+Φ^2)]
def vB(P0,xi,Phi,rho,S,a,eps):
    C=np.sum(Phi*np.cos(th0+n*dth)*rho*S)
    return eps**5*((P0+xi)*(1+a*P0)+C)/((1+a*P0)*np.prod(eps+Phi**2))

Phi=np.array([1.0,0.8,1.2,0.5,0.9]); rho=np.array([0.5,0.6,0.4,0.7,0.5]); S=np.ones(5)/5

print("ПРОВЕРКА 1: при Φ->0  Ψ -> Ψ0+ξ  (заявленный базовый режим)")
for sc in [1.0,0.1,0.01,1e-4]:
    print(f"  Φ_scale={sc:7.0e}: Ψ={vB(0.3,0.1,Phi*sc,rho,S,0.5,0.1):.6f}  (ожид Ψ0+ξ=0.4)")
print()

print("ПРОВЕРКА 2: масштаб статического Ψ по ε (объясняет 'Ψ~1e-7')")
for eps in [0.001,0.01,0.1,0.5,1.0]:
    print(f"  ε={eps:5.3f}: Ψ={vB(0.0,0.0,Phi,rho,S,0.5,eps):.4e}")
print()

print("ПРОВЕРКА 3 (КЛЮЧЕВАЯ): ограниченность БЕЗ клиппинга, правильная формула")
def sim(eps,alpha=0.5,K=5000,clip=None,seed=1,phi_scale=1.0,jumps=False):
    rng=np.random.default_rng(seed)
    Phi=np.array([1.0,0.8,1.2,0.5,0.9])*phi_scale; rho=np.array([0.5,0.6,0.4,0.7,0.5])
    Eref=np.linspace(-2,2,5); Psi=0.0; tau=0.5; eta,sg=0.05,0.1; g,Sb,rmax=0.02,0.2,1.0; lam=0.05
    mx=0.0; dv=False
    for k in range(K):
        Ek=rng.normal(); Sw=np.exp(-(Ek-Eref)**2/tau); Sw/=Sw.sum()
        Phi=Phi+eta*(phi_scale-Phi)+sg*rng.normal(size=5)
        rho=rho+g*(Sw-Sb)*rho*(1-rho/rmax); rho=np.clip(rho,0,rmax)
        xi=rng.normal(0,1.0) if (jumps and rng.random()<lam) else 0.0
        Psi=vB(Psi,xi,Phi,rho,Sw,alpha,eps)
        if clip: Psi=np.clip(Psi,-clip,clip)
        if not np.isfinite(Psi) or abs(Psi)>1e12: dv=True; break
        mx=max(mx,abs(Psi))
    return mx,dv,Psi
for eps in [0.001,0.01,0.1,0.5,1.0]:
    for ps in [1.0,0.1]:
        mx,dv,_=sim(eps,clip=None,phi_scale=ps)
        print(f"  ε={eps:5.3f} Φ_scale={ps:4.1f} БЕЗ клипа: |Ψ|max={mx:.4e} diverged={dv}")
mx,dv,_=sim(0.1,clip=None,jumps=True)
print(f"  ε=0.1 со скачками, без клипа: |Ψ|max={mx:.4e} diverged={dv}")
print()

print("ПРОВЕРКА 4: достигается ли реально полюс Ψ=-1/α? (мин Ψ за прогон)")
def minPsi(eps,alpha=0.5,K=20000,seed=7):
    rng=np.random.default_rng(seed)
    Phi=np.array([1.0,0.8,1.2,0.5,0.9]); rho=np.array([0.5,0.6,0.4,0.7,0.5])
    Eref=np.linspace(-2,2,5); Psi=0.0; mn=0.0
    for k in range(K):
        Ek=rng.normal(); Sw=np.exp(-(Ek-Eref)**2/0.5); Sw/=Sw.sum()
        Phi=Phi+0.05*(1-Phi)+0.1*rng.normal(size=5)
        Psi=vB(Psi,0.0,Phi,rho,Sw,alpha,eps); mn=min(mn,Psi)
    return mn
for eps in [0.001,0.1,1.0]:
    print(f"  ε={eps:5.3f}: minΨ={minPsi(eps):.4e}  полюс при Ψ=-1/α=-2.0  -> запас огромный")
print()

print("ПРОВЕРКА 5: 'уравнение третьей степени' — повтор с ПРАВИЛЬНОЙ формулой")
eps,a=0.1,0.5; Dc=np.prod(eps+Phi**2); C=np.sum(Phi*np.cos(th0+n*dth)*rho*S)
# (Dc-eps^5)(a Ψ^2+Ψ) - eps^5 C = 0
rt=np.roots([(Dc-eps**5)*a,(Dc-eps**5),-eps**5*C])
print(f"  степень=2, корни={np.round(rt,8)}  -> КВАДРАТНОЕ, не кубическое (ошибка ОСТАЁТСЯ)")
