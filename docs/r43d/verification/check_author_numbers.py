import numpy as np
def Psi(Psi0,xi,alpha,eps,th0,dth,Phi,rho,S):
    Phi=np.array(Phi,float);rho=np.array(rho,float);S=np.array(S,float);n=np.arange(5)
    phase=np.sum(Phi*np.cos(th0+n*dth)*rho*S)
    num=eps**5*((Psi0+xi)*(1+alpha*Psi0)+phase)
    den=(1+alpha*Psi0)*np.prod(eps+Phi**2)
    return num/den

cases=[
 ("Симуляция 1 (стр.184)",dict(Psi0=1.5,xi=0.8,alpha=0.3,eps=0.05,th0=0.2,dth=0.4,
    Phi=[0.4,0.5,0.6,0.7,0.8],rho=[1.0,1.2,1.4,1.6,1.8],S=[0.9,0.85,0.8,0.75,0.7]),9.8207e-5),
 ("Отчёт Фаза I (стр.442)",dict(Psi0=2.0,xi=1.2,alpha=0.5,eps=0.01,th0=0.3,dth=0.45,
    Phi=[0.2,0.35,0.5,0.65,0.8],rho=[1.0,1.25,1.5,1.75,2.0],S=[0.95,0.88,0.81,0.74,0.67]),6.686e-7),
 ("Нагрузка 98.5% (стр.616)",dict(Psi0=2.5,xi=1.7,alpha=0.65,eps=0.009,th0=0.28,dth=0.49,
    Phi=[0.25,0.39,0.53,0.67,0.81],rho=[1.05,1.33,1.61,1.89,2.17],S=[0.93,0.865,0.80,0.735,0.67]),2.4097e-7),
 ("Нагрузка 97% (стр.758)",dict(Psi0=2.1,xi=1.4,alpha=0.55,eps=0.007,th0=0.27,dth=0.44,
    Phi=[0.23,0.36,0.49,0.62,0.75],rho=[1.01,1.28,1.55,1.82,2.09],S=[0.91,0.85,0.79,0.73,0.67]),1.351e-7),
]
print(f"{'случай':28} {'заявлено':>12} {'пересчёт':>14} {'совпадает':>10}")
for name,p,claimed in cases:
    got=Psi(**p)
    ok="ДА" if abs(got-claimed)/claimed<0.02 else f"НЕТ ({got/claimed:.2f}x)"
    print(f"{name:28} {claimed:>12.4e} {got:>14.4e} {ok:>10}")

print("\nПроверка тождества: 'маска фильтрации' ∏(1 - Φ²/(Φ²+ε)) == ε⁵/∏(ε+Φ²) ?")
Phi=np.array([0.2,0.35,0.5,0.65,0.8]);eps=0.01
mask=np.prod(1-Phi**2/(Phi**2+eps)); denomfac=eps**5/np.prod(eps+Phi**2)
print(f"  маска={mask:.6e}  ε⁵/∏(ε+Φ²)={denomfac:.6e}  тождество: {np.isclose(mask,denomfac)}")
print("  => 'флуктуационная маска' — это в точности множитель знаменателя, переименованный.")
