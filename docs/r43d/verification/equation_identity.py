import numpy as np
np.random.seed(1)
def form_eps5(Psi0,xi,alpha,eps,th0,dth,Phi,rho,S):
    n=np.arange(5); ph=np.sum(Phi*np.cos(th0+n*dth)*rho*S)
    return eps**5*((Psi0+xi)*(1+alpha*Psi0)+ph)/((1+alpha*Psi0)*np.prod(eps+Phi**2))
def form_mask(Psi0,xi,alpha,eps,th0,dth,Phi,rho,S):     # строка 1072 файла
    n=np.arange(5); ph=np.sum(Phi*np.cos(th0+n*dth)*rho*S)
    mask=np.prod(1-Phi**2/(Phi**2+eps))
    return (Psi0+xi + ph/(1+alpha*Psi0))*mask
print(f"{'eps5-форма':>16} {'mask-форма':>16} {'идентичны':>10}")
for _ in range(5):
    a=dict(Psi0=np.random.uniform(0,3),xi=np.random.uniform(0,2),alpha=np.random.uniform(0,1),
           eps=np.random.uniform(0.005,0.1),th0=np.random.uniform(0,1),dth=np.random.uniform(0,1),
           Phi=np.random.uniform(0.1,1,5),rho=np.random.uniform(0.5,2,5),S=np.random.uniform(0.5,1,5))
    x,y=form_eps5(**a),form_mask(**a)
    print(f"{x:16.6e} {y:16.6e} {np.isclose(x,y,rtol=1e-12)!s:>10}")
print("\n=> две 'разные' главные формулы файлов — это одно уравнение, записанное двояко.")
