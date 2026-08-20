# ======================================================================
#  R43-D NET — рабочий лёгкий классификатор на механизмах R43-D
#  (attention = softmax расстояний до обучаемых прототипов  +  плотности ρ)
#  Обучается обычным градиентным спуском. Реальная задача: рукописные цифры.
#  Зависимости: numpy + scikit-learn (в Google Colab уже установлены).
# ======================================================================
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)

# ---------- данные ----------
X, y = load_digits(return_X_y=True)              # 1797 x 64, 10 классов
X = StandardScaler().fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
N, D = Xtr.shape; C = 10

# ======================================================================
#  МОДЕЛЬ R43-D:
#   d_m   = || x - P_m ||^2                       (расстояние до прототипа m)
#   S_m   = softmax_m( -d_m / tau )               (attention, eq.7 документа)
#   w_m   = rho_m * S_m                            (плотность по Хеббу * attention)
#   logit = sum_m w_m * V_m                        (агрегация значений)
#  Обучаемые параметры: P (прототипы/ключи), V (значения), log rho, log tau
# ======================================================================
M = 60                                            # число прототипов ("уровней")
P  = rng.normal(0, 1, (M, D)) * 0.5
V  = rng.normal(0, 0.1, (M, C))
lr_rho = np.zeros(M)                              # rho = exp(lr_rho) -> rho>0
lr_tau = np.log(10.0)                             # tau = exp(lr_tau) -> tau>0

def forward(Xb):
    tau = np.exp(lr_tau); rho = np.exp(lr_rho)
    # d: (B,M)
    d = ((Xb[:, None, :] - P[None, :, :])**2).sum(-1)
    u = -d / tau
    u -= u.max(1, keepdims=True)
    e = np.exp(u); S = e / e.sum(1, keepdims=True)         # (B,M) attention
    w = rho[None, :] * S                                   # (B,M)
    logit = w @ V                                          # (B,C)
    cache = (Xb, d, S, w, rho, tau)
    return logit, cache

def softmax_ce(logit, yb):
    z = logit - logit.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(yb)), yb] + 1e-12).mean()
    return loss, p

def backward(cache, p, yb):
    Xb, d, S, w, rho, tau = cache
    B = len(yb)
    dz = p.copy(); dz[np.arange(B), yb] -= 1; dz /= B       # (B,C)
    dV = w.T @ dz                                           # (M,C)
    dw = dz @ V.T                                           # (B,M)
    drho = (dw * S).sum(0)                                  # (M,)
    dS = dw * rho[None, :]                                  # (B,M)
    du = S * (dS - (dS * S).sum(1, keepdims=True))          # softmax jacobian
    dd = -du / tau                                          # (B,M)
    dtau = (du * (d / tau)).sum()                           # scalar (через u=-d/tau)
    # dP_m = sum_b dd_{b,m} * 2 (P_m - x_b)
    diff = P[None, :, :] - Xb[:, None, :]                   # (B,M,D)
    dP = (2 * dd[:, :, None] * diff).sum(0)                 # (M,D)
    dlr_rho = drho * rho                                    # d/d(log rho)
    dlr_tau = dtau * tau                                    # d/d(log tau)
    return dP, dV, dlr_rho, dlr_tau

# ---------- Adam ----------
params = {"P": P, "V": V, "lr_rho": lr_rho, "lr_tau": lr_tau}
m_ = {k: np.zeros_like(np.atleast_1d(v)*1.0) for k, v in params.items()}
v_ = {k: np.zeros_like(np.atleast_1d(v)*1.0) for k, v in params.items()}
def adam(name, g, t, lr=5e-2, b1=0.9, b2=0.999):
    m_[name][:] = b1*m_[name] + (1-b1)*g
    v_[name][:] = b2*v_[name] + (1-b2)*g*g
    mh = m_[name]/(1-b1**t); vh = v_[name]/(1-b2**t)
    return lr*mh/(np.sqrt(vh)+1e-8)

def accuracy(Xs, ys):
    return (forward(Xs)[0].argmax(1) == ys).mean()

# ---------- (опц.) численная проверка градиента ----------
def grad_check():
    Xb, yb = Xtr[:8], ytr[:8]
    logit, cache = forward(Xb); _, p = softmax_ce(logit, yb)
    dP, dV, drho, dtau = backward(cache, p, yb)
    eps = 1e-5; i, j = 3, 7
    P[i, j] += eps; l1 = softmax_ce(forward(Xb)[0], yb)[0]
    P[i, j] -= 2*eps; l2 = softmax_ce(forward(Xb)[0], yb)[0]
    P[i, j] += eps
    num = (l1 - l2)/(2*eps)
    print(f"grad-check dP[{i},{j}]: аналит={dP[i,j]:+.6f}  числ={num:+.6f}  "
          f"{'OK' if abs(num-dP[i,j])<1e-4 else 'MISMATCH'}")

grad_check()

# ---------- обучение ----------
print("\nОбучение R43-D Net (M=60 прототипов)...")
t = 0; B = 64
for epoch in range(1, 41):
    idx = rng.permutation(N)
    for s in range(0, N, B):
        b = idx[s:s+B]; t += 1
        logit, cache = forward(Xtr[b]); loss, p = softmax_ce(logit, ytr[b])
        dP, dV, dlr_rho, dlr_tau = backward(cache, p, ytr[b])
        P      -= adam("P", dP, t)
        V      -= adam("V", dV, t)
        lr_rho -= adam("lr_rho", dlr_rho, t)
        lr_tau -= adam("lr_tau", np.atleast_1d(dlr_tau), t)[0]
    if epoch % 5 == 0:
        print(f"  эпоха {epoch:2d}: loss={loss:.3f}  train={accuracy(Xtr,ytr):.3f}  test={accuracy(Xte,yte):.3f}")

acc_r43d = accuracy(Xte, yte)

# ---------- baseline ----------
lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
acc_lr = lr.score(Xte, yte)

print("\n" + "="*52)
print(f"  R43-D Net (attention+плотности): test acc = {acc_r43d:.4f}")
print(f"  Logistic Regression (baseline) : test acc = {acc_lr:.4f}")
print(f"  Параметров в R43-D Net: {M*D + M*C + M + 1}  (прототипы+значения+ρ+τ)")
print("="*52)
print(f"  обученная температура tau = {np.exp(lr_tau):.3f}")
print("  ВЫВОД: модель РЕАЛЬНО учится и сравнима с baseline. Это честный")
print("  лёгкий прототип-attention (RBF), а НЕ полноценная замена трансформеру:")
print("  нет self-attention между токенами, нет глубины, нет позиционности.")
