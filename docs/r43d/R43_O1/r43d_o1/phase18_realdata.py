"""
R43 O1 — Ф18: РЕАЛЬНЫЕ ИЗМЕРЕННЫЕ ДАННЫЕ (V1+V2). НЕ симуляция, НЕ Курамото.

Данные: S&P 500, дневные цены за 5 лет (619k строк) + сектора GICS (ground truth).
Это идеальный реальный тест нашей ниши:
  • рынок в целом = СИЛЬНЫЙ ОБЩИЙ ДРАЙВЕР (все акции коррелируют через рынок);
  • сектора = сообщества ПОД этим драйвером (ground truth).
  • фаза извлекается преобразованием Гильберта (реальные сигналы, НЕ Курамото).

Сравнение восстановления секторов (ARI/NMI vs GICS):
  наивная корреляция      — должна тонуть в рыночном факторе;
  R43 O1 (corr→coh, Ф11/16) — ранг убирает рыночный 'пол' → сектора всплывают;
  дифференциальный (dφ/dt) — производный baseline.

Источники (raw GitHub, выкачаны curl):
  prices: plotly/datasets all_stocks_5yr.csv
  sectors: datasets/s-and-p-500-companies constituents.csv
Зависимости: numpy, pandas, scipy, scikit-learn.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.signal import hilbert
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from phase16_hier import hier_corr_coh
import os, urllib.request

CACHE = "/tmp/r43o1_realdata"
URLS = {
    "stocks.csv": "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv",
    "sectors.csv": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
}

def _ensure(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        print(f"  скачиваю {name} ...")
        urllib.request.urlretrieve(URLS[name], path)
    return path


def load():
    px = pd.read_csv(_ensure("stocks.csv"))
    close = px.pivot_table(index="date", columns="Name", values="close")
    close = close.dropna(axis=1)                       # тикеры с полной историей
    sec = pd.read_csv(_ensure("sectors.csv")).set_index("Symbol")["GICS Sector"]
    common = [t for t in close.columns if t in sec.index]
    close = close[common]
    sectors = sec.loc[common]
    # топ-6 секторов по числу тикеров (баланс/чёткость)
    top = sectors.value_counts().index[:6]
    keep = sectors[sectors.isin(top)].index
    close = close[keep]; sectors = sectors.loc[keep]
    labels = pd.Categorical(sectors).codes
    return close.values, labels, list(top), keep


def phases_from_prices(close):
    """Лог-доходности → фаза через Гильберт (реальные сигналы, derivative-free для R43)."""
    ret = np.diff(np.log(close), axis=0)               # (T-1, N)
    ret = ret - ret.mean(0, keepdims=True)
    analytic = hilbert(ret, axis=0)                     # аналитический сигнал
    return np.angle(analytic)                           # (T-1, N) фазы


def corr_naive(phi, K):
    C = np.abs(np.nan_to_num(np.corrcoef(np.cos(phi).T)))
    np.fill_diagonal(C, 1.0)
    return SpectralClustering(K, affinity="precomputed", random_state=0).fit_predict(C)

def deriv_method(phi, K):
    D = np.gradient(phi, axis=0); N = phi.shape[1]; C = np.zeros((N, N))
    for i in range(N):
        F = np.sin(phi - phi[:, [i]]); X = np.hstack([np.ones((F.shape[0],1)), F])
        C[i] = np.linalg.solve(X.T@X + np.eye(X.shape[1]), X.T@D[:, i])[1:]
    A = np.abs(C)+np.abs(C.T); np.fill_diagonal(A, A.max())
    return SpectralClustering(K, affinity="precomputed", random_state=0).fit_predict(A)


if __name__ == "__main__":
    print("="*66)
    print("R43 O1 · Ф18 · РЕАЛЬНЫЕ данные S&P 500 (V1+V2, НЕ Курамото)")
    print("="*66)
    close, labels, sectors, tickers = load()
    K = len(sectors)
    print(f"  акций: {close.shape[1]}, дней: {close.shape[0]}, секторов K={K}")
    print(f"  сектора: {sectors}")
    phi = phases_from_prices(close)

    def sc(lab): return adjusted_rand_score(labels, lab), normalized_mutual_info_score(labels, lab)
    ar_h, nmi_h = sc(hier_corr_coh(phi, K, k=15))
    ar_c, nmi_c = sc(corr_naive(phi, K))
    ar_d, nmi_d = sc(deriv_method(phi, K))

    print(f"\n  {'метод':28} {'ARI':>7} {'NMI':>7}")
    print(f"  {'R43 O1 (corr→coh)':28} {ar_h:>7.3f} {nmi_h:>7.3f}")
    print(f"  {'наивная корреляция':28} {ar_c:>7.3f} {nmi_c:>7.3f}")
    print(f"  {'дифференциальный (dφ/dt)':28} {ar_d:>7.3f} {nmi_d:>7.3f}")

    print("\n[ВЫВОД]")
    win = ar_h > max(ar_c, ar_d)
    print(f"  R43 O1 {'ЛУЧШЕ' if win else 'не лучше'} прочих на РЕАЛЬНЫХ данных "
          f"(рынок=общий драйвер, сектора=сообщества).")
    print(f"  Это первая валидация на настоящих измеренных сигналах (НЕ Курамото).")
    print(f"  Замечание: сектора — не идеальные фазовые сообщества, поэтому ARI скромный;")
    print(f"  важно ОТНОСИТЕЛЬНОЕ преимущество corr→coh (ранг убирает рыночный 'пол').")
