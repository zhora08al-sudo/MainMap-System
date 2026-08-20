"""
R43 O1 — Ф26: РЕАЛЬНАЯ погода — фазовый лаг суточного цикла = долгота.

Идея теста. Почасовая температура многих городов (Open-Meteo, по UTC). Доминирующая
осцилляция — СУТОЧНЫЙ цикл (период 24 ч), общий почти для всех (глобальный драйвер).
НО локальный полдень сдвигается с ДОЛГОТОЙ → по UTC пик температуры у разных городов в
разные часы = РЕАЛЬНЫЙ постоянный фазовый ЛАГ. Это чистый физический фазовый признак.

Что проверяем (ground truth = КОНТИНЕНТ ≈ долготная полоса):
  • наивная |корреляция| температуры — ЛАГ-чувствительна «не туда»: города со сдвигом
    6 ч (≈90° долготы) почти НЕ коррелируют, со сдвигом 12 ч — анти-коррелируют (|corr|↑).
  • ФАЗА суточного цикла (комплексный фазор на 1/24ч): угол = местный полдень = долгота →
    кластеризация по фазовому СДВИГУ должна чисто давать континенты.
  • лаг-ИНВАРИАНТный corr→coh (наше ядро) — здесь должен ПРОИГРАТЬ: он стирает именно тот
    сдвиг, в котором сидит группировка (все 24ч «когерентны»). Честный контр-пример: когда
    структура = ЗНАЧЕНИЕ лага, нужна лаг-ЧУВСТВИТЕЛЬНАЯ мера (урок Ф22 в обратную сторону).

Зависимости: numpy scipy scikit-learn + интернет (archive-api.open-meteo.com).
"""
from __future__ import annotations
import warnings, os, json, urllib.request, urllib.parse; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

CACHE = "/tmp/r43o1_weather"
START, END = "2024-01-01", "2024-02-09"     # 40 суток почасово

CITIES = [
    # name, lat, lon, continent
    ("NewYork", 40.71, -74.01, "America"), ("Chicago", 41.85, -87.65, "America"),
    ("Denver", 39.74, -104.99, "America"), ("LosAngeles", 34.05, -118.24, "America"),
    ("Toronto", 43.65, -79.38, "America"), ("MexicoCity", 19.43, -99.13, "America"),
    ("Vancouver", 49.28, -123.12, "America"), ("Houston", 29.76, -95.37, "America"),
    ("London", 51.51, -0.13, "Europe"), ("Paris", 48.85, 2.35, "Europe"),
    ("Berlin", 52.52, 13.41, "Europe"), ("Madrid", 40.42, -3.70, "Europe"),
    ("Rome", 41.89, 12.50, "Europe"), ("Warsaw", 52.23, 21.01, "Europe"),
    ("Oslo", 59.91, 10.75, "Europe"), ("Athens", 37.98, 23.73, "Europe"),
    ("Tokyo", 35.68, 139.69, "EastAsia"), ("Beijing", 39.90, 116.41, "EastAsia"),
    ("Seoul", 37.57, 126.98, "EastAsia"), ("Shanghai", 31.23, 121.47, "EastAsia"),
    ("Bangkok", 13.76, 100.50, "EastAsia"), ("Singapore", 1.35, 103.82, "EastAsia"),
    ("Manila", 14.60, 120.98, "EastAsia"), ("Taipei", 25.03, 121.57, "EastAsia"),
    ("Sydney", -33.87, 151.21, "Oceania"), ("Melbourne", -37.81, 144.96, "Oceania"),
    ("Brisbane", -27.47, 153.03, "Oceania"), ("Perth", -31.95, 115.86, "Oceania"),
    ("Adelaide", -34.93, 138.60, "Oceania"), ("Auckland", -36.85, 174.76, "Oceania"),
    ("Wellington", -41.29, 174.78, "Oceania"), ("Canberra", -35.28, 149.13, "Oceania"),
]


def fetch(name, lat, lon):
    os.makedirs(CACHE, exist_ok=True); path = os.path.join(CACHE, f"{name}.json")
    if not os.path.exists(path):
        q = urllib.parse.urlencode(dict(latitude=lat, longitude=lon, start_date=START,
            end_date=END, hourly="temperature_2m", timezone="GMT"))
        urllib.request.urlretrieve(f"https://archive-api.open-meteo.com/v1/archive?{q}", path)
    with open(path) as f: d = json.load(f)
    return np.array(d["hourly"]["temperature_2m"], float)


def load_all():
    rows, names, cont = [], [], []
    for nm, la, lo, co in CITIES:
        try:
            t = fetch(nm, la, lo)
            if np.isnan(t).mean() < 0.05:
                t = np.nan_to_num(t, nan=np.nanmean(t)); rows.append(t); names.append(nm); cont.append(co)
        except Exception as e:
            print(f"  пропуск {nm}: {type(e).__name__}")
    L = min(len(r) for r in rows)
    X = np.array([r[:L] for r in rows]).T          # (T, N)
    codes = np.unique(cont, return_inverse=True)[1]
    return X, np.array(names), codes, np.array(cont)


def diurnal_phasor(X):
    """Комплексный фазор суточного цикла на канал: угол = местный полдень (долгота)."""
    T = X.shape[0]; t = np.arange(T)
    w = np.exp(-1j*2*np.pi*t/24.0)                 # 24-часовая частота (почасовые данные)
    Xc = X - X.mean(0)
    return (Xc * w[:, None]).sum(0)                # (N,) комплексные фазоры


def sc(aff, K):
    A = np.clip(0.5*(aff+aff.T), 0, None); np.fill_diagonal(A, 1.0)
    return SpectralClustering(K, affinity="precomputed", assign_labels="kmeans",
                              random_state=0).fit_predict(A)


def naive_corr(X, K):
    C = np.abs(np.nan_to_num(np.corrcoef(X.T))); return sc(C, K)

def phase_offset(X, K):                            # ЛАГ-ЧУВСТВИТЕЛЬНО: близость фазовых углов
    p = diurnal_phasor(X); u = p/(np.abs(p)+1e-12)
    A = (np.real(np.outer(u, np.conj(u))) + 1.0)/2.0   # cos(Δугол)→[0,1]
    return sc(A, K)

def lag_invariant(X, K):                           # наше ядро (здесь ожидаем проигрыш)
    Z = X - X.mean(0); G = Z.conj().T @ Z
    d = np.sqrt(np.real(np.diag(G))); d[d == 0] = 1
    C = np.abs(G)/np.outer(d, d); return sc(C, K)


if __name__ == "__main__":
    print("="*72)
    print("R43 O1 · Ф26 · РЕАЛЬНАЯ погода: фазовый лаг суточного цикла = долгота/континент")
    print("="*72)
    X, names, codes, cont = load_all()
    K = len(np.unique(codes))
    print(f"  городов: {X.shape[1]}, отсчётов(ч): {X.shape[0]}, континентов: {K} ({sorted(set(cont))})")
    print(f"\n  {'метод':34} {'ARI':>7} {'NMI':>7}")
    print("  " + "-"*50)
    res = {}
    for nm, lab in [("наивная |корреляция| T", naive_corr(X, K)),
                    ("ФАЗА суточного цикла (лаг-чувств.)", phase_offset(X, K)),
                    ("лаг-ИНВАРИАНТный corr→coh (ядро)", lag_invariant(X, K))]:
        a = adjusted_rand_score(codes, lab); n = normalized_mutual_info_score(codes, lab)
        res[nm] = a; print(f"  {nm:34} {a:>7.3f} {n:>7.3f}")

    print("\n[ЧЕСТНЫЙ ВЫВОД — по фактам]")
    best = max(res, key=res.get)
    print(f"  Лучший: «{best}» (ARI {res[best]:.2f}).")
    print("  Ожидание теста: фазовый сдвиг суточного цикла кодирует долготу→континент, поэтому")
    print("  ЛАГ-ЧУВСТВИТЕЛЬНАЯ фаза должна бить и наивную корреляцию, и лаг-инвариантное ядро.")
    print("  Это обратная сторона урока Ф22: когда структура = ЗНАЧЕНИЕ лага, инвариантность ВРЕДНА.")
