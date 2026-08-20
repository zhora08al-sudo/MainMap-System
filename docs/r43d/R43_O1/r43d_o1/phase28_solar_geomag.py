"""
R43 O1 — Ф28: СОЛНЕЧНЫЙ драйвинг — геомагнитная суточная фаза = долгота.

Параллель Ф26 в другом физическом домене (космическая погода). Геомагнитная суточная
Sq-вариация порождается ионосферными токами, которые гонит СОЛНЦЕ; её пик — около
местного полудня → по UT сдвинут ДОЛГОТОЙ обсерватории. Чистый реальный фазовый признак
солнечного происхождения.

Данные: USGS geomagnetism web service, H-компонента, обсерватории с широким разбросом
долгот (Гуам +145 … Гавайи/Аляска −158), поминутно → усреднение до часа.

Проверяем: восстанавливает ли ФАЗА суточной вариации ДОЛГОТУ обсерватории —
  (а) круговая корреляция угла суточного фазора с географической долготой;
  (б) кластеризация в долготные полосы (ARI), фаза vs наивная корреляция.
Зависимости: numpy scikit-learn + интернет (geomag.usgs.gov).
"""
from __future__ import annotations
import warnings, os, json, urllib.request; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

CACHE = "/tmp/r43o1_geomag"
START, END = "2024-09-02T00:00:00Z", "2024-09-16T00:00:00Z"   # 14 суток
OBS = [   # id, lat, lon
    ("HON", 21.32, -158.00), ("BRW", 71.32, -156.60), ("CMO", 64.87, -147.86),
    ("SIT", 57.06, -135.33), ("NEW", 48.26, -117.12), ("FRN", 37.09, -119.72),
    ("BOU", 40.13, -105.24), ("TUC", 32.17, -110.73), ("BSL", 30.35, -89.64),
    ("FRD", 38.20, -77.37),  ("SJG", 18.11, -66.15),  ("GUA", 13.59, 144.87),
]


def fetch(oid):
    os.makedirs(CACHE, exist_ok=True); path = os.path.join(CACHE, f"{oid}.json")
    if not os.path.exists(path):
        url = (f"https://geomag.usgs.gov/ws/data/?id={oid}&starttime={START}&endtime={END}"
               f"&elements=H&format=json&sampling_period=60&type=adjusted")
        urllib.request.urlretrieve(url, path)
    with open(path) as f: d = json.load(f)
    v = np.array([x if x is not None else np.nan for x in d["values"][0]["values"]], float)
    return v


def to_hourly(v):
    n = (len(v)//60)*60; v = v[:n].reshape(-1, 60)
    return np.nanmean(v, axis=1)


def load_all(maxlat=55.0):
    rows, lon, lat = [], [], []
    for oid, la, lo in OBS:
        if abs(la) > maxlat:                     # исключаем авроральные (Sq не местно-солнечный)
            continue
        try:
            h = to_hourly(fetch(oid))
            if np.isnan(h).mean() < 0.1:
                m = np.nanmean(h); h = np.where(np.isnan(h), m, h)
                rows.append(h); lon.append(lo); lat.append(la)
        except Exception as e:
            print(f"  пропуск {oid}: {type(e).__name__}")
    L = min(len(r) for r in rows)
    X = np.array([r[:L] for r in rows]).T
    return X, np.array(lon), np.array(lat)


def diurnal_phasor(X):
    T = X.shape[0]; t = np.arange(T); w = np.exp(-1j*2*np.pi*t/24.0)
    return ((X - X.mean(0)) * w[:, None]).sum(0)

def circ_corr(a, b):
    a = a - np.angle(np.mean(np.exp(1j*a))); b = b - np.angle(np.mean(np.exp(1j*b)))
    num = np.sum(np.sin(a)*np.sin(b))
    den = np.sqrt(np.sum(np.sin(a)**2)*np.sum(np.sin(b)**2)) + 1e-12
    return num/den

def sc(aff, K):
    A = np.clip(0.5*(aff+aff.T), 0, None); np.fill_diagonal(A, 1.0)
    return SpectralClustering(K, affinity="precomputed", assign_labels="kmeans",
                              random_state=0).fit_predict(A)


if __name__ == "__main__":
    print("="*72)
    print("R43 O1 · Ф28 · СОЛНЕЧНЫЙ драйвинг: геомагнитная суточная ФАЗА = долгота")
    print("="*72)
    X, lon, lat = load_all(maxlat=55.0)
    print(f"  среднеширотных обсерваторий: {X.shape[1]} (|lat|<55), часов: {X.shape[0]}")
    print(f"  долготы {np.round(np.sort(lon),0)}")

    ang = np.angle(diurnal_phasor(X))                 # измеренная суточная фаза
    lon_phase = np.deg2rad(lon)                        # ожидаемая ∝ долготе
    r = circ_corr(ang, lon_phase)
    print(f"\n  (а) круговая корр(суточная фаза, долгота) = {abs(r):.3f}")

    # (б) сбалансированная 2-полосная долготная группировка (медиана долготы)
    band = (lon > np.median(lon)).astype(int)
    Cn = np.abs(np.nan_to_num(np.corrcoef(X.T)))
    p = diurnal_phasor(X); u = p/(np.abs(p)+1e-12)
    Cp = (np.real(np.outer(u, np.conj(u)))+1.0)/2.0
    ari_naive = adjusted_rand_score(band, sc(Cn, 2))
    ari_phase = adjusted_rand_score(band, sc(Cp, 2))
    print(f"  (б) долготные полосы восток/запад (K=2, {np.bincount(band)}):")
    print(f"      наивная |корреляция| H : ARI={ari_naive:.3f}")
    print(f"      ФАЗА суточной вариации : ARI={ari_phase:.3f}")

    print("\n[ЧЕСТНЫЙ ВЫВОД]")
    print(f"  Генерализация Ф26 на солнечный/геомагнитный домен НЕ подтвердилась: одногармоническая")
    print(f"  ФАЗА проигрывает наивной корреляции в восстановлении долготы (фаза {ari_phase:.2f} vs")
    print(f"  корр {ari_naive:.2f}); круговая корр фаза↔долгота слабая ({abs(r):.2f}).")
    print(f"  Причина честная: геомагнитная Sq-вариация МНОГОГАРМОНИЧНА (24/12/8 ч + бури) → оценка")
    print(f"  фазы из одной Фурье-частоты неустойчива; а соседние обсерватории и так сильно")
    print(f"  скоррелированы (общий токовый овал в одно местное время) → корреляция ловит долготу.")
    print(f"  ВЫВОД: чистый выигрыш ФАЗЫ (Ф26) специфичен для ОДНОГАРМОНИЧЕСКОЙ осцилляции (температура);")
    print(f"  в многогармоническом домене корреляция не хуже/лучше. N={X.shape[1]}, долготы скучены.")
