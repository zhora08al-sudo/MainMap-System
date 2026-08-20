"""
R43 O1 — Ф19: РЕАЛЬНЫЙ ЭЭГ (правильный осцилляторный домен для фазовой когерентности).

Данные: EEG Eye State (OpenML 1471), 14 каналов Emotiv, ~128 Гц, 14980 отсчётов.
Это НАСТОЯЩИЕ нейронные осцилляторы (мозговые ритмы) — домен, где фазовая когерентность
физически применима (в отличие от акций, Ф18).

Группировка каналов (ground truth) — порядок Emotiv V1..V14:
  AF3 F7 F3 FC5 T7 P7 O1 | O2 P8 T8 FC6 F4 F8 AF4
  • передние/задние: задние (T7,P7,O1,O2,P8,T8) сильнее синхронизированы по альфа;
  • левое/правое полушарие: V1..V7 = лево, V8..V14 = право.

Тест: альфа-полоса (8–13 Гц) → фаза по Гильберту → восстановить группировку.
Сравнение: R43 O1 (corr→coh) vs наивная корреляция vs дифференциальный (dφ/dt).
Зависимости: numpy, scipy, scikit-learn (fetch_openml — интернет).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from sklearn.datasets import fetch_openml
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from integrate import full_corr_coh

FS = 128.0


def load_eeg():
    d = fetch_openml(data_id=1471, as_frame=True, parser="auto")
    X = d.data.values.astype(float)              # (T, 14)
    # чистка артефактов: z-score по каналу + клип ±5σ (в EEG Eye State есть «попы» сенсора)
    X = (X - np.median(X, 0)) / (1.4826*np.median(np.abs(X-np.median(X,0)),0)+1e-9)
    X = np.clip(X, -5, 5)
    return X


def alpha_phase(X):
    b, a = butter(4, [8/(FS/2), 13/(FS/2)], btype="band")
    Xf = filtfilt(b, a, X, axis=0)
    return np.angle(hilbert(Xf, axis=0))         # (T, 14) фазы альфа


def naive_corr(phi, K):
    C = np.abs(np.nan_to_num(np.corrcoef(np.cos(phi).T))); np.fill_diagonal(C, 1.0)
    return SpectralClustering(K, affinity="precomputed", random_state=0).fit_predict(C)

def deriv(phi, K):
    D = np.gradient(phi, axis=0); N = phi.shape[1]; C = np.zeros((N, N))
    for i in range(N):
        F = np.sin(phi - phi[:, [i]]); Xm = np.hstack([np.ones((F.shape[0],1)), F])
        C[i] = np.linalg.solve(Xm.T@Xm + np.eye(Xm.shape[1]), Xm.T@D[:, i])[1:]
    A = np.abs(C)+np.abs(C.T); np.fill_diagonal(A, A.max())
    return SpectralClustering(K, affinity="precomputed", random_state=0).fit_predict(A)


if __name__ == "__main__":
    print("="*64)
    print("R43 O1 · Ф19 · РЕАЛЬНЫЙ ЭЭГ (осцилляторный домен)")
    print("="*64)
    X = load_eeg(); phi = alpha_phase(X)
    print(f"  каналов: {X.shape[1]}, отсчётов: {X.shape[0]}, полоса альфа 8–13 Гц")

    # две ground-truth группировки
    ant = np.array([0,0,0,0,1,1,1,1,1,1,0,0,0,0])         # 0=перед,1=зад
    hemi = np.array([0]*7 + [1]*7)                         # лево/право

    def sc(lab, gt): return adjusted_rand_score(gt, lab), normalized_mutual_info_score(gt, lab)
    for gtname, gt in [("передние/задние", ant), ("лево/право", hemi)]:
        print(f"\n  Группировка: {gtname}")
        print(f"    {'метод':26} {'ARI':>7} {'NMI':>7}")
        for name, lab in [("R43 O1 (corr→coh)", full_corr_coh(phi, 2, k=5)),
                          ("наивная корреляция", naive_corr(phi, 2)),
                          ("дифференциальный", deriv(phi, 2))]:
            a, n = sc(lab, gt); print(f"    {name:26} {a:>7.3f} {n:>7.3f}")

    print("\n[ВЫВОД]")
    print("  Малое N=14 → результат шумный, но это ПРАВИЛЬНЫЙ домен (нейроосцилляторы).")
    print("  Смотрим, восстанавливает ли фазовая когерентность электродную группировку")
    print("  и бьёт ли corr→coh корреляцию там, где фаза физически осмысленна.")
