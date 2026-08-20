"""
R43 O1 — Ф20: ПЛОТНЫЙ реальный ЭЭГ (усиление V1, N=64).

Данные: PhysioNet EEGMMI, 64 канала, fs=160 Гц, стандартный монтаж 10-10.
Берём запись R02 (ГЛАЗА ЗАКРЫТЫ → сильный затылочный альфа-ритм).
Ground truth — группировка каналов по 5 долям мозга (из имён электродов):
  frontal / central / temporal / parietal / occipital.
Соседние/однодолевые электроды сильнее фазово-когерентны (региональная синхрония +
объёмная проводимость = общий драйвер) → это наша ниша.

Сравнение восстановления долей: R43 O1 (corr→coh) vs наивная корреляция vs дифф.
Зависимости: numpy, scipy, scikit-learn, pyedflib (+ интернет: physionet.org).
"""
from __future__ import annotations
import warnings, os, urllib.request; warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pyedflib
from integrate import full_corr_coh

CACHE = "/tmp/r43o1_eeg"
URL = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R02.edf?download"  # eyes closed


def region(lbl):
    l = lbl.strip().strip(".").upper()
    if l == "IZ" or l.startswith("O") or l.startswith("PO"): return "occipital"
    if l.startswith("TP") or l.startswith("FT") or l.startswith("T"): return "temporal"
    if l.startswith("CP") or l.startswith("FC") or l.startswith("C"): return "central"
    if l.startswith("P"): return "parietal"
    if l.startswith("F") or l.startswith("AF") or l.startswith("FP"): return "frontal"
    return "other"


def load():
    os.makedirs(CACHE, exist_ok=True); path = os.path.join(CACHE, "S001R02.edf")
    if not os.path.exists(path):
        print("  скачиваю ЭЭГ..."); urllib.request.urlretrieve(URL, path)
    f = pyedflib.EdfReader(path); n = f.signals_in_file
    labels = [f.getLabel(i) for i in range(n)]
    fs = f.getSampleFrequency(0)
    X = np.array([f.readSignal(i) for i in range(n)]).T   # (T, n)
    f._close()
    regs = np.array([region(l) for l in labels])
    return X, regs, fs


def alpha_phase(X, fs):
    b, a = butter(4, [8/(fs/2), 13/(fs/2)], btype="band")
    Xf = filtfilt(b, a, X, axis=0)
    return np.angle(hilbert(Xf, axis=0))


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
    print("R43 O1 · Ф20 · ПЛОТНЫЙ реальный ЭЭГ 64 канала (V1, глаза закрыты)")
    print("="*64)
    X, regs, fs = load()
    from collections import Counter
    print(f"  каналов: {X.shape[1]}, отсчётов: {X.shape[0]}, fs={fs}")
    print(f"  доли (ground truth): {dict(Counter(regs))}")
    codes = np.unique(regs, return_inverse=True)[1]; K = len(np.unique(codes))
    phi = alpha_phase(X, fs)

    coarse = np.isin(regs, ["frontal", "central"]).astype(int)   # передние vs задние
    for gtname, gt, Kk in [("5 долей (мелко)", codes, K), ("передние/задние (грубо)", coarse, 2)]:
        print(f"\n  Группировка: {gtname}")
        print(f"    {'метод':26} {'ARI':>7} {'NMI':>7}")
        res = {}
        for name, lab in [("R43 O1 (corr→coh)", full_corr_coh(phi, Kk, k=8)),
                          ("наивная корреляция", naive_corr(phi, Kk)),
                          ("дифференциальный", deriv(phi, Kk))]:
            a = adjusted_rand_score(gt, lab); n = normalized_mutual_info_score(gt, lab)
            res[name] = a; print(f"    {name:26} {a:>7.3f} {n:>7.3f}")

    print("\n[ЧЕСТНЫЙ ВЫВОД]")
    print("  Устойчиво: дифференциальный метод СТАБИЛЬНО ХУЖЕ всех (derivative-free тезис ✓).")
    print("  НЕустойчиво: фазовая когеренция vs простая корреляция — СМЕШАННО: R43 O1")
    print("  выигрывает на мелкой 5-долевой группировке, но проигрывает на грубой 2-долевой.")
    print("  Значит фазовый край НЕ робастен на реальном ЭЭГ (в отличие от синтетики Курамото);")
    print("  N=14 (Ф19) был слишком мал/удачен. Честная V1: derivative-free>дифф — да;")
    print("  фаза>корреляция — не доказано робастно на реальных данных.")
