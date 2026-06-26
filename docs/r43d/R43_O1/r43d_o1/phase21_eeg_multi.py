"""
R43 O1 — Ф21: СТАТИСТИЧЕСКАЯ РЕПЛИКАЦИЯ на МНОГИХ испытуемых ЭЭГ.

Зачем. Вердикт Ф20 («фаза > корреляция НЕ робастна на реале») стоял на ОДНОМ
субъекте (S001). Один субъект — это шум, а не вывод. Здесь честно реплицируем на
многих субъектах PhysioNet EEGMMI (S001..S0NN, запись R02 — глаза закрыты, сильный
затылочный альфа-ритм) и считаем mean±std ARI и WIN-RATE R43 O1 vs корреляция.

Гипотеза, которую ПРОВЕРЯЕМ (а не подтверждаем): если фазовый край реален, R43 O1
должен бить корреляцию у БОЛЬШИНСТВА субъектов на обеих группировках. Если win-rate
≈ 50% — фазовый край на реальном ЭЭГ действительно НЕ робастен (Ф20 подтверждается).

Без черри-пикинга: считаем ВСЕ субъекты, что скачались, обе группировки, печатаем
полную таблицу и агрегат. Зависимости: numpy scipy scikit-learn pyedflib (+ интернет).
"""
from __future__ import annotations
import warnings, os, urllib.request; warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
import pyedflib
from integrate import full_corr_coh

CACHE = "/tmp/r43o1_eeg"
BASE = "https://physionet.org/files/eegmmidb/1.0.0/{s}/{s}R02.edf?download"
N_SUBJ = 20   # S001..S020


def region(lbl):
    l = lbl.strip().strip(".").upper()
    if l == "IZ" or l.startswith("O") or l.startswith("PO"): return "occipital"
    if l.startswith("TP") or l.startswith("FT") or l.startswith("T"): return "temporal"
    if l.startswith("CP") or l.startswith("FC") or l.startswith("C"): return "central"
    if l.startswith("P"): return "parietal"
    if l.startswith("F") or l.startswith("AF") or l.startswith("FP"): return "frontal"
    return "other"


def load(sid):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{sid}R02.edf")
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE.format(s=sid), path)
    f = pyedflib.EdfReader(path); n = f.signals_in_file
    labels = [f.getLabel(i) for i in range(n)]
    fs = f.getSampleFrequency(0)
    X = np.array([f.readSignal(i) for i in range(n)]).T
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


def run_subject(sid):
    X, regs, fs = load(sid)
    codes = np.unique(regs, return_inverse=True)[1]; K = len(np.unique(codes))
    coarse = np.isin(regs, ["frontal", "central"]).astype(int)
    phi = alpha_phase(X, fs)
    out = {}
    for gtname, gt, Kk in [("fine5", codes, K), ("coarse2", coarse, 2)]:
        out[gtname] = {
            "o1":    adjusted_rand_score(gt, full_corr_coh(phi, Kk, k=8)),
            "corr":  adjusted_rand_score(gt, naive_corr(phi, Kk)),
            "deriv": adjusted_rand_score(gt, deriv(phi, Kk)),
        }
    return out


if __name__ == "__main__":
    print("="*72)
    print("R43 O1 · Ф21 · СТАТИСТИЧЕСКАЯ РЕПЛИКАЦИЯ на многих ЭЭГ-субъектах")
    print("="*72)
    subjects = [f"S{ i:03d}" for i in range(1, N_SUBJ+1)]
    rows = {}
    for sid in subjects:
        try:
            rows[sid] = run_subject(sid)
        except Exception as e:
            print(f"  {sid}: пропуск ({type(e).__name__}: {str(e)[:40]})")
    print(f"\n  успешно обработано субъектов: {len(rows)}")

    for gtname, label in [("fine5", "5 долей (мелко)"), ("coarse2", "передние/задние (грубо)")]:
        o1  = np.array([rows[s][gtname]["o1"]    for s in rows])
        cor = np.array([rows[s][gtname]["corr"]  for s in rows])
        der = np.array([rows[s][gtname]["deriv"] for s in rows])
        print(f"\n  ── Группировка: {label} ──")
        print(f"    {'метод':22} {'mean ARI':>10} {'std':>8} {'медиана':>9}")
        for nm, arr in [("R43 O1 (corr→coh)", o1), ("наивная корреляция", cor), ("дифференциальный", der)]:
            print(f"    {nm:22} {arr.mean():>10.3f} {arr.std():>8.3f} {np.median(arr):>9.3f}")
        win_o1_corr  = float(np.mean(o1  > cor))
        win_o1_der   = float(np.mean(o1  > der))
        win_corr_der = float(np.mean(cor > der))
        print(f"    win-rate  R43O1 > корреляция : {win_o1_corr:.2%}  ({int(win_o1_corr*len(o1))}/{len(o1)})")
        print(f"    win-rate  R43O1 > дифф       : {win_o1_der:.2%}")
        print(f"    win-rate  корреляция > дифф  : {win_corr_der:.2%}")

    print("\n[ЧЕСТНЫЙ ВЫВОД — заполнится фактами после прогона]")
    print("  Решающий критерий: если win-rate(R43O1>корр) около 50% на обеих группировках,")
    print("  фазовый край на реальном ЭЭГ НЕ робастен (Ф20 устоял на многих субъектах).")
    print("  Если устойчиво >>50% хотя бы на мелкой — Ф20 был пессимистичен из-за 1 субъекта.")
    print("  derivative-free тезис: ждём win-rate(*>дифф) близко к 100% на всех субъектах.")
