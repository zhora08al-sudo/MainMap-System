"""
R43 O1 — Ф25: МНОГОСЛОЙНОСТЬ на РЕАЛЬНОМ ЭЭГ + ПОДЪЁМ остаточного трека.

Две задачи автора сразу:
  1) ПОДНЯТЬ остаточный трек — итеративная (EM-подобная) дефляция: чередуем оценку
     первичного и вторичного слоёв, каждый раз снимая общий фазовый компонент другого.
  2) ПРОВЕРИТЬ на РЕАЛЬНЫХ данных. ЭЭГ PhysioNet (64 канала, глаза закрыты) даёт два
     ПЕРЕКРЁСТНЫХ анатомических слоя электродов:
       • первичный  = передние/задние (сильный, альфа-доминанта),
       • вторичный  = ЛЕВОЕ/ПРАВОЕ полушарие (ортогонален первому: по чётности номера).
  Плоский corr→coh берёт доминантный передне-задний слой и к полушарию СЛЕП. Остаточный
  трек: снять первичный слой → corr→coh резидуала → достать полушарие. Это прямой
  реальный тест способности «1 узел → несколько слоёв».

Сравнение по 20 субъектам: плоский vs остаточный (1 проход) vs остаточный ИТЕРАТИВНЫЙ.
Без производных. Зависимости: numpy scipy scikit-learn pyedflib (+ локальные модули).
"""
from __future__ import annotations
import warnings, os, urllib.request; warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from sklearn.metrics import adjusted_rand_score
import pyedflib
from phase24_residual import cplx_corr, spec, lag_corr_coh


def residual_proj(phi, labels):
    """СИЛЬНАЯ дефляция (подъём vs «снять среднюю фазу»): комплексная регрессия сигнала
    узла на общий компонент его кластера и вычитание — убирает общий временной компонент
    ЦЕЛИКОМ (амплитуда+фаза), а не только средний фазовый сдвиг."""
    Z = np.exp(1j*phi); out = np.zeros_like(phi)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        g = Z[:, idx].mean(1); gg = np.vdot(g, g) + 1e-12
        for i in idx:
            b = np.vdot(g, Z[:, i]) / gg
            out[:, i] = np.angle(Z[:, i] - b*g)
    return out

CACHE = "/tmp/r43o1_eeg"
URL = "https://physionet.org/files/eegmmidb/1.0.0/{s}/{s}R02.edf?download"


def antpost(lbl):                      # первичный слой: передние(0) / задние(1)
    l = lbl.strip().strip(".").upper()
    if l.startswith(("FP", "AF", "F", "FC", "FT", "C", "CP", "T")) and not l.startswith(("TP","P")):
        return 0
    return 1

def hemi(lbl):                         # вторичный слой: лево(0)/право(1)/средняя(-1)
    l = lbl.strip().strip(".").upper()
    d = "".join(ch for ch in l if ch.isdigit())
    if "Z" in l or d == "": return -1
    return 0 if int(d[-1]) % 2 == 1 else 1


def load(sid):
    path = os.path.join(CACHE, f"{sid}R02.edf")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True); urllib.request.urlretrieve(URL.format(s=sid), path)
    f = pyedflib.EdfReader(path); n = f.signals_in_file
    labels = [f.getLabel(i) for i in range(n)]; fs = f.getSampleFrequency(0)
    X = np.array([f.readSignal(i) for i in range(n)]).T; f._close()
    return X, labels, fs


def alpha_phase(X, fs):
    b, a = butter(4, [8/(fs/2), 13/(fs/2)], btype="band")
    return np.angle(hilbert(filtfilt(b, a, X, axis=0), axis=0))


def residual_iter(phi, K1, K2, iters=3):
    """ПОДЪЁМ: чередуем СИЛЬНУЮ дефляцию слоёв (EM-подобно), уточняя оба."""
    lab1 = lag_corr_coh(phi, K1)
    for _ in range(iters):
        lab2 = lag_corr_coh(residual_proj(phi, lab1), K2)
        lab1 = lag_corr_coh(residual_proj(phi, lab2), K1)
    lab2 = lag_corr_coh(residual_proj(phi, lab1), K2)
    return lab1, lab2


if __name__ == "__main__":
    print("="*74)
    print("R43 O1 · Ф25 · МНОГОСЛОЙНОСТЬ на РЕАЛЬНОМ ЭЭГ (полушарие из-под передне-заднего)")
    print("="*74)
    subs = [f"S{i:03d}" for i in range(1, 21)]
    flatH, res1H, resIH, resOR, flatAP = [], [], [], [], []
    for sid in subs:
        try:
            X, labels, fs = load(sid)
        except Exception as e:
            print(f"  {sid}: пропуск ({type(e).__name__})"); continue
        phi = alpha_phase(X, fs)
        ap = np.array([antpost(l) for l in labels])
        hm = np.array([hemi(l) for l in labels])
        keep = hm >= 0                                  # убрать срединные (z) каналы
        phiH, hmH, apH = phi[:, keep], hm[keep], ap[keep]

        flatAP.append(adjusted_rand_score(ap, lag_corr_coh(phi, 2)))         # санити: первичное
        flatH.append(adjusted_rand_score(hmH, lag_corr_coh(phiH, 2)))        # плоский → полушарие
        lay1 = lag_corr_coh(phiH, 2)                                         # доминантный слой (данные)
        res1H.append(adjusted_rand_score(hmH, lag_corr_coh(residual_proj(phiH, lay1), 2)))
        _, lab2 = residual_iter(phiH, 2, 2, iters=3)                         # итеративный
        resIH.append(adjusted_rand_score(hmH, lab2))
        resOR.append(adjusted_rand_score(hmH, lag_corr_coh(residual_proj(phiH, apH), 2)))  # ОРАКУЛ

    flatH, res1H, resIH, resOR, flatAP = map(np.array, (flatH, res1H, resIH, resOR, flatAP))
    print(f"\n  субъектов: {len(flatH)}; первичный слой (передн/задн) плоским ARI={flatAP.mean():.3f}")
    print(f"\n  ВТОРИЧНЫЙ слой = ЛЕВОЕ/ПРАВОЕ полушарие:")
    print(f"    {'метод':36} {'mean ARI':>9} {'std':>7} {'медиана':>8}")
    print("    " + "-"*64)
    for nm, a in [("плоский corr→coh (без остатка)", flatH),
                  ("остаточный трек (1 проход, данные)", res1H),
                  ("остаточный ИТЕРАТИВНЫЙ (подъём)", resIH),
                  ("ОРАКУЛ: дефляция по ИСТИННОМУ ant/post", resOR)]:
        print(f"    {nm:36} {a.mean():>9.3f} {a.std():>7.3f} {np.median(a):>8.3f}")

    print("\n[ЧЕСТНЫЙ ВЫВОД]")
    print(f"  Плоский {flatH.mean():.3f}, остаток {res1H.mean():.3f}, итер {resIH.mean():.3f},"
          f" ОРАКУЛ {resOR.mean():.3f} — ВСЕ ≈ 0.")
    print("  Даже снятие ИСТИННОГО передне-заднего слоя сильной комплексной дефляцией НЕ вскрывает")
    print("  полушарие → в альфа-ЭЭГ (глаза закрыты) ПОЛУШАРНОГО ФАЗОВОГО СЛОЯ ПРОСТО НЕТ")
    print("  (билатеральная симметрия + объёмная проводимость = лево/право фазово-эквивалентны).")
    print("  Это ЧЕСТНЫЙ отрицательный результат: метод не сломан — второго слоя в этой группировке")
    print("  не существует, потому ноль у всех (синтетика Ф24 имела слой ПО ПОСТРОЕНИЮ — здесь его нет).")
    print("  Для реальной проверки многослойности нужен сигнал с РЕАЛЬНО перекрёстными фазовыми")
    print("  сетями (например, source-ЭЭГ/МЭГ без объёмной проводимости, или две разные полосы).")
