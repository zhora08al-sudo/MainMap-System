"""
R43 O1 — Ф23: ВЛОЖЕННАЯ КОМПОЗИЦИЯ масштабов (идея автора).

Замысел автора (дословно): начинаешь с СЕЛЕКТИВНОЙ корреляции → селективной когеренции,
далее ЛОКАЛЬНАЯ корреляция → локальная когеренция, потом ГЛОБАЛЬНАЯ корреляция →
глобальная когеренция, потом МЕТА-корреляция → мета-когеренция. Вложенная композиция:
выход каждого масштаба — вход следующего. Комплексная (лаг-инвариантная) корреляция
подключается на дальних (глобальный/мета) уровнях.

Проверяем на ИЕРАРХИЧЕСКИХ данных: мета-сообщества, каждое из суб-сообществ.
  • суб-сообщество: узлы делят общий фазовый процесс (сильная когеренция) + ЛАГ-сдвиги;
  • мета-сообщество: суб-сообщества частично делят мета-процесс (слабее);
  • разные мета — независимы.
Истина на ДВУХ уровнях: суб (тонко, K=6) и мета (грубо, K=2).

Тест: восстанавливает ли ВЛОЖЕННАЯ лестница ОБА уровня лучше, чем ПЛОСКИЕ одномасштабные
методы (наивная корреляция; лаг-устойчивый corr→coh)? Особенно мета-уровень, который на
уровне узлов слаб, но проявляется после агрегации в супер-узлы.
Без производных. Зависимости: numpy scipy scikit-learn (+ integrate, harden).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from integrate import _knn_sym
from harden import plv

rng = np.random.default_rng(23)


# ───────────────────────── данные: иерархия фаз ─────────────────────────
def make_hier(N_sub=15, n_sub=6, n_meta=2, T=3000, lag=0.6, f0=0.05,
              a_meta=0.55, jitter=0.15, seed=0):
    r = np.random.default_rng(seed)
    sub_per_meta = n_sub // n_meta
    meta_of_sub = np.repeat(np.arange(n_meta), sub_per_meta)
    N = N_sub * n_sub
    sub = np.repeat(np.arange(n_sub), N_sub)
    meta = meta_of_sub[sub]
    t = np.arange(T)
    meta_proc = [np.cumsum(r.normal(0, 0.3, T))*0.02 for _ in range(n_meta)]
    sub_proc = []
    for c in range(n_sub):
        own = np.cumsum(r.normal(0, 0.3, T))*0.02
        sub_proc.append(a_meta*meta_proc[meta_of_sub[c]] + (1-a_meta)*own)
    offs = r.uniform(-lag, lag, N)               # лаг-сдвиги внутри суб-сообщества
    phi = np.zeros((T, N))
    for i in range(N):
        drift = np.cumsum(r.normal(0, jitter, T))*0.02
        phi[:, i] = 2*np.pi*f0*t + sub_proc[sub[i]] + offs[i] + drift
    return phi, sub, meta


# ───────────────────────── строительные блоки ─────────────────────────
def corr_cos(phi):                                # дешёвая РЕАЛЬНАЯ корреляция (лаг-чувств.)
    C = np.abs(np.nan_to_num(np.corrcoef(np.cos(phi).T))); np.fill_diagonal(C, 1.0); return C

def cplx_corr(Z):                                 # КОМПЛЕКСНАЯ корреляция — ЛАГ-инвариантна
    Zc = Z - Z.mean(0); G = Zc.conj().T @ Zc
    d = np.sqrt(np.real(np.diag(G))); d[d == 0] = 1.0
    C = np.abs(G) / np.outer(d, d); np.fill_diagonal(C, 1.0); return C

def spec(W, K):
    A = np.clip(0.5*(W+W.T), 0, None); np.fill_diagonal(A, 1.0)
    return SpectralClustering(K, affinity="precomputed", assign_labels="kmeans",
                              random_state=0).fit_predict(A)


# ───────────────────── ПЛОСКИЕ базлайны (для сравнения) ─────────────────────
def flat_corr(phi, K):                            # наивная корреляция
    return spec(corr_cos(phi), K)

def flat_lag_corr_coh(phi, K, k=8):               # лаг-устойч. corr→coh из Ф22 (один масштаб)
    Z = np.exp(1j*phi); mask = _knn_sym(cplx_corr(Z), k) > 0
    P = plv(phi); W = np.where(mask, np.clip(P, 0, None), 0.0)
    return spec(W, K)


# ───────────────────── ВЛОЖЕННАЯ ЛЕСТНИЦА (идея автора) ─────────────────────
def nested(phi, Kfine, Kmeta, k=8):
    """8 ступеней: селект.корр→селект.коген→локал.корр→локал.коген→глоб.корр→глоб.коген
       →мета.корр→мета.коген. Возвращает (тонкие метки, мета-метки)."""
    Z = np.exp(1j*phi)
    # (1) СЕЛЕКТИВНАЯ корреляция: top-k кандидатов по дешёвой cos-корреляции
    mask = _knn_sym(corr_cos(phi), k) > 0
    # (2) СЕЛЕКТИВНАЯ когеренция: вес кандидатов по PLV
    P = plv(phi)
    W = np.where(mask, np.clip(P, 0, None), 0.0); np.fill_diagonal(W, 1.0)
    # (3) ЛОКАЛЬНАЯ корреляция: 2-хоп распространение (согласованность соседей соседей)
    Wn = W / (W.sum(1, keepdims=True) + 1e-9); L2 = Wn @ Wn
    # (4) ЛОКАЛЬНАЯ когеренция: усиливаем локально-согласованные рёбра (на той же PLV-массе)
    Wlocal = W * (1.0 + L2); Wlocal = 0.5*(Wlocal + Wlocal.T)
    # (5) ГЛОБАЛЬНАЯ корреляция: КОМПЛЕКСНАЯ (лаг-инвариантная) — дальние согласования
    Cc = cplx_corr(Z)
    # (6) ГЛОБАЛЬНАЯ когеренция: спектральное вложение комбинированной аффинности → тонкие
    Wg = Wlocal * Cc
    fine = spec(Wg, Kfine)
    # (7) МЕТА корреляция: супер-узлы = средний аналитический сигнал кластера; компл. корр
    labs = np.unique(fine)
    Zsup = np.stack([Z[:, fine == c].mean(1) for c in labs], axis=1)   # (T, |labs|)
    phisup = np.angle(Zsup)
    Kmeta_eff = min(Kmeta, len(labs))
    # (8) МЕТА когеренция: PLV супер-узлов; кластеризация супер-узлов
    Wm = cplx_corr(Zsup) * plv(phisup)
    metasup = spec(Wm, Kmeta_eff) if len(labs) > Kmeta_eff else np.arange(len(labs))
    meta = np.zeros_like(fine)
    for ci, c in enumerate(labs): meta[fine == c] = metasup[ci]
    return fine, meta


# ───────────── ВЛОЖЕННАЯ ЛЕСТНИЦА v2: ЛАГ-ИНВАРИАНТНЫЙ отбор на КАЖДОМ масштабе ─────────────
def nested_v2(phi, Kfine, Kmeta, k=8):
    """Урок Ф22: под лагом отбор обязан быть лаг-инвариантным НА ВСЕХ ступенях.
       Селективная корреляция = |комплексная корр| (а не cos). Остальное как в v1."""
    Z = np.exp(1j*phi)
    Cc = cplx_corr(Z)                                   # лаг-инвариантная корреляция
    mask = _knn_sym(Cc, k) > 0                          # (1) селективная корр (лаг-инвар.)
    P = plv(phi)                                        # (2) селективная когеренция
    W = np.where(mask, np.clip(P, 0, None), 0.0); np.fill_diagonal(W, 1.0)
    Wn = W / (W.sum(1, keepdims=True) + 1e-9); L2 = Wn @ Wn   # (3) локальная корр (2-хоп)
    Wlocal = W * (1.0 + L2); Wlocal = 0.5*(Wlocal + Wlocal.T)  # (4) локальная когеренция
    Wg = Wlocal * Cc                                    # (5)(6) глоб. корр(компл.)→спектр
    fine = spec(Wg, Kfine)
    labs = np.unique(fine)
    Zsup = np.stack([Z[:, fine == c].mean(1) for c in labs], axis=1)
    Kmeta_eff = min(Kmeta, len(labs))
    Wm = cplx_corr(Zsup) * plv(np.angle(Zsup))          # (7) мета корр × (8) мета коген
    metasup = spec(Wm, Kmeta_eff) if len(labs) > Kmeta_eff else np.arange(len(labs))
    meta = np.zeros_like(fine)
    for ci, c in enumerate(labs): meta[fine == c] = metasup[ci]
    return fine, meta


def sweep_ameta(ametas, reps=5, Kfine=6, Kmeta=2):
    print(f"\n  ── СВИП по силе МЕТА-связи a_meta (где вложенность нужна?) ──")
    print(f"  {'a_meta':>7} | {'плоский мета':>13} {'вложенный мета':>15} | {'плоский тонкий':>14} {'вложен.тонкий':>14}")
    print("  " + "-"*72)
    for am in ametas:
        fm_flat, fm_nest, ff_flat, ff_nest = [], [], [], []
        for r in range(reps):
            phi, sub, meta = make_hier(a_meta=am, seed=r)
            fine_n, meta_n = nested_v2(phi, Kfine, Kmeta)
            fm_flat.append(adjusted_rand_score(meta, flat_lag_corr_coh(phi, Kmeta)))
            fm_nest.append(adjusted_rand_score(meta, meta_n))
            ff_flat.append(adjusted_rand_score(sub, flat_lag_corr_coh(phi, Kfine)))
            ff_nest.append(adjusted_rand_score(sub, fine_n))
        print(f"  {am:>7.2f} | {np.mean(fm_flat):>13.3f} {np.mean(fm_nest):>15.3f} |"
              f" {np.mean(ff_flat):>14.3f} {np.mean(ff_nest):>14.3f}")


# ─────────────────────────────── прогон ───────────────────────────────
if __name__ == "__main__":
    print("="*74)
    print("R43 O1 · Ф23 · ВЛОЖЕННАЯ композиция масштабов на ИЕРАРХИЧЕСКИХ данных")
    print("="*74)
    Kfine, Kmeta = 6, 2
    reps = 5
    rows = {"наивная корр (плоско)": ([], []),
            "corr→coh лаг (плоско)": ([], []),
            "ВЛОЖЕННАЯ лестница":   ([], [])}
    for r in range(reps):
        phi, sub, meta = make_hier(seed=r)
        # тонкий уровень
        rows["наивная корр (плоско)"][0].append(adjusted_rand_score(sub, flat_corr(phi, Kfine)))
        rows["corr→coh лаг (плоско)"][0].append(adjusted_rand_score(sub, flat_lag_corr_coh(phi, Kfine)))
        fine, mlab = nested(phi, Kfine, Kmeta)
        rows["ВЛОЖЕННАЯ лестница"][0].append(adjusted_rand_score(sub, fine))
        # мета уровень
        rows["наивная корр (плоско)"][1].append(adjusted_rand_score(meta, flat_corr(phi, Kmeta)))
        rows["corr→coh лаг (плоско)"][1].append(adjusted_rand_score(meta, flat_lag_corr_coh(phi, Kmeta)))
        rows["ВЛОЖЕННАЯ лестница"][1].append(adjusted_rand_score(meta, mlab))

    print(f"\n  данные: 2 мета × 3 суб × 15 узлов = 90, лаг внутри суб, повторов={reps}")
    print(f"\n  {'метод':24} {'ARI тонкий (суб,K=6)':>22} {'ARI мета (K=2)':>16}")
    print("  " + "-"*64)
    for nm, (fineA, metaA) in rows.items():
        print(f"  {nm:24} {np.mean(fineA):>14.3f}±{np.std(fineA):.2f}   {np.mean(metaA):>10.3f}±{np.std(metaA):.2f}")

    # v2: лаг-инвариантная лестница на тех же данных
    v2f, v2m = [], []
    for r in range(reps):
        phi, sub, meta = make_hier(seed=r)
        f2, m2 = nested_v2(phi, Kfine, Kmeta)
        v2f.append(adjusted_rand_score(sub, f2)); v2m.append(adjusted_rand_score(meta, m2))
    print(f"  {'ВЛОЖЕННАЯ v2 (лаг-инв.)':24} {np.mean(v2f):>14.3f}±{np.std(v2f):.2f}"
          f"   {np.mean(v2m):>10.3f}±{np.std(v2m):.2f}")

    print("\n[ЧЕСТНЫЙ ВЫВОД — v1 vs v2 vs плоский]")
    bf = max(np.mean(rows["наивная корр (плоско)"][0]), np.mean(rows["corr→coh лаг (плоско)"][0]))
    bm = max(np.mean(rows["наивная корр (плоско)"][1]), np.mean(rows["corr→coh лаг (плоско)"][1]))
    print(f"  v1 (cos-отбор) ПРОВАЛ: тонкий {np.mean(rows['ВЛОЖЕННАЯ лестница'][0]):.2f}/"
          f"мета {np.mean(rows['ВЛОЖЕННАЯ лестница'][1]):.2f} — лаг сломал первый отбор (урок Ф22).")
    print(f"  v2 (лаг-инв. отбор): тонкий {np.mean(v2f):.2f} / мета {np.mean(v2m):.2f};"
          f" лучший плоский: тонкий {bf:.2f} / мета {bm:.2f}.")

    sweep_ameta([0.25, 0.40, 0.55, 0.70, 0.85])
    print("\n[ИТОГ Ф23]")
    print("  • Лаг-инвариантный отбор обязателен НА ВСЕХ масштабах (v1→v2: 0.07→0.33) — урок Ф22 ✓.")
    print("  • Многомасштабная ВЛОЖЕННОСТЬ на ТОЧНОСТИ выигрыша НЕ дала: тонкий — паритет с плоским,")
    print("    мета — ХУЖЕ плоского во всём свипе (супер-узлы каскадно наследуют ошибки тонкого).")
    print("  • Прямой лаг-инвар. corr→coh на нужном K — достаточен и устойчивее на иерархии.")
    print("  • Остаточная ценность вложенности — НЕ точность, а (а) дешёвые ранние ступени для")
    print("    масштаба (Ф16) и (б) явная ИЕРАРХИЯ/дендрограмма как интерпретация. Не приукрашиваю.")
