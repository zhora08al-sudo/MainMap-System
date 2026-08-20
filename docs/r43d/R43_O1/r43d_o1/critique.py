"""
R43 O1 — Ф9: КОНСТРУКТИВНАЯ САМОКРИТИКА. Нужна ли вообще фазовая когерентность?

Жёсткий вопрос к собственному методу: может, тривиальная корреляция сигналов
восстанавливает сообщества не хуже PLV? Если да — фазовая машинерия R43 O1
избыточна, и это надо признать честно. Очная ставка считывателей-аффинностей
(все derivative-free), одна и та же спектральная кластеризация, под шумом σ_obs=0.2:

  PLV        |⟨e^{i(φi−φj)}⟩|              — фазовая когерентность (R43 O1)
  corr_phase |corr(φ_i, φ_j)|             — корреляция сырых фаз (тривиальный baseline)
  corr_cos   |corr(cos φ_i, cos φ_j)|     — корреляция наблюдаемой волны
  diff       |ridge(dφ/dt)|               — дифференциальный (производный) метод

Датасеты: синтетический SBM, Karate Club (реальный), LFR μ=0.3 (реалистичный).
Зависимости: numpy, scikit-learn, networkx.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_rand_score
from harden import gen, plv, spec
from phase8_real import simulate_on_graph, affinity_derivative


def aff_corr_phase(phi):
    C = np.corrcoef(phi.T)
    return np.abs(np.nan_to_num(C))

def aff_corr_cos(phi):
    C = np.corrcoef(np.cos(phi).T)
    return np.abs(np.nan_to_num(C))

AFFS = {
    "PLV (R43 O1)":   plv,
    "corr_phase":     aff_corr_phase,
    "corr_cos":       aff_corr_cos,
    "diff (dφ/dt)":   affinity_derivative,
}


def evaluate(phi, comm, K):
    out = {}
    for name, fn in AFFS.items():
        try:
            out[name] = adjusted_rand_score(comm, spec(fn(phi), K))
        except Exception:
            out[name] = float("nan")
    return out


def karate():
    G = nx.karate_club_graph(); A = nx.to_numpy_array(G)
    comm = np.array([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in G.nodes])
    return simulate_on_graph(A, comm, obs_noise=0.2, seed=1), comm, 2

def lfr(mu=0.3, seed=7):
    G = nx.LFR_benchmark_graph(n=250, tau1=3, tau2=1.5, mu=mu,
                               average_degree=10, min_community=25, seed=seed)
    G.remove_edges_from(nx.selfloop_edges(G))
    cm = {}
    for v in G.nodes:
        c = frozenset(G.nodes[v]["community"]); cm.setdefault(c, len(cm))
    lab = np.array([cm[frozenset(G.nodes[v]["community"])] for v in G.nodes])
    return simulate_on_graph(nx.to_numpy_array(G), lab, obs_noise=0.2, seed=1), lab, len(cm)


if __name__ == "__main__":
    print("="*72)
    print("R43 O1 · Ф9 · самокритика: нужна ли фазовая когерентность? (σ_obs=0.2)")
    print("="*72)

    cases = []
    phi, comm = gen(150, 3, seed=1, obs_noise=0.2); cases.append(("SBM синт.", phi, comm, 3))
    (phi, comm, K) = karate(); cases.append(("Karate реальн.", phi, comm, K))
    try:
        (phi, comm, K) = lfr(); cases.append(("LFR μ=0.3", phi, comm, K))
    except Exception as e:
        print("  LFR пропущен:", e)

    names = list(AFFS.keys())
    print(f"\n{'датасет':16} " + " ".join(f"{n:>14}" for n in names))
    table = []
    for label, phi, comm, K in cases:
        res = evaluate(phi, comm, K)
        table.append((label, res))
        print(f"{label:16} " + " ".join(f"{res[n]:>14.3f}" for n in names))

    print("\n[КОНСТРУКТИВНЫЙ ВЫВОД]")
    # сравнение PLV vs лучший тривиальный корреляционный
    for label, res in table:
        plv_v = res["PLV (R43 O1)"]
        triv = max(res["corr_phase"], res["corr_cos"])
        diff = res["diff (dφ/dt)"]
        verdict = ("PLV ≳ corr (фаза оправдана)" if plv_v - triv >= 0.05
                   else "corr ≈ PLV (фаза НЕ обязательна тут)" if abs(plv_v-triv) < 0.05
                   else "corr > PLV (тривиальное лучше!)")
        print(f"  {label:16}: PLV={plv_v:.3f} | лучш.corr={triv:.3f} | diff={diff:.3f}  -> {verdict}")
    print("\n  Честно: где corr≈PLV — фазовая когерентность не даёт преимущества над")
    print("  обычной корреляцией; уникальная ценность R43 O1 — derivative-free устойчивость")
    print("  под шумом ОТНОСИТЕЛЬНО ПРОИЗВОДНОГО метода, а не относительно корреляции.")
