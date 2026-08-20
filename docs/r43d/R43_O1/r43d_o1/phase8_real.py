"""
R43 O1 — Ф8: проверка ниши на НАСТОЯЩИХ топологиях (не синтетический SBM).

Реальные/реалистичные графы с известными сообществами:
  • Zachary Karate Club — реальная соц. сеть, 34 узла, 2 фракции (ground truth).
  • LFR benchmark — стандартный реалистичный генератор сообществ (гетерогенные
    степени/размеры), ground truth известен.

Протокол (всё derivative-free):
  на РЕАЛЬНОЙ матрице смежности запускаем дискретную фазовую динамику (у каждого
  сообщества свой ритм) → наблюдаем зашумлённые фазы → восстанавливаем сообщества
  через PLV-когерентность + спектральную кластеризацию → ARI/NMI против истины.
  Контроль: тот же расчёт через ДИФФЕРЕНЦИАЛЬНУЮ аффинность (ridge на dφ/dt).

Зависимости: numpy, scikit-learn, networkx.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from harden import plv, spec, eigengap_K


def simulate_on_graph(A, comm, K0=1.6, steps=1200, dt=0.05, tail=400,
                      obs_noise=0.0, seed=0):
    """Дискретная derivative-free фазовая динамика на заданной смежности A."""
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    deg = A.sum(1); deg[deg == 0] = 1
    M = len(np.unique(comm))
    band = np.linspace(-1.5, 1.5, M)
    omega = band[comm] + 0.10 * rng.standard_normal(N)
    phi = rng.uniform(-np.pi, np.pi, N)
    Z = np.empty((tail, N), complex); t0 = steps - tail
    for k in range(steps):
        s, c = np.sin(phi), np.cos(phi)
        coup = K0 * (c * (A @ s) - s * (A @ c)) / deg     # Σ_j A_ij sin(φ_j−φ_i)
        phi = phi + dt * (omega + coup)
        if k >= t0:
            Z[k - t0] = np.exp(1j * phi)
    pho = np.angle(Z)
    if obs_noise > 0:
        pho = pho + obs_noise * rng.standard_normal(pho.shape)
    return pho


def affinity_derivative(phi, dt=0.05, alpha=1.0):
    """Дифференциальная аффинность: ridge-регрессия на конечно-разностную dφ/dt."""
    D = np.gradient(phi, dt, axis=0)
    N = phi.shape[1]; C = np.zeros((N, N))
    for i in range(N):
        F = np.sin(phi - phi[:, [i]])
        X = np.hstack([np.ones((F.shape[0], 1)), F])
        coef = np.linalg.solve(X.T @ X + alpha*np.eye(X.shape[1]), X.T @ D[:, i])
        C[i] = coef[1:]
    return np.abs(C) + np.abs(C.T)


def score(comm, lab):
    return adjusted_rand_score(comm, lab), normalized_mutual_info_score(comm, lab)


def run_case(name, A, comm, K):
    print(f"\n--- {name}: N={A.shape[0]}, сообществ={K} ---")
    print(f"  {'σ_obs':>6} {'ARI R43O1':>10} {'NMI R43O1':>10} {'ARI дифф':>9} | eigengap K̂")
    for s in (0.0, 0.2):
        phi = simulate_on_graph(A, comm, obs_noise=s, seed=1)
        aff = plv(phi)
        a_r, n_r = score(comm, spec(aff, K))
        a_d, _ = score(comm, spec(affinity_derivative(phi), K))
        khat = eigengap_K(aff)
        print(f"  {s:>6.1f} {a_r:>10.3f} {n_r:>10.3f} {a_d:>9.3f} |  K̂={khat}")
    return


if __name__ == "__main__":
    print("="*66)
    print("R43 O1 · Ф8 · проверка ниши на НАСТОЯЩИХ топологиях")
    print("="*66)

    # ---- 1. Zachary Karate Club (реальная сеть, 2 фракции) ----
    G = nx.karate_club_graph()
    A = nx.to_numpy_array(G)
    comm = np.array([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in G.nodes])
    run_case("Karate Club (реальная сеть)", A, comm, K=2)

    # ---- 2. LFR benchmark (реалистичные сообщества, ground truth) ----
    try:
        GL = nx.LFR_benchmark_graph(n=250, tau1=3, tau2=1.5, mu=0.1,
                                    average_degree=10, min_community=25, seed=10)
        GL.remove_edges_from(nx.selfloop_edges(GL))
        comms = {}
        for v in GL.nodes:
            c = frozenset(GL.nodes[v]["community"])
            comms.setdefault(c, len(comms))
        lab = np.array([comms[frozenset(GL.nodes[v]["community"])] for v in GL.nodes])
        AL = nx.to_numpy_array(GL)
        run_case(f"LFR benchmark μ=0.1", AL, lab, K=len(comms))
    except Exception as e:
        print(f"\n  LFR пропущен: {e}")

    # ---- 3. LFR посложнее (μ=0.3 — больше межкластерных рёбер) ----
    try:
        GL2 = nx.LFR_benchmark_graph(n=250, tau1=3, tau2=1.5, mu=0.3,
                                     average_degree=10, min_community=25, seed=7)
        GL2.remove_edges_from(nx.selfloop_edges(GL2))
        comms2 = {}
        for v in GL2.nodes:
            c = frozenset(GL2.nodes[v]["community"]); comms2.setdefault(c, len(comms2))
        lab2 = np.array([comms2[frozenset(GL2.nodes[v]["community"])] for v in GL2.nodes])
        run_case(f"LFR benchmark μ=0.3 (труднее)", nx.to_numpy_array(GL2), lab2, K=len(comms2))
    except Exception as e:
        print(f"\n  LFR μ=0.3 пропущен: {e}")

    print("\n[ИТОГ Ф8] Реальные топологии: см. таблицы. Ожидаем — R43 O1 держит структуру")
    print("под шумом, дифференциальный метод деградирует (как и на синтетике).")
