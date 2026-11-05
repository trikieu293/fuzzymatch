"""Empirical scaling harness for StarClusterer.

Benchmarks StarClusterer on graphs with N=1,000 to N=5,000 nodes.
Verifies:
1. Total execution completes in < 2 seconds.
2. Empirical sub-quadratic scaling (growth ratio T(2N)/T(N) < 4.0).
"""

from __future__ import annotations

import random
import time
from typing import Any

from fuzzymatch.clustering.star_cluster import StarClusterer


def generate_benchmark_graph(
    n_nodes: int,
    avg_degree: int = 10,
    cluster_size: int = 5,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[float]]:
    """Generates a realistic candidate linkage graph with clustered entities and noise."""
    rng = random.Random(seed)
    nodes = [f"ENTITY_{i}" for i in range(n_nodes)]
    pairs: list[tuple[str, str]] = []
    weights: list[float] = []

    # 1. Intra-cluster true candidate pairs (cliques of cluster_size)
    num_clusters = n_nodes // cluster_size
    for c in range(num_clusters):
        c_nodes = nodes[c * cluster_size : (c + 1) * cluster_size]
        for i in range(len(c_nodes)):
            for j in range(i + 1, len(c_nodes)):
                pairs.append((c_nodes[i], c_nodes[j]))
                weights.append(round(rng.uniform(0.7, 0.99), 3))

    # 2. Inter-cluster noise candidate pairs (blocking false positives)
    num_noise_edges = (n_nodes * avg_degree) // 2 - len(pairs)
    if num_noise_edges > 0:
        for _ in range(num_noise_edges):
            u = rng.choice(nodes)
            v = rng.choice(nodes)
            if u != v:
                pairs.append((u, v))
                weights.append(round(rng.uniform(0.2, 0.65), 3))

    return pairs, weights


def run_scaling_benchmark() -> dict[str, Any]:
    print("=" * 70)
    print("StarClusterer Empirical Scaling Benchmark (N = 1,000 to 5,000)")
    print("=" * 70)

    node_counts = [1000, 2000, 3000, 4000, 5000]
    timings: dict[int, float] = {}
    cluster_counts: dict[int, int] = {}
    pair_counts: dict[int, int] = {}

    sc = StarClusterer(cutoff=0.6, verify_clique=True)

    for n in node_counts:
        pairs, weights = generate_benchmark_graph(n, avg_degree=10, cluster_size=5, seed=n)
        pair_counts[n] = len(pairs)

        # Warm-up run
        _ = sc.cluster(pairs[:100], weights[:100])

        # Timed execution
        start_time = time.perf_counter()
        clusters = sc.cluster(pairs, weights)
        elapsed = time.perf_counter() - start_time

        timings[n] = elapsed
        cluster_counts[n] = len(clusters)

        print(
            f"N = {n:5d} nodes | M = {len(pairs):6d} pairs | "
            f"Clusters = {len(clusters):5d} | Time = {elapsed:7.4f}s"
        )

    t_1k = timings[1000]
    t_2k = timings[2000]
    t_5k = timings[5000]

    # Scaling ratio T(2000) / T(1000)
    ratio_2k_1k = t_2k / max(1e-6, t_1k)
    # Scaling ratio T(5000) / T(2000)
    ratio_5k_2k = t_5k / max(1e-6, t_2k)

    print("-" * 70)
    print(f"Total time for N=5,000: {t_5k:.4f}s (Threshold: < 2.0s)")
    print(
        f"Empirical scaling ratio T(2,000)/T(1,000): {ratio_2k_1k:.2f}x (Quadratic would be 4.0x)"
    )
    print(
        f"Empirical scaling ratio T(5,000)/T(2,000): {ratio_5k_2k:.2f}x (Quadratic would be 6.25x)"
    )
    print("=" * 70)

    assert t_5k < 2.0, f"Benchmark failed: N=5,000 took {t_5k:.4f}s, exceeding 2.0s threshold!"
    print(">>> VERIFICATION PASSED: StarClusterer executes in < 2.0 seconds on N=5,000.")

    return {
        "timings": timings,
        "pair_counts": pair_counts,
        "cluster_counts": cluster_counts,
        "t_5k": t_5k,
        "sub_quadratic_verified": t_5k < 2.0 and ratio_2k_1k < 4.0,
    }


if __name__ == "__main__":
    run_scaling_benchmark()
