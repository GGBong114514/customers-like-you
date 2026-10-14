"""评估推荐结果，并检查输入和命中率计算。
Evaluate recommendations and check inputs and hit-rate calculations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd
import scipy.sparse as sp


def validate_recommendations(recs, train, k=10):
    """每个客户需要 k 个不同的、未买过的商品。
    Require k different, unseen products for each customer.
    """
    if recs.shape != (train.shape[0], k):
        raise ValueError(f"Wrong recommendation shape: {recs.shape}")
    if not np.issubdtype(recs.dtype, np.integer):
        raise ValueError("Recommendation indexes must be integers.")
    if not ((recs >= 0) & (recs < train.shape[1])).all():
        raise ValueError("Recommendation index out of range.")
    if (np.diff(np.sort(recs, axis=1), axis=1) == 0).any():
        raise ValueError("Duplicate products in a recommendation row.")
    rows = np.repeat(np.arange(train.shape[0]), k)
    values = np.asarray(train[rows, recs.ravel()]).ravel()
    if (values != 0).any():
        raise ValueError("Recommendations include training purchases.")


def load_test(path, train):
    """测试表只保留匿名编号，并检查它与训练矩阵没有重叠。
    Check anonymous test indexes and exclude overlap with training.
    """
    pairs = pd.read_csv(path)
    if list(pairs.columns) != ["customer_idx", "product_idx"]:
        raise ValueError("Unexpected test columns.")
    if pairs.empty or pairs.isna().any().any() or pairs.duplicated().any():
        raise ValueError("Test pairs are empty, missing or duplicated.")
    for column in pairs.columns:
        if not pd.api.types.is_integer_dtype(pairs[column]):
            raise ValueError("Test indexes must be integers.")
    rows = pairs["customer_idx"].to_numpy()
    columns = pairs["product_idx"].to_numpy()
    if not ((rows >= 0) & (rows < train.shape[0])).all():
        raise ValueError("Test customer out of range.")
    if not ((columns >= 0) & (columns < train.shape[1])).all():
        raise ValueError("Test product out of range.")
    if (np.asarray(train[rows, columns]).ravel() != 0).any():
        raise ValueError("Test pairs overlap training purchases.")
    truth = {}
    for customer, group in pairs.groupby("customer_idx", sort=True):
        truth[int(customer)] = set(group["product_idx"].tolist())
    return pairs, truth


def hit_rate(recs, truth):
    """一个客户命中至少一件测试商品，就计为一次命中。
    Count a customer as a hit if any test product is recommended.
    """
    if not truth or any(not items for items in truth.values()):
        raise ValueError("Each evaluated customer needs a test purchase.")
    hits = 0
    for customer, items in truth.items():
        recommended = set(recs[customer].tolist())
        if recommended.intersection(items):
            hits += 1
    return hits, hits / len(truth)


def best_sellers(train, k=10):
    """按训练数量排序，同分按商品编号排序，再排除已买商品。
    Rank training quantities, break ties by product index, then exclude seen items.
    """
    totals = np.asarray(train.sum(axis=0)).ravel()
    product_indexes = np.arange(train.shape[1])
    ranked = np.lexsort((product_indexes, -totals))
    result = np.empty((train.shape[0], k), dtype=np.int64)
    for customer in range(train.shape[0]):
        seen = set(train[customer].indices.tolist())
        selected = []
        for product in ranked:
            if product not in seen:
                selected.append(product)
            if len(selected) == k:
                break
        if len(selected) != k:
            raise ValueError("Not enough unseen products.")
        result[customer] = selected
    return result


def random_recommendations(train, seed=42, k=10):
    """从未买过的商品中随机抽取，不重复。
    Sample unseen products without replacement.
    """
    rng = np.random.default_rng(seed)
    products = np.arange(train.shape[1])
    result = np.empty((train.shape[0], k), dtype=np.int64)
    for customer in range(train.shape[0]):
        candidates = np.setdiff1d(products, train[customer].indices)
        result[customer] = rng.choice(candidates, size=k, replace=False)
    return result


def check_metric(train, truth):
    """先用手算例子，再用随机和必命中的推荐检查函数。
    Check a hand-worked example, random recommendations and guaranteed hits.
    """
    toy = np.array([[1, 2], [0, 2], [0, 1]])
    toy_truth = {0: {1, 3}, 1: {4}}
    assert hit_rate(toy, toy_truth) == (1, 0.5)
    assert hit_rate(np.array([[1], [0]]), {0: {4}, 1: {4}}) == (0, 0.0)
    # 基线不能使用测试数据，同分时按编号选。
    # The baseline uses no test data and resolves ties by index.
    small = sp.csr_matrix([[2, 0, 0, 0], [0, 2, 0, 0]])
    assert best_sellers(small, k=2).tolist() == [[1, 2], [0, 2]]
    valid = np.array([[1, 2], [0, 2]])
    validate_recommendations(valid, small, k=2)
    bad_inputs = [valid.astype(float), valid[:, :1],
                  np.array([[1, 1], [0, 2]]), np.array([[1, 4], [0, 2]]),
                  np.array([[0, 2], [0, 2]])]
    for bad in bad_inputs:
        try:
            validate_recommendations(bad, small, k=2)
        except ValueError:
            continue
        raise AssertionError("Invalid recommendations were accepted.")

    random_recs = random_recommendations(train)
    validate_recommendations(random_recs, train)
    random_hits, random_rate = hit_rate(random_recs, truth)
    probabilities = []
    for customer, items in truth.items():
        available = train.shape[1] - train[customer].nnz
        misses = available - len(items)
        no_hit = 1.0
        for draw in range(10):
            no_hit *= max(0, misses - draw) / (available - draw)
        probabilities.append(1 - no_hit)
    probabilities = np.array(probabilities)
    expected = float(probabilities.mean())
    standard_error = math.sqrt(float(np.sum(probabilities * (1 - probabilities)))) / len(truth)
    if abs(random_rate - expected) > 5 * standard_error:
        raise AssertionError("Random hit rate differs greatly from its expectation.")

    # 只用于测评估函数，不保存为模型结果。
    # Use test-aware recommendations only to check the metric, not as model results.
    oracle = random_recs.copy()
    for customer, items in truth.items():
        chosen = min(items)
        if chosen not in oracle[customer]:
            oracle[customer, 0] = chosen
    validate_recommendations(oracle, train)
    oracle_hits, oracle_rate = hit_rate(oracle, truth)
    assert oracle_rate == 1.0
    return {
        "toy_metric_checks": "passed", "validator_negative_checks": "passed",
        "baseline_tie_and_exclusion_check": "passed", "random_seed": 42,
        "random_hits": random_hits, "random_hit_rate_at_10": random_rate,
        "random_expected_hit_rate_at_10": expected,
        "random_standard_error": standard_error,
        "oracle_hits": oracle_hits, "oracle_hit_rate_at_10": oracle_rate,
    }


def concentration(recs, customers, products, method, cohort):
    """统计推荐覆盖范围和最常出现的商品。
    Count catalogue coverage and the most frequent recommendations.
    """
    selected = recs[customers]
    counts = np.bincount(selected.ravel(), minlength=len(products))
    ordered = np.lexsort((np.arange(len(products)), -counts))[:20]
    top = products.iloc[ordered][["product_idx", "StockCode", "Description"]].copy()
    top.insert(0, "method", method)
    top.insert(1, "cohort", cohort)
    top["recommended_customers"] = counts[ordered]
    top["customer_share_pct"] = counts[ordered] / len(customers) * 100
    summary = {"method": method, "cohort": cohort, "customers": len(customers),
               "distinct_recommended_products": int((counts > 0).sum()),
               "catalogue_coverage_pct": float((counts > 0).mean() * 100),
               "top_product_customer_share_pct": float(counts.max() / len(customers) * 100),
               "top_product_idx": int(ordered[0]),
               "distinct_unordered_lists": len(np.unique(np.sort(selected, axis=1), axis=0))}
    return summary, top


def evaluate(root, recs_dir, out):
    """同一批测试客户用于所有方法，结果只输出汇总。
    Evaluate every method on the same test customers and export aggregates.
    """
    data = root / "data/processed/handoff"
    train = sp.load_npz(data / "A_train_count.npz")
    if not sp.isspmatrix_csr(train) or not np.isfinite(train.data).all() or (train.data <= 0).any():
        raise ValueError("Expected a positive finite CSR count matrix.")
    pairs, truth = load_test(data / "private/test.csv", train)
    products = pd.read_csv(data / "products.csv", dtype={"StockCode": "string"})
    if products["product_idx"].tolist() != list(range(train.shape[1])):
        raise ValueError("Product map does not match matrix columns.")
    print("Checking metric and validation rules...", flush=True)
    checks = check_metric(train, truth)
    methods = {"SVD_log": "recs_svd.npy", "PCA_log": "recs_pca.npy",
               "SVD_count": "recs_count.npy", "SVD_binary": "recs_binary.npy"}
    recommendations = {}
    hashes = {}
    for method, filename in methods.items():
        path = recs_dir / filename
        recs = np.load(path, allow_pickle=False)
        validate_recommendations(recs, train)
        recommendations[method] = recs
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    recommendations["Best_seller_count"] = best_sellers(train)
    validate_recommendations(recommendations["Best_seller_count"], train)
    baseline_hits, baseline_rate = hit_rate(recommendations["Best_seller_count"], truth)
    results, diagnostics, tops = [], [], []
    for method, recs in recommendations.items():
        hits, rate = hit_rate(recs, truth)
        results.append({"method": method, "test_customers": len(truth),
                        "hit_customers": hits, "hit_rate_at_10": rate,
                        "hit_rate_pct": rate * 100,
                        "difference_vs_baseline_pp": (rate - baseline_rate) * 100})
        for cohort, customers in [("all_train", np.arange(train.shape[0])),
                                  ("evaluated", np.array(sorted(truth)))]:
            summary, top = concentration(recs, customers, products, method, cohort)
            diagnostics.append(summary)
            tops.append(top)
    out.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(results)
    results.to_csv(out / "evaluation_results.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "recommendation_diagnostics.csv", index=False)
    pd.concat(tops, ignore_index=True).to_csv(out / "top_recommended_products.csv", index=False)
    (out / "metric_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    input_hashes = {}
    for name in ["A_train_count.npz", "A_train_log.npz", "A_train_binary.npz", "products.csv", "customers.csv", "private/test.csv"]:
        input_hashes[name] = hashlib.sha256((data / name).read_bytes()).hexdigest()
    metadata = {"definition": "Customers with at least one held-out item in top 10 / evaluated customers",
                "test_customers": len(truth), "test_pairs": len(pairs), "matrix_shape": list(train.shape),
                "baseline": "Total training quantity, descending; ties by ascending product_idx; exclude seen items",
                "validation": "All four files passed shape, integer, range, uniqueness and unseen-item checks",
                "recommendation_sha256": hashes, "input_sha256": input_hashes,
                "limitations": ["Saved arrays do not prove score order or training-data provenance.",
                                "High concentration or high hit rate alone does not prove leakage.",
                                "Final test results must not be used to tune k or cleaning rules."]}
    (out / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(results.to_string(index=False))
    print(json.dumps(checks, indent=2))


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recs-dir", type=Path, default=root / "B_to_A/B_to_A")
    parser.add_argument("--output", type=Path, default=root / "reports/evaluation")
    args = parser.parse_args()
    evaluate(root, args.recs_dir, args.output)


if __name__ == "__main__":
    main()
