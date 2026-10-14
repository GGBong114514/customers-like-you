# Recommendation evaluation

## Results
All methods use the same 1,906 customers with at least one eligible held-out purchase. The test set contains 44,578 distinct customer-product pairs. Customers with no eligible test purchase are excluded from the denominator. Each customer counts once, even when several recommended products match.

| Method | Customers with a hit | Hit rate at 10 | Difference from baseline (percentage points) |
|---|---:|---:|---:|
| SVD_log | 448 / 1906 | 23.50% | +6.14 |
| PCA_log | 432 / 1906 | 22.67% | +5.30 |
| SVD_count | 272 / 1906 | 14.27% | -3.10 |
| SVD_binary | 467 / 1906 | 24.50% | +7.14 |
| Best_seller_count | 331 / 1906 | 17.37% | +0.00 |

The main SVD-log method exceeds the baseline by 6.14 percentage points. PCA-log also exceeds it. Count is below the baseline by 3.10 points. Binary has the highest observed rate, 1.00 point above log (19 more customers); this is a descriptive difference, not a significance claim. Keep the planned log result as the main result and report count/binary under purchase-value sensitivity. These final test results should not be used for further tuning or cleaning changes.

## Baseline and metric
Best sellers are ranked by total retained training quantity from A_train_count.npz. Equal totals are ordered by product_idx. For each customer, skip all training purchases and take the first 10 remaining products. Popularity uses training data only. Ranking by buyer count would be a different baseline.

Hit rate at 10 = customers with at least one held-out item in their top 10 / customers with eligible held-out items. It is not the percentage of test purchases recovered and not precision at 10.

## Input checks
All four recommendation files passed:
- Shape (5463, 10), integer type, product indexes in range.
- Ten distinct products per customer.
- No recommendation is a retained training purchase for that customer.

The test pairs are unique, have valid indexes and do not overlap retained training purchases. The notebook describes k = 100, training-only product similarities and exclusion of seen products. Its stored outputs contain no errors. Static inspection found no reads of held-out data in the model code. The returned README uses the expected shape and counts, but no training-input checksum was supplied by B, so exact training-file provenance is not independently established. Saved arrays alone also cannot prove descending score order. The supplied notebook was inspected, not rerun.

## Metric checks
- Hand-worked example: 1 hit out of 2 eligible customers = 50%; an ineligible extra row does not change the denominator.
- Zero-hit example: 0%.
- Invalid shape, dtype, index range, duplicate and seen-item cases are rejected.
- Baseline tie order and seen-item exclusion pass a small example.
- Random unseen-item recommendations (seed 42): 119 / 1906 = 6.24%. The exact expected rate is 6.26%; random does not mean zero.
- Test-aware oracle used only for checking the metric: 1906 / 1906 = 100%. It is not a model result and is not saved as a recommendation file.
- An independent sparse-matrix calculation reproduced all four hit counts.

## Recommendation review
Among evaluated customers, the most frequently recommended product reaches 2.52% for SVD-log, 2.83% for PCA-log, 11.54% for SVD-count and 2.57% for SVD-binary. None of the model results recommends one item to almost everyone. Count is more concentrated than the other model outputs, but this alone does not establish the cause of its lower hit rate.

The best-seller baseline reaches 94.49% with its most common product; this is expected from its global popularity ranking. Full top-product counts and coverage are in the accompanying CSV files. A high hit rate or concentrated recommendations alone cannot prove or rule out leakage.

## Notes for the model report
The four returned arrays are usable for ML4, ML5 and ML7 evaluation. Add the results above to the comparison and sensitivity tables.

The notebook's recommendation timer excludes product-similarity matrix creation. Label that measurement accurately; a full nightly-job time should include the steps that actually run nightly. Peak memory measured with tracemalloc should keep its stated limitation about native allocations. Matrix sparsity and factor compression are different comparisons; use consistent MB or MiB units.

The first-component table lists high absolute loadings and their signs. The written explanation should describe what those loadings mean rather than relying only on product overlap between PCA and SVD.

## Reproduce
From the project folder, run:
```text
python evaluate.py
```
PyCharm can run evaluate.py directly with the project .venv. The defaults read B_to_A/B_to_A/ and data/processed/handoff/private/test.csv, then write reports/evaluation/. Use --recs-dir if the recommendation files move.

Only aggregate results are included here. Original recommendation files and the model notebook are unchanged. Detailed customer test purchases and real ID maps are not included.
