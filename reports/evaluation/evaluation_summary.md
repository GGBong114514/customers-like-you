# Recommendation evaluation

## Results

All methods are scored on the same 1,906 customers who have at least one eligible held-out purchase (44,578 customer-product pairs in total). A customer counts as one hit if at least one of their 10 recommendations is a held-out product, even when several match.

| Method | Customers with a hit | Hit Rate@10 | vs baseline (pp) |
|---|---:|---:|---:|
| SVD, log (main) | 448 / 1906 | 23.50% | +6.14 |
| PCA, log | 432 / 1906 | 22.67% | +5.30 |
| SVD, count | 272 / 1906 | 14.27% | -3.10 |
| SVD, binary | 467 / 1906 | 24.50% | +7.14 |
| Best sellers | 331 / 1906 | 17.37% | 0.00 |

The main SVD log model beats the baseline by 6.14 percentage points, and PCA log also beats it. Count is 3.10 points below. Binary is 1.00 point above log (19 more customers). Log was chosen as the main method before these results were computed, so it stays the main result, and count and binary are reported as the purchase-value comparison.

## Baseline and metric

Best sellers are ranked by total training quantity from `A_train_count.npz`, with ties broken by `product_idx`. Each customer gets the first 10 products they did not buy in training. The ranking uses training data only.

Hit Rate@10 = customers with at least one held-out product in their top 10 / customers with at least one eligible held-out product.

## Input checks

All four recommendation files passed: shape (5463, 10), integer type, indices in range, 10 different products per customer, and no product the customer already bought in training. The held-out pairs are unique, have valid indices and do not appear in the training matrix.

## Metric checks

- Hand-worked example: 1 hit out of 2 customers gives 50%; an extra customer with no held-out purchase does not change the denominator.
- No hits gives 0%.
- Wrong shape, wrong type, out-of-range indices, duplicates and already-bought products are rejected.
- Baseline tie order and exclusion of bought products pass a small example.
- Random unseen products (seed 42): 119 / 1906 = 6.24%, close to the expected 6.26%.
- Putting each customer's held-out products into their list on purpose gives 1906 / 1906 = 100%. This only checks the metric; it is not a model.
- A separate sparse-matrix calculation gives the same four hit counts.

## How varied the recommendations are

Among evaluated customers, the single most recommended product goes to 2.52% of customers for SVD log, 2.83% for PCA log, 2.57% for SVD binary and 11.54% for SVD count. The best-seller list gives its top product to 94.49% of customers, as expected from one global ranking.

## Reproduce

From the project root:

```text
python evaluate.py
```

The script reads the `recs_*.npy` files from the project root and the held-out file from `data/processed/handoff/private/test.csv`, and writes the results to `reports/evaluation/`. Only aggregate results are saved; no customer IDs or held-out purchases are included.
