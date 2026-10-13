# A to B training-data handoff

## Success targets and roles
Project goals: reduce stored factor size compared with the dense matrix; measure fit and recommendation time; compare hit rate at 10 with unseen-item best sellers; describe customer groups using products and countries. Numerical time/storage targets still need agreement with B. These are goals, not achieved model results.

A owns data cleaning, split, matrices and private evaluation under our agreed division. B owns PCA/SVD, recommendation outputs and the customer map. Memo and slides are outside this package.

## Ready-to-use files
- A_train_log.npz: CSR, log(1 + summed remaining training Quantity).
- A_train_count.npz: CSR, summed remaining training Quantity.
- A_train_binary.npz: CSR, 1 for each positive training pair.
- Shape: (5463, 3684); nonzeros: 411374. All three use identical positions.
- products.csv: product_idx, StockCode, Description, n_customers.
- customers.csv: customer_idx, Country. No real Customer ID.
- profile.csv, cleaning_log.csv, return_summary.csv: public aggregate reports.
- matrix_report.json and validation.json: sizes, time range and matrix checksums.
- decision_log.md: rules, reasons and limitations.

This replaces the older (5473, 3688) build. Replace ALL three matrices and BOTH metadata tables together. Rerun recommendations; old product/customer indexes must not be reused.

## Load the data
```python
import pandas as pd
import scipy.sparse as sp

A = sp.load_npz("A_train_log.npz")
products = pd.read_csv("products.csv", dtype={"StockCode": "string"})
customers = pd.read_csv("customers.csv")
assert A.shape == (len(customers), len(products))
```

Row i is customer_idx i; column j is product_idx j. Indexes start at zero and have no gaps. Read matrix dimensions from A.shape. At least 10 distinct training buyers are required per product. Descriptions and countries use training modes, with alphabetical tie-breaking.

## Data and split
Both Online Retail II sheets contain 1,067,371 raw rows. Cutoff: 2011-10-09 12:50:00. Train timestamps are strictly before cutoff; test timestamps are at or after it. This build has 654,635 training lines after cleaning.

Returns reduce matching earlier purchases before cancellation lines are removed. Training is frozen before processing test-period returns. The matching rule is an assumption because the data does not link every return to an original invoice; unmatched and ambiguous cases are recorded. See decision_log.md.

PADS remains despite its suspicious low price; this is a documented uncertainty. SP1002 passes code cleaning but is absent from the final matrix because it has fewer than 10 training buyers.

Raw Excel, transaction tables, the real customer map, return details and held-out pairs are not in this ZIP. A retains them locally. Aggregate held-out counts in matrix_report.json are not model input or a basis for tuning.

## Return files from B
recs_svd.npy (main log method), recs_pca.npy, recs_count.npy, recs_binary.npy.

Each is an integer array of shape (A.shape[0], 10). Row i contains 10 different product_idx values for customer i, ranked by decreasing score. Exclude products already bought in training. Run:
```text
python validate_recommendations.py recs_svd.npy
```
The validator checks shape, type, range, uniqueness and exclusion of training items. It cannot verify score order without the scores. A evaluates the files on private held-out purchases and the best-seller baseline.

The assignment also asks B for product-to-product recommendations by factor cosine similarity, k comparisons, PCA/SVD timing and memory, and the customer map. These are not replaced by the four return arrays.

## Rebuild from raw
Use Python 3.12 and install requirements.txt in the project virtual environment. Obtain the UCI Excel separately and put it at data/raw/online_retail_II.xlsx beside build_matrix.py. Then run:
```text
python build_matrix.py --value log
```
This writes all three matrices and reports to data/processed/handoff/. --value count and --value binary also generate all three, with the same maps; the option only identifies the preferred matrix. --raw and --output can override paths.

The private/ output contains train.parquet, test_transactions.parquet, customer_id_map.csv, return_audit.csv and test.csv. Do not send that folder to B or add it to Git.

## Source
Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D. CC BY 4.0. This project cleans, filters and aggregates the source data.


# B to A ML Modeling and Recommendation Results

## 1. Method Selection (ML1)

We selected Truncated SVD as our main method because the customer-product matrix is highly sparse (about 97.96%). Truncated SVD can work directly with sparse matrices without centering the data.

We also used PCA as a comparison method.

## 2. Selecting the Number of Components (ML2)

We tested Truncated SVD with k = 10, 20, 50, 100, and 200 and compared their relative reconstruction errors.

The reconstruction error decreased as k increased. We selected **k = 100** as a balance between reconstruction quality and model size.

## 3. Product Similarity and Customer Recommendations (ML3)

We used the log purchase matrix as our main input and trained Truncated SVD with 100 components.

For product similarity, we normalized the product factors and calculated cosine similarity between products.

For customer recommendations, we combined the similarity scores of products each customer had purchased, excluded products already purchased in training, and selected the top 10 remaining products.

The results were saved as `recs_svd.npy`.

## 4. Model Time and Storage (ML4)

We measured the training time, recommendation generation time, and stored factor size for Truncated SVD.

| Measure | Result |
|---|---:|
| Number of components | 100 |
| SVD fit time | 0.2996 seconds |
| Recommendation time for all customers | 1.0880 seconds |
| Stored customer and product factors | 6.98 MiB |

The recommendation time does not include model training or product similarity matrix creation.

Training times were measured in separate runs and may vary slightly.

The Hit Rate@10 evaluation is performed by A using private held-out purchases.

## 5. PCA and Truncated SVD Comparison (ML5)

We compared PCA and Truncated SVD using the same log purchase matrix and k = 100.

| Measure | PCA | Truncated SVD |
|---|---:|---:|
| Fit time | 0.3140 s | 0.2035 s |
| Peak Python memory during fit | 15.89 MiB | 17.00 MiB |
| Stored factor size | 7.01 MiB | 6.98 MiB |
| Relative reconstruction error | 0.7257 | 0.7269 |

The memory measurements were collected using `tracemalloc` and may not include all memory used by NumPy and SciPy.

Both methods had similar reconstruction errors, but Truncated SVD was faster in this comparison. Their first components also shared 9 of the top 10 products, mainly including baking supplies, lunch bags, and home decorations.

We generated PCA recommendations and saved them as `recs_pca.npy`.

## 6. Customer Map and Clustering (ML6)

We normalized each customer's purchase vector and used Truncated SVD to project 5,463 customers into a 2D space.

We compared KMeans clustering with 3, 4, 5, and 6 clusters. Based on the plots, we selected 5 clusters for further analysis.

We examined the top products and countries in each group.

| Cluster | Customers | Segment Name | Main Products |
|---|---:|---|---|
| 0 | 586 | Baking & Home | Cake cases, cake stands, decorations |
| 1 | 1,864 | Mixed Shoppers | Cake stands, shopping bags, party supplies |
| 2 | 1,163 | Home Decoration | Heart decorations, ornaments, chalkboards |
| 3 | 488 | Bags & Lunch | Lunch bags, jumbo bags, cake cases |
| 4 | 1,362 | Baking & Tea | Baking sets, tea sets, cake stands |

About 93.2% of customers were from the United Kingdom. Cluster 3 had a relatively higher proportion of customers from other countries.

## 7. Purchase Value Sensitivity (ML7)

We compared three purchase value representations using the same Truncated SVD recommendation method:

- Log: `log(1 + quantity)`
- Count: original purchase quantity
- Binary: 1 if purchased, otherwise 0

We kept k = 100 and used the same cosine similarity recommendation method for all three matrices.

The log recommendations were generated in ML3. We additionally generated `recs_count.npy` and `recs_binary.npy`.

We will compare their Hit Rate@10 results after A completes the private evaluation.

## 8. Recommendation Files and Validation

B generated four recommendation files:

| File | Method | Purchase Value |
|---|---|---|
| `recs_svd.npy` | Truncated SVD | Log |
| `recs_pca.npy` | PCA | Log |
| `recs_count.npy` | Truncated SVD | Count |
| `recs_binary.npy` | Truncated SVD | Binary |

Each file contains an integer array with shape `(5463, 10)`, representing 10 ranked product recommendations for each customer.

All four files passed `validate_recommendations.py`, including checks for shape, data type, valid product indexes, uniqueness, and exclusion of training purchases.

The validator does not check recommendation score order.

## 9. Pending Evaluation

A will evaluate the four recommendation files using private held-out purchases and compare them with the unseen-item best-seller baseline.

The final Hit Rate@10 results will be added after the evaluation is complete.