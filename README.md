# Customers Like You: PCA or SVD for a Real Retail Problem

A customer-product recommendation and segmentation project using the [UCI Online Retail II dataset](https://doi.org/10.24432/C5CG6D). We use sparse purchase histories to recommend products, compare Truncated SVD with PCA, evaluate against a popularity baseline, and explore customer groups.

## Project goals and responsibilities

- **Data Engineer (A):** Clean transactions, handle returns, create a time-based train/test split, build sparse matrices, and evaluate recommendations using private held-out purchases.
- **ML Engineer (B):** Select and compare dimensionality-reduction methods, generate recommendations, measure efficiency, test purchase-value representations, and visualize customer segments.
- **Both:** Prepare the project memo, presentation, documentation, and final submission.

We evaluate recommendation quality with Hit Rate@10 and report model timing and stored factor size. The timing and storage measurements below describe this experiment, not a production deployment or a guaranteed performance target.

## Dataset and preprocessing

The source contains **1,067,371 raw transaction rows** across two sheets. After cleaning, the training period contains **654,635 rows**. We use a time-based cutoff of **2011-10-09 12:50:00**:

- **Training:** Transactions strictly before the cutoff.
- **Held-out test:** Transactions at or after the cutoff.

Returns are matched against earlier purchases where possible and reduce retained purchase quantities before cancellation lines are removed. Because the source does not always identify the original invoice for a return, some matches are uncertain; unmatched and ambiguous cases are documented in `decision_log.md`. Training data is frozen before test-period returns are processed.

Additional documented decisions:

- `PADS` is retained despite its suspiciously low price; this remains a known data-quality uncertainty.
- `SP1002` passes stock-code cleaning but is excluded from the final matrix because it has fewer than 10 training buyers.
- Products must have at least **10 distinct training buyers** to be included.
- Product descriptions and customer countries use training-data modes, with alphabetical tie-breaking.

### Customer-product matrices

| Property | Value |
|---|---:|
| Customers | 5,463 |
| Products | 3,684 |
| Matrix shape | `(5463, 3684)` |
| Nonzero customer-product pairs | 411,374 |
| Sparsity | 97.96% |

All three matrices use CSR sparse format and share the same nonzero positions:

| File | Purchase value |
|---|---|
| `A_train_log.npz` | `log(1 + summed retained quantity)` |
| `A_train_count.npz` | Summed retained quantity |
| `A_train_binary.npz` | 1 for a positive training purchase, otherwise 0 |

`products.csv` maps `product_idx` to `StockCode`, `Description`, and `n_customers`; `customers.csv` maps `customer_idx` to `Country` without exposing original customer IDs. Indices start at zero, are consecutive, and correspond to matrix columns and rows respectively.

**Version warning:** These matrices replace an older `(5473, 3688)` build. Never mix old recommendation files, index mappings, or matrices with this version.

### Data handoff and reports

The Data Engineer's aggregate handoff includes:

- `profile.csv`, `cleaning_log.csv`, `return_summary.csv`
- `matrix_report.json`, `validation.json`
- `decision_log.md`
- `products.csv`, `customers.csv`, and the three training `.npz` matrices

Raw Excel files, transaction-level tables, original customer ID maps, detailed return audits, and held-out customer-product pairs are **not** included in the public repository. Aggregate held-out counts are not model inputs and were not used to tune the models.

## ML methods and results

### ML1: Method selection

We chose **Truncated SVD** as the main method because the customer-product matrix is approximately 98% sparse. Truncated SVD works directly with sparse input without centering. PCA centers features, which can complicate sparse-data processing. PCA is retained as a comparison method.

### ML2: Number of components

We tested Truncated SVD with `k = 10, 20, 50, 100, 200` and measured relative reconstruction error.

| Components (k) | Relative reconstruction error |
|---:|---:|
| 10 | 0.8757 |
| 20 | 0.8454 |
| 50 | 0.7890 |
| 100 | 0.7272 |
| 200 | 0.6391 |

We selected **k = 100** as a practical balance between reconstruction error and the size of the retained representation. Increasing k further reduces error, but requires more components. The reconstruction-error plot is included in the ML notebook.

### ML3: Product similarity and recommendations

Using the log-value training matrix and Truncated SVD (`k = 100`), we:

1. Extract product factor vectors from the SVD components.
2. L2-normalize product vectors and calculate product-to-product cosine similarity.
3. Score candidate products for each customer using similarities to products they purchased in training.
4. Exclude all products already purchased by that customer in training.
5. Return the 10 highest-scoring remaining products.

The main output is `recs_svd.npy`, an integer array of shape `(5463, 10)`. The notebook also demonstrates top-10 similar products for an example product. Product similarity reflects **co-purchase patterns**, not necessarily functional or semantic similarity.

### ML4: Efficiency and recommendation quality

| Measure | Observed result |
|---|---:|
| SVD components | 100 |
| Stored customer and product factors | 6.98 MiB |
| SVD fit time | Approximately 0.3 seconds in the reported timing run |
| Recommendation scoring, masking and top-10 selection | Approximately 1 second for 5,463 customers |
| SVD Log Hit Rate@10 | **23.50%** |
| Best-seller baseline Hit Rate@10 | **17.37%** |

Timing measurements vary between runs. The recommendation timing **excludes SVD fitting and construction of the product-similarity matrix**; it must not be interpreted as end-to-end runtime. The 6.98 MiB factor size excludes the original matrix, similarity matrix, and intermediate arrays.

The main SVD Log model exceeded the best-seller baseline by **6.14 percentage points** (448 versus 331 customers with a hit, out of 1,906 eligible customers).

### ML5: PCA versus Truncated SVD

Both methods use the same log training matrix and `k = 100`.

| Measure | PCA | Truncated SVD |
|---|---:|---:|
| Relative reconstruction error | 0.7257 | 0.7269 |
| Stored factor size | 7.01 MiB | 6.98 MiB |
| Peak Python memory during fit | 15.89 MiB | 17.00 MiB |
| Hit Rate@10 | **22.67%** | **23.50%** |

Fit times were only measured in individual runs and varied across notebook executions, so we do not claim a consistent speed advantage for either method. `tracemalloc` measures traced Python allocations and may omit native memory allocations made by NumPy, SciPy, or other libraries.

The first PCA and SVD components share 9 of their 10 largest absolute product loadings, including baking supplies, lunch bags, and home decorations. Large absolute loadings indicate products that contribute strongly to a component; the displayed top loadings are positive in this run. PCA describes variation relative to average purchases, while SVD works with the original matrix. Both outperform the baseline in this held-out evaluation, and SVD has a slightly higher observed hit rate.

### ML6: Customer map and segmentation

We L2-normalized customer purchase vectors, projected them into two dimensions with Truncated SVD, and compared KMeans with 3, 4, 5, and 6 groups. We chose **5 groups** based on the visual separation and interpretability of their purchase patterns.

| Cluster | Customers | Segment name | Common products |
|---:|---:|---|---|
| 0 | 586 | Baking & Home | Cake cases, cake stands, decorations |
| 1 | 1,864 | Mixed Shoppers | Cake stands, shopping bags, party supplies |
| 2 | 1,163 | Home Decoration | Heart decorations, ornaments, chalkboards |
| 3 | 488 | Bags & Lunch | Lunch bags, jumbo bags, cake cases |
| 4 | 1,362 | Baking & Tea | Baking sets, tea sets, cake stands |

Approximately **93.2%** of customers are from the United Kingdom, limiting the usefulness of country for separating most groups. Cluster 3 has a relatively higher share of customers from other countries. The notebook includes the customer map and top products and countries for each group; it does not display real customer IDs.

### ML7: Purchase-value sensitivity

We used the same Truncated SVD recommendation method (`k = 100`) with three input representations:

| Purchase representation | Hit Rate@10 | Difference from baseline |
|---|---:|---:|
| Log | 23.50% | +6.14 percentage points |
| Count | 14.27% | −3.10 percentage points |
| Binary | **24.50%** | +7.14 percentage points |

Binary achieved the highest **observed** hit rate, 1.00 percentage point above Log (19 additional customers with a hit). This is a descriptive difference, not a statistical significance claim. Count was below the best-seller baseline. We retain the preselected **SVD Log** method as our main result and report Binary and Count as sensitivity comparisons; held-out test results were not used for further tuning.

## Held-out evaluation

The Data Engineer evaluated the four recommendation arrays against the **same 1,906 customers** who had at least one eligible held-out purchase. The test set contains **44,578 distinct customer-product pairs**. Customers without eligible held-out purchases are excluded from the denominator, and each eligible customer contributes at most one hit.

**Hit Rate@10** is the fraction of eligible customers with at least one held-out product among their top 10 recommendations. It is **not** precision@10 and is **not** the fraction of all test purchases recovered.

| Method | Customers with a hit | Hit Rate@10 | Difference from baseline |
|---|---:|---:|---:|
| SVD Log | 448 / 1906 | 23.50% | +6.14 pp |
| PCA Log | 432 / 1906 | 22.67% | +5.30 pp |
| SVD Count | 272 / 1906 | 14.27% | −3.10 pp |
| SVD Binary | 467 / 1906 | **24.50%** | +7.14 pp |
| Best-seller Count | 331 / 1906 | 17.37% | 0.00 pp |

### Best-seller baseline

The baseline ranks products by **total retained training quantity** from `A_train_count.npz`, breaks ties by `product_idx`, excludes each customer's training purchases, and takes the first 10 remaining products. Popularity is computed **only from training data**. Ranking by the number of buyers would be a different baseline.

### Evaluation checks and limitations

All four recommendation arrays passed validation for shape `(5463, 10)`, integer dtype, valid product indices, 10 distinct products per customer, and exclusion of training purchases. Held-out pairs were checked for valid indices, uniqueness, and no overlap with retained training purchases. Independent sparse-matrix calculations reproduced the four hit counts.

Additional metric sanity checks included a hand-worked example, zero-hit case, invalid-input rejection, a random unseen-item baseline (**6.24%**, seed 42), and a test-aware oracle (**100%**, used solely to test the metric and **not** as a model).

Recommendation concentration differed across methods. Among evaluated customers, the most commonly recommended item appeared for **2.52%** of SVD Log customers, **2.83%** of PCA Log customers, **11.54%** of SVD Count customers, and **2.57%** of SVD Binary customers. The corresponding figure for the global best-seller baseline was **94.49%**. Concentration alone does not explain differences in hit rate or establish the presence or absence of leakage.

The evaluation inspected the supplied notebook and saved recommendation arrays but **did not rerun the notebook**. A training-input checksum was not supplied with B's outputs, so exact training-file provenance was not independently established. The validator also cannot prove that saved recommendations are in descending score order without access to scores. See the evaluation reports for further details.

## Recommendation outputs

| File | Model | Input |
|---|---|---|
| `recs_svd.npy` | Truncated SVD | Log |
| `recs_pca.npy` | PCA | Log |
| `recs_count.npy` | Truncated SVD | Count |
| `recs_binary.npy` | Truncated SVD | Binary |

Each file has shape `(5463, 10)`, where row `i` contains ten recommended zero-based `product_idx` values for customer `i`. The notebook generates recommendations in descending score order; the validator independently checks shape, dtype, index range, uniqueness, and exclusion of seen training items, but **not score ordering**.

## How to reproduce

Use **Python 3.12** and install the packages in `requirements.txt` into a virtual environment. Run commands from the project root, with paths adjusted if the repository uses different directories.

### 1. Rebuild the data handoff

Obtain the UCI Online Retail II Excel file separately and place it at `data/raw/online_retail_II.xlsx` beside the expected project structure. Then run:

```bash
python build_matrix.py --value log
```

The script writes all three matrices and aggregate reports to `data/processed/handoff/`. The `--value` argument identifies the preferred matrix but does not change which three matrices are generated. `--raw` and `--output` can override the paths.

The private output directory contains `train.parquet`, `test_transactions.parquet`, `customer_id_map.csv`, `return_audit.csv`, and `test.csv`. **Do not commit or share this directory.**

### 2. Run the ML notebook

Open `ml_experiments.ipynb` and ensure the following files are accessible from its working directory:

```text
A_train_log.npz
A_train_count.npz
A_train_binary.npz
products.csv
customers.csv
validate_recommendations.py
```

Run the notebook from the beginning to reproduce the ML figures, comparison tables, customer groups, and four recommendation arrays. The notebook uses fixed random seeds where applicable. Its saved outputs reflect particular runs; short runtime measurements may vary.

### 3. Validate recommendation arrays

```bash
python validate_recommendations.py recs_svd.npy
python validate_recommendations.py recs_pca.npy
python validate_recommendations.py recs_count.npy
python validate_recommendations.py recs_binary.npy
```

### 4. Evaluate against held-out purchases

Evaluation must be run by the Data Engineer, who retains the private held-out data:

```bash
python evaluate.py
```

By default, the evaluation script reads recommendation arrays from `B_to_A/B_to_A/` and the private test file from `data/processed/handoff/private/test.csv`, and writes aggregate reports to `reports/evaluation/`. Use `--recs-dir` if recommendation arrays are elsewhere. The resulting public reports include `evaluation_results.csv`, `evaluation_summary.md`, `evaluation_metadata.json`, `metric_checks.json`, `recommendation_diagnostics.csv`, and `top_recommended_products.csv`.

**Path note:** The data-build instructions and notebook currently use different default working directories. Keep the input files where the notebook expects them or update the notebook paths before running from a clean checkout. Verify that the named scripts and reports are committed in the locations described here.

## Limitations

- The dataset is sparse, and purchase-based product similarity does not necessarily reflect product function or meaning.
- Some return matching is uncertain because the raw data lacks complete invoice links; known decisions are recorded in `decision_log.md`.
- Hit Rate@10 is a single held-out metric; it does not measure purchase value, ranking quality beyond a hit, diversity, or statistical significance.
- The observed Binary advantage is not a reason to retune on the final held-out set.
- Short fit-time measurements vary between runs. Stored factor size is not full model or pipeline memory; `tracemalloc` may miss native allocations.
- A 2D projection and KMeans clusters are exploratory summaries, not definitive customer categories.
- The public repository excludes original customer IDs and detailed held-out transactions.

## Dataset citation

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D. Licensed under CC BY 4.0. This project cleans, filters, and aggregates the source data.
