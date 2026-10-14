"""生成训练矩阵和本地测试文件。
Build the training matrices and local test files.

在 PyCharm 中运行，或使用下面的命令。
Run in PyCharm, or use: python build_matrix.py --value log

不修改原始 Excel。
Keep the raw Excel unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_data(raw_path):
    """读取两张表，先检查列，再合并。
    Read both sheets and check the columns before joining them.
    """
    sheets = pd.read_excel(raw_path, sheet_name=None)
    tables = list(sheets.values())
    expected_columns = list(tables[0].columns)

    for table in tables:
        current_columns = list(table.columns)
        if current_columns != expected_columns:
            raise ValueError("Sheet columns differ. Align them first.")

    df = pd.concat(tables, ignore_index=True)
    required_columns = {
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country",
    }
    if set(df.columns) != required_columns:
        raise ValueError("Unexpected Excel columns.")

    # 编号和描述用文本，缺失值保持为空。
    # Keep IDs and descriptions as text without losing missing values.
    text_columns = ["Invoice", "StockCode", "Description", "Country"]
    for column in text_columns:
        text = df[column].astype("string")
        df[column] = text.str.strip()

    df["StockCode"] = df["StockCode"].str.upper()
    return df


def make_profile(df):
    """检查类型、缺失值和不同值的数量。
    Check types, missing values and unique values.
    """
    types = df.dtypes.astype(str)
    missing = df.isna().sum()
    missing_percent = df.isna().mean() * 100
    missing_percent = missing_percent.round(2)
    unique_counts = df.nunique()

    profile = pd.DataFrame({
        "dtype": types,
        "missing": missing,
        "missing_%": missing_percent,
        "n_unique": unique_counts,
    })
    return profile


def apply_rule(data, name, keep, log):
    """保留符合条件的行，记录排除的行数。
    Keep matching rows and record how many rows were removed.
    """
    before = len(data)
    result = data.loc[keep].copy()
    removed = before - len(result)
    record = {
        "rule": name,
        "removed": removed,
        "remaining": len(result),
    }
    log.append(record)
    return result


def reconcile_returns(data, cutoff):
    """按时间配对退款，在截止时固定训练数量。
    Match returns in time order and freeze training quantities at the cutoff.
    """
    ordered = data.sort_values("InvoiceDate", kind="stable")
    remaining = data["Quantity"].copy()
    purchases = {}
    audit = []
    snapshots = []

    # 测试期继续使用购买记录，但不改已经保存的训练快照。
    # Keep purchase history in test, but do not change the saved train snapshot.
    for period in ["train", "test"]:
        before_cutoff = ordered["InvoiceDate"] < cutoff
        if period == "train":
            part = ordered.loc[before_cutoff]
        else:
            part = ordered.loc[~before_cutoff]

        columns = ["Invoice", "StockCode", "Description", "Quantity",
                   "InvoiceDate", "Price", "Customer ID", "Country"]
        for row in part[columns].itertuples(index=True, name=None):
            index = row[0]
            invoice, stock_code, description, quantity, date, price, customer, country = row[1:]
            valid_key = pd.notna(customer) and pd.notna(stock_code) and price > 0
            if not valid_key or not np.isfinite(quantity):
                continue
            key = (customer, stock_code, price)
            cancelled = str(invoice).upper().startswith("C")

            if quantity > 0 and not cancelled:
                purchases.setdefault(key, []).append(index)
                continue
            if quantity >= 0:
                continue

            needed = -quantity
            candidates = purchases.get(key, [])
            active = [i for i in candidates if remaining.at[i] > 0]
            exact = [i for i in active if remaining.at[i] == needed]
            # 优先最近一笔数量相同的购买，其余按从近到远扣除。
            # Prefer the latest exact quantity; otherwise work from latest to oldest.
            if exact:
                chosen = [exact[-1]]
            else:
                chosen = list(reversed(active))

            matched = 0
            prior_train_units = 0
            for purchase_index in chosen:
                units = min(needed, remaining.at[purchase_index])
                remaining.at[purchase_index] -= units
                needed -= units
                matched += units
                if period == "test" and data.at[purchase_index, "InvoiceDate"] < cutoff:
                    prior_train_units += units
                if needed == 0:
                    break

            audit.append({
                "period": period,
                "source_row": index,
                "Invoice": invoice,
                "Customer ID": customer,
                "StockCode": stock_code,
                "return_units": -quantity,
                "matched_units": matched,
                "unmatched_units": needed,
                "test_return_against_train_units": prior_train_units,
                "candidate_purchases": len(active),
                "exact_candidates": len(exact),
            })

        snapshot = part.copy()
        snapshot["Quantity"] = remaining.loc[part.index]
        snapshots.append(snapshot)

    adjusted = pd.concat(snapshots).sort_index()
    columns = ["period", "source_row", "Invoice", "Customer ID", "StockCode",
               "return_units", "matched_units", "unmatched_units",
               "test_return_against_train_units", "candidate_purchases", "exact_candidates"]
    return adjusted, pd.DataFrame(audit, columns=columns)


def clean_data(df, cutoff):
    """逐条执行清洗规则。
    Apply the cleaning rules one at a time.
    """
    clean_df = df.copy()
    log = [{"rule": "raw", "removed": 0, "remaining": len(clean_df)}]

    keep = ~clean_df.duplicated()
    clean_df = apply_rule(clean_df, "exact duplicates", keep, log)

    original_quantity = clean_df["Quantity"].copy()
    clean_df, return_audit = reconcile_returns(clean_df, cutoff)
    fully_returned = original_quantity.gt(0) & clean_df["Quantity"].eq(0)
    clean_df = apply_rule(clean_df, "fully returned purchases", ~fully_returned, log)

    invoice = clean_df["Invoice"].str.upper()
    cancelled = invoice.str.startswith("C", na=False)
    keep = ~cancelled
    clean_df = apply_rule(clean_df, "cancellation", keep, log)

    keep = clean_df["Quantity"] > 0
    clean_df = apply_rule(clean_df, "Quantity <= 0", keep, log)

    keep = clean_df["Price"] > 0
    clean_df = apply_rule(clean_df, "Price <= 0", keep, log)

    keep = clean_df["Customer ID"].notna()
    clean_df = apply_rule(clean_df, "missing Customer ID", keep, log)

    has_invoice = clean_df["Invoice"].notna() & clean_df["Invoice"].ne("")
    has_code = clean_df["StockCode"].notna() & clean_df["StockCode"].ne("")
    keep = has_invoice & has_code
    clean_df = apply_rule(clean_df, "missing Invoice or StockCode", keep, log)

    # 排除邮费、费用、调整、折扣和测试记录。
    # Remove postage, fees, adjustments, discounts and test entries.
    # 保留 PADS 和 SP1002，不能只看编码格式。
    # Keep PADS and SP1002; code format alone is not enough.
    non_product_codes = {
        "POST", "M", "C2", "BANK CHARGES", "ADJUST",
        "DOT", "TEST001", "D", "ADJUST2", "TEST002", "AMAZONFEE",
    }
    is_non_product = clean_df["StockCode"].isin(non_product_codes)
    keep = ~is_non_product
    clean_df = apply_rule(clean_df, "non-product codes", keep, log)

    ids = pd.to_numeric(clean_df["Customer ID"], errors="raise")
    whole_numbers = ids % 1 == 0
    finite_numbers = np.isfinite(ids)
    if not (whole_numbers & finite_numbers).all():
        raise ValueError("Customer IDs must be finite whole numbers.")
    clean_df["Customer ID"] = ids.astype("int64")

    dates = pd.to_datetime(clean_df["InvoiceDate"], errors="raise")
    if dates.isna().any():
        raise ValueError("Missing transaction dates.")
    clean_df["InvoiceDate"] = dates

    cleaning_log = pd.DataFrame(log)
    return clean_df, cleaning_log, return_audit


def split_by_time(clean_df, cutoff):
    """早期购买用于训练，最后两个月用于测试。
    Use earlier purchases for training and the last two months for testing.
    """
    is_train = clean_df["InvoiceDate"] < cutoff
    is_test = clean_df["InvoiceDate"] >= cutoff
    train = clean_df.loc[is_train].copy()
    test = clean_df.loc[is_test].copy()

    assert len(train) + len(test) == len(clean_df)
    assert train["InvoiceDate"].max() < cutoff
    assert test["InvoiceDate"].min() >= cutoff
    return train, test, cutoff


def aggregate_purchases(train, min_customers):
    """先加总数量，再保留训练买家足够多的商品。
    Sum quantities first, then keep products with enough training buyers.
    """
    grouped = train.groupby(["Customer ID", "StockCode"], as_index=False)
    agg = grouped["Quantity"].sum()

    product_groups = agg.groupby("StockCode")
    n_customers = product_groups["Customer ID"].nunique()
    popular_products = n_customers.loc[n_customers >= min_customers]
    kept_codes = popular_products.index

    keep = agg["StockCode"].isin(kept_codes)
    agg = agg.loc[keep].copy()
    if agg.empty:
        raise ValueError("No purchases remain after the training product filter.")
    return agg, n_customers


def make_index_maps(agg):
    """给客户分配行号，给商品分配列号。
    Give each customer a row number and each product a column number.
    """
    customer_ids = agg["Customer ID"].unique()
    customer_ids = np.sort(customer_ids)
    stock_codes = agg["StockCode"].unique()
    stock_codes = np.sort(stock_codes)

    cust_idx = {}
    for row_number, customer_id in enumerate(customer_ids):
        cust_idx[int(customer_id)] = row_number

    prod_idx = {}
    for column_number, stock_code in enumerate(stock_codes):
        prod_idx[stock_code] = column_number

    return customer_ids, stock_codes, cust_idx, prod_idx


def make_matrices(agg, customer_ids, stock_codes, cust_idx, prod_idx):
    """三种矩阵使用相同的行列位置。
    Use the same positions for count, log and binary values.
    """
    rows = agg["Customer ID"].map(cust_idx).to_numpy()
    columns = agg["StockCode"].map(prod_idx).to_numpy()
    quantities = agg["Quantity"].to_numpy(dtype=np.float64)
    matrix_shape = (len(customer_ids), len(stock_codes))

    count = sp.csr_matrix((quantities, (rows, columns)), shape=matrix_shape)
    count.sort_indices()
    assert count.nnz == len(agg)
    assert (count.data > 0).all()

    log_matrix = count.copy()
    log_matrix.data = np.log1p(count.data)

    binary = count.copy()
    binary.data[:] = 1
    matrices = {"count": count, "log": log_matrix, "binary": binary}
    return matrices


def most_common_values(train, key_column, value_column):
    """选训练中最常见的值，次数相同按文字顺序选。
    Use the most common training value. Break ties by text order.
    """
    available = train.dropna(subset=[value_column])
    groups = available.groupby([key_column, value_column])
    counts = groups.size()
    counts = counts.rename("frequency")
    frequency_table = counts.reset_index()

    sort_columns = [key_column, "frequency", value_column]
    directions = [True, False, True]
    ordered = frequency_table.sort_values(sort_columns, ascending=directions)
    first_per_key = ordered.drop_duplicates(subset=[key_column])
    indexed = first_per_key.set_index(key_column)
    values = indexed[value_column]
    return values


def make_metadata(train, customer_ids, stock_codes, n_customers):
    """共享表里不放真实客户编号。
    Keep real customer IDs out of the shared tables.
    """
    descriptions = most_common_values(train, "StockCode", "Description")
    countries = most_common_values(train, "Customer ID", "Country")

    products = pd.DataFrame({
        "product_idx": np.arange(len(stock_codes)),
        "StockCode": stock_codes,
    })
    products["Description"] = products["StockCode"].map(descriptions)
    support = products["StockCode"].map(n_customers)
    products["n_customers"] = support.astype("int64")

    country_values = pd.Series(customer_ids).map(countries)
    customers = pd.DataFrame({
        "customer_idx": np.arange(len(customer_ids)),
        "Country": country_values,
    })
    private_map = pd.DataFrame({
        "customer_idx": np.arange(len(customer_ids)),
        "Customer ID": customer_ids,
    })
    return products, customers, private_map


def make_held_out(test, count, cust_idx, prod_idx):
    """保留训练期间没买过的商品的未来购买记录。
    Keep future purchases of products not bought during training.
    """
    known_customer = test["Customer ID"].isin(cust_idx)
    known_product = test["StockCode"].isin(prod_idx)
    keep = known_customer & known_product
    eligible = test.loc[keep, ["Customer ID", "StockCode"]]
    eligible = eligible.drop_duplicates()

    rows = eligible["Customer ID"].map(cust_idx).to_numpy(dtype=np.int64)
    columns = eligible["StockCode"].map(prod_idx).to_numpy(dtype=np.int64)
    training_values = np.asarray(count[rows, columns]).ravel()
    is_unseen = training_values == 0

    held_out = pd.DataFrame({
        "customer_idx": rows[is_unseen],
        "product_idx": columns[is_unseen],
    })
    held_out = held_out.sort_values(["customer_idx", "product_idx"])
    held_out = held_out.reset_index(drop=True)
    assert not held_out.duplicated().any()
    return held_out, len(eligible)


def make_report(df, clean_df, train, test, cutoff, count, held_out,
                eligible_pairs, min_customers):
    """记录规模、时间范围和内存。
    Record sizes, dates and memory use.
    """
    rows, columns = count.shape
    total_cells = rows * columns
    sparse_bytes = count.data.nbytes + count.indices.nbytes + count.indptr.nbytes
    report = {
        "raw_rows": len(df),
        "clean_rows": len(clean_df),
        "train_rows": len(train),
        "test_transaction_rows": len(test),
        "cutoff": str(cutoff),
        "train_start": str(train["InvoiceDate"].min()),
        "train_end": str(train["InvoiceDate"].max()),
        "test_start": str(test["InvoiceDate"].min()),
        "test_end": str(test["InvoiceDate"].max()),
        "shape": [rows, columns],
        "nnz": count.nnz,
        "min_customers": min_customers,
        "sparsity_pct": 100 * (1 - count.nnz / total_cells),
        "dense_mb_float64": total_cells * 8 / 1e6,
        "sparse_mb": sparse_bytes / 1e6,
        "held_out_purchases": len(held_out),
        "held_out_customers": int(held_out["customer_idx"].nunique()),
        "test_unique_known_pairs": eligible_pairs,
    }
    return report


def save_and_check(out, matrices, products, customers, private_map,
                   held_out, profile, cleaning_log, report, return_audit):
    """保存文件，再读回来检查。
    Save the files, then load them again and check the result.
    """
    out.mkdir(parents=True, exist_ok=True)
    private = out / "private"
    private.mkdir(exist_ok=True)
    count = matrices["count"]
    validation = {}

    for value_name, matrix in matrices.items():
        path = out / f"A_train_{value_name}.npz"
        sp.save_npz(path, matrix)
        restored = sp.load_npz(path)
        assert sp.isspmatrix_csr(restored)
        assert restored.shape == count.shape
        assert np.array_equal(restored.indptr, count.indptr)
        assert np.array_equal(restored.indices, count.indices)
        assert np.array_equal(restored.data, matrix.data)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        validation[value_name] = {
            "sha256": digest,
            "shape": list(restored.shape),
            "nnz": restored.nnz,
        }

    assert products["product_idx"].tolist() == list(range(count.shape[1]))
    assert customers["customer_idx"].tolist() == list(range(count.shape[0]))
    assert "Customer ID" not in customers.columns
    actual_support = np.asarray((count > 0).sum(axis=0)).ravel()
    assert np.array_equal(products["n_customers"].to_numpy(), actual_support)
    held_rows = held_out["customer_idx"].to_numpy()
    held_columns = held_out["product_idx"].to_numpy()
    held_training_values = np.asarray(count[held_rows, held_columns]).ravel()
    assert (held_training_values == 0).all()

    products.to_csv(out / "products.csv", index=False)
    customers.to_csv(out / "customers.csv", index=False)
    private_map.to_csv(private / "customer_id_map.csv", index=False)
    held_out.to_csv(private / "test.csv", index=False)
    profile.to_csv(out / "profile.csv")
    cleaning_log.to_csv(out / "cleaning_log.csv", index=False)
    return_audit.to_csv(private / "return_audit.csv", index=False)
    totals = ["return_units", "matched_units", "unmatched_units",
              "test_return_against_train_units"]
    return_summary = return_audit.groupby("period")[totals].sum()
    return_summary["return_rows"] = return_audit.groupby("period").size()
    return_summary["rows_with_multiple_candidates"] = return_audit.groupby("period")["candidate_purchases"].apply(lambda s: (s > 1).sum())
    return_summary.to_csv(out / "return_summary.csv")
    report_text = json.dumps(report, indent=2)
    (out / "matrix_report.json").write_text(report_text, encoding="utf-8")
    validation_text = json.dumps(validation, indent=2)
    (out / "validation.json").write_text(validation_text, encoding="utf-8")


def build(raw_path, out, min_customers=10):
    """按顺序运行各个步骤。
    Run the steps in order.
    """
    print("1/8 Read Excel", flush=True)
    df = load_data(raw_path)
    profile = make_profile(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="raise")
    if df["InvoiceDate"].isna().any():
        raise ValueError("Missing transaction dates.")
    cutoff = df["InvoiceDate"].max() - pd.DateOffset(months=2)

    print("2/8 Clean rows", flush=True)
    clean_df, cleaning_log, return_audit = clean_data(df, cutoff)

    print("3/8 Split by time", flush=True)
    train, test, cutoff = split_by_time(clean_df, cutoff)

    print("4/8 Sum purchases and filter products", flush=True)
    agg, n_customers = aggregate_purchases(train, min_customers)
    customer_ids, stock_codes, cust_idx, prod_idx = make_index_maps(agg)

    print("5/8 Build three matrices", flush=True)
    matrices = make_matrices(agg, customer_ids, stock_codes, cust_idx, prod_idx)
    products, customers, private_map = make_metadata(
        train, customer_ids, stock_codes, n_customers
    )

    print("6/8 Prepare private test purchases", flush=True)
    count = matrices["count"]
    held_out, eligible_pairs = make_held_out(test, count, cust_idx, prod_idx)

    print("7/8 Make the report", flush=True)
    report = make_report(
        df, clean_df, train, test, cutoff, count,
        held_out, eligible_pairs, min_customers
    )

    print("8/8 Save and check", flush=True)
    save_and_check(
        out, matrices, products, customers, private_map,
        held_out, profile, cleaning_log, report, return_audit
    )
    train.to_parquet(out / "private/train.parquet", index=False)
    test.to_parquet(out / "private/test_transactions.parquet", index=False)
    print(json.dumps(report, indent=2))


def main():
    # 这里读取运行选项。
    # Read the run options here.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value", choices=["log", "count", "binary"], default="log")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-customers", type=int, default=10)
    args = parser.parse_args()
    if args.min_customers < 1:
        parser.error("--min-customers must be positive.")

    # 路径从脚本目录开始。
    # Start paths from the script folder.
    root = Path(__file__).resolve().parent
    raw_path = args.raw
    if raw_path is None:
        raw_path = root / "data/raw/online_retail_II.xlsx"
    out = args.output
    if out is None:
        out = root / "data/processed/handoff"

    build(raw_path, out, args.min_customers)
    print(f"Selected matrix: A_train_{args.value}.npz; all three matrices are saved.")


if __name__ == "__main__":
    main()
