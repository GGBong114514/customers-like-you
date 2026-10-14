import sys
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from build_matrix import reconcile_returns

columns = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country']
def table(events):
    rows = []
    for invoice, quantity, day, price in events:
        rows.append([invoice, 'A', 'Item', quantity, pd.Timestamp('2011-01-01') + pd.Timedelta(days=day), price, 1, 'UK'])
    return pd.DataFrame(rows, columns=columns)

cutoff = pd.Timestamp('2011-01-10')
cases = [
    ([('1', 10, 0, 1), ('C1', -10, 1, 1)], [0, -10]),
    ([('1', 10, 0, 1), ('C1', -4, 1, 1)], [6, -4]),
    ([('1', 10, 0, 1), ('2', 10, 1, 1), ('C1', -10, 2, 1)], [10, 0, -10]),
    ([('1', 4, 0, 1), ('2', 6, 1, 1), ('C1', -9, 2, 1)], [1, 0, -9]),
    ([('C1', -10, 0, 1), ('1', 10, 1, 1)], [-10, 10]),
    ([('1', 10, 0, 1), ('C1', -10, 1, 2)], [10, -10]),
    ([('1', 10, 0, 1), ('C1', -10, 10, 1)], [10, -10]),
    ([('1', 10, 0, 1), ('2', 4, 10, 1), ('C1', -10, 11, 1)], [10, 4, -10]),
    ([('1', 10, 10, 1), ('C1', -10, 11, 1)], [0, -10]),
]
for events, expected in cases:
    result, audit = reconcile_returns(table(events), cutoff)
    assert result.Quantity.tolist() == expected, (events, result.Quantity.tolist())
    assert (audit.matched_units + audit.unmatched_units == audit.return_units).all()
    assert (audit.unmatched_units >= 0).all()

# 截止点固定时，测试期退款不能改变训练快照。
# Test-period changes must not alter the training snapshot at a fixed cutoff.
base = table([('1', 10, 0, 1), ('2', 4, 1, 1)])
extended = table([('1', 10, 0, 1), ('2', 4, 1, 1), ('C1', -14, 10, 1)])
a, _ = reconcile_returns(base, cutoff)
b, _ = reconcile_returns(extended, cutoff)
pd.testing.assert_frame_equal(a, b.iloc[:2])
print('PASS: 9 return cases, unit conservation, fixed-cutoff train independence.')
