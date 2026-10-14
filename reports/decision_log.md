# Data decisions

## Purchase values
Use remaining positive quantities as of each period end. Sum by customer and product before taking log(1 + quantity). Log is the main matrix because large wholesale orders otherwise dominate. Count and binary use the same positions for ML7. Binary means a positive retained purchase, not every historical order ever placed.

## Returns and cancellations
Remove exact duplicate rows first. Treat identical rows as repeated records; without a line ID, some real repeat lines could be lost.

Process events by InvoiceDate, using source-row order to break timestamp ties. A negative quantity can offset only an earlier positive purchase with the same customer, StockCode and exact unit price. First choose the most recent purchase with an equal remaining quantity. Otherwise reduce remaining quantities from newest to oldest, allowing partial returns. Never reuse units already returned. Remove fully returned purchases, C-prefixed lines and other nonpositive quantities afterward.

The data has no reliable original-invoice link for each return. This allocation is a reproducible assumption, not proof of the original purchase. Repeated equal orders can be ambiguous. Different-price returns, missing keys, and purchases before the dataset began may not match. Unmatched returns do not reduce unrelated purchases. The private return audit records eligible return rows, candidate counts and unmatched units. Returns without a usable customer, code or positive price are excluded from matching and removed by the later cleaning rules. The public return summary covers matching-eligible rows, including non-product codes before their later removal.

Freeze training quantities at the cutoff. Continue the purchase ledger into test so returns can refer to older purchases, but never edit the saved training snapshot with a test-period return. Test purchases are reduced by returns observed by the dataset end. Later returns are unknown. A fixed-cutoff regression test checks that future returns cannot change training quantities.

## Other cleaning rules
Remove nonpositive prices, missing customer IDs, and missing invoice or product codes. Check IDs are finite whole numbers. Keep missing descriptions because a product code can still identify the item. Preserve missing text as missing values, not the text "nan".

Exclude POST, M, C2, BANK CHARGES, ADJUST, DOT, TEST001, D, ADJUST2, TEST002 and AMAZONFEE as postage, charges, adjustments, discounts or test entries. Cleaning-log removals are sequential: a row removed earlier is not counted again under its code. Other fee codes may still be missing from this list.

PADS stays because its description names paper pads. Its 0.001 training price is suspicious; price alone does not prove it is a fee. This remains a limitation, and an exclusion sensitivity check is appropriate if it affects recommendations. SP1002 passes this code rule but fails the final threshold of 10 training buyers. Passing cleaning does not imply inclusion in the matrix.

## Time split and held-out purchases
The last two calendar months are test: cutoff = latest raw timestamp minus two months. Train is strictly before cutoff; test starts at cutoff. Fix the cutoff before return matching. No model or fitted transformation uses test purchases.

Use training purchases only to choose products with at least 10 distinct buyers. This reduces very weak item histories. Drop customers with no retained products. Sort IDs and codes for stable zero-based maps within a build. A changed dataset or cleaning rule can change the maps.

Test pairs must have a known training customer and retained product. Keep one pair per customer/product and require a zero in the training count matrix. Thus "new" means absent from retained, net-positive training purchases. Cold-start customers and filtered products are outside this evaluation. Evaluate customers with at least one eligible held-out purchase; report that denominator.

Descriptions and countries use the most frequent nonmissing training value; alphabetical order breaks ties. Missing training metadata stays missing; test values are not used to fill it.

## Reproducibility and privacy
The builder has no random step. Planned seed for model and clustering steps: 42. Real customer maps, transactions and test purchases stay private. Public reports contain only aggregate results. Raw data and notebook outputs with real customer IDs stay local.

The best-seller baseline and hit rate will be computed when recommendation outputs arrive. All methods will use the same eligible test customers. Evaluation results are pending.

## Evaluation
Use customer hit rate at 10: at least one recommended product in the customer's held-out set. Evaluate only the 1,906 customers with eligible held-out purchases, using the same denominator for every method. The other training customers have no eligible test target.

The baseline ranks total retained training quantity, with product_idx as the tie-breaker. Exclude each customer's seen products before taking 10. This quantity-based choice was fixed before calculating model hit rates. Random checks use seed 42; the test-aware oracle is used only to check the metric. Report the planned log method and count/binary sensitivity results without tuning on the final test set.
