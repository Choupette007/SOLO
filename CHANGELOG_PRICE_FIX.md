# Price Change and Momentum Fix

This branch contains fixes for two critical issues in eligibility.py:

## 1. Fail-Closed Momentum Checks
The `_recent_momentum_allowed` function now properly fails-closed when:
- A minimum threshold is configured (> 0)
- The corresponding price_change field is missing from the token data

Config parameters:
- `discovery.min_price_change_1h` (default: 0.0)
- `discovery.min_price_change_6h` (default: 0.0)

## 2. Signed-Aware Price Change Handling
The `_price_change_allowed` function now correctly handles:
- Negative price changes (drops) separately from positive changes
- Uses signed values instead of abs() for positive changes
- New config parameter `discovery.max_price_drop` to control acceptable drops

Config parameters:
- `discovery.max_price_drop` (default: max_price_change)
- `discovery.max_price_change` (default: 350)
- `discovery.max_price_change_hard` (default: max(max_pct*4, 2000))
- `discovery.min_liq_for_big_pct` (default: 5000.0)
- `discovery.min_vol_for_big_pct` (default: 10000.0)
- `discovery.min_price_for_pct` (default: 0.0001)

## Bug Fixed
Previous implementation used `abs(price_change_pct)`, treating -50% drops the same as +50% pumps, causing incorrect rejections of legitimate price drops.
