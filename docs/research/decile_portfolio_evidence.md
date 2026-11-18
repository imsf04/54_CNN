# Decile portfolio protocol evidence

## Question

Does the current portfolio evaluator match the paper's 20-day portfolio construction closely enough for the replication line?

## Verified observations

- The paper says that for each forecast period it sorts stocks into decile portfolios by out-of-sample probability of a positive subsequent return.
- The paper defines the H-L spread as long decile 10 and short decile 1.
- The holding period matches the forecast horizon; for the 20-day model this is the subsequent 20-day return.
- The paper reports equal-weight decile portfolio performance and annualized Sharpe ratios.
- The course data uses monthly 20-day image dates, producing 240 date-level observations in the current full test artifact.

Source: `docs/references/jiang_kelly_xiu_2023.txt`, training/evaluation discussion around lines 560-585 and Table I description immediately following it.

## Decision

The primary implementation uses date-level equal-weight top/bottom 10% portfolios, equivalent to decile 10 versus decile 1 when each date has enough assets. The evaluator also emits top/bottom 20% as a sensitivity result. It fails loudly when a date has fewer than ten assets instead of silently reporting a malformed decile portfolio.

## Limits

The current result is gross of transaction costs and does not estimate turnover or statistical significance. It is therefore a descriptive replication metric, not a claim of implementable trading performance. A later extension may add turnover, costs, and Newey-West or bootstrap inference.

## Recommendation

PROCEED with the current evaluator for the local and formal 20-day replication, while keeping the limitations in the report. Do not add costs or inferential machinery to the GPU run unless the experiment plan explicitly reserves those outputs.
