# ADR 0001: Rigorous replication protocol

## Status

Accepted

## Context

The course brief asks for a 20-day image replication with a development/test time split, while the paper defines the training protocol and portfolio evaluation. A single short run is not enough evidence for architecture claims, and test-set statistics must not influence training or model selection.

## Decision

- The primary replication target is the supplied 20-day image with the 20-day return-direction label. Other horizons using the same image are extensions.
- Pixel mean and population standard deviation are computed from the training split only. The same statistics are applied to validation and test images and saved with every experiment.
- The primary portfolio metric is an equal-weight top/bottom 10% date-level long-short portfolio. A top/bottom 20% portfolio is retained as a sensitivity analysis. Neither includes transaction costs unless explicitly added in a later experiment. Monthly cross-section returns are annualized with 12 periods per year. Transaction costs, realized turnover, Newey-West statistics, and a block-bootstrap interval are reported from the saved test predictions.
- Local ablations use the same split, four complete epochs, patience greater than the budget, and seeds 7, 17, and 27. Raw per-seed results and aggregate mean/std are both retained.
- Grad-CAM uses a fixed 16-image test protocol: four true positives, four true negatives, four false positives, and four false negatives, with metadata for every image.
- Remote GPU runs begin only after local tests, normalized baseline/ablation outputs, decile metrics, and Grad-CAM artifacts pass review.

## Consequences

The results are more expensive than a single smoke run but are comparable, auditable, and less likely to turn a seed- or budget-specific observation into a scientific claim. The course-mandated 20-day scope remains clear while extensions can be reported separately.
