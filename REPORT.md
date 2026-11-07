# Project report: how the bot-query classifier was built

This is the history behind the [README](README.md): the decisions, the evidence for each, the two full runs and what they changed. The README describes the system as it stands. Sources are `DESIGN.md` (decisions and probe findings, sections 12 and 13), `PLAN.md` (tasks, reviews and the run reports under M6.T1 and M7.T7) and the git history.

## 1. Setup

**Goal.** Decide from the query text alone whether a furniture shop's search query is real or bot traffic, with the false positive rate (FPR) capped at 1% first and bot recall maximised inside the cap.

**Constraints that shaped everything.**

- The only real legitimate queries are the 480 in the public WANDS dataset (`DESIGN.md` 3), and there are no real bot queries at all. Positives therefore have to be generated, and a model can learn the generator instead of the signal.
- The real queries are too few to train on and also the only honest test set for the FPR, so every design choice is about not letting them leak into fitting.

**Working method.** Four roles with a fixed loop: a design owner records each decision and its rationale; a planner writes tasks with acceptance criteria; a developer implements one task at a time test-first; a reviewer validates each task, including a mutation pass in an isolated copy (every surviving mutant became a correction), and the design owner signs off per milestone after running the suite and reading the diff. The suite grew from 336 tests at the end of M2 to 941 at the end of M7.

## 2. The decisions, in order

| # | Decision | Evidence | Where |
|---|---|---|---|
| 1 | Character TF-IDF (1 to 4 characters, sublinear, `min_df=2`), fitted inside each training fold only | Probes of the vectoriser on empty, unknown and very long input | `DESIGN.md` 5, 13.3 |
| 2 | Group-aware folds: each generator template and each real query is one group, so nothing straddles a fold | With fewer than 6 templates per family a family vanished from some inner folds (7 of 75); with 6 it never does | 4 rule 6, 13.10, 13.14 |
| 3 | Threshold chosen on real queries, never on training data | A model fitted on the threshold set would flatter its own FPR | 8.2, 8.4 |
| 4 | No LLM-generated positives, no second query source | Paid, not reproducible, and templates sufficed | 12.1 items 2, 7 |
| 5 | A blind-labelled 50-query sanity set, never fitted on | The authors of the generators must not see it, so collisions are caught by an audit, not by authors steering around it | 8.6.1 |
| 6 | **Q13: training negatives are word spans cut from product names, not the 480 real queries** | Probe: spans at 20,000 reached 83.4% recall at FPR <= 1% against 71.0% for real queries as negatives (384 of them); whole product names transfer badly (18% FPR on real queries) while spans with the queries' length distribution transfer (0.2%); 5,000 spans capture most of the gain of 40,000 | 3.2, 13.19 |
| 7 | Catalogue-gap check: remove the products of the test queries' classes and re-measure FPR | The new failure mode of decision 6 is a shopper asking for something the catalogue does not describe | 8.6 item 5 |
| 8 | SMOTE dropped, embeddings abandoned, SVD rejected for the classifier | SMOTE generates nothing on a balanced set (13.4); embeddings cost latency; SVD brought no gain (13.21, 13.22) | 12.1 items 15 to 17 |
| 9 | Part-number hard negatives and the `lr_hand` model (M7) | The first full run, section 4 | 3.3, 6.5 |

## 3. The first full run (M6, 2026-11-06)

Three models, both arms, leave-one-family-out (LOFO), catalogue gap, clustering. Wall time 65 minutes; one MLP fit at 5,000 / 5,000 took 38 seconds.

Primary arm (spans as negatives), pooled over 15 folds:

| Model | Recall at FPR <= 1% | Achieved FPR |
|---|---|---|
| `lr` | 77.0% | 0.49% |
| `mlp` | 76.3% | 1.18% |
| `trivial` (length, word count, punctuation ratio) | 61.3% | 0.90% |

Comparison arm (the 480 real queries as negatives): `lr` 70.4%, `mlp` 66.5%, `trivial` 58.6%.

**What it decided.**

- Logistic regression matched the MLP within noise (recall -0.8 points, standard error 0.9) and the MLP failed the FPR gate, so logistic regression became the recommendation (`DESIGN.md` 12.1 item 18). The MLP stayed in the harness as the comparison.
- Spans beat real queries as training negatives (`lr` +6.6 points, gate holds), confirming decision 6.
- The catalogue-gap check passed in every arm and model.
- Two things did not look right: held-out-family recall of `lr` was low on B (34%) and D (30%) while the trivial baseline transferred better there (49% and 69%), and 85% of 100 generated part numbers were flagged although part numbers are legitimate.

**Reasoning about the first finding.** The tempting fix was to rework the B and D generators. The per-template table argued against it: the trivial baseline scored 100% on scripts, URLs, emails and phone numbers because real payloads of that kind are long and punctuation-heavy, so a generator that avoided those cues would be less realistic, not harder. The finding was about the features: character n-grams do not transfer to an unseen family while length and punctuation do. The fix was therefore a model change, not a data change.

**Clustering.** With labels hidden, HDBSCAN on TF-IDF reduced to 50 dimensions did not separate the classes (default cell: 2 clusters, 1.5% noise, adjusted Rand index 0.002, seed-labelling accuracy 0.49). The cells with a high index (0.87 to 0.90) labelled 84% of rows as noise. Finding recorded; no decision depended on it.

## 4. The response (M7, 2026-11-07)

Two changes, both approved by the user:

1. **`lr_hand`** (`DESIGN.md` 6.5): the TF-IDF matrix plus the three standardised hand features of the trivial baseline, into the same logistic regression. A probe first confirmed the combined matrix stays sparse and the scaler is fitted on training rows only (13.23).
2. **100 part-number hard negatives** (3.3): generated in the four shapes of the part-number diagnostic, training-only, never in a threshold set or a test side, with their own tag, drop lists against every other negative list and the diagnostic strings, and rulings on quotas, seeds and the inner-split groups.

## 5. The final run (M7, 2026-11-07)

Four models, both arms, clean clone, wall time 80 minutes, one final MLP fit 37 seconds.

### Headline

| Arm | Model | Recall | FPR | Pooled ROC-AUC |
|---|---|---|---|---|
| primary | `lr_hand` | 81.9% | 0.69% | 0.953 |
| primary | `lr` | 75.3% | 0.62% | 0.973 |
| primary | `mlp` | 73.5% | 0.83% | 0.967 |
| primary | `trivial` | 61.1% | 0.76% | 0.854 |
| comparison | `lr_hand` | 79.3% | 0.63% | 0.911 |
| comparison | `lr` | 64.0% | 0.76% | 0.937 |
| comparison | `mlp` | 57.9% | 0.35% | 0.924 |
| comparison | `trivial` | 58.5% | 0.49% | 0.861 |

### Paired comparisons (15 folds, first minus second, rule of `DESIGN.md` section 7)

| Comparison | Arm | Recall difference | Verdict |
|---|---|---|---|
| `lr_hand` minus `lr` | primary | +6.2 points (standard error 2.5, 12 of 15 folds) | **justified**, FPR gate holds |
| `lr_hand` minus `lr` | comparison | +14.8 points (3.9, 11 of 15) | justified |
| `mlp` minus `lr` | primary | -1.8 points (1.0) | within noise |
| `mlp` minus `lr` | comparison | -6.3 points (2.5) | worse |
| spans minus real queries | `lr`, `mlp`, `trivial` | positive | justified |
| spans minus real queries | `lr_hand` | +2.7 points (2.0) | not justified: its comparison arm is already strong |

The pre-registered rule made `lr_hand` the recommendation. ROC-AUC, informational, went the other way: it is lower for `lr_hand` (-0.020 in the primary arm), so the gain is at the operating point, not in ranking.

### Per family, primary arm

| Model | A | B | C | D | E (off-topic, never trained on) |
|---|---|---|---|---|---|
| `lr_hand` | 84.4% | 75.5% | 84.7% | 83.1% | 9.0% |
| `lr` | 64.7% | 64.6% | 90.6% | 81.5% | 19.6% |

### Leave-one-family-out, recall on the held-out family

| Arm | Model | A | B | C | D |
|---|---|---|---|---|---|
| primary | `lr_hand` | 78.3% | 49.7% | 82.5% | 70.8% |
| primary | `lr` | 61.1% | 37.9% | 69.3% | 22.8% |
| primary | `trivial` | 35.0% | 49.3% | 70.2% | 69.4% |
| comparison | `lr_hand` minus `lr` | +9.2 | +47.5 | +28.7 | +50.7 (points) |

The B and D transfer now matches the trivial baseline, which was the goal of `lr_hand`.

### Part-number diagnostic (share of 100 generated part numbers flagged)

| Arm | Model | First run | Final run |
|---|---|---|---|
| primary | `lr` | 85% | 78% |
| primary | `lr_hand` | not run | 61% |
| primary | `mlp` | 76% | 32% |
| primary | `trivial` | 3% | 3% |
| comparison | `lr` | 85% | 26% |
| comparison | `lr_hand` | not run | 19% |
| comparison | `mlp` | 81% | 8% |

Primary-arm `lr` still flags 100% of the three bare shapes (`abc000456`, digit runs, `kx-7731b`); only the product-words shape improved (40% to 12%). Sanity rows 39 and 43 (`SKU...`) are still flagged.

### Other checks, final run

- **Sanity set, 50 rows** (confusion matrix [[TN, FP], [FN, TP]]): `lr_hand` [[34, 3], [3, 10]], `lr` [[33, 4], [3, 10]]. Rows 5, 7 and 28 sit above the near-duplicate threshold and are all caught by every model; rows 47 and 49 are caught by none.
- **Catalogue gap**: passes in every arm and model; primary pooled FPR `lr_hand` 0.49%, `lr` 0.76%, `trivial` 0.42%, `mlp` 1.04%.
- **One-word real queries flagged** (of 23): `lr_hand` 2, `lr` 4.
- **False positives** reviewed by eye: none looks machine-generated; `drudge report` and `promo codes or discounts` are borderline; `lr_hand` adds long ordinary product titles.
- **Hard negatives built**: 1,500 requested and 1,500 realised per arm over the 15 folds (375 per shape), no drops, no shortfall.
- **Clustering**: identical to the first run.

## 6. What the changes cost

- **`lr_hand`**: lower ranking quality (AUC), `C.walk` falls from 80% to 23%, `B.template` from 33% to 24%, family E from 20% to 9%. The gain at the operating point is partly length, which is also what makes it transfer.
- **Part-number negatives**: in the comparison arm, where the 100 rows are 21% of the negatives, `lr` family A recall fell from 60% to 33% (`A.hex` 100% to 2%), and primary-arm MLP `A.hex` fell from 96% to 41%. The bare part-number shapes sit on the taxonomy's own boundary with `A.alnum` and `A.hex` (`DESIGN.md` 2.2), so part of this is inherent. `lr_hand` keeps `A.hex` at 100%.
- **The 13.22 recall drop**: after M2.T13 widened two templates, recall on `B.template` fell in every probe and run (92% in the probe, 60% after the widening, 48% then 33% in the full runs). Its cause is not isolated.

## 7. Open items

Recorded in `DESIGN.md` section 14, none scheduled:

- Isolate `lr_hand`'s losses (a length-only variant first), then per template.
- Decide the bare part-number shapes against family A: a sharper rule for what a bare letter-digit string is, or accept that bare part numbers are undecidable per query.
- Isolate the `B.template` recall drop.
- The unlabelled-logs scenario needs behavioural features (rate, session), which are outside this project.
- Housekeeping: the CLI tests that fit real models take about three minutes of the four-minute suite (follow-up F16 in `PLAN.md`).
