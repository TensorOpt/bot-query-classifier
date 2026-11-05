# DESIGN.md: bot-query-classifier

Owner: `fable`. Status: draft v1, 2026-11-05, awaiting user review of section 12.

Each decision is written as **Decision** / **Rationale**. Unverified assumptions and
probe findings live in section 13. Deferred ideas live in section 14 ("Later").

## 1. Goal and scope

Classify a search query string as **legitimate** (serve it) or **illegitimate**
(discard it), using only the query text.

- Negative class (label `0`) = legitimate = the 480 real customer queries in WANDS.
- Positive class (label `1`) = illegitimate = synthetically generated queries.
- A **false positive** (legitimate query discarded) is the costly error: a lost sale.
  The whole evaluation is organised around bounding the false-positive rate (FPR)
  first and maximising recall of illegitimate queries second.

This is an experiment-style project: a reproducible pipeline that trains, evaluates
and reports, plus one separate clustering experiment. Deployment, serving, latency
and online learning are out of scope (section 12, Q9 asks whether that stays true).

**Honest limit of the approach.** A text-only classifier can only catch queries whose
*text* is anomalous. A scraper that searches for "leather sofa" is indistinguishable
from a shopper by this model. Behavioural signals (rate, session, IP, UA) are the
usual way to catch those; they are out of scope here and listed in section 14.

## 2. What "illegitimate" means: proposed taxonomy

The user's request does not define "illegitimate", and the definition drives the
synthetic generator, so this is the first thing to agree (section 12, Q1).

**Decision (proposed):** v1 positive families, each a separate generator module and
a separate `family` tag on every sample:

| id | family | examples | why it is machine-origin |
|----|--------|----------|--------------------------|
| A | gibberish | `asdkjh qwe`, `xk9f2m`, `aaaaaaaa`, UUID/hash-like strings | keyboard mashing, fuzzers, random IDs |
| B | injection / probing payloads | `sofa' or 1=1--`, `<script>alert(1)</script>`, `${{7*7}}`, `../../etc/passwd`, `{"$gt":""}` | security scanners |
| C | enumeration / scraping | bare SKU/ID sequences (`w003412987`), `a`, `aa`, `ab`, ... dictionary walks, URL paths, `page=2&sort=price` | catalogue scrapers |
| D | spam / promotional | URLs, emails, phone numbers, `buy cheap ... click here`, crypto/SEO phrases | comment-spam bots hitting every text field |

**Not in v1, raise with the user:**

| id | family | status |
|----|--------|--------|
| E | off-topic natural language (`what is the capital of france`, chat/LLM prompts, code snippets) | **Open** (Q1). A real customer typing `drudge report` (which is in WANDS, see 13.1) still gets served. Recommend: generate E but use it as an *evaluation-only* family in v1 so we learn what the model does with it before deciding to train on it. |
| F | non-English shopping queries | **Legitimate**, not a positive family. WANDS has no non-ASCII query (13.1), so the model cannot learn this; see 11.6. |
| - | typos and misspellings | **Legitimate**. WANDS contains `desk for kids tjat ate 10 year old`, `trinaic towel rod`, `cool desins`. Generators must not produce "typo" positives. |
| - | single-word queries | **Legitimate**. WANDS has 23 one-word queries incl. `zen`, `owl`, `gold`, `canadian`. Family C "dictionary walk" positives are 1-2 character strings only, never real words. |

**Rationale.** A-D are unambiguously not produced by a shopper and have textual
fingerprints (character classes, punctuation patterns, structure) that TF-IDF over
character n-grams can see. E is a judgement call about product policy, not about
bots, and is the family most likely to produce false positives on unusual but real
queries.

**Hard positives (required in every family).** Each family must include a sub-generator
that *wraps real product vocabulary* (sampled from the training-fold WANDS queries,
see 8.3) in the illegitimate pattern, e.g. `leather sofa' union select`, `cheap area
rug http://x.co`, `accent chair asdfgh`. Otherwise the model learns "furniture words
present = legitimate", which is a dataset artefact, not a bot signal.

## 3. Data: WANDS (verified 2026-11-05)

Facts verified by downloading and inspecting `dataset/query.csv` (probe 13.1), the
repository `LICENSE`, the repository README, and Wayfair's release blog post:

- Repository: `https://github.com/wayfair/WANDS`. Licence: **MIT** (`LICENSE`,
  "Copyright (c) 2021 ecir2022"). Compatible with this repo's Apache-2.0.
- `dataset/query.csv`: **tab-separated** despite the `.csv` extension, UTF-8, header
  `query_id\tquery\tquery_class`, **480 data rows**, all 480 queries unique (also
  unique after strip+lower). `query_id` is `0..479`.
- `query_class` is a product category (`Wall Art`, `Accent Chairs`, ...); 6 rows have
  an empty class. It is not used by this project.
- The queries are **real customer queries**: Wayfair's blog states they were obtained
  by "stratified-sampling of search queries from a pool of historical customer queries
  stored in the e-commerce customer behavior logs", stratified over on-site organic vs
  marketing-redirected searches, with/without engagement, and product popularity.
- Provider normalisation: **all lowercase, all ASCII, no leading/trailing whitespace**,
  3 queries contain a double space. Punctuation is rare (`/` x6, `"` x3, `-` x3, `.`,
  `,`, `(`, `)`, `&`). 44 queries contain digits (`48"`, `7qt`, `3/4 size`).
- Length: 3-67 chars (median 20); 1-10 words (median 3, mean 3.4). Word vocabulary
  834 types, of which **595 (71%) occur once**.
- Content: furniture/home shopping, including brand/collection names (`kisner`,
  `argenteuil`, `minnestrista`), typos, and at least one off-topic query
  (`drudge report`). No query looks machine-generated.
- Other files (`product.csv` 90 MB, `label.csv` 5.7 MB) are **not downloaded**; only
  `query.csv` (~10 KB) is needed. See 14 for the catalogue-anchor idea that would need
  `product.csv`.

**Decision:** `query.csv` is fetched by a script into `data/raw/` (gitignored) and its
SHA-256 recorded in `data/CHECKSUMS` (committed). The loader refuses a file whose hash
does not match. **Rationale:** the download is untrusted data and the dataset is small;
pinning the hash makes every run reproducible.

**Decision:** WANDS is used as negatives only. The 6 empty-class rows and the
off-topic `drudge report` stay in. **Rationale:** they are real customer queries that
were served; removing them would be label leakage by hand-curation.

## 4. Synthetic positives

**Decision:** rule/template-based generators in Python, one module per family
(section 2), driven by a seeded `random.Random`. No LLM-generated positives in v1
(paid API, non-reproducible, and the user must approve any paid call; Q2).

**Decision:** generate 480 positives to match the negatives, with equal quotas per
family (120 each for A-D; if E is added, quotas rebalance). **Rationale:** balanced
classes keep every metric interpretable and avoid any need for re-balancing.

**Every synthetic sample carries metadata:** `family`, `template_id` (the
sub-generator or template that produced it), `seed`. This is what makes the
anti-inflation measures in section 8 possible.

**Generator rules (to stop the model learning the generator instead of the signal):**

1. Output is lowercased and ASCII-only, matching WANDS' provider normalisation
   (section 3). Otherwise case or non-ASCII would be a trivial, false signal.
2. Length distribution per family is logged next to WANDS' (3-67 chars). Length is
   *not* forced to match (payloads are legitimately longer) but the length-only
   baseline in 6.3 quantifies how much of the result is explained by length alone.
3. Every family has a hard-positive sub-generator that embeds real product words
   (section 2). Product words are sampled only from the *training fold's* WANDS
   queries (8.3) so test-fold vocabulary never leaks into training positives.
4. Near-duplicate audit after generation: character-3-gram Jaccard similarity between
   every positive and every negative; any pair above 0.6 is reported, and the
   generation fails if any positive is identical to a negative after strip+lower.
5. Template-level diversity: each template must be able to produce at least 50
   distinct strings; the generator test asserts this.

## 5. Features: TF-IDF

**Decision:** `TfidfVectorizer(analyzer="char", ngram_range=(1, 4), lowercase=True,
sublinear_tf=True, min_df=2)`, fitted inside each training fold only.

**Rationale (char vs word, asked as decision Q6 in the brief):**

- Queries are very short (median 3 words) and 71% of WANDS word types are hapaxes
  (13.1). Word features would be mostly singletons and an unseen-brand query would
  be an all-zero vector.
- The default word `token_pattern` **drops punctuation and single characters**
  entirely (probe 13.3): `sofa' or 1=1--` becomes `[sofa, or]`, and
  `../../etc/passwd` becomes `[etc, passwd]`. Word TF-IDF would blind the model to
  exactly the characters that distinguish families B and C.
- Char n-grams see punctuation sequences, digit runs, and letter-pair plausibility
  (gibberish), and degrade gracefully on out-of-vocabulary words.
- `(1, 4)` with `min_df=2` gives ~5.2k features on 960 strings (13.3); `(2, 5)` gives
  ~6.4k with no obvious benefit at this size. `analyzer="char"` rather than
  `"char_wb"` so that cross-token patterns like `' or` are kept.
- `min_df=2` removes n-grams that occur in one string only; these can only memorise.

**Decision:** preprocessing before vectorisation is exactly: Unicode NFKC normalise,
`str.strip()`, collapse internal whitespace runs to one space, lowercase. Applied
identically to negatives and positives, in one function with one home (`DRY`). An
empty string after preprocessing is a valid input (11.1).

## 6. Models

**Decision:** scikit-learn for everything (vectoriser, models, CV, metrics, HDBSCAN).
No PyTorch/TensorFlow.

**Rationale (standard-library-first, Q7):** the standard library cannot reasonably do
TF-IDF, stratified grouped CV, ROC-AUC and an MLP; a hand-rolled numpy MLP would be
code we must test and maintain for no gain. scikit-learn is required for the
vectoriser, folds and metrics anyway, and `MLPClassifier` adds nothing to the
dependency set. PyTorch is a ~2 GB dependency for a 2-layer network on 960 samples.
Dependencies: `numpy`, `scikit-learn>=1.3` (HDBSCAN is in sklearn from 1.3), and
`imbalanced-learn` for the SMOTE experiment only. `scipy` is transitive.

### 6.1 Neural net (the user's model)

`MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam",
alpha=1e-3, early_stopping=True, validation_fraction=0.2, max_iter=500,
random_state=SEED)`. Input is the sparse TF-IDF matrix (sklearn accepts it).
One hidden layer; the data cannot support more. Early stopping uses a split taken
from the *training* data it is given, so it never sees the test fold (13.6).

### 6.2 Baseline (must be reported next to the MLP)

`LogisticRegression(C=1.0, max_iter=1000)` on identical features. Hypothesis: at
960 samples it will match the MLP; if it does, the recommendation is to ship the
simpler model. The MLP is kept because the user asked for it and the comparison is
the point.

### 6.3 Trivial baseline (artefact detector)

`LogisticRegression` on three hand features: character length, word count,
non-alphanumeric ratio. If this gets close to the MLP, the synthetic set is too easy
and the generators need rework. It is reported, never selected.

### 6.4 Model port

Stages depend on a `Scorer` protocol (`fit(texts, y)`, `score(texts) -> ndarray of
P(illegitimate)`) that wraps `vectoriser -> [SMOTE] -> estimator`. The three models
above are three adapters; the evaluation code never imports sklearn estimators
directly.

## 7. SMOTE

**Decision:** SMOTE is an *experiment flag*, default **off**, run only inside
training folds, compared against the no-SMOTE run on identical folds and seeds.

**Rationale and evidence (Q3):**

- SMOTE balances classes by oversampling the minority. On an exactly balanced
  480/480 set, `SMOTE()` with defaults **generates nothing** (probe 13.4: 960 -> 960
  rows). To "upsample" both classes you must pass
  `sampling_strategy={0: n, 1: n}` with `n > 480`; this works on sparse input (13.4).
- What it then produces is a convex combination of two TF-IDF vectors of the same
  class. Such rows have **L2 norm 0.72-1.0 instead of exactly 1.0 and ~1.7x the
  non-zeros of any real query** (13.4): they lie off the unit sphere that every real
  query lies on, and correspond to no string. For a small MLP this acts as a within-
  class mixup regulariser, which might help a little or not at all. It is not more
  data.
- Therefore the design does not promise an improvement; it measures one. Report:
  paired per-fold difference in the headline metric (8.5), with and without SMOTE,
  for the MLP and the LR. If the difference is within fold-to-fold noise, the
  conclusion is "not justified" and the flag stays off.
- Leakage rule: SMOTE must be fitted and applied to the training portion only.
  `imblearn.pipeline.Pipeline` does this correctly (sampling steps are skipped at
  `predict` time); the pipeline test asserts that the test fold row count is
  unchanged.

A more meaningful augmentation for text (string-level perturbation of real negatives)
is in section 14.

## 8. Evaluation protocol

### 8.1 Folds

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)`, repeated with 3
different seeds (15 outer fits per model; seconds of compute). Groups: each WANDS
query is its own group (all unique, 13.1); each synthetic sample's group is
`(family, template_id)` so that near-duplicates from one template never straddle a
fold. Stratification keeps the 50/50 class ratio per fold (13.7).

### 8.2 Nested fitting: everything inside the training fold

Within each outer training fold, in order: fit preprocessing/vectoriser, (optionally)
SMOTE, estimator. Nothing is fitted on data outside the training fold, including
threshold selection:

- Inner 5-fold CV on the training fold produces out-of-fold scores for every training
  sample. The threshold `t` is the smallest score such that inner FPR <= target FPR
  (8.4). With ~384 training negatives the inner FPR resolution is 1/384 ~ 0.26%.
- The model is then refitted on the whole training fold and `t` is applied to the
  outer test fold.

### 8.3 Positives are regenerated per fold

Hard-positive sub-generators sample product vocabulary from WANDS queries. To avoid
leaking test-fold vocabulary into training positives, the folds are defined on
negatives and on the *(family, template_id)* groups first; the hard-positive
templates then draw vocabulary only from training-fold negatives when producing
training positives, and from test-fold negatives when producing test positives.
Implementation detail for the plan: generation takes a vocabulary argument; the
runner passes the right one.

### 8.4 Operating point: "minimise false positives"

**Decision:** the operating point is a **target FPR of 1%**, chosen on inner
validation data (8.2). Headline number: **recall of illegitimate queries at
FPR <= 1%**, pooled over outer test folds, with the **achieved pooled FPR and its 95%
Wilson interval**.

**Rationale:** "minimise FPR" alone has the trivial solution of serving everything.
A target makes it an operating point. Why 1%: with 480 negatives, one false positive
per outer fold of 96 negatives is already 1.04%, and an observed 0/480 has a 95%
upper bound of 0.77% (rule of three ~0.6%). Anything below 1% is not demonstrable
with this data; the design says so rather than report a number it cannot support.
The user may pick another target (Q4); the code takes it as a parameter.

### 8.5 Metrics reported

For each model x SMOTE setting, per repeat and pooled:

- Headline: recall at FPR <= 1% (8.4), achieved FPR with Wilson 95% CI.
- Confusion matrix at the chosen thresholds (pooled out-of-fold).
- Macro F1, precision, recall (positive class) at the chosen thresholds.
- ROC-AUC, mean +/- sd over folds and pooled (threshold-free sanity number).
- Per-family recall (which kinds of illegitimate queries are missed).
- The list of all false positives (the actual WANDS queries that would be discarded).
  With ~5 FPs at 1% this list *is* the most useful output of the whole project.

Macro F1 and ROC-AUC are reported because the user asked for them; they are not used
for any selection. Selection (threshold) uses only FPR on inner validation data.

### 8.6 Anti-inflation: how would we know the model generalises?

No real bot data exists, so the standard CV above measures "can the model separate
WANDS from *these generators*". Three additional checks make that limitation visible:

1. **Leave-one-family-out (LOFO).** For each family F: train on all other families
   (plus training-fold negatives, same nested protocol), test on F plus test-fold
   negatives. Report recall on F at the threshold chosen without F. This is the
   closest proxy for "a kind of bot we did not anticipate". Expect it to be much
   lower than in-distribution recall; that gap is the honest generalisation number.
2. **Trivial baseline gap** (6.3). If length + punctuation ratio alone reaches the
   MLP's recall, the generators are too easy.
3. **Hand-written sanity set** (`data/sanity.tsv`, committed, ~50 legitimate and
   ~50 illegitimate queries written by a person without reading the generator code,
   legitimate ones deliberately from *other* shop domains: electronics, groceries,
   clothing). Scored by the final models, reported separately, never used for
   fitting or selection. Who writes it is Q5.

All three are reported, not optimised. A run that looks great on 8.5 and poor on
8.6 is reported as exactly that.

### 8.7 Final model for inspection

After evaluation, one model per configuration is fitted on all 960 samples (threshold
from a 5-fold OOF on all data) and saved with `pickle` to `results/` for the
sanity-set scoring and manual probing. Not a deployment artefact.

## 9. Label discovery with HDBSCAN (separate experiment)

**Decision:** clustering is a **separate experiment** (`experiments/cluster`), not a
stage of the training pipeline. The training pipeline always uses known labels.

**Rationale:** the user's scenario is "in production we may not have labels". That
is a question about whether the two classes are *discoverable*, which is answered by
clustering the pooled data with labels hidden and then scoring against the labels.
Mixing it into training would only add a noisier label source to a pipeline that
already has labels.

### 9.1 Pipeline

1. Same preprocessing and char TF-IDF as section 5, fitted on all 960 strings.
2. `TruncatedSVD(n_components=50, random_state=SEED)` then L2-normalise rows, so
   that Euclidean distance approximates cosine distance.
3. `HDBSCAN(min_cluster_size=15, min_samples=None)` on the 50-d vectors.

**Evidence (probe 13.5):** HDBSCAN on the raw 25k-dim TF-IDF matrix, sparse
Euclidean or dense cosine, returns **zero clusters and 100% noise** (distances
concentrate in high dimensions). On SVD-50 + L2 it returns 4 clusters, 41% noise,
ARI 0.76 against the true labels, on crude random-string positives. So the reduction
step is required, not optional. `n_components` and `min_cluster_size` are the only
knobs; a small grid (`n_components` in {20, 50, 100}, `min_cluster_size` in {10, 15,
30}) is reported.

### 9.2 Assigning a class to a cluster

Clusters carry no class. Two rules are implemented and both reported:

- **Oracle majority vote** (measures separability only): each cluster takes the
  majority true label of its members. Report purity per cluster, and overall
  ARI, AMI, homogeneity, completeness, noise fraction, cluster count.
- **Seed labelling** (the deliverable rule for the no-label scenario): a small random
  set of `k` queries (`k` in {10, 20, 40}) is labelled (simulating a human looking at
  a few queries). Each cluster takes the majority label of the seeds it contains.
  Clusters with no seed, and all **noise points, are labelled legitimate (fail
  open)**, because a false positive is the expensive error. Report resulting
  accuracy, FPR and recall against the true labels, averaged over 20 random seed
  draws.

**Many clusters** is expected (product categories cluster among the legitimate
queries; families cluster among the positives) and is not a problem for either rule:
labels are per cluster, not "one cluster per class". **Noise points** are the
problem: 41% noise in the probe means 41% of queries get the fail-open label. That
number is a result to report, not something to hide.

A catalogue-anchored rule (legitimate = close to product names from `product.csv`)
would remove the need for seeds, but needs the 90 MB file; section 14.

## 10. Code structure, reproducibility, commands

```
bqc/                      package
  preprocess.py           normalise() (section 5), one home
  wands.py                download + hash check + load (section 3)
  synth/                  one module per family + registry; sample metadata
  features.py             vectoriser factory (section 5)
  models.py               Scorer protocol + three adapters (section 6)
  evaluate.py             folds, nested threshold selection, metrics (section 8)
  cluster.py              HDBSCAN experiment (section 9)
  cli.py                  python -m bqc <command>
tests/                    unittest; python -m unittest discover -s tests
data/raw/                 gitignored downloads; data/CHECKSUMS committed; data/sanity.tsv committed
results/                  gitignored JSON reports and pickles, one file per run
scratch/                  gitignored probes
```

- **Seeds:** one `SEED = 20261105` constant with one home; every `random_state`,
  `random.Random` and repeat seed derives from it (`SEED + repeat_index`).
  `MLPClassifier` is deterministic given `random_state` (13.6).
- **Results** are JSON: config (all parameters, seed, git commit hash, library
  versions), per-fold and pooled metrics, FP list, per-family recall. One file per
  run; never overwritten (file name includes config hash and timestamp).
- **Commands** (for CLAUDE.md "Commands" once the user fills it):
  `python -m bqc fetch`, `python -m bqc generate`, `python -m bqc evaluate
  [--smote] [--target-fpr 0.01] [--lofo]`, `python -m bqc cluster`,
  `python -m unittest discover -s tests`.
- **Environment:** Python 3.12, `python3 -m venv .venv`, `pyproject.toml` with the
  dependencies from section 6. The user's pyenv 3.12.16 has no `pip` on the shim
  path but `python3 -m venv` works and gives a venv with pip (verified 2026-11-05).
- **Tests do not download.** The WANDS loader is tested against a 10-row fixture
  with the same header and delimiter; the hash check is tested with a wrong hash.
  Model tests use tiny synthetic inputs and assert behaviour (no leakage, row counts,
  determinism), not metric values.

## 11. Edge cases and failure modes (starting list)

Each item names the component and the designed behaviour.

1. **Empty or whitespace-only query.** `normalise()` returns `""`; the vectoriser
   produces an all-zero row; the model returns a score (LR: sigmoid of the intercept).
   The CLI/`Scorer` accepts it and returns a score; the evaluation reports it. Not an
   error. (A real search box can submit empty input.)
2. **Very long input** (e.g. 10 kB pasted text or a 1 MB payload). `normalise()`
   truncates to 512 characters before vectorising; the truncation is logged in the
   score output. Rationale: WANDS max is 67; nothing beyond 512 is a query, and char
   n-gram extraction on megabytes is a DoS surface.
3. **Out-of-vocabulary-only query** (e.g. an unseen brand, `minnestrista`). With char
   n-grams some n-grams will be known; with `min_df=2` a fully unknown string yields
   a zero row and gets the same score as the empty string. The test asserts a zero row
   does not raise and the FP list lets us see whether such queries are discarded.
4. **Unicode: non-ASCII letters, combining marks, emoji, zero-width characters,
   full-width ASCII.** NFKC normalisation maps full-width and compatibility forms;
   zero-width and control characters are stripped in `normalise()`; everything else
   is kept as-is. The model has never seen non-ASCII negatives (section 3), so
   non-ASCII will likely score as illegitimate; this is recorded as a known
   limitation (Q6), and the sanity set includes a few accented legitimate queries so
   the effect is measured.
5. **Case.** Lowercased in `normalise()`; the model cannot use case (section 4).
6. **Non-English legitimate queries** (`sofá de cuero`). Same as 4: likely false
   positives. Not solvable with WANDS as the only negative source; see 14.
7. **Synthetic positive identical or near-identical to a WANDS query.** Generation
   fails on exact matches and reports near-duplicates (section 4, rule 4).
8. **Synthetic near-duplicates straddling folds.** Prevented by grouping on
   `(family, template_id)` (8.1); the fold test asserts no group appears in both sides.
9. **Leakage of vectoriser / SMOTE / threshold.** All fitted inside the training
   fold (8.2). Test: fit on a training fold, assert vocabulary contains no n-gram
   that exists only in the test fold; assert test-fold row count is unchanged after
   the pipeline.
10. **Threshold not achievable.** If no threshold reaches the target FPR on inner
    validation (every positive overlaps with some negative), choose the threshold
    with the smallest FPR and flag `target_unmet=true` in the report.
11. **Target FPR below resolution.** If `target_fpr < 1/n_inner_negatives`, warn in
    the report; do not silently round to zero FPs.
12. **MLP fails to converge** (`ConvergenceWarning`). Captured and recorded in the
    run JSON per fold; not an error.
13. **WANDS download fails or changes.** `fetch` fails loudly on network error or
    hash mismatch; nothing partial is written (download to a temp file, then rename).
14. **WANDS file quirks.** Tab delimiter with `.csv` extension; 3 rows contain `"`
    and are standard-CSV quoted (`"fawkes 36"" blue vanity"`), so the `csv` module
    with `delimiter="\t"` and **default quoting** is correct and `QUOTE_NONE` would be
    wrong (13.8). The loader test uses a fixture with such a row.
15. **Empty `query_class`.** Ignored; the loader does not require it.
16. **Degenerate fold** (a family or group missing from a training fold). With 5
    folds and 120 samples per family this cannot happen for whole families; LOFO
    removes a family on purpose. The runner asserts both classes are present in every
    training and test fold.
17. **All-noise HDBSCAN result.** Reported as 0 clusters, 100% noise; the seed rule
    then labels everything legitimate (FPR 0, recall 0). Not an error.
18. **Non-determinism.** Two runs with the same config and seed must produce
    identical JSON apart from the timestamp; this is a test on a tiny config.
19. **Scores exactly at the threshold.** Decision is `score >= t` is illegitimate;
    stated once in `evaluate.py` and tested.

## 12. Open questions for the user

Each with the recommended answer. The plan should not start until Q1-Q4 are settled.

1. **Taxonomy (section 2).** Are families A-D the right definition of
   "illegitimate", and is E (off-topic natural language) in, out, or
   evaluation-only? *Recommend: A-D in; E evaluation-only in v1.*
2. **LLM-generated positives.** Should an LLM be used (paid API) to add diversity to
   the generators or to write family E? *Recommend: no in v1; templates are
   reproducible and free. Revisit if LOFO recall is poor.*
3. **SMOTE.** Keep it as a measured experiment with default off (section 7)?
   *Recommend: yes.*
4. **Target FPR.** 1% (section 8.4)? *Recommend: 1%; lower is not measurable with
   480 negatives.*
5. **Sanity set authorship (8.6).** Will the user write or review the ~100-query
   hand-written set? *Recommend: the user writes the ~50 legitimate ones from memory
   of real shop searches (highest value, 15 minutes); sonnet drafts the illegitimate
   ones and the user reviews.*
6. **Non-ASCII / non-English legitimate queries (11.4, 11.6).** Accept as a known
   limitation of v1? *Recommend: yes, record it, and note the extra negative source
   in section 14 as the fix.*
7. **Second real negative source for evaluation only** (e.g. a 500-query sample from
   Amazon ESCI, public shopping queries). Would show whether the FPR holds on a
   different shop. *Recommend: defer to section 14; decide after v1 numbers.*
8. **Branch name.** CLAUDE.md says commit on `master`; the repo's branch is `main`.
   *Recommend: change CLAUDE.md to `main`.*
9. **Deployment.** Stays out of scope? *Recommend: yes. The only serving-related
   decision taken is input truncation (11.2), because it affects the model's input
   contract.*

## 13. Probe findings and unverified assumptions

Format: date, versions, what was run, finding. Probe scripts were run from a
scratch directory with `python -I`, with the untrusted download in its own directory.

### 13.1 WANDS `query.csv` inspection (2026-11-05, Python 3.12.16, stdlib `csv`)

Downloaded `https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv`.
Findings as listed in section 3: TSV, 480 rows, 480 unique, 3 columns, lowercase,
ASCII-only, no empty rows, lengths 3-67 chars / 1-10 words, 834 word types with 595
hapaxes, 6 empty `query_class`, punctuation rare, 44 rows with digits, `drudge
report` present, typos present. Licence file: MIT.

### 13.2 WANDS provenance (2026-11-05, web)

Wayfair tech blog ("Wayfair releases WANDS...") states the queries were stratified-
sampled from historical customer queries in behaviour logs. Neither the README nor the
blog mentions bot filtering or normalisation. **Unverified assumption:** the 480
queries contain no bot traffic. Mitigation: the FP list (8.5) is reviewed by eye.

### 13.3 TfidfVectorizer behaviour (2026-11-05, scikit-learn 1.9.1, numpy 2.5.4)

- Default word analyser on `sofa' or 1=1--`, `<script>alert(1)</script>`,
  `../../etc/passwd`, `a` yields vocabulary `[alert, etc, or, passwd, script, sofa]`:
  all punctuation, digits-with-operators and single characters are lost.
- Char analyser on 480 WANDS + 480 crude random-string positives, `sublinear_tf`:
  `(1,3)` 12,872 features (3,749 with `min_df=2`); `(1,4)` 25,711 (5,234);
  `(2,5)` 39,065 (6,402); `(3,5)` 37,046 (4,454). Word analyser on WANDS alone: 814.

### 13.4 SMOTE on balanced sparse TF-IDF (2026-11-05, imbalanced-learn 0.14.2)

- `SMOTE(random_state=0).fit_resample(X, y)` with 480/480: output 960 rows, i.e.
  **no samples generated**.
- `SMOTE(sampling_strategy={0: 1000, 1: 1000})`: output 2000 rows, sparse CSR
  preserved. Synthetic rows: mean 122.8 non-zeros vs 71.3 for real rows; L2 norms
  min 0.72 / mean 0.85 / max 1.00 (real rows are exactly 1.0).

### 13.5 HDBSCAN on TF-IDF (2026-11-05, scikit-learn 1.9.1)

- `sklearn.cluster.HDBSCAN` **accepts sparse CSR input** (contrary to the author's
  prior belief; this is why it was probed). It emits a `FutureWarning` about the
  `copy` parameter default changing in 1.10; set `copy` explicitly.
- Raw TF-IDF `(1,4)`, 25,711 dims, `min_cluster_size=10`: sparse Euclidean -> 0
  clusters, 100% noise; dense `metric="cosine"` -> 0 clusters, 100% noise.
- `TruncatedSVD(50)` + L2 normalise, Euclidean, `min_cluster_size=10`: 4 clusters,
  41.5% noise, ARI 0.759 vs true labels (positives were crude random strings, so
  this is an upper-bound-ish sanity figure, not a result).
- `metric="cosine"` on the dense 50-d matrix is accepted.

### 13.6 MLPClassifier (2026-11-05, scikit-learn 1.9.1)

- Two fits with `random_state=0` on identical sparse input give bit-identical
  `predict_proba`.
- `early_stopping=True, validation_fraction=0.2` fits fine on sparse input (stopped
  at iteration 23 on the probe data). The validation split is drawn from the data
  passed to `fit`, so inside a CV training fold it does not touch the test fold.

### 13.7 StratifiedGroupKFold (2026-11-05, scikit-learn 1.9.1)

480 singleton groups (negatives) + 120 groups of 4 (positives), `n_splits=5,
shuffle=True`: every fold has 192 test rows, exactly 50% positive, and no group on
both sides.

### 13.8 WANDS quoting (2026-11-05, Python 3.12.16, stdlib `csv`)

The three queries containing `"` are stored standard-CSV quoted:
`"fawkes 36"" blue vanity"`, `"48"" sliding single track , barn door for laundry"`,
`"writing desk 48"""`. Default `csv` quoting reads them as `fawkes 36" blue vanity`
etc.; `quoting=QUOTE_NONE` returns the raw quoted form. Use default quoting.

### 13.9 Unverified assumptions (to be probed by the plan's probe tasks)

- `imblearn.pipeline.Pipeline` skips the SMOTE step at `predict`/`predict_proba`
  time (documented behaviour; not yet probed here).
- Wilson interval implementation: written in-house with the standard library
  (`math`), tested against known values; no `statsmodels`.
- The 480 WANDS queries contain no bot traffic (13.2); not provable, mitigated by
  reviewing the FP list by eye.

## 14. Later (deferred, not in scope)

- **Behavioural features** (rate, session, IP reputation): the only way to catch
  scrapers that send well-formed queries. Different data, different project.
- **Second real negative source** (Amazon ESCI sample, Home Depot queries) for
  evaluation-only FPR on a different shop and for non-English coverage.
- **Catalogue-anchored cluster labelling**: label a cluster legitimate if its members
  are close to product names from WANDS `product.csv` (90 MB). Removes the need for
  seed labels in section 9.2.
- **String-level augmentation of negatives** (typo injection, word dropout) as a more
  meaningful alternative to SMOTE for text.
- **Word + char feature union**, and a `char_wb` comparison, if the FP list suggests
  word-level cues are being missed.
- **Partial AUC** (FPR in [0, 0.05]) as a threshold-free number closer to the
  operating region than full ROC-AUC.
- **Calibration** of scores (not needed while the decision is a thresholded rank).
- **Family E (off-topic natural language) as a training family**, if the
  evaluation-only run shows it is separable without new false positives.
- **LLM-generated positives** for diversity, with the user's approval of paid calls.
- **Serving**: input contract (11.2) is the only serving decision taken; a model
  file format, latency budget and monitoring would come with a deployment decision.
