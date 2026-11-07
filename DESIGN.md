# DESIGN.md: bot-query-classifier

Owner: `fable`. Status: v2.8, 2026-11-07 (after the M6 run: logistic regression chosen,
12.1 item 18; a combined-feature LR adapter, 6.5, and part-number hard negatives, 3.3,
added for M7). v2.7, 2026-11-06, was Q13: product-name spans are the training
negatives; the real queries set thresholds and test. The user answered Q1-Q11 and
Q13 (12.1);
`opus`'s plan-review questions and additions are resolved in 12.3 and applied in
the body. One question (12.2, Q12, synthetic part-number negatives) is open; it
waits for the diagnostic of 8.6 item 4 and blocks nothing before M6.

Each decision is written as **Decision** / **Rationale**. Unverified assumptions and
probe findings live in section 13. Deferred ideas live in section 14 ("Later").

## 1. Goal and scope

Classify a search query string as **legitimate** (serve it) or **illegitimate**
(discard it), using only the query text.

**The use case is a furniture and home-goods shop**, i.e. WANDS's own domain (user
decision 2026-11-05, section 12 Q7). Every data source, including the hand-written
sanity set, stays in that domain; queries for products such a shop does not sell
(`iphone 11 case`) are not mixed in as legitimate (and are not labelled illegitimate
either; see 2.1 and Q10).

- Negative class (label `0`) = legitimate. **Training negatives are word spans cut
  from WANDS product names** (3.2; user decision 2026-11-05, Q13). The **480 real
  customer queries** in WANDS are the legitimate class as it is *measured*: they
  never enter a model fit; they set the threshold and are the test set, through
  cross-fitting (8.2). The v1 setup (real queries as training negatives) is kept as
  a comparison arm.
- Positive class (label `1`) = illegitimate = synthetically generated queries, as
  many as there are training negatives (section 4).
- A **false positive** (legitimate query discarded) is the costly error: a lost sale.
  The whole evaluation is organised around bounding the false-positive rate (FPR)
  first and maximising recall of illegitimate queries second.

This is an experiment-style project: a reproducible pipeline that trains, evaluates
and reports, plus one separate clustering experiment. Deployment, serving, latency
and online learning are out of scope (decided, section 12 Q9). The only
serving-related decision taken is the input truncation in 11.2, because it defines
the model's input contract.

**Honest limit of the approach.** A text-only classifier can only catch queries whose
*text* is anomalous. A scraper that searches for "leather sofa" is indistinguishable
from a shopper by this model. Behavioural signals (rate, session, IP, UA) are the
usual way to catch those; they are out of scope here and listed in section 14.

## 2. What "illegitimate" means: taxonomy

The user's request does not define "illegitimate", and the definition drives the
synthetic generator. The taxonomy below was accepted by the user on 2026-11-05
(section 12, Q1).

**Decision:** v1 positive families, each a separate generator module and a separate
`family` tag on every sample:

| id | family | examples | why it is machine-origin |
|----|--------|----------|--------------------------|
| A | gibberish | `asdkjh qwe`, `xk9f2m`, `aaaaaaaa`, UUID/hash-like strings | keyboard mashing, fuzzers, random IDs |
| B | injection / probing payloads | `sofa' or 1=1--`, `<script>alert(7)</script>`, `${{7*7}}`, `../../../etc/hosts`, `{"$gt":""}` | security scanners |
| C | enumeration / scraping | `a`, `aa`, `ab`, ... dictionary walks, URL paths (`/category/sofas?page=3`), query-string fragments (`page=2&sort=price`), pagination/offset sweeps (`offset=340&limit=20`) | catalogue scrapers walking the search engine rather than the catalogue |
| D | spam / promotional | URLs, emails, phone numbers, `buy cheap ... click here`, crypto/SEO phrases | comment-spam bots hitting every text field |

**Not trained on in v1:**

| id | family | status |
|----|--------|--------|
| E | off-topic natural language (`what is the capital of france`, chat/LLM prompts, code snippets) | **Evaluation-only** (decided, Q1). Generated like A-D (own module, own `family` tag) but excluded from every training fold; scored by the fold models and reported as a per-family recall number in 8.5. A real customer typing `drudge report` (which is in WANDS, see 13.1) still gets served, which is why E is not a training family until we have seen what the model does with it (14). |
| F | non-English shopping queries | **Legitimate**, not a positive family. WANDS has no non-ASCII query (13.1), so the model cannot learn this; accepted as a known v1 limitation (Q6), see 11.6. |
| - | off-domain shopping queries (`iphone 11 case`) | **No label in v1**; excluded from every data set. See 2.1 (decided by the user, Q10 in 12.1). |
| - | typos and misspellings | **Legitimate**. WANDS contains `desk for kids tjat ate 10 year old`, `trinaic towel rod`, `cool desins`. Generators must not produce "typo" positives. |
| - | single-word queries | **Legitimate**. WANDS has 23 one-word queries incl. `zen`, `owl`, `gold`, `canadian`. Family C "dictionary walk" positives are 1-2 character strings only, never real words. |
| - | SKUs, part numbers, model numbers (`abc000456`, `w003096809`, `kx-7731b desk`) | **Legitimate** (user decision 2026-11-05, Q11: "SKUs should remain legit, customers might search by part numbers"). Removed from family C. WANDS itself contains `ines shag rug w003096809` (13.15). No generator may emit a part-number shape as a whole query (2.2). |

**Rationale.** A-D are unambiguously not produced by a shopper and have textual
fingerprints (character classes, punctuation patterns, structure) that TF-IDF over
character n-grams can see. E is a judgement call about product policy, not about
bots, and is the family most likely to produce false positives on unusual but real
queries.

### 2.1 Off-domain shopping queries

A query such as `iphone 11 case` or `running shoes size 10` is a well-formed product
search for something a furniture and home-goods shop does not sell. The user ruled
(Q7) that such queries are **not** to be mixed in as legitimate; the user did not
say they are illegitimate, and this design does not invent that label.

**Decision:** off-domain shopping queries are **outside both classes in v1**. They
are not generated as a positive family, they are not added as negatives, and the
sanity set (8.6) must not contain any. They are distinct from family E: E is text
that is *not a product search at all* (questions, chat, code), whereas an off-domain
query is a product search for the wrong shop. Whether a shop should discard
off-domain searches is a product-policy question, not a bot-detection one, and is
was confirmed by the user on 2026-11-05 (12.1, Q10).

**Consequence to record:** the model's behaviour on off-domain queries is undefined
and unmeasured in v1. WANDS itself contains a few queries at the edge of the domain
(`printers`, `dumbbells`, `drudge report`); they stay legitimate because Wayfair
served them to real customers (section 3).

**Hard positives (required in every family).** Each family must include a sub-generator
that *wraps real product vocabulary* (sampled from the training-fold WANDS queries,
see 8.3) in the illegitimate pattern, e.g. `leather sofa' union select`, `cheap area
rug http://x.co`, `accent chair asdfgh`. Otherwise the model learns "furniture words
present = legitimate", which is a dataset artefact, not a bot signal.

### 2.2 Part numbers vs gibberish: the boundary for template authors

Removing SKUs from the positives is not enough if family A can still emit strings
that look like part numbers. The user's decision draws the line: `rt4kq8wvz` and
`zzzzxxxxcccc 5190284` are illegitimate, `kx-7731b desk` and `abc000456` are
legitimate. The distinguishing feature is **block structure**, not the presence of
digits:

- A **part-number shape** is a short alphabetic prefix (1-4 letters), an optional
  `-` or `/`, a run of 3 or more digits, and at most one short alphanumeric suffix;
  or a bare run of digits. Letters and digits form at most three contiguous blocks.
  Regex, applied to a whole token after `normalise()`:
  `^[a-z]{0,4}[-/]?\d{3,}[a-z0-9]{0,2}$`.
- **Gibberish with digits** (family A) interleaves letters and digits in four or
  more blocks (`rt4kq8wvz` is `rt|4|kq|8|wvz`). A keyboard mash next to a bare
  digit run (`zzzzxxxxcccc 5190284`) is illegitimate too but is **not** a
  generator form: its digit token is one block and would fail the rule below, and
  a bare digit run next to anything is too close to a part number for a template
  to be trusted with. Family A does not generate it (decided 2026-11-05; the rule
  below holds, the description that previously allowed the form was the
  contradiction).

**Decision (extended 2026-11-05 after M2.T1 validation):** every template output,
in every family, must contain **at least one token that is not benign**. A token
is *benign* if, after `normalise()` and whitespace splitting, it is either

- a **vocabulary word**: a member of `set(vocabulary)`, where the vocabulary
  passed to the generator is the list of whitespace-token occurrences of the
  caller's negatives, with repeats (8.3: the product spans for training positives
  in the primary arm, the training-side real queries in the comparison arm, the
  test-side real queries for test positives); membership ignores the repeats; or
- a **number-shaped token**: it contains a digit, matches
  `^(?=.*\d)[a-z0-9]+([-/.x][a-z0-9]+)*"?$`, **and has at most three letter/digit
  blocks** (a block is a maximal run of letters or of digits; separators do not
  count). This covers bare numbers (`72`), part numbers (`w003412`, `abc000456`,
  `kx-7731b`), and the measurement forms WANDS uses (`18x18`, `3/4`, `72.5`,
  `3-3/4`, `36"`, `7qt`, `e12/candelabra`); all 36 digit-bearing WANDS tokens match
  and have at most 3 blocks (verified 2026-11-05). The block limit is what keeps
  `rt4kq8wvz` (5 blocks) and UUID/hash-like tokens non-benign, as the table below
  requires; without it the regex alone would match them. (Corrected 2026-11-05
  after `opus` found the omission.)

A string made only of benign tokens is the shape of a legitimate query
(`leather sofa w003412`, `area rug 18x18`, `sofa`, `w003412`, `72`), and no family
may emit one as a positive. The earlier whole-query ban (a lone token matching
the part-number regex of the bullet above, or a bare number) is **kept as a
separate check in family C's tests**: the two overlap but neither contains the
other (`/123` matches the part-number regex yet is not benign).

**Accepted consequence:** a real part number with four or more blocks (`a1b2c3`)
is non-benign, so a template may lawfully emit `leather sofa a1b2c3` as a
positive. This is accepted because the shop's own part numbers have at most three
blocks (every digit-bearing WANDS token does), because family A's four-block rule
makes four-plus blocks the gibberish signature on purpose, and because the
part-number diagnostic (8.6 item 4) generates the three-block shapes the user's
decision is about. If such part numbers turn up in real traffic, the block limit
is one constant to change.

Why "plain word" means *vocabulary word* and not *letters-only token*: family A's
keyboard mashes (`asdkjh qwe`) and family E's sentences (`what is the capital of
france`) are letters-only and must stay allowed; what makes `leather sofa w003412`
legitimate is that its words are the shop's words. A letters-only token outside
the vocabulary (`asdkjh`, `capital`, `select`) is therefore non-benign and
satisfies the rule on its own.

What passes and what fails (vocabulary = WANDS words):

| output | verdict | why |
|---|---|---|
| `leather sofa w003412`, `area rug 18x18`, `sofa`, `w003412`, `72` | **fails** | all tokens benign |
| `leather sofa' union select`, `sofa' or 1=1--` | passes | `sofa'`, `union`, `select` are not benign |
| `area rug&page=2`, `/search?q=accent+chair`, `page=17` | passes | `&`, `/`, `?`, `=` tokens |
| `cheap area rug http://x.co`, `call (555) 012-3456 now` | passes | URL; `(555)`, `call`, `now` (M2.T5: phone numbers are never a bare digit run) |
| `accent chair asdfgh`, `rt4kq8wvz`, `aaaaaaaa` | passes | non-vocabulary letter tokens; 5-block token |
| `aa`, `ab` (dictionary walk) | passes | not vocabulary words |
| `a`, `x`, `or`, `tv` (dictionary walk) | **fails** | WANDS vocabulary words; the walk template must skip vocabulary tokens |
| `what is the capital of france` | passes | `capital`, `france` outside the vocabulary |

Family A templates that use digits must additionally produce at least four
letter/digit blocks in **every** digit-bearing token (so a token never has a
part-number shape even inside a longer string), and family A's hard-positive
templates combine vocabulary words with gibberish tokens only, never with
digits. Four is the floor, not the target: the implemented A template uses a
minimum of **seven** blocks (M2.T2), which keeps training positives further
from three-block part numbers; the examples `xk9f2m` and `rt4kq8wvz` (five blocks
each) are illustrations of the shape, not strings the template must be able to
produce (the revisit trigger for the minimum is recorded in 8.6.1). The
benign-token predicate, the block-count function and the
"at least one non-benign token" check have one home, `bqc/synth/registry.py`;
the registry contract test (`assert_template_rules`) applies them to every
output of every template in every family, with a fold-sized vocabulary. The
property is structural per template (a template either always carries a
non-benign marker or does not), so it is a contract test, not a generation-time
audit; the generation-time audit stays the exact-match check of section 4 rule 4.

**Family C after the change (section 4 rule 6 still holds: six templates, at least
two hard-positive):** (1) dictionary walks; (2) URL paths; (3) generic query-string
fragments with random keys; (4) pagination and offset sweeps with the usual key
names; (5) hard-positive: product words plus a query-string fragment
(`area rug&page=2`); (6) hard-positive: a search-URL form around product words
(`/search?q=accent+chair`). Templates 3 and 4 are distinct in what they vary:
3 varies the keys, 4 varies the numbers.

## 3. Data: WANDS (verified 2026-11-05)

Facts verified by downloading and inspecting `dataset/query.csv` (probe 13.1), the
repository `LICENSE`, the repository README, and Wayfair's release blog post:

- Repository: `https://github.com/wayfair/WANDS`. Licence: **MIT** (`LICENSE`,
  "Copyright (c) 2021 ecir2022"). Compatible with this repo's Apache-2.0.
- `dataset/query.csv`: **tab-separated** despite the `.csv` extension, UTF-8, header
  `query_id\tquery\tquery_class`, **480 data rows**, all 480 queries unique (also
  unique after strip+lower). `query_id` runs `0..487` in ascending order, 480
  distinct values with 8 gaps (154, 187, 192, 217, 299, 350, 450, 477); it is an
  identifier, not a row index (corrected 2026-11-05; the v1 draft said `0..479`).
- `query_class` is a product category (`Wall Art`, `Accent Chairs`, ...); 6 rows have
  an empty class. Its only use is the catalogue-gap check (8.6 item 5, 11.15), which
  compares it, whole, against the `|`-split classes of `product.csv` (3.1); nothing
  else reads it.
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
- Other files: `product.csv` (90 MB) **is downloaded since v2.0** (3.1, Q13);
  `label.csv` (5.7 MB) is not needed and not downloaded. The catalogue-anchor idea
  for clustering (14) could now reuse the product file.

**Decision:** `query.csv` is fetched by a script into `data/raw/` (gitignored) and its
SHA-256 recorded in `data/CHECKSUMS` (committed). The loader refuses a file whose hash
does not match. **Rationale:** the download is untrusted data and the dataset is small;
pinning the hash makes every run reproducible.

**Decision (v1, superseded 2026-11-05 by Q13, see 3.2):** ~~WANDS queries are the
training negatives.~~ The real queries are now the **threshold and test set only**.
What stays: WANDS queries are the only real legitimate data; the 6 empty-class rows
and the off-topic `drudge report` stay in; nothing is hand-curated out.

### 3.1 WANDS `product.csv` (verified 2026-11-05, 13.18)

- `dataset/product.csv`, **90.6 MB**, tab-separated, UTF-8, header `product_id,
  product_name, product_class, category hierarchy, product_description,
  product_features, rating_count, average_rating, review_count` (the fourth name has
  a space). **42,994 rows, 0 malformed**, 0 empty names, **42,577 distinct names**.
- Names are lowercase (0 with upper case), **918 contain non-ASCII**, 1-49 words,
  **median 6 words** (queries: median 3). `product_class` is a `|`-separated list in
  the same vocabulary as the queries' `query_class`.
- Coverage: product names contain **94.8% of real-query token occurrences (90.9% of
  types)**; other real queries cover only 63.3% / 28.7% of a held-out query's tokens.
  That is the whole reason for 3.2: the catalogue phrases what shoppers type.
- Fields other than `product_id`, `product_name`, `product_class` are not read. The
  description column is long: the loader raises `csv.field_size_limit`.

**Decision:** `bqc fetch` downloads `product.csv` next to `query.csv`, hash-pinned in
`data/CHECKSUMS` (SHA-256
`d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6`), same atomic
write; `bqc/wands.py` gains `load_products()` returning `(product_id, name,
classes)` with the three columns only, raw, in file order. The 90 MB download is
accepted because it is the only real legitimate text we have at scale (13.19).

### 3.2 Training negatives are product-name spans (user decision 2026-11-05, Q13)

**Decision:** a training negative is a **contiguous span of consecutive words cut
from one product name**. Construction, one home in `bqc/negatives.py`, seeded by
`random.Random(SEED + repeat_index)`:

1. Preprocess every product name with `normalise()` (section 5), then fold to
   ASCII (NFKD, drop non-ASCII, re-collapse whitespace). WANDS queries are ASCII
   (section 3), so negatives must be too, or non-ASCII would become a class signal
   (rule 1 of section 4 applies to negatives for the same reason).
2. Draw a span length `k` from the real queries' word-count distribution recorded in
   13.1 (1 word: 23, 2: 109, 3: 169, 4: 99, 5: 41, 6: 16, 7: 11, 8: 8, 9: 3, 10: 1,
   out of 480), a product uniformly among those with at least `k` words, and a
   start position uniformly; take the `k` consecutive words. The distribution is a
   published statistic of the dataset, written as a constant; it is not recomputed
   from the fold's queries.
3. Drop the span if, after `normalise()`, it is empty, already drawn (exact
   duplicate), **equal to any of the 480 real queries**, equal to a part-number
   diagnostic string (8.6 item 4), or **equal to a sanity query**. All four
   removals go through the audit's dropping entry point (`drop_exact` in
   `bqc/synth/audit.py`, additive next to the raising `check_exact` used for
   positives, rule 4): it takes the candidate strings and the lists, returns the
   survivors and per-list counts, and never returns or logs the removed text. It is
   the only code that touches the sanity queries; the span builder never reads the
   file (rule 8). The counts go in the run JSON; the removed text does not.
4. Stop at `n` spans (default **5,000**, CLI `--negatives-size`).

**Known skew, accepted at M2 sign-off (13.20):** exact deduplication after drawing
removes short spans more often than long ones (one-word spans collide most), so
the realised mix is slightly longer than the target: at 5,000, one-word spans
3.9-4.5% against 4.8%, two-word 20-21% against 22.7%, mean 3.46-3.49 words against
3.38; at 20,000 the one-word share falls to 3.3%. Accepted because the deviation
is under one percentage point per length at the default size, the threshold is
set on the real queries (which carry the true mix), and the one-word false
positives of 8.5 measure the consequence directly. Every run JSON records the
realised span length mix next to the target; if the one-word false-positive count
moves between 5,000 and the 20,000 sensitivity run, per-length quotas (14) are the
fix.

Every span carries metadata: `product_id`, `k`, `seed`. Spans from the same product
are near-duplicates of each other; they do not need group-aware folds because
**spans are never on the test side** (8.1): the test side is real queries.

**Why spans and not whole names (13.19):** full names (median 6 words, P1) train a
model whose score distribution does not transfer to 3-word queries (threshold from
held-out names gives 18% FPR on real queries); spans with the queries' length
distribution (P2) transfer (0.2%).

**Why 5,000 (13.19):** the learning curve at 500 / 2,000 / 5,000 / 20,000 / 40,000
spans gives recall 73.6 / 79.2 / 81.2 / 83.4 / 84.9% at FPR <= 1% and AUC .958 /
.967 / .973 / .980 / .983; FPR stays within noise (0.4-1.1%). Five thousand
captures three quarters of the gain of 40,000 at an eighth of the data, keeps the
MLP at about 17 s per fit (measured, 13.22) instead of 16 minutes, and keeps the
Jaccard audit (5,000 x 5,000 pairs) in seconds. The size is a parameter so the 20,000 point can be re-run
for logistic regression as a reported sensitivity.

**What the real queries are for now:** per outer fold, the 384 training-side real
queries set the threshold (scored by a model that never saw any real query) and
the 96 test-side ones are the FPR test set; every query is a test item exactly once
per repeat, so the 480-query Wilson arithmetic of 8.4 is unchanged. Adding real
queries to the training spans was tested (BP arm) and rejected: no gain, more false
positives (13.19).

### 3.3 Part-number hard negatives (user decision 2026-11-07, Q12)

**Decision:** every training side also gets **100 part-number hard negatives**
(label 0), generated, one home in `bqc/negatives.py`, reusing the part-number
shape machinery of `bqc/synth/partnumbers.py` (8.6 item 4; one home for the
shapes), seeded by `random.Random(SEED + PART_NUMBER_SEED_OFFSET + repeat_index)`
with `PART_NUMBER_SEED_OFFSET = 100_000` so no other seed of this design (spans
`SEED + r`, inner folds `SEED + r`, draws `SEED + i`, the diagnostic set `SEED`)
can coincide with it:

1. The four shapes of 8.6 item 4 in equal share (25 each): bare prefix+digits, bare
   digit runs, prefix-digits-suffix, and **product words plus a part number**. The
   product words are sampled from the token occurrences of the arm's training
   negatives (the spans in the primary arm, the training-side real queries in the
   comparison arm), exactly as hard positives are (rule 3 of section 4), so no
   test-side vocabulary reaches training.
2. Drop a candidate equal, after `normalise()`, to a span, a real query, a
   diagnostic P string (8.6 item 4), a sanity query or an earlier candidate, through
   `drop_exact` (3.2 step 3); counts in the run JSON, text never. Draw until 100
   survive or a cap of 20 x 100 candidates is reached, then raise (`PoolExhausted`
   form of 11.25); never pad.
3. They are **training-only material** like spans: never on the test side, never in
   a threshold set, never in the clustering experiment (section 9: the scenario is a
   query log, and these are not queries), rebuilt per repeat, included in both arms
   and in the final-model fits (8.7). Positives keep rendering from the spans' or
   queries' occurrences only: part-number tokens never enter a positive's
   vocabulary, or the model would learn "part number = bot" from the other class.
4. Metadata: `source="part_number"`, `template_id` = the shape, `seed`. The fold
   report counts them under their own key next to the spans, and the class ratio
   of section 4 counts them as negatives (positives render at
   `(n + 100) // n_training_templates` each so the training side stays 1:1).

**Rulings on the M7 plan (`fable`, 2026-11-07, PQ-7 to PQ-9; recorded in 12.3):**

- **Quota (PQ-7a).** The 100 count as negatives wherever the negative count sets
  the positive quota: primary main run and LOFO `(n + 100) // n_training_templates`,
  comparison LOFO `(384 + 100) // n_remaining`, primary final model
  `(n + 100) // 24`. The comparison arm's fixed 20 per template (8.2, main run and
  final model) stays. In the comparison arm the 100 are about 21% of the negatives;
  accepted: both arms get the same fix so `arm_comparison` keeps isolating spans
  against real queries, and the diagnostic is reported per arm.
- **Seed without a repeat (PQ-7b).** The final models use
  `SEED + PART_NUMBER_SEED_OFFSET`, as their spans use `SEED`.
- **Inner-split groups (PQ-8a).** One group per shape, `("part_number", shape)`,
  like a template: rows of one shape are near-duplicates and belong together. They
  never enter the out-of-fold threshold scores (step 3).
- **Rule 4 (PQ-8b).** A training positive equal to a part-number negative is
  dropped and counted, as for any training negative. The test side is unchanged.
- **"A span" (PQ-8c).** Step 2 drops against the arm's own training negatives (the
  spans in the primary arm, the training-side real queries in the comparison arm),
  all 480 real queries, the P strings, the sanity queries and earlier candidates.
- **Per fold where the vocabulary is per fold (PQ-8d).** The comparison arm and the
  catalogue-gap pass build them per fold with the repeat's seed, as the gap pass
  builds its spans; "per repeat" in step 3 describes the primary main run.
- **`generate` (PQ-9).** Unchanged: it inspects positives, spans and the
  diagnostic set; the hard negatives are reported by `evaluate`'s fold reports.

**Rationale:** the M6 run's diagnostic (8.6 item 4) flagged 85% of the 100
part-number strings with the chosen model and both `SKU...` sanity rows, although
part numbers are legitimate (2.2); WANDS has no bare part-number query (13.15), so
nothing in the negatives teaches the boundary, and family A teaches the opposite.
12.2 named this proportionate fix for a large share; 100 rows are 2% of the
negatives, with their own tag so folds and reports tell them apart. The diagnostic
set stays disjoint from them (step 2) and is re-reported after the change, so it
measures the fix rather than memorisation.

## 4. Synthetic positives

**Decision:** rule/template-based generators in Python, one module per family
(section 2), driven by a seeded `random.Random`. **No LLM-generated positives**
(user decision, Q2): paid API, non-reproducible, and templates are sufficient for
families A-E.

**Decision (revised 2026-11-05 with Q13):** generate **as many training positives
as training negatives** (5,000 by default, 3.2), split equally over the training
templates of A-D: `n_neg / n_training_templates` renders per template (5,000 / 24
~ 209 in a full fit, 5,000 / ~19 in a fold where some templates are on the test
side). The 20-per-template figure of rule 6 remains the **nominal group size used
for fold assignment** (8.1), which is what probes 13.10 and 13.14 validated; the
renders are then made per fold for the templates on each side. Family E stays
evaluation-only, 120 per fold, `family="E"`. **Rationale:** a balanced training
side needs no class weights or re-balancing (the test side is not balanced, about
five synthetic positives per real query; 8.5 says how precision and macro F1 are
read); the probe shows the gain comes from the quantity of
negatives, and 384 positives against 20,000 negatives already reach most of it
(P2LP arm, 13.19), so matching counts costs nothing and avoids re-weighting.

Rendering more than 20 per template uses the registry's existing seeded
`generate()` repeatedly with `seed + i` and keeps the first `per_template` outputs
of each wanted template (a `render(vocabulary, seed, templates, per_template)`
helper in `bqc/synth/registry.py`; the templates themselves do not change). Exact
duplicates within a template are dropped and the shortfall reported per template;
rule 5's floor of 50 distinct strings is far below 209, so the report also lists
each template's distinct count at the required size (a template under 50% distinct
is a finding, not a failure). A shortfall is **not redistributed** to other
templates or families: that would change the family mix to hide a template's lack
of variety. So the training side is balanced *up to reported shortfalls*. At M2
sign-off it was 4,804 positives to 5,000 spans because `B.script` yielded 60
distinct strings and `B.traversal` 168 of 208; after M2.T13 widened both (13.20)
it is **4,992 to 5,000**, the 8 missing being `5,000 // 24` rounding, with no
template short. `B.script`'s capacity is exactly 396 distinct strings, so at the
20,000 sensitivity run (833 per template) it is a reported finding again (47.5%
distinct), accepted there. `B.traversal` outputs are long (median 53 characters
against the spans' 19); bounded and measured by the trivial baseline (6.3). The
CLI's "positives:" count includes the 120 E strings. Training and test positives equal, after
`normalise()`, to a training negative, a real query or a P string are dropped by
the audit (rule 4), counted, never shown; one equal to a sanity query fails the
generation (rule 4). Test-side templates render 100 each (8.3).

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
   (section 2). Product words are sampled only from the **training negatives of
   the arm** (the product spans in the primary arm, the training-side real queries
   in the comparison arm; 8.3) so test-side vocabulary never leaks into training
   positives, and they are sampled **from token occurrences, with repeats**, so
   common words appear in hard positives as often as in the negatives (8.3, 13.17).
4. Exact-match audit and near-duplicate report (clarified 2026-11-05, PQ-1). The
   exact-match check compares every generated positive, after `normalise()`
   (section 5), with four lists, on every generation without exception
   (`generate`, the per-fold ones (8.3), the final-model fit (8.7), the clustering
   experiment (9)). The outcome depends on the list:
   - equal to a **sanity query**: the generation **fails** (`check_exact` raises);
     the template changes, never the sanity file. This is a template defect: the
     sanity set must stay outside the generators (8.6.1).
   - equal to a **training negative, a real query or a P string**: the positive is
     **dropped and counted** in the run JSON, never shown. These collisions are
     data-dependent (a hard positive borrowing catalogue words can coincide with a
     span; a scratch run saw 0 of 5,016), not a template defect, so they are not a
     failure.
   Product spans (3.2) follow the second rule for all four lists, sanity included:
   a span equal to a sanity query is dropped and counted, because spans are cut from
   real catalogue text, not written by a template.
   The character-3-gram Jaccard report runs only in `bqc generate` (section 10):
   pairs above 0.6 against negatives, and, **report-only, pairs above 0.5 against
   sanity queries** (8.6.1); neither threshold fails a run. The example strings in
   section 2 are illustrations, not a list templates may copy.
8. **Template authors do not open `data/sanity.csv`** (ruled 2026-11-05). The
   delegation that writes or corrects a generator may contain **exactly**: the
   family's row in section 2 (and, for the part-number task, 8.6 item 4), sections
   2.1, 2.2 and 4, and the task entry; nothing from the sanity file and nothing
   from 8.6.1, 11, 12, 13 or 14 (those name rows to explain decisions). Those
   author-facing sections contain only invented examples (8.6.1). The author does
   not read the file, and no generator source, docstring, comment or test quotes a
   sanity query or reads the real file. Collisions are the
   audit's job (rule 4), not the author's. Rationale in 8.6.1: the earlier practice
   of asking authors to steer around sanity queries made them read the file, and
   phrases traceable only to it ended up in templates.
9. **Spam contact details are invented, not reserved** (ruled 2026-11-05,
   overruling an earlier brief). Family D builds domains from its own label parts
   plus a list of common spam TLDs (`.biz`, `.info`, `.xyz`, `.top`, `.club`,
   `.online`, `.shop`, `.site`, `.com`, `.net`), and phone numbers from random area
   codes and random 7-digit shapes, never a bare digit run (2.2). **Banned:** real
   brand names, real people, real company or product names, and the reserved
   `.example`/`.test`/`.invalid` and `555-01xx` forms as the *only* shapes (a scratch
   model learned `exam`, `xamp`, `.exa` and `555-` as the spam signal, stronger than
   `http`). An invented domain or number may coincide with a real one; that is
   accepted because the strings are generated at run time, never fetched or
   dialled, `results/` is gitignored, and the source holds only TLDs and label
   parts.
5. Template-level diversity: each template must be able to produce at least 50
   distinct strings; the generator test asserts this.
6. **Templates per family (decided 2026-11-05, OQ-3).** Each family has **exactly 6
   templates**, of which **at least 2 are hard-positive** (rule 3). The family's 120
   is split equally, 20 per template, so every `(family, template_id)` group has 20
   samples. Family E follows the same rule (6 templates, 20 each). **Rationale:**
   groups are the fold unit (8.1). Probe 13.10 shows that with 3 templates per
   family (groups of 40) a family is absent from 7 of 75 inner training folds and
   test-fold class share swings 0.46-0.56 (outer) and 0.34-0.51 (inner); with 5-8
   templates no family is ever absent and shares stay within 0.44-0.56. Six is the
   smallest count in that safe range that leaves room for two hard-positive
   templates plus four pattern templates per family. The runner still asserts family
   presence (11.16) rather than relying on this.
7. Every generated string is at most 512 code points after `normalise()` (11.2), so
   no positive is ever truncated; the generator test asserts this.

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
- `(1, 4)` with `min_df=2` gives ~5.2k features on the 960 strings of the
  comparison arm (13.3) and **24,056-29,728 features per fold in the primary arm**
  at 5,000 spans plus 5,000 positives (13.21); `(2, 5)` gives
  ~6.4k with no obvious benefit at this size. `analyzer="char"` rather than
  `"char_wb"` so that cross-token patterns like `' or` are kept.
- `min_df=2` removes n-grams that occur in one string only; these can only memorise.

**Decision (`normalise()`, resolved 2026-11-05, OQ-1; order revised 2026-11-05
after M1.T3 validation, see 13.16):** preprocessing before vectorisation is exactly
these six steps, in this order, in one function with one home
(`bqc/preprocess.py`):

1. Remove every character whose Unicode category is `Cc` (control) or `Cf` (format,
   which includes zero-width characters) **unless `str.isspace()` is true for it**.
   So `\t` and `\n` survive this step and `a\tb` becomes `a b` in step 5, while
   U+200B and U+FEFF are deleted. Removal before the first NFKC is the specified
   order; with step 4 present, no known input distinguishes it from removing after
   the first NFKC or collapsing earlier (`opus` searched every code point in six
   embeddings and 500,000 random strings, 2026-11-05), so the behaviour tests
   cannot pin this position and validation checks it by reading the code. The
   behaviour that matters (`e` + U+200B + U+0301 -> `é`, stable on a second call)
   is pinned by test (13.16).
2. Unicode NFKC normalisation (`unicodedata.normalize("NFKC", s)`).
3. Lowercase with `str.lower()` (the same operation `TfidfVectorizer(lowercase=True)`
   applies, so the two agree).
4. NFKC again. Lowercasing can produce a sequence that composes (`Α` + U+0342 has no
   precomposed uppercase form; `α` + U+0342 composes to U+1FB6), so without this
   step a second call would change the string (13.16). It is a second call to the
   same function on an already-normalised string: microseconds.
5. Collapse whitespace: `" ".join(s.split())`. This also strips both ends.
6. Truncate to the first **512 code points**, then `rstrip()` so a cut inside a
   whitespace run leaves no trailing space.

Truncation is **silent**: `normalise()` is a pure `str -> str` function and nothing
in the pipeline reports whether it truncated (11.2). Truncation comes last because
NFKC can lengthen a string (`ﬁ` x10 becomes 20 code points) and the earlier steps
are linear and cheap: a 1 MB input normalises in ~70 ms (13.11); the cost
truncation guards against is character n-gram extraction, which only ever sees 512
code points. A prefix of an NFKC string is NFKC (nothing new can compose in a
prefix, and canonical order is preserved), so truncating after the last NFKC does
not reopen the problem.

**Idempotence is a tested property, not a proof.** `normalise(normalise(s)) ==
normalise(s)` held for every class probed in 13.16 (every single code point, every
code point followed by a combining mark, base + format character + mark for every
`Cf` character, 200,000 random mixed strings, strings straddling the truncation
boundary) and failed for none. No exhaustive proof over all strings exists, so the
rule for where it runs (settled 2026-11-05) is: **loaders return raw text; the
`Scorer` adapters normalise every text they receive, in both `fit` and `score`.**
The WANDS and sanity loaders do not normalise (the raw query is the record, and
the FP list in 8.5 shows it as typed); comparisons that need normal form (the
sanity-vs-WANDS duplicate check, the generator exact-match audit, the fold
vocabulary) call `normalise()` on both sides at the point of comparison; the
generator registry emits strings for which `normalise(s) == s` already holds and
asserts it. So pipeline strings reach an adapter once, raw, and are normalised
there, exactly like ad-hoc input; the adapter cannot and need not tell the two
apart. Idempotence is the safety net that makes any extra call harmless. Applied
identically to negatives, positives and the sanity set. An empty string after
preprocessing is a valid input (11.1).

## 6. Models

**Decision:** scikit-learn for everything (vectoriser, models, CV, metrics, HDBSCAN).
No PyTorch/TensorFlow.

**Rationale (standard-library-first):** the standard library cannot reasonably do
TF-IDF, stratified grouped CV, ROC-AUC and an MLP; a hand-rolled numpy MLP would be
code we must test and maintain for no gain. scikit-learn is required for the
vectoriser, folds and metrics anyway, and `MLPClassifier` adds nothing to the
dependency set. PyTorch is a ~2 GB dependency for a 2-layer network on 10,000
samples (960 in the comparison arm).
Dependencies: `numpy` and `scikit-learn>=1.3` (HDBSCAN is in sklearn from 1.3).
`scipy` is transitive. `imbalanced-learn` was listed for the SMOTE experiment
until the user dropped SMOTE (section 7, 12.1 item 17); it leaves
`pyproject.toml` and the report's library list.

### 6.1 Neural net (the user's model)

`MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam",
alpha=1e-3, early_stopping=True, validation_fraction=0.2, max_iter=500,
random_state=SEED)`. Input is the sparse TF-IDF matrix (sklearn accepts it).
One hidden layer; the data cannot support more. Early stopping uses a split taken
from the *training* data it is given, so it never sees the test fold (13.6).

### 6.2 Baseline (must be reported next to the MLP)

`LogisticRegression(C=1.0, max_iter=1000)` on identical features. Hypothesis: at
10,000 samples it will match the MLP (the probe saw 83.4% vs 84.1% recall at
20,000, within noise, and the MLP *worse* at 960, 13.19); if it does, the
recommendation is to ship the
simpler model. The MLP is kept because the user asked for it and the comparison is
the point.

### 6.3 Trivial baseline (artefact detector)

`LogisticRegression` on three hand features: character length, word count,
non-alphanumeric ratio. If this gets close to the MLP, the synthetic set is too easy
and the generators need rework. It is reported, never selected.

### 6.4 Model port

Stages depend on a `Scorer` protocol (`fit(texts, y)`, `score(texts) -> ndarray of
P(illegitimate)`) that wraps `vectoriser -> estimator`. The four models
of this section are four adapters; the evaluation code never imports sklearn
estimators directly.

### 6.5 Combined-feature logistic regression, `lr_hand` (fable, user-approved 2026-11-07)

**Decision:** a fourth adapter: `LogisticRegression(C=1.0, max_iter=1000)` (6.2)
on the section 5 TF-IDF matrix **concatenated with the three hand features of 6.3**
(character length, word count, non-alphanumeric ratio, computed by the one
`_hand_features` home), the three columns standardised (`StandardScaler`, mean and
variance from the training rows only), joined by a sparse horizontal stack. Model
name `lr_hand`; in `--models` by default; it runs everywhere `lr` runs, LOFO and
the catalogue gap included.

**Rationale (M6.T1, 2026-11-06):** leave-one-family-out recall of `lr` falls to
34% on B and 30% on D, and the trivial baseline (6.3) beats both `lr` and the MLP on
those held-out families (49% and 69%) while trailing them by 15 points on the
headline. So the headline is not inflated by surface artefacts, but char n-grams
do not transfer to an unseen family and length and punctuation do. Reworking the
B and D generators was considered and rejected: it would lower the yardstick (the
long, punctuation-heavy strings are what scripts, URLs and contact details look
like) without changing what the classifier can see. Adding the three features to
the chosen model is the smallest change that targets the finding, and the same
harness measures it: `model_comparison_hand` (section 7, `lr_hand` minus `lr`)
decides whether it replaces `lr` as the recommendation; the LOFO rows show whether
the transfer improves. If the gain is not `justified`, `lr` stays and this
subsection is the record of the attempt.

**Probe before implementing (section 13):** that `scipy.sparse.hstack` of the
TF-IDF matrix with the standardised dense columns feeds `LogisticRegression`
without densifying, that `StandardScaler` on three columns is fitted on the
training rows only inside the adapter, and the fit time at 5,000 / 5,000.

## 7. SMOTE (dropped by the user, 2026-11-05)

**Decision:** SMOTE is **not part of the design** in either arm. The user dropped it
on 2026-11-05 (12.1 item 17): the training data now comes from product spans
(3.2), so there is enough of it, and SMOTE is not needed. This supersedes Q3 (SMOTE
as a measured experiment, default off) and the plan-review rulings OQ-6 (sizing)
and OQ-7 (`--smote`, `smote_comparison`, `smote_justified`). There is no `--smote`
flag, no SMOTE arm, no comparison section, no `imbalanced-learn` dependency, and
the `Scorer` adapters are `vectoriser -> estimator` (6.4).

What the earlier analysis established, kept for the record: on a balanced set
`SMOTE()` with defaults generates nothing (13.4); forced upsampling produces rows
off the unit sphere that correspond to no string, a within-class mixup rather
than data; and `imblearn.pipeline.Pipeline` skips the sampler at predict time
(13.12). None of that is needed now. The section number is kept so that
references to 7 and 7.1 elsewhere resolve to this note; the paired-comparison
shape once defined in 7.1 is now stated in 8.2 (`arm_comparison`) and 12.1 item
13 (`model_comparison`), with one verdict rule (settled 2026-11-06, PQ-6):

- **Direction.** Every paired difference is *first-named minus second-named*:
  `arm_comparison` is primary arm minus comparison arm; `model_comparison` is
  **MLP minus logistic regression** (the user's model minus the baseline). "Wins"
  counts the folds where the first-named is better.
- **`justified` (per comparison, one boolean)** is one-sided: the first-named is
  better. True only if the mean paired difference in **headline recall is
  positive and greater than** twice its standard error (`sd / sqrt(15)`, `sd` the
  sample standard deviation of the 15 per-fold differences, `ddof=1`), **and** the
  FPR gate holds. A negative mean, however large, is never `justified`; a `worse`
  boolean (mean negative and beyond twice its standard error) is reported for
  symmetry. ROC-AUC gets the same two booleans, informational only.
- **FPR gate.** Achieved FPR is not a gain metric: both arms target the same FPR,
  so its paired difference should be noise. The gate is: the mean paired
  difference in achieved FPR is **not positive beyond twice its standard error**
  (a one-false-positive wobble does not flip the verdict; a systematic rise does).
  Its mean, sd and the strict sign are reported next to the gate's boolean.
- With `justified` false and `worse` false the reading is "within fold-to-fold
  noise"; for `model_comparison` that means the 6.2 recommendation is the
  simpler model.
- **`model_comparison_hand` (added 2026-11-07, 6.5):** the same shape and rule,
  first-named `lr_hand`, second-named `lr`, under each arm, present when both ran.
  `justified` here means `lr_hand` replaces `lr` as the recommended model (the
  user decides, as for 12.1 item 18); `worse` or within noise keeps `lr`.

A more meaningful augmentation for text (string-level perturbation of real negatives)
is in section 14.

## 8. Evaluation protocol

### 8.1 Folds

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)`, repeated with 3
different seeds (15 outer fits per model; seconds of compute). Groups: each WANDS
query is its own group (all unique, 13.1); each synthetic sample's group is
`(family, template_id)` so that near-duplicates from one template never straddle a
fold. With groups of 20 (section 4, rule 6) stratification keeps the test-fold
positive share within about 0.45-0.51 rather than exactly 0.50 (probe 13.10). The
splitter stratifies on class only, so a family's six groups spread unevenly over the
five test folds: with the real seeds a training family has **zero test samples in
12 of the 15 outer folds** as implemented (8 of 15 in the probe's own group
encoding, 13.14), while no training fold ever lacks a family. That is
expected and harmless: every group is in exactly one test fold per repeat, so each
family is fully scored once per repeat (8.5). The runner therefore records each
fold's class and per-family counts in the run JSON and asserts, **on the training
side only**, that both classes and all four training families are present (outer
and inner, 11.16). On the test side it asserts both classes only.

**Under Q13 the fold assignment is unchanged:** the units are still the 480 real
queries (singleton groups) and the 24 template groups at their nominal size of 20,
so 13.10 and 13.14 still describe the folds. Product spans are not fold units: they
are training-only material, rebuilt per repeat (3.2), never on the test side, so
spans from one product never straddle a fold in any sense that matters.

### 8.2 Fitting and threshold selection: cross-fitting the real queries

**Primary arm (`--negatives product`, default; Q13).** Per outer fold `(r, k)`:

1. **Training negatives:** `n` product spans built with seed `SEED + r` (3.2). They
   are the same for every fold of a repeat; what changes per fold is the real-query
   split and the template split.
2. **Training positives:** rendered for the training-side templates from the
   training negatives' token occurrences (8.3), `n / n_training_templates` each.
3. **Fit** vectoriser and estimator on 1 + 2. No real query is in the fit, ever.
4. **Threshold:** score the **384 training-side real queries** with the fitted
   model; `t` is the 8.4 rule applied to those scores (smallest score with FPR <=
   target over the 384; resolution 1/384 ~ 0.26%). This is cross-fitting: the
   queries that set `t` were not trained on, so no inner loop is needed.
5. **Test:** apply `t` to the 96 test-side real queries, the test-side templates'
   positives (100 renders each, from the test-side queries' vocabulary, 8.3) and
   family E.

Across a repeat every real query is a threshold-setter four times and a test item
once, so the 480-query Wilson arithmetic (8.4, 8.5) is unchanged. The probe did
exactly this (13.19).

**Comparison arm (`--negatives real`, the v1 protocol).** Training negatives are
the 384 training-side real queries, positives 20 per training-side template from
their vocabulary, and the threshold comes from an **inner** 5-fold
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + repeat_index)`
over the outer training rows with the 8.1 groups, producing out-of-fold scores for
the training queries; the model is then refitted on the whole training fold. Positives
are not regenerated per inner fold (accepted: the inner loop only chooses `t`, and
any bias shows in the achieved outer FPR). Probe 13.10 shows this inner split keeps
every family present and the class share within 0.44-0.51.

**Arms in one run (PQ-3):** `--negatives` takes a comma list, default `product`;
`--negatives product,real` runs both arms **in the same run**, on the same outer
folds, the same test-side strings and the same seeds, and writes **one JSON** with
each arm under its own key plus an `arm_comparison` section: per-fold paired
differences (primary minus comparison) in headline recall, achieved FPR and
ROC-AUC, each as the list of 15 per-fold values, their mean, their sample standard
deviation (`ddof=1`), the standard error (`sd / sqrt(15)`) and the count of folds
where the primary arm wins; one `justified` and one `worse` boolean for the
comparison and the FPR gate, by the verdict rule of section 7. The MLP-minus-LR
difference lives inside a run under `model_comparison` with the same shape and
direction rule. There is no
command that compares two run files; the full run (M6) is one invocation with both
arms, and reports the primary arm as the headline with the paired differences.

Nothing is fitted on data outside the training side in either arm.

### 8.3 Positives are regenerated per fold

Hard-positive sub-generators borrow product words from the negatives. To avoid
leaking test-fold vocabulary into training positives, the folds are defined on the
480 real queries and on the *(family, template_id)* groups first; then:

- **Training positives** draw vocabulary from the **training negatives** of the arm:
  the product spans' token occurrences in the primary arm (so a hard positive like
  `tufted club chair' union select` borrows catalogue words, never a real query's
  words), or the training-side real queries' tokens in the comparison arm.
- **Test positives** (100 per test-side template) and **family E** draw vocabulary
  from the **test-side real queries' token occurrences**, as before: the test asks
  whether the illegitimate pattern is still caught around words the model saw only
  in the form a shopper uses them. The threshold-setting queries are not used for
  any vocabulary.

Leakage check: no real query token reaches a training positive in the primary arm;
in the comparison arm only training-side query tokens do, as in v1. Implementation
detail for the plan: generation takes a vocabulary argument; the runner passes the
right one. The probe capped the vocabulary list at 10,000 occurrences for speed
(generators rebuild `set(vocabulary)` per render); the plan may keep that cap and
must record it in the run config.

**The vocabulary is the list of token occurrences, not the set of distinct words
(ruled 2026-11-05 after M2.T2 validation, 13.17).** Callers (`bqc generate`, the
fold builder, the final-model fit) pass every whitespace token of every normalised
negative on their side, **with repeats, in file order**, and hard-positive
templates sample from that list uniformly. Sampling from the set of distinct
words would draw one-off words (brands, collection names) in 71% of word slots
against 37% of real token occurrences, and `chair` in 0.1% of slots against 2.2%
of real tokens; the model would then learn "common furniture word = legitimate,
rare or brand word = illegitimate", which is the section 2 artefact and a false
positive on every real brand query. With occurrences, hard positives carry product
words at their real frequencies and the remaining signal is the illegitimate
pattern itself. The benign-token check of 2.2 tests membership and uses
`set(vocabulary)`; a list with repeats changes nothing there. A fold's training
vocabulary still comes from training negatives only.

### 8.4 Operating point: "minimise false positives"

**Decision:** the operating point is a **target FPR of 1%** (confirmed by the user,
Q4), chosen on the threshold queries of 8.2 (cross-fit in the primary arm, inner
out-of-fold in the comparison arm). Headline number: **recall of
illegitimate queries at FPR <= 1%**, pooled over outer test folds, with the
**achieved pooled FPR and its 95% Wilson interval**.

**Rationale:** "minimise FPR" alone has the trivial solution of serving everything.
A target makes it an operating point. Why 1%: with 480 negatives, one false positive
per outer fold of 96 negatives is already 1.04%, and an observed 0/480 has a 95%
Wilson upper bound of 0.79% (13.13; rule of three ~0.6%). Anything below 1% is not demonstrable
with this data; the design says so rather than report a number it cannot support.
The target is a CLI parameter with default `0.01` so that the sensitivity of recall
to the target can be reported; 1% is the number that decisions are made on.

### 8.5 Metrics reported

For each model and arm, per repeat and pooled:

- Headline: recall at FPR <= 1% (8.4), achieved FPR with Wilson 95% CI.
  **Pooling and `n` (resolved 2026-11-05, OQ-5):** each repeat scores every
  negative exactly once, so a repeat's FPR is `k_r / 480` with a Wilson interval at
  `n = 480`. The pooled point estimate is the total over the 3 repeats divided by
  1440, but its interval is computed at **`n = 480` with `k = total / 3`** (the mean
  FP count per repeat; the Wilson formula takes a proportion and needs no integer
  `k`). The 1440 observations are three re-scorings of the same 480 queries, not
  independent, and an interval at `n = 1440` (0 FPs: upper bound 0.27%) would claim
  a precision the data does not have; at `n = 480` it is 0.79% (13.13), which is
  the figure 8.4's argument rests on. No confidence interval is reported on recall:
  the positives are synthetic and 2,400 per repeat in v2.0, and an interval there
  would claim a precision about *our generators*, not about bots.
- Confusion matrix at the chosen thresholds (pooled out-of-fold).
- Macro F1, precision, recall (positive class) at the chosen thresholds.
- ROC-AUC, mean +/- sd over folds and pooled (threshold-free sanity number).
- Per-family recall (which kinds of illegitimate queries are missed), including
  family E, which is scored by every outer-fold model at that fold's threshold but
  never trained on (section 2). E's number is reported in its own row, marked
  evaluation-only, and is never pooled into the headline recall. **Per-family recall
  is a pooled quantity**: caught / total over all test folds of a repeat and over
  the three repeats, never a mean of per-fold values, because a training family is
  absent from the test side of about half the outer folds (8.1, 13.14). For A-D
  the totals are 6 templates x 100 test renders = 600 per family per repeat and
  1,800 overall (each template is on the test side exactly once per repeat; a
  template whose 100 renders contain duplicates or audited-out strings counts what
  survives); for E, which is scored in every outer fold, they are 120 per fold, 600
  per repeat and 1,800 overall. The per-fold breakdown in the JSON records each
  family's test count and caught count, with recall `null` where the count is 0.
  Per-template recall is reported too (24 rows), because the probe showed the
  spread (`A.hard_mash` 59% vs `A.alnum` 100% at 20,000, 13.19) is where the
  information is.
- The list of all false positives (the actual WANDS queries that would be discarded,
  as stored in `query.csv`: the loader does not normalise, section 5, so the three
  double-space queries appear with their double space). With ~5 FPs at 1% this
  list *is* the most useful output of the whole project.
- One-word false positives, reported against the 23 one-word WANDS negatives
  (13.1): the count flagged out of 23 and the words themselves, per model. Short
  real queries (`zen`, `owl`, brand names) are the legitimate inputs closest to
  gibberish, so this subset is where an artefact would show first (added
  2026-11-05).

Macro F1 and ROC-AUC are reported because the user asked for them; they are not used
for any selection. Selection (threshold) uses only FPR on the threshold queries
(8.2). Since v2.0 the test side is unbalanced, about 500 synthetic positives to 96
real queries per fold, so precision and macro F1 depend on that ratio; the run JSON
states the ratio next to them, and they are read as comparisons between models on
the same data, not as absolute numbers.

### 8.6 Anti-inflation: how would we know the model generalises?

No real bot data exists, so the standard CV above measures "can the model separate
WANDS from *these generators*". Five additional checks make that limitation visible:

1. **Leave-one-family-out (LOFO)** (restated for v2.0, PQ-2). For each training
   family F in A-D (E is never trained on, so it has no LOFO row; its
   evaluation-only recall in 8.5 already is the "unseen family" number for E), per
   outer fold of 8.1 and per arm:
   - **Training:** the arm's training negatives unchanged (spans in the primary
     arm; training-side real queries in the comparison arm) and positives from the
     training-side templates of the other three families only, rendered at
     `n / n_remaining_training_templates` each so the training side stays 1:1
     (there is no "3:4": the class ratio is held fixed and only the family mix
     changes). `n` here is the arm's training-negative count: the span count in
     the primary arm, the 384 training-side queries in the comparison arm, whose
     main run renders a fixed 20 per template (8.2) rather than a 1:1 quota.
     Ruled by `fable` on M3.T9, 2026-11-06.
   - **Threshold:** the arm's own rule of 8.2 (cross-fit on the 384 training-side
     queries in the primary arm; the nested inner loop in the comparison arm),
     chosen without F.
   - **Test:** the **same test strings as the main run** for that fold: the 96
     test-side queries, and F's test-side templates at their 100 renders each from
     the test-side queries' vocabulary. Not all six of F's templates: using the
     identical strings makes LOFO recall on F directly comparable, fold by fold,
     with the main run's per-family recall on F (600 per repeat, 8.5).
   - **Report:** recall on F at that threshold, the achieved FPR, and the paired
     difference to the main run's recall on F. This is the closest proxy for "a
     kind of bot we did not anticipate"; expect it to be much lower, and that gap
     is the honest generalisation number.
   Runs in **every arm that runs**, for logistic regression and the trivial
   baseline always; for the MLP in the primary arm under the runtime budget (10;
   the comparison arm's nested LOFO is 360 fits of 384 / 384, seconds each). Off by
   default; `--lofo` adds it to the run JSON as its own section.
2. **Trivial baseline gap** (6.3). If length + punctuation ratio alone reaches the
   MLP's recall, the generators are too easy.
3. **Blind-labelled sanity set** (`data/sanity.csv`, committed). The agents write
   **50 queries**; the **user supplies every label** without being told the
   author's intent (user decision, Q5). Full mechanics in 8.6.1. Scored by the final
   models (8.7), reported separately, never used for fitting or selection.
4. **Part-number diagnostic** (added 2026-11-05 with Q11). Part numbers are
   legitimate (section 2) but WANDS, the only negative source, has one such token
   in 480 queries and no bare part-number query (13.15), so the 1% FPR measured on
   WANDS says nothing about them, and family A teaches the model that letter-digit
   mixtures are illegitimate. The run therefore generates **100 part-number-shaped
   strings** (`bqc/synth/partnumbers.py`, tag `family="P"`, label 0, same seeded
   machinery; shapes: bare prefix+digits like `abc000456`, bare digit runs,
   prefix-digits-suffix like `kx-7731b`, and product words from the full WANDS
   vocabulary plus a part number like `ines shag rug w003096809`), never used in any
   fit, threshold or audit-as-positive, and scores them with each final model (8.7).
   The P strings must pass the same exact-match check as positives (no P string
   equals a WANDS or sanity query after `normalise()`), so a flagged P string is
   never a memorised one; `bqc generate` writes them in its report under their own
   key, separate from the positives (section 10).
   Reported per model: the **fraction flagged at the model's threshold** (the
   expected FPR on such queries), with min/median/max score. The sanity set
   (8.6.1) gives a second, human-labelled look at legitimate shapes.
   Since 3.3 (2026-11-07) the training sides hold 100 part-number hard negatives
   of the same four shapes; the diagnostic strings are kept disjoint from them
   (3.3 step 2), so the fraction flagged now measures whether the negatives teach
   the boundary, not whether a string was memorised. The M6 baseline for the
   chosen model is 85% flagged (12.1 item 20).

5. **Catalogue-gap check** (required, user decision Q13). The primary arm learns
   "legitimate" from the catalogue, so a shopper searching for something the
   catalogue does not describe is the new failure mode (brands such as `kisner`
   were flagged in the probe). Per outer fold: take the `product_class` values of
   the test-side real queries (their `query_class`; the 6 empty-class queries
   contribute none), **remove every product carrying any of those classes** from
   the span pool before building the training negatives, and **remove the
   training-side real queries of those classes** from the threshold set. The
   **2,852 products with an empty `product_class`** (6.6%, 13.18) can never be
   removed; if some of them describe a test fold's category, the simulated gap is
   smaller than intended, so the check errs on the lenient side. Accepted: it is a
   simulation of coverage loss, not a measurement of it; the run JSON records the
   number of products removed per fold and the unclassified count, and the
   unclassified share is small; fit and
   threshold as in 8.2; test on the test-side queries. Report pooled FPR with
   Wilson CI, recall, and the flagged queries. **Failure rule (confirmed by the
   user 2026-11-05, 12.1 item 14, knowing it is lenient):** the check fails only
   when the Wilson **lower** bound of the pooled FPR exceeds the 1% target, i.e.
   when we are confident the gap pushes FPR past target. The probe's result (FPR
   1.67% [0.85, 3.25], recall 86.3%; comparison arm under the same removal 0.90%
   [0.36, 2.22], recall 66.2%) passes the rule and is reported as "point estimate
   above target". What to do on failure (category-aware thresholds, real-query
   negatives for uncovered categories) is a user decision, not a design default.
   **Applies to every arm that runs** (PQ-4): in the comparison arm the removal
   is of the training-side real queries of those classes from both its training
   negatives and its threshold set (the probe's BASEX arm, which is where the 0.90%
   above comes from), so the two arms' gap results are paired like everything
   else. `--catalogue-gap` therefore never is a usage error with either arm. Runs
   for logistic regression and the trivial baseline always; for the MLP under the
   runtime budget (10).

All five are reported, not optimised. A run that looks great on 8.5 and poor on
8.6 is reported as exactly that. If the diagnostic shows part numbers flagged at a
high rate, the remedy (synthetic part-number negatives) would add a third negative
source with no real counterpart, so it is a question for the user (12.2, Q12), not
something this design does on its own.

#### 8.6.1 Sanity set mechanics

*Purpose.* The set exists to test the model on queries written **outside the
generators** and labelled by a person, not by the taxonomy. It measures agreement
with the user's own notion of legitimate, which may differ from section 2; a
disagreement between the user's label and the taxonomy is a finding, not an error.

*File.* `data/sanity.csv`, comma-separated, UTF-8, header `id,query,label`.
`id` is `1..50`. `label` is written **empty** by the author; the user fills in `0`
(legitimate, serve) or `1` (illegitimate, discard). There is **no family, intended
class, or comment column**, and the author's intent is not recorded anywhere, so
the user labels blind. Rows are shuffled with `random.Random(SEED)` before writing
so their order carries no information.

*Authorship, kept independent of the generators.* The set is written in a plan task
that is scheduled and committed **before any generator module exists** (`bqc/synth/`
must be empty or absent at that commit; git history is the evidence), by a
delegation whose brief contains **this section, the `SEED` value (section 10), the
WANDS `query.csv` URL (13.1) and the plan task's own text, and nothing else from
`DESIGN.md` or `PLAN.md`**. The seed is needed for the shuffle and the URL for the
overlap check; neither says anything about the generators. The delegate must not
open `DESIGN.md` or any other plan task. After that commit the file is modified only
by the user adding labels. The loader test asserts that no sanity query equals a
WANDS query after `normalise()`.

*Independence is enforced, not only requested (decided 2026-11-05).* On validating
the committed set, `opus` found two rows verbatim equal to example strings in
section 2. The restricted brief makes that plausibly coincidence (the examples are
the obvious instances of their patterns), but it cannot be proven from the
repository, and the real risk is forward-looking: a generator written later could
emit those strings, and the rows would then be inside the generators. The rows
stay as they are, because after the authoring commit only the user edits the file
and labelling was in progress. Instead, the exact-match audit of section 4 rule 4
also fails any generation whose output equals a sanity query after `normalise()`,
on every generation including the per-fold ones. So whatever the author read, no
string in the sanity set can be produced by the generators that are actually used.
The loader therefore exposes the queries regardless of label state, and the
generator audit depends on it.

*What independence means after M2 (ruled 2026-11-05).* While families A-D were
written, the briefs asked authors to make sanity queries unreachable, which led
them to read the file, and `opus` found phrases in families B and D that trace
only to four sanity rows (character-3-gram Jaccard 0.55-0.81 against them), plus
docstrings and tests quoting rows or reading the real file. Decisions:

- Those phrases are replaced by a delegate that has not read the file (section 4
  rule 8); the quoting tests and comments are removed; from now on no author opens
  the file. The exact-match audit (rule 4) remains the only mechanism, and it
  guarantees exactly one thing: no generator emits a sanity query verbatim.
- The honest claim is therefore: the sanity set is **human-labelled and outside
  the training data**, and **no sanity query is a generator output**; it is *not*
  guaranteed to be outside the generators' *style*, and four rows are known to be
  close to it. The `generate` report lists every sanity row with Jaccard above 0.5
  to any generated positive (rule 4), and M6 reports the sanity results twice:
  over all 50 rows, and over the rows below that threshold, with the excluded
  rows named by id. A row the generators are close to tells us about the
  generators, not about the model.
- Revisit trigger for family A's seven-block minimum (2.2): if the M6 sanity
  results miss the short five-block gibberish row, lower the minimum toward the
  four-block floor.

*Author-facing text.* Sections that generator briefs may contain (section 4 rule
8) are kept free of sanity queries: every example there is invented, and none
says which queries are in the sanity set. Checked by script on 2026-11-05 (every
backtick-quoted string in those sections compared with the file after
`normalise()`): 0 matches. Sections 8.6.1, 11, 12, 13 and 14 may name rows to
explain a decision, because authors do not read them.

*Content guideline for the author (not recorded in the file).* All queries stay in
the furniture and home-goods domain, with no off-domain shopping queries (2.1).
Aim for roughly: 30 legitimate-looking queries that are not in WANDS, covering the
patterns WANDS shows (brands and collection names, dimensions like `72 inch`, typos,
one-word queries, `with`/`for` phrasing, punctuation such as `"` and `/`); about 15
clearly illegitimate ones spanning gibberish, payloads, enumeration and spam
patterns written from scratch without consulting the taxonomy tables' examples; and
about 5 **deliberately ambiguous** ones where a legitimate furniture query contains
tokens that look like a payload or bot pattern (`drop leaf table`, `select comfort
mattress`, `union jack rug`, `null bed frame`, `script wall art`, a query with a
model number like `wf-2049x`). The ambiguous ones are the point: they are the kind
of false positive that costs a sale. The author does not know how the user will
label them, and must not reveal the intended mix in the commit message.

*Evaluation behaviour.* `bqc evaluate` reads the file. The loader strips leading
and trailing whitespace from each `label` cell and nothing else; if **any** cell is
then blank or not exactly `0` or `1`, the sanity section is **skipped** with a
message naming the count of such rows (`sanity set: 50 of 50 rows unlabelled,
skipping`). So ` 1` is a valid `1` and `2`, `yes`, `1.0` are not (resolved
2026-11-05, OQ-2; a person labelling in a text editor will leave stray spaces, and
rejecting those would make the user re-edit for nothing). The run JSON records
`"sanity": {"status": "skipped", "unlabelled": n}` when skipping. Blank
is never treated as a class and a partially labelled file is not scored (a subset
chosen by the labeller could be biased). Everything else in the run proceeds.
The user's labels were committed on 2026-11-05 (`7332c6b`, 50 of 50 rows), so
every real run from then on scores the sanity section; the skip path remains for
fixtures and for any future re-authoring.

*What 50 examples can support, stated plainly.* With about 30 legitimate and 20
illegitimate queries (the final split is whatever the user's labels say), one false
positive is already ~3%, and an observed 0/30 has a 95% Wilson upper bound of about
11%. The sanity set therefore **cannot confirm the 1% FPR target**; it is a
qualitative check. Reported: the confusion matrix (raw counts, no rates with
decimals), and the full list of disagreements between model and user, each with
the query, the user's label, and the model's score. No macro F1, no ROC-AUC, no
confidence intervals on this set.

### 8.7 Final model for inspection

After evaluation, one model per configuration is fitted and saved with `pickle` to
`results/` for the sanity-set scoring, the part-number diagnostic and manual probing.
**Primary arm:** `n` spans with seed `SEED` (3.2), positives from all 24 templates
rendered from the spans' vocabulary, `n / 24` each; the threshold is the 8.4 rule
over **all 480 real queries** scored by that model (none were trained on; no
out-of-fold pass needed). **Comparison arm:** as in v1, all 960 nominal samples, the
threshold from a 5-fold out-of-fold pass (`StratifiedGroupKFold(n_splits=5,
shuffle=True, random_state=SEED)`, 8.1 groups). Family E is excluded from both. Not
a deployment artefact. The pickle holds the scorer and its threshold.

## 9. Label discovery with HDBSCAN (separate experiment)

**Decision:** clustering is a **separate experiment**, implemented in
`bqc/cluster.py` and run with `python -m bqc cluster` (section 10; resolved
2026-11-05, OQ-8a), not a stage of the training pipeline. Its data is unchanged by
Q13: the 480 real queries plus the 480 nominal positives (20 per template, real-query
vocabulary), because the scenario is an unlabelled *query log*, and product spans
are not queries. It shares the
preprocessing and vectoriser factory with the pipeline and nothing else; the
training pipeline always uses known labels.

**Rationale:** the user's scenario is "in production we may not have labels". That
is a question about whether the two classes are *discoverable*, which is answered by
clustering the pooled data with labels hidden and then scoring against the labels.
Mixing it into training would only add a noisier label source to a pipeline that
already has labels.

### 9.1 Pipeline

1. Same preprocessing and char TF-IDF as section 5, fitted on all 960 strings.
2. `TruncatedSVD(n_components=50, random_state=SEED)` then L2-normalise rows, so
   that Euclidean distance approximates cosine distance. This SVD stands by the
   user's decision of 2026-11-06 (12.1 item 16): the rejection of SVD applies to
   the classifier's features, not to this experiment.
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
  draws. Seed draws use `random.Random(SEED + draw_index)` over all 960 rows,
  including noise points (a human does not know which queries are noise).

**Ties and noise in the metrics (resolved 2026-11-05, OQ-8b, 8c):**

- A tie, in either rule (seed 1-1, or an oracle cluster that is exactly half and
  half), resolves to **legitimate**. Fail open is the rule everywhere in this design;
  an oracle tie also means the cluster has purity 0.5, which the purity report shows.
- ARI, AMI, homogeneity and completeness are computed **on non-noise points only**
  (rows with HDBSCAN label `-1` removed from both the true and the predicted label
  arrays), with the noise fraction reported next to them. Treating noise as one
  extra cluster would reward dumping everything into noise. The seed rule's
  accuracy, FPR and recall are computed **over all 960 rows** with noise labelled
  legitimate, because that is the decision the rule would actually make. An
  all-noise result (11.17) therefore has no ARI/AMI (reported as `null` with
  `"non_noise": 0`) and seed-rule FPR 0, recall 0.

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
  __init__.py             SEED constant (one home)
  __main__.py             python -m bqc entry point
  preprocess.py           normalise() (section 5), one home
  wands.py                download + hash check + load (section 3)
  sanity.py               sanity set loader and validity check (8.6.1, 11.20, 11.21)
  synth/                  one module per family + registry + audit; sample metadata
  features.py             vectoriser factory (section 5)
  models.py               Scorer protocol + four adapters (section 6)
  negatives.py            product-name spans (3.2) and part-number hard negatives (3.3)
  evaluate.py             folds, cross-fit / nested threshold selection, metrics (section 8)
  cluster.py              HDBSCAN experiment (section 9)
  report.py               write_report(): config fields, naming, no overwrite
  cli.py                  command implementations
tests/                    unittest; python -m unittest discover -s tests
data/raw/                 gitignored downloads; data/CHECKSUMS committed; data/sanity.csv committed (8.6.1)
results/                  gitignored JSON reports and pickles, one file per run
scratch/                  gitignored probes
```

`python -m bqc generate [--negatives-size 5000]` builds the default training set
for inspection: the product spans with seed `SEED` (3.2), the matching positives of
A-D rendered from the spans' vocabulary (section 4) **and** the 120 of family E
(from the real queries' vocabulary), each sample tagged with its family so E is
never confused with the training set; runs the audit (section 4, rules 2 and 4)
over all of them; and writes one JSON report to `results/` holding the samples
with their metadata, the per-family and per-template length statistics and
distinct counts next to WANDS', the near-duplicate pairs (positives vs negatives
above 0.6; positives vs sanity above 0.5, report-only), the **span audit counts**
(removed as equal to a real query, a sanity query, a P string; expected share of
real queries with a span above Jaccard 0.6: 27% at 20,000 spans in the probe, about
12% at the default 5,000 in `opus`'s scratch build, both desired, 13.19),
the 100 part-number diagnostic strings (8.6 item 4) under their own key, and the
two occurrence-rule numbers of 13.17 computed against the spans' tokens. It exists
so the data can be inspected by eye; fold runs rebuild their own negatives and
positives (8.2, 8.3) and never read its output.

Every JSON file under `results/` (from `generate`, `evaluate` and `cluster`) is
written by one `write_report()` function in `bqc/report.py`: it adds the config
fields (parameters, seed, git commit hash, library versions), names the file by
config hash and timestamp, and refuses to overwrite. It lives in its own module
rather than in `cli.py` because `evaluate.py` and `cluster.py` call it, and they
must not import the command layer.

- **Seeds:** one `SEED = 20261105` constant with one home; every `random_state`,
  `random.Random` and repeat seed derives from it (`SEED + repeat_index`).
  `MLPClassifier` is deterministic given `random_state` (13.6).
- **Results** are JSON: config (all parameters, seed, git commit hash, library
  versions), per-fold and pooled metrics, FP list, per-family recall. One file per
  run; never overwritten (file name includes config hash and timestamp).
- **Commands** (for CLAUDE.md "Commands" once the user fills it):
  `python -m bqc fetch`, `python -m bqc generate [--negatives-size N]`,
  `python -m bqc evaluate [--negatives product[,real]] [--negatives-size 5000]
  [--models lr,mlp,trivial,lr_hand] [--target-fpr 0.01] [--lofo]
  [--catalogue-gap]` (all four models by default since 6.5), `python -m bqc cluster`,
  `python -m unittest discover -s tests`.
- **Runtime budget (estimate, to be measured and recorded in 13 by the first full
  run):** logistic regression and the trivial baseline are seconds per fit at any
  size. The MLP took 987 s per fit at 20,000 / 20,000 with 67k features (13.19);
  **measured at 5,000 / 5,000 by the SVD probe: 9-29 s per fit, median 17 s**
  (13.22; early stopping ends most fits in 19-43 epochs). The full primary run is
  15 outer fits per model (no inner loop, 8.2), plus 60 for LOFO, 15 for the
  catalogue gap and 1 final fit: **91 MLP fits, roughly 25 minutes** on the
  probe's machine, plus the comparison arm's small nested fits (90 at 384 / 384,
  seconds each, plus its LOFO and gap variants). `--models lr` gives a complete run
  in minutes for iteration; the MLP is run last. **"On request" (PQ-5):** `--lofo`
  and `--catalogue-gap` take an optional comma list of models, default all of
  `--models`; naming or omitting `mlp` there is the request. The main fits and
  the final model always cover every model in `--models`. M4.T4 records the
  measured MLP fit time in the run JSON and in 13.9; if it exceeds 5 minutes per
  fit, the M6 invocation is `--lofo lr,trivial --catalogue-gap lr,trivial` and the
  MLP's LOFO and gap rows are reported as not run, with the reason. The rule
  stands as the safeguard; on the measured 17 s it will not trigger, so the M6
  run covers the MLP everywhere by default.
- **Environment:** Python 3.12, `python3 -m venv .venv`, `pyproject.toml` with the
  dependencies from section 6. The user's pyenv 3.12.16 has no `pip` on the shim
  path but `python3 -m venv` works and gives a venv with pip (verified 2026-11-05).
- **Tests do not download.** The WANDS loader is tested against a 10-row fixture
  with the same header and delimiter; the hash check is tested with a wrong hash.
  Model tests use tiny synthetic inputs and assert behaviour (no leakage, row counts,
  determinism), not metric values. The sanity-set loader is tested with a fixture
  containing blank and invalid labels (11.20) and with a row equal to a WANDS
  fixture query (11.21).

## 11. Edge cases and failure modes (starting list)

Each item names the component and the designed behaviour.

1. **Empty or whitespace-only query.** `normalise()` returns `""`; the vectoriser
   produces an all-zero row; the model returns a score (LR: sigmoid of the intercept).
   The CLI/`Scorer` accepts it and returns a score; the evaluation reports it. Not an
   error. (A real search box can submit empty input.)
2. **Very long input** (e.g. 10 kB pasted text or a 1 MB payload). `normalise()`
   truncates to 512 code points as its last step, silently (section 5). Rationale:
   WANDS max is 67; nothing beyond 512 is a query, and char n-gram extraction on
   megabytes is a DoS surface. No data source in v1 can exceed 512 (WANDS, the
   generators by rule 7, and the sanity set by its loader test), so truncation only
   ever applies to ad-hoc input, and there is nothing to report.
3. **Out-of-vocabulary-only query** (e.g. an unseen brand, `minnestrista`). With char
   n-grams some n-grams will be known; with `min_df=2` a fully unknown string yields
   a zero row and gets the same score as the empty string. The test asserts a zero row
   does not raise and the FP list lets us see whether such queries are discarded.
4. **Unicode: non-ASCII letters, combining marks, emoji, zero-width characters,
   full-width ASCII.** NFKC normalisation maps full-width and compatibility forms;
   zero-width and control characters are stripped in `normalise()`; everything else
   is kept as-is. The model has never seen non-ASCII negatives (section 3), so
   non-ASCII will likely score as illegitimate. **Accepted as a known v1
   limitation** (user decision, Q6); it is recorded, not measured, in v1.
5. **Case.** Lowercased in `normalise()`; the model cannot use case (section 4).
6. **Non-English legitimate queries** (`sofá de cuero`). Same as 4: likely false
   positives. Not solvable with WANDS as the only negative source; accepted as a
   known v1 limitation (Q6). No fix is planned: the user rejected mixing in another
   query source (Q7).
7. **Synthetic positive identical to a sanity query:** generation fails (section 4
   rule 4; 8.6.1). **Identical to a training negative, a real query or a P
   string:** dropped and counted. **Near-identical to a WANDS query:** reported
   (Jaccard above 0.6), never a failure.
8. **Synthetic near-duplicates straddling folds.** Prevented by grouping on
   `(family, template_id)` (8.1); the fold test asserts no group appears in both sides.
9. **Leakage of vectoriser / threshold.** The vectoriser is fitted on the training
   side only and the threshold on the threshold queries only (8.2). Test: fit on a
   training fold, assert vocabulary contains no n-gram that exists only in the
   test fold; assert the test-side row count is unchanged by scoring.
10. **Threshold not achievable.** If no threshold reaches the target FPR on the
    threshold queries of 8.2 (the top-scoring negatives tie above every
    candidate), choose the threshold with the smallest FPR and flag
    `target_unmet=true` in the report.
11. **Target FPR below resolution.** If `target_fpr < 1/n_threshold_negatives`
    (384 in the primary arm), warn in
    the report; do not silently round to zero FPs.
12. **MLP fails to converge** (`ConvergenceWarning`). Captured and recorded in the
    run JSON per fold; not an error.
13. **WANDS download fails or changes.** `fetch` fails loudly on network error or
    hash mismatch; nothing partial is written (download to a temp file, then rename).
14. **WANDS file quirks.** Tab delimiter with `.csv` extension; 3 rows contain `"`
    and are standard-CSV quoted (`"fawkes 36"" blue vanity"`), so the `csv` module
    with `delimiter="\t"` and **default quoting** is correct and `QUOTE_NONE` would be
    wrong (13.8). The loader test uses a fixture with such a row.
15. **Empty `query_class`.** Ignored; the loader does not require it. The loader
    still has to carry the value (empty string allowed) once the catalogue-gap check
    exists, because 8.6 item 5 reads it per real query: `build_run_report` takes the
    classes as a sequence parallel to the real queries (M3.T14), and the `evaluate`
    command gets them from the loader. Today `load_queries` drops the column, so
    exposing it is its own small task before M4.T4 (a change to validated M1.T4
    code). Ruled by `fable` on M3.T14, 2026-11-06.
16. **Degenerate fold** (a family or group missing from a training fold).
    `StratifiedGroupKFold` stratifies on class, not family, so with few large
    template groups a family *can* land entirely in one test fold (probe 13.10: it
    happened in 7 of 75 inner folds with 3 templates per family, never with 5 or
    more). Section 4 rule 6 fixes 6 templates per family, and the runner **asserts**
    that both classes and every training family (A-D) are present in every outer
    and inner training fold and raises otherwise. The **test side is not asserted
    for families**: a family missing from a test fold is normal (12 of 15 outer
    folds with the real seeds as implemented, 13.14) and is handled by pooling
    per-family recall (8.5).
    LOFO removes one family on purpose and asserts the other three.
17. **All-noise HDBSCAN result.** Reported as 0 clusters, 100% noise; the seed rule
    then labels everything legitimate (FPR 0, recall 0). Not an error.
18. **Non-determinism.** Two runs with the same config and seed must produce
    identical JSON apart from the timestamp and the `timing` section (the measured
    MLP fit time of section 10, M4.T4); this is a test on a tiny config.
19. **Scores exactly at the threshold.** Decision is `score >= t` is illegitimate;
    stated once in `evaluate.py` and tested.
20. **Sanity set unlabelled, partially labelled, or with a bad label** (`2`, `yes`,
    `1.0`). The loader strips surrounding whitespace from each label cell (so ` 1`
    is a valid `1`) and nothing else, then the sanity section is skipped with a
    message giving the count of rows that are not exactly `0` or `1`; blanks are
    never a class; nothing else in the run is affected (8.6.1, OQ-2).
21. **Sanity query duplicates a WANDS query** after `normalise()`. The loader fails
    with the offending `id`; the set must be outside the training data.
22. **Off-domain shopping query** (`iphone 11 case`) at scoring time. No defined
    behaviour in v1 (2.1); the score is whatever the model gives. Must not appear in
    the sanity set.
24. **`product.csv` quirks.** 90 MB; a description field can exceed the `csv`
    module's default field limit, so the loader raises `csv.field_size_limit`
    before parsing; a row with the wrong column count is an error naming the row
    number (the real file has none, 13.18); the header's fourth column is
    `category hierarchy` with a space and is matched by name, not position.
25. **Span equals a real query, a sanity query or a P string** after
    `normalise()`: dropped and counted by the audit's `drop_exact` (3.2), which
    returns survivors and counts only; the span builder never sees sanity text,
    and no report carries it. **Span pool exhausted** (`n` larger than the
    distinct spans available after removals, or the catalogue-gap removal leaving
    too few products): the builder raises with the count reached; it never pads.
26. **Non-ASCII product names** (918): ASCII-folded in 3.2 step 1; a name that is
    empty after folding is skipped.
27. **Product with fewer words than the drawn `k`:** redraw the product, not the
    length, so the length distribution is preserved.
28. **Real query with empty `query_class`** (6 rows): takes part in folds and
    thresholds as usual; defines no class for the catalogue-gap removal.
29. **Part-number or SKU query** (`sku000123`, `w003096809`, `wf-2049x desk`).
    Legitimate by decision (section 2, Q11) but nearly absent from the negatives
    (13.15), so the model may flag it. Not an error at scoring time; measured by the
    part-number diagnostic (8.6 item 4). Generators must not emit such a shape as a
    whole query (2.2), and the registry test asserts it. (Numbered 23 before v2.0;
    references to 11.23 mean this item.)
30. **Non-finite score** (`NaN`, `+inf`, `-inf`) from a `Scorer` (ruled 2026-11-05).
    **Every score array stops the run**: threshold queries, test queries, test
    positives, family E, part-number strings, sanity queries, LOFO and
    catalogue-gap scores, and the final-model scores alike. One function in
    `bqc/evaluate.py` checks each array as it is produced and raises an error
    naming the repeat, fold, arm, model and which score set held the value;
    `select_threshold`'s own check (M3.T4) stays as a second line. Rationale: a
    `NaN` compared with the threshold is never `>= t`, so an unchecked array would
    read as "not flagged" and flatter the FPR silently; there is no legitimate
    source of a non-finite probability from the adapters of section 6, so it is
    always a defect, and a defect must not become a number in the report.

### 12.1 Resolved on 2026-11-05

The v1 draft asked nine questions. The user's answers, now applied in the body:

1. **Taxonomy (section 2).** Families A-D are the training definition of
   "illegitimate"; E (off-topic natural language) is generated but
   evaluation-only. Accepted as proposed.
2. **LLM-generated positives (section 4).** No. Template generators only.
3. **SMOTE (section 7).** A measured experiment, default off. Yes. **Superseded
   by item 17 (dropped).**
4. **Target FPR (8.4).** 1%. Yes.
5. **Sanity set (8.6.1).** The user's words: "I don't write, you write 50 queries
   in a CSV, and I'll label". The agents write 50 queries in one CSV with an empty
   label column; the user supplies every label, blind. This replaces the draft's
   ~100-query, author-labelled set.
6. **Non-ASCII / non-English legitimate queries (11.4, 11.6).** Accepted as a known
   v1 limitation.
7. **Second real negative source.** Rejected, not deferred. The user's words:
   "Amazon ESCI contains all sorts of queries not just furtinure, home accessories.
   let's try to model a real use case, not not mix in irrelevant queris like
   \"iphone 11 case\", etc." Consequences: the use case is a furniture and
   home-goods shop (section 1); the sanity set's legitimate queries are in-domain
   queries not in WANDS (8.6.1), not queries from other shop domains as the draft
   proposed; ESCI is removed from section 14; off-domain shopping queries get no
   label (2.1, Q10).
8. **Branch.** Commit only to `master`. The local branch was renamed from `main` to
   `master`; CLAUDE.md stays as written. (The remote still has only `main`;
   pushing is out of scope.)
9. **Deployment.** Out of scope. Only the input truncation (11.2) is a
   serving-related decision.
10. **Off-domain shopping queries (2.1).** Answered "ok" on 2026-11-05 to keeping
    them outside both classes in v1: not generated, not in the sanity set,
    behaviour unmeasured. Discarding them would be a product-policy choice, not
    bot detection. (The question as asked: positive family, negative family, or
    neither; recommendation was neither.)
11. **SKUs and part numbers are legitimate.** The user's words, 2026-11-05: "SKUs
    should remain legit, customers might search by part numbers." Prompted by the
    user labelling the sanity rows `sku000123` and `sku000124` as legitimate, which
    contradicted family C's "bare SKU/ID sequences". Applied: removed from family C
    and listed as legitimate (section 2); family C re-specified with six templates
    (2.2); boundary rule between part numbers and family A gibberish, asserted by
    test (2.2); part-number diagnostic added (8.6 item 4); edge case 11.23; WANDS
    measured (13.15).

13. **Product-name spans as training negatives (2026-11-05).** The coordinator
    recommended, on the probe evidence in 13.19, and the user approved ("Dew it"):
    (1) word spans from product names become the training negatives (3.2); (2) the
    v1 setup, real queries as training negatives, stays as a comparison arm (8.2);
    (3) all 480 real queries stay out of every model fit and are used only through
    cross-fitting, to set thresholds and to test (8.2); (4) the catalogue-gap check
    is required (8.6 item 5); (5) product-derived negatives must never exactly
    match a sanity query, enforced by the audit without anyone opening the file
    (3.2 step 3); (6) the user was told 5,000 spans is a possible starting size and
    left the size to `fable`: 5,000, justified in 3.2. This supersedes the v1
    decisions that negatives are real WANDS queries (section 3), that there are
    480 positives (section 4) and the nested inner loop for the primary arm (8.2).
    Not decided by the user and not decided here: whether logistic regression
    replaces the MLP as the recommended model. The MLP stays in; 6.2's hypothesis
    (LR matches it) is tested by the paired difference in headline recall with the
    7.1 rule, and the M6 report carries a recommendation for the user to decide
    on. Q12 (part numbers) remains open: the product data does not fix the bare
    part-number shapes (13.19).
14. **Catalogue-gap failure rule (2026-11-05).** The user confirmed the lenient
    rule of 8.6 item 5, having been told it is lenient: the check fails only when
    the Wilson lower bound of the pooled FPR exceeds the 1% target, so the probe's
    1.67% [0.85, 3.25] passes and is reported as "point estimate above target".
15. **Embedding probe (2026-11-05).** The user approved a read-only probe of a
    1024-dimension embedding model as a possible replacement for the TF-IDF
    features of section 5. **Resolved the same day: embeddings are abandoned
    because of their added latency** (about 100 ms per query against 0.21 ms for
    TF-IDF, as reported to the user), although the probe showed a real accuracy
    gain (13.21). Work continues with TF-IDF; section 5 is unchanged; section 6's
    no-PyTorch decision stands.
16. **SVD (2026-11-05).** The user asked for truncated SVD on the TF-IDF features
    to be tried; a read-only probe ran (13.22). **Resolved the same day: SVD is
    rejected by the user.** It lost accuracy at every dimension tried (recall
    75-80% against 79.7% for plain TF-IDF, AUC down, FPR at 90% recall worse) and
    its scoring latency (1.8-18 ms per query against 0.22 ms) misses the bound
    Q17 set. **Not to be re-proposed for the classifier**: dimensionality
    reduction of the section 5 features before the estimator is closed for this
    project unless the user reopens it. Section 5 is unchanged. The M6
    recommendation between logistic regression and the MLP is unaffected (12.1
    item 13). **Clarified by the user on 2026-11-06: the rejection covers the
    classifier only; the HDBSCAN experiment keeps its `TruncatedSVD(50)` step
    (9.1)**, which is required there (13.5: without it HDBSCAN returns no
    clusters) and has no latency role.
17. **SMOTE dropped (2026-11-05).** The user's decision: with training data from
    product spans there is enough of it, and SMOTE is not needed. Applies to both
    arms, the comparison arm included. Supersedes Q3 and the plan-review rulings
    OQ-6 and OQ-7; removes `--smote`, the SMOTE arm, `smote_comparison`,
    `smote_justified` and the `imbalanced-learn` dependency (sections 6, 7, 8.2,
    8.5, 8.7, 10, 11.9). Probes 13.4 and 13.12 stay as history. The MLP fit count
    of the full primary run falls from about 105 to 91.

Also noted by the user: the `.claude/agents` files were cleaned of their previous
project's references in commit `e2e4421`; nothing in this design depends on them.

18. **Logistic regression is the recommended model (2026-11-07).** The M6.T1 run's
    `model_comparison` (section 7 rule) in the primary arm: recall within noise
    (-0.77 pp, se 0.93), the MLP fails the FPR gate (1.18% against 1%); in the
    comparison arm the MLP is `worse`. `fable` recommended logistic regression under
    6.2 and the user accepted. The MLP stays in the harness as the comparison model;
    the final model for inspection and any deployment reading is the LR one.
    Confirmed in the same run: the product-span arm is `justified` for LR
    (+6.63 pp recall, FPR gate holds), so Q13 stands; the catalogue-gap check
    passes in every arm and model.

19. **LOFO finding: a combined-feature adapter, not a generator rework
    (2026-11-07).** `fable` first recommended reworking the B and D generators at
    the M6 sign-off, then withdrew it on the per-template evidence (6.5) and
    recommended the `lr_hand` adapter instead; the user accepted the adapter.
    Generators are unchanged. The LOFO rows stay the honest generalisation number
    (8.6 item 1).
20. **Q12 resolved: part-number hard negatives (2026-11-07).** The M6 diagnostic
    flagged 85% of the 100 part-number strings with `lr` (76-81% with the MLP,
    2-3% with the trivial baseline) and both `SKU...` sanity rows (ids 39, 43)
    with every LR and MLP run. The user accepted the 12.2 fix: 100 generated hard
    negatives with their own tag, 3.3.

### 12.2 Still open

No question is open. Q12 was resolved on 2026-11-07 (12.1 item 20); its text is
kept below for the record.

12. **Synthetic part-number negatives?** (Resolved, see above.) WANDS has no bare part-number query
    (13.15), so the model cannot learn from the negatives that part numbers are
    legitimate, and the 1% FPR target does not cover them. The design measures the
    problem (8.6 item 4) but does not fix it: product spans (Q13) do not help
    either, since the three bare part-number shapes were flagged 100% in every
    probe arm (13.19), and generated part-number negatives would be a third
    negative source with no real counterpart. *Recommend: wait for the full
    run's diagnostic.* If the final
    models flag a large share of the 100 part-number strings, the proportionate fix
    is a small set of generated hard negatives (real product words plus a part
    number, ~50-100 rows) added to the negatives with their own group tag so folds
    and the FP list can tell them apart; if the share is small, do nothing. Does
    not block any task before M6.
(Q16, embeddings, and Q17, SVD, were both resolved by the user on 2026-11-05:
abandoned and rejected respectively; 12.1 items 15 and 16. Q12 is the only open
question.)

### 12.3 Resolved by `fable` on plan review (2026-11-05, OQ-1 to OQ-8; 2026-11-07, PQ-7 to PQ-9)

M7 plan review (2026-11-07, `PLAN.md` at `59379e4`): PQ-7 (quotas and the
final-model seed), PQ-8 (inner-split groups, rule 4, the drop lists per arm,
per-fold builds) and PQ-9 (`generate` unchanged; section 10 tree wording) are
ruled in 3.3 under "Rulings on the M7 plan". None alters a user decision.


`opus` raised eight questions while planning (`PLAN.md` at commit `5d76c8e`,
"Open questions for `fable`"; later plan revisions removed them). None alters a user
decision in 12.1, changes scope, or adds cost, so they were resolved here and
recorded in the body:

| OQ | Resolved in | One line |
|----|-------------|----------|
| 1 | 5, 11.2, 11.4 | `normalise()` is five ordered steps; `Cc`/`Cf` removal spares whitespace; truncation last, silent. |
| 2 | 8.6.1, 11.20 | Labels are whitespace-stripped, then must be exactly `0`/`1`; ` 1` is valid. |
| 3 | 4 rule 6, 8.1, 11.16 | 6 templates per family, 20 samples each, at least 2 hard-positive; runner asserts family presence. Probed (13.10), so no plan probe task is needed. |
| 4 | 8.2 | Inner folds: same splitter and groups, seeded `SEED + repeat_index`; positives not regenerated per inner fold. |
| 5 | 8.5 | Pooled interval at `n = 480`, `k = total / 3`; never `n = 1440`. |
| 6 | 7.1 | (Superseded: SMOTE dropped, 12.1 item 17.) Was: `sampling_strategy` as a callable doubling each class; probed (13.12). |
| 7 | 7.1 | (Superseded: SMOTE dropped, 12.1 item 17.) Was: `--smote`, `smote_comparison`, `smote_justified`; the verdict rule survives in section 7 and 8.2. |
| 8 | 9, 9.2 | `bqc/cluster.py`; ties resolve to legitimate; ARI/AMI/homogeneity/completeness on non-noise points, seed-rule metrics over all rows. |

Readings `opus` took (same commit, "Readings taken"): all six accepted and written
into the body (LOFO in 8.6; SMOTE arms in 7.1; `generate` in 10; exact-match audit
in 4 rule 4; `bqc/sanity.py` and `SEED` home in 10; no results document: the v1
numbers go into the M6 sign-off record in `PLAN.md`, and `results/` stays
gitignored).

Plan additions `opus` made at commit `91a1058`, ruled on 2026-11-05: accepted a
`cli.py` skeleton in the scaffold task with `sanity.py` left to the sanity-loader
task; accepted one shared report writer, but placed in `bqc/report.py`, not
`cli.py` (section 10); accepted the module-level SMOTE callable (7.1); accepted
the sanity real-file test checking the 512-code-point bound (11.2); accepted the
generator registry depending on `normalise()` (rule 4 and rule 7 need it);
accepted the fold builder running the per-fold exact-match check and returning
class and per-family counts (8.1). The 8.1/11.16 contradiction it found is fixed:
only the training side asserts family presence (13.14).

Rulings from M2.T5 validation, 2026-11-05 (opus notes in `PLAN.md` at `428b969`):
sanity-only phrases in families B and D are replaced by a delegate that has not
read the file, authors no longer open the file, quoting tests and comments go,
and a report-only Jaccard against sanity queries is added (section 4 rules 4 and
8, 8.6.1); spam contact details become invented rather than reserved (rule 9);
one-word false positives are reported (8.5). None alters a user decision.

Rulings on `opus`'s v2.0 plan questions, 2026-11-05 (`PLAN.md` at `530de4f`):
PQ-1, a positive equal to a sanity query fails the generation, one equal to a
negative, a real query or a P string is dropped and counted, spans always dropped
and counted (4 rule 4, 11.7); PQ-2, LOFO restated for both arms with a 1:1
training side and the main run's test strings (8.6 item 1); PQ-3, both arms in one
run and one JSON with `arm_comparison` (8.2, 10); PQ-4, the catalogue-gap check
runs in every arm (8.6 item 5). Nits: "inner validation" wording (8.4, 8.5, 11.10,
11.11), no recall interval (8.5), five checks (8.6), vocabulary definition (2.2,
rule 3), 13.19 deviation list, the near-duplicate share at 5,000 (10), balance
wording (4). The user then confirmed the catalogue-gap failure rule (12.1 item 14)
and approved the embedding probe (12.1 item 15; since resolved: embeddings
abandoned for latency, section 5 unchanged; SVD then probed and rejected by the
user, 12.1 item 16, 13.22).

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

### 13.4 SMOTE on balanced sparse TF-IDF (2026-11-05, imbalanced-learn 0.14.2; history, SMOTE dropped per 12.1 item 17)

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
- (2026-11-06, M5.T1, scikit-learn 1.9.1.) `HDBSCAN(copy=...)` defaults to `"warn"`
  (becoming `True` in 1.10); on dense finite input `True` and `False` give identical
  labels and neither mutates the input, so `bqc/cluster.py` sets `copy=True`. Details
  in its module docstring.

### 13.6 MLPClassifier (2026-11-05, scikit-learn 1.9.1)

- Two fits with `random_state=0` on identical sparse input give bit-identical
  `predict_proba`.
- `early_stopping=True, validation_fraction=0.2` fits fine on sparse input (stopped
  at iteration 23 on the probe data). The validation split is drawn from the data
  passed to `fit`, so inside a CV training fold it does not touch the test fold.
- (2026-11-06, scikit-learn 1.9.1, M4.T2 probe, re-run by `opus`.) With
  `validation_fraction=0.2` the fit needs at least 3 rows of each class: 4 or 5
  balanced rows raise `ValueError` (the validation split would hold fewer rows than
  classes), 6 rows fit. Tiny-config tests that fit the MLP (11.18) need fixtures of
  at least that size. `max_iter=1` emits `ConvergenceWarning` and the fit returns.

### 13.7 StratifiedGroupKFold (2026-11-05, scikit-learn 1.9.1)

480 singleton groups (negatives) + 120 groups of 4 (positives), `n_splits=5,
shuffle=True`: every fold has 192 test rows, exactly 50% positive, and no group on
both sides.

### 13.8 WANDS quoting (2026-11-05, Python 3.12.16, stdlib `csv`)

The three queries containing `"` are stored standard-CSV quoted:
`"fawkes 36"" blue vanity"`, `"48"" sliding single track , barn door for laundry"`,
`"writing desk 48"""`. Default `csv` quoting reads them as `fawkes 36" blue vanity`
etc.; `quoting=QUOTE_NONE` returns the raw quoted form. Use default quoting.

### 13.9 Unverified assumptions

- The 480 WANDS queries contain no bot traffic (13.2); not provable, mitigated by
  reviewing the FP list by eye. (Reviewed 2026-11-06, M6.T1: no false positive of any
  arm or model looks machine-generated; borderline `promo codes or discounts`,
  `drudge report`, `ines shag rug w003096809`. The assumption stands.)
- (Closed 2026-11-05.) The MLP fit at 5,000 / 5,000 was estimated at 1-2 minutes
  from 987 s at 20,000 / 20,000; the SVD probe measured 9-29 s, median 17 s (13.22).
  The first full run re-records it in the run JSON.
  (Measured 2026-11-06, M4.T4, `python -m bqc evaluate` defaults, primary arm only, Python
  3.13.18, scikit-learn 1.9.1, numpy 2.5.4, 4 cores, working tree on top of `880d384`.) One
  `MLPScorer.fit` at 5,000 / 5,000 took 39.7 s (`timing.mlp_final_fit_seconds` in the run JSON: a
  `time.perf_counter()` around that one fit, not the spans or the threshold scores), under the
  5-minute rule and above 13.22's median of 17 s (range 9-29 s). The whole run (15 outer folds x
  3 models, the three final fits, sanity, part-number diagnostic, no `--lofo` or
  `--catalogue-gap`) took 12 min 1 s of wall time (an earlier identical run: 12 min 57 s).
  (M6.T1, 2026-11-06, clean clone at `a20073f`, same software.) The full invocation,
  `evaluate --negatives product,real --lofo --catalogue-gap` with all three models, took
  64 min 58 s of wall time; the final primary-arm MLP fit took 38.4 s. `fetch` 6 s, `generate`
  56 s, `cluster` 7 s. (M7.T7, 2026-11-07, clean clone at `4d19467`, four models and the
  part-number hard negatives.) The same invocation took 79 min 35 s; the final MLP fit 37.1 s;
  `fetch` 3 s, `generate` 67 s, `cluster` 6 s.
- The probe agent's `facts.py` coverage numbers (13.18) were read, not re-run.
- (Closed 2026-11-06.) No WANDS `query_class` contains `|`: checked by `fable` with
  `grep -c '|'` on a `query.csv` whose SHA-256 matches `data/CHECKSUMS`, 0 lines.
  The catalogue-gap check's whole-string comparison (M3.T14) is therefore exact.

These are the unverified assumptions left. Of the three items this section held
in v1.1, the imblearn `Pipeline` behaviour and the Wilson reference values were
probed on 2026-11-05 (13.12, 13.13); the in-house Wilson function is a decision
(standard library `math`, tested against 13.13), not an assumption.

### 13.10 StratifiedGroupKFold with template-sized groups (2026-11-05, scikit-learn 1.9.1)

480 singleton negatives plus 4 families x 120 positives split equally over `T`
templates per family (groups of 120/T), `n_splits=5, shuffle=True`, 3 seeds (15
outer folds); for each outer training fold an inner 5-fold split with the same
splitter and groups (75 inner folds). No split raised.

| T (group size) | outer test positive share | family absent from outer train | inner validation positive share | family absent from inner train |
|---|---|---|---|---|
| 3 (40) | 0.455-0.556 | 0/15 | 0.342-0.513 | **7/75** |
| 5 (24) | 0.500-0.500 | 0/15 | 0.483-0.558 | 0/75 |
| 6 (20) | 0.455-0.510 | 0/15 | 0.438-0.513 | 0/75 |
| 8 (15) | 0.484-0.522 | 0/15 | 0.493-0.542 | 0/75 |

Basis for section 4 rule 6 (6 templates), 8.1, 8.2 and 11.16.

### 13.11 `normalise()` order, idempotence and cost (2026-11-05, Python 3.12.16)

The original five-step order of section 5 (NFKC, remove `Cc`/`Cf`, collapse,
lower, truncate): `a\tb` -> `a b`; `so​fa` -> `sofa`; full-width `ＳＯＦＡ` ->
`sofa`; `café 🛋 é` kept (accents, emoji, combining mark). Applying it twice
equals applying it once for every Unicode code point taken **alone** (all code
points except surrogates, each alone and embedded as `x<c>y`): 0 exceptions. A
1,008,000-character input normalises to 512 code points in 73 ms. NFKC can
lengthen input (`ﬁ` x10 -> 20 code points), which is why truncation is the last
step. **This probe tested single code points only; it did not establish
idempotence on multi-character input, and the five-step order is not idempotent
there (13.16).** The order in section 5 was revised as a result.

### 13.12 imblearn `Pipeline` with SMOTE at predict time (2026-11-05, imbalanced-learn 0.14.2, scikit-learn 1.9.1; history, SMOTE dropped per 12.1 item 17)

`Pipeline([TfidfVectorizer(char), SMOTE(sampling_strategy=<callable doubling each
class>), estimator])` on 300/200 sparse rows:

- A callable `sampling_strategy` is accepted; the estimator's `fit` saw 1000 rows
  (600/400), i.e. the callable ran against the fitted `y`.
- The sampler's `fit_resample` was called once during `fit` and **zero times during
  `predict_proba`**; `predict_proba` on 7 rows returned 7 rows. Same for
  `LogisticRegression` and `MLPClassifier`.
- `SMOTE(sampling_strategy={0: 100, 1: 100})` with class counts 300/200 raises
  `ValueError` ("the number of samples in a class should be greater or equal to the
  original number of samples"), so an absolute count must never be below any class
  count it is used with; basis for the factor in 7.1.

### 13.13 Wilson and Clopper-Pearson reference values (2026-11-05, scipy 1.18.1, `binomtest(k, n).proportion_ci`)

95% intervals, `[low, high]`:

| k/n | Wilson | exact (Clopper-Pearson) |
|---|---|---|
| 0/480 | [0, 0.00794] | [0, 0.00766] |
| 5/480 | [0.00446, 0.02415] | [0.00339, 0.02414] |
| 0/30 | [0, 0.11351] | [0, 0.11570] |
| 1/96 | [0.00184, 0.05667] | [0.00026, 0.05667] |
| 15/1440 | [0.00632, 0.01712] | [0.00584, 0.01712] |
| 5/1440 | [0.00148, 0.00810] | [0.00113, 0.00808] |
| 480/480 | [0.99206, 1] | [0.99234, 1] |

The 0.77% quoted in the v1.1 draft of 8.4 was the exact bound; the design reports
Wilson, so 8.4 now says 0.79%. These values are the test oracle for the in-house
Wilson function (13.9).

### 13.14 Family presence per fold side, 6 x 20 groups (2026-11-05, scikit-learn 1.9.1, numpy 2.5.4)

Reproduction of counts `opus` reported while planning. 480 singleton negatives plus
4 families x 6 templates x 20 samples (the shape of section 4 rule 6), groups
`(family, template_id)`, `StratifiedGroupKFold(5, shuffle=True)`; inner splits use
the same splitter and seed on the outer training rows; LOFO inner splits drop one
family's groups first.

| seeds | outer folds with a family absent from **test** | from outer **train** | inner train | LOFO inner train |
|---|---|---|---|---|
| `SEED`, `SEED+1`, `SEED+2` | **8 / 15** | 0 / 15 | 0 / 75 | 0 / 300 |
| 0..99 | 359 / 500 | 0 / 500 | 2 / 2500 | 16 / 10000 |

Test-side positives per family with the real seeds take values in {0, 20, 40, 60};
e.g. seed `SEED` fold 4 has [0, 0, 60, 20]. `opus`'s own counts (369, 3 and 7 for
the "other seeds" cells) differ slightly, as expected when groups are encoded
differently; the real-seed row matched `opus`'s scratch count exactly. **The
implemented fold builder (M3.T3) gives 12 of 15** with the real seeds, because its
group labels are encoded differently from this probe's; the training-side counts
(0 absent) are unchanged, which is the property that matters. Basis for 8.1
(assert families on the training side only), 8.5 (pooled per-family recall) and
11.16.

### 13.15 Part-number-shaped queries in WANDS (2026-11-05, Python 3.12.16, stdlib `csv`/`re`)

Over the 480 queries: 44 contain a digit. Tokens that mix letters and digits:
`12v`, `18x18`, `5x6`, `70s`, `7qt`, `e12/candelabra`, `w003096809` (7 tokens, 7
queries). Excluding measurement and spec forms (`18x18`, `7qt`, `70s`, `5x6`),
**3 queries contain a code-like token** (`ines shag rug w003096809`,
`e12/candelabra`, `12v dirt bike`), of which **one** (`w003096809`) is a SKU/part
number in the sense of Q11. **Zero** queries are a bare SKU or part number, and
zero are all digits. The negatives therefore contain almost no evidence that
part-number shapes are legitimate; basis for 8.6 item 4, 11.23 and Q12.

### 13.16 `normalise()` idempotence on multi-character input (2026-11-05, Python 3.12.16, Unicode data 15.0.0)

Reproduces the M1.T3 validation finding by `opus` and tests the fix. Orders
compared, each ending with collapse-whitespace and truncate-to-512:

| order | opus cases (3) | base + `Cf` + mark, all 161 `Cf` x 60 marks x 8 bases | single code point | every code point + 5 marks | 200,000 random mixed strings | 20,000 straddling the 512 cut |
|---|---|---|---|---|---|---|
| v1.5: NFKC, remove, (collapse), lower | **fail** | 282 fail | 0 | not run | 13 fail | 1 fail |
| remove, NFKC, lower | pass | **2 fail** (`Α`+`Cf`+U+0342) | 0 | not run | 0 | 0 |
| remove, lower, NFKC | pass | 0 | **629 fail** | 3145 fail | 5467 fail | not run |
| **remove, NFKC, lower, NFKC** (section 5 now) | pass | 0 | 0 | 0 | 0 | not run |

Opus's cases: `e` + U+200B + U+0301 (first pass `0065 0301`, second `00E9` under
v1.5), Hangul jamo split by U+200B, and a double-mark variant. Cause confirmed:
removing a character after NFKC can leave a composable sequence that only the next
call composes. The reordering alone leaves lowercase-induced compositions (`Α` +
U+0342 lowercases to `α` + U+0342, which NFKC composes to U+1FB6), hence the
second NFKC after lowercasing. Lowercasing before the first NFKC is wrong for
compatibility characters that NFKC maps to uppercase letters. NFKC never emits a
`Cc`/`Cf` character from a non-`Cc`/`Cf` code point (all code points checked), so
removing before NFKC cannot leave a format character behind. The random strings
drew from 4,000 random code points, 40 `Cf` characters, 200 marks, whitespace,
uppercase, full-width and `İ`/`ẞ`. WANDS is ASCII and the generators are ASCII by
rule, so no v1 data is affected either way; the change matters for ad-hoc input
and for the honesty of the claim in section 5.

### 13.17 Vocabulary types vs occurrences (2026-11-05, Python 3.12.16, stdlib)

WANDS: 1,623 token occurrences, 834 types, 595 one-off types. One-off words are
**36.7% of occurrences** but **71.3% of types**; `chair` is **2.16% of
occurrences** but one type in 834 (0.12%). `opus` measured the implemented hard
positives at M2.T2 validation: 70.7% of hard-positive word slots were one-off
WANDS words and `chair` filled 0.15% of them, i.e. sampling was uniform over
types. Basis for the occurrence rule in 8.3 and the two report numbers in 10.

### 13.18 WANDS `product.csv` (2026-11-05, Python 3.12.16, stdlib `csv`, checked by `fable`)

Downloaded by the probe agent to an isolated directory; re-inspected by `fable`
with `python -I`. SHA-256
`d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6`, 90,621,131
bytes. Tab-separated; header `product_id, product_name, product_class, category
hierarchy, product_description, product_features, rating_count, average_rating,
review_count`. 42,994 rows, 0 with a wrong column count (`csv.field_size_limit`
raised), 0 empty names, 42,577 distinct names, 1-49 words, median 6.0, mean 6.75;
0 names with upper case; 918 with non-ASCII characters; **2,852 rows (6.6%) have
an empty `product_class`** (found by `opus` at M1.T8 validation, re-counted by
`fable`), which bears on the catalogue-gap check (8.6 item 5). First rows: `solid wood
platform bed`, `all-clad 7 qt . slow cooker`. Token coverage of the 480 real
queries by product names: 94.8% of occurrences, 90.9% of types; by the other real
queries (leave-one-out): 63.3% / 28.7% (probe agent's `facts.py`, read, not re-run).

### 13.19 Product-name spans as training negatives (2026-11-05, probe agent; key numbers verified by `fable` against its outputs)

Stand-in pipeline: char TF-IDF (1,4), `min_df=2`; logistic regression C=1 unless
noted; 5 x 3 folds as 8.1; threshold = 8.4 rule on negatives only; test strings
identical across arms per fold (100 renders per test template, E, 100 P strings);
positives rendered from the arm's training-negative vocabulary (capped at 10,000
occurrences); spans equal to a real query, a sanity query (via `check_exact`) or
a P string removed and counted. Scripts and outputs in the session scratch
directory (`product_probe/`), not in the repository.

Arms at 20,000 negatives unless noted (BASE = v1 protocol, 384 / 384):

| arm | FPR % [Wilson] | recall % at FPR <= 1% | E % | AUC | FPR at 90% recall | one-word FP / 23 | words+part-number flagged % |
|---|---|---|---|---|---|---|---|
| BASE | 0.62 [0.21, 1.82] | 71.0 | 16.3 | .9603 | 10.8 | 1.67 | 31 |
| P1 whole names | 0.56 [0.18, 1.72] | 74.7 | 14.1 | .9686 | 9.4 | 2.33 | 22 |
| **P2 spans** | 0.83 [0.32, 2.12] | **83.4** | 20.6 | **.9801** | **3.9** | 2.33 | 30 |
| P3 spans + typos | 0.56 [0.18, 1.72] | 83.5 | 20.2 | .9808 | 3.8 | 2.00 | 31 |
| P4 spans + taxonomy terms | 0.83 | 84.0 | 20.8 | .9806 | 4.7 | 2.33 | 34 |
| BP real + P2 spans | 1.32 [0.62, 2.79] | 85.6 | 21.8 | .9798 | 3.8 | 4.33 | 37 |

The three bare part-number shapes were flagged 100% in every arm. Paired P2 vs
BASE (intervals from the probe's bootstrap over queries and templates in
`compare.py` and `analyze.py`, not from the 15 folds): FPR +0.21 pp [-0.49, +0.90],
recall +12.4 pp [+6.2, +18.9],
AUC +.0198 [+.0043, +.0399], FPR at 90% recall -6.9 pp [-15.3, -0.8]. No arm has
lower FPR than BASE at the 1% target; P3 and P4 add nothing clear over P2; BP
shows no gain and more false positives (rejected); P1's score distribution does not
transfer (threshold from held-out names gives 18% FPR on real queries; P2: 0.2%;
rejected).

Learning curve, P2, negatives = positives: 500 / 2,000 / 5,000 / 20,000 / 40,000:
FPR 1.11 / 0.76 / 0.42 / 0.83 / 0.90 %; recall 73.6 / 79.2 / 81.2 / 83.4 / 84.9 %;
AUC .9578 / .9672 / .9732 / .9801 / .9827. BASE with 25 / 50 / 75 / 100 % of its
real negatives: recall 59.3 / 62.7 / 68.1 / 71.0, AUC .9454 / .9528 / .9566 /
.9603. The gain comes from the quantity of negatives, not their text: P2 at 500
gives 73.6% and no AUC gain; real negatives with 20,000 positives (class-weighted)
67.4%; P2 20,000 negatives with only 384 positives 84.3%.

MLP vs LR: BASE MLP recall 63.3%, AUC .9231, FPR 0.35% vs LR 71.0 / .9603 / 0.62;
P2 at 20,000 MLP 84.1 / .9819 / 0.90 vs LR 83.4 / .9801 / 0.83 (within noise).
MLP fit time 987 s per fold at 20,000 / 20,000 (67k features); LR 2 s.

Simulated catalogue gap (products of the test fold's classes removed from the
span pool and the threshold set): P2 FPR 1.67% [0.85, 3.25], recall 86.3%; BASE
under the same removal 0.90% [0.36, 2.22], recall 66.2%; brands such as `kisner`
get flagged. Near-duplicates: 27.1% of real queries have a P2 span above Jaccard
0.6 (BASE leave-one-out: 3.5%); those queries are flagged *less* often (FPR 0.51%
vs 0.95%). Spans cannot reproduce 276 of the 457 multi-word real queries. Product
names contain `null` and `order` 0 times, `union` 8, `select` 3, `drop` 40,
`script` 5 (so the sanity-style false positives on those words are not explained
by the catalogue). Sanity gate: P2 removed 1 span equal to a sanity query at
20,000, P3 removed 2 (counts only; text never shown).

Not covered by the probe: real bot traffic, legitimate queries outside the
catalogue beyond the simulated gap, the sanity set (not scored), the trivial
baseline, SMOTE, final-model fits. Deviations from the design: vocabulary capped
at 10,000; P3's typo rate and P2's word counts taken from all 480 queries (a small
leak, replaced in 3.2 by the published constant); 100 renders per test template.
(The threshold on negative scores only is the 8.4 rule itself, not a deviation.)

### 13.20 `bqc generate` on the real data at M2 sign-off (2026-11-05, `fable`, commit `468a367`)

`python -m bqc generate` with the defaults, 38 s: 5,000 spans built (removed:
689 exact duplicates, 17 equal to a real query, 0 equal to a part-number string,
0 equal to a sanity query); 4,804 training positives kept of 5,000 requested, 0
dropped by the audit, the shortfall entirely from `B.script` (60 distinct of 208)
and `B.traversal` (168 of 208); 120 E positives; 100 P strings; 2 positive-span
pairs above Jaccard 0.6; 3 sanity queries above 0.5 to some positive (ids and
scores in the report, no text). `opus` measured the span length mix across seeds:
one-word 3.9-4.5% (target 4.8%), two-word 20.0-21.0% (22.7%), mean 3.46-3.49 words
(3.38); at 20,000 spans one-word 3.3%. Basis for the acceptances in 3.2 and 4.
After M2.T13 (`e60416d`, widened `B.script` and `B.traversal`), the same run by
`fable` gives 4,992 training positives of 5,000 (rounding only), 5,112 with E, no
template short, 0 audit drops, 2 near-duplicate pairs, 3 sanity ids above 0.5.
Sonnet's first attempt (traversal depth 2) tripped the sanity gate; the author
was told only to restore the validated minimum of 3, which is at most one bit of
information about the sanity set (rule 8, recorded).

### 13.21 Embedding probe (2026-11-05, probe agent, read-only; numbers checked by `fable` against its `analysis.txt` and `extras.txt`)

Logistic regression on 5,000 product spans and matched positives (design v2.0
templates), the 8.2 cross-fit protocol, 5 x 3 folds; arms: `T` = section 5 TF-IDF,
`E` = a 1,024-dimension sentence embedding, `TE` = both concatenated, `EM` = the MLP
on `E`. Intervals are bootstraps over queries and templates.

| arm | FPR % [Wilson] | recall % at FPR <= 1% | E % | AUC | FPR at 90% recall | features per fold |
|---|---|---|---|---|---|---|
| T | 0.76 [0.29, 2.02] | 81.7 | 18.4 | .9729 | 6.46 | 24,056-29,728 |
| E | 0.69 [0.25, 1.92] | 88.7 | 60.9 | .9921 | 0.83 | 1,024 |
| TE | 0.83 [0.32, 2.12] | 91.5 | 49.9 | .9942 | 0.62 | 25,080-30,752 |
| EM | 0.56 [0.18, 1.72] | 90.1 | 55.3 | .9881 | 0.76 | 1,024 |

Paired `TE - T`: FPR +0.07 pp [-0.83, +1.04], recall +9.9 pp [+1.7, +19.5], AUC
+.0213 [+.0066, +.0392], FPR at 90% recall -5.8 pp [-15.1, -0.2]; family B +22.1
pp and family E +31.5 pp (both beyond a Bonferroni correction), family C -4.4 pp
(the dictionary-walk template collapses under embeddings: `C.walk` 84% -> 46%,
5% with `E` alone). Catalogue gap: T 1.39% [0.66, 2.89] vs TE 1.46% [0.71, 2.98],
recall 83.4 vs 93.9. One-word false positives 3.0/23 (T) vs 1.7 (TE), within
noise. Part numbers: embeddings flag bare digit runs less (-20 to -48 pp) but
`words + part number` more (+16 to +25 pp). The gain concentrates in the hard
positives (+17 pp) and `D.promo` (+83 pp). Latency as reported to the user: about
0.21 ms per query for TF-IDF against about 100 ms for the embedding model (the
output files carry per-fold feature times, 0.21-0.24 s for T against 0.06-0.93 s
for E on cached vectors, not the per-query figure). **Decision: abandoned by the
user for latency** (12.1 item 15), the accuracy gain notwithstanding.

### 13.22 SVD probe (2026-11-05, probe agent, read-only; numbers checked by `fable` against its `analysis.txt`, `rare.txt`, `extras.txt`, `latency.json`)

Same protocol as 13.21 on the templates at commit `28cabaa` (after M2.T13 and
distinct positives); `T` = section 5 TF-IDF, `S<k>` = `TruncatedSVD(k)` after the
vectoriser, then L2; `-MLP` = the 6.1 network instead of logistic regression.

| arm | FPR % [Wilson] | recall % at FPR <= 1% | AUC | FPR at 90% recall | fit s per fold | p50 scoring ms per query |
|---|---|---|---|---|---|---|
| T | 0.69 [0.25, 1.92] | 79.7 | .9723 | 7.64 | 0.27-0.43 | 0.22 |
| S100 | 1.04 [0.45, 2.42] | 75.2 | .9492 | 13.89 | 0.8-1.7 | 1.8 |
| S300 | 0.97 [0.40, 2.32] | 78.8 | .9611 | 8.19 | 2.2-3.8 | 4.9 |
| S1000 | 0.83 [0.32, 2.12] | 80.2 | .9625 | 9.17 | 9.9-12.5 | 17.9 |
| T-MLP | 0.83 [0.32, 2.12] | 79.2 | .9754 | 6.39 | **9.0-28.7, median 16.8** | 0.23 |
| S300-MLP | 1.04 [0.45, 2.42] | 77.4 | .9523 | 11.04 | 2.7-4.5 | 4.8 |

Explained variance of the SVD: 27% at 100, 44-46% at 300, 66-67% at 1,000. The
SVD keeps the least of the TF-IDF length for exactly the strings the model must
catch (repeats, mashes, part-number shapes, walks: 21-31% at 100), which is why
recall on those falls. Catalogue gap (logistic regression): T 1.81% [0.94, 3.43],
recall 82.2 (passes the 8.6 item 5 rule, lower bound close to the line); S300
1.11% [0.49, 2.51]. One-word false positives: within noise for every SVD arm;
T-MLP 2.3 fewer than T. Part numbers: `words + part number` flagged 5-10 pp less
under SVD, bare shapes unchanged. Pickles: SVD matrix 20 MB (100), 60 MB (300),
199 MB (1,000) against 0.4 MB for the vectoriser. T's features on these
templates: **24,916-30,461 per fold** (13.21 measured 24,056-29,728 on the
earlier templates). **Decision: SVD rejected by the user** (12.1 item 16).

**Finding, cause not isolated:** on these templates T's recall is 79.7% against
81.7% in 13.21, and `B.template` fell from 92% to 60% (`B` overall 74.4 -> 67.3).
Between the two probes the templates changed (M2.T13 widened `B.script` and
`B.traversal`) and positives became distinct per template. Whether the drop is
the wider B positives diluting the family, the distinct rendering, or seed noise
is not known; the full run's per-template table (8.5) and LOFO are where it will
show, and 14 lists the isolation as a follow-up.

### 13.23 Combined-feature matrix probe for `lr_hand` (2026-11-07, `sonnet`, Python 3.13.18, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.4)

Script `scratch/m7t1_probe.py` (not committed), run with the repo `.venv` on the real
data under `data/raw/`. Training side as the default primary arm of `fit_final_model`:
5,000 spans (`build_spans`, seed `SEED`, real queries, part-number strings and sanity
set as in `evaluate`), `n // 24` = 208 positives per A-D template from the spans'
token occurrences, screened as `_render_screened`: 9,992 rows (5,000 negatives, 4,992
positives; no shortfall, nothing dropped). Texts normalised once, as the adapters do.

1. **Sparse stack (criterion 1).**
   - `TfidfVectorizer` output: `csr_matrix`, float64, 9,992 x 29,425, 710,955 non-zeros.
   - `scipy.sparse.hstack([T, csr_matrix(hand_scaled)], format="csr")`: `csr_matrix`,
     float64, 9,992 x 29,428, 740,931 non-zeros (density 0.25%): 8.9 MB against 2.35 GB
     for the dense equivalent (263 times).
   - **Pitfall:** `hstack([T, ndarray])` with the dense 3-column array directly (no `format`)
     returns a `coo_matrix`; with `csr_matrix(...)` and the default format it returns csr.
     Pass `format="csr"` so the type is not left to scipy's choice.
   - `LogisticRegression(C=1.0, max_iter=1000).fit(X, y)` accepts the csr input; the input
     stays `csr_matrix` at every step (also `predict_proba` on a csr batch, which returns an
     ndarray of shape (n, 2), `classes_` [0, 1]). `coef_` is a dense (1, 29,428) ndarray, as
     for plain `lr`. Nothing was densified.
   - **`FeatureUnion([("tfidf", make_vectoriser()), ("hand", make_pipeline(FunctionTransformer(_hand_features), StandardScaler()))])`
     also keeps the output sparse:** `fit_transform` and `transform` both return a
     `csr_matrix` float64, 9,992 x 29,428, equal to the manual stack (max abs difference 0.0),
     and a classifier fitted on it has coefficients within 2e-11 of the manual one. A union
     of only the dense hand branch returns an ndarray, so sparsity comes from the TF-IDF
     branch; with it present both compositions keep the matrix sparse. Either composition
     works; the `FeatureUnion` has the advantage that one `fit_transform` / `transform`
     owns the scaler's state (item 2). Empty and all-unknown texts give an all-zero TF-IDF
     part and the (negative) standardised hand values, with no error.
2. **Scaler on training rows only (criterion 2).** Inside the union, after `fit`,
   `StandardScaler.mean_` = [24.1496, 2.9410, 0.17654] and `scale_` = [13.4292, 1.6588,
   0.14250], equal (`np.allclose`) to the mean and population standard deviation of the
   training rows' hand features, for the manual scaler and the union's. Calling
   `transform` on two other batches (a batch of long strings and short ones, and 7
   training rows) left `mean_` and `scale_` bit-identical. A text's score is the same
   alone, in a batch of 25, and in a batch of 1,905: maximum absolute difference 0.0.
   Smallest `scale_` 0.1425, so no zero-variance column on this data (a constant column
   would get `scale_` 1.0 from sklearn, not a division by zero).
3. **Fit time and convergence (criterion 3).** Wall time of vectoriser + classifier on the
   9,992 rows, three runs each, on this machine (noisy, shared host):

   | model | fit s (3 runs) | `n_iter_` | `ConvergenceWarning` |
   |---|---|---|---|
   | `lr` (TF-IDF only) | 1.15, 1.57, 1.16 | 27 | none |
   | `lr_hand` (`FeatureUnion`) | 0.98, 2.20, 2.90 | 42 | none |

   `lr_hand` takes more lbfgs iterations (42 against 27), so its classifier fit is slower, by
   about 1.9 times at the medians; the spread between runs is as large as the difference.
   Neither is near `max_iter=1000`. Without the scaler (raw length up to hundreds in the
   stack) the same fit took 132 iterations and 6.0 s, still without a warning, which is why
   6.5 standardises. A pickled `lr_hand` is 751,305 bytes against 750,446 for `lr`.
4. **Consequence for M7.T5.** Use `FeatureUnion` or the manual `hstack(..., format="csr")`;
   both are sparse, and the scaler is fitted in `fit` only. The adapter's `_hand_features`
   branch must receive normalised text (the one `normalise` call in the base class).
   The fit-time ratio is a few seconds in absolute terms; it adds nothing material to the
   per-fold cost of 13.22.

### 13.24 Inner split with the part-number groups (2026-11-07, `sonnet`, Python 3.13, scikit-learn 1.9.1)

Comparison arm, real fold structure (`assign_folds` over 480 placeholder queries, the registry's
`group_sizes`, `QUOTA` 20 per template, the 100 part-number rows as 4 groups of 25, inner
`StratifiedGroupKFold(5, shuffle=True, random_state=SEED + repeat)`): over 15 outer folds x 5 inner
folds no inner training fold lacks a family of A-D, with and without the part-number groups
(0 of 75 both ways). With the 6-template stub registry and 3 outer folds a fixture can lose a
family (family A with two training templates), which is a fixture effect, not the real structure.
Script: `scratch/probe_inner.py` (gitignored).

## 14. Later (deferred, not in scope)

Not in this list because the user rejected them (12.1): LLM-generated positives
(Q2) and a second query source such as Amazon ESCI (Q7).

- **Behavioural features** (rate, session, IP reputation): the only way to catch
  scrapers that send well-formed queries. Different data, different project.
- **Catalogue-anchored cluster labelling**: label a cluster legitimate if its members
  are close to product names from WANDS `product.csv` (90 MB, in-domain). Removes
  the need for seed labels in section 9.2.
- **String-level augmentation of negatives** (typo injection, word dropout), the
  text-native form of augmentation; P3 in 13.19 showed no clear gain over plain
  spans, so only if the FP list points at typo-bearing queries. (SMOTE itself is
  dropped, 12.1 item 17, and not to be re-proposed.)
- **Word + char feature union**, and a `char_wb` comparison, if the FP list suggests
  word-level cues are being missed.
- **Partial AUC** (FPR in [0, 0.05]) as a threshold-free number closer to the
  operating region than full ROC-AUC.
- **Calibration** of scores (not needed while the decision is a thresholded rank).
- **Family E (off-topic natural language) as a training family**, if the
  evaluation-only run shows it is separable without new false positives.
- **Sequential enumeration** (a bot walking `sku000123`, `sku000124`, ...): each
  query is a legitimate part number on its own (Q11), so a per-query text model
  cannot see the pattern; it belongs to a rate or session check, with the
  behavioural features above.
- **Rejected or unproven negative constructions (13.19):** whole product names
  (P1, distribution shift), spans with injected typos (P3) and taxonomy terms
  (P4), both without a clear gain over plain spans, and real queries mixed into
  the training spans (BP, more false positives). Revisit P3 only if the FP list
  shows typo-bearing queries being flagged.
- **Per-length quotas for spans** (draw and deduplicate within each word-count
  bucket so the realised mix equals the target), if the one-word false-positive
  count differs between the 5,000 and 20,000 runs (3.2, 13.20).
- **Isolate the 13.22 recall drop** (T 81.7 -> 79.7%, `B.template` 92 -> 60%)
  by re-running the 13.21 protocol with the M2.T13 templates and with distinct
  rendering switched separately, if the full run's per-template table confirms it.
- **Dimensionality reduction of the classifier's TF-IDF features**: closed by the
  user (SVD rejected for the classifier, 12.1 item 16); not to be re-proposed.
  The HDBSCAN experiment's SVD (9.1) is unaffected.
- **More variety in `B.script`** beyond its 396 distinct strings, only if the
  20,000 sensitivity run's per-template recall for it looks off (section 4; done
  for 5,000 by M2.T13).
- **Category-aware thresholds or real-query negatives for uncovered categories**,
  if the catalogue-gap check (8.6 item 5) fails in the full run; the user decides.
- **Serving**: input contract (11.2) is the only serving decision taken; a model
  file format, latency budget and monitoring would come with a deployment decision.
