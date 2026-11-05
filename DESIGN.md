# DESIGN.md: bot-query-classifier

Owner: `fable`. Status: v1.9, 2026-11-05. The user answered Q1-Q11 (12.1);
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

- Negative class (label `0`) = legitimate = the 480 real customer queries in WANDS.
- Positive class (label `1`) = illegitimate = synthetically generated queries.
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
| B | injection / probing payloads | `sofa' or 1=1--`, `<script>alert(1)</script>`, `${{7*7}}`, `../../etc/passwd`, `{"$gt":""}` | security scanners |
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
| - | SKUs, part numbers, model numbers (`sku000123`, `w003096809`, `wf-2049x desk`) | **Legitimate** (user decision 2026-11-05, Q11: "SKUs should remain legit, customers might search by part numbers"). Removed from family C. WANDS itself contains `ines shag rug w003096809` (13.15). No generator may emit a part-number shape as a whole query (2.2). |

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
that look like part numbers. The user's sanity labels draw the line: `xk9qz7vplm`
and `qqqqwwwweeee 8374651` are illegitimate, `wf-2049x desk` and `sku000123` are
legitimate. The distinguishing feature is **block structure**, not the presence of
digits:

- A **part-number shape** is a short alphabetic prefix (1-4 letters), an optional
  `-` or `/`, a run of 3 or more digits, and at most one short alphanumeric suffix;
  or a bare run of digits. Letters and digits form at most three contiguous blocks.
  Regex, applied to a whole token after `normalise()`:
  `^[a-z]{0,4}[-/]?\d{3,}[a-z0-9]{0,2}$`.
- **Gibberish with digits** (family A) interleaves letters and digits in four or
  more blocks (`xk9qz7vplm` is `xk|9|qz|7|vplm`). The other illegitimate form in
  the sanity set, a keyboard mash next to a bare digit run (`qqqqwwwweeee
  8374651`), is **not** a generator form: its digit token is one block and would
  fail the rule below, and a bare digit run next to anything is too close to a
  part number for a template to be trusted with. The sanity set covers that form;
  family A does not generate it (decided 2026-11-05; the rule below holds, the
  description that previously allowed the form was the contradiction).

**Decision (extended 2026-11-05 after M2.T1 validation):** every template output,
in every family, must contain **at least one token that is not benign**. A token
is *benign* if, after `normalise()` and whitespace splitting, it is either

- a **vocabulary word**: a member of `set(vocabulary)`, where the vocabulary
  passed to the generator is the list of whitespace-token occurrences of the fold's
  normalised WANDS queries, with repeats (8.3); membership ignores the repeats; or
- a **number-shaped token**: it contains a digit, matches
  `^(?=.*\d)[a-z0-9]+([-/.x][a-z0-9]+)*"?$`, **and has at most three letter/digit
  blocks** (a block is a maximal run of letters or of digits; separators do not
  count). This covers bare numbers (`72`), part numbers (`w003412`, `sku000123`,
  `wf-2049x`), and the measurement forms WANDS uses (`18x18`, `3/4`, `72.5`,
  `3-3/4`, `36"`, `7qt`, `e12/candelabra`); all 36 digit-bearing WANDS tokens match
  and have at most 3 blocks (verified 2026-11-05). The block limit is what keeps
  `xk9qz7vplm` (5 blocks) and UUID/hash-like tokens non-benign, as the table below
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
| `accent chair asdfgh`, `xk9qz7vplm`, `aaaaaaaa` | passes | non-vocabulary letter tokens; 5-block token |
| `aa`, `ab` (dictionary walk) | passes | not vocabulary words |
| `a`, `x`, `or`, `tv` (dictionary walk) | **fails** | WANDS vocabulary words; the walk template must skip vocabulary tokens |
| `what is the capital of france` | passes | `capital`, `france` outside the vocabulary |

Family A templates that use digits must additionally produce at least four
letter/digit blocks in **every** digit-bearing token (so a token never has a
part-number shape even inside a longer string), and family A's hard-positive
templates combine vocabulary words with gibberish tokens only, never with
digits. Four is the floor, not the target: the implemented A template uses a
minimum of **seven** blocks (M2.T2), which keeps training positives further
from three-block part numbers; the section 2 examples `xk9f2m` and sanity row
`xk9qz7vplm` (five blocks each) are illustrations of the shape, not strings the
template must be able to produce. Revisit the minimum if the M6 run misses that
sanity row. The benign-token predicate, the block-count function and the
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
(section 2), driven by a seeded `random.Random`. **No LLM-generated positives**
(user decision, Q2): paid API, non-reproducible, and templates are sufficient for
families A-E.

**Decision:** generate 480 training positives to match the negatives, with equal
quotas per family (120 each for A-D). Family E is generated separately (120
samples, same machinery, `family="E"`) and is **never part of the 480**: it is
scored, not trained on (section 2, 8.5). **Rationale:** balanced classes keep every
metric interpretable and avoid any need for re-balancing.

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
   queries (8.3) so test-fold vocabulary never leaks into training positives, and
   they are sampled **from token occurrences, with repeats**, so common words
   appear in hard positives as often as in real queries (8.3, 13.17).
4. Near-duplicate audit after generation: character-3-gram Jaccard similarity between
   every positive and every negative; any pair above 0.6 is reported, and the
   generation fails if any positive is identical, after `normalise()` (section 5),
   to a negative **or to a sanity-set query** (8.6.1). The exact-match check runs
   on every generation without exception: `generate`, the per-fold ones (8.3), the
   final-model fit (8.7) and the clustering experiment (9), against both lists; the
   Jaccard report runs only in `bqc generate` (section 10) and covers negatives
   only. The sanity half of the check is what keeps the sanity set outside the
   generators by construction (added 2026-11-05, see 8.6.1): if a template can emit
   a sanity query, the template changes, never the sanity file. The example strings
   in section 2 are illustrations, not a list templates may copy; a template whose
   output included one of them would be caught by this check only if that string
   is in the sanity set, which is the only case that matters.
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
- `(1, 4)` with `min_df=2` gives ~5.2k features on 960 strings (13.3); `(2, 5)` gives
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
training folds, compared against the no-SMOTE run on identical folds and seeds
(confirmed by the user, Q3).

**Rationale and evidence:**

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
- Therefore the design does not promise an improvement; it measures one (7.1).
- Leakage rule: SMOTE must be fitted and applied to the training portion only.
  `imblearn.pipeline.Pipeline` does this: the sampler's `fit_resample` is called
  once during `fit` and zero times during `predict_proba`, for both `LogisticRegression`
  and `MLPClassifier` on sparse input (probe 13.12); the pipeline test asserts that
  the test fold row count is unchanged.

### 7.1 Sizing, flag semantics and verdict (resolved 2026-11-05, OQ-6, OQ-7)

**Sizing.** `sampling_strategy` is a **callable** that doubles each class present in
the `y` passed to `fit`: `{c: 2 * count_c}`. This is a factor, not an absolute
count, so the same adapter works for outer training folds (~384 per class), inner
training folds (~307) and the final fit on all 960 (8.7). An absolute `n` would
have to exceed the largest class count it is ever used with, and `imblearn` raises
`ValueError` when a requested count is below the existing one (13.12). The factor 2
is the only knob and is not tuned; the experiment asks whether SMOTE helps at all.
`k_neighbors` stays at the default 5, which every fold's class counts satisfy.

**Flag semantics.** `python -m bqc evaluate --smote` runs **both arms in the same
run**: the three models without SMOTE, plus the MLP and the LR with SMOTE (the
trivial baseline has no SMOTE arm; its three hand features are not what SMOTE is
being tested on). Both arms use identical folds, seeds and generated positives
because they come from one fold construction. The run JSON holds every arm's
metrics under its own configuration key, plus a `smote_comparison` section with,
for the MLP and the LR separately: the per-fold paired differences in headline
recall (SMOTE minus no-SMOTE, 15 values), their mean and standard deviation, the
number of folds where SMOTE wins, and the paired difference in achieved FPR.
Without `--smote` there is no SMOTE arm and no comparison section.

**Verdict rule.** The report computes a boolean `smote_justified` per model: true
only if the mean paired difference in headline recall is positive **and** exceeds
twice its standard error (`sd / sqrt(15)`, where `sd` is the **sample standard
deviation of the 15 paired differences, `ddof=1`**, i.e. `statistics.stdev`),
**and** the mean paired difference in achieved FPR is not positive. The callable
`sampling_strategy` is a **module-level function** (not a lambda or closure) so a
fitted SMOTE scorer pickles for 8.7. Otherwise the difference is "within fold-to-fold
noise", SMOTE is not justified, and the default stays off. The rule is a crude
paired t at about 2 sigma; it is stated so the conclusion is reproducible, not
because 15 correlated folds support a formal test. The numbers behind it are all
in the JSON for a reader who wants to judge differently.

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
five test folds: with the real seeds a training family has **zero test samples in 8
of the 15 outer folds**, while no training fold ever lacks a family (13.14). That is
expected and harmless: every group is in exactly one test fold per repeat, so each
family is fully scored once per repeat (8.5). The runner therefore records each
fold's class and per-family counts in the run JSON and asserts, **on the training
side only**, that both classes and all four training families are present (outer
and inner, 11.16). On the test side it asserts both classes only.

### 8.2 Nested fitting: everything inside the training fold

Within each outer training fold, in order: fit preprocessing/vectoriser, (optionally)
SMOTE, estimator. Nothing is fitted on data outside the training fold, including
threshold selection:

- Inner 5-fold CV on the training fold produces out-of-fold scores for every training
  sample. The threshold `t` is the smallest score such that inner FPR <= target FPR
  (8.4). With ~384 training negatives the inner FPR resolution is 1/384 ~ 0.26%.
- The model is then refitted on the whole training fold and `t` is applied to the
  outer test fold.

**Inner splitter (resolved 2026-11-05, OQ-4):** the inner folds are
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + repeat_index)`
over the outer training rows, with the **same groups** as 8.1. Positives are **not
regenerated per inner fold**: the outer training fold's positives (generated from the
outer training vocabulary, 8.3) are split as they are. So an inner-training hard
positive may contain a product word that also occurs in an inner-validation
negative. This is accepted because the inner loop only chooses a threshold, and
any bias it introduces shows up in the *achieved* outer FPR, which is what 8.5
reports; regenerating per inner fold would add a second generation layer for a
threshold whose quality is measured anyway. Probe 13.10 shows the inner split with
groups of 20 keeps every family present and the class share within 0.44-0.51.

### 8.3 Positives are regenerated per fold

Hard-positive sub-generators sample product vocabulary from WANDS queries. To avoid
leaking test-fold vocabulary into training positives, the folds are defined on
negatives and on the *(family, template_id)* groups first; the hard-positive
templates then draw vocabulary only from training-fold negatives when producing
training positives, and from test-fold negatives when producing test positives.
Implementation detail for the plan: generation takes a vocabulary argument; the
runner passes the right one.

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
Q4), chosen on inner validation data (8.2). Headline number: **recall of
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

For each model x SMOTE setting, per repeat and pooled:

- Headline: recall at FPR <= 1% (8.4), achieved FPR with Wilson 95% CI.
  **Pooling and `n` (resolved 2026-11-05, OQ-5):** each repeat scores every
  negative exactly once, so a repeat's FPR is `k_r / 480` with a Wilson interval at
  `n = 480`. The pooled point estimate is the total over the 3 repeats divided by
  1440, but its interval is computed at **`n = 480` with `k = total / 3`** (the mean
  FP count per repeat; the Wilson formula takes a proportion and needs no integer
  `k`). The 1440 observations are three re-scorings of the same 480 queries, not
  independent, and an interval at `n = 1440` (0 FPs: upper bound 0.27%) would claim
  a precision the data does not have; at `n = 480` it is 0.79% (13.13), which is
  the figure 8.4's argument rests on. The same rule applies to the recall interval
  if one is reported: `n = 480` positives.
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
  the totals are 120 per family per repeat and 360 overall (each group is in
  exactly one test fold per repeat); for E, which is scored in every outer fold,
  they are 120 per fold, 600 per repeat and 1800 overall. The per-fold breakdown in the JSON records each family's test
  count and caught count, with recall `null` where the count is 0.
- The list of all false positives (the actual WANDS queries that would be discarded,
  as stored in `query.csv`: the loader does not normalise, section 5, so the three
  double-space queries appear with their double space). With ~5 FPs at 1% this
  list *is* the most useful output of the whole project.

Macro F1 and ROC-AUC are reported because the user asked for them; they are not used
for any selection. Selection (threshold) uses only FPR on inner validation data.

### 8.6 Anti-inflation: how would we know the model generalises?

No real bot data exists, so the standard CV above measures "can the model separate
WANDS from *these generators*". Four additional checks make that limitation visible:

1. **Leave-one-family-out (LOFO).** For each training family F in A-D (E is never
   trained on, so it has no LOFO row; its evaluation-only recall in 8.5 already is
   the "unseen family" number for E): reuse the outer folds of 8.1, drop F's groups
   from the training side, train on the remaining three families plus
   training-fold negatives with the same nested protocol (threshold chosen without
   F), and test on all of F generated from test-fold vocabulary plus the test-fold
   negatives. The training side is then 3:4 positive to negative; no re-balancing
   is applied, because the point is the threshold's transfer, not the class ratio.
   Report recall on F at that threshold, and the achieved FPR. This is the closest
   proxy for "a kind of bot we did not anticipate". Expect it to be much lower than
   in-distribution recall; that gap is the honest generalisation number. Off by
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
   machinery; shapes: bare prefix+digits like `sku000123`, bare digit runs,
   prefix-digits-suffix like `wf-2049x`, and product words from the full WANDS
   vocabulary plus a part number like `ines shag rug w003096809`), never used in any
   fit, threshold or audit-as-positive, and scores them with each final model (8.7).
   The P strings must pass the same exact-match check as positives (no P string
   equals a WANDS or sanity query after `normalise()`), so a flagged P string is
   never a memorised one; `bqc generate` writes them in its report under their own
   key, separate from the positives (section 10).
   Reported per model: the **fraction flagged at the model's threshold** (the
   expected FPR on such queries), with min/median/max score. The three part-number
   rows of the sanity set give a second, human-labelled look at the same thing.

All four are reported, not optimised. A run that looks great on 8.5 and poor on
8.6 is reported as exactly that. If the diagnostic shows part numbers flagged at a
high rate, the remedy (synthetic part-number negatives) changes the decision that
negatives are real WANDS queries, so it is a question for the user (12.2, Q12),
not something this design does on its own.

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

After evaluation, one model per configuration is fitted on all 960 samples and saved
with `pickle` to `results/` for the sanity-set scoring and manual probing. Its
threshold comes from the inner procedure of 8.2 applied to all 960 rows:
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)` with the groups
of 8.1, out-of-fold scores, then the threshold rule of 8.4. Positives are generated
once from the vocabulary of all 480 negatives; family E is excluded. Not a
deployment artefact. The pickle holds the scorer and its threshold; the SMOTE
callable of 7.1 is a module-level function so that SMOTE configurations pickle too.

## 9. Label discovery with HDBSCAN (separate experiment)

**Decision:** clustering is a **separate experiment**, implemented in
`bqc/cluster.py` and run with `python -m bqc cluster` (section 10; resolved
2026-11-05, OQ-8a), not a stage of the training pipeline. It shares the
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
  models.py               Scorer protocol + three adapters (section 6)
  evaluate.py             folds, nested threshold selection, metrics (section 8)
  cluster.py              HDBSCAN experiment (section 9)
  report.py               write_report(): config fields, naming, no overwrite
  cli.py                  command implementations
tests/                    unittest; python -m unittest discover -s tests
data/raw/                 gitignored downloads; data/CHECKSUMS committed; data/sanity.csv committed (8.6.1)
results/                  gitignored JSON reports and pickles, one file per run
scratch/                  gitignored probes
```

`python -m bqc generate` generates the positives with vocabulary from all 480
negatives (the 480 training positives of A-D **and** the 120 of family E, each
sample tagged with its family so E is never confused with the training set), runs
the audit (section 4, rules 2 and 4) over all 600, and writes one JSON report to
`results/` holding the generated samples with their metadata, the per-family length
statistics next to WANDS', the near-duplicate pairs, the 100 part-number
diagnostic strings (8.6 item 4) under their own key, never mixed with the
positives, and two numbers that keep 8.3's occurrence rule visible: the share of
hard-positive word slots filled by WANDS one-off words, next to the one-off share
of WANDS token occurrences (expected to be close; 13.17). It exists so the positives
can be inspected by eye; fold runs regenerate their own positives (8.3) and never
read its output.

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
  `python -m bqc fetch`, `python -m bqc generate`, `python -m bqc evaluate
  [--smote] [--target-fpr 0.01] [--lofo]`, `python -m bqc cluster`,
  `python -m unittest discover -s tests`.
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
7. **Synthetic positive identical or near-identical to a WANDS query, or identical
   to a sanity query.** Generation fails on exact matches against either list and
   reports near-duplicates against WANDS (section 4, rule 4; 8.6.1).
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
16. **Degenerate fold** (a family or group missing from a training fold).
    `StratifiedGroupKFold` stratifies on class, not family, so with few large
    template groups a family *can* land entirely in one test fold (probe 13.10: it
    happened in 7 of 75 inner folds with 3 templates per family, never with 5 or
    more). Section 4 rule 6 fixes 6 templates per family, and the runner **asserts**
    that both classes and every training family (A-D) are present in every outer
    and inner training fold and raises otherwise. The **test side is not asserted
    for families**: a family missing from a test fold is normal (8 of 15 outer folds
    with the real seeds, 13.14) and is handled by pooling per-family recall (8.5).
    LOFO removes one family on purpose and asserts the other three.
17. **All-noise HDBSCAN result.** Reported as 0 clusters, 100% noise; the seed rule
    then labels everything legitimate (FPR 0, recall 0). Not an error.
18. **Non-determinism.** Two runs with the same config and seed must produce
    identical JSON apart from the timestamp; this is a test on a tiny config.
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
23. **Part-number or SKU query** (`sku000123`, `w003096809`, `wf-2049x desk`).
    Legitimate by decision (section 2, Q11) but nearly absent from the negatives
    (13.15), so the model may flag it. Not an error at scoring time; measured by the
    part-number diagnostic (8.6 item 4). Generators must not emit such a shape as a
    whole query (2.2), and the registry test asserts it.

## 12. Decisions taken by the user, and open questions

### 12.1 Resolved on 2026-11-05

The v1 draft asked nine questions. The user's answers, now applied in the body:

1. **Taxonomy (section 2).** Families A-D are the training definition of
   "illegitimate"; E (off-topic natural language) is generated but
   evaluation-only. Accepted as proposed.
2. **LLM-generated positives (section 4).** No. Template generators only.
3. **SMOTE (section 7).** A measured experiment, default off. Yes.
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

Also noted by the user: the `.claude/agents` files were cleaned of their previous
project's references in commit `e2e4421`; nothing in this design depends on them.

### 12.2 Still open

12. **Synthetic part-number negatives?** WANDS has no bare part-number query
    (13.15), so the model cannot learn from the negatives that part numbers are
    legitimate, and the 1% FPR target does not cover them. The design measures the
    problem (8.6 item 4) but does not fix it, because adding generated negatives
    would change the recorded decision that negatives are real WANDS queries
    (section 1, section 3). *Recommend: wait for the diagnostic.* If the final
    models flag a large share of the 100 part-number strings, the proportionate fix
    is a small set of generated hard negatives (real product words plus a part
    number, ~50-100 rows) added to the negatives with their own group tag so folds
    and the FP list can tell them apart; if the share is small, do nothing. Does
    not block any task before M6.

### 12.3 Resolved by `fable` on plan review (2026-11-05, OQ-1 to OQ-8)

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
| 6 | 7.1 | SMOTE `sampling_strategy` is a callable doubling each class in the fitted `y`. Probed (13.12). |
| 7 | 7.1 | `--smote` runs both arms in one run and one JSON; `smote_comparison` section; `smote_justified` boolean with a stated rule. |
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

### 13.9 Unverified assumptions

- The 480 WANDS queries contain no bot traffic (13.2); not provable, mitigated by
  reviewing the FP list by eye.

This is the only unverified assumption left. Of the three items this section held
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

### 13.12 imblearn `Pipeline` with SMOTE at predict time (2026-11-05, imbalanced-learn 0.14.2, scikit-learn 1.9.1)

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
differently; the real-seed row matches exactly. Basis for 8.1 (assert families on
the training side only), 8.5 (pooled per-family recall) and 11.16.

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

## 14. Later (deferred, not in scope)

Not in this list because the user rejected them (12.1): LLM-generated positives
(Q2) and a second query source such as Amazon ESCI (Q7).

- **Behavioural features** (rate, session, IP reputation): the only way to catch
  scrapers that send well-formed queries. Different data, different project.
- **Catalogue-anchored cluster labelling**: label a cluster legitimate if its members
  are close to product names from WANDS `product.csv` (90 MB, in-domain). Removes
  the need for seed labels in section 9.2.
- **String-level augmentation of negatives** (typo injection, word dropout) as a more
  meaningful alternative to SMOTE for text.
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
- **Synthetic part-number hard negatives**, only if the user decides Q12 that way.
- **Serving**: input contract (11.2) is the only serving decision taken; a model
  file format, latency budget and monitoring would come with a deployment decision.
