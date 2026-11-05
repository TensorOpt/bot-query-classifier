# PLAN.md: bot-query-classifier

Owner: `opus`. Plans `DESIGN.md` v1.1 (commit `f450f97`). Created 2026-11-05.

Status values: `todo` -> `in progress` -> `done` (sonnet) -> `validated` (opus) ->
`signed off` (fable, per milestone). `blocked (OQ-n)` means the task can't start until
`fable` answers that open question (see "Open questions"). A task whose dependency is
blocked stays `todo`.

Test command, everywhere: `.venv/bin/python -m unittest discover -s tests`.

## Overview

| Milestone | Content | Tasks | Blocked |
|-----------|---------|-------|---------|
| M1 | Sanity set, scaffold, preprocessing, WANDS, sanity loader, fold probe | 7 | M1.T3, M1.T6 |
| M2 | Synthetic generators, audit, `generate` | 8 | M2.T1 |
| M3 | Features and evaluation harness against a fake `Scorer` | 11 | M3.T5, M3.T7 |
| M4 | Real models, `evaluate` command, SMOTE experiment | 7 | M4.T6, M4.T7 |
| M5 | HDBSCAN experiment | 3 | M5.T1, M5.T2 |
| M6 | Full run on real data | 1 | |

### Order constraints

1. **M1.T1 is the first task of the project.** `data/sanity.csv` must be committed
   before anything exists under `bqc/synth/` (8.6.1). The first `bqc/synth/` file
   appears in M2.T1, and M2.T1's acceptance checks the order in git history.
2. Probe tasks come before the task that depends on them (table below).
3. Generators (M2) come before the harness (M3), and the harness runs against a fake
   `Scorer` before any real model exists (M4).

### Probe tasks

| Probe | Assumption | Schedule before |
|-------|------------|-----------------|
| M1.T7 | `StratifiedGroupKFold` on template-level groups. 13.7 only tested 120 groups of 4, while 8.1 uses `(family, template_id)` groups, which are far fewer and larger | M2.T1 (informs OQ-3), M3.T3 |
| M3.T6 | Wilson interval reference values (13.9) | M3.T7 |
| M4.T5 | `imblearn.pipeline.Pipeline` skips SMOTE at predict time (13.9) | M4.T6 |

The third item in 13.9 (no bot traffic in the 480 queries) can't be probed; the design
says so. Its mitigation, reviewing the false-positive list by eye, is an acceptance
criterion of M6.T1.

### Where the user acts

The user has **one step**: labelling `data/sanity.csv`.

- **When:** any time after M1.T1 is committed. The best moment is **before M6.T1**, so
  the labels are given without having seen any model output.
- **How:** fill the `label` column of each row with `0` (legitimate, serve) or `1`
  (illegitimate, discard). Use a plain text editor: a spreadsheet can reinterpret
  queries as formulas or dates. Afterwards, `git diff data/sanity.csv` must show changes
  to the label column only.
- **Nothing waits for it.** No task needs the labels to reach `done`, `validated` or
  `signed off`. Until the file is fully labelled, `bqc evaluate` skips the sanity
  section (8.6.1, 11.20) and everything else runs. Scored-path tests use labelled
  fixtures.
- **Afterwards:** re-running `python -m bqc evaluate` produces the sanity section. This
  is an ordinary run, not a plan task.

No agent modifies `data/sanity.csv` after M1.T1.

## Open questions for `fable`

Each one blocks the listed task. Q10 (12.2) isn't here: it is non-blocking and the plan
follows the design's position, so off-domain queries are outside both classes and no
task exists for them.

| Id | Blocks | Question |
|----|--------|----------|
| OQ-1 | M1.T3 | **What exactly does `normalise()` do, and in what order?** Section 5 says preprocessing is "exactly" NFKC, strip, collapse whitespace, lowercase. 11.2 adds truncation to 512 characters and 11.4 adds stripping of zero-width and control characters. (a) What is the order? Truncating before NFKC changes the result, and truncating after it means normalising the whole 1 MB input that truncation is meant to avoid. (b) Which characters count? `\t` and `\n` are control characters, so stripping them before collapsing turns `a\tb` into `ab`, and collapsing first gives `a b`. (c) 11.2 says truncation is "logged in the score output", but `Scorer.score` returns only an ndarray (6.4) and there is no scoring command (section 10). Where does the log go? |
| OQ-2 | M1.T6 | **Is ` 1` a valid sanity label?** 11.20 lists ` 1` as a bad label, then says the loader strips surrounding whitespace before checking, which makes it a valid `1`. 8.6.1 says "not exactly `0` or `1`" is skipped. The test for this case can't be written until one of them wins. |
| OQ-3 | M2.T1 | **How many templates per family, and how is each family's 120 split among them, including the hard-positive share?** Section 4 fixes only 120 per family and 50 distinct strings per template. Because groups are `(family, template_id)` (8.1), this decides group sizes. So it also decides whether folds stay 50/50, and 13.7 doesn't cover that shape. It also bears on 11.16's claim that a whole family can't be missing from a training fold: `StratifiedGroupKFold` stratifies on class, not family, so a family with few templates could land entirely in one test fold. Should the runner assert family presence too? M1.T7 produces the evidence; answer after it. |
| OQ-4 | M3.T5 | **How are the inner folds built?** 8.2 says "inner 5-fold CV" without naming the splitter. Does it use `StratifiedGroupKFold` with the same groups? 8.3 regenerates positives per outer fold only, so inner-training hard positives can contain vocabulary from inner-validation negatives. Is that accepted, or are positives also regenerated per inner fold? |
| OQ-5 | M3.T7 | **What is `n` for the "pooled" Wilson interval?** 8.5 reports "per repeat and pooled". Each negative is scored once per repeat, so pooling all 3 repeats gives `n = 1440` non-independent observations. That makes the interval too narrow (0 FPs: upper bound 0.27% at `n = 1440`, 0.79% at `n = 480`), and undercuts 8.4's argument that sub-1% isn't demonstrable. Does the headline CI use `n = 480` per repeat, some aggregate of those, or `n = 1440`? |
| OQ-6 | M4.T6 | **What is `n` in `sampling_strategy={0: n, 1: n}`?** Section 7 says only `n > 480`, and that comes from the full set. Training folds hold about 384 per class and inner-training folds about 307, and each fit needs a value at least as large as its class counts. Is it an absolute count, or a multiple of the class size within each fit? |
| OQ-7 | M4.T7 | **What does `--smote` do?** Does `evaluate --smote` run both arms in one run and one JSON file (which is where the paired per-fold difference of section 7 would live), or only the SMOTE arm, with the comparison made across two files by something the design doesn't name? Also, is "within fold-to-fold noise" a computed verdict with a rule, or does the report give mean and sd of the differences and leave the conclusion to a reader? |
| OQ-8 | M5.T1, M5.T2 | **Clustering details.** (a) Section 9 says `experiments/cluster`, while section 10 says `bqc/cluster.py` and `python -m bqc cluster`. Which? (b) What label does a cluster get on a seed tie (9.2), for example 1-1, and on an oracle tie? (c) How do ARI, AMI, homogeneity and completeness treat noise points: as one cluster, or excluded? |

### Readings taken (not blocking; correct them if wrong)

These follow from the design's text, but they are choices a reader could contest.

1. **LOFO (8.6, item 1) covers families A-D only**, since E is never trained on. It reuses the
   outer folds, drops F's groups from training, and tests on all of F (with test-fold
   vocabulary) plus the test-fold negatives. The resulting class imbalance in training
   is left as is, because the design specifies no re-balancing.
2. **SMOTE arms exist for the MLP and LR only** (section 7), not the trivial baseline.
3. **`bqc generate`** generates the positives with vocabulary from all 480 negatives,
   runs the audit, and writes a JSON report to `results/`. Fold runs regenerate their
   own positives and don't read its output (8.3).
4. **The exact-match failure** (section 4 rule 4) runs on every generation, including
   per-fold, and compares through `normalise()`, which subsumes strip+lower. The
   near-duplicate report runs only in `generate`.
5. **`bqc/sanity.py`** is a new module not in section 10's tree, because the sanity
   loader is a separate responsibility from `wands.py` and `evaluate.py`. `SEED` lives in
   `bqc/__init__.py`, since section 10 names no module for it.
6. **No results document is planned**, since the design names none. M6.T1 reports its
   numbers to `fable`.

### Design nits (no task affected)

- 8.4 quotes 0.77% as the 95% upper bound for 0/480. That is the exact (Clopper-Pearson)
  figure; the Wilson bound the design reports is 0.79%. M3.T7's tests use Wilson values.
- 8.6.1's brief needs two facts from outside that section: the `SEED` value (section 10)
  and the WANDS URL (13.1). M1.T1 supplies both. Neither concerns the taxonomy.

### Fact checked while planning

All 480 WANDS queries remain distinct after NFKC, strip, whitespace collapse and
lowercase (2026-11-05, Python 3.12, `query.csv` SHA-256 `63b61660...68e95`), so 8.1's
singleton groups hold under any answer to OQ-1 that keeps those steps.

---

## M1: Foundation

Sign-off: pending.

### M1.T1: Author the unlabelled sanity set

- **Goal:** commit 50 unlabelled queries in `data/sanity.csv`, written before any
  generator exists.
- **Design refs:** 8.6.1 only.
- **Files:** `data/sanity.csv`. No other file is in the commit.
- **Delegation (restricted brief, required by 8.6.1):**
  - `fable` pastes this task entry and the text of 8.6.1 into the delegation, and adds
    the two facts the section doesn't contain: `SEED = 20261105`, and the WANDS URL
    `https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv`.
  - The delegate must **not** open `DESIGN.md` or the rest of `PLAN.md`. It must not
    read any other section, or any other task, since those describe the generators.
  - As an exception to the usual workflow, the delegate doesn't edit `PLAN.md`. `fable`
    records this task's status changes in separate commits, so the task commit holds
    only the data file.
- **Acceptance criteria:**
  1. Header is exactly `id,query,label`; comma-separated, UTF-8; `id` runs `1..50`;
     every `label` is empty; there is no other column.
  2. Rows are shuffled with `random.Random(20261105)` before ids are assigned.
  3. The content follows the guideline in 8.6.1, and no query is off-domain.
  4. No query equals a WANDS query. Check with a throwaway script in `scratch/` comparing
     strip + whitespace collapse + lowercase (`normalise()` doesn't exist yet; M1.T6
     enforces it from then on). Keep the download in its own empty directory, and run
     the script from elsewhere with `python -I`.
  5. `git ls-tree -r --name-only HEAD | grep '^bqc/synth/'` prints nothing at this
     commit.
  6. **Blindness:** the author's intent appears nowhere in the commit, in the commit
     message, or in the report. Delete the scratch authoring file after use. The
     suggested message is `chore(M1.T1): add unlabelled sanity query set`.
- **Tests:** no code in this task. The validator checks the criteria above with
  read-only commands. M1.T6 adds an automated structure test on the real file.
  The validator doesn't record any breakdown of the content by intended class, in
  `PLAN.md` or in its report, because the user reads both before labelling.
- **Depends on:** none.
- **Status:** todo

### M1.T2: Project scaffold

- **Goal:** create an installable, empty package with a working test command.
- **Design refs:** 6 (dependencies), 10.
- **Files:** `pyproject.toml`, `bqc/__init__.py` (`SEED = 20261105`), `tests/test_seed.py`,
  `.gitignore` (add `data/raw/`, `results/`).
- **Acceptance criteria:**
  1. `python3 -m venv .venv && .venv/bin/pip install -e .` succeeds on Python 3.12.
  2. Dependencies are exactly `numpy`, `scikit-learn>=1.3`, `imbalanced-learn`, each
     justified in the commit message.
  3. The test command runs and passes.
  4. `git check-ignore data/raw/x results/x scratch/x .venv` matches all four, and
     `data/sanity.csv` isn't ignored.
  5. The report states the installed versions of numpy, scikit-learn and imbalanced-learn
     and flags any that differ from those probed in 13.3-13.7.
- **Tests:** `SEED == 20261105`.
- **Depends on:** M1.T1 (schedule order only).
- **Status:** todo

### M1.T3: `normalise()`

- **Goal:** one function holds all query preprocessing.
- **Design refs:** 5, 11.1, 11.2, 11.4, 11.5.
- **Files:** `bqc/preprocess.py`, `tests/test_preprocess.py`.
- **Acceptance criteria:**
  1. `normalise(str) -> str` applies NFKC, strip, whitespace collapse and lowercase, plus
     the truncation and character stripping as specified by the answer to OQ-1.
  2. It is idempotent, and it is the only preprocessing implementation in the package.
  3. Truncation reporting follows the answer to OQ-1(c).
- **Tests:**
  - Empty and whitespace-only input return `""` without raising (11.1).
  - Lengths 511, 512, 513 and a 1 MB string (11.2), including the reporting.
  - Full-width ASCII maps to ASCII; zero-width and control characters are removed;
    combining marks, emoji and non-ASCII letters are kept (11.4).
  - Tabs, newlines and internal runs of whitespace, per OQ-1(b).
  - Mixed case (11.5).
  - Idempotence over all the above.
- **Depends on:** M1.T2.
- **Status:** blocked (OQ-1)

### M1.T4: WANDS loader with hash check

- **Goal:** load `query.csv`, refusing a file whose hash doesn't match.
- **Design refs:** 3, 10, 11.14, 11.15, 13.8.
- **Files:** `bqc/wands.py`, `data/CHECKSUMS` (format and loader support only; M1.T5 adds
  the real hash), `tests/test_wands.py`, `tests/fixtures/query.tsv`.
- **Acceptance criteria:**
  1. Reads with `csv`, `delimiter="\t"`, default quoting, UTF-8.
  2. Returns queries in file order, and doesn't normalise or use `query_class`.
  3. A hash mismatch or a missing file raises an error naming the path, before parsing.
  4. Tests never download.
- **Tests:** a 10-row fixture with the real header and delimiter, containing:
  - a quoted row such as `"fawkes 36"" blue vanity"`, read as `fawkes 36" blue vanity`
    (11.14);
  - an empty `query_class` (11.15);
  - a double-space query kept as is.
  Failure paths: wrong hash; missing file; wrong header.
- **Depends on:** M1.T2.
- **Status:** todo

### M1.T5: `bqc fetch` and the real checksum

- **Goal:** download `query.csv` atomically and pin its hash.
- **Design refs:** 3, 10, 11.13.
- **Files:** `bqc/wands.py`, `bqc/cli.py`, `bqc/__main__.py`, `data/CHECKSUMS`,
  `tests/test_fetch.py`.
- **Acceptance criteria:**
  1. `python -m bqc fetch` downloads with the standard library into `data/raw/query.csv`,
     writing to a temp file and renaming only after the hash check passes.
  2. A network error or hash mismatch exits non-zero and leaves no file at the target or
     temp path.
  3. `data/CHECKSUMS` holds the real SHA-256 from one manual run (the planning check saw
     `63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95`; report any
     difference rather than overwriting it silently).
  4. The real file loads as 480 rows with `query_id` `0..479`.
- **Tests:** with the URL opener replaced by a local stub, so nothing downloads:
  success; network error; wrong bytes; an existing good file isn't destroyed by a failed
  fetch (11.13).
- **Depends on:** M1.T4.
- **Status:** todo

### M1.T6: Sanity set loader

- **Goal:** read `data/sanity.csv`, and report whether it can be scored.
- **Design refs:** 8.6.1, 10, 11.20, 11.21.
- **Files:** `bqc/sanity.py`, `tests/test_sanity_loader.py`, `tests/fixtures/sanity_*.csv`.
- **Acceptance criteria:**
  1. Returns either all rows with integer labels, or a "not scorable" result carrying the
     count of invalid rows and the message `sanity set: N of M rows unlabelled, skipping`.
  2. Blank is never a class, and a partially labelled file is never scored.
  3. A query equal to a WANDS query after `normalise()` raises an error naming the `id`,
     regardless of the labels' state.
  4. A wrong header raises an error naming the file.
  5. The loader doesn't write to the file.
- **Tests:**
  - All blank; all labelled; partly labelled; bad labels `2`, `yes`, and ` 1` per OQ-2
    (11.20).
  - A row equal to a WANDS fixture query after case and whitespace differences (11.21).
  - Quoted queries containing commas; CRLF line endings.
  - **The real `data/sanity.csv`**: header, 50 rows, ids `1..50`, no duplicate queries.
    This test must pass both blank and fully labelled, so it asserts nothing about label
    values.
- **Depends on:** M1.T1, M1.T3.
- **Status:** blocked (OQ-2)

### M1.T7: Probe `StratifiedGroupKFold` on template-level groups

- **Goal:** measure fold behaviour for the group shape 8.1 really uses, as evidence for
  OQ-3.
- **Design refs:** 8.1, 11.16, 13.7.
- **Files:** `scratch/` (not committed); `DESIGN.md` section 13 (a new dated finding).
- **Acceptance criteria:** the finding is recorded with date and versions and covers,
  for 480 singleton negatives plus positives in 4 families with 3, 5 and 8 templates per
  family (equal and unequal quotas), `n_splits=5`, 3 seeds:
  1. the positive share per test fold (min and max);
  2. whether any family is ever absent from a training fold;
  3. the same two measures for an inner 5-fold split of one training fold;
  4. whether any split raises an error.
- **Tests:** none (probe).
- **Depends on:** M1.T2.
- **Status:** todo

---

## M2: Synthetic generators

Sign-off: pending.

### M2.T1: Sample type, registry and quotas

- **Goal:** a seeded, vocabulary-parameterised entry point for generating positives.
- **Design refs:** 2, 4, 8.3, 10, 11.18.
- **Files:** `bqc/synth/__init__.py`, `bqc/synth/registry.py`, `tests/test_synth_registry.py`.
- **Acceptance criteria:**
  1. **History:** the commit of `data/sanity.csv` is an ancestor of the first commit
     touching `bqc/synth/` (`git merge-base --is-ancestor`).
  2. A sample carries `text`, `family`, `template_id`, `seed`.
  3. The generation function takes a vocabulary and a seed. Its default output is the
     training set (A-D, 120 each, 480 total); family E is requested separately and is
     never in the 480.
  4. Templates per family and quota split follow the answer to OQ-3, and group sizes can
     be read without generating text (M3.T3 needs them before vocabulary is known).
  5. An empty vocabulary raises `ValueError`.
- **Tests** (with stub templates; they also run against every real template once
  registered, so M2.T2-T6 inherit them):
  - Output lowercased and ASCII-only (section 4 rule 1).
  - Same seed gives identical output, including across two subprocesses with different
    `PYTHONHASHSEED`; a different seed gives different output (11.18).
  - Every family has at least one hard-positive template, which uses the vocabulary: a
    sentinel word appears in the output, and a word outside the given vocabulary doesn't
    (rules 3, 8.3).
  - Each template reaches at least 50 distinct strings with a fold-sized vocabulary
    (rule 5).
  - Every sample has its metadata.
- **Depends on:** M1.T1, M1.T7.
- **Status:** blocked (OQ-3)

### M2.T2: Family A, gibberish

- **Goal:** generate keyboard mashing, random IDs and hash-like strings, plus hard
  positives.
- **Design refs:** 2 (family A, hard positives), 4.
- **Files:** `bqc/synth/gibberish.py`, `tests/test_synth_gibberish.py`.
- **Acceptance criteria:** covers mashing, repeated characters and UUID/hash-like
  strings, and a hard-positive template such as `accent chair asdfgh`. Registered with
  family `A`. Produces no typo-style positives (section 2).
- **Tests:** registry tests pass for this family; template-specific shape checks (for
  example, a UUID template has UUID shape); output isn't a perturbed product word.
- **Depends on:** M2.T1.
- **Status:** todo

### M2.T3: Family B, injection and probing payloads

- **Goal:** generate scanner payloads, plus hard positives.
- **Design refs:** 2 (family B, hard positives), 4.
- **Files:** `bqc/synth/payloads.py`, `tests/test_synth_payloads.py`.
- **Acceptance criteria:** covers SQL, script, template, path-traversal and NoSQL
  patterns, and a hard-positive template such as `leather sofa' union select`.
  Registered with family `B`. Payloads are only ever generated as text, never executed.
- **Tests:** registry tests pass for this family; each pattern kind appears in the
  output; payload characters survive lowercasing.
- **Depends on:** M2.T1.
- **Status:** todo

### M2.T4: Family C, enumeration and scraping

- **Goal:** generate SKU sequences, short dictionary walks, URL paths and query strings,
  plus hard positives.
- **Design refs:** 2 (family C, single-word row, hard positives), 4.
- **Files:** `bqc/synth/enumeration.py`, `tests/test_synth_enumeration.py`.
- **Acceptance criteria:** covers bare SKUs, walks, paths and `page=2&sort=price` forms,
  and a hard-positive template. Registered with family `C`.
- **Tests:** registry tests pass for this family; every dictionary-walk output is 1-2
  characters, so real single-word queries such as `zen` or `owl` can never appear; the
  walk template still reaches 50 distinct strings.
- **Depends on:** M2.T1.
- **Status:** todo

### M2.T5: Family D, spam and promotional

- **Goal:** generate URLs, emails, phone numbers and promotional phrases, plus hard
  positives.
- **Design refs:** 2 (family D, hard positives), 4.
- **Files:** `bqc/synth/spam.py`, `tests/test_synth_spam.py`.
- **Acceptance criteria:** covers each listed kind, and a hard-positive template such as
  `cheap area rug http://x.co`. Registered with family `D`.
- **Tests:** registry tests pass for this family; each kind appears in the output.
- **Depends on:** M2.T1.
- **Status:** todo

### M2.T6: Family E, off-topic natural language (evaluation-only)

- **Goal:** generate 120 questions, chat prompts and code snippets, tagged `E`.
- **Design refs:** 2 (family E, 2.1), 4.
- **Files:** `bqc/synth/offtopic.py`, `tests/test_synth_offtopic.py`.
- **Acceptance criteria:**
  1. Generated by the same machinery, with `family="E"` and a hard-positive template.
  2. Never part of the training 480.
  3. Output is text that is not a product search; no template produces an off-domain
     shopping query (2.1).
- **Tests:** registry tests pass for this family; the training output contains no `E`
  sample; requesting E returns exactly 120.
- **Depends on:** M2.T1.
- **Status:** todo

### M2.T7: Generation audit

- **Goal:** fail on exact collisions with negatives, report near-duplicates, and log
  length distributions.
- **Design refs:** 4 (rules 2 and 4), 11.7.
- **Files:** `bqc/synth/audit.py`, `tests/test_synth_audit.py`.
- **Acceptance criteria:**
  1. A positive identical to a negative after `normalise()` raises an error naming both
     strings.
  2. Every positive-negative pair with character-3-gram Jaccard above 0.6 is reported.
  3. Length statistics are returned per family, alongside those of WANDS.
- **Tests:**
  - Identical pair; pair differing only in case and whitespace.
  - Jaccard exactly 0.6 (not reported) and just above it (reported).
  - Strings shorter than 3 characters, such as family C's `a`, don't raise.
- **Depends on:** M1.T3, M2.T1.
- **Status:** todo

### M2.T8: `bqc generate`

- **Goal:** generate the full-vocabulary positives from the command line and write the
  audit report.
- **Design refs:** 4, 10.
- **Files:** `bqc/cli.py`, `tests/test_cli_generate.py`.
- **Acceptance criteria:**
  1. `python -m bqc generate` uses vocabulary from all 480 negatives and writes a JSON
     report to `results/` without overwriting an earlier one.
  2. An audit failure exits non-zero, and a missing `data/raw/query.csv` exits non-zero
     with a message pointing at `fetch`.
  3. The real run passes the audit, and its near-duplicate count goes in the task report.
- **Tests:** on fixture paths: success writes a report; collision exits non-zero; missing
  raw file exits non-zero.
- **Depends on:** M1.T4, M1.T5, M2.T2, M2.T3, M2.T4, M2.T5, M2.T6, M2.T7.
- **Status:** todo

---

## M3: Features and evaluation harness (fake `Scorer`)

Sign-off: pending.

### M3.T1: Vectoriser factory

- **Goal:** one factory for the TF-IDF configuration.
- **Design refs:** 5, 11.1, 11.3, 11.9.
- **Files:** `bqc/features.py`, `tests/test_features.py`.
- **Acceptance criteria:** returns an unfitted vectoriser with exactly the parameters of
  section 5. Empty and unknown strings give an all-zero row without raising.
- **Tests:**
  - Parameter values.
  - Empty string and all-unknown string give a zero row (11.1, 11.3).
  - Cross-token n-grams such as `' or` are kept, and punctuation survives.
  - `min_df=2` drops a one-document n-gram.
  - After fitting on a training split, no n-gram exclusive to the test split is in the
    vocabulary (11.9).
- **Depends on:** M1.T3.
- **Status:** todo

### M3.T2: `Scorer` port and fake

- **Goal:** define the protocol the evaluation depends on, and a deterministic fake.
- **Design refs:** 6.4.
- **Files:** `bqc/models.py` (protocol only), `tests/fakes.py`, `tests/test_scorer_port.py`.
- **Acceptance criteria:** protocol is `fit(texts, y)` and `score(texts) -> ndarray`. The
  fake satisfies it, returns scores from a fixed rule, records what it was fitted on (so
  later tests can assert leakage properties), and is picklable.
- **Tests:** the fake meets the protocol; it returns one score per text; it records its
  fit input.
- **Depends on:** M1.T2.
- **Status:** todo

### M3.T3: Outer folds and per-fold positives

- **Goal:** build repeated grouped folds, and regenerate positives from each side's
  vocabulary.
- **Design refs:** 8.1, 8.3, 10 (seeds), 11.8, 11.16.
- **Files:** `bqc/evaluate.py`, `tests/test_folds.py`.
- **Acceptance criteria:**
  1. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + repeat_index)`,
     3 repeats, with both parameters overridable for tiny test configs.
  2. Groups: each query is its own group; each positive's group is `(family, template_id)`.
  3. Training hard positives draw only on training-fold negatives, and test hard positives
     only on test-fold negatives.
  4. Family E is generated from test-fold vocabulary for scoring, and never appears in a
     training set.
  5. The runner raises if either class is missing from any training or test fold (11.16);
     any family-presence check follows the answer to OQ-3.
- **Tests:**
  - No group on both sides of a fold (11.8).
  - A sentinel word present only in a test-fold negative never appears in a training
    positive, and the reverse (8.3).
  - No E sample is in a training set.
  - Same seed gives the same folds; different repeats give different folds.
  - A degenerate input raises the designed error (11.16).
- **Depends on:** M1.T7, M2.T1. Real families aren't required, because the tests use stub
  templates.
- **Status:** todo

### M3.T4: Threshold selection

- **Goal:** pick the threshold from out-of-fold scores, as a pure function.
- **Design refs:** 8.2, 8.4, 11.10, 11.11, 11.19.
- **Files:** `bqc/evaluate.py`, `tests/test_threshold.py`.
- **Acceptance criteria:**
  1. Returns the smallest score `t` with inner FPR at or below the target.
  2. The decision rule `score >= t` is illegitimate lives in one place.
  3. If no threshold reaches the target, returns the one with the smallest FPR and sets
     `target_unmet=true` (11.10).
  4. `target_fpr < 1/n_negatives` sets a warning field, with no silent rounding (11.11).
- **Tests:** hand-computed cases:
  - clean separation;
  - full overlap and tied scores, including ties straddling the target;
  - a score exactly at `t` counted illegitimate (11.19);
  - a target below resolution;
  - `target_unmet`;
  - no negatives or no positives raises `ValueError`.
- **Depends on:** M1.T2.
- **Status:** todo

### M3.T5: Nested outer-fold runner

- **Goal:** for each outer fold, select a threshold on inner out-of-fold scores, refit on
  the training fold, and score the test fold and family E.
- **Design refs:** 8.2, 8.5 (family E), 11.9, 11.12.
- **Files:** `bqc/evaluate.py`, `tests/test_runner.py`.
- **Acceptance criteria:**
  1. Depends only on the `Scorer` port, via a factory; imports no sklearn estimator.
  2. Inner folds follow the answer to OQ-4.
  3. Scorer fits see only outer-training rows, and the threshold comes only from inner
     scores.
  4. Family E is scored by each fold's model at that fold's threshold.
  5. Warnings raised during `fit` (11.12) are captured per fold and returned, and don't
     raise.
- **Tests** (with the fake):
  - No test-fold or E text in any recorded fit input (11.9).
  - Number of fits: outer folds x (inner folds + 1).
  - Changing test-fold scores doesn't change the threshold.
  - The test-fold row count is unchanged at scoring.
  - A fake emitting `ConvergenceWarning` is captured, not raised (11.12).
- **Depends on:** M3.T2, M3.T3, M3.T4.
- **Status:** blocked (OQ-4)

### M3.T6: Probe Wilson reference values

- **Goal:** obtain independent reference values for the in-house Wilson interval.
- **Design refs:** 8.4, 13.9.
- **Files:** `scratch/` (not committed); `DESIGN.md` section 13 (a new dated finding).
- **Acceptance criteria:** the finding records the values for `0/480`, `5/480`, `0/30`,
  `15/1440` and `480/480` at 95%, obtained from an implementation independent of this
  project (`scipy.stats.binomtest(...).proportion_ci(method="wilson")`; scipy is already
  transitive), with its version. It also closes the Wilson item in 13.9.
- **Tests:** none (probe).
- **Depends on:** M1.T2.
- **Status:** todo

### M3.T7: Metrics and aggregation

- **Goal:** compute every number in 8.5 from fold results.
- **Design refs:** 8.4, 8.5, 13.9.
- **Files:** `bqc/evaluate.py`, `tests/test_metrics.py`.
- **Acceptance criteria:**
  1. Per repeat and pooled: headline recall and achieved FPR, Wilson 95% CI (standard
     library `math` only, with `n` per the answer to OQ-5), confusion matrix, macro F1,
     precision, recall, ROC-AUC as mean and sd over folds and pooled.
  2. Per-family recall. E is in its own row, marked evaluation-only, and never pooled into
     headline recall.
  3. The full list of false positives, with their query text.
  4. No metric feeds any selection.
- **Tests:**
  - Wilson against the M3.T6 values, including `k = 0` and `k = n`.
  - Hand-computed confusion matrix and per-family recall.
  - Adding E samples leaves headline recall unchanged.
  - The FP list holds exactly the expected queries.
  - A single-class input to ROC-AUC raises rather than reporting a number.
- **Depends on:** M3.T5, M3.T6.
- **Status:** blocked (OQ-5)

### M3.T8: Run report JSON

- **Goal:** write one never-overwritten, reproducible JSON file per run.
- **Design refs:** 10, 11.12, 11.18.
- **Files:** `bqc/evaluate.py`, `tests/test_report.py`.
- **Acceptance criteria:**
  1. Contains config (all parameters, seed, git commit hash, library versions), per-fold
     and pooled metrics, FP list, per-family recall, `target_unmet` and warning fields,
     and per-fold convergence warnings.
  2. The file name includes the config hash and timestamp; an existing file is never
     overwritten.
- **Tests:**
  - Two tiny-config runs give identical JSON apart from the timestamp (11.18).
  - A name collision doesn't overwrite.
  - Output round-trips through `json` (no numpy types left).
- **Depends on:** M3.T7.
- **Status:** todo

### M3.T9: Leave-one-family-out

- **Goal:** report recall on each family at a threshold chosen without it.
- **Design refs:** 8.6 (item 1), 11.16.
- **Files:** `bqc/evaluate.py`, `tests/test_lofo.py`.
- **Acceptance criteria:**
  1. Runs per the reading above: families A-D, same nested protocol, F absent from every
     fit and from threshold selection.
  2. Results are added to the run JSON as their own section, separate from the headline.
  3. Off by default, enabled by a parameter (`--lofo` is wired in M4.T4).
- **Tests** (with the fake):
  - No F text in any fit input.
  - The test side holds F plus test-fold negatives only.
  - One result row per family.
  - Both classes are present on both sides.
- **Depends on:** M3.T8.
- **Status:** todo

### M3.T10: Final models

- **Goal:** fit one model per configuration on all 960 samples, and pickle it.
- **Design refs:** 8.7, 10.
- **Files:** `bqc/evaluate.py`, `tests/test_final_model.py`.
- **Acceptance criteria:**
  1. The threshold comes from a 5-fold out-of-fold pass on all data; the model is then
     fitted on all of it.
  2. Positives use vocabulary from all negatives, and family E is excluded.
  3. The pickle goes to `results/` and holds the scorer and threshold, and loading it
     reproduces the same scores.
- **Tests** (with the fake): 960-equivalent fit input with no E; pickle round trip gives
  identical scores; the threshold is produced by the M3.T4 function.
- **Depends on:** M3.T5.
- **Status:** todo

### M3.T11: Sanity section of the run

- **Goal:** score the sanity set with the final models when it's fully labelled, and skip
  it otherwise.
- **Design refs:** 8.6.1, 11.20, 11.21.
- **Files:** `bqc/evaluate.py`, `tests/test_sanity_section.py`.
- **Acceptance criteria:**
  1. **Blank or partial file:** the message is printed, the run JSON records
     `"sanity": {"status": "skipped", "unlabelled": n}`, and everything else proceeds.
  2. **Fully labelled file:** reports the raw-count confusion matrix and the full list of
     disagreements (query, user's label, model's score), per configuration, using the
     threshold of M3.T10. There is no macro F1, ROC-AUC, confidence interval or decimal
     rate.
  3. A duplicate of a WANDS query fails the run with the offending `id` (11.21).
  4. Sanity data is never passed to any `fit` or threshold selection.
- **Tests** (with the fake, and fixture files; none uses the real labels):
  - Blank, partial and bad-label fixtures give the skip record, with the rest of the
    report present (11.20).
  - A labelled fixture gives the expected counts and disagreements, and no forbidden
    metric keys.
  - Fit inputs contain no sanity text.
  - Duplicate raises (11.21).
- **Depends on:** M1.T6, M3.T8, M3.T10.
- **Status:** todo

---

## M4: Real models and the SMOTE experiment

Sign-off: pending.

### M4.T1: Logistic regression adapter

- **Goal:** the `Scorer` for 6.2.
- **Design refs:** 5, 6.2, 6.4, 11.1, 11.3.
- **Files:** `bqc/models.py`, `tests/test_models_lr.py`.
- **Acceptance criteria:** `LogisticRegression(C=1.0, max_iter=1000)` on the M3.T1
  vectoriser. Every text passes through `normalise()` exactly once. Scores are
  P(illegitimate), the probability of label `1`, in `[0, 1]`.
- **Tests:**
  - On tiny inputs: `score` is aligned with the class-1 probability.
  - `"  SOFA "` and `sofa` score the same.
  - Empty and all-unknown strings give the same score, and don't raise (11.1, 11.3).
  - Calling `score` before `fit` raises.
  - No metric values are asserted (section 10).
- **Depends on:** M3.T1, M3.T2.
- **Status:** todo

### M4.T2: MLP adapter

- **Goal:** the `Scorer` for 6.1.
- **Design refs:** 6.1, 6.4, 11.12, 13.6.
- **Files:** `bqc/models.py`, `tests/test_models_mlp.py`.
- **Acceptance criteria:** exact parameters of 6.1, on the same vectoriser, and
  `random_state` is passed in from the seed. Doesn't suppress `ConvergenceWarning`, so
  the runner can capture it (M3.T5).
- **Tests:**
  - Two fits with the same seed give identical scores (13.6, 11.18).
  - Empty string doesn't raise (11.1).
  - Parameter values.
  - A forced `max_iter=1` fit emits `ConvergenceWarning`, and the fit still returns (11.12).
- **Depends on:** M4.T1.
- **Status:** todo

### M4.T3: Trivial baseline adapter

- **Goal:** the `Scorer` for 6.3, the artefact detector.
- **Design refs:** 6.3, 6.4, 8.6.
- **Files:** `bqc/models.py`, `tests/test_models_trivial.py`.
- **Acceptance criteria:** logistic regression on exactly three features (character
  length, word count, non-alphanumeric ratio), computed on normalised text. Marked as
  reported, never selected.
- **Tests:** hand-computed features for a few strings, including `""` with no division by
  zero (11.1); scores in `[0, 1]`.
- **Depends on:** M4.T1.
- **Status:** todo

### M4.T4: `bqc evaluate` command

- **Goal:** run the harness with the three real models from the command line.
- **Design refs:** 8.4, 8.5, 8.6, 10.
- **Files:** `bqc/cli.py`, `tests/test_cli_evaluate.py`.
- **Acceptance criteria:**
  1. `python -m bqc evaluate [--target-fpr 0.01] [--lofo]` runs the three models and writes
     the run JSON, including the sanity section or its skip record.
  2. `--target-fpr` outside `(0, 1)` is a usage error; a missing raw file exits non-zero
     with a message pointing at `fetch`.
  3. The real run completes whatever the state of `data/sanity.csv`; while the file isn't
     fully labelled, it reports the skip message.
- **Tests:** on a tiny config and fixtures: end-to-end with each flag; JSON has one entry
  per model; the trivial baseline's report row is present; bad `--target-fpr`; missing
  raw file; **real model and fold determinism end to end** (11.18).
- **Depends on:** M2.T8, M3.T9, M3.T11, M4.T1, M4.T2, M4.T3.
- **Status:** todo

### M4.T5: Probe `imblearn` Pipeline at predict time

- **Goal:** verify that sampling steps are skipped at predict time, as 7 assumes.
- **Design refs:** 7, 13.9.
- **Files:** `scratch/` (not committed); `DESIGN.md` section 13 (a new dated finding).
- **Acceptance criteria:** the finding records date and versions and shows, for
  `TfidfVectorizer -> SMOTE(sampling_strategy={0: n, 1: n}) -> estimator` on sparse input:
  1. the estimator's `fit` sees more rows than the input;
  2. `predict_proba` on `m` rows returns `m` rows and calls the sampler zero times;
  3. the same with `MLPClassifier` and `LogisticRegression`;
  4. what error is raised when `n` is below a class count (evidence for OQ-6).
  It also closes the `imblearn` item in 13.9.
- **Tests:** none (probe).
- **Depends on:** M1.T2.
- **Status:** todo

### M4.T6: SMOTE option in the adapters

- **Goal:** let the LR and MLP adapters run `vectoriser -> SMOTE -> estimator`.
- **Design refs:** 6.4, 7, 13.4, 11.9.
- **Files:** `bqc/models.py`, `tests/test_models_smote.py`.
- **Acceptance criteria:**
  1. Off by default. When on, it uses `imblearn.pipeline.Pipeline`, with `sampling_strategy`
     per the answer to OQ-6 and `random_state` from the seed.
  2. Applied to training data only.
  3. `imblearn` is imported only in this code path; stages don't import it.
- **Tests:**
  - The test-fold row count is unchanged after the pipeline (7, 11.9).
  - The estimator saw more rows than the training input (so the sampler really ran,
    unlike default SMOTE on balanced data, 13.4).
  - Same seed gives identical scores.
  - With SMOTE off, scores equal those of the adapter before this task.
- **Depends on:** M4.T1, M4.T2, M4.T5.
- **Status:** blocked (OQ-6)

### M4.T7: SMOTE comparison and `--smote`

- **Goal:** report the paired per-fold difference in the headline metric, for MLP and LR.
- **Design refs:** 7, 8.5, 10.
- **Files:** `bqc/evaluate.py`, `bqc/cli.py`, `tests/test_smote_comparison.py`.
- **Acceptance criteria:**
  1. Flag semantics and location of the comparison follow the answer to OQ-7.
  2. Both arms use identical folds, seeds and generated positives.
  3. The report gives per-fold differences for MLP and LR, and no SMOTE arm for the trivial
     baseline.
  4. Final models and sanity scoring cover the SMOTE configurations too (8.7).
- **Tests** (with fakes): identical folds and texts across arms; hand-computed paired
  differences; default run has no SMOTE arm.
- **Depends on:** M4.T4, M4.T6.
- **Status:** blocked (OQ-7)

---

## M5: HDBSCAN experiment

Sign-off: pending.

### M5.T1: Clustering pipeline and grid

- **Goal:** TF-IDF, SVD, L2 normalisation and HDBSCAN on all 960 strings with labels
  hidden.
- **Design refs:** 9, 9.1, 11.17, 13.5.
- **Files:** per OQ-8(a) (section 10 names `bqc/cluster.py`); `tests/test_cluster.py`.
- **Acceptance criteria:**
  1. Uses the M3.T1 vectoriser; `TruncatedSVD(random_state=SEED)`; `copy` set explicitly on
     `HDBSCAN` (13.5).
  2. Defaults `n_components=50`, `min_cluster_size=15`; the grid of 9.1 (`n_components`
     in {20, 50, 100}, `min_cluster_size` in {10, 15, 30}) is run and reported.
  3. Labels aren't passed to any clustering step, and the module is independent of the
     training pipeline.
- **Tests:**
  - On tiny inputs: rows are unit-norm after normalising.
  - Same seed gives identical labels.
  - An all-noise result returns 0 clusters and 100% noise without raising (11.17).
- **Depends on:** M3.T1, M2.T8.
- **Status:** blocked (OQ-8)

### M5.T2: Cluster labelling rules and metrics

- **Goal:** assign classes to clusters by both rules of 9.2, and compute their metrics.
- **Design refs:** 9.2, 11.17.
- **Files:** same module as M5.T1; `tests/test_cluster_labels.py`.
- **Acceptance criteria:**
  1. **Oracle majority vote:** purity per cluster, ARI, AMI, homogeneity, completeness,
     noise fraction, cluster count.
  2. **Seed labelling:** `k` in {10, 20, 40}, 20 draws averaged, reporting accuracy, FPR and
     recall. Clusters with no seed and all noise points are labelled legitimate.
  3. Ties and noise handling per OQ-8(b, c). Seed draws derive from `SEED`.
- **Tests:** hand-built label arrays:
  - cluster with no seed and noise points fail open;
  - all noise gives FPR 0 and recall 0 (11.17);
  - many clusters per class;
  - tie cases per OQ-8(b);
  - same seed gives the same draws.
- **Depends on:** M5.T1.
- **Status:** blocked (OQ-8)

### M5.T3: `bqc cluster`

- **Goal:** run the experiment from the command line and write its JSON.
- **Design refs:** 9, 10.
- **Files:** `bqc/cli.py`, `tests/test_cli_cluster.py`.
- **Acceptance criteria:** `python -m bqc cluster` writes a separate JSON file to `results/`
  with the same config fields and no-overwrite rule as M3.T8, and shares no state with
  `evaluate`. A missing raw file exits non-zero with a message pointing at `fetch`.
- **Tests:** tiny config end to end; report round-trips through `json`; missing raw file;
  two runs identical apart from the timestamp (11.18).
- **Depends on:** M3.T8, M5.T2.
- **Status:** todo

---

## M6: Full run

Sign-off: pending.

### M6.T1: Run everything on real data and report

- **Goal:** produce the v1 numbers and hand them to `fable`.
- **Design refs:** 6.2, 6.3, 7, 8.5, 8.6, 9, 13.2, 13.9.
- **Files:** none committed (`results/` is gitignored), apart from this task's status.
- **Acceptance criteria:**
  1. `fetch`, `generate`, `evaluate` (with `--lofo` and the SMOTE comparison) and `cluster`
     all complete on the real data from a clean clone with a fresh `.venv`.
  2. The report to `fable` gives, for each model and SMOTE setting: headline recall with
     achieved FPR and CI; per-family recall including E; LOFO recalls; the trivial baseline
     gap; the paired SMOTE differences; cluster results for both rules, including noise
     fraction.
  3. The false-positive list is included, and reviewed by eye for queries that look
     machine-generated (13.2, 13.9).
  4. The sanity section's state is stated: skipped with its count, or scored. **Skipped is
     an acceptable outcome for `done`.**
  5. If the trivial baseline gets close to the MLP, or LOFO shows generators are too easy,
     this is reported as a finding. The rework is `fable`'s decision (6.3, 8.6), and this
     task doesn't change generators.
- **Tests:** none (no code). The determinism test of M4.T4 covers reproducibility.
- **Depends on:** M2.T8, M4.T7, M5.T3.
- **Status:** todo
