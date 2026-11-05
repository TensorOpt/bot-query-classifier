# PLAN.md: bot-query-classifier

Owner: `opus`. Plans `DESIGN.md` v1.3 (commit `8888a74`). Created 2026-11-05.

Status values: `todo` -> `in progress` -> `done` (sonnet) -> `validated` (opus) ->
`signed off` (fable, per milestone). `blocked` means the task can't start until `fable`
settles the design point stated in the task. A task whose dependency is blocked stays
`todo`.

Test command, everywhere: `.venv/bin/python -m unittest discover -s tests`.

## Overview

| Milestone | Content | Tasks |
|-----------|---------|-------|
| M1 | Sanity set, scaffold, preprocessing, WANDS, sanity loader | 6 |
| M2 | Synthetic generators, audit, `generate` | 8 |
| M3 | Features and evaluation harness against a fake `Scorer` | 10 |
| M4 | Real models, `evaluate` command, SMOTE experiment | 6 |
| M5 | HDBSCAN experiment | 3 |
| M6 | Full run on real data | 1 |

Task ids stay stable: `M1.T7`, `M3.T6` and `M4.T5` were probe tasks, are retired, and
aren't reused.

### Order constraints

1. **M1.T1 is the first task of the project.** `data/sanity.csv` must be committed
   before anything exists under `bqc/synth/` (8.6.1). The first `bqc/synth/` file
   appears in M2.T1, and M2.T1's acceptance checks the order in git history.
2. Generators (M2) come before the harness (M3), and the harness runs against a fake
   `Scorer` before any real model exists (M4).

### Probes

No probe task is scheduled: section 13 already answers every question a task depends on.

| Question | Finding | Used by |
|----------|---------|---------|
| `StratifiedGroupKFold` with 6 templates x 20 per family, outer and inner | 13.10 | M2.T1, M3.T3, M3.T5 |
| Family presence per fold side, with the real seeds | 13.14 | M3.T3, M3.T5, M3.T7, M3.T9 |
| `normalise()` order, idempotence and cost | 13.11 | M1.T3 |
| `imblearn` `Pipeline` skips SMOTE at `predict_proba` time; callable `sampling_strategy` | 13.12 | M4.T6 |
| Wilson reference values | 13.13 | M3.T7 |

13.9 keeps one item. "No bot traffic in the 480 queries" can't be probed; its
mitigation, reviewing the false-positive list by eye, is an acceptance criterion of
M6.T1.

### File ownership

Each file in section 10's tree is created by exactly one task. Any other task listing it
changes it.

| Path | Created by |
|------|------------|
| `data/sanity.csv` | M1.T1 |
| `pyproject.toml`, `bqc/__init__.py`, `bqc/__main__.py`, `bqc/cli.py` | M1.T2 |
| `bqc/preprocess.py` | M1.T3 |
| `bqc/wands.py` | M1.T4 |
| `data/CHECKSUMS` | M1.T5 |
| `bqc/sanity.py` | M1.T6 |
| `bqc/synth/__init__.py`, `bqc/synth/registry.py` | M2.T1 |
| `bqc/synth/gibberish.py`, `payloads.py`, `enumeration.py`, `spam.py`, `offtopic.py` | M2.T2 to M2.T6 |
| `bqc/synth/audit.py` | M2.T7 |
| `bqc/report.py` | M2.T8 |
| `bqc/features.py` | M3.T1 |
| `bqc/models.py` | M3.T2 |
| `bqc/evaluate.py` | M3.T3 |
| `bqc/cluster.py` | M5.T1 |

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

Q10 (12.2) stays open and is non-blocking. The plan follows the design's position, so
off-domain queries are outside both classes and no task exists for them.

### Fact checked while planning

All 480 WANDS queries remain distinct after the five `normalise()` steps of section 5
(2026-11-05, Python 3.12, `query.csv` SHA-256 `63b61660...68e95`), so 8.1's singleton
groups hold.

---

## M1: Foundation

Sign-off: pending.

### M1.T1: Author the unlabelled sanity set

- **Goal:** commit 50 unlabelled queries in `data/sanity.csv`, written before any
  generator exists.
- **Design refs:** 8.6.1 only (it defines this task's restricted brief).
- **Files:** `data/sanity.csv`. No other file is in the commit.
- **Delegation (restricted brief, required by 8.6.1):**
  - `fable` pastes this task entry and the text of 8.6.1 into the delegation, and adds
    `SEED = 20261105` and the WANDS URL
    `https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv`. The brief
    holds nothing else from `DESIGN.md` or `PLAN.md`.
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

- **Goal:** create an installable package with a command entry point and a working test
  command.
- **Design refs:** 6 (dependencies), 10.
- **Files:** `pyproject.toml`, `bqc/__init__.py` (`SEED = 20261105`), `bqc/__main__.py`,
  `bqc/cli.py` (argument parser with no commands yet), `tests/test_seed.py`,
  `tests/test_cli.py`, `.gitignore` (add `data/raw/`, `results/`).
- **Acceptance criteria:**
  1. `python3 -m venv .venv && .venv/bin/pip install -e .` succeeds on Python 3.12.
  2. Dependencies are exactly `numpy`, `scikit-learn>=1.3`, `imbalanced-learn`, each
     justified in the commit message.
  3. The test command runs and passes.
  4. `git check-ignore data/raw/x results/x scratch/x .venv` matches all four, and
     `data/sanity.csv` isn't ignored.
  5. `python -m bqc` with no command prints usage and exits non-zero.
  6. The report states the installed versions of numpy, scikit-learn and imbalanced-learn
     and flags any that differ from those used by the section 13 probes (scikit-learn
     1.9.1, numpy 2.5.4, imbalanced-learn 0.14.2).
- **Tests:** `SEED == 20261105`; an unknown command exits non-zero.
- **Depends on:** M1.T1 (schedule order only).
- **Status:** todo

### M1.T3: `normalise()`

- **Goal:** one function holds all query preprocessing.
- **Design refs:** 5, 11.1, 11.2, 11.4, 11.5, 13.11.
- **Files:** `bqc/preprocess.py`, `tests/test_preprocess.py`.
- **Acceptance criteria:**
  1. `normalise(str) -> str` is pure and applies the five steps of section 5, in order:
     NFKC; remove `Cc` and `Cf` characters unless `str.isspace()` is true for them;
     `" ".join(s.split())`; `str.lower()`; truncate to 512 code points, then `rstrip()`.
  2. Truncation is silent: nothing logs or reports it (section 5, 11.2).
  3. It is idempotent, and it is the only preprocessing implementation in the package.
- **Tests:**
  - Empty and whitespace-only input return `""` without raising (11.1).
  - `\t` and `\n` become a space: `a\tb` gives `a b`.
  - U+200B, U+FEFF and a non-whitespace control character such as `\x00` are removed,
    joining their neighbours: `so\u200bfa` gives `sofa`.
  - Full-width ASCII maps to ASCII; combining marks, emoji and non-ASCII letters are kept
    (11.4).
  - Results of 511, 512 and 513 code points, and a 1 MB input, give at most 512 code points
    with no trailing space, including a cut that lands inside a space (11.2).
  - An input that NFKC lengthens past 512 (`\ufb01` x 300) is truncated to 512.
  - Mixed case (11.5).
  - Idempotence over all the above.
- **Depends on:** M1.T2.
- **Status:** todo

### M1.T4: WANDS loader with hash check

- **Goal:** load `query.csv`, refusing a file whose hash doesn't match.
- **Design refs:** 3, 10, 11.14, 11.15, 13.8.
- **Files:** `bqc/wands.py`, `tests/test_wands.py`, `tests/fixtures/query.tsv`,
  `tests/fixtures/CHECKSUMS`.
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
- **Files:** `bqc/wands.py`, `bqc/cli.py`, `data/CHECKSUMS`, `tests/test_fetch.py`.
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
- **Design refs:** 8.6.1, 10, 11.2, 11.20, 11.21.
- **Files:** `bqc/sanity.py`, `tests/test_sanity_loader.py`, `tests/fixtures/sanity_*.csv`.
- **Acceptance criteria:**
  1. Each `label` cell is stripped of leading and trailing whitespace and nothing else, then
     must be exactly `0` or `1`. So ` 1` is a valid `1`, and `2`, `yes`, `1.0` are not.
  2. Returns either all rows with integer labels, or a "not scorable" result carrying the
     count of invalid rows and the message `sanity set: N of M rows unlabelled, skipping`.
     Blank is never a class, and a partially labelled file is never scored.
  3. A query equal to a WANDS query after `normalise()` raises an error naming the `id`,
     regardless of the labels' state.
  4. A wrong header raises an error naming the file.
  5. The loader doesn't write to the file.
- **Tests:**
  - All blank; all labelled; partly labelled; whitespace-only labels count as blank
    (11.20).
  - ` 1` and `0 ` are valid; `2`, `yes` and `1.0` make the file not scorable (11.20).
  - A row equal to a WANDS fixture query after case and whitespace differences (11.21).
  - Quoted queries containing commas; CRLF line endings.
  - **The real `data/sanity.csv`**: header, 50 rows, ids `1..50`, no duplicate queries,
    every query at most 512 code points after `normalise()` (11.2). This test must pass both blank and fully labelled, so it asserts nothing about label
    values.
- **Depends on:** M1.T1, M1.T3.
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
  4. The registry enforces section 4 rule 6: each family, E included, has exactly 6
     templates with 20 samples each, at least 2 of them hard-positive. A registry that breaks
     this raises `ValueError`.
  5. Group sizes can be read without generating text (M3.T3 needs them before vocabulary is
     known).
  6. An empty vocabulary raises `ValueError`.
- **Tests** (with stub templates; they also run against every real template once
  registered, so M2.T2-T6 inherit them):
  - Output lowercased and ASCII-only (section 4 rule 1).
  - Same seed gives identical output, including across two subprocesses with different
    `PYTHONHASHSEED`; a different seed gives different output (11.18).
  - Hard-positive templates use the vocabulary: a sentinel word appears in the output, and a
    word outside the given vocabulary doesn't (rules 3, 8.3).
  - Each template reaches at least 50 distinct strings with a fold-sized vocabulary
    (rule 5).
  - Every string is at most 512 code points after `normalise()` (rule 7).
  - Every sample has its metadata.
  - A family with 5 templates, with a quota other than 20, or with one hard-positive
    template raises (rule 6).
- **Depends on:** M1.T1, M1.T3.
- **Status:** todo

### M2.T2: Family A, gibberish

- **Goal:** generate keyboard mashing, random IDs and hash-like strings, plus hard
  positives.
- **Design refs:** 2 (family A, hard positives), 4.
- **Files:** `bqc/synth/gibberish.py`, `tests/test_synth_gibberish.py`.
- **Acceptance criteria:** Exactly 6 templates, at least 2 hard-positive (rule 6). Covers mashing, repeated characters and
  UUID/hash-like strings, with hard positives such as `accent chair asdfgh`. Registered with
  family `A`. Produces no typo-style positives (section 2).
- **Tests:** registry tests pass for this family; template-specific shape checks (for
  example, a UUID template has UUID shape); output isn't a perturbed product word.
- **Depends on:** M2.T1.
- **Status:** todo

### M2.T3: Family B, injection and probing payloads

- **Goal:** generate scanner payloads, plus hard positives.
- **Design refs:** 2 (family B, hard positives), 4.
- **Files:** `bqc/synth/payloads.py`, `tests/test_synth_payloads.py`.
- **Acceptance criteria:** Exactly 6 templates, at least 2 hard-positive (rule 6). Covers SQL, script, template, path-traversal and
  NoSQL patterns, with hard positives such as `leather sofa' union select`.
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
- **Acceptance criteria:** Exactly 6 templates, at least 2 hard-positive (rule 6). Covers bare SKUs, walks, paths and
  `page=2&sort=price` forms, with hard positives. Registered with family `C`.
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
- **Acceptance criteria:** Exactly 6 templates, at least 2 hard-positive (rule 6). Covers each listed kind, with hard positives such
  as `cheap area rug http://x.co`. Registered with family `D`.
- **Tests:** registry tests pass for this family; each kind appears in the output.
- **Depends on:** M2.T1.
- **Status:** todo

### M2.T6: Family E, off-topic natural language (evaluation-only)

- **Goal:** generate 120 questions, chat prompts and code snippets, tagged `E`.
- **Design refs:** 2 (family E, 2.1), 4.
- **Files:** `bqc/synth/offtopic.py`, `tests/test_synth_offtopic.py`.
- **Acceptance criteria:**
  1. Generated by the same machinery, with `family="E"`, exactly 6 templates of 20 and at
     least 2 hard-positive (rule 6).
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
     strings. This check is callable on its own, because it runs on every generation,
     including the per-fold ones (rule 4, M3.T3).
  2. Every positive-negative pair with character-3-gram Jaccard above 0.6 is reported. This
     report runs only in `bqc generate` (rule 4).
  3. Length statistics are returned per family, alongside those of WANDS.
- **Tests:**
  - Identical pair; pair differing only in case and whitespace.
  - Jaccard exactly 0.6 (not reported) and just above it (reported).
  - Strings shorter than 3 characters, such as family C's `a`, don't raise.
- **Depends on:** M1.T3, M2.T1.
- **Status:** todo

### M2.T8: `bqc generate`

- **Goal:** generate the full-vocabulary positives from the command line and write the
  report for inspection by eye.
- **Design refs:** 4, 10.
- **Files:** `bqc/report.py`, `bqc/cli.py`, `tests/test_report.py`,
  `tests/test_cli_generate.py`.
- **Acceptance criteria:**
  1. `python -m bqc generate` uses vocabulary from all 480 negatives to generate the 480
     training positives of A-D and the 120 of family E, each tagged with its family, and runs
     the audit over all 600. It writes one JSON report to `results/` holding the samples with
     their metadata, the per-family length statistics next to WANDS', and the near-duplicate
     pairs (section 10).
  2. `write_report()` in `bqc/report.py` is the one home for writing results: it adds the
     config fields of section 10, names the file by config hash and timestamp, and refuses to
     overwrite. M3.T8 and M5.T3 import it from there.
  3. An audit failure exits non-zero, and a missing `data/raw/query.csv` exits non-zero
     with a message pointing at `fetch`.
  4. The real run passes the audit, and its near-duplicate count goes in the task report.
- **Tests:** on fixture paths:
  - Success writes a report with samples and `family`, `template_id`, `seed` on each.
  - Length statistics and near-duplicate pairs are present.
  - `write_report`: config fields present; a name collision doesn't overwrite.
  - Collision with a negative exits non-zero, for a family E sample too.
  - Missing raw file exits non-zero.
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
- **Design refs:** 4 (rule 4), 8.1, 8.3, 10 (seeds), 11.8, 11.16, 13.10, 13.14.
- **Files:** `bqc/evaluate.py`, `tests/test_folds.py`.
- **Acceptance criteria:**
  1. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + repeat_index)`,
     3 repeats, with both parameters overridable for tiny test configs.
  2. Groups: each query is its own group; each positive's group is `(family, template_id)`.
  3. Training hard positives draw only on training-fold negatives, and test hard positives
     only on test-fold negatives.
  4. Family E is generated from test-fold vocabulary for scoring, and never appears in a
     training set.
  5. Every per-fold generation runs the exact-match check of M2.T7 (rule 4).
  6. A single fold-check function raises if either class or any training family (A-D) is
     missing from a training fold, or if either class is missing from a test fold. Families
     are never asserted on the test side (8.1, 11.16). M3.T5 and M3.T9 use it too.
  7. Each fold's class and per-family counts are returned for the run JSON (8.1). The positive
     share isn't assumed to be 0.50: about 0.45-0.51 is expected (13.10).
- **Tests:**
  - No group on both sides of a fold (11.8).
  - A sentinel word present only in a test-fold negative never appears in a training
    positive, and the reverse (8.3).
  - No E sample is in a training set.
  - Same seed gives the same folds; different repeats give different folds.
  - A class missing from a training or a test fold raises (11.16).
  - A training fold missing a family raises (11.16).
  - A family present in training but absent from a test fold doesn't raise, and its test count
    is returned as 0 (8.1, 13.14).
  - A per-fold collision with a negative raises (rule 4).
- **Depends on:** M2.T1, M2.T7. Real families aren't required, because the tests use stub
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
- **Design refs:** 8.2, 8.5 (family E), 11.9, 11.12, 11.16, 13.10.
- **Files:** `bqc/evaluate.py`, `tests/test_runner.py`.
- **Acceptance criteria:**
  1. Depends only on the `Scorer` port, via a factory; imports no sklearn estimator.
  2. Inner folds are `StratifiedGroupKFold(n_splits=5, shuffle=True,
     random_state=SEED + repeat_index)` over the outer training rows, with the same groups.
     Positives aren't regenerated per inner fold (8.2).
  3. Scorer fits see only outer-training rows, and the threshold comes only from inner
     scores.
  4. Family E is scored by each fold's model at that fold's threshold.
  5. Warnings raised during `fit` (11.12) are captured per fold and returned, and don't
     raise.
  6. Every inner training fold passes the M3.T3 fold check (both classes and all training
     families, 11.16).
- **Tests** (with the fake):
  - No test-fold or E text in any recorded fit input (11.9).
  - No group on both sides of an inner fold.
  - Inner fits see only the outer fold's positives, unchanged: generation runs once per
    outer fold (8.2).
  - An inner training fold missing a family raises (11.16).
  - Number of fits: outer folds x (inner folds + 1).
  - Changing test-fold scores doesn't change the threshold.
  - The test-fold row count is unchanged at scoring.
  - A fake emitting `ConvergenceWarning` is captured, not raised (11.12).
- **Depends on:** M3.T2, M3.T3, M3.T4.
- **Status:** todo

### M3.T7: Metrics and aggregation

- **Goal:** compute every number in 8.5 from fold results.
- **Design refs:** 8.1, 8.4, 8.5, 13.13, 13.14.
- **Files:** `bqc/evaluate.py`, `tests/test_metrics.py`.
- **Acceptance criteria:**
  1. Per repeat and pooled: headline recall and achieved FPR, Wilson 95% CI (standard
     library `math` only), confusion matrix, macro F1, precision, recall, ROC-AUC as mean and
     sd over folds and pooled.
  2. **Intervals (8.5):** a repeat's interval uses `k_r` at `n = 480`. The pooled point
     estimate is the total over all repeats divided by 1440, and the pooled interval uses
     `k = total / 3` at `n = 480`; `k` need not be an integer. Never `n = 1440`. The same
     rule applies to a recall interval, if one is reported. (`n` and the repeat count come
     from the run, so tiny configs work.)
  3. Per-family recall is caught / total, pooled over all test folds of a repeat and over all
     repeats, never a mean of per-fold values (8.5). For A-D that is 120 per repeat and 360
     overall. E is scored in every fold, so its total is 120 per fold. E is in its own row,
     marked evaluation-only, and never pooled into headline recall.
  4. The per-fold breakdown records each family's test count and caught count, with recall
     `null` where the count is 0. A family with zero test samples in a fold is normal (13.14)
     and doesn't raise or divide by zero.
  5. The full list of false positives, with their query text.
  6. No metric feeds any selection.
- **Tests:**
  - Wilson against the 13.13 values, to 5 decimals: `0/480`, `5/480`, `1/96`, `0/30`,
    `480/480`, including `k = 0` and `k = n`.
  - Pooled total 15 over 3 repeats gives point 15/1440 and interval `[0.00446, 0.02415]`,
    not the `15/1440` row. Pooled total 0 gives upper bound 0.00794.
  - Non-integer `k` (total 16) gives bounds between those of `5/480` and `6/480`.
  - Hand-computed confusion matrix and pooled per-family recall.
  - A fold where a family has zero test samples: count 0, caught 0, recall `null`, and the
    pooled recall is unaffected by it.
  - Folds of unequal family counts give caught / total, which differs from the mean of the
    per-fold recalls.
  - Adding E samples leaves headline recall unchanged.
  - The FP list holds exactly the expected queries.
  - A single-class input to ROC-AUC raises rather than reporting a number.
- **Depends on:** M3.T5.
- **Status:** todo

### M3.T8: Run report JSON

- **Goal:** build one reproducible JSON report per run, written through `write_report`.
- **Design refs:** 8.1, 10, 11.12, 11.18.
- **Files:** `bqc/evaluate.py`, `tests/test_run_report.py`.
- **Acceptance criteria:**
  1. Contains config (all parameters, seed, git commit hash, library versions), per-fold
     and pooled metrics, each fold's class and per-family counts, FP list, per-family recall,
     `target_unmet` and warning fields, and per-fold convergence warnings.
  2. Written by `write_report` imported from `bqc/report.py`; there is no second writer, and
     `bqc/evaluate.py` doesn't import `bqc/cli.py` (section 10).
- **Tests:**
  - Two tiny-config runs give identical JSON apart from the timestamp (11.18).
  - Output round-trips through `json` (no numpy types left).
  - Fold class and per-family counts are present.
- **Depends on:** M2.T8, M3.T7.
- **Status:** todo

### M3.T9: Leave-one-family-out

- **Goal:** report recall and achieved FPR on each family at a threshold chosen without it.
- **Design refs:** 8.6 (item 1), 11.16.
- **Files:** `bqc/evaluate.py`, `tests/test_lofo.py`.
- **Acceptance criteria:**
  1. For each F in A-D (E has no LOFO row): reuse the outer folds of 8.1, drop F's groups from
     the training side, and apply the same nested protocol, so F is absent from every fit and
     from threshold selection. Test on all of F, generated from test-fold vocabulary, plus
     the test-fold negatives. No re-balancing.
  2. Each row reports recall on F and the achieved FPR at that threshold.
  3. Results go in the run JSON as their own section, separate from the headline.
  4. Off by default, enabled by a parameter (`--lofo` is wired in M4.T4).
  5. The fold check asserts the other three families and both classes (11.16).
- **Tests** (with the fake):
  - No F text in any fit input.
  - The test side holds all of F plus test-fold negatives only.
  - One row per family A-D, none for E, each with recall and achieved FPR.
  - A hand-computed achieved FPR.
  - A training side missing one of the other three families raises.
- **Depends on:** M3.T8.
- **Status:** todo

### M3.T10: Final models

- **Goal:** fit one model per configuration on all 960 samples, and pickle it.
- **Design refs:** 8.2, 8.7, 10.
- **Files:** `bqc/evaluate.py`, `tests/test_final_model.py`.
- **Acceptance criteria:**
  1. The threshold comes from the M3.T5 inner procedure applied to all 960 rows, as 8.7 names
     it: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)` with the groups of
     8.1, out-of-fold scores, then the threshold rule of 8.4. The model is then fitted on all
     960.
  2. Positives are generated once from the vocabulary of all 480 negatives, and family E is
     excluded.
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

### M4.T6: SMOTE option in the adapters

- **Goal:** let the LR and MLP adapters run `vectoriser -> SMOTE -> estimator`.
- **Design refs:** 6.4, 7, 7.1, 13.4, 13.12, 11.9.
- **Files:** `bqc/models.py`, `tests/test_models_smote.py`.
- **Acceptance criteria:**
  1. Off by default. When on, it uses `imblearn.pipeline.Pipeline` with `random_state` from
     the seed and default `k_neighbors`.
  2. `sampling_strategy` is a callable returning `{c: 2 * count_c}` for each class in the `y`
     passed to `fit` (7.1). It is a module-level function, so the scorer can be pickled
     (M3.T10).
  3. Applied to training data only.
  4. `imblearn` is imported only in this code path; stages don't import it.
- **Tests:**
  - The test-fold row count is unchanged after the pipeline (7, 11.9).
  - The estimator saw exactly twice the training rows, at two different input sizes, and
    with unequal class counts (7.1).
  - Same seed gives identical scores.
  - A fitted SMOTE scorer survives a pickle round trip with identical scores.
  - With SMOTE off, scores equal those of the adapter before this task.
- **Depends on:** M4.T1, M4.T2.
- **Status:** todo

### M4.T7: SMOTE comparison and `--smote`

- **Goal:** run both arms in one run, and report the paired comparison and verdict.
- **Design refs:** 7.1, 8.5, 8.7, 10.
- **Files:** `bqc/evaluate.py`, `bqc/cli.py`, `tests/test_smote_comparison.py`.
- **Acceptance criteria:**
  1. `python -m bqc evaluate --smote` runs, in one run and one JSON file, the three models
     without SMOTE plus the MLP and LR with SMOTE. The trivial baseline has no SMOTE arm.
     Every arm's metrics sit under its own configuration key.
  2. Both arms come from one fold construction, so folds, seeds and generated positives are
     identical.
  3. A `smote_comparison` section gives, for MLP and LR separately: the per-fold paired
     differences in headline recall (SMOTE minus no-SMOTE, 15 values), their mean and sample
     standard deviation (`ddof=1`, `statistics.stdev`), the number of folds where SMOTE wins, and the paired difference in achieved
     FPR.
  4. `smote_justified` per model is true only if the mean paired recall difference is positive
     and exceeds twice `sd / sqrt(15)`, and the mean paired FPR difference isn't positive.
  5. Without `--smote` there is no SMOTE arm and no comparison section.
  6. Final models and sanity scoring cover the SMOTE configurations too (8.7). They pickle
     because the `sampling_strategy` callable of M4.T6 is a module-level function (7.1).
- **Tests** (with fakes):
  - Identical folds and texts across arms.
  - Hand-computed paired differences, mean, sample sd and win count; a case where `ddof=0`
    would flip `smote_justified`.
  - `smote_justified`: true case; mean exactly at the 2-standard-error bound (false); positive
    mean with positive FPR difference (false); zero FPR difference (allowed); negative mean
    (false).
  - No trivial-baseline SMOTE arm.
  - A default run has no SMOTE arm and no `smote_comparison` key.
- **Depends on:** M4.T4, M4.T6.
- **Status:** todo

---

## M5: HDBSCAN experiment

Sign-off: pending.

### M5.T1: Clustering pipeline and grid

- **Goal:** TF-IDF, SVD, L2 normalisation and HDBSCAN on given strings with labels hidden.
- **Design refs:** 9, 9.1, 11.17, 13.5.
- **Files:** `bqc/cluster.py`, `tests/test_cluster.py`.
- **Acceptance criteria:**
  1. Uses the M3.T1 vectoriser; `TruncatedSVD(random_state=SEED)`; `copy` set explicitly on
     `HDBSCAN` (13.5).
  2. Defaults `n_components=50`, `min_cluster_size=15`; the grid of 9.1 (`n_components`
     in {20, 50, 100}, `min_cluster_size` in {10, 15, 30}) is run and reported.
  3. Labels aren't passed to any clustering step. The module takes texts as input and shares
     only preprocessing and the vectoriser factory with the training pipeline (section 9).
- **Tests:**
  - On tiny inputs: rows are unit-norm after normalising.
  - Same seed gives identical labels.
  - An all-noise result returns 0 clusters and 100% noise without raising (11.17).
  - The module doesn't import `bqc.evaluate` or `bqc.models`.
- **Depends on:** M3.T1.
- **Status:** todo

### M5.T2: Cluster labelling rules and metrics

- **Goal:** assign classes to clusters by both rules of 9.2, and compute their metrics.
- **Design refs:** 9.2, 11.17.
- **Files:** `bqc/cluster.py`, `tests/test_cluster_labels.py`.
- **Acceptance criteria:**
  1. **Oracle majority vote:** purity per cluster, ARI, AMI, homogeneity, completeness,
     noise fraction, cluster count.
  2. **Seed labelling:** `k` in {10, 20, 40}, 20 draws averaged, each using
     `random.Random(SEED + draw_index)` over all rows, including noise points. Clusters with no
     seed and all noise points are labelled legitimate.
  3. A tie in either rule resolves to legitimate.
  4. ARI, AMI, homogeneity and completeness are computed on non-noise points only, with the
     noise fraction next to them. Seed-rule accuracy, FPR and recall are computed over all
     rows, with noise labelled legitimate.
  5. An all-noise result reports these metrics as `null` with `"non_noise": 0`, and seed-rule
     FPR 0 and recall 0 (11.17).
- **Tests:** hand-built label arrays:
  - Cluster with no seed and noise points fail open.
  - Seed tie 1-1 and half-and-half oracle cluster both give legitimate.
  - Moving noise rows' true labels doesn't change ARI, and does change seed-rule recall.
  - All noise gives `null`, `"non_noise": 0`, FPR 0, recall 0 (11.17).
  - Many clusters per class.
  - A seed drawn on a noise point counts as a draw but labels nothing.
  - Same `draw_index` gives the same draw; different ones differ.
- **Depends on:** M5.T1.
- **Status:** todo

### M5.T3: `bqc cluster`

- **Goal:** run the experiment from the command line and write its JSON.
- **Design refs:** 9, 10.
- **Files:** `bqc/cluster.py`, `bqc/cli.py`, `tests/test_cli_cluster.py`.
- **Acceptance criteria:** `python -m bqc cluster` builds the 960 strings (WANDS plus positives
  from all-negative vocabulary, no family E) and writes a separate JSON file with
  `write_report` imported from `bqc/report.py`; `bqc/cluster.py` doesn't import `bqc/cli.py`
  (section 10). It shares no state with `evaluate`. A missing raw file exits non-zero with a
  message pointing at `fetch`.
- **Tests:** tiny config end to end; report round-trips through `json`; missing raw file;
  two runs identical apart from the timestamp (11.18).
- **Depends on:** M2.T8, M5.T2.
- **Status:** todo

---

## M6: Full run

Sign-off: pending.

### M6.T1: Run everything on real data and report

- **Goal:** produce the v1 numbers and hand them to `fable`, who records them in the M6
  sign-off (12.3).
- **Design refs:** 6.2, 6.3, 7, 8.5, 8.6, 9, 13.2, 13.9.
- **Files:** none committed (`results/` is gitignored), apart from this task's status.
- **Acceptance criteria:**
  1. `fetch`, `generate`, `evaluate` (with `--lofo` and the SMOTE comparison) and `cluster`
     all complete on the real data from a clean clone with a fresh `.venv`.
  2. The report to `fable` gives, for each model and SMOTE setting: headline recall with
     achieved FPR and CI; per-family recall including E; LOFO recall and achieved FPR; the
     trivial baseline gap; `smote_comparison` and `smote_justified`; cluster results for both
     rules, including noise fraction.
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
