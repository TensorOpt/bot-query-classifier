# PLAN.md: bot-query-classifier

Owner: `opus`. Plans `DESIGN.md` v2.7 (commit `b025fde`) for M1-M6 and v2.8 (`cafa701`, rulings PQ-7 to PQ-9 in `cbc2169`) for M7.
Created 2026-11-05.

Status values: `todo` -> `in progress` -> `done` (sonnet) -> `validated` (opus) ->
`signed off` (fable, per milestone). `blocked (PQ-n)` means the task can't start until `fable`
answers that question (see "Open questions for `fable`"). A task whose dependency is blocked
stays `todo`.

Test command, everywhere: `.venv/bin/python -m unittest discover -s tests`. `python -m unittest -v tests.test_x`
fails on import because tests import each other as top-level modules; only the `discover -s tests` form works.

## Overview

| Milestone | Content | Tasks |
|-----------|---------|-------|
| M1 | Sanity set, scaffold, preprocessing, WANDS, sanity loader, fetch hardening, product file | 8 |
| M2 | Synthetic generators, audit, part-number set, render helper, product-name spans, `generate` | 13 |
| M3 | Features and evaluation harness against a fake `Scorer`, both arms, catalogue-gap check, query classes in the loader, SMOTE dependency removal | 15 |
| M4 | Real models, `evaluate` command, paired comparisons | 5 |
| M5 | HDBSCAN experiment (unchanged by v2.0) | 3 |
| M6 | Full run on real data | 1 |
| M7 | Part-number hard negatives (3.3), `lr_hand` adapter (6.5), `model_comparison_hand`, re-run (v2.8) | 7 |

Task ids stay stable: `M3.T6` and `M4.T5` were probe tasks, and `M4.T6` and `M4.T7` were the SMOTE tasks
(dropped by the user, 12.1 item 17). All four are retired and aren't reused.
`M1.T7` was also a dropped probe; the id now names the fetch fix below, and the old probe
appears only in earlier versions of this plan. Tasks run in file order, which isn't always id
order: M2.T9, M2.T10, M2.T11 and M2.T12 come before M2.T8, M3.T13 comes after M3.T5, M3.T15 comes
after M3.T13, M3.T14 comes after M3.T9, and M3.T16 comes after M3.T14. M7 runs in id order.

v2.0 doesn't reopen validated work: changes to validated code are new tasks (M1.T8, M2.T10,
M2.T11, M3.T16), and earlier entries keep their statuses and the design version they were validated against.

**Next task:** M7.T6. M7.T1 to M7.T5 are validated; M7.T6 and M7.T7 follow
in id order (PQ-7 to PQ-9 are ruled, `DESIGN.md` `cbc2169`).

### Order constraints

1. **M1.T1 is the first task of the project.** `data/sanity.csv` must be committed
   before anything exists under `bqc/synth/` (8.6.1). The first `bqc/synth/` file
   appears in M2.T1, and M2.T1's acceptance checks the order in git history.
2. Generators (M2) come before the harness (M3), and the harness runs against a fake
   `Scorer` before any real model exists (M4).

### Probes

Up to M6 no probe task was needed: section 13 answered every question a task depended on. M7.T1
probes 6.5's paragraph (sparse stack, scaler on training rows, fit time) before M7.T5.

| Question | Finding | Used by |
|----------|---------|---------|
| `StratifiedGroupKFold` with 6 templates x 20 per family, outer and inner | 13.10 | M2.T1, M3.T3, M3.T5 |
| Family presence per fold side, with the real seeds | 13.14 | M3.T3, M3.T5, M3.T7, M3.T9 |
| `normalise()` order, idempotence and cost | 13.11 | M1.T3 |
| Wilson reference values | 13.13 | M3.T7 |
| Part-number shapes in WANDS | 13.15 | M2.T9, M3.T12 |
| `product.csv` format and coverage | 13.18 | M1.T8, M2.T12 |
| Product-name spans as training negatives | 13.19 | M2.T12, M3.T5, M3.T14 |
| Sparse TF-IDF + standardised hand columns into `LogisticRegression` (6.5) | 13.23 (M7.T1) | M7.T5 |

13.9 keeps two open items. "No bot traffic in the 480 queries" can't be probed; its
mitigation, reviewing the false-positive list by eye, is an acceptance criterion of
M6.T1. The MLP fit time at 5,000 / 5,000 is closed by 13.22 (9-29 s, median 17 s), and M4.T4
still measures and records it (section 10). The coverage
numbers of 13.18 were read, not re-run; no task depends on their exact values.

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
| `bqc/synth/partnumbers.py` | M2.T9 |
| `bqc/negatives.py` | M2.T12 |
| `bqc/report.py` | M2.T8 |
| `bqc/features.py` | M3.T1 |
| `bqc/models.py` | M3.T2 |
| `bqc/evaluate.py` | M3.T3 |
| `bqc/cluster.py` | M5.T1 |

M7 creates no file under `bqc/`: it changes `bqc/negatives.py`, `bqc/synth/partnumbers.py`,
`bqc/evaluate.py`, `bqc/models.py` and `bqc/cli.py`. Its one new file is a test module,
`tests/test_models_lr_hand.py` (M7.T5).

### Where the user acts

- **Sanity labels: done.** All 50 rows are labelled (`7332c6b`), so real runs score the sanity
  section. No agent modifies `data/sanity.csv`. A template that can emit a sanity query
  changes; the file doesn't (rule 4).
- **Q12: answered 2026-11-07** (12.1 item 20): part-number hard negatives, M7.T2 to M7.T4.
- **LR or MLP: answered 2026-11-07** (12.1 item 18): logistic regression.
- **`lr_hand` or `lr`: after M7.T7.** M7.T7 carries a recommendation under section 7; the user
  decides (6.5).
- **Catalogue gap: only if the check fails in M6.T1.** What to do then is a user decision
  (8.6 item 5).

12.1 items 1 to 11 and 13 to 17 are closed; embeddings were abandoned for latency (item 15), truncated SVD
on the section 5 features was rejected (item 16), and SMOTE was dropped (item 17). Off-domain queries stay outside both classes, so no task
exists for them. SKUs and part numbers are legitimate (Q11, 2.2).

### Open questions for `fable`

PQ-1 to PQ-4 are answered in `DESIGN.md` 12.3 and PQ-5 in section 10, and all are applied below; their
text is in `530de4f` and `a3cf42e`. PQ-6 is answered in section 7, and its text is in `f0c12cd`.

PQ-7 to PQ-9 (M7, raised 2026-11-07 against `cafa701`) are answered in `DESIGN.md` 3.3 under "Rulings on
the M7 plan" and recorded in 12.3 (`cbc2169`), and are applied to M7.T2 to M7.T4 below; their text is in
`59379e4`. In short: (PQ-7a) the 100 count as negatives wherever the negative count sets the quota, and
the comparison arm's fixed 20 stays in its main run and final model; (PQ-7b) the final models seed
`SEED + PART_NUMBER_SEED_OFFSET`; (PQ-8a) one inner-split group per shape, `("part_number", shape)`;
(PQ-8b) a training positive equal to a part-number negative is dropped and counted; (PQ-8c) drop against
the arm's own training negatives, all 480 real queries, the P strings, the sanity queries and earlier
candidates; (PQ-8d) the comparison arm and the catalogue-gap pass build per fold with the repeat's seed;
(PQ-9) `generate` unchanged, section 10's tree wording fixed by `fable`.

No question is open.

### Facts checked while planning

All 480 WANDS queries remain distinct after the six `normalise()` steps of section 5
(2026-11-05, Python 3.12, `query.csv` SHA-256 `63b61660...68e95`), so 8.1's singleton
groups hold.

A scratch build of 5,000 spans per 3.2 (2026-11-05, the probe's `product.csv`, the committed
generators):

- 0 of 14,400 renders of A-D were all benign against the spans' vocabulary (17,393 occurrences,
  3,763 types). Against the whole catalogue's vocabulary, some `C.walk` and `D.promo` outputs were,
  so the check belongs on the vocabulary a template was given (M2.T8 criterion 8).
- `B.script` reaches only about 60 distinct strings and `B.template` about 146 in 600 renders, so
  their shortfall at `n / 24` renders is real and goes in the report (section 4).
- `near_duplicates` costs about 23 s at 5,000 x 5,000.
- 11.9% of real queries have a span above Jaccard 0.6 at 5,000 spans (27% at 20,000, section 10). No
  test asserts either figure.

Read in the code at `cafa701` while planning M7 (2026-11-07):

- `run_comparison_fold` sets the threshold from rows whose family is `None` (`bqc/evaluate.py:535`), and
  `fit_final_model`'s comparison arm slices the out-of-fold scores with `[:len(negatives)]`
  (`bqc/evaluate.py:1081`). Part-number negatives added with family `None`, or appended to `negatives`,
  would enter the threshold set; M7.T3 and M7.T4 guard both.
- `bqc/synth/partnumbers.py` keeps its shapes in the private `_SHAPES` and seeds each shape with the
  string `f"{seed}/P/{shape_id}"`, so the diagnostic set can't share a stream with an integer seed
  `SEED + 100_000 + r`. M7.T2 makes the table public without changing the diagnostic output.
- `bqc/cluster.py` imports nothing from `bqc.negatives` or `bqc.evaluate`; section 9's exclusion holds
  without a change.

---

## M1: Foundation

**Sign-off: 2026-11-05, `fable`, verdict POSITIVE** (audited at `589c4b7`).

- Suite: `.venv/bin/python -m unittest discover -s tests`, 54 tests, OK, run by `fable`.
- Diff read: `a36823a..589c4b7` for `bqc/`, `tests/`, `pyproject.toml`, `.gitignore`,
  `data/CHECKSUMS` (24 files, 751 insertions). `normalise()` is the six steps of
  DESIGN.md section 5 in the specified order; `fetch` is hash-gated before any write
  and atomic (`<target>.tmp` then `os.replace`); loaders return raw text; the sanity
  loader strips labels and nothing else and normalises both sides only for the WANDS
  duplicate check; `SEED` has one home; dependencies are exactly those of section 6.
- Real-data check by `fable`: `python -m bqc fetch` exits 0 against the pinned hash;
  `load_queries` returns 480 rows, `query_id` 0..487, 480 distinct, the 3 double-space
  queries intact; `load_sanity` on the real file with the real WANDS list is scorable
  (0 invalid rows, no overlap).
- Invariants: pinned hash, atomic write, no secrets, tests never download, YAGNI
  (nothing beyond the six tasks; the `opener` parameter exists for the tests only).
- Probes: all recorded (13.8-13.16); none performed in M1 without a record.
- Process record, noted without penalty: M1.T2 skipped the failing-test stage
  (disclosed; `opus` confirmed by mutation that the tests catch breakage); M1.T3's fix
  commit `7789930` has no failing-run record because the agent was stopped by accident
  before reporting; M1.T5's report has 2 unexplained errors of 9, with test strength
  confirmed by reading. Follow-up F1 below. Early commits carry a wrong co-author line
  (`Claude Opus 5.5`), disclosed to the user, no history rewrite.

Follow-ups:

- **F1 (process, revised at the M2.T13 review):** Every sonnet report pastes, unedited, (a) the failing run: `python -m unittest discover -s tests -v -k <task's tests>`, its per-test lines and summary; and (b) the final run: the summary line of the full `python -m unittest discover -s tests` plus the `-v -k <task's tests>` per-test lines for the task's own tests. Tracebacks are included only for errors other than `NotImplementedError`. Nothing in these blocks is retyped or edited; a report without them, or with an edited block, is rejected by the coordinator before validation. Files are changed with the Edit tool; no `sed -i` or other in-place rewriting of source or test files.
- **F2 (new task M1.T7, `fix(M1.T7)`, `bqc/wands.py`, for `opus` to add; may run in
  parallel with M2):** in `fetch`, (a) catch `http.client.HTTPException` alongside
  `OSError` so a truncated download (`IncompleteRead`) becomes `WandsError` as the
  docstring promises; (b) call the opener with `timeout=60` so a stalled connection
  cannot hang (`urllib.request.urlopen` accepts it; test stubs already take `**k`);
  (c) a test that an opener raising `IncompleteRead` mid-read raises `WandsError` and
  leaves nothing. The `finally` cleanup stays untested: the temp file exists only
  between `write_bytes` and `os.replace`, and the mid-download test covers the path
  that matters.
- **F3 (README, user's file, not a plan task):** the status line "3 of 36 tasks are
  validated" is out of date (M1 signed off, 6 of 36); and the arrow from "Extra
  checks" to the fold metrics is a simplification: the 50 hand-labelled queries and
  the 100 part numbers are scored by the final model (DESIGN.md 8.6.1, 8.6 item 4,
  8.7), while the 120 off-topic queries and the held-out families are scored per
  fold. Everything else in the diagram matches the design.

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
- **Status:** signed off (implemented in `74def5b`; `opus` PASS; status recorded by
  `fable` on 2026-11-05 per the delegation note above). Not verifiable from the
  repository: criterion 2 (the shuffle), because the authoring script was deleted as
  criterion 6 requires; and whether the queries were written from scratch and what
  the delegate read. Mitigation for the latter is now structural: the generator
  audit fails on any output equal to a sanity query (DESIGN.md 4 rule 4, 8.6.1;
  M2.T7, M3.T3).

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
- **Annotation (v2.6):** criteria 2 and 6 stand as validated and signed off. `imbalanced-learn` leaves the
  dependencies in M3.T15, since SMOTE was dropped (12.1 item 17).
- **Tests:** `SEED == 20261105`; an unknown command exits non-zero.
- **Depends on:** M1.T1 (schedule order only).
- **Status:** signed off

### M1.T3: `normalise()`

- **Goal:** one function holds all query preprocessing.
- **Design refs:** `DESIGN.md` `ba6aec5`: 5, 11.1, 11.2, 11.4, 11.5, 13.11, 13.16.
- **Files:** `bqc/preprocess.py`, `tests/test_preprocess.py`.
- **Acceptance criteria:**
  1. `normalise(str) -> str` is pure and applies the six steps of section 5, in this order:
     remove `Cc` and `Cf` characters unless `str.isspace()` is true for them; NFKC;
     `str.lower()`; NFKC again; `" ".join(s.split())`; truncate to 512 code points, then
     `rstrip()`. Validation checks the order by reading the code, because behaviour tests can't
     pin the position of the removal relative to the first NFKC (section 5).
  2. Truncation is silent: nothing logs or reports it (section 5, 11.2).
  3. `normalise(normalise(s)) == normalise(s)` for every case in the test list; the design
     states idempotence as a tested property (section 5, 13.16), not as a guarantee over all
     strings. It is the only preprocessing implementation in the package.
- **Tests:** every case also asserts that a second call leaves the result unchanged.
  - Empty and whitespace-only input return `""` without raising (11.1).
  - `\t` and `\n` become a space: `a\tb` gives `a b`.
  - U+200B, U+FEFF and a non-whitespace control character such as `\x00` are removed,
    joining their neighbours: `so\u200bfa` gives `sofa`.
  - Removal runs before collapsing: `a \u200b b` gives `a b`.
  - A format character between a base and a mark: `e\u200b\u0301` gives U+00E9 on the first call.
  - Hangul jamo split by a format character: `\u1100\u200b\u1161` gives U+AC00.
  - The second NFKC: `\u0391\u0342` gives U+1FB6.
  - Lowercasing runs after the first NFKC: `\u2103` gives `°c`.
  - Full-width ASCII maps to ASCII; `x\u0301`, emoji and non-ASCII letters are kept (11.4).
  - Results of 511, 512 and 513 code points, and a 1 MB input, give at most 512 code points
    with no trailing space, including a cut that lands inside a space (11.2).
  - An input that NFKC lengthens past 512 (`\ufb01` x 300) is truncated to 512.
  - Truncation runs last: `" " * 1000 + "sofa"` and `"\u200b" * 1000 + "sofa"` give `sofa`.
  - Mixed case (11.5).
- **Depends on:** M1.T2.
- **Status:** signed off
- **Status note:** implemented in `1bcb0e3`; `7789930` moved it to the six steps and added the
  behaviour cases; `b73ace9` renamed the test; `a8ba6f6` completed the idempotence samples
  (`b73ace9`'s message wrongly claims that). `opus` requested changes twice: the lists are in
  `9ddddec`/`d6502b2` and `2128700`.

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
- **Status:** signed off

### M1.T5: `bqc fetch` and the real checksum

- **Goal:** download `query.csv` atomically and pin its hash.
- **Design refs:** 3, 10, 11.13.
- **Files:** `bqc/wands.py`, `bqc/cli.py`, `data/CHECKSUMS`, `tests/test_fetch.py`.
- **Acceptance criteria:**
  1. `python -m bqc fetch` downloads with the standard library into `data/raw/query.csv`,
     writing to a temp file and renaming only after the hash check passes.
  2. A network error or hash mismatch exits non-zero and leaves no file at the target or
     temp path.
  3. `data/CHECKSUMS` holds the real SHA-256 from one manual run, as the line
     `<hex>  query.csv` that `load_queries` looks up by file name (M1.T4). The planning check
     saw `63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95`; report any
     difference rather than overwriting it silently.
  4. The real file loads as 480 rows with 480 distinct `query_id` values. The ids run from 0 to
     487 with 8 gaps, not `0..479` as section 3 says (checked 2026-11-05).
- **Tests:** with the URL opener replaced by a local stub, so nothing downloads:
  success; network error; wrong bytes; an existing good file isn't destroyed by a failed
  fetch (11.13); a non-zero return from `main` becomes the exit status of `python -m bqc`
  (run `bqc.__main__` with `runpy` and `main` patched, because in-process tests of `main`
  can't see `__main__` dropping it).
- **Depends on:** M1.T4.
- **Status:** signed off

### M1.T6: Sanity set loader

- **Goal:** read `data/sanity.csv`, and report whether it can be scored.
- **Design refs:** 5, 8.6.1, 10, 11.2, 11.20, 11.21.
- **Files:** `bqc/sanity.py`, `tests/test_sanity_loader.py`, `tests/fixtures/sanity_*.csv`.
- **Acceptance criteria:**
  1. Each `label` cell is stripped of leading and trailing whitespace and nothing else, then
     must be exactly `0` or `1`. So ` 1` is a valid `1`, and `2`, `yes`, `1.0` are not.
  2. Returns the queries raw, in every label state, because the generator audit needs them
     (section 5, 8.6.1, M2.T7). Alongside them, it returns either all integer labels, or a "not scorable" result
     carrying the count of invalid rows and the message
     `sanity set: N of M rows unlabelled, skipping`. Blank is never a class, and a partially
     labelled file is never scored.
  3. A query equal to a WANDS query after `normalise()` raises an error naming the `id`,
     regardless of the labels' state. The loader normalises both sides for this check only.
  4. A wrong header raises an error naming the file.
  5. The loader doesn't write to the file.
- **Tests:**
  - All blank; all labelled; partly labelled; whitespace-only labels count as blank
    (11.20). The queries are returned in all of these.
  - ` 1` and `0 ` are valid; `2`, `yes` and `1.0` make the file not scorable (11.20).
  - A row equal to a WANDS fixture query after case and whitespace differences (11.21).
  - A query with upper case and a double space comes back unchanged.
  - Quoted queries containing commas; CRLF line endings.
  - **The real `data/sanity.csv`**: header, 50 rows, ids `1..50`, no duplicate queries,
    every query at most 512 code points after `normalise()` (11.2). This test must pass both blank and fully labelled, so it asserts nothing about label
    values.
- **Depends on:** M1.T1, M1.T3.
- **Status:** signed off

### M1.T7: Harden `fetch` against truncated and stalled downloads

- **Goal:** make every download failure a `WandsError`, and stop a stalled connection hanging
  (M1 sign-off, follow-up F2). Commit scope: `fix(M1.T7)`.
- **Design refs:** 11.13.
- **Files:** `bqc/wands.py`, `tests/test_fetch.py`.
- **Acceptance criteria:**
  1. `fetch` catches `http.client.HTTPException` alongside `OSError`, so `IncompleteRead`
     becomes `WandsError`, as its docstring promises.
  2. `fetch` calls the opener with `timeout=60`.
- **Tests:**
  - An opener whose response raises `http.client.IncompleteRead` on read raises `WandsError`
    and leaves nothing.
  - The opener receives `timeout=60`.
  The `finally` cleanup stays untested, as accepted at sign-off.
- **Depends on:** M1.T5.
- **Status:** signed off
- **Status note:** implemented in `2eeef65`. `fable`'s light review was positive: the commit is exactly
  follow-up F2, suite 230 OK. M1's verdict stands.

### M1.T8: Download and load `product.csv`

- **Goal:** fetch the product file next to `query.csv`, hash-pinned, and load the three columns the
  project reads.
- **Design refs:** `DESIGN.md` `5ba7cee`: 3, 3.1, 11.13, 11.24, 13.18.
- **Files:** `bqc/wands.py`, `bqc/cli.py`, `data/CHECKSUMS`, `tests/test_products.py`,
  `tests/test_fetch.py`, `tests/fixtures/product.tsv`, `tests/fixtures/CHECKSUMS`.
- **Acceptance criteria:**
  1. `python -m bqc fetch` downloads both files through the existing `fetch`, each atomically and
     checked against its own line. If either fails, it exits non-zero, and any file already in place
     stays.
  2. `data/CHECKSUMS` gains the line `d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6  product.csv`.
  3. `load_products(path, checksums)` returns `(product_id, name, classes)` for every row, raw and in
     file order. `classes` is the `|`-separated `product_class` as a tuple, empty for an empty field.
     No other column is kept.
  4. Hash refused, missing file and wrong header raise `WandsError` naming the path. Columns are
     matched by name, including `category hierarchy` with its space. A row with the wrong column
     count raises `WandsError` naming the row number (11.24).
  5. The loader raises `csv.field_size_limit` before parsing (3.1, 11.24).
  6. The real run gives 42,994 rows, 42,577 distinct names and no error (13.18). The report gives
     the time and peak memory of one load.
- **Tests:** fixture with the real header, tab delimiter, and:
  - a description field longer than the default `csv` limit, which loads;
  - a quoted field and a non-ASCII name, returned raw;
  - an empty `product_class`, and a multi-class one split into a tuple.
  Failure paths: wrong hash; missing file; wrong header; wrong column count names the row number.
  In `tests/test_fetch.py`: the command fetches both files; a failure on the second returns non-zero
  and leaves the first; nothing downloads.
- **Depends on:** M1.T7.
- **Status:** signed off
- **Status note:** implemented in `1b02ff3`. Criterion 6, real load re-run by `opus` on a copy with the
  pinned hash at `data/raw/product.csv`: 42,994 rows, 42,577 distinct names, no error, 0.85 s, peak RSS
  577.5 MB. `fable`'s review: diff read, suite 245 OK. Accepted known issues: the failing run was
  reported as a summary, not verbatim (F1); the real run was closed with a hash-matching copy; and
  `python -m bqc fetch` hasn't completed a real `product.csv` download, so only stubbed tests cover
  criterion 1 (the URL answers with the expected 90,621,131 bytes). M6.T1 exercises it first, from a
  clean clone.

---

## M2: Synthetic generators

**Sign-off: 2026-11-05, `fable`, verdict POSITIVE** (audited at `468a367`; `DESIGN.md` `5654502`).

- Suite: `.venv/bin/python -m unittest discover -s tests`, 336 tests, OK. `fable`'s own `generate` run
  matched (13.20).
- Accepted in the design: the realised span length mix (3.2) and the template shortfalls (section 4).

Follow-ups:

- **F1 (process, revised at the M2.T13 review):** Every sonnet report pastes, unedited, (a) the failing run: `python -m unittest discover -s tests -v -k <task's tests>`, its per-test lines and summary; and (b) the final run: the summary line of the full `python -m unittest discover -s tests` plus the `-v -k <task's tests>` per-test lines for the task's own tests. Tracebacks are included only for errors other than `NotImplementedError`. Nothing in these blocks is retyped or edited; a report without them, or with an edited block, is rejected by the coordinator before validation. Files are changed with the Edit tool; no `sed -i` or other in-place rewriting of source or test files.
  Deviations at M1.T8, M2.T10, M2.T11, M2.T12 and M2.T8's `sed -i` are recorded without penalty.
- **F5:** widen `B.script` and `B.traversal`; task M2.T13, before M6.T1, not a condition of M2.
- **F6:** fix the misplaced comment at `tests/test_synth_audit.py:84`, and add a `token_occurrences` test
  with a cap below the token count, so `bqc/negatives.py:132` is exercised. M3.T3 carries it, as the
  first scheduled task to build on both modules.

### M2.T1: Sample type, registry and quotas

- **Goal:** a seeded, vocabulary-parameterised entry point for generating positives.
- **Design refs:** `DESIGN.md` `3dbe549`: 2, 2.2, 4, 5, 8.3, 10, 11.18, 11.23.
- **Files:** `bqc/synth/__init__.py`, `bqc/synth/registry.py`, `tests/test_synth_registry.py`,
  `tests/synth_stubs.py`.
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
  7. The benign-token predicate, the "at least one non-benign token" check, the letter/digit
     block count and the part-number regex of 2.2 have one home here.
  8. **Wiring:** `tests/test_synth_registry.py` has `IMPLEMENTED_FAMILIES`, empty at this task.
     Each of M2.T2 to M2.T6 imports its module in `bqc/synth/__init__.py` and adds its family
     there, so a family that isn't imported fails the suite.
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
  - `normalise(s) == s` for every template output (section 5).
  - **Every output of every template in every family has at least one non-benign token (2.2,
    11.23),** with a fold-sized vocabulary that includes short WANDS words such as `a`, `x`, `or`
    and number tokens such as `36"` and `3-3/4`.
  - A stub emitting only vocabulary words plus a part number fails. Stubs emitting a non-vocabulary
    letter token, a punctuation-bearing token or a 5-block gibberish token pass. With `a` in the
    vocabulary, `a` fails and `aa` passes.
  - **Every digit-bearing token in family A has at least 4 letter/digit blocks, and family A's
    hard-positive output has no digit (2.2).**
  - Every sample has its metadata.
  - A family with 5 templates, with a quota other than 20, or with one hard-positive
    template raises (rule 6).
- **Depends on:** M1.T1, M1.T3.
- **Status:** signed off
- **Status note:** implemented in `2bbfe9e`, corrected in `bf8a6f5`. `opus`'s review lists are in
  `99ff23d` and `8878e13`.

### M2.T2: Family A, gibberish

- **Goal:** generate keyboard mashing, random IDs and hash-like strings, plus hard
  positives, without producing part-number shapes.
- **Design refs:** `DESIGN.md` `7482987`: 2 (family A, hard positives), 2.2, 4.
- **Files:** `bqc/synth/gibberish.py`, `tests/test_synth_gibberish.py`, `bqc/synth/__init__.py`,
  `tests/test_synth_registry.py` (wiring, M2.T1 criterion 8).
- **Brief:** the delegate's brief contains the family's section 2 row, sections 2.1, 2.2 and 4, and the task entry; nothing from `data/sanity.csv`. The delegate doesn't open that file or
  `tests/fixtures/sanity_*.csv` beyond what its task needs (section 4 rule 8).
- **Acceptance criteria:**
  1. Exactly 6 templates, at least 2 hard-positive (rule 6). Covers mashing, repeated
     characters and UUID/hash-like strings, with hard positives such as
     `accent chair asdfgh`. Registered with family `A`.
  2. Every digit-bearing token has at least 4 letter/digit blocks in every template, by
     construction rather than by chance: random hex must not occasionally produce fewer
     (2.2). So a separate digit run, as a token of its own, isn't produced. Four blocks is the
     design floor; seven is this template's implementation choice (2.2, `0a29571`).
  3. Hard-positive templates use only vocabulary words without digits, because real WANDS
     words such as `18x18` and `w003096809` would break the block rule.
  4. Random letter tokens that serve as the non-benign token can't be vocabulary words: a template
     checks them against its vocabulary, so the contract holds by construction (2.2).
  5. Produces no typo-style positives (section 2).
- **Tests:**
  - Registry tests pass for this family, including the 2.2 assertions.
  - The block rule holds over many seeds for each digit-using template.
  - A vocabulary containing `18x18` and `w003096809` never brings those words into the output.
  - Template-specific shape checks (for example, a UUID template has UUID shape).
  - Output isn't a perturbed product word.
  Also remove `test_is_part_number_query_is_gone` from `tests/test_synth_registry.py`: asserting
  that a function is absent has no future value.
- **Depends on:** M2.T1.
- **Status:** signed off
- **Status note:** implemented in `4265190`, corrected in `0e61958`. The correction also replaced example
  strings in `bqc/synth/registry.py`'s docstring and `tests/test_synth_registry.py`, as listed in
  `d7f95e3`. `opus`'s correction list is in `839923b`.

### M2.T3: Family B, injection and probing payloads

- **Goal:** generate scanner payloads, plus hard positives.
- **Design refs:** `DESIGN.md` `7482987`: 2 (family B, hard positives), 2.2, 4.
- **Files:** `bqc/synth/payloads.py`, `tests/test_synth_payloads.py`, `bqc/synth/__init__.py`,
  `tests/test_synth_registry.py` (wiring, M2.T1 criterion 8).
- **Brief:** the delegate's brief contains the family's section 2 row, sections 2.1, 2.2 and 4, and the task entry; nothing from `data/sanity.csv`. The delegate doesn't open that file or
  `tests/fixtures/sanity_*.csv` beyond what its task needs (section 4 rule 8).
- **Acceptance criteria:** Exactly 6 templates, at least 2 hard-positive (rule 6). Covers SQL, script, template, path-traversal and
  NoSQL patterns, with hard positives such as `leather sofa' union select`.
  Registered with family `B`. Payloads are only ever generated as text, never executed.
- **Tests:** registry tests pass for this family; each pattern kind appears in the
  output; payload characters survive lowercasing.
- **Depends on:** M2.T1.
- **Status:** signed off
- **Status note:** implemented in `b21290a`, corrected in `92dc5da`. `opus`'s correction list is in
  `5915427`.

### M2.T4: Family C, enumeration and scraping

- **Goal:** generate dictionary walks, URL paths and query-string forms that walk the search
  engine, plus hard positives, and no SKUs or part numbers.
- **Design refs:** `DESIGN.md` `0a29571`: 2 (family C, single-word and SKU rows), 2.2, 4, 11.23.
- **Files:** `bqc/synth/enumeration.py`, `tests/test_synth_enumeration.py`, `bqc/synth/__init__.py`,
  `tests/test_synth_registry.py` (wiring, M2.T1 criterion 8).
- **Brief:** the delegate's brief contains the family's section 2 row, sections 2.1, 2.2 and 4, and the task entry; nothing from `data/sanity.csv`. The delegate doesn't open that file or
  `tests/fixtures/sanity_*.csv` beyond what its task needs (section 4 rule 8).
- **Acceptance criteria:** registered with family `C`, with exactly the six templates of 2.2:
  1. dictionary walks (1-2 letters), skipping vocabulary tokens (2.2);
  2. URL paths;
  3. generic query-string fragments with random keys;
  4. pagination and offset sweeps with the usual key names;
  5. hard-positive: product words plus a query-string fragment (`area rug&page=2`);
  6. hard-positive: a search-URL form around product words (`/search?q=accent+chair`).
  Templates 3 and 4 differ in what they vary: 3 varies the keys, 4 the numbers.
- **Tests:**
  - Registry tests pass for this family.
  - **No output matches the part-number regex or is a bare number (2.2)**, including URL paths
    such as `/123`, which the regex matches.
  - Every dictionary-walk output is 1-2 letters, so `zen`, `owl` and digits can never appear,
    and the walk template still reaches 50 distinct strings.
  - With `a` and `x` in the vocabulary, the walk never emits them.
  Also, in `tests/test_synth_registry.py`: `assert_template_rules` accepts a vocabulary list with
  repeats, with one test using a repeated word (2.2, `0a29571`; M2.T1 is already validated).
  - Template 3 varies keys with fixed-form numbers; template 4 uses a fixed set of key names.
- **Depends on:** M2.T1.
- **Status:** signed off

### M2.T5: Family D, spam and promotional

- **Goal:** generate URLs, emails, phone numbers and promotional phrases, plus hard
  positives.
- **Design refs:** `DESIGN.md` `d7f95e3`: 2 (family D, hard positives), 2.2, 4 (rules 8, 9).
- **Files:** `bqc/synth/spam.py`, `tests/test_synth_spam.py`, `bqc/synth/__init__.py`,
  `tests/test_synth_registry.py` (wiring, M2.T1 criterion 8).
- **Brief:** the delegate's brief contains the family's section 2 row, sections 2.1, 2.2 and 4, and the task entry; nothing from `data/sanity.csv`. The delegate doesn't open that file or
  `tests/fixtures/sanity_*.csv` beyond what its task needs (section 4 rule 8).
- **Acceptance criteria:** Exactly 6 templates, at least 2 hard-positive (rule 6). Covers each
  listed kind, with hard positives such as `cheap area rug http://x.co`. Registered with
  family `D`. Phone numbers are never a bare digit run (2.2). Domains and numbers follow rule 9.
- **Tests:** registry tests pass for this family; each kind appears in the output; the phone
  template's output never matches the part-number regex.
- **Depends on:** M2.T1.
- **Status:** signed off
- **Status note:** implemented in `c0dcd80`, corrected in `f78aca4`, `2dac920` and `496645e`. Review history
  is in git (`428b969`, `34c01e6`, `d29b06e`); it isn't part of any brief.

### M2.T6: Family E, off-topic natural language (evaluation-only)

- **Goal:** generate 120 questions, chat prompts and code snippets, tagged `E`.
- **Design refs:** `DESIGN.md` `d7f95e3`: 2 (family E, 2.1), 2.2, 4 (rule 8).
- **Files:** `bqc/synth/offtopic.py`, `tests/test_synth_offtopic.py`, `bqc/synth/__init__.py`,
  `tests/test_synth_registry.py` (wiring, M2.T1 criterion 8).
- **Brief:** the delegate's brief contains the family's section 2 row, sections 2.1, 2.2 and 4, and the task entry; nothing from `data/sanity.csv`. The delegate doesn't open that file or
  `tests/fixtures/sanity_*.csv` beyond what its task needs (section 4 rule 8).
- **Acceptance criteria:**
  1. Generated by the same machinery, with `family="E"`, exactly 6 templates of 20 and at
     least 2 hard-positive (rule 6).
  2. Never part of the training 480.
  3. Output is text that is not a product search; no template produces an off-domain
     shopping query (2.1).
- **Tests:** registry tests pass for this family; the training output contains no `E`
  sample; requesting E returns exactly 120. As the last family, it replaces the conditional
  test `test_module_level_generate_uses_default_registry`: module-level `generate` returns the 480 and
  `REGISTRY.validate(ALL_FAMILIES)` passes.
- **Depends on:** M2.T1.
- **Status:** signed off

### M2.T7: Generation audit

- **Goal:** fail on exact collisions with negatives or sanity queries, report
  near-duplicates, and log length distributions.
- **Design refs:** `DESIGN.md` `7482987`: 4 (rules 2 and 4), 8.6.1 (independence paragraph), 11.7.
- **Files:** `bqc/synth/audit.py`, `tests/test_synth_audit.py`.
- **Brief:** this task isn't template authoring. Code reads the sanity queries through the M1.T6
  loader; the delegate has no need to view them, and reports give ids and counts, not query text.
- **Acceptance criteria:**
  1. A positive identical, after `normalise()`, to a negative **or to a sanity query**
     raises an error naming both strings and which list was hit. This check is callable
     on its own, because it runs on every generation, including the per-fold ones (rule
     4, M3.T3). The sanity queries come from the M1.T6 loader, which exposes them
     regardless of label state.
  2. One function reports every pair of positive and other string with character-3-gram Jaccard
     above a given threshold. `bqc generate` is its only caller and uses it twice: negatives at 0.6,
     sanity queries at 0.5 (rule 4, M2.T8). Neither threshold fails a run.
  3. Length statistics are returned per family, alongside those of WANDS.
- **Tests:**
  - Identical pair; pair differing only in case and whitespace; the same two cases
    against a sanity fixture query (the fixture is not the real file).
  - Jaccard exactly at the threshold (not reported) and just above it (reported), at 0.6 and at 0.5.
  - Strings shorter than 3 characters, such as family C's `a`, don't raise.
- **Depends on:** M1.T3, M1.T6, M2.T1.
- **Status:** signed off
- **Status note:** implemented in `411330a`, corrected in `968c199`. `opus`'s correction list is in `d6ef691`.

### M2.T9: Part-number strings

- **Goal:** generate the 100 legitimate part-number strings for the diagnostic of 8.6 item 4.
- **Design refs:** `DESIGN.md` `d7f95e3`: 2 (SKU row), 2.2, 8.6 (item 4).
- **Files:** `bqc/synth/partnumbers.py`, `tests/test_synth_partnumbers.py`.
- **Brief:** the delegate's brief contains the family's section 2 row, sections 2.1, 2.2 and 4, 8.6 item 4, and the task entry; nothing from `data/sanity.csv`. The delegate doesn't open that file
  or `tests/fixtures/sanity_*.csv` beyond what its task needs (section 4 rule 8).
- **Acceptance criteria:**
  1. Generates 100 distinct strings, `family="P"`, label 0, seeded, with `template_id` and
     `seed` metadata, 25 per shape: bare prefix+digits (`ab004512`), bare digit runs,
     prefix-digits-suffix (`bf-7731x`), and product words plus a part number
     (`leather sofa w003412`).
  2. Product words come from a vocabulary argument; the callers pass the occurrence list of all
     480 negatives (8.3).
  3. It isn't a registered positive family, so it is outside the 480 and the 600, the rule 6
     template check, the 2.2 registry assertions and the Jaccard report.
  4. It must not collide with a WANDS or sanity query after `normalise()`: it reuses the
     exact-match check of M2.T7, with both lists passed by its caller. On a collision the generator changes, never the sanity file.
- **Tests:**
  - Exactly 100 distinct strings; 25 per shape; each has `family="P"` and label 0.
  - The first three shapes match the part-number regex as a whole, and the last token of the
    fourth does.
  - Output is lowercased and ASCII-only.
  - Same seed gives identical output; a different seed differs.
  - Product words come only from the given vocabulary; an empty vocabulary raises `ValueError`.
  - No `P` sample appears in the training output or in family E.
  - Collision with a WANDS fixture query, and with a sanity fixture query, raises.
- **Depends on:** M2.T1, M2.T7.
- **Status:** signed off
- **Status note:** implemented in `7b0678a`, corrected in `e3f2229`. `opus`'s correction list is in `3ef106b`.

### M2.T10: Render any number of positives per template

- **Goal:** render `per_template` distinct strings for chosen templates, without changing them.
- **Design refs:** `DESIGN.md` `5ba7cee`: 2.2, 4, 8.3, 11.18.
- **Files:** `bqc/synth/registry.py`, `tests/test_synth_registry.py`.
- **Acceptance criteria:**
  1. `render(vocabulary, seed, templates, per_template)` calls the registry's `generate()` with
     `seed + i` for `i = 0, 1, ...` and keeps, per wanted template, the first `per_template` distinct
     outputs. `templates` holds `(family, template_id)` pairs; an unknown pair raises `ValueError`.
  2. It stops after a fixed number of rounds in proportion to `per_template / QUOTA`, and returns
     the samples and each template's shortfall. A shortfall never pads and never raises.
  3. Output order follows `templates`. The templates themselves and `generate()` don't change.
- **Tests:**
  - 209 per template over several templates; every sample has its metadata.
  - A stub with fewer distinct strings than asked gives its shortfall and stops.
  - Same seed gives identical output, across `PYTHONHASHSEED` values too; a different seed differs.
  - `per_template` of 20 or fewer works; an empty vocabulary raises `ValueError`.
  - `assert_template_rules` runs on every real template with a fold-sized vocabulary that also holds
    product-span tokens: repeats, `''`, `.`, `-`, `&`, `w/`, `all-clad`, `qt`, `24''`.
- **Depends on:** M2.T1, M2.T6.
- **Status:** signed off
- **Status note:** implemented in `2a9d1ec`, corrected in `eebc638`. `opus`'s correction list is in
  `554d7da`. `ROUND_FACTOR` is 5: at 833 per template (the 20,000 sensitivity), `B.template` stays about
  9 short of what a larger cap reaches, which is negligible.

### M2.T11: Drop exact matches without showing them

- **Goal:** the audit's dropping entry point, next to the raising `check_exact`: it removes strings
  equal to any of several lists and returns only survivors and counts (3.2, 11.25).
- **Design refs:** `DESIGN.md` `99037a5`: 3.2, 4 (rule 4), 8.6.1, 11.25.
- **Files:** `bqc/synth/audit.py`, `tests/test_synth_audit.py`.
- **Brief:** this task isn't template authoring, and it doesn't read the sanity file.
- **Acceptance criteria:**
  1. `drop_exact(strings, lists)` takes the candidate strings and a mapping from list name to strings.
     It returns the candidates that aren't equal to any listed string after `normalise()`, in order,
     and a count per list name. A candidate in several lists counts once, under the first.
  2. It never returns, logs or raises the removed text.
  3. `check_exact` and `near_duplicates` don't change.
- **Tests:** identical; differing only in case and whitespace; listed strings normalised too; order
  kept; per-list counts, including a candidate in two lists and a list with no hits; empty inputs;
  input not mutated; removed text absent from the return value and from captured logs.
- **Depends on:** M2.T7.
- **Status:** signed off
- **Status note:** implemented in `480160f`. Process deviation (F1), the third: the failing run was
  reported with its tracebacks collapsed, as "Ran 35 tests, FAILED (errors=11)", all
  `NotImplementedError`. The count matches the 11 new tests. Minor, for the next edit of that file: the
  `# Sets of 3-grams` comment at `tests/test_synth_audit.py:84` now sits above `DropExactTest`, not
  `NearDuplicatesTest`.

### M2.T12: Product-name spans

- **Goal:** build the training negatives of 3.2 in one home.
- **Design refs:** `DESIGN.md` `99037a5`: 3.2, 8.3, 11.25, 11.26, 11.27, 13.17, 13.19.
- **Files:** `bqc/negatives.py`, `tests/test_negatives.py`.
- **Brief:** this task isn't template authoring. The builder takes the sanity queries as a list of
  strings and never reads the file; tests use invented strings.
- **Acceptance criteria:**
  1. Each product name is `normalise()`d, then folded to ASCII (NFKD, drop non-ASCII, re-collapse
     whitespace). A name empty after folding is skipped (11.26).
  2. `build_spans(products, n, seed, real_queries, part_numbers, sanity)`, using
     `random.Random(seed)`, draws `k` from the constant word-count distribution of 3.2, a product
     uniformly among those with at least `k` words, and a start uniformly. So a short product redraws
     the product, not the length (11.27).
  3. A span empty after `normalise()` is dropped and counted. The other four removals go through
     `drop_exact`: equal to an already drawn span, a real query, a P string or a sanity query (3.2 step
     3). So candidates are drawn in batches, deduplicated within the batch into the same "drawn" count.
     It returns the spans and a count per kind of removal, with no removed text.
  4. Each span carries `product_id`, `k` and `seed`. It stops at `n` (default 5,000, one home here).
  5. If `n` can't be reached, it raises an error with the count reached, and never pads (11.25).
  6. `token_occurrences(strings)` returns every whitespace token of the normalised strings, with
     repeats and in order: the one vocabulary helper for M2.T8, M3.T3, M3.T10 and M5.T3. A constant
     caps the product-side list at the probe's 10,000 occurrences, and callers record it in the run
     config (8.3).
  7. `bqc/negatives.py` doesn't import `bqc/sanity.py` or open any file.
- **Tests:**
  - **Length draw:** the constant sums to 480 and matches 3.2; over a large draw, the share of each
    `k` is close to it.
  - **ASCII fold:** a non-ASCII name folds, an all-non-ASCII name is skipped, and every span is ASCII
    with `normalise(s) == s`.
  - **Removals:** one test per kind, each counted, with no removed text anywhere in the return value.
  - **Exhaustion:** `n` above the distinct spans available raises with the count; so does a pool with
    no product long enough for a drawn `k`.
  - **Determinism:** same seed gives identical spans, across `PYTHONHASHSEED` values too; a different
    seed differs.
  - Spans are contiguous words of their product, and `product_id` and `k` match.
  - `token_occurrences` keeps repeats and order; the cap keeps the first 10,000.
- **Depends on:** M1.T8, M2.T11.
- **Status:** signed off
- **Status note:** implemented in `17ed0ac`. Real builds by `opus`: 5,000 spans take 0.34 s; 589,376
  distinct spans are available, so exhaustion can't fire on real data at 40,000 or on half the catalogue.
  Deduplication shifts the `k` shares a little: `k=1` 3.9-4.5% against 4.8%, `k=2` 20-21% against 22.7%,
  mean words 3.47 against 3.38; 3.3% for `k=1` at 20,000. Minor, for the next edit of that file:
  `test_cap_keeps_the_first_tokens` lands exactly on the cap, so it doesn't exercise the slice at
  `bqc/negatives.py:132`.

### M2.T8: `bqc generate`

- **Goal:** build the default training set from the command line and write the report for
  inspection by eye.
- **Design refs:** `DESIGN.md` `6cec6c6`: 3.2, 4 (rule 4), 8.3, 8.6 (item 4), 8.6.1, 10, 11.7, 13.17, 13.19.
- **Files:** `bqc/report.py`, `bqc/cli.py`, `tests/test_report.py`,
  `tests/test_cli_generate.py`.
- **Brief:** this task isn't template authoring. Code reads the sanity queries through the M1.T6
  loader; the delegate has no need to view them, and reports give ids and counts, not query text.
- **Acceptance criteria:**
  1. `python -m bqc generate [--negatives-size 5000]` builds, in this order: the 100 part-number
     strings of M2.T9 (vocabulary from the real queries); the spans with seed `SEED` (M2.T12); the
     positives of A-D rendered from the spans' token occurrences at `n // 24` per template
     (M2.T10); and the 120 of family E from the real queries' occurrences. Each sample is tagged with
     its family, so E is never confused with the training set.
  2. It runs the audit over all of them and writes one JSON report to `results/` holding: spans and
     positives with their metadata; per-family and per-template length statistics and distinct
     counts next to WANDS', with any template under 50% distinct marked as a finding (section 4);
     each template's shortfall; the span audit counts; the share of real queries with a span above
     Jaccard 0.6; near-duplicate pairs of positives against spans above 0.6; and the one-off share of
     hard-positive word slots next to the one-off share of the spans' token occurrences (section 10,
     13.17). The report also records the vocabulary cap.
  3. The report holds the 100 part-number strings under their own key, `part_numbers`. They aren't
     counted among the positives, aren't in the positives' samples or length statistics, and aren't
     in the near-duplicate report.
  4. `write_report()` in `bqc/report.py` is the one home for writing results: it adds the
     config fields of section 10, names the file by config hash and timestamp, and refuses to
     overwrite. M3.T8 and M5.T3 import it from there.
  5. **Rule 4, for every positive including E:** one equal to a sanity query fails the generation
     through `check_exact`. One equal to a span, a real query or a P string is dropped through
     `drop_exact` and counted in the report, never shown.
  6. A missing `data/raw/query.csv` or `data/raw/product.csv`, and a span pool that runs out, exit
     non-zero with a message; the missing-file one points at `fetch`.
  7. `main` dispatches on `args.command`, replacing the M1.T5 shortcut that runs `fetch` for any
     parsed command, with each command's error handling kept to that command.
  8. In the real run, every output has a non-benign token against the vocabulary it was rendered
     from (2.2), which the contract's synthetic vocabulary can't show.
  9. The report lists, report-only, every sanity query with character-3-gram Jaccard above 0.5 to any
     generated positive, as its id and its maximum Jaccard, computed with the M2.T7 function. It holds
     no query text, and it never fails a run (rule 4, 8.6.1).
  10. The real run takes the sanity queries through `load_sanity` with the real WANDS queries, passed
     as `[q.query for q in sanity.queries]`, and gives `build_spans` the real queries, the real P strings
     and those sanity queries: M2.T12's real run had the last two empty, so this is the first proof of
     those filters on real data. Its near-duplicate and span audit counts, which count draws rather
     than distinct strings, and its runtime go in the task report.
- **Tests:** on fixture paths:
  - Success writes a report with spans and positives, with their metadata.
  - Length statistics, distinct counts, shortfalls, span audit counts, near-duplicate pairs and both
    one-off shares are present.
  - `--negatives-size` sets the span count and the renders per template; a non-positive value is a
    usage error.
  - A vocabulary with `chair` x10 and `zen` x1 yields `chair` far more often than `zen` across
    outputs.
  - E's hard positives use only the real queries' words, and A-D's only the spans' words.
  - `part_numbers` holds 100 entries, and no `P` sample is among the positives.
  - `write_report`: config fields present; a name collision doesn't overwrite.
  - A collision with a sanity fixture query exits non-zero; one with a span, a real query or a P
    string is dropped and counted, with no text in the report. Both hold for a family E sample too.
  - Either raw file missing exits non-zero; an exhausted span pool exits non-zero.
  - With a sanity fixture: a query just above 0.5 is listed with its id and maximum Jaccard, one at 0.5
    isn't, and the run still exits zero.
  - `generate` doesn't call `fetch`, and `fetch` doesn't call `generate`.
- **Depends on:** M1.T4, M1.T5, M1.T8, M2.T2, M2.T3, M2.T4, M2.T5, M2.T6, M2.T7, M2.T9, M2.T10, M2.T11, M2.T12.
- **Status:** signed off
- **Status note:** real run (5,000 spans, 35 s; `opus` re-ran it twice, identical JSON): 4,804 positives
  of A-D plus 120 of E; shortfalls only `B.script` (148, flagged as a finding, 60 of 208 distinct) and
  `B.traversal` (40, not flagged, 168 of 208); 0 outputs without a non-benign token; span removals drawn
  689, real queries 17, P strings 0, sanity 0; 0 positives dropped; 2 near-duplicate pairs above 0.6; 57
  of 480 real queries (11.9%) have a span above 0.6; 3 sanity queries above 0.5; one-off share 18.3% of
  hard-positive word slots against 16.8% of span occurrences. `--negatives-size` below 24 is a usage
  error (no render per template). Real-data proof of the span filter: all 50 sanity queries and 100 P
  strings reached `build_spans`, and at 20,000 and 40,000 spans it removed 2 and 4 spans for sanity.
  Implemented in `aeb7f41`, tests added in `ffc6694` and `0f635f0`. `opus`'s correction lists are in
  `a57249b` and `f23b5ef`.

### M2.T13: Widen `B.script` and `B.traversal`

- **Goal:** give the two short templates enough variety that family B reaches its quarter of the
  positives (M2 sign-off, follow-up F5). Commit scope: `fix(M2.T3)`.
- **Design refs:** `DESIGN.md` `5654502`: 2 (family B), 2.2, 4 (rules 4, 5, 6, 8), 13.20.
- **Files:** `bqc/synth/payloads.py`, `tests/test_synth_payloads.py`.
- **Brief:** the delegate's brief contains the family's section 2 row, sections 2.1, 2.2 and 4, and the
  task entry; nothing from `data/sanity.csv`. The delegate doesn't open that file or
  `tests/fixtures/sanity_*.csv` beyond what its task needs (section 4 rule 8). This is template
  authoring: the delegate chooses the new forms and arguments, and doesn't add examples quoted from
  anywhere else.
- **Acceptance criteria:**
  1. `B.script` and `B.traversal` have more forms or arguments, and each reaches 263 distinct strings
     through `render` (`5,000 // 19`, a fold's training size) with no shortfall.
  2. Family B keeps exactly its 6 templates, and every output keeps passing `assert_template_rules`.
  3. A real `python -m bqc generate` exits 0 with no shortfall on either template. The task report gives
     the positives total and only the count of sanity queries above 0.5.
- **Tests:**
  - Each of the two templates reaches 263 with no shortfall, on `SPAN_VOCAB`.
  - Each new form appears in the output.
  - Registry tests pass for this family.
- **Depends on:** M2.T3, M2.T8, M2.T10.
- **Status:** signed off
- **Status note:** implemented in `e60416d`. `fable`'s review: `DESIGN.md` `b385eb3`, suite 339 OK, own
  `generate` run matched. Real `generate` re-run by `opus`: 4,992 positives of A-D
  (`n // 24` x 24, 8 short of 5,000 by rounding only) plus 120 of E. The summary's "positives: 5112"
  includes E, as "positives: 4924" did before this task. No template is short; 0 dropped; 0 all-benign;
  3 sanity queries above 0.5. Longest output 141 characters, all ASCII and normalised. The rule 4 gate
  worked: a first attempt at a minimum depth of 2 failed `generate` on a sanity collision, and the
  coordinator told the author only to restore the validated minimum of 3. The leak to the author is
  one bit at most. Process deviation (F1): a hand-typed `errors=0` in the failing block, corrected by the
  author, and only the last lines of the final `-v` run. M2's sign-off doesn't cover this task; `fable`
  reviews it as for M1.T8. It isn't a condition of M2.

---

## M3: Features and evaluation harness (fake `Scorer`)

**Sign-off: 2026-11-06, `fable`, verdict POSITIVE** (audited at `7e03854`; `DESIGN.md` `1152683`).

- Suite: `.venv/bin/python -m unittest discover -s tests`, 636 tests, OK, run by `fable`.
- Diff read: `bqc/features.py`, `bqc/models.py` and the whole of `bqc/evaluate.py` at `7e03854`,
  plus every M3 commit's review record. Invariants checked by `fable`: runners and sections see
  only `fit` and `score` of the `Scorer` port (section 6); no real query is in any primary fit,
  thresholds come from training-side or out-of-fold scores only, and the sanity and part-number
  sets reach no fit or threshold (section 8); every split and render is seeded from `SEED`
  (no unseeded randomness in `bqc/`, section 10); no test downloads or reads `data/raw/`.
- Probes recorded: 13.9 (no `|` in `query_class`, closed by `fable`), the pickle finding in the
  `save_final_model` docstring (M3.T10), the real-file cell count under M3.T16.
- Rulings made during M3 and recorded in `DESIGN.md`: `n` per arm for the LOFO quota (8.6 item 1,
  `7bcea5e`), the loader carries `query_class` (11.15, `352f9b5`; task M3.T16), section 3 aligned
  (`061c01c`).

Follow-ups:

- **F1 (process, revised at this sign-off):** the pasted-run rule of the M2 sign-off was not
  followed in M3: `sonnet`'s reports summarised their runs instead of pasting them. What stood in
  for it, and is now the rule: `opus` runs its own mutation pass in an isolated copy for every
  task, each mutant alone, and lists the survivors as corrections. A task whose validation has no
  mutation pass is not validated. Recorded without penalty for M3.
- **F8 (cost, not a defect):** the LOFO and catalogue-gap passes each rebuild the spans and the
  test side (four and one extra `iter_fold_data` passes). Deterministic and asserted identical to
  the main run. M6.T1 reports the wall time; optimise only if the section 10 budget bites.
- **F9:** M4.T4 wires what M3 left as separate calls: `fit_final_model` and `save_final_model`
  per arm and model, `sanity_section`, `part_number_section`, and `query_classes` from the
  loader (M3.T16) for `--catalogue-gap`. Already in M4.T4's criteria.
- **F10 (accepted):** `leave_out` and `gap` on the runners are not mutually exclusive by
  construction; `build_run_report` is their only caller and never combines them.

### M3.T1: Vectoriser factory

- **Goal:** one factory for the TF-IDF configuration.
- **Design refs:** `DESIGN.md` `f07a5da`: 5, 11.1, 11.3, 11.9, 13.3, 13.21.
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
- **Status:** validated
- **Status note:** implemented in `540e125`.

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
- **Status:** validated
- **Status note:** implemented in `c6911b0`. Process deviation (F1): the fake's bodies were filled in by a
  Python script rewriting `tests/fakes.py`, not the Edit tool; disclosed. For later tasks: `FakeScorer`
  keeps only its last fit, so tests count fits through a factory that records each instance (M3.T5,
  M3.T13); and an all-letters string scores 0.0, so threshold fixtures need queries with varied scores.
  `tests/test_scorer_port.py:8` imports `tests.fakes`, while other tests import helpers as top-level
  modules (`from synth_stubs ...`); later tests should use one form, `from fakes import FakeScorer`.

### M3.T3: Outer folds and per-fold data

- **Goal:** build repeated grouped folds, and each fold's training, threshold and test data for
  either arm.
- **Design refs:** `DESIGN.md` `6cec6c6`: 3.2, 4 (rules 3 and 4), 8.1, 8.2, 8.3, 10 (seeds), 11.7, 11.8,
  11.16, 13.10, 13.14, 13.17, 13.19.
- **Files:** `bqc/evaluate.py`, `tests/test_folds.py`; for follow-up F6, `tests/test_synth_audit.py` and
  `tests/test_negatives.py`.
- **Acceptance criteria:**
  1. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + repeat_index)`,
     3 repeats, with both parameters overridable for tiny test configs.
  2. Fold units: each real query is its own group, and each template of A-D is a group of nominal
     size 20 from `group_sizes()`. Spans are never fold units and never on the test side (8.1).
  3. **Primary arm:** training negatives are the spans built once per repeat with seed
     `SEED + repeat_index`, shared by that repeat's folds. Training positives are rendered for the
     training-side templates from the spans' occurrences (capped, M2.T12), `n // n_training_templates`
     each. The threshold set is the training-side real queries.
  4. **Comparison arm:** training negatives are the training-side real queries, with positives 20 per
     training-side template from those queries' occurrences.
  5. **Both arms:** the test side is the test-side real queries, 100 renders per test-side template,
     and 120 of family E, all from the test-side queries' occurrences. The test-side strings are
     identical across arms for a given repeat and fold. E never appears in a training set.
  6. Vocabulary: no real query's token reaches a training positive in the primary arm (8.3). Every
     list comes from `token_occurrences` (M2.T12).
  7. **Rule 4, for every positive, training or test, E included:** one equal to a sanity query fails the
     generation through `check_exact`. One equal to a training negative, any of the 480 real queries or
     a P string is dropped through `drop_exact`, and the per-fold counts are returned. The sanity
     queries come from the M1.T6 loader as `[q.query for q in sanity.queries]`.
  8. A single fold-check function raises if either class or any training family (A-D) is
     missing from a training fold, or if either class is missing from a test fold. Families
     are never asserted on the test side (8.1, 11.16). M3.T5, M3.T13 and M3.T9 use it too.
  9. Each fold's class and per-family counts, span removal counts and render shortfalls are returned
     for the run JSON (8.1). The assignment's nominal positive share is about 0.45-0.51 (13.10); the
     test side's actual share is far higher, with 100 renders per template.
- **Tests** (stub templates, invented strings):
  - No group on both sides of a fold (11.8).
  - Primary: a sentinel word only in a real query never reaches a training positive; a sentinel word
    only in a span never reaches a test positive or E.
  - Comparison: a sentinel word only in a test-side query never reaches a training positive, and the
    reverse (8.3).
  - No real query in the primary arm's training negatives; the spans are identical across a repeat's
    folds and differ across repeats.
  - Test-side strings are identical across arms.
  - A stub template given a side whose negatives hold `chair` x10 and `zen` x1 yields `chair` far
    more often than `zen`.
  - No E sample is in a training set.
  - Same seed gives the same folds; different repeats give different folds.
  - A class missing from a training or a test fold raises (11.16).
  - A training fold missing a family raises (11.16).
  - A family present in training but absent from a test fold doesn't raise, and its test count
    is returned as 0 (8.1, 13.14).
  - A collision with a sanity fixture query raises, on the training and the test side.
  - A collision with a training negative, a real query on either side or a P string is dropped and
    counted, with no text in what's returned.
  - Follow-up F6: `token_occurrences` with a cap below the token count returns exactly the cap, so
    `bqc/negatives.py:132` is exercised; and the `# Sets of 3-grams` comment at
    `tests/test_synth_audit.py:84` moves back above `NearDuplicatesTest`.
- **Depends on:** M2.T1, M2.T7, M2.T10, M2.T11, M2.T12. Real families aren't required by the tests,
  which use stub templates.
- **Status:** validated
- **Status note:** implemented in `9c7eb85`, with F6, corrected in `99927f2`. `opus`'s correction list is
  in `1db79a5`. Real run by `opus`, all 15 folds of both arms: no `FoldError`, so the sanity gate doesn't
  stop the run at the real seeds; every query and template is on the test side once per repeat; training
  4,996-5,000 positives to 5,000 spans in the primary arm and 380-400 to 384 in the comparison arm; no
  shortfall; about 10 s per primary fold to build. A training family is never absent; one is absent from
  the test side in 12 of 15 folds. The move of rule 4 to `screen_samples` changed `cli.py` in a
  signed-off milestone: `generate`'s report body is identical before and after. Process shortfall: the
  tests for items 2 and 5 and for `screen_samples` had no failing skeleton stage; disclosed.

### M3.T4: Threshold selection

- **Goal:** pick the threshold from the real queries' scores, as a pure function.
- **Design refs:** `DESIGN.md` `5ba7cee`: 8.2, 8.4, 11.10, 11.11, 11.19.
- **Files:** `bqc/evaluate.py`, `tests/test_threshold.py`.
- **Acceptance criteria:**
  1. Takes the real queries' scores only, in both arms (8.2), and returns the smallest of those
     scores with FPR at or below the target.
  2. The decision rule `score >= t` is illegitimate lives in one place.
  3. If no score reaches the target, returns the one with the smallest FPR and sets
     `target_unmet=true` (11.10).
  4. `target_fpr < 1/n_negatives` sets a warning field, with no silent rounding (11.11).
- **Tests:** hand-computed cases:
  - 384 scores at a 1% target give the third-highest score;
  - tied scores, including ties straddling the target;
  - a score exactly at `t` counted illegitimate (11.19);
  - a target below resolution;
  - `target_unmet`;
  - no scores raises `ValueError`.
- **Depends on:** M1.T2.
- **Status:** validated
- **Status note:** implemented in `52f8df7`. The code is right: no mismatch with a brute-force oracle of
  8.4 on 20,000 random cases, ties as one block; at a 1% target, 384 scores flag 3, 233 flag 2 and 190
  flag 1, none with a warning. Additions beyond the criteria, accepted: `fpr` on `Threshold`, and
  `ValueError` on a non-finite score (without it `sorted` gives an undefined order; 11 has no path for
  this, reported to `fable`). Process shortfall, third task running: the implementation went in by an
  inline Python string replace and the skeleton and tests by Bash heredocs, not the Edit tool; disclosed.
  Three test gaps, listed in `959d090`, were closed in `e390bb8`, which changes only
  `tests/test_threshold.py`; the file changes there used Edit, as `sonnet` reports. 19 of 20 mutations
  are now caught; the survivor is equivalent, since FPR only rises down the sorted scores.

### M3.T5: Cross-fit outer-fold runner (primary arm)

- **Goal:** for each outer fold, fit on spans and positives, set the threshold on the training-side
  real queries, and score the test side and family E.
- **Design refs:** `DESIGN.md` `5ba7cee`: 8.2, 8.5 (family E), 11.9, 11.12, 11.16, 13.19; `a8b8600`: 11.30.
- **Files:** `bqc/evaluate.py`, `tests/test_runner.py`.
- **Acceptance criteria:**
  1. Depends only on the `Scorer` port, via a factory; imports no sklearn estimator.
  2. One fit per outer fold, with no inner loop. No real query is ever in a fit (8.2).
  3. The threshold is the M3.T4 rule over the training-side real queries, scored by that fit.
  4. The same `t` applies to the test-side queries, the test-side positives and family E.
  5. Warnings raised during `fit` (11.12) are captured per fold and returned, and don't
     raise.
  6. One finite-score check lives in `bqc/evaluate.py` (11.30): a function taking a score array and a
     context of repeat, fold, arm, model and score-set name, which raises on any `NaN` or `inf` with all
     five in the message. The runner calls it on every score array as it's produced: threshold queries,
     test queries, test positives and E. `select_threshold`'s own check stays as a second line.
- **Tests** (with the fake):
  - A `NaN` in test-query scores raises, with the repeat, fold, arm, model and `test queries` in the
    message; the same for E (11.30).
  - The same error, not `select_threshold`'s `ValueError`, for a `NaN` in threshold-query scores; and for
    `+inf` and `-inf` in test positives.
  - A finite array passes the check untouched.
  - No real query, test-side text or E text in any recorded fit input (11.9).
  - Number of fits equals the number of outer folds.
  - Changing test-side scores doesn't change the threshold; changing training-side query scores does.
  - Test-side row counts are unchanged at scoring.
  - Across a repeat, each real query is a test item exactly once.
  - A fake emitting `ConvergenceWarning` is captured, not raised (11.12).
- **Depends on:** M3.T2, M3.T3, M3.T4.
- **Status:** validated
- **Status note:** implemented in `cdc1fcb`. Real fold by `opus`, repeat 0 fold 0, real vectoriser and a
  scratch logistic regression: no overlap between the fit and any real query or the test side, raw or
  normalised; no E in the fit; scores `float64` before and after the runner; `t`'s FPR over the 384 is
  3/384; no warning from fit or score. Warnings are captured around `fit` only, which is what 11.12
  covers. `test_evaluate_imports_no_sklearn_estimator` passed at the skeleton stage, as a guard. `sonnet`
  reports no scripted edits, apart from an empty no-op heredoc. Corrected in `d999ca9`, list in `9ab1b52`:
  `FoldResult` now carries `test_texts`, and a comment, an import and the docstring are fixed. All 27
  mutations are caught, and both new ones on `test_texts`.

### M3.T13: Nested runner (comparison arm)

- **Goal:** run the v1 protocol on the same outer folds, so that results pair fold by fold with the
  primary arm.
- **Design refs:** `DESIGN.md` `5ba7cee`: 8.2, 11.9, 11.16, 13.10; `a8b8600`: 11.30.
- **Files:** `bqc/evaluate.py`, `tests/test_runner_comparison.py`.
- **Acceptance criteria:**
  1. Inner folds are `StratifiedGroupKFold(n_splits=5, shuffle=True,
     random_state=SEED + repeat_index)` over the outer training rows, with the same groups.
     Positives aren't regenerated per inner fold (8.2).
  2. Out-of-fold scores for the training-side queries give `t` by the M3.T4 rule; the model is then
     refitted on the whole training side.
  3. Every inner training fold passes the M3.T3 fold check (both classes and all training
     families, 11.16).
  4. It returns the same result shape as M3.T5, under its own arm name, with the same test-side
     strings.
  5. Every score array, the out-of-fold threshold scores included, goes through the M3.T5 finite-score
     check (11.30).
  6. The test-side scoring of `run_primary_fold` is shared, not repeated. Warnings from the inner fits and
     the refit are captured per outer fold (11.12).
- **Tests** (with the fake):
  - A `NaN` in the out-of-fold scores raises the 11.30 error, naming the score set.
  - No test-side or E text in any recorded fit input (11.9).
  - No group on both sides of an inner fold.
  - Generation runs once per outer fold.
  - An inner training fold missing a family raises (11.16).
  - Number of fits: outer folds x (inner folds + 1).
  - Changing test-side scores doesn't change the threshold.
  - Test-side strings equal the primary arm's for each repeat and fold.
- **Depends on:** M3.T5.
- **Status:** validated
- **Status note:** implemented in `8ca9124`. The `_test_result` refactor leaves the primary arm unchanged:
  a real fold gives byte-identical scores and the same `t` before and after. Real runs by `opus`, all 15
  comparison folds: test strings equal the primary arm's in 15 of 15; 0 of 75 inner training folds lose a
  family or a class, and 0 of 75 held-out sides have one class (at least 60 positives); with each family's
  rows removed, as LOFO will do, 0 of 300 lose another family or have a one-class held-out side. One real
  fold with a scratch logistic regression: 6 fits, `float64` scores, no overlap between fit and test side.
  A one-class held-out side raises `FoldError` from `check_fold`, not an sklearn crash; it never occurs on
  the real data and isn't tested, accepted. Corrected in `0faf062`, list in `ae52a87`: the whole
  out-of-fold array is checked, as `out-of-fold scores`, and tests now pin which model scores what. All
  mutations are caught except that one-class check. Process deviation
  (F1): `sonnet`'s report on `0faf062` didn't paste the per-test lines and may have mistranscribed one
  `AssertionError`; the coordinator re-ran the per-test command on the final state instead of returning it. The
  failing stage of that commit therefore rests on the mutation run, not the report.

### M3.T15: Remove the SMOTE dependency

- **Goal:** remove what SMOTE left in the code, before later reports record library versions.
- **Design refs:** `DESIGN.md` `0386ca8`: 6, 6.4, 7, 12.1 (item 17).
- **Files:** `pyproject.toml`, `bqc/report.py`, `bqc/models.py`, `tests/test_report.py`.
- **Acceptance criteria:**
  1. The dependencies in `pyproject.toml` are exactly `numpy` and `scikit-learn>=1.3`.
  2. `bqc/report.py`'s `LIBRARIES` is `numpy` and `scikit-learn`.
  3. The `Scorer` docstring in `bqc/models.py` says `vectoriser -> estimator`.
  4. `.venv` is rebuilt with `python3 -m venv --clear .venv && .venv/bin/pip install -e .`,
     `imbalanced-learn` is no longer installed, and the full suite passes on it.
  5. The report states the installed numpy and scikit-learn versions and flags any that differ from those
     of the section 13 probes (scikit-learn 1.9.1, numpy 2.5.4): nothing pinned them, and removing a
     dependency can change what `pip` resolves.
  6. A search for `smote`, `imblearn` and `imbalanced` in `bqc/`, `tests/` and `pyproject.toml`, ignoring
     case, finds nothing.
- **Tests:** in `tests/test_report.py`, `config["libraries"]` has exactly the keys `python`, `numpy` and
  `scikit-learn`. No other new tests.
- **Depends on:** M2.T8, M3.T2.
- **Status:** validated
- **Status note:** implemented in `5facb56`. The rebuilt `.venv` has numpy 2.5.4 and scikit-learn 1.9.1, as
  the probes did, and no `imbalanced-learn`; the search finds nothing. Process deviation (F1), disclosed:
  the failing block in `sonnet`'s report had `0.5s` retyped for `0.400s`. The test is confirmed to fail by
  a mutation restoring the old library list.

### M3.T7: Metrics and aggregation

- **Goal:** compute every number in 8.5 from fold results.
- **Design refs:** `DESIGN.md` `5ba7cee`: 8.1, 8.4, 8.5, 13.13, 13.14, 13.19.
- **Files:** `bqc/evaluate.py`, `tests/test_metrics.py`.
- **Acceptance criteria:**
  1. Per repeat and pooled: headline recall and achieved FPR, Wilson 95% CI (standard
     library `math` only), confusion matrix, macro F1, precision and recall. ROC-AUC is given both ways,
     as mean and sd over folds and pooled (8.5).
  2. **FPR intervals (8.5):** a repeat's interval uses `k_r` at `n = 480`. The pooled point
     estimate is the total over all repeats divided by 1440, and the pooled interval uses
     `k = total / 3` at `n = 480`; `k` need not be an integer. Never `n = 1440`. No recall interval
     is reported. (`n` and the repeat count come from the run, so tiny configs work.)
  3. Per-family recall is caught / total, pooled over all test folds of a repeat and over all
     repeats, never a mean of per-fold values (8.5). For A-D that is 6 templates x 100 renders, 600
     per repeat and 1,800 overall, counting what survives. E is scored in every fold, so its total
     is 120 per fold, 600 per repeat and 1,800 overall. E is in its own row, marked evaluation-only,
     and never pooled into headline recall.
  4. Per-template recall is reported the same way, in 24 rows (8.5).
  5. The per-fold breakdown records each family's test count and caught count, with recall
     `null` where the count is 0. A family with zero test samples in a fold is normal (13.14)
     and doesn't raise or divide by zero.
  6. One-word false positives: per model, the count flagged out of the one-word negatives, and the words
     (23 in real WANDS; 8.5).
  7. The full list of false positives, each with the raw WANDS query string as stored in
     `query.csv` (8.5).
  8. No metric feeds any selection. The test side's positive-to-negative ratio is stated next to
     precision and macro F1 (8.5).
  9. Every flag comes from `is_illegitimate` (M3.T4, 11.19) applied to `FoldResult.test_scores` at that
     result's `threshold`. M3.T5 returns scores and `t` without applying `t`, so this is where the rule
     first decides a metric.
- **Tests:**
  - A test-side score exactly at `t` counts as flagged, in FPR and in recall (11.19).
  - Wilson against the 13.13 values, to 5 decimals: `0/480`, `5/480`, `1/96`, `0/30`,
    `480/480`, including `k = 0` and `k = n`.
  - Pooled total 15 over 3 repeats gives point 15/1440 and interval `[0.00446, 0.02415]`,
    not the `15/1440` row. Pooled total 0 gives upper bound 0.00794.
  - Non-integer `k` (total 16) gives bounds between those of `5/480` and `6/480`.
  - Hand-computed one-word count and words, with a one-word query counted once after `normalise()`.
  - Hand-computed confusion matrix and pooled per-family and per-template recall.
  - A template with fewer than 100 surviving renders counts what survives.
  - A fold where a family has zero test samples: count 0, caught 0, recall `null`, and the
    pooled recall is unaffected by it.
  - Folds of unequal family counts give caught / total, which differs from the mean of the
    per-fold recalls.
  - Adding E samples leaves headline recall unchanged.
  - The FP list holds exactly the expected queries, and a double-space query keeps its double
    space.
  - A single-class input to ROC-AUC raises rather than reporting a number.
- **Depends on:** M3.T5, M3.T15 (schedule order only).
- **Status:** validated
- **Status note:** implemented in `bba4137`. Criterion 1 is met: 8.5 asks for mean and sd over folds on
  ROC-AUC only, and the criterion's wording, which could be read otherwise, is corrected above. Widening
  the M3.T5 guard to `sklearn.metrics` is fine: the check is on the exact set, and estimator imports in
  each form still fail it. Equal negatives per repeat hold on the real data (480), so that error can't fire
  there. Real run by `opus`, all 15 folds, scratch logistic regression at the M4.T1 settings, pooled: primary
  recall 77.0%, FPR 0.49% (7 of 1,440), AUC .9711, one-word 2.33 mean per repeat; comparison 70.4%, 1.04%,
  .9552, 3.00. That's plausible against 13.22's T arm (79.7%, 0.69%, .9723) and 13.19's BASE (71.0%, 0.62%,
  .9603), with fewer false positives and less recall in the primary arm, but not a match. Family E totals
  600 per repeat, A-D 1,799-1,800 pooled, no `target_unmet`. Corrected in `a74213e`, list in `e48a6fa`: no
  numpy types in the metrics, `flagged_mean_per_repeat` added, E out of `templates`, and tests for macro F1 and
  for two repeats. `opus`'s re-run of all 15 folds: `json.dumps` round-trips in both arms, with 24 template
  rows and the same numbers. Process deviation (F1), disclosed: `sonnet` mistranscribed a timing in one mutant
  block. Follow-up F7, carried by M3.T8: no test tells `flagged_mean_per_repeat` from the union over the
  repeat count, as no test has one word flagged in two repeats. The real values are right (2.33, 3.00).

### M3.T8: Run report JSON

- **Goal:** build one reproducible JSON report per run, written through `write_report`.
- **Design refs:** `DESIGN.md` `5654502`: 3.2, 8.1, 10, 11.12, 11.18, 13.20.
- **Files:** `bqc/evaluate.py`, `tests/test_run_report.py`, `tests/test_metrics.py` (F7).
- **Acceptance criteria:**
  1. Contains config (all parameters, seed, git commit hash, library versions, arms, negatives size,
     vocabulary cap), and under each arm's own key (8.2): per-fold and pooled metrics, each fold's class
     and per-family counts, span removal and dropped-positive counts with no removed text, render
     shortfalls, FP list, per-family and per-template recall, `target_unmet` and warning fields, and
     per-fold convergence warnings. The counts come from `FoldData.report`, once per (repeat, fold, arm);
     `FoldResult` is per model and doesn't carry them.
  2. In the primary arm, it records the realised span length mix per repeat next to the target of 3.2
     (3.2, 13.20).
  3. Written by `write_report` imported from `bqc/report.py`; there is no second writer, and
     `bqc/evaluate.py` doesn't import `bqc/cli.py` (section 10).
- **Tests:**
  - Two tiny-config runs give identical JSON apart from the timestamp (11.18).
  - Output round-trips through `json` (no numpy types left).
  - Fold class and per-family counts, span removal and dropped-positive counts and shortfalls are
    present, and a run with both arms has both keys in one file.
  - A hand-checked span length mix sits next to the target; the comparison arm has none.
  - Removed span and positive text appears nowhere in the report.
  - Follow-up F7, in `tests/test_metrics.py`: one word flagged in both of two repeats gives `flagged` 1 and
    `flagged_mean_per_repeat` 1.0.
- **Depends on:** M2.T8, M3.T7, M3.T13.
- **Status:** validated
- **Status note:** implemented in `3d6f709`, with F7. Real run by `opus`, all 15 folds of both arms, scratch
  logistic regression, twice in separate processes with different `PYTHONHASHSEED`: the two files are
  byte-identical, and their names differ only in the timestamp. In the file: no sanity or part-number string,
  no span or positive text (the only matches are schema keys and the model name `lr`), and every false
  positive is a WANDS query. The realised mix sits next to the target (mean 3.46-3.49 against 3.38), as in
  13.20. The seed is recorded by `write_report` from the same `SEED`, so criterion 1 holds with no argument.
  `config_hash` changes with each of the nine parameters. Process, disclosed: an empty heredoc against
  `bqc/evaluate.py` that changed nothing, and the commit message through a heredoc. Corrected in `5463bb2`,
  list in `aeccc7f`: `iter_fold_data` is called by keyword, and tests pin each option, the stamp and a
  warning on one fold. `opus` ran 17 mutants on it, each in isolation, all caught, `sanity=[]` and the
  wrong-fold warnings included; the commit's `bqc/` changes only that call. Process, disclosed: during
  `sonnet`'s mutation run a failed revert left one mutant in place for the next, so its run of that mutant
  was not in isolation.

### M3.T9: Leave-one-family-out

- **Goal:** report recall and achieved FPR on each family at a threshold chosen without it.
- **Design refs:** `DESIGN.md` `6cec6c6`: 8.2, 8.6 (item 1), 11.16; `a8b8600`: 11.30; `7bcea5e`: 8.6
  (item 1, `n` per arm).
- **Files:** `bqc/evaluate.py`, `tests/test_lofo.py`.
- **Acceptance criteria:**
  1. For each F in A-D (E has no LOFO row), per outer fold and per arm: the arm's training negatives
     stay unchanged, and positives come from the training-side templates of the other three families
     only, at `n // n_remaining_training_templates` each, so the training side stays 1:1. `n` is the
     arm's training-negative count: the span count in the primary arm, the 384 training-side queries in
     the comparison arm (`7bcea5e`); the comparison arm's main run keeps its fixed 20 per template.
  2. The threshold follows the arm's own rule (M3.T5 or M3.T13), chosen without F, so F is absent from
     every fit and from threshold selection.
  3. The test side is the same strings as the main run for that fold: the test-side queries, and F's
     test-side templates at their 100 renders. A fold with none of F's templates on the test side
     still fits and counts towards FPR, with `null` recall on F for that fold. Such a fold has one class, so
     `compute_metrics` raises on it (ROC-AUC, M3.T7): LOFO rows are computed without that function's ROC-AUC.
  4. Each row reports recall on F, the achieved FPR at that threshold, and the paired difference to the
     main run's recall on F.
  5. Results go in the run JSON as their own section, under each arm, separate from the headline.
  6. Off by default, enabled by a parameter (`--lofo` is wired in M4.T4).
  7. The fold check asserts the other three families and both classes, in the inner folds too (11.16).
  8. Every LOFO score array goes through the M3.T5 finite-score check, with its own score-set name (11.30).
     The runners' score-set names are fixed at M3.T5, so this task, followed by M3.T14, adds the way to
     label them.
- **Tests** (with the fake), in both arms:
  - A `NaN` in LOFO scores raises the 11.30 error, naming the score set.
  - No F text in any fit input.
  - Training positives equal training negatives in count, up to rounding, with F's templates removed.
  - The test strings equal the main run's for that fold, restricted to F's templates and the queries.
  - One row per family A-D, none for E, each with recall, achieved FPR and the paired difference.
  - A hand-computed achieved FPR and paired difference.
  - A fold with F absent from the test side gives `null` recall and doesn't raise.
  - A training side missing one of the other three families raises.
  - Fit counts: 1 per family per fold in the primary arm, inner folds + 1 in the comparison arm.
- **Depends on:** M3.T8.
- **Status:** validated
- **Status note:** `iter_fold_data(leave_out=F)` re-renders the training side without F and keeps the
  test-side queries and F's rows; both runners take `leave_out`, which prefixes score-set names
  (`LOFO A test queries`) and narrows the inner family check; `lofo_section` builds the rows;
  `build_run_report(lofo=True)` adds `body[arm]["lofo"][model][F]` with `folds` and `pooled`. Comparison
  arm: positives per template are `len(train_queries) // n_remaining` (the main run uses 20), so it stays
  1:1; the criteria's `n` is read that way there. Tests use 5 outer folds: with 3, an inner training fold
  can lose a family (the 11.16 error). The `--lofo` CLI flag is not wired (M4.T4).
- **Validation (`opus`, `3ecd83f` against `7bcea5e`): CHANGES REQUESTED.** Suite: 536 tests, OK (33 s).
  `fable`'s ruling matches the code: `n // len(train_templates)` in the primary arm, `len(train_queries) //
  n_remaining` in the comparison arm, `QUOTA` when `leave_out` is None (pinned by
  `test_folds.FoldDataTest.test_comparison_training_side`). Criteria 1-7 are met and covered; the fold check
  sees the full test side before the restriction, and `compute_metrics` is never called on a LOFO fold.
  Mutation run by `opus`, 20 mutants, each in its own copy: 19 caught (F text in training, both quotas, the
  main-run quota, the test-row restriction two ways, the check on the restricted test side, the outer and
  inner family lists three ways, the score-set prefix, the `difference` sign, `null` recall as 0, main
  recall over all families, pooled FPR, FPR over all rows, the threshold source, `lofo` on by default). One
  survives, item 1. No real data in this checkout, so no real LOFO run; 13.14 and the M3.T13 note (0 of 300
  LOFO inner folds lose a family) cover the fold check on the real seeds. Process deviation (F1), disclosed
  by `sonnet`: the `leave_out` plumbing was written before its tests and no mutation pass was run.
  1. `tests/test_lofo.py:200` (`ReportTest`): no test reaches a `NaN` that appears only in a LOFO pass of
     `build_run_report`. Mutant: at `bqc/evaluate.py:731` pass `leave_out=None` to the primary runner; the
     whole suite still passes, so criterion 8's LOFO score-set name is unpinned where the run produces it
     (the comparison arm is pinned only indirectly, by the inner family check raising). Fix: add a test,
     in both arms, with a scorer factory whose scorers return `NaN` only once the main run's fits are
     spent (or only when fitted without one family's texts), so that `build(lofo=True)` raises
     `NonFiniteScoreError` naming `LOFO <F>` and the arm, and `build()` with the same factory doesn't raise.
  2. `bqc/evaluate.py:696-701`: `parameters` doesn't record `lofo`, so a run with and without it has the
     same `config_hash` and file-name prefix; M3.T8 criterion 1 puts every parameter in the config, and
     criterion 6 here makes `lofo` one. Fix: add `"lofo": lofo` to `parameters`, and test that it is
     `False` by default and `True` with `lofo=True`.
  3. `bqc/evaluate.py:480` (109 columns) and `tests/test_lofo.py:214` (101): every other line of both
     files is within 100. Fix: wrap them.
  Follow-up, cost only (not a correction): the LOFO pass calls `iter_fold_data` once per family, so spans
  and the test side are rebuilt four more times per run. The output is deterministic and the test strings
  are asserted identical to the main run's, so nothing is wrong; if M6 timing shows it matters, build the
  test side once and reuse it.
  Corrections addressed in the `fix(M3.T9)` commit that follows `ab29b45`.
- **Correction validation (`opus`, `0520ede` against `ab29b45`): PASS.** Suite: 538 tests, OK (35 s).
  `0520ede` touches only the three items (plus the one-line pointer above); `3ecd83f` is unchanged in
  history (no amend). Item 1: `ReportTest.test_nan_only_in_a_lofo_pass_names_the_lofo_score_set` runs both
  arms with a factory whose scorers return `NaN` only after the main run's fits (10 primary, 60
  comparison), asserts `LOFO A` and `arm <arm>` in the `NonFiniteScoreError`, and runs the same factory
  without `lofo` without raising. The surviving mutant (primary runner given `leave_out=None` in
  `build_run_report`) is now caught, by that test alone; a second mutant (comparison out-of-fold score set
  without the prefix) is caught by it and by `RunnerTest`. Both reverted. Item 2: `"lofo": lofo` is in
  `parameters`; `test_parameters_record_lofo` pins `False` by default and `True` when enabled, and
  `test_run_report.StructureTest` includes `"lofo": False` in the whole-config check. Item 3: no line of
  `bqc/evaluate.py` or `tests/test_lofo.py` exceeds 100 columns. Process deviations: the correction
  commit carries the `fable` co-author trailer rather than `sonnet`'s, and it adds a line to `PLAN.md`,
  which `opus` owns; neither changes the code under review.

### M3.T14: Catalogue-gap check

- **Goal:** measure what happens when a shopper searches for something the catalogue doesn't
  describe (8.6 item 5).
- **Design refs:** `DESIGN.md` `99037a5`: 3.1, 3.2, 8.2, 8.6 (item 5), 11.25, 11.28, 12.1 (item 14), 13.18, 13.19; `a8b8600`: 11.30.
- **Files:** `bqc/evaluate.py`, `tests/test_catalogue_gap.py`.
- **Acceptance criteria:**
  1. Per outer fold of every arm that runs, take the `query_class` values of the test-side real
     queries, then fit and set the threshold by the arm's rule, and test on the test-side queries and
     positives. **Primary arm:** remove every product whose `classes` contains any of them before
     building that fold's spans, and remove the training-side real queries of those classes from the
     threshold set. **Comparison arm:** remove the training-side real queries of those classes from
     both its training negatives and its threshold set.
  2. Queries with an empty class define no removal and stay in folds and thresholds (11.28).
  3. Too few products left raises the M2.T12 error; it never pads (11.25).
  4. A `catalogue_gap` section under each arm in the run JSON gives pooled FPR with the Wilson CI of
     M3.T7, recall, the flagged queries and `gap_failed`, which is true when the CI's lower bound exceeds
     the target. A pass with its point estimate above target is reported as
     `point_estimate_above_target`. The two arms' results come from the same folds and are paired. In
     the primary arm it also records the number of products removed per fold and the unclassified count:
     products with an empty `classes` can never be removed, so the check errs lenient (8.6 item 5).
  5. Its result never stops a run, and nothing acts on it: failure is a user decision.
  6. Off by default, enabled by a parameter (`--catalogue-gap` is wired in M4.T4).
  7. Every gap score array goes through the M3.T5 finite-score check, with its own score-set name; a
     non-finite score stops the run like any other (11.30).
- **Tests** (with the fake), in both arms:
  - A `NaN` in gap scores raises the 11.30 error, naming the score set.
  - Primary: no product of a test-side class contributes a span in that fold, including a multi-class
    product with one matching class.
  - Comparison: no training-side query of those classes is in any fit or the threshold set.
  - The threshold set excludes those classes' training-side queries; empty-class queries stay.
  - An empty-class product is never removed; the per-fold removed count and the unclassified count are
    hand-checked.
  - The removal differs per fold, and the normal run's spans are unaffected.
  - `gap_failed` at lower bound just above, exactly at, and just below the target.
  - Hand-computed pooled FPR uses `k = total / 3` at `n = 480`.
  - An exhausted pool raises.
- **Depends on:** M2.T12, M3.T8.
- **Status:** validated
- **Status note:** implemented in `98af9bb`. `iter_fold_data(query_classes=...)` is the gap pass: per fold,
  products with a test-side class leave the pool, spans are built from the rest with seed `SEED + repeat`,
  and the training-side queries of those classes leave the primary threshold set and the comparison
  training negatives; the runners take `gap=True` (`catalogue gap <name>`); `catalogue_gap_section` builds
  the per-model result from `compute_metrics(...)["pooled"]`; `build_run_report(catalogue_gap=True,
  query_classes=...)` adds `body[arm]["catalogue_gap"]`, with `folds` (`products_removed`) and
  `unclassified_products` in the primary arm, and records `catalogue_gap` in `parameters`. `fable`'s
  ruling `352f9b5` (11.15) makes `query_classes` a sequence parallel to the real queries, as built; the
  loader change is M3.T16.
- **Validation (`opus`, `98af9bb` against `352f9b5`): CHANGES REQUESTED.** Suite: 579 tests, OK (41 s).
  Criteria 1-7 are met. Points raised by `sonnet`: (1) spans per fold are the only reading of 8.6 item 5
  ("per outer fold ... remove ... before building the training negatives"), and `SEED + repeat` is 3.2's
  seed; the normal run's spans are provably unaffected, since the gap pass is a separate
  `iter_fold_data` call and the `query_classes is None` branch is unchanged (pinned by
  `test_the_main_sections_are_unchanged_by_the_gap_pass` and `AllEmptyClassesTest`). The gap pass screens
  test positives against its own spans, so its test side can differ from the main run's by a few drops;
  the criteria pair the two arms, not the gap with the main run, and the arms share one test side.
  (2) `gap` beside `leave_out` is acceptable: no caller combines them, and a guard would be speculative;
  revisit only if a task ever runs LOFO under a gap. (3) the Wilson interval is right: the gap pass keeps
  every test-side query, so `compute_metrics` sees 480 negatives per repeat and `k = total / 3`;
  `SectionTest` uses 96 per fold and pins `wilson_interval(4, 480)` against `wilson_interval(12, 1440)`.
  (4) criterion 4 and 8.6 item 5 ask for products removed and the unclassified count only; removed
  training queries aren't needed. (5) the `ValueError` without `query_classes` is right and tested. No
  leakage, no download, no new runtime assumption to probe (`mock.patch` of `build_spans` only).
  Mutation run by `opus`, 20 mutants, each in its own copy of `bqc/` and `tests/` (the copies lack
  `data/sanity.csv`, so `RealSanityFileTest` errors in each, the unmutated copy included; that error is
  discounted): 19 caught. `sonnet`'s 8 re-run, all caught (classes as a subset instead of an intersection,
  no query filter, `>=` and the point estimate in `gap_failed`, `""` in the removal set, unfiltered
  comparison negatives, unfiltered primary threshold set, runner without `gap=True`). Also caught:
  empty-class products removed, `products_removed` as `len(kept_products)` and as `len(gone)`, gap results
  appended to the main `models`, gap reports appended to the main `folds`, `point_estimate_above_target`
  true on a failure, E rows dropped from the gap test side, gap seed `+ fold` and `+ 1`, the unclassified
  count inverted, `catalogue_gap` on by default. One survives, item 1.
  1. `tests/test_catalogue_gap.py:30` and `:115-117`: every gap test runs one repeat, so the repeat part of
     the seed is unpinned. Mutant: at `bqc/evaluate.py:270` pass `seed` instead of `render_seed`; the whole
     suite still passes, though every repeat's gap spans would then share `SEED` (3.2 says `SEED +
     repeat`). Fix: run the `build_spans` spy with `repeats=2` and assert the seeds of the six builds are
     `[SEED] * 3 + [SEED + 1] * 3`, in repeat order.
  2. `bqc/evaluate.py:748-752`, `:764-769`, `:776-781`: the same twelve-argument `iter_fold_data` call is
     now written three times (main run, LOFO, gap), differing in one keyword. A run option threaded into
     one copy and not another would give the gap or LOFO pass a different config, and no test pins every
     option reaching the gap pass. Fix: bind the shared arguments once in `build_run_report` (for example
     `fold_data = functools.partial(iter_fold_data, real_queries=..., vocabulary_cap=vocabulary_cap)`)
     and call `fold_data()`, `fold_data(leave_out=family)` and `fold_data(query_classes=query_classes)`.
     No behaviour change; the suite stays green.
  3. `tests/test_catalogue_gap.py:416-419`: `test_a_failed_gap_does_not_stop_the_run_or_change_anything_else`
     asserts only that the key exists. The fixture does fail today (`opus` checked: lower bound 0.161 against
     0.0001 in both arms), but the test would pass vacuously if it stopped failing. Fix: assert
     `gap_failed` is `True` in both arms.
  - Corrections addressed in the `fix(M3.T14)` commit that follows `e7157d7`.
- **Correction validation (`opus`, `a09e361` against `e7157d7`): PASS.** Suite: 579 tests, OK (40 s).
  `a09e361` touches only the three items (plus the one-line pointer above); `98af9bb` is unchanged in
  history (no amend). Item 1: `PrimaryRemovalTest.test_the_repeat_seed_is_kept` runs the `build_spans` spy
  with `repeats=2` and asserts the build seeds in order, `[SEED] * 3 + [SEED + 1] * 3`. The surviving
  mutant (`seed` instead of `render_seed` at the gap span build, `bqc/evaluate.py:271`) is now caught, by
  that test alone (1 failure in 579); reverted (run in a scratch copy). Item 2: `build_run_report` binds the
  shared `iter_fold_data` arguments once with `functools.partial` and calls `fold_data()`,
  `fold_data(leave_out=family)` and `fold_data(query_classes=query_classes)`; the M3.T8 option-reaching
  tests (`test_run_report`, "Every option of build_run_report must reach iter_fold_data") still pass.
  Item 3: the failed-gap test asserts `gap_failed is True` in both arms. No line of `bqc/evaluate.py` or
  `tests/test_catalogue_gap.py` exceeds 100 columns. Process deviations, as at M3.T9: the correction commit
  carries the `fable` co-author trailer rather than `sonnet`'s, and it adds a line to `PLAN.md`, which
  `opus` owns; neither changes the code under review.

### M3.T16: Query classes in the loader

- **Goal:** carry each real query's `query_class` from `load_queries`, so the catalogue-gap check can run
  on real data (11.15).
- **Design refs:** `DESIGN.md` `352f9b5`: 3, 8.6 (item 5), 11.15, 11.28.
- **Files:** `bqc/wands.py`, `tests/test_wands.py`.
- **Brief:** a change to validated M1.T4 code, made here rather than by reopening M1.T4. Section 3's
  sentence that `query_class` "is not used by this project" predates 11.15; 11.15 rules.
- **Acceptance criteria:**
  1. `WandsQuery` gains `query_class: str`, after `query_id` and `query`. The value is the raw cell, not
     normalised, stripped or split; an empty cell gives `""` and is not an error (11.15, 11.28).
  2. Everything else `load_queries` does is unchanged: the hash check before parsing, the header check,
     file order, the `query` text as stored.
  3. Existing callers keep working unchanged (`bqc/cli.py` reads `.query` only).
  4. `[q.query_class for q in load_queries(...)]` is the sequence `build_run_report` takes as
     `query_classes` (M3.T14), parallel to `[q.query for q in ...]`; M4.T4 passes it when
     `--catalogue-gap` is given.
- **Tests** (on the existing 10-row fixture):
  - The classes of all ten rows, in file order, hand-read from the fixture (`Wall Art`, ..., `Outdoor`).
  - The two empty-class rows (`drudge report`, `kisner`) give `""`.
  - A class with a space (`Wall Art`) is kept whole and as stored.
  - `_fields` is `("query_id", "query", "query_class")`; this replaces `test_query_class_is_not_returned`,
    the M1.T4 test that pinned the drop.
  - The hash, missing-file and header failure paths still raise as before (existing tests, unchanged).
- **Depends on:** M1.T4, M3.T14.
- **Status:** validated
- **Validation (`opus`, `40f1334`): PASS.** Suite: 585 tests, OK (40 s). `40f1334` touches only
  `bqc/wands.py` (default dropped, comment kept), `tests/test_sanity_loader.py:11,74` (`""` passed),
  `tests/test_wands.py` (items 2 and 3 as specified, plus `test_query_class_is_required` pinning
  item 1) and a one-line `PLAN.md` note; `386bf88` is unchanged. No other two-argument `WandsQuery`
  construction remains. Probe: the no-pipe assumption is closed in `DESIGN.md` 13.9 (`1152683`); the
  real-file probe above stands. Mutants, each in isolation then reverted: `row[2].strip()` caught by
  `test_query_class_is_not_stripped`; `row[2] if len(row) > 2 else ""` caught by
  `test_row_without_query_class_cell_raises_naming_path`. Lines within 100 columns. Process note:
  `sonnet` added the "Corrections addressed" line to `PLAN.md`, which `opus` owns; harmless,
  recorded as at M3.T14.
- **Validation (`opus`, `386bf88`): CHANGES REQUESTED.** Suite: 582 tests, OK (40 s). Criteria 2-4 are met:
  `load_queries` changes only in reading `row[2]`, so the hash-before-parse, header, order and raw-query
  tests pass unchanged; `bqc/cli.py:157-159` and `bqc/sanity.py:64` read `.query` only; the new field is
  parallel to `.query` by construction. Probe by `opus` (2026-11-06, Python 3 `csv`): the real `query.csv`,
  fetched into a scratch directory (SHA-256 matches `data/CHECKSUMS`), has 3 cells on all 481 lines, 6 empty
  classes, no class with leading or trailing whitespace; the changed loader returns 480 rows, 6 with `""`,
  189 distinct classes. So `sonnet`'s deviation 2 can't fire on the pinned file. Mutation run by `opus`,
  11 mutants, each in its own copy (`test_wands`, `test_fetch`, `test_sanity_loader`, `test_cli_generate`;
  the unmutated copy passes): 9 caught (class `""`, class from `row[1]`, `.lower()`, split on space,
  `or None`, field dropped, field before `query`, first data row skipped, `query` whitespace-collapsed).
  Two survive, items 2 and 3. Item 1 is `sonnet`'s deviation 1.
  1. `bqc/wands.py:38`: `query_class: str = ""` makes the field optional, where criterion 1 asks for a
     required field. Under 11.28 `""` means "defines no class", so a construction that forgets the class
     silently drops that query from the gap removal instead of failing with `TypeError`. Criterion 3 is
     about the production caller (`bqc/cli.py`), which never constructs `WandsQuery`; the only two-argument
     constructions are in tests. Fix: drop the default (keep the comment), and pass a class in
     `tests/test_sanity_loader.py:11` (`WandsQuery(0, "smoky mountain tray", "")`, likewise for id 3) and
     `:74` (`WandsQuery(0, "OAK  DESK", "")`). The value is irrelevant there; `load_sanity` reads `.query`.
     Touching `tests/test_sanity_loader.py` is authorised by this item.
  2. `tests/test_wands.py:116-121`: a row with two cells now raises `WandsError` ("malformed row", via the
     `IndexError` handler), where before it loaded; nothing pins this. Mutant `row[2] if len(row) > 2 else
     ""` survives. The loader's failure paths are pinned (M1.T4), and this is a new one. Fix: add
     `test_row_without_query_class_cell_raises_naming_path`, writing `query_id\tquery\tquery_class\n0\tfoo\n`
     to a temp file with its own checksum (as `test_malformed_row_raises_naming_path` does) and asserting
     `WandsError` naming the path.
  3. `tests/test_wands.py:73-74`: criterion 1 says the cell is "not normalised, stripped or split", but no
     fixture class has surrounding whitespace, so mutant `row[2].strip()` survives. Fix: add a temp-file test
     (same pattern as item 2, the shared fixture stays unchanged) with a class cell ` Rugs ` and assert
     `query_class == " Rugs "`.
  - Corrections 1-3 addressed by `sonnet` in the `fix(M3.T16)` commit following `d1ffea3`.
  - Process deviation, as at M3.T14: `386bf88` carries the `fable` co-author trailer and sets the status in
    `PLAN.md`; neither changes the code under review.

### M3.T10: Final models

- **Goal:** fit one final model per model type in each arm, and pickle it with its threshold.
- **Design refs:** `DESIGN.md` `5ba7cee`: 3.2, 8.2, 8.3, 8.7, 10; `a8b8600`: 11.30.
- **Files:** `bqc/evaluate.py`, `tests/test_final_model.py`.
- **Acceptance criteria:**
  1. **Primary arm:** `n` spans with seed `SEED`, and positives from all 24 templates rendered from the
     spans' occurrences at `n // 24` each. The threshold is the M3.T4 rule over all 480 real queries,
     scored by that model, with no out-of-fold pass. No real query is in the fit.
  2. **Comparison arm:** all 960 nominal samples, with positives from the occurrences of all 480 real
     queries. The threshold comes from the M3.T13 inner procedure on all 960 rows, as 8.7 names it:
     `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)` with the groups of 8.1,
     out-of-fold scores, then the M3.T4 rule. The model is then fitted on all 960.
  3. Family E is excluded from both. Rule 4 applies as in M3.T3: a positive equal to a sanity query
     fails; one equal to a training negative, a real query or a P string is dropped and counted.
  4. The pickle goes to `results/` and holds the scorer and threshold, and loading it
     reproduces the same scores.
  5. Every final-model score array, inner out-of-fold scores included, goes through the M3.T5
     finite-score check (11.30). A final model has no outer fold, and the context says so.
- **Tests** (with the fake):
  - A `NaN` in final-model threshold scores raises the 11.30 error, naming arm, model and score set.
  - Primary: no real query or E in the fit input; one fit; the threshold is the M3.T4 function over all
    480 scores.
  - Comparison: 960-equivalent fit input with no E; the threshold comes from out-of-fold scores.
  - Pickle round trip gives identical scores for both arms.
- **Depends on:** M3.T5, M3.T13.
- **Status:** validated
  - `fit_final_model` / `save_final_model` / `load_final_model` in `bqc/evaluate.py`; `FinalModel` (arm, model,
    scorer, threshold, fit_warnings, report) is what is pickled, as `final-<arm>-<model>-<UTC stamp>.pkl`, never
    overwritten. `check_finite` takes `repeat=None` (and `fold=None`) for a final model: the message says
    "final model". The inner out-of-fold loop of `run_comparison_fold` moved to `_inner_out_of_fold`, shared.
    Real scorers (M4) must be module-level classes to pickle.
  - **Validation (opus, 2026-11-06): CHANGES REQUESTED.** Suite 601 OK. Mutation pass (copy of `bqc/`, `tests/`,
    `data/`): 15 of 18 mutants killed (E rendered, all 30 templates in the quota, `n // 25`, refit on fewer rows,
    3 real queries missing from the comparison negatives, a fake `repeat=0` in the context, drops zeroed in the
    report, real queries in the primary fit, inner seed `SEED + 1`, sanity list not screened, part numbers not
    dropped, span seed `SEED + 1`, no finite check on the primary threshold scores). Three survive:
    1. `tests/test_final_model.py:30`: `N = 120` gives `120 // 23 == 120 // 24 == 5`, so a primary quota of
       `n // 23` (`bqc/evaluate.py:971`) passes every test (mutant run: OK). Required: pin the quota with an `n`
       where `n // 23`, `n // 24` and `n // 25` all differ (e.g. 299: 13, 12, 11), in the primary
       positive-count assertion of `test_one_fit_on_spans_and_all_24_templates_without_e_or_real`.
    2. `tests/test_final_model.py:85` and `:124`: every `REAL` string is alphabetic, so `FakeScorer` scores all
       of them 0.0 and `select_threshold` gives `(0.0, 1.0, ...)` over any subset. Scoring the threshold over
       `real_queries[:-5]` (primary, `bqc/evaluate.py:986`) or `out_of_fold[:len(negatives) - 5]`
       (comparison, `bqc/evaluate.py:993`) passes every test. Required: in both tests give the real queries
       distinct scores through `Factory` overrides (e.g. `REAL[i] -> (i + 1) / 100`, so the dropped tail
       holds the top scores) and assert the full `Threshold` (value and `fpr`) equals `select_threshold` over
       all `len(REAL)` scores at a target where a subset would differ (e.g. 0.1).
    3. `bqc/evaluate.py:980`: rendering positives with `SEED + 1` instead of `SEED` passes every test. 8.7 and
       section 10 make `SEED` the final model's seed. Required: a test that the primary fit's positives equal the
       screened kept samples of `registry.render(vocabulary, SEED, templates, n // 24)` with the span
       vocabulary (or an equivalent pin that fails under `SEED + 1`).
    Corrections addressed in `4682cf4` (`test(M3.T10)`, after `f2424dd`; tests only, no production
    change).
    Disclosed points, accepted: the `_inner_out_of_fold` extraction is behaviour-preserving (same groups,
    `SEED + repeat`, `leave_out`; all M3.T13 tests pass) and belongs in this commit since the final model is
    its second caller (DRY). `check_finite(repeat=None)` saying "final model" meets criterion 5 and 11.30.
    Pickling the whole `FinalModel` is fine: `arm` and `model` identify it for M3.T11/M3.T12, and `report`
    holds only `_side` counts and `build_spans` removal counts, no text. The pickling finding sits next to
    the code that depends on it (allowed by `CLAUDE.md`); its consequence for M4 is now an acceptance
    criterion of M4.T1, M4.T2 and M4.T3. No `check_fold` on the final fit's input: not required, and the
    comparison arm's inner folds are checked.
  - **Re-validation (opus, 2026-11-06): PASS.** Suite 605 OK. `4682cf4` touches only
    `tests/test_final_model.py` and `PLAN.md`; `84b4a33` is untouched (`git diff 84b4a33 HEAD` lists
    no `bqc/` file). The three surviving mutants, each run alone in a copy and reverted: `n // 23`
    fails `test_quota_is_n_floor_24_per_template`; `real_queries[:-5]` and
    `out_of_fold[:len(negatives) - 5]` fail
    `test_threshold_covers_every_real_query_with_distinct_scores` in `PrimaryTest` and
    `ComparisonTest`; `SEED + 1` in the primary render fails
    `test_positives_are_rendered_with_seed_from_the_span_vocabulary`. `Threshold(0.925, 0.1, False,
    None)` checked by hand against M3.T4: scores 1/40..40/40, four flagged at 37/40 gives FPR 0.1
    within target, five gives 0.125, and 0.1 is not below 1/40, so no warning; over 35 scores it
    would be 33/40 at 3/35. Process notes: the render-seed test filters by raw equality with `REAL`
    and the spans instead of `screen_samples` (which compares after `normalise`); accepted, since
    the fixtures have no sanity or part-number lists and a normalised-only collision would make the
    test fail loudly, not pass falsely. The `Threshold` import sits out of alphabetical order
    (cosmetic; no linter is configured).

### M3.T11: Sanity section of the run

- **Goal:** score the sanity set with the final models when it's fully labelled, and skip
  it otherwise.
- **Design refs:** 8.6.1, 11.20, 11.21, 11.30.
- **Files:** `bqc/evaluate.py`, `bqc/sanity.py` (`SanityError.query_id`, criterion 3), `tests/test_sanity_section.py`.
- **Acceptance criteria:**
  1. **Blank or partial file:** the message is printed, the run JSON records
     `"sanity": {"status": "skipped", "unlabelled": n}`, and everything else proceeds.
  2. **Fully labelled file:** reports the raw-count confusion matrix and the full list of
     disagreements (id, query, user's label, model's score), per model and arm, using the
     threshold of M3.T10. There is no macro F1, ROC-AUC, confidence interval or decimal
     rate.
  3. A duplicate of a WANDS query fails the run with the offending `id` (11.21) and without the query
     text: `SanityError`'s message quotes the query, so the id needs its own attribute. The loader is
     given all the WANDS queries of the run, never an empty list: the real-file test in M1.T6
     can't make this check, because tests don't read `data/raw/`.
  4. Sanity data is never passed to any `fit` or threshold selection.
  5. Sanity scores go through the M3.T5 finite-score check (11.30); the message carries no query text.
- **Tests** (with the fake, and fixture files; none uses the real labels):
  - A `NaN` in sanity scores raises the 11.30 error, naming the score set.
  - Blank, partial and bad-label fixtures give the skip record, with the rest of the
    report present (11.20).
  - A labelled fixture gives the expected counts and disagreements, and no forbidden
    metric keys.
  - Fit inputs contain no sanity text.
  - Duplicate raises (11.21).
- **Depends on:** M1.T6, M3.T8, M3.T10.
- **Status:** validated
  - `sanity_section(path, wands, finals)` in `bqc/evaluate.py` returns the `sanity` body section; it is
    separate from `build_run_report` (the finals are fitted by `fit_final_model`, M3.T10) and not wired
    into the CLI. Scored form: `{"status": "scored", "models": {arm: {model: {"threshold",
    "confusion_matrix" [[tn, fp], [fn, tp]] by user label x flagged, "disagreements"}}}}`. Scores use
    score-set name "sanity scores". `SanityError` (in `bqc/sanity.py`, one small edit outside the Files
    list) gained `query_id`; `sanity_section` re-raises a duplicate without the query text.
  - **Validation (opus, 2026-11-06): CHANGES REQUESTED.** Suite 624 OK. Mutation pass, each mutant alone in a
    fresh copy of `bqc/`, `tests/`, `data/`, run against `test_sanity_section`, `test_sanity_loader` and
    `test_final_model`: 26 of 28 killed (matrix transposed, `confusion_matrix` arguments swapped, disagreement
    `score` from the reversed row, `label` as a string, loader labels left as strings, flagging at another
    model's threshold, `>` for `>=`, `check_finite` removed, score-set name changed, duplicate text in the
    message, original duplicate re-raised, loader given `wands[:1]`, `wands[1:]` or `[]`, empty-wands guard
    removed, `unlabelled` as the row count, print removed, disagreements narrowed to FPs or widened to all,
    a refit on the sanity set, `query_id` not passed by the loader, one model per arm overwriting another,
    an `fp_rate` key added, the query text altered, sanity texts appended to either arm's negatives). The
    real `data/sanity.csv` is read by no test of this task (`SANITY_PATH` appears only in the M1.T6
    structure test, which asserts no label). Two survive:
    1. `tests/test_sanity_section.py:98`: the test named "...own entry and threshold" never asserts
       `threshold`, so reporting `finals[0].threshold.threshold` for every model (`bqc/evaluate.py:1060`)
       passes the suite (mutant run: OK). Required: assert `models[PRIMARY]["mlp"]["threshold"] == 0.15` and
       `models[COMPARISON]["lr"]["threshold"] == 0.85` (and `0.5` for the primary `lr`) in that test.
    2. `tests/test_sanity_section.py:198`: criterion 4's test is vacuous for the primary arm's span screen.
       No `PRODUCTS` span equals a `sanity_full.csv` query, so passing `[]` instead of `sanity` to
       `build_spans` in `fit_final_model` (`bqc/evaluate.py:969`) passes the full suite (624 OK). Required:
       make the sanity list hold a string the primary fit would otherwise contain (e.g. a negative taken from
       a `fit_final_model(..., sanity=[], ...)` fit; `pwhhaj pwhhba` at `n=120` with `build_registry()`), then
       assert it is absent from every fit input and `report["span_removals"]["sanity"] >= 1` (checked by
       hand: 1). Keep the fixture queries in the list too.
    Disclosed points: (1) TDD order slipped, recorded as process deviation F1; with the mutation passes above
    the tests are shown to bite except for items 1-2, so no re-run of the order is required. (2) The
    `bqc/sanity.py` edit is minimal (an optional `query_id` constructor argument, defaulting to None, and one
    call-site argument); accepted and added to the Files line. (3) `sanity_section` as a separate call is
    accepted: M4.T4 criterion 1 assembles the run JSON "including the sanity section or its skip record",
    and the M3.T11 skip test composes it with a `build_run_report` body through `write_report`, which shows
    the rest proceeds. (4) The `threshold` key is kept: criterion 2 names "the threshold of M3.T10", and a
    disagreement's score is unreadable without it; it is a cut-off, not a rate. It must be pinned (item 1).
    (5) `in progress` was not set (process); the "Next task" line now reads M3.T11 (corrections), then M3.T12.
  - **Corrections (sonnet, 2026-11-06):** items 1 and 2 addressed in tests only; see the `test(M3.T11)` commit.
  - **Re-validation (opus, 2026-11-06): PASS.** `aa3a1d8` touches tests and PLAN.md only; `bqc/` is
    unchanged since `d95f843`. Mutant `finals[0].threshold.threshold` is caught by
    `ScoredTest.test_every_model_and_arm_has_its_own_entry_and_threshold` (0.5 != 0.15); mutant `[]` for
    `sanity` at `build_spans` in `fit_final_model` is caught by
    `NoLeakTest.test_fit_inputs_contain_no_sanity_text` [product]
    (`span_removals["sanity"]` 0 < 1). Suite: 624 tests OK.

### M3.T12: Part-number diagnostic

- **Goal:** score the 100 part-number strings with each final model, and report how many would
  be discarded.
- **Design refs:** 8.6 (item 4), 8.7, 11.23, 11.30.
- **Files:** `bqc/evaluate.py`, `tests/test_part_number_diagnostic.py`.
- **Acceptance criteria:**
  1. Scores the M2.T9 set with each final model of M3.T10, at that model's threshold.
  2. The run JSON has a `part_number_diagnostic` section, separate from the headline, giving per
     model the fraction flagged, overall and per shape (`template_id`), and the min, median and max
     score.
  3. Flagging uses the one `score >= t` rule of M3.T4.
  4. The set is never used in any fit or threshold selection, and doesn't change any other
     metric. Part-number scores are reported, not selected on.
  5. Part-number scores go through the M3.T5 finite-score check (11.30).
- **Tests** (with the fake):
  - A `NaN` in part-number scores raises the 11.30 error, naming the score set.
  - No `P` text in any fit input, including the inner folds.
  - 100 scores for each model.
  - The fraction is computed with `score >= t`: a score exactly at `t` counts as flagged.
  - Hand-computed fraction, overall and per shape, min, median and max.
  - The other sections of the report are unchanged with and without the diagnostic.
- **Depends on:** M2.T9, M3.T8, M3.T10.
- **Status:** validated
  - `part_number_section(part_numbers, finals)` in `bqc/evaluate.py` returns the `part_number_diagnostic` body
    section; separate from `build_run_report` (as `sanity_section`) and not wired into the CLI. Form:
    `{"models": {arm: {model: {"threshold", "n", "flagged_fraction", "per_shape": {template_id: {"n",
    "flagged_fraction"}}, "min", "median", "max"}}}}`. Score-set name "part-number scores".
  - **Validation (opus, 2026-11-06): PASS.** Suite 636 OK. `7c9e244` adds `part_number_section` only
    (no other `bqc/` change). Mutation pass, each mutant alone in a fresh copy of `bqc/`, `tests/`, `data/`,
    full suite: 23 of 23 killed (`>` for `>=`; one-key grouping; mean for median; per-shape fraction over the
    overall count; score-set name changed; min/max swapped; a fit on the part numbers; per-shape `n` as the
    overall count and off by one; overall `n` off by one; another model's threshold for the flag, the key,
    and both; a sample scored twice; a sample skipped; `check_finite` removed; `max` and `median` swapped;
    `check_finite` given `repeat=0, fold=0` (loses "final model"); wrong model in the context; empty guard
    removed; every arm collapsed to the first; overall fraction denominator changed; only the first final
    scored). Criteria: (1) each final scored at its own threshold, 100 scores per model on the generated set;
    (2) the section is a separate function, as `sanity_section`; `{"models": {arm: {model: ...}}}` gives
    per model the overall and per-`template_id` fraction and min/median/max, which meets "per model" (arm
    and model together identify a final, as in M3.T11); (3) `is_illegitimate`, exact-`t` test; (4) a
    raising `fit` and fit-input checks over both arms, comparison inner folds included; (5) `check_finite`
    with `repeat=None`, "part-number scores". Weighed: the `threshold` key per entry follows the accepted
    M3.T11 ruling (a cut-off, not a rate). The P drop list was not mutated here; it is already pinned by
    `tests/test_folds.py:404-420` (M3.T3) and `tests/test_final_model.py:199-205` (M3.T10, mutant "part
    numbers not dropped" killed in its validation), so `NoLeakTest.test_fit_inputs_contain_no_part_number_text`
    guards only the diagnostic's own fit-freedom, which is the M3.T12 concern. Note for M4.T4: the section is
    not wired into the CLI yet, as with M3.T11.

---

## M4: Real models and paired comparisons

**Sign-off: 2026-11-06, `fable`, verdict POSITIVE** (audited at `07ee3ac`; `DESIGN.md` `7034165`).

- Suite: `.venv/bin/python -m unittest discover -s tests`, 768 tests, OK in 134 s, run by `fable`.
- Diff read: the whole of `bqc/models.py`, `paired_comparison` and the `build_run_report` changes
  in `bqc/evaluate.py`, and the `evaluate` parts of `bqc/cli.py` (parser, `_FitTimer`,
  `_not_run_reasons`, `_evaluate`), plus every M4 review record. Invariants: the three adapters
  are module-level `vectoriser -> estimator` classes behind the `Scorer` port with exactly the
  parameters of 6.1, 6.2 and 6.3, pinned by full `get_params()`; stages still import no vendor
  SDK (`bqc/evaluate.py` imports nothing from `bqc/cli.py` or `sklearn` estimators); the MLP's
  `random_state` is `SEED`; the paired comparisons follow the one verdict rule of section 7; a
  failed run writes no run JSON; tests never download or read `data/raw/`.
- Real run (M4.T4, recorded in 13.9): 12 min 1 s wall, one MLP fit at 5,000 / 5,000 in 39.7 s,
  under the 5-minute rule; sanity section `scored`.
- Rulings recorded in `DESIGN.md`: 13.6 (early stopping needs 3 + 3 rows), 11.18 (`timing`
  excluded from the determinism rule).

Follow-ups:

- **F1 (process):** the TDD order slipped at M3.T11, M4.T4 and M4.T8 (implementation before
  tests). The M3 sign-off rule held: every task had an isolated mutation pass by `opus`, and each
  survivor became a correction. Recorded without penalty.
- **F11 (for the user, via M6.T1):** the first real run's pooled primary-arm numbers at target
  FPR 0.01: `lr` recall 0.770 / FPR 0.0049; `mlp` 0.763 / 0.0118 (point estimate above target,
  CI 0.0053-0.0261); `trivial` 0.613 / 0.0090. The trivial baseline is close to the MLP on recall
  (8.6 item 2), so the generators may be too easy. Nothing acts on this before M6.T1 reports it
  with the paired comparisons and the user decides.
- **F12 (optional):** two tests in `tests/test_cli_evaluate.py` (the over-five-minutes message
  pair) still fit the real MLP; a fake pins them equally.
- **F13 (noted):** the MLP fit time (39.7 s) is above the 13.22 median of 17 s; different
  machine, same software versions. The section 10 budget is unaffected.

### M4.T1: Logistic regression adapter

- **Goal:** the `Scorer` for 6.2.
- **Design refs:** 5, 6.2, 6.4, 11.1, 11.3.
- **Files:** `bqc/models.py`, `tests/test_models_lr.py`.
- **Acceptance criteria:** `LogisticRegression(C=1.0, max_iter=1000)` on the M3.T1
  vectoriser. The adapter calls `normalise()` on every text in `fit` and `score`, once, inside
  the adapter (section 5). Scores are P(illegitimate), the probability of label `1`, in `[0, 1]`.
- **Tests:**
  - On tiny inputs: `score` is aligned with the class-1 probability.
  - `"  SOFA "` and `sofa` score the same.
  - Fitting on raw variants (`"  SOFA "`) gives the same scores as fitting on their normal forms.
  - Empty and all-unknown strings give the same score, and don't raise (11.1, 11.3).
  - Calling `score` before `fit` raises.
  - No metric values are asserted (section 10).
  - Pickle round trip of a fitted adapter (module-level class, M3.T10) gives identical scores.
- **Depends on:** M3.T1, M3.T2.
- **Status:** validated
  - **Validation (opus, 2026-11-06): CHANGES REQUESTED.** Suite 648 OK (43 s). Met by reading: the
    adapter builds its vectoriser with `make_vectoriser()`, calls `normalise()` once per text in `fit` and
    `score`, returns `predict_proba[:, 1]`, and checks fit state on the empty-list path; no new dependency; no
    metric value asserted. Mutation pass (`bqc/models.py` alone in a `git archive` copy, `tests.test_models_lr`
    run per mutant), 20 mutants: killed are column 0, labels inverted, `C=100`, `max_iter=100` and `=5`,
    `char_wb` analyser, vectoriser refitted in `score` (`fit_transform`), `check_is_fitted` dropped, `[]`
    returned for empty input, `fit` returning None. Equivalent and accepted: labels cast to `bool` or `str`
    (`classes_` sorts the same, column 1 is still label 1), `C=1` int, `random_state=0` added (lbfgs ignores
    it), `float32` output. Gaps:
    1. `tests/test_models_lr.py:62`: criterion 1 ("on the M3.T1 vectoriser") is not pinned. Replacing
       `make_vectoriser()` at `bqc/models.py:34` with a locally built `TfidfVectorizer` with one parameter
       changed survives for `min_df=1`, `ngram_range=(1, 3)` and `sublinear_tf=False` (all OK). Required: in
       `test_configuration`, assert `s._vectoriser.get_params() == make_vectoriser().get_params()`, or patch
       `bqc.models.make_vectoriser` and assert the adapter's vectoriser is the object it returned.
    2. `tests/test_models_lr.py:36-45`: the criterion says `normalise()` runs once per text, and
       `normalise(normalise(t))` in `fit` (`bqc/models.py:39`) or `score` (`:51`) survives because
       `normalise` is idempotent. Required: one test that patches `bqc.models.normalise` with
       `unittest.mock.patch(..., wraps=normalise)` and asserts `call_count == len(texts)` after `fit` and,
       reset, after `score`.
    3. `tests/test_models_lr.py:55` and `:59`: `assertRaises(Exception)` passes for any error (a `TypeError`
       from a broken `score` would satisfy it). The designed error is sklearn's `NotFittedError` (docstring at
       `bqc/models.py:45`; probed now, sklearn 1.9.1: the non-empty path raises it from the vectoriser, the
       empty path from `check_is_fitted`). Required: `assertRaises(NotFittedError)` in both tests.
    Weighed and accepted: (a) no `random_state`: 6.2 names only `C` and `max_iter`, lbfgs does not use
    `random_state`, and `test_refit_is_deterministic` pins it; section 10's seed rule covers estimators that
    take one. (b) Single-class training raises sklearn's `ValueError` unhandled: the `Scorer` port does not
    promise more, and 11.x puts the both-classes guard on the folds (`check_fold`), not on the adapter.
    (c) `np.empty(0)` (float64, shape `(0,)`) after `check_is_fitted` matches the port's 1-D float array and
    avoids sklearn's 0-sample `ValueError` (re-probed). (d) Probe findings as comments next to the code
    (`bqc/models.py:36`, `:46-48`, `:50`) satisfy `CLAUDE.md` ("in DESIGN.md section 13 or in a comment next to
    the code that depends on it"); re-run now, same results. Notes, not required: `scratch/models.orig` is a
    leftover backup identical to `bqc/models.py` (gitignored; delete it); stray blank line between the two
    sklearn imports at `bqc/models.py:7`.
  - **Corrections (sonnet, 2026-11-06):** items 1-3 addressed in tests; see the `test(M4.T1)` commit.
  - **Re-validation (opus, 2026-11-06): PASS.** Suite 650 OK (47 s). `f653dee` addresses items 1-3:
    `test_uses_m3_vectoriser` compares `get_params()` with `make_vectoriser()`; `test_normalise_applied_once_per_text`
    spies with `wraps=normalise` and asserts 24 calls in `fit`, 2 in `score`; both not-fitted tests use
    `NotFittedError`. The only production change is the removed blank line at `bqc/models.py:7`; `b04b00d` is
    unchanged in history. The five surviving mutants, re-run in a `git archive` copy, are now killed: local
    `TfidfVectorizer` with `min_df=1`, `ngram_range=(1, 3)` or `sublinear_tf=False` (by
    `test_uses_m3_vectoriser`), `normalise` twice in `fit` (48 != 24) and in `score` (4 != 2).

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
  - Pickle round trip of a fitted adapter (module-level class, M3.T10) gives identical scores.
- **Depends on:** M4.T1.
- **Status:** validated
  - **Corrections (sonnet, 2026-11-06):** items 1-3 below addressed in `52be56a` (`fix(M4.T2)`).
  - **Re-validation (opus, 2026-11-06): PASS.** Suite 666 OK (44 s; 667 less the dropped
    `test_seed_is_passed_through`). `52be56a` touches only `bqc/models.py` (seed argument dropped,
    `random_state=SEED`), `tests/test_models_mlp.py`, `tests/test_models_lr.py` and `PLAN.md`; `111b602`
    untouched; no caller passed a seed. Item 1: scorer built and `max_iter=1` set inside the
    `catch_warnings` block. Item 2: full `get_params()` equality for both adapters. Item 3: `fitted()` takes
    no seed, the pass-through test is gone, `test_configuration` pins `SEED`. The three survivors, re-run
    in a `git archive` copy, are now killed: `simplefilter("ignore", ConvergenceWarning)` in `__init__` by
    `test_models_mlp.test_convergence_warning_not_suppressed`; MLP `tol=1e-2` by
    `test_models_mlp.test_configuration`; LR `class_weight="balanced"` by
    `test_models_lr.test_configuration`. Changed code lines within 100 columns.
  - **Validation (opus, 2026-11-06): CHANGES REQUESTED.** Suite 667 OK (43 s). Met by reading and by test:
    the 6.1 parameters (named keys only, see item 2), the M3.T1 vectoriser (`get_params()` equality),
    `random_state=SEED`, fit returns self, same seed gives identical scores, `""` and `"   "` score without
    raising, `max_iter=1` warns and fit returns, pickle round trip, `NotFittedError` on both paths; no metric
    asserted; no new dependency; the committed `bqc/models.py` has no stray import (all eleven imports used).
    Mutation pass (`git archive` copy, `tests.test_models_mlp` + `tests.test_models_lr` per mutant), 26
    mutants, 22 killed: every 6.1 parameter changed, `early_stopping=False`, `validation_fraction` 0.1 and
    0.25, `n_iter_no_change=5`, `batch_size=32`, `random_state` `seed + 1`, `0`, dropped, or moved to the
    vectoriser, `normalise` dropped in `fit` only and in `score` only, column 0, `check_is_fitted` dropped, `fit`
    returning None, LR `C=10` and `max_iter=100`, LR with a local vectoriser. Equivalent: `warnings.filterwarnings
    ("once", ConvergenceWarning)` in `fit` (probed, Python 3.13.18: re-registering a filter clears the once
    registry, so each fit still warns inside the runner's `catch_warnings`). Survivors:
    1. `tests/test_models_mlp.py:94-101`: `warnings.simplefilter("ignore", ConvergenceWarning)` in
       `MLPScorer.__init__` survives, because the test builds the scorer at `:95`, outside its
       `catch_warnings` block, so its own `simplefilter("always")` overrides the mutant. The runner builds the
       scorer inside the block (`bqc/evaluate.py:433-435`, `make_scorer().fit(...)`), where the mutant would
       hide the warning (11.12). Required: construct the scorer (and set `max_iter=1`) inside the
       `with warnings.catch_warnings(record=True)` block.
    2. `tests/test_models_mlp.py:83-88`: "exact parameters of 6.1" is pinned only for the named keys; adding
       `tol=1e-2` survives. Required: assert `MLPScorer()._classifier.get_params() ==
       MLPClassifier(<6.1 parameters>, random_state=SEED).get_params()`. Same gap in the LR class this commit
       rewrote: `class_weight="balanced"` survives `tests/test_models_lr.py:74-76`; required: the same full
       `get_params()` comparison against `LogisticRegression(C=1.0, max_iter=1000)`. (Both checked to kill
       their mutant in the copy.)
    3. `bqc/models.py:76` and `:80`: `seed: int = SEED` is YAGNI. 6.1 fixes `random_state=SEED`; every caller
       is a zero-argument factory (`Callable[[], Scorer]`, `bqc/evaluate.py:416`, `:482`, `:505`, `:725`,
       `:942`); `SEED + repeat` (8.2, section 10) seeds splits and spans, not the estimator. Required: drop the
       argument and pass `random_state=SEED`; in the tests drop the `seed` parameter of `fitted()` (`:23`) and
       `test_seed_is_passed_through` (`:35-37`); `test_configuration` already pins `SEED`.
    Weighed and accepted: (a) the `_VectorisedScorer` base class in the feature commit: DRY says to extract on
    the second occurrence, which is this task, and TDD step 5 puts the refactor inside the task; it can't be
    split now without rewriting history. The untouched LR tests pass and kill LR mutants on `C`, `max_iter`, the
    vectoriser and `normalise`; item 2 closes the one LR gap found. (b) `max_iter=1` set on the private
    `_classifier` in the test: no option added to production code, the tests already read `_classifier`.
    (c) Probe findings in the `MLPScorer` docstring and the fixture comment satisfy `CLAUDE.md`; re-checked
    (3 + 3 rows is the smallest balanced fit). For `fable`, not required: 13.6 could carry the minimum-rows
    finding, since the 11.18 tiny-config determinism test will fit the MLP on small data.

### M4.T3: Trivial baseline adapter

- **Goal:** the `Scorer` for 6.3, the artefact detector.
- **Design refs:** 6.3, 6.4, 8.6.
- **Files:** `bqc/models.py`, `tests/test_models_trivial.py`.
- **Acceptance criteria:** logistic regression on exactly three features (character
  length, word count, non-alphanumeric ratio), computed on normalised text. Marked as
  reported, never selected.
- **Tests:** hand-computed features for a few strings, including `""` with no division by
  zero (11.1); scores in `[0, 1]`; pickle round trip of a fitted adapter (module-level class,
  M3.T10) gives identical scores.
- **Depends on:** M4.T1.
- **Status:** validated
  - **Validation (opus, 2026-11-06): CHANGES REQUESTED.** Suite 682 OK (43 s). Met by reading and by test:
    exactly three features in the documented order (`test_hand_computed`, `(0, 3)` for no texts); the ratio
    counts spaces and is 0.0 for `""` (11.1); length, ratio and normalise-before-features pinned by
    `test_features_are_computed_on_normalised_text` and `test_fit_normalises`; scores in `[0, 1]`; pickle round
    trip; full `get_params()` against `LogisticRegression()`; `normalise` once per text (24 / 2);
    `NotFittedError` on both paths; "reported, never selected" in the docstring, and no code references
    `TrivialScorer` yet, so nothing selects on it; no metric asserted; no new dependency; probe in the docstring.
    Mutation pass (`git archive` copy, `tests.test_models_trivial` + `_lr` + `_mlp` per mutant), 22 mutants,
    17 killed: missing `.reshape(len(texts), 3)` (re-run: killed by `test_no_texts_gives_zero_by_three`), ratio
    counting only punctuation (`and not c.isspace()`), `isalpha` for `isalnum`, ratio over `len + 1`, `""` ratio
    1.0, length in UTF-8 bytes, `split(" ")`, columns swapped, a fourth feature (with and without the reshape
    adjusted), features on raw text with `normalise` still called once, no `normalise`, `FunctionTransformer`
    `validate=True`, LR `C=10`, `max_iter=1000`, `class_weight="balanced"`, base `score` using `fit_transform`
    (killed by the LR and MLP tests; for this stateless transformer it is equivalent). Equivalent:
    `/ max(len(t), 1)`; `FunctionTransformer` with `accept_sparse=True`, `check_inverse=False` or
    `feature_names_out="one-to-one"` (no effect with `validate=False` and no inverse). Survivor:
    1. `tests/test_models_trivial.py:61-65`: word count taken from the raw text (length and ratio still from
       the normalised text) survives, because every input there has the same word count raw and normalised
       (`str.split` already ignores the extra spaces). The criterion "computed on normalised text" is pinned for
       two of the three features. Required: add an input whose word count changes under `normalise`, e.g.
       `self.assertEqual(m.score(["sofa \u200b bed"]).tolist(), m.score(["sofa bed"]).tolist())` (U+200B is
       Cf and not whitespace, so the raw text splits into 3 words and the normalised `"sofa bed"` into 2;
       checked to kill the mutant in the copy).
    Weighed and accepted: (a) `FunctionTransformer(_hand_features)` as `_vectoriser`: honest DRY. The base
    owns normalise-once, `predict_proba[:, 1]` and the empty / not-fitted path, the knowledge 6.4 puts in one
    place; a hand-feature transformer is a vectoriser in 6.4's `vectoriser -> estimator` sense. Skipping
    `super().__init__()` avoids building an unused TF-IDF. Note, not required: the base docstring
    (`bqc/models.py:33`) says "Subclasses set `_classifier`"; it could add "and may replace `_vectoriser`".
    (b) `LogisticRegression()` defaults on unscaled features: 6.3 names no settings, the runner records every
    fit warning (`simplefilter("always")`, `bqc/evaluate.py:433-435`), so a `ConvergenceWarning` would be
    visible per fold. Re-probed (scikit-learn 1.9.1) on 10,000 synthetic rows, lengths up to 512: no warning,
    `n_iter_` 29. 6.3 needs nothing more.
  - **Re-validation (opus, 2026-11-06): PASS.** Correction `f017dbf`: item 1 met. The new assertion
    `m.score(["sofa \u200b bed"]) == m.score(["sofa bed"])` is in
    `test_features_are_computed_on_normalised_text`. The surviving mutant (word count from the raw text,
    length and ratio from the normalised text), re-run in a `git archive` copy: killed by exactly that test
    (1 of 16 in `tests.test_models_trivial`); with the pre-correction test file it still survives (16 OK).
    `bqc/models.py` change is docstring only (the optional note). Suite 682 OK (43 s); no line over 100 columns.

### M4.T4: `bqc evaluate` command

- **Goal:** run the harness with the real models from the command line, in either arm.
- **Design refs:** `DESIGN.md` `0386ca8`: 8.2, 8.4, 8.5, 8.6, 10, 13.9, 13.22.
- **Files:** `bqc/cli.py`, `bqc/evaluate.py` (`lofo_not_run`, `gap_not_run` on `build_run_report`), `tests/test_cli_evaluate.py`, `tests/test_cli_generate.py` (shared fixture: query classes `C0`-`C7`), `DESIGN.md` (section 13.9, the measurement only).
- **Acceptance criteria:**
  1. `python -m bqc evaluate [--negatives product[,real]] [--negatives-size 5000]
     [--models lr,mlp,trivial] [--target-fpr 0.01] [--lofo [MODELS]] [--catalogue-gap [MODELS]]` runs the chosen models
     and arms and writes one run JSON, including the sanity section or its skip record, and
     `part_number_diagnostic` for each model. `--negatives` takes a comma list. The defaults are
     `product`, 5,000 and all three models.
  2. `--target-fpr` outside `(0, 1)`, a non-positive `--negatives-size`, an unknown model or arm, and a
     `--lofo` or `--catalogue-gap` model outside `--models` are usage errors. A missing raw file exits non-zero with a message pointing at `fetch`. An
     exhausted span pool exits non-zero with its count. A non-finite score (11.30) exits non-zero with its message.
  3. The real run scores the committed sanity labels. The skip path stays covered by fixtures
     (M3.T11).
  4. `--lofo` and `--catalogue-gap` take an optional comma list of models, default all of `--models`;
     naming or omitting `mlp` there is the request (section 10). The main fits and the final
     model always cover every model in `--models`. A model left out has its rows reported as not run,
     with the reason. `--lofo` never runs the MLP in the comparison arm (8.6 item 1). `--catalogue-gap`
     is valid with either arm, and passes the loader's classes (M3.T16) as `query_classes` (11.15).
  5. **Runtime (section 10):** the real run measures one MLP fit at 5,000 / 5,000, and records it in the
     run JSON and, with date and versions, in 13.9. The task report says if it exceeds 5 minutes. For
     scale, 13.22 measured a median of 17 s, about 25 minutes for the primary arm's 91 MLP fits, so the
     5-minute rule is a safeguard that isn't expected to trigger.
- **Tests:** on a tiny config and fixtures: end-to-end with each flag and each arm setting; `product,real`
  writes one file with both arms; JSON has one entry per model, including the diagnostic; `--models`
  limits the entries; `--lofo` has no MLP rows in the comparison arm; `--lofo lr,trivial` and
  `--catalogue-gap lr,trivial` report the MLP's rows as not run, with the reason, while its main fit and
  final model are present; bare `--lofo` covers all of `--models`; the fit time is in the JSON; the
  trivial baseline's report row is present; `--catalogue-gap` adds its section; each usage error, including `--target-fpr nan` (`select_threshold` would flag every query with `target_unmet` false); each
  missing raw file; a stub model returning `NaN` exits non-zero and writes no run JSON;
  **real model and fold determinism end to end** (11.18).
- **Depends on:** M2.T8, M3.T9, M3.T11, M3.T12, M3.T14, M3.T16, M4.T1, M4.T2, M4.T3.
- **Status:** validated
  - **Implementation (sonnet, 2026-11-06):** `_evaluate` in `bqc/cli.py` composes `build_run_report`,
    one `fit_final_model` per arm and model, `sanity_section` and `part_number_section` into one
    `write_report` call (pickles first; a failure before the writes leaves nothing, a
    `write_report` failure leaves the pickles). `build_run_report` gained
    `lofo_not_run` and `gap_not_run` (arm -> model -> reason); the CLI builds them, including the
    comparison-arm MLP rule. Real run, defaults, primary arm: 12 min 1 s, one MLP fit 39.7 s (under
    5 minutes), sanity `scored`; recorded in 13.9. `tests/test_cli_generate.py` fixture: query classes
    `C0`-`C7` instead of `C` (the gap pass needs several classes). Parameters in the run JSON also
    carry `lofo_models` and `catalogue_gap_models`.
  - **Validation (opus, 2026-11-06): CHANGES REQUESTED.** Suite 740 OK in 162 s. Criteria 1-5 are met;
    the gaps are in what the tests pin. Mutation pass in isolated copies, one mutant each, `test_cli_evaluate`
    and `test_run_report`: 39 mutants, 33 killed, 1 equivalent (`query_classes` passed with the gap off,
    ignored by `build_run_report`), 5 survivors, items 1-3 below.
    1. `tests/test_cli_evaluate.py:51` and `:184`: `ClockedScorer` advances the clock only in `fit`, so a
       timer around the whole `fit_final_model` call (spans, renders, threshold scores) still reads 5.0 and
       survives; `:159` survives it too (two `perf_counter` calls either way). Make the fake clock also move
       outside the fit (for example in `score`, or in a patched `build_spans`) and assert the fit's 5.0 alone.
    2. `bqc/cli.py:454`: `lofo_models` and `catalogue_gap_models` in `parameters` are untested; dropping
       either survives. Assert both in the run JSON: `None` when the flag is absent, the named list, and all
       of `--models` for a bare flag (through `_section_models`).
    3. `bqc/cli.py:469-478`: no test reaches the `OSError` branch or the save-before-write order; removing
       `OSError` from the `except` and writing the run JSON before the pickles both survive. Add a test
       where `save_final_model` raises an `OSError`: exit 1, the message on stderr, no `evaluate-*.json`.
       The status note's "a failure leaves neither" holds for every failure before the writes; a
       `write_report` failure after the pickles leaves the pickles (accepted, no run JSON refers to them).
       Reword the note's claim when you record the correction.
    4. `tests/test_cli_evaluate.py:358`: `assert_usage_error` calls `main` with the real `_evaluate`. If a
       parser check breaks, the test falls through to a real run on `data/raw/` writing to `results/`
       (12 minutes on this machine) instead of failing. Patch `cli._evaluate` there, as
       `test_a_usage_error_runs_nothing` does, and assert it was not called.
    5. Suite time (43 s to 162 s; this file is 120 s). Tests that pin a name-based rule fit the real MLP
       where a fake pins the same thing: `:282` (14 s), `:290` (9 s), `:297` (8 s), `:306` (8 s), `:159`
       (7 s, all three models in both arms), `:191` (6 s). Patch `cli.MODEL_FACTORIES["mlp"]` with a
       fake (as `:184` does) in these. Keep the real MLP in the primary end-to-end test, the
       saved-scorer test and `DeterminismTest` (11.18 needs the real models).
    6. `bqc/cli.py:83`: `GenerateError` is now raised by the shared `_load_inputs` and `_part_numbers` for
       both commands; a shared error named after one command misleads. Rename it (for example
       `CommandError`) with its docstring; no test imports it.
    7. `bqc/cli.py:80`: the comment says the constant leaves the MLP out of the sections above 5 minutes;
       the code only reports it (the M6 invocation applies the rule, section 10). Say that.
    Rulings on the disclosed points: `--negatives-size` below 24 as a usage error matches M2.T8 (accepted);
    the `tests/test_cli_generate.py` fixture change is accepted and added to Files, as is `bqc/evaluate.py`;
    `timing` is excluded from the 11.18 comparison by necessity (section 10 puts a wall-clock time in the run
    JSON), and `lofo_models` / `catalogue_gap_models` belong in `parameters` (M3.T8 criterion 1, all
    parameters); sanity text appears only in `sanity.disagreements` of the real run JSON, as 8.6.1 asks;
    13.9 has the date, versions and core count, and the 5-minute rule did not trigger (39.7 s).
  - **Corrections (sonnet, 2026-11-06):** items 1-7 addressed in the `fix(M4.T4)` commit after `c35c96d`.
  - **Correction validation (opus, 2026-11-06): PASS.** `182ea3b` touches only `bqc/cli.py`,
    `tests/test_cli_evaluate.py` and this note, and addresses items 1-7 and nothing else; `c93a92f`
    is unchanged. Suite 743 OK in 124 s (125 s wall; was 162 s). The five survivors plus a disabled
    parser check, one per isolated copy, are all killed: timer around the whole `fit_final_model` by
    `test_the_time_surrounds_the_fit_alone`; `lofo_models` or `catalogue_gap_models` dropped by
    `test_the_section_models_are_in_the_run_parameters` and
    `test_a_bare_flag_records_all_of_models_in_the_run_parameters`; `OSError` removed and run JSON
    before the pickles by `test_a_save_failure_exits_non_zero_and_writes_no_run_json`; early return
    in `_check_evaluate_args` by four `UsageErrorTest` tests in 0.09 s with `_evaluate` never
    started. `GenerateError` survives only in the item-6 quote above. Lines within 100 columns.
    Optional: the over-five-minutes pair (`:207`, `:214`) keys on the model name and the patched
    `perf_counter`, so a fake MLP would pin it equally and save the two real fits.

### M4.T8: MLP-vs-LR and between-arm paired comparisons

- **Goal:** compute the paired differences M6 reports: MLP against LR, and primary against comparison
  arm.
- **Design refs:** `DESIGN.md` `b025fde`: 6.2, 7, 8.2, 12.1 (items 13, 17).
- **Files:** `bqc/evaluate.py`, `bqc/cli.py` (imports the `LR`/`MLP` names), `tests/test_paired_comparison.py`,
  `tests/test_run_report.py` (body key sets).
- **Acceptance criteria:**
  1. One function computes, for each of headline recall, achieved FPR and ROC-AUC: the list of per-fold
     paired differences, first-named minus second-named (15 in a real run), their mean, sample standard
     deviation (`ddof=1`, `statistics.stdev`), standard error (`sd / sqrt(folds)`), and the count of folds
     won. A win is a fold where the first-named is better: higher recall or ROC-AUC, lower FPR.
  2. `justified` is true only if the mean recall difference is positive and greater than twice its standard
     error, and `fpr_gate` holds. `worse` is true if that mean is negative and beyond twice its standard
     error. ROC-AUC gets the same two booleans, informational only (section 7).
  3. `fpr_gate` is true unless the mean FPR difference is positive beyond twice its standard error. Its
     mean, sd and strict sign are reported next to it.
  4. A run with both arms has an `arm_comparison` section: primary minus comparison, per model (8.2).
  5. A run with both LR and MLP has a `model_comparison` section under each arm, in the same shape: MLP
     minus LR (section 7, 12.1 item 13).
  6. No command compares two run files (8.2). There is no `smote_comparison` or `smote_justified`.
- **Tests** (with fakes), hand-computed:
  - Differences, means, sample sd, standard error and win counts, including a lower FPR counting as a win
    and a tied fold counting as none; a case where `ddof=0` would flip `justified`.
  - A recall mean positive beyond noise with a passing gate: `justified`, not `worse`.
  - A negative recall mean beyond noise: `worse`, not `justified`.
  - A recall mean within noise: neither, and a mean exactly at the 2-standard-error bound: neither.
  - An FPR rise beyond 2 standard errors blocks an otherwise justified gain; a positive FPR mean within
    noise, or exactly at the bound, doesn't, and its strict sign is still reported.
  - ROC-AUC's booleans don't change recall's `justified`.
  - The sign is primary minus comparison, and MLP minus LR.
  - No `arm_comparison` with one arm; no `model_comparison` unless both models ran.
- **Depends on:** M4.T4.
- **Status:** validated
  - `paired_comparison(first, second)` in `bqc/evaluate.py` over two `compute_metrics` `folds` lists;
    returns `folds`, `recall`/`fpr`/`roc_auc` (`differences`, `mean`, `sd`, `se`, `wins`), `justified`,
    `worse`, `fpr_gate` (`holds`, `mean`, `sd`, `sign` -1/0/1) and `roc_auc.justified`/`.worse`. Raises
    ValueError on mismatched or duplicate (repeat, fold) pairs and on fewer than two folds (sd undefined;
    no verdict is better than a silent False). `build_run_report` adds top-level `arm_comparison`
    (per model) with both arms and `model_comparison` under each arm when `lr` and `mlp` both ran; the
    names are `LR`/`MLP` constants in `evaluate.py`, which `cli.py` now imports for `MODEL_FACTORIES`.
    Three existing run-report assertions on the body's key set now include `arm_comparison`.
  - **Validation (opus, 2026-11-06): CHANGES REQUESTED.** Suite 767 OK in 178 s. Criteria 1-6 are met by
    reading: one `paired_comparison`, first-named minus second-named, `statistics.stdev`/`fmean` (stdlib,
    no new dependency), `se = sd / sqrt(folds)`, strict `>` against twice the se for `justified`, `worse`,
    the ROC-AUC pair and the FPR gate, the gate in `justified` only, `fpr_gate` with `holds`/`mean`/`sd`/
    strict `sign`; `arm_comparison` only with both arms, `model_comparison` (MLP minus LR) only with both;
    no `smote` key, no cross-file command. `LR`/`MLP` have one home in `evaluate.py`; `evaluate.py` imports
    nothing from `cli.py`. Hand checks redone: `[0.5, 0.25, 0.0, 0.25]` gives ss 0.125, sd sqrt(0.125/3),
    se sd/2, 3 wins; `[0.625, 0.125, 0.125, 0.125]` gives mean 0.25, ss 0.1875, sd 0.25, 2 se 0.25 exactly
    (dyadic, exact in floats), so neither boolean, and the gate holds at the bound. Mutation pass, one
    mutant per isolated copy, `test_paired_comparison`: 13 mutants, 10 killed (se from `pstdev` with sd
    sample; wins on the second side; gate sign from the mean of absolute values; `arm_comparison` over
    `lr`/`mlp` only, so `trivial` dropped; `worse` at `>=`; median for mean; gate on ROC-AUC; model
    comparison in the first arm only; gate `sd` set to se; `folds` off by one), 1 equivalent (minimum
    lowered to one fold: `statistics.stdev` then raises `StatisticsError`, a `ValueError` subclass, probed
    on Python 3.12), 2 survivors, items 1-2.
    1. `tests/test_paired_comparison.py:44-49`: the ROC-AUC differences `[0.125, -0.125, 0.0]` are
       symmetric, so inverting the ROC-AUC win direction (`d < 0`, `bqc/evaluate.py:768`) still counts 1
       and survives. Use asymmetric differences (for example `[0.125, 0.0625, -0.125, 0.0]`: 2 wins, 1 if
       inverted).
    2. `tests/test_paired_comparison.py:119` and `:155`: no test has the FPR gate failing while ROC-AUC is
       justified, so `"justified": better and (gate["holds"] or roc_auc["justified"])`
       (`bqc/evaluate.py:775`) survives. Add a case: recall beyond noise, FPR rise beyond noise, ROC-AUC
       justified; assert `justified` false and `fpr_gate.holds` false.
    Rulings: (F1) TDD order slipped again, recorded as process deviation F1; the tests are hand-computed
    and with items 1-2 the mutation evidence is sufficient, so no rework beyond them. Fewer than two
    folds raising `ValueError` is accepted: the run always has at least 2 x 2 folds, so it is a
    programming error, and section 11 fails loudly rather than reporting a misleading verdict.
    Top-level `justified`/`worse` (recall), nested `roc_auc.justified`/`.worse` and `fpr_gate`
    {`holds`, `mean`, `sd`, `sign`} match section 7 and 8.2 ("one `justified` and one `worse` boolean
    for the comparison and the FPR gate"; ROC-AUC's pair informational). Top-level `arm_comparison`
    keyed by model, `trivial` included, matches 8.2 ("each arm under its own key plus an
    `arm_comparison` section"); `model_comparison` under each arm matches 8.2 ("inside a run"). The
    `tests/test_run_report.py` key-set changes are accepted and added to Files with `bqc/cli.py`.
  - **Corrections (sonnet, 2026-11-06):** items 1-2 addressed in tests; see the `test(M4.T8)` commit.
  - **Validation of corrections (opus, 2026-11-06): PASS.** `e7bd2e8` touches tests and
    PLAN.md only. Item 1: ROC-AUC differences `[0.125, 0.0625, -0.125, 0.0]`, 2 wins (1 if
    inverted), mean 0.015625. Item 2: new
    `test_roc_auc_justified_does_not_override_a_failed_fpr_gate`; hand check: recall mean 0.5 >
    2 se 0.102; FPR mean 0.140625 > 2 se 0.03125 (ss 3/1024, sd 1/32, exact); ROC-AUC the same,
    so `roc_auc.justified` true, gate fails, `justified` false. Both survivors now killed in an
    isolated copy: `d < 0` for ROC-AUC fails `test_roc_auc_higher_is_a_win`; `gate["holds"] or
    roc_auc["justified"]` fails the new test. Full suite: 768 tests OK. Lines within 100 columns.

---

## M5: HDBSCAN experiment

**Sign-off: 2026-11-06, `fable`, verdict POSITIVE** (audited at `30221fe`; `DESIGN.md` `f8855ca`).

- Suite: `.venv/bin/python -m unittest discover -s tests`, 849 tests, OK in 138 s, run by `fable` on a
  clean tree (the in-place mutant of the M5.T3 re-validation was reverted; `bqc/cli.py:533` holds
  the sum).
- Diff read: the whole of `bqc/cluster.py`, `grid_report`, and the `cluster` parts of `bqc/cli.py`
  (`build_cluster_report`, `_cluster`), plus every M5 review record. Invariants: labels reach no
  clustering step (`run_grid` takes texts only, the truth is applied after it returns); the
  experiment shares only `normalise` and `make_vectoriser` with the pipeline and imports neither
  `bqc.evaluate` nor `bqc.models`; the pipeline is exactly 9.1 with `random_state=SEED` and
  `copy=True`; both 9.2 rules resolve ties and noise to legitimate; the oracle metrics are on
  non-noise rows and the seed rule over all rows; seeds come from `random.Random(SEED + i)`; the
  960 strings are the real queries and 20 per A-D template from their occurrences, screened by
  rule 4; one JSON through `write_report`; tests never download or read `data/raw/`.
- Probes recorded: 13.5 (`copy`), the module docstring of `bqc/cluster.py` (SVD and HDBSCAN size
  limits, the sklearn metrics on degenerate and empty input).

Follow-ups:

- **F14 (for the user, via M6.T1):** the first real `cluster` run (6 s, file under `results/`, from
  the tree one commit before `cc7dc12`): at the default cell (50, 15) 2 clusters, 1.5% noise, ARI
  0.002, seed-rule accuracy 0.493 / FPR 0.45 / recall 0.435 at k=20; the best ARI, 0.895 at
  (100, 30), comes with 84% noise, i.e. the fail-open label for most rows. Read: the two classes
  are not discoverable by clustering at this feature set. M6.T1 reruns it at the signed-off commit
  and reports it; nothing acts on it before the user reads it.
- **F15 (noted):** `cluster` loads and hash-checks `product.csv` through the shared loader although
  it never uses it; harmless (the file is already fetched for `evaluate`).

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
- **Status:** validated

  Implementation notes (`sonnet`): `make_pipeline` is a sklearn `Pipeline` (vectoriser, SVD,
  L2 `Normalizer`, `HDBSCAN(copy=True)`); `cluster_labels` normalises once per text; `run_grid`
  returns 9 dicts (n_components, min_cluster_size, labels, n_clusters, noise_fraction).
  Probes (2026-11-06) recorded in the `bqc/cluster.py` docstring: `copy` has no observable
  effect on dense finite input (set True); SVD raises ValueError when n_components exceeds the
  feature count, not when it exceeds the row count; HDBSCAN raises ValueError when rows are
  fewer than min_cluster_size, so the all-noise test uses rows == min_cluster_size. Mutation
  pass in an isolated copy: 12 mutants, all killed. Full suite: 787 tests OK.

  Validation (`opus`, 2026-11-06): CHANGES REQUESTED. Suite: 787 tests OK. Criteria 1 and 2
  are met and pinned (full `get_params()` of every step, defaults 50/15, grid order and
  values), unit norm, same-seed, all-noise (11.17) and the import check are tested, and there
  are no new dependencies or downloads. Mutation pass in an isolated copy: 21 mutants, 19
  killed (L1 normaliser, half-fit with label reuse, grid reordered or reversed, noise counted
  as a cluster, noise fraction over non-noise rows, SVD seed dropped or +1, L2 dropped,
  `min_samples=5`, `copy` unset or False, each default, `normalise` dropped or doubled, grid
  value dropped, `labels` key dropped, `metric="cosine"`). Two survived:
  1. `tests/test_cluster.py:102`: the no-labels test passes by accident. With
     `cluster_labels(texts, labels=None, n_components=..., min_cluster_size=...)`, the call
     `cluster_labels(TEXTS, 5, 5, [0] * 60)` passes a list as `min_cluster_size`, and
     sklearn's `InvalidParameterError` subclasses `TypeError`, so the test still passes.
     `run_grid(texts, labels=None)` isn't checked at all. Fix: pin the exact parameter names
     with `inspect.signature`: `cluster_labels` and `make_pipeline` take
     `["texts", "n_components", "min_cluster_size"]` (`make_pipeline` without `texts`), and
     `run_grid` takes `["texts"]`.
  2. `tests/test_cluster.py:60-80`: nothing checks that the steps are fitted on all rows
     (9.1 step 1). If the vectoriser and SVD are fitted on the first half and then transform
     every row before `HDBSCAN.fit_predict`, the tests still pass. Fix: add a test that wraps
     `bqc.cluster.make_pipeline` to capture the fitted pipeline from
     `cluster_labels(TEXTS, 5, 5)`. Assert that its `vectoriser.vocabulary_` and
     `svd.components_` equal those of a reference `make_pipeline(5, 5)` fitted on all
     normalised texts, and that `len(hdbscan.labels_) == len(TEXTS)`. `TEXTS` halves use
     disjoint alphabets, so a half-fit vocabulary differs.
  - **Corrections (sonnet, 2026-11-06):** items 1-2 addressed in tests; see the `test(M5.T1)`
    commit.

  Re-validation (`opus`, 2026-11-06): PASS. `09d186c` touches tests and `PLAN.md` only.
  Item 1: `test_signatures_take_texts_only` pins exact parameter names of `cluster_labels`,
  `make_pipeline` and `run_grid`. Item 2: `test_every_step_fitted_on_all_rows` wraps
  `make_pipeline` and compares vocabulary and SVD components with an all-row reference.
  Mutants re-run in an isolated copy, all killed: `labels=None` on `cluster_labels` and on
  `run_grid` (by the signature test); vectoriser and SVD fitted on the first half (by the
  all-rows test). Suite: 789 tests OK.

  Rulings: rows below `min_cluster_size` raising sklearn's `ValueError` is accepted, because
  11.17 is about the all-noise result, which is tested, and that case is documented and
  pinned. Refitting the vectoriser per grid cell is accepted: a shared fit would split the
  `Pipeline` and isn't simpler. Returning `labels` per cell is accepted because M5.T2 needs
  them, and nothing else speculative is built. For `fable`: 13.5 says only "set `copy`
  explicitly". The `bqc/cluster.py` docstring records the finding (no observable effect on
  dense input, so True was chosen). A one-line pointer in 13.5 would keep the HDBSCAN
  findings in one place (optional).

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
- **Status:** validated

  Implementation notes (`sonnet`): `oracle_vote(clusters, truth)`, `draw_seeds(n_rows, k,
  draw_index)` (= `random.Random(SEED + draw_index).sample(range(n_rows), k)`),
  `seed_draw(clusters, truth, k, draw_index)` and `seed_rule(clusters, truth)` (k in
  `SEED_KS`, `N_DRAWS` = 20; mean plus per-draw values) in `bqc/cluster.py`. Inputs: 1 =
  illegitimate. Decisions: `k` > rows raises `ValueError` (from `random.sample`); a draw with
  no seed in any cluster predicts all legitimate; FPR with no true negatives and recall with
  no true positives are `None` (a mean stays `None`); empty or unequal-length input raises
  `ValueError`; oracle purity is majority count over cluster size (0.5 on a tie). Probes
  (2026-11-06) in the `bqc/cluster.py` docstring: sklearn's four metrics never raise or warn on
  one cluster, one class or empty input, and return 1.0 on empty input, hence the explicit
  `None` for zero non-noise rows. Mutation pass in an isolated copy: 19 mutants, all killed.
  Full suite: 821 tests OK.

  Validation (`opus`, 2026-11-06): CHANGES REQUESTED. Suite: 821 tests OK (152 s). Criteria
  1-5 are met, and checked by hand: ARI 12/37 (contingency [[2,1],[0,3]]); one draw 12/14,
  1/9, 4/5; noise moved recall 4/5 to 4/9, FPR 1/9 to 1/5. `draw_seeds` matches 9.2. Both
  rules resolve ties to legitimate. No-seed clusters and noise fail open. All-noise output
  follows 11.17. No `bqc.evaluate` or `bqc.models` import, no new dependency, and the probe
  is recorded. Mutants run in an isolated copy, 14 in total. Killed: seed majority `>=`,
  oracle `>=`, oracle label from the minority, purity from the minority, `draws` reversed,
  AMI `average_method` geometric and max, `non_noise` with `>= -1`, `clusters`
  reverse-sorted, noise seeds voting, metric arguments swapped. Three survived:
  1. `tests/test_cluster_labels.py:239-251`: the mean over draws isn't pinned. The fake
     returns accuracy 0..19, which is symmetric, so `statistics.median` also gives 9.5 and
     survives (criterion 2, "20 draws averaged"). Fix: make the fake's values asymmetric,
     for example `float(draw_index ** 2)`, and assert the mean 123.5 (the median is 110.5).
  2. `bqc/cluster.py:119`: nothing pins ascending cluster-id order. Every test lists cluster
     ids in ascending order of appearance, so iterating `dict.fromkeys(...)` (first
     appearance) survives. HDBSCAN ids don't appear in order. Fix: add an `oracle_vote` test
     whose ids appear out of order, for example `[1, 1, 0, 0, 0]`, and assert that
     `[c["cluster"] for c in r["clusters"]] == [0, 1]` with the matching sizes.
  3. `non_noise` counted with `clusters >= 0` survives, but it is equivalent: HDBSCAN on
     dense, finite input gives only ids >= -1. No change is needed.

  Rulings: `k` larger than the row count raising from `random.sample`, and empty or
  mismatched input raising `ValueError`, are accepted as tested failure paths. FPR or recall
  `None` with no negatives or positives, and the mean staying `None`, are accepted: the value
  is undefined, and 11.17 only fixes FPR 0 and recall 0 for all-noise with both classes
  present, which is tested. All-noise accuracy equals the legitimate share, 0.5 on a
  balanced set, which is what predicting everything legitimate means. The per-draw list
  is accepted: 60 dicts of three floats is about 5 KB per call, which is fine in the run
  JSON. If M5.T3 calls `seed_rule` for each of the 9 grid cells, that is about 45 KB, still
  fine. The "Next task" line read M5.T3. It is now M5.T2 (corrections), then M5.T3.

  Corrections (`sonnet`): items 1 and 2 done in a `test(M5.T2)` commit after `0fd1926`.

  Re-validation (`opus`, after `201b723`): CHANGES REQUESTED. Both items are addressed and both
  mutants are now caught: `statistics.median` for `fmean` fails
  `SeedRuleTest.test_draw_indices_and_k_grid`; `dict.fromkeys` for `sorted(set(...))` fails
  `OracleVoteTest.test_clusters_listed_in_ascending_id_order`. Suite: 822 tests, OK.

  1. `tests/test_cluster_labels.py:256`: the comment says the median of `i**2` for 0..19 is
     110.5. It is 90.5 (mean of 81 and 100; `statistics.median` confirms). The comment is there
     to show why the fixture tells mean from median, so the wrong number misleads. Fix: change
     110.5 to 90.5. Comment only; no other change.

  Comment fix (`sonnet`): `9ac6e76`, one line in `tests/test_cluster_labels.py` (110.5 to 90.5).

  Final validation (`opus`, after `9ac6e76`): PASS, status `validated`. Commits: `248b068`
  (implementation), `201b723` (corrections), `9ac6e76` (comment fix). `git show 9ac6e76` is the
  one comment line and nothing else; 90.5 is correct (median of 81 and 100). Mutants from the
  earlier passes: `statistics.median` for `fmean` and `dict.fromkeys` for `sorted(set(...))` are
  caught; `clusters >= 0` for `non_noise` is equivalent. Module tests
  (`-k test_cluster_labels`): 33 OK. No further mutant run: the change is comment-only.

### M5.T3: `bqc cluster`

- **Goal:** run the experiment from the command line and write its JSON.
- **Design refs:** 9, 10.
- **Files:** `bqc/cluster.py`, `bqc/cli.py`, `tests/test_cli_cluster.py`.
- **Acceptance criteria:** `python -m bqc cluster` builds the nominal 960 strings: the 480 real queries
  and 20 positives per template of A-D from their occurrences, with no family E or P and no product
  span (section 9). Rule 4 applies (`DESIGN.md` `6cec6c6`): a positive equal to a sanity query fails; one
  equal to a real query or a P string is dropped and counted in the report. It writes a separate JSON
  file with
  `write_report` imported from `bqc/report.py`; `bqc/cluster.py` doesn't import `bqc/cli.py`
  (section 10). It shares no state with `evaluate`. A missing raw file exits non-zero with a
  message pointing at `fetch`.
- **Tests:** tiny config end to end; report round-trips through `json`; missing raw file;
  two runs identical apart from the timestamp (11.18).
- **Depends on:** M2.T8, M5.T2.
- **Status:** validated

  Implementation notes (`sonnet`): `cli._cluster` / `build_cluster_report` compose the strings
  (real queries, then 20 per template of A-D from `token_occurrences(real)`, rule 4 through
  `_screen` with `real_queries` and `part_numbers` as drop lists); `cluster.grid_report(texts,
  truth)` runs `run_grid(texts)` and only then `oracle_vote` / `seed_rule` per cell. Report body:
  `rows`, `positives_dropped`, `shortfall` (sum), `grid` (n_components, min_cluster_size,
  n_clusters, noise_fraction, `oracle`, `seed_rule` with its per-draw lists). The per-cell
  `labels` arrays are not written (9.2 asks for metrics, not assignments). Parameters hold
  families, per_template, the grid, seed ks and draws. `registry` and `per_template` are
  test-only arguments of `_cluster`; tests patch `GRID_*` to a tiny grid. Mutation pass in an
  isolated copy: 17 mutants, all killed (one needed a retarget of my own replace). Full suite:
  847 tests OK. Real run: see the report to the caller.

  Validation (`opus`, 2026-11-06, `cc7dc12`): CHANGES REQUESTED. Suite: 847 tests OK (147 s).
  Criteria met and pinned: real queries first, then the A-D positives rendered with `SEED` from
  `token_occurrences(real)` (exact texts compared); no E, P or span text; `per_template` and
  `registry` defaults are `QUOTA` and `REGISTRY`; rule 4 (sanity collision exits 1 with no text
  and no file; real-query and P matches dropped after `normalise()` and counted); one file via
  `write_report` from `bqc/report.py`; `bqc/cluster.py` doesn't import `bqc.cli` (AST test); no
  `evaluate` state is used; a missing raw file exits 1 naming `fetch` and the file; two runs give
  equal JSON (11.18); truth `[0]*real + [1]*positives` reaches `oracle_vote` and `seed_rule`
  only after the single `run_grid(texts)` call; `labels` aren't written; the parser tests patch
  `_cluster`; no downloads. Mutation pass in an isolated copy: 24 mutants, 20 killed (positives
  from the product names' vocabulary, both `per_template` defaults, truth offset by one row
  either way, truth reversed, `seed_rule` k=40 row or key dropped, `rows.total` +1 or nominal,
  `rows.positives` from the rendered list, a partial file then `OSError` in place of
  `write_report`, a non-serialisable body value, positives before real, `real_queries` drop list
  removed, sanity bypassed, E rendered, `SEED + 1`, `labels` kept, fetch message, oracle
  stubbed). Four survived:
  1. `bqc/cli.py:533`: `"shortfall": 0` survives; the reported shortfall is never checked. On
     the real data it is the only signal that a template fell short of the nominal 20. Fix: in
     `tests/test_cli_cluster.py`, add a test that wraps `self.registry.render` (as
     `replace_first` does) to drop the first 3 samples of one template and the first 2 of
     another and to return a shortfall dict with those two keys set to 3 and 2; assert
     `r["shortfall"] == 5` and `r["rows"]["positives"] == POSITIVES - 5`.
  2. `bqc/cli.py:527-528`: `"seed_ks": [10]` and `"seed_draws": 5` survive;
     `tests/test_cli_cluster.py:126-130` checks only three of the five parameters. Fix: assert
     the whole dict, with `"seed_ks": [10, 20, 40]` and `"seed_draws": 20` written as the 9.2
     values (not read back from `cluster`).
  3. `tests/test_cli_cluster.py:318-321`: the test passes without the check at
     `bqc/cluster.py:189`, because `run_grid` on two rows raises sklearn's `ValueError` anyway
     (rows below `min_cluster_size`, M5.T1). Fix: patch `cluster.run_grid` with a spy and assert
     it was not called, so the check must come before any clustering.

  Corrections (`sonnet`): items 1-3 done in a `test(M5.T3)` commit after `8e16655`.

  Re-validation (`opus`, 2026-11-06, `6617968`): PASS. Tests only; items 1-3 addressed. The
  extra zero-shortfall test is accepted (cheap, pins the no-drop case). Survivors now caught:
  `"shortfall": 0` by `ShortfallTest.test_the_reported_shortfall_is_the_sum_over_templates`;
  `"seed_ks": [10]` and `"seed_draws": 5` by `EndToEndTest.test_parameters_are_recorded`; the
  `grid_report` length check removed by `GridReportTest.test_the_inputs_must_be_equally_long`.
  Suite: 849 tests OK (139 s).

  Rulings: `grid_report` in `bqc/cluster.py` (texts and truth in, JSON-ready metrics out,
  clustering before scoring) with the string composition in `bqc/cli.py` is accepted: it
  matches `generate`, keeps the experiment free of the command layer (10), and `write_report`
  is still imported from `bqc/report.py`. `registry` and `per_template` as test-only
  arguments are accepted (M4.T4 precedent); the defaults are pinned. The per-draw lists are
  accepted (M5.T2); the real file is 66 KB. Reusing `_load_inputs` means `cluster` also needs
  and hash-checks `product.csv`, which it doesn't use; accepted for DRY, since `fetch` gets
  both files and the run takes 6 s. A failure inside `json.dump` leaves a partial file: that is
  `write_report` (M1), not this task, and the body is fully built before the write; noted, no
  change.

  Real run for `fable` (not acted on): `results/cluster-f74b9197dfe4-20261106T164127978984Z.json`,
  960 rows (480 + 480), nothing dropped, shortfall 0. Default cell (50, 15): 2 clusters, 1.5%
  noise, ARI 0.002, seed rule k=20 accuracy 0.493, FPR 0.45, recall 0.435. Best ARI 0.895 at
  (100, 30) with 84.4% noise (seed rule k=20: accuracy 0.533, FPR 0.009, recall 0.074). The file
  records `git_commit` `62ede06`, the parent of `cc7dc12`: the run came from the uncommitted
  tree; M6 reruns it.

---

## M6: Full run

**Sign-off: 2026-11-06, `fable`, verdict POSITIVE for the run** (audited at `9ba84da`; runs at
`a20073f`; `DESIGN.md` 13.9 updated with the wall times). The decisions it raises are the user's
(12.1 item 13, 12.2, 8.6 items 2 and 5) and are listed below, not taken.

- Checked by `fable` against `evaluate-250d3c345ade-20261106T184818872494Z.json` in the M6 clone:
  every headline recall, FPR and CI in the report, both `model_comparison` verdicts, all three
  `arm_comparison` verdicts, every `catalogue_gap` verdict, the primary-arm LR family-B and
  `B.template` recall, the sanity confusion matrix, the LOFO D row and the MLP fit time match.
- Suite at the audited commit: 849 tests, OK (M5 sign-off; M6.T1 changes `PLAN.md` only).
- Criterion 1: all four commands completed from a clean clone with a fresh venv; the MLP ran
  everywhere the design allows (fit 38.4 s, under the 5-minute rule). The run was restarted once
  after a container restart; the recorded run is the complete one.

Findings and the decisions they need:

1. **LR or MLP (12.1 item 13).** Primary arm: `model_comparison` is within noise on recall
   (-0.77 pp, se 0.93) and the MLP fails the FPR gate (+0.69 pp, se 0.26, point estimate 1.18%
   against the 1% target). Comparison arm: the MLP is `worse`. Recommendation under 6.2 and
   section 7: **logistic regression**. User decides.
2. **Arm (Q13 confirmed).** `arm_comparison`: LR `justified` (+6.63 pp recall, gate holds),
   trivial `justified`; the MLP's +9.74 pp is not `justified` because its FPR gate fails. The
   product-span arm stands as the headline.
3. **Generators are too easy on held-out B and D (8.6 items 1 and 2).** LOFO recall for LR drops
   to 34.1% on B (-30.6 pp) and 29.7% on D (-52.3 pp), and the trivial baseline beats LR and MLP
   on held-out B and D. On the headline the trivial baseline is 15 pp behind, so the in-family
   numbers are not inflated by artefacts, but the out-of-family numbers say the B and D
   templates share surface cues that the length-and-punctuation features capture. Rework of B
   and D is a design change: `fable` recommends it as the next piece of work (section 14 entry
   to be written), **after** the user confirms.
4. **The 13.22 recall drop is confirmed and deeper.** Primary-arm LR: overall 77.0% (probe 81.7
   to 79.7), family B 64.6% (74.4 to 67.3), `B.template` 47.7% (92 to 60); the comparison arm's
   LR gets `B.template` 100%, so the loss is specific to span negatives. Cause isolation stays a
   section 14 follow-up.
5. **Catalogue gap (8.6 item 5): passes in every arm and model.** Primary LR 1.25% (CI 0.57 to
   2.70), MLP 1.39%: point estimates above target, lower bounds below it, so
   `point_estimate_above_target` and no failure. No decision needed; reported.
6. **Q12, part numbers (12.2).** LR flags 85% of the 100 synthetic part-number strings in both
   arms, the MLP 76 to 81%, trivial 2 to 3%; on the sanity set LR flags all three part-number
   rows (16, 39, 43) and the two `SKU...` rows (39, 43) are flagged by every LR and MLP run
   though labelled legitimate. The user decides whether to add synthetic part-number negatives.
7. **Sanity set.** Primary LR over all 50 rows `[[33, 4], [3, 10]]`; ids 8, 47, 49 are missed by
   every LR and MLP run; the Jaccard-excluded ids 5, 7, 28 are all true positives caught by every
   model. Family E recall is 16 to 19% for LR and MLP: off-topic natural language is not caught,
   as the design expects (E is evaluation-only).
8. **Clustering (section 9, F14).** At the default cell the two classes are not discoverable
   (ARI 0.002, seed rule at chance); the high-ARI cells label 84% of rows noise. Finding, no
   decision: the no-label scenario is not served by this feature set.

Open after this sign-off: the user's answers to 1, 3 and 6; the push (blocked all session by
GitHub access, 67+ commits local); section 14 entries for 3 and 4.

### M6.T1: Run everything on real data and report

- **Goal:** produce the v1 numbers and hand them to `fable`, who records them in the M6
  sign-off (12.3).
- **Design refs:** `DESIGN.md` `0386ca8`: 6.2, 6.3, 7, 8.2, 8.5, 8.6, 9, 10, 12.1 (items 13, 14), 12.2
  (Q12), 13.2, 13.9, 13.15, 13.19, 13.22, 14.
- **Files:** none committed (`results/` is gitignored), apart from this task's status.
- **Acceptance criteria:**
  1. `fetch`, `generate`, one `evaluate --negatives product,real` invocation (with `--lofo`
     and `--catalogue-gap`) and `cluster` all complete on the real data from a clean clone with a fresh
     `.venv`. This is the first real `product.csv` download by `fetch`. If M4.T4's measured MLP fit
     exceeds 5 minutes, the invocation is `--lofo lr,trivial --catalogue-gap lr,trivial`, and the report
     says the MLP's rows weren't run and why (section 10). The primary arm is the headline. The primary arm's
     MLP runs 91 fits (15 outer, 60 LOFO, 15 catalogue gap, 1 final); the report states the count and
     the wall time.
  2. The report to `fable` gives, for each arm and model: headline recall with
     achieved FPR and CI; per-family recall including E; per-template recall; LOFO recall, achieved
     FPR and the paired difference to the main run; the trivial baseline gap; and cluster results for
     both rules, including noise fraction. It also gives `arm_comparison` (M4.T8).
  3. The false-positive list is included, and reviewed by eye for queries that look
     machine-generated (13.2, 13.9).
  4. The sanity section is reported per model and arm, twice: over all 50 rows, and over the rows at or
     below Jaccard 0.5 in the `generate` report, naming the excluded ids (8.6.1). Each gives the
     raw-count confusion matrix and full list of disagreements.
  5. **Part-number diagnostic:** the report gives, per model, the fraction flagged, overall and per
     shape, and the min, median and max score, next to the model's result on the three part-number rows of the
     sanity set (ids 16, 39, 43). It states these so the user can answer Q12. The answer isn't
     needed for `done`.
  6. **One-word false positives:** per model, the count flagged out of 23, as mean per repeat and as the union
     over repeats, and the words (8.5).
  7. **Catalogue-gap check:** for each arm, pooled FPR with CI, recall, the flagged queries, and whether
     it failed the rule of 8.6 item 5. A failure is reported for the user to decide; this task doesn't
     act on it.
  8. **LR or MLP:** `model_comparison` under the rule of section 7, and a recommendation for the user to
     decide on (12.1 item 13). With `justified` and `worse` both false the reading is "within fold-to-fold
     noise", and the 6.2 recommendation is the simpler model.
  9. If the trivial baseline gets close to the MLP, or LOFO shows generators are too easy,
     this is reported as a finding. The rework is `fable`'s decision (6.3, 8.6), and this
     task doesn't change generators.
  10. **The 13.22 recall drop:** the primary arm's logistic regression recall overall, on family B and on
     `B.template`, next to the probes' figures (81.7 to 79.7%, 74.4 to 67.3%, 92 to 60%), with a statement
     that the full run confirms or dismisses the drop. Isolating its cause is a follow-up in section 14, not
     this task.
- **Tests:** none (no code). The determinism test of M4.T4 covers reproducibility.
- **Depends on:** M2.T8, M2.T13, M4.T4, M4.T8, M5.T3.
- **Status:** done

#### Report (run on 2026-11-06 by `sonnet`, resumed after a container restart)

Clean clone `/tmp/claude-0/-home-user-bot-query-classifier/23a87975-f427-5451-b1ca-6a3bd01b7bc3/scratchpad/m6/clone`
at `a20073f` (the signed-off M5 commit; every run JSON records `git_commit`
`a20073f387b7364277e9df8183c3ba68fbbdac1b`, python 3.13.18, numpy 2.5.4, scikit-learn 1.9.1, seed
20261105), fresh `.venv`. Nothing under `results/` or `data/raw/` is committed. Arm names in the JSON:
`product` = primary arm (product-name spans as negatives, headline), `real` = comparison arm
(training-side real queries). Differences are first minus second (`arm_comparison` = product minus
real; `model_comparison` = MLP minus LR).

**Commands and wall times** (all exit 0; logs in `.../scratchpad/m6/`: `fetch.log`, `generate.log`,
`evaluate.log`, `cluster.log`):

| step | wall time | output |
|---|---|---|
| `fetch` | 6 s (checksums matched; first real `product.csv` download) | `data/raw/{query,product}.csv` |
| `generate` | 56 s | `results/generate-40ffc15c606d-20261106T170957825460Z.json` |
| `evaluate --negatives product,real --lofo --catalogue-gap` (all 3 models) | **64 min 58 s** (3,889.6 s; `user` 210 min on multiple cores) | `results/evaluate-250d3c345ade-20261106T184818872494Z.json` (+ 6 `final-*.pkl`) |
| `cluster` | 7 s (6.1 s) | `results/cluster-b6afd0c5a7b2-20261106T184834583413Z.json` |

Criterion 1: M4.T4's MLP fit (39.7 s) is under 5 minutes, so the MLP ran in every place the design
allows: main run, LOFO and catalogue gap in the primary arm (MLP LOFO is never run in the comparison
arm, 8.6 item 1, recorded in the JSON as `not run`), and the comparison arm's main run and catalogue
gap. The primary arm's MLP is 91 fits by construction (15 outer + 60 LOFO + 15 catalogue gap + 1 final);
I did not instrument the count, only the final fit was timed (38.4 s, `timing.mlp_final_fit_seconds`).
13.9 should get the final wall time, about 65 minutes for the full invocation.

Recall has no CI (8.5). FPR is pooled FP / (3 x 480) with the Wilson interval at n = 480, k = total/3.
Per-family and per-template recall are pooled caught/total. Tables below are generated from the JSON.


##### Headline, arm `product`

| model | recall | caught/n | FPR (pooled) | FP/n | Wilson 95% CI (n=480) | ROC-AUC pooled |
|---|---|---|---|---|---|---|
| lr | 77.0% | 5542/7199 | 0.49% | 7/480 | 0.15% to 1.61% | 0.971 |
| mlp | 76.3% | 5493/7199 | 1.18% | 17/480 | 0.53% to 2.61% | 0.975 |
| trivial | 61.3% | 4410/7199 | 0.90% | 13/480 | 0.36% to 2.22% | 0.852 |

Per-family recall, arm `product` (E evaluation-only):

| model | A | B | C | D | E |
|---|---|---|---|---|---|
| lr | 70.9% | 64.6% | 90.4% | 82.0% | 18.4% |
| mlp | 74.5% | 62.9% | 84.8% | 83.0% | 16.6% |
| trivial | 38.7% | 59.0% | 76.2% | 71.2% | 5.8% |

Per-template recall, arm `product`:

| template | lr | mlp | trivial |
|---|---|---|---|
| A.alnum | 99.3% | 98.7% | 93.3% |
| A.hard_mash | 16.7% | 26.3% | 2.3% |
| A.hard_repeat | 25.3% | 37.7% | 20.3% |
| A.hex | 100.0% | 96.3% | 100.0% |
| A.mash | 95.7% | 97.3% | 16.3% |
| A.repeat | 88.3% | 90.7% | 0.0% |
| B.hard_nosql | 35.3% | 39.7% | 84.0% |
| B.hard_sql | 59.3% | 59.7% | 20.3% |
| B.script | 75.0% | 69.0% | 100.0% |
| B.sql | 71.7% | 71.3% | 26.3% |
| B.template | 47.7% | 53.7% | 23.3% |
| B.traversal | 98.7% | 84.3% | 100.0% |
| C.hard_fragment | 89.7% | 84.3% | 95.3% |
| C.hard_search_url | 88.0% | 64.3% | 100.0% |
| C.paging | 100.0% | 100.0% | 85.3% |
| C.path | 84.7% | 78.0% | 96.3% |
| C.random_keys | 100.0% | 100.0% | 79.7% |
| C.walk | 80.3% | 81.9% | 0.0% |
| D.email | 99.7% | 100.0% | 100.0% |
| D.hard_contact | 90.0% | 86.0% | 65.0% |
| D.hard_url | 99.7% | 99.7% | 99.7% |
| D.phone | 100.0% | 100.0% | 62.3% |
| D.promo | 2.7% | 12.3% | 0.0% |
| D.url | 100.0% | 100.0% | 100.0% |

##### Headline, arm `real`

| model | recall | caught/n | FPR (pooled) | FP/n | Wilson 95% CI (n=480) | ROC-AUC pooled |
|---|---|---|---|---|---|---|
| lr | 70.4% | 5067/7199 | 1.04% | 15/480 | 0.45% to 2.42% | 0.955 |
| mlp | 66.5% | 4787/7199 | 0.42% | 6/480 | 0.11% to 1.51% | 0.922 |
| trivial | 58.6% | 4221/7199 | 0.56% | 8/480 | 0.18% to 1.72% | 0.849 |

Per-family recall, arm `real` (E evaluation-only):

| model | A | B | C | D | E |
|---|---|---|---|---|---|
| lr | 60.2% | 67.8% | 78.2% | 75.4% | 19.1% |
| mlp | 55.0% | 59.6% | 71.7% | 79.7% | 14.6% |
| trivial | 35.9% | 56.5% | 71.8% | 70.4% | 5.8% |

Per-template recall, arm `real`:

| template | lr | mlp | trivial |
|---|---|---|---|
| A.alnum | 100.0% | 93.7% | 84.3% |
| A.hard_mash | 7.0% | 7.0% | 1.0% |
| A.hard_repeat | 11.0% | 7.0% | 19.3% |
| A.hex | 100.0% | 100.0% | 100.0% |
| A.mash | 74.3% | 75.0% | 10.7% |
| A.repeat | 69.0% | 47.3% | 0.0% |
| B.hard_nosql | 23.3% | 19.0% | 81.7% |
| B.hard_sql | 45.0% | 43.7% | 17.3% |
| B.script | 66.0% | 47.0% | 100.0% |
| B.sql | 72.3% | 65.7% | 16.3% |
| B.template | 100.0% | 84.7% | 23.7% |
| B.traversal | 100.0% | 97.3% | 100.0% |
| C.hard_fragment | 79.7% | 68.0% | 94.0% |
| C.hard_search_url | 45.3% | 29.0% | 98.0% |
| C.paging | 100.0% | 100.0% | 81.0% |
| C.path | 78.7% | 77.7% | 86.7% |
| C.random_keys | 100.0% | 99.0% | 70.7% |
| C.walk | 65.2% | 56.5% | 0.0% |
| D.email | 91.7% | 100.0% | 100.0% |
| D.hard_contact | 67.0% | 76.3% | 63.3% |
| D.hard_url | 90.0% | 100.0% | 99.3% |
| D.phone | 100.0% | 100.0% | 59.7% |
| D.promo | 3.7% | 2.0% | 0.0% |
| D.url | 100.0% | 100.0% | 100.0% |

##### LOFO, arm `product` (recall on held-out family F; FPR; paired difference to main run's recall on F)

| model | F | LOFO recall | FPR | main recall on F | diff |
|---|---|---|---|---|---|
| lr | A | 62.7% | 0.76% | 70.9% | -8.2 pp |
| lr | B | 34.1% | 0.49% | 64.6% | -30.6 pp |
| lr | C | 71.3% | 0.49% | 90.4% | -19.1 pp |
| lr | D | 29.7% | 0.83% | 82.0% | -52.3 pp |
| mlp | A | 65.4% | 0.83% | 74.5% | -9.1 pp |
| mlp | B | 35.6% | 0.62% | 62.9% | -27.4 pp |
| mlp | C | 57.5% | 0.90% | 84.8% | -27.3 pp |
| mlp | D | 22.4% | 0.97% | 83.0% | -60.6 pp |
| trivial | A | 35.2% | 0.83% | 38.7% | -3.6 pp |
| trivial | B | 49.2% | 0.69% | 59.0% | -9.8 pp |
| trivial | C | 70.5% | 0.69% | 76.2% | -5.6 pp |
| trivial | D | 68.8% | 0.83% | 71.2% | -2.4 pp |

##### LOFO, arm `real` (recall on held-out family F; FPR; paired difference to main run's recall on F)

| model | F | LOFO recall | FPR | main recall on F | diff |
|---|---|---|---|---|---|
| lr | A | 50.2% | 0.90% | 60.2% | -10.1 pp |
| lr | B | 44.3% | 0.76% | 67.8% | -23.4 pp |
| lr | C | 65.8% | 0.56% | 78.2% | -12.4 pp |
| lr | D | 30.0% | 0.90% | 75.4% | -45.4 pp |
| mlp | all | not run: the MLP is never run in the comparison arm's leave-one-family-out (8.6 item 1) | | | |
| trivial | A | 32.1% | 0.49% | 35.9% | -3.8 pp |
| trivial | B | 48.0% | 0.56% | 56.5% | -8.5 pp |
| trivial | C | 64.2% | 0.69% | 71.8% | -7.6 pp |
| trivial | D | 67.5% | 0.56% | 70.4% | -2.9 pp |

##### Comparisons

| comparison | recall | FPR | fpr_gate | ROC-AUC | justified | worse |
|---|---|---|---|---|---|---|
| arm_comparison lr (product - real) | recall diff +6.63 pp (se 2.01, wins 12/15) | FPR diff -0.56 pp (se 0.40) | gate holds=True | AUC diff +0.0143 (se 0.0048; just=True, worse=False) | justified=True | worse=False |
| arm_comparison mlp (product - real) | recall diff +9.74 pp (se 2.26, wins 12/15) | FPR diff +0.76 pp (se 0.24) | gate holds=False | AUC diff +0.0122 (se 0.0049; just=True, worse=False) | justified=False | worse=False |
| arm_comparison trivial (product - real) | recall diff +2.75 pp (se 0.81, wins 13/15) | FPR diff +0.35 pp (se 0.26) | gate holds=True | AUC diff +0.0026 (se 0.0020; just=False, worse=False) | justified=True | worse=False |
| model_comparison product | recall diff -0.77 pp (se 0.93, wins 8/15) | FPR diff +0.69 pp (se 0.26) | gate holds=False | AUC diff +0.0024 (se 0.0024; just=False, worse=False) | justified=False | worse=False |
| model_comparison real | recall diff -3.88 pp (se 1.78, wins 4/15) | FPR diff -0.62 pp (se 0.50) | gate holds=True | AUC diff +0.0045 (se 0.0027; just=False, worse=False) | justified=False | worse=True |


##### Cluster (`cluster`, 960 rows: 480 real + 480 positives; oracle ARI/AMI/homogeneity/completeness; seed rule at k = 10 / 20 / 40 as accuracy/FPR/recall)

Both rules are in the table: the oracle (ARI against the true class) and the seed rule.

| n_comp | min_size | clusters | noise | ARI | AMI | homog | compl | seed rule |
|---|---|---|---|---|---|---|---|---|
| 20 | 10 | 2 | 0.000 | 0.002 | 0.036 | 0.021 | 0.145 | 0.499/0.45/0.45 ; 0.500/0.45/0.45 ; 0.507/0.40/0.41 |
| 20 | 15 | 2 | 0.008 | 0.002 | 0.036 | 0.022 | 0.147 | 0.495/0.45/0.44 ; 0.496/0.45/0.44 ; 0.505/0.35/0.36 |
| 20 | 30 | 4 | 0.796 | 0.376 | 0.469 | 0.729 | 0.351 | 0.550/0.00/0.10 ; 0.572/0.01/0.16 ; 0.602/0.02/0.22 |
| 50 | 10 | 2 | 0.000 | 0.002 | 0.036 | 0.021 | 0.145 | 0.499/0.45/0.45 ; 0.500/0.45/0.45 ; 0.507/0.40/0.41 |
| 50 | 15 | 2 | 0.015 | 0.002 | 0.030 | 0.017 | 0.141 | 0.492/0.45/0.43 ; 0.493/0.45/0.44 ; 0.499/0.35/0.35 |
| 50 | 30 | 2 | 0.848 | 0.867 | 0.790 | 0.795 | 0.787 | 0.542/0.00/0.09 ; 0.568/0.00/0.14 ; 0.576/0.00/0.15 |
| 100 | 10 | 3 | 0.033 | 0.005 | 0.045 | 0.029 | 0.139 | 0.487/0.40/0.37 ; 0.487/0.40/0.37 ; 0.500/0.25/0.25 |
| 100 | 15 | 3 | 0.848 | 0.433 | 0.558 | 0.793 | 0.435 | 0.535/0.00/0.07 ; 0.562/0.00/0.12 ; 0.589/0.00/0.18 |
| 100 | 30 | 2 | 0.844 | 0.895 | 0.847 | 0.843 | 0.851 | 0.519/0.01/0.05 ; 0.533/0.01/0.07 ; 0.560/0.00/0.12 |

Cluster reading (F14): at the default cell (50, 15) there are 2 clusters, 1.5% noise, ARI 0.002 and the
seed rule is at chance (accuracy 0.49, FPR 0.45). Only cells with 84-85% noise reach a high ARI (0.867 at
(50, 30), 0.895 at (100, 30)); there the seed rule's FPR is ~0 but recall 0.05-0.15, i.e. the fail-open
label for most rows. This reproduces F14: the two classes are not discoverable by clustering at this
feature set.

##### Sanity set (criterion 4)

Labels: 37 legitimate (0), 13 illegitimate (1). Jaccard > 0.5 in the `generate` report: ids 5, 7 and 28
(max Jaccard 1.0, 1.0, 0.714); all three are labelled 1, so the second view drops 3 illegitimate rows
(47 rows: 37 / 10). No model disagrees on ids 5, 7 or 28 (all three are caught by every model in both
arms), so the second view differs from the first only by 3 fewer true positives. Confusion matrices are raw
counts, rows = user label (0, 1), columns = model (0, 1): `[[TN, FP], [FN, TP]]`.

| arm | model | threshold | all 50 | at or below 0.5 (47 rows, excl. 5, 7, 28) |
|---|---|---|---|---|
| product | lr | 0.529 | [[33, 4], [3, 10]] | [[33, 4], [3, 7]] |
| product | mlp | 0.681 | [[35, 2], [3, 10]] | [[35, 2], [3, 7]] |
| product | trivial | 0.664 | [[36, 1], [6, 7]] | [[36, 1], [6, 4]] |
| real | lr | 0.559 | [[34, 3], [4, 9]] | [[34, 3], [4, 6]] |
| real | mlp | 0.552 | [[34, 3], [3, 10]] | [[34, 3], [3, 7]] |
| real | trivial | 0.679 | [[36, 1], [7, 6]] | [[36, 1], [7, 3]] |

Disagreements (id, query, user label, score; identical in both views because no excluded id is among
them):

- product lr: 8 `1; DROP TABLE users;--` (1, 0.30); 10 `union jack rug` (0, 0.54); 16 `wf-2049x desk` (0, 0.71);
  39 `SKU000124` (0, 0.91); 43 `SKU000123` (0, 0.91); 47 `sofa sofa sofa sofa sofa sofa sofa` (1, 0.20);
  49 `best casino bonus free spins sofa` (1, 0.28).
- product mlp: 8 (1, 0.20); 39 (0, 0.97); 43 (0, 0.98); 47 (1, 0.02); 49 (1, 0.01).
- product trivial: 1 `xk9qz7vplm` (1, 0.40); 8 (1, 0.40); 20 `asdfghjkl qwerty zxcv` (1, 0.34);
  22 `loveseat/recliner combo` (0, 0.76); 25 `qqqqwwwweeee 8374651` (1, 0.58); 47 (1, 0.03); 49 (1, 0.08).
- real lr: 8 (1, 0.36); 16 (0, 0.76); 39 (0, 0.88); 43 (0, 0.88); 44 `buy cheap followers click here www.promo-deals.biz` (1, 0.52);
  47 (1, 0.43); 49 (1, 0.34).
- real mlp: 8 (1, 0.30); 16 (0, 0.61); 39 (0, 0.74); 43 (0, 0.74); 47 (1, 0.32); 49 (1, 0.31).
- real trivial: 1 (1, 0.41); 8 (1, 0.42); 20 (1, 0.35); 22 (0, 0.75); 25 (1, 0.58); 44 (1, 0.67); 47 (1, 0.03); 49 (1, 0.09).

Every LR and MLP run misses ids 8, 47 and 49 (SQL payload, repeated word, casino spam on a furniture
word), and every LR and MLP run flags the two `SKU...` rows.

##### Part-number diagnostic (criterion 5; Q12 input)

Fraction of the 100 generated part-number strings flagged at each final model's threshold; shapes are
prefix+digits / bare digits / prefix-digits-suffix / product words + part number (25 each).

| arm | model | flagged | per shape (4 shapes in that order) | score min / median / max | sanity 16 `wf-2049x desk` | sanity 39 `SKU000124` | sanity 43 `SKU000123` |
|---|---|---|---|---|---|---|---|
| product | lr | 85% | 100 / 100 / 100 / 40% | 0.135 / 0.891 / 0.975 | flagged (0.71) | flagged (0.91) | flagged (0.91) |
| product | mlp | 76% | 92 / 84 / 100 / 28% | 0.009 / 0.949 / 0.999 | not flagged | flagged (0.97) | flagged (0.98) |
| product | trivial | 3% | 0 / 0 / 8 / 4% | 0.127 / 0.312 / 0.729 | not flagged | not flagged | not flagged |
| real | lr | 85% | 100 / 100 / 100 / 40% | 0.351 / 0.880 / 0.940 | flagged (0.76) | flagged (0.88) | flagged (0.88) |
| real | mlp | 81% | 100 / 100 / 100 / 24% | 0.215 / 0.744 / 0.805 | flagged (0.61) | flagged (0.74) | flagged (0.74) |
| real | trivial | 2% | 0 / 0 / 8 / 0% | 0.139 / 0.325 / 0.725 | not flagged | not flagged | not flagged |

(The three sanity rows are labelled 0 by the user; "flagged" is therefore a false positive. Sanity
results come from the disagreement lists above.) The trained models flag most bare part numbers in both
arms, as 8.6 item 4 feared; the user answers Q12.

##### One-word false positives (criterion 6; 23 one-word WANDS negatives, 3 repeats)

| arm | model | flagged (union over repeats) / 23 | mean per repeat | words |
|---|---|---|---|---|
| product | lr | 3 | 2.33 | dumbbells, owl, zen |
| product | mlp | 6 | 3.00 | argenteuil, gold, kisner, meditation, owl, zen |
| product | trivial | 1 | 1.00 | e12/candelabra |
| real | lr | 6 | 3.00 | dumbbells, gold, peacock, picasso, pineapple, zen |
| real | mlp | 4 | 1.67 | e12/candelabra, owl, printers, zen |
| real | trivial | 0 | 0.00 | none |

##### False positives (criterion 3; pooled over 3 repeats, `xN` = number of repeats in which it was flagged)

| arm | model | FP (count, distinct) | queries |
|---|---|---|---|
| product | lr | 7, 3 | zen x3, dumbbells x3, owl x1 |
| product | mlp | 17, 11 | zen x3, drudge report x3, kisner x2, mom urn x2, outdoor privacy wall, argenteuil, rug plum, promo codes or discounts, gold, owl, meditation |
| product | trivial | 13, 6 | artificial orchid arrangement x3, certified international melamine x3, e12/candelabra x3, bathroom freestanding cabinet x2, brockham solid blackout thermal grommet curtain panels, white splashproof shiplap wallpaper |
| real | lr | 15, 9 | zen x3, zachary 72.5 x3, picasso x2, ines shag rug w003096809 x2, pineapple, peacock, gold, dumbbells, carpet 5x6 |
| real | mlp | 6, 5 | zen x2, zachary 72.5, printers, e12/candelabra, owl |
| real | trivial | 8, 4 | certified international melamine x3, white splashproof shiplap wallpaper x2, artificial orchid arrangement x2, bathroom freestanding cabinet |

Eye review: none of these looks machine-generated; all are real WANDS queries (brand or collection names,
short nouns, dimensions). Borderline only: `promo codes or discounts` (MLP, product arm) and `drudge
report` (off-domain but a human title) read like spam-family text and sit near D.promo; `ines shag rug
w003096809` is a product name plus a part number (the 13.15 token); `e12/candelabra` and `zachary 72.5`
look like catalogue fragments. The trivial baseline's FPs are longer ordinary phrases (a length artefact).

##### Catalogue gap (criterion 7; rule of 8.6 item 5: fails only if the Wilson lower bound of the pooled FPR exceeds 1%)

2,852 unclassified products; products removed per fold are in the JSON (`product.catalogue_gap.folds`, e.g. 21,995 in repeat 0 fold 0).

| arm | model | pooled FPR [Wilson 95%] | FP | recall | verdict |
|---|---|---|---|---|---|
| product | lr | 1.25% [0.57, 2.70] | 18 | 78.3% | pass; point estimate above target |
| product | mlp | 1.39% [0.66, 2.89] | 20 | 76.0% | pass; point estimate above target |
| product | trivial | 0.42% [0.11, 1.51] | 6 | 58.5% | pass |
| real | lr | 0.69% [0.25, 1.92] | 10 | 67.5% | pass |
| real | mlp | 1.39% [0.66, 2.89] | 20 | 61.1% | pass; point estimate above target |
| real | trivial | 0.35% [0.09, 1.40] | 5 | 56.4% | pass |

No arm or model fails the rule (lower bounds 0.09% to 0.66%). Flagged queries (all in the JSON,
`<arm>.catalogue_gap.models.<model>.flagged_queries`, with repeat and fold). Product arm LR: zen x3,
dumbbells x2, printers x2, mom urn x2, promo codes or discounts, drudge report, 12v dirt bike, picasso,
kisner, bee, gold, carpet 5x6, ines shag rug w003096809. Product arm MLP: zen x2, promo codes or discounts x2,
mom urn x2, peacock x2, midcentury tv unit x2, turquoise pillows x2, gold, kisner, printers, drudge report,
owl, rug plum, picasso, 12v dirt bike. Product trivial: certified international melamine x3, bathroom
freestanding cabinet, e12/candelabra, artificial orchid arrangement. Real arm LR: zen x2, zachary 72.5 x2,
3/4 size mattress, owl, picasso, dumbbells, gold, ines shag rug w003096809. Real arm MLP: zen, dumbbells x2,
pineapple x2, midcentury tv unit, promo codes or discounts, printers, kisner, 7qt slow cooker, camper,
surge protector, argenteuil, auburn throw pillows, coffee container, blk 18x18 seat cushions, sunflower,
zodiac pillow, 12v dirt bike, owl. Real trivial: certified international melamine x3, brockham solid blackout
thermal grommet curtain panels, artificial orchid arrangement. The primary arm's gap result (LR 1.25% /
78.3%) is close to the probe's 13.19 figure (1.67% / 86.3%) in FPR; recall is lower.

##### Findings (criteria 8, 9, 10)

**Criterion 8, LR or MLP.** `product.model_comparison` (MLP minus LR): recall -0.77 pp (se 0.93, MLP wins
8 of 15 folds), `justified` false, `worse` false, so the reading is "within fold-to-fold noise". The FPR
gate fails for the MLP (mean FPR difference +0.69 pp, se 0.26, the MLP has the lower FPR in only 1 of 15 folds), its
achieved FPR is 1.18% (CI 0.53-2.61, point estimate above the 1% target) against LR's 0.49%, and ROC-AUC
is within noise (+0.002). In the comparison arm the MLP is `worse` (recall -3.88 pp, se 1.78;
`justified` false). The MLP is also more expensive (final fit 38.4 s; 91 fits in the primary arm). Under
6.2 and section 7 the recommendation is **logistic regression**: nothing the MLP adds is outside noise, and
it overshoots the FPR target. The user decides (12.1 item 13). Note `arm_comparison`: LR `justified` true
(recall +6.63 pp, se 2.01, gate holds), MLP `justified` false although recall rises +9.74 pp (se 2.26)
because its FPR gate fails (+0.76 pp, se 0.24), trivial `justified` true (+2.75 pp, se 0.81, gate holds). So the
product-name negatives raise recall in every model at the same FPR, and the arm choice is supported for LR
and the trivial baseline.

**Criterion 9, trivial baseline and LOFO.** Headline recall (primary arm): trivial 61.3% vs LR 77.0% and
MLP 76.3%, a gap of about 15 pp; so on the headline the trivial baseline is not close. But it is
the best or tied model, at 100% recall, on A.hex, B.script, B.traversal, C.hard_search_url, D.url, and far
above LR on B.hard_nosql (84.0% vs 35.3%), so those templates are separable by length, word count and
punctuation alone. LOFO (primary arm; recall on the held-out family vs main run): LR A 62.7 (-8.2 pp),
B 34.1 (-30.6), C 71.3 (-19.1), D 29.7 (-52.3); MLP A 65.4 (-9.1), B 35.6 (-27.4), C 57.5 (-27.3),
D 22.4 (-60.6); trivial A 35.2 (-3.6), B 49.2 (-9.8), C 70.5 (-5.6), D 68.8 (-2.4). On a held-out family
the trivial baseline beats LR and MLP on B and D, ties LR on C, and loses on A. Family E (never trained on)
is 18.4% (LR) / 16.6% (MLP) / 5.8% (trivial). Weak spots in the main run regardless of model:
A.hard_mash 16.7%, A.hard_repeat 25.3%, D.promo 2.7% (LR). Reported as findings for `fable`: generators are
partly easy (several templates at 100% for trivial features), and unseen-family generalisation is poor;
nothing was changed. The MLP's LOFO is no better than LR's.

**Criterion 10, 13.22 recall drop (primary-arm LR, this full run, vs probes):**

| quantity | probes (13.22) | this run |
|---|---|---|
| recall overall | 81.7% to 79.7% | 77.0% (5542/7199) |
| family B | 74.4% to 67.3% | 64.6% (1163/1800) |
| B.template | 92% to 60% | 47.7% |

The full run **confirms the drop** and goes beyond the post-drop probe figures on all three. In the
comparison arm the same LR gets B.template 100% (MLP 84.7%), so the loss is specific to the primary arm's
negatives. Isolating the cause is a section 14 follow-up.

**Other observations (not acted on).** (a) Q12 and the part-number numbers above. (b) The `real` arm's MLP
FPR overall is 0.42% and the primary-arm MLP's is 1.18%; thresholds in the primary arm are cross-fit on 384
queries, so the FPR varies by fold (primary LR FPR 0.49% sits under the target, the MLP's above). (c) The 3
sanity rows above Jaccard 0.5 are caught by every model. (d) The `cluster` result in F14 is reproduced.

**Deviations.** None from the procedure. The previous session's `fetch` and `generate` outputs were reused
(same clean clone, same commit, state verified: clean tree, HEAD `a20073f`, `bqc` importable). `evaluate` was
rerun from scratch. MLP fit count not instrumented. Nothing under `results/` or `data/raw/` is committed.

---

## M7: Part-number hard negatives and the combined-feature adapter (design v2.8)

Plans `DESIGN.md` v2.8 (`cafa701`): 3.3 (part-number hard negatives, Q12 resolved, 12.1 item 20),
6.5 (`lr_hand`, 12.1 item 19), section 7 (`model_comparison_hand`), 8.6 item 4 (the diagnostic now
measures the fix). Logistic regression is the recommended model (12.1 item 18); the MLP stays as the
comparison model. No new file: every task changes files created earlier (see "File ownership").

Order: the probe (T1) before the adapter (T5); the builder (T2) before its wiring (T3, T4); the
adapter (T5) before its report section (T6); the run (T7) last. PQ-7 to PQ-9 are ruled in 3.3
("Rulings on the M7 plan", `cbc2169`) and written into T2 to T4. `generate` is unchanged (PQ-9).

Conventions carried over: adapters follow M4.T1-T3 (module-level class on `_VectorisedScorer`,
`normalise` once per text, full `get_params()` pinned, pickle round trip, `NotFittedError` on both
score paths); every removal goes through `drop_exact` and reports counts, never text (3.2 step 3,
11.25); every task closes with an `opus` mutation pass (F1). F8 stands: the extra builds per pass
cost little and aren't optimised.

### M7.T1: Probe the combined feature matrix for `lr_hand`

- **Goal:** answer 6.5's probe paragraph before the adapter is written.
- **Design refs:** `DESIGN.md` `cafa701`: 5, 6.2, 6.3, 6.5 (probe paragraph), 10, 13 (format), 13.22.
- **Files:** `DESIGN.md` (a new subsection 13.23 only, written by `sonnet`); scripts in `scratch/`
  (gitignored, not committed).
- **Acceptance criteria:**
  1. `scipy.sparse.hstack` of the fitted TF-IDF matrix with the three standardised hand columns gives a
     sparse matrix (format and dtype recorded), and `LogisticRegression(C=1.0, max_iter=1000)` fits and
     predicts on it without densifying (checked by type at each step, and by the peak memory or matrix
     size against the dense equivalent). If `sklearn.pipeline.FeatureUnion` (TF-IDF next to
     `FunctionTransformer(_hand_features)` + `StandardScaler`) is the composition the adapter would use,
     the same is checked for it, and the finding says which composition keeps the output sparse.
  2. `StandardScaler` inside the composition is fitted on the training rows only: after `fit`, `mean_`
     and `scale_` equal the training rows' statistics, and scoring a different batch leaves them
     unchanged; a text's score doesn't depend on the batch it is scored in.
  3. Fit time of the combined adapter at 5,000 spans / about 5,000 positives (the default training side,
     real data, seed `SEED`), next to plain `lr` on the same rows, and whether either emits a
     `ConvergenceWarning` at `max_iter=1000`.
  4. 13.23 records each finding with the date and the Python, scikit-learn, scipy and numpy versions,
     in the format of 13.3 to 13.22. No code under `bqc/` or `tests/` changes.
- **Tests:** none (probe). `opus` re-runs the scratch script, or reads its pasted output, when validating.
- **Depends on:** none (M6 signed off).
- **Status:** validated (opus, 2026-11-07). Re-ran `scratch/m7t1_probe.py` with `.venv` (Python 3.13.18,
  sklearn 1.9.1, scipy 1.18.1, numpy 2.5.4, as in 13.23): every structural figure reproduced exactly
  (shapes, nnz 740,931, csr from `hstack(format="csr")` and from the `FeatureUnion`, coo for
  `hstack([T, ndarray])`, union equal to the manual stack, `mean_`/`scale_` equal to the training rows'
  mean and population sd and bit-identical after two other batches, batch-independence diff 0.0,
  `n_iter_` 27 and 42, unscaled 132, no warnings). Wall times differed (rerun medians: `lr` 1.99 s,
  `lr_hand` 1.31 s, so `lr_hand` was not slower this time); 13.23 already says the spread is as large
  as the difference, and `n_iter_` is the reliable comparison. Minor, not blocking: the script's
  `timed()` keeps the warnings of the last of its three runs only; the deterministic `n_iter_` far below
  `max_iter` rules a `ConvergenceWarning` out for all three. `git show --stat edf9445`: `DESIGN.md` and
  `PLAN.md` only. Suite: 849 tests OK (`.venv/bin/python -m unittest discover -s tests`, 144 s).

### M7.T2: Part-number hard negatives builder

- **Goal:** one function in `bqc/negatives.py` that builds the 100 part-number hard negatives of one
  training side.
- **Design refs:** `DESIGN.md` `cbc2169`: 2.2, 3.2 (step 3, `drop_exact`), 3.3 (steps 1, 2 and 4, seed
  offset; rulings PQ-7b, PQ-8c), 4 (rule 3), 8.3 (occurrences), 8.6 item 4 (the four shapes), 10
  (seeds), 11.4, 11.25, 11.29, 13.15, 13.17.
- **Files:** `bqc/negatives.py`, `bqc/synth/partnumbers.py` (expose the shape table, no behaviour
  change), `tests/test_negatives.py`, `tests/test_synth_partnumbers.py`.
- **Acceptance criteria:**
  1. **One home for the shapes:** the builder renders with the four shape functions of
     `bqc/synth/partnumbers.py`, made public there (for example `_SHAPES` -> `SHAPES`); no shape logic
     is copied. `generate_part_numbers` returns exactly the same 100 strings as before for the same
     arguments (pinned by a digest or the full list taken before the change).
  2. Constants with one home in `bqc/negatives.py`: `PART_NUMBER_SEED_OFFSET = 100_000`, the count
     (100) and the cap factor (20); the source tag `"part_number"`.
  3. The builder takes the vocabulary (an occurrence list, used as given), the seed, and the lists to
     drop against (the arm's own training negatives, all real queries, P strings, sanity queries; PQ-8c),
     and opens no file. It draws with one `random.Random(seed)`; callers pass
     `SEED + PART_NUMBER_SEED_OFFSET + repeat_index` in the folds and `SEED + PART_NUMBER_SEED_OFFSET`
     for the final models (PQ-7b; wiring is M7.T3/M7.T4). Product words come only from the vocabulary, through the
     `words_part_number` shape, sampled uniformly from the occurrences (rule 3).
  4. 25 per shape, in `SHAPE_IDS` order. Each candidate equal, after `normalise()`, to a training
     negative, a real query, a P string, a sanity query or an earlier candidate is dropped through
     `drop_exact`; the counts per kind are returned and hold no text.
  5. Drawing stops at 100 survivors. If 2,000 candidates (20 x 100, counted over the whole call) are
     drawn first, it raises `PoolExhausted` (the existing class) with the count reached; it never pads
     and its message holds no candidate text (11.25).
  6. Each output carries `source="part_number"`, `template_id` (one of the four shape names in
     `SHAPE_IDS`) and `seed`. Every output is ASCII and equal to its own `normalise()`.
  7. An empty vocabulary raises `ValueError` (as `generate_part_numbers` does).
- **Tests:**
  - 100 outputs, 25 per shape, `template_id` values exactly `SHAPE_IDS`, `source` and `seed` set.
  - Same seed gives the same list; seeds `SEED + 100_000 + 0` and `+ 1` give different lists.
  - A sentinel word present only in the vocabulary appears in `words_part_number` outputs, and no word
    outside the vocabulary does; a vocabulary of `chair` x10 and `zen` x1 yields `chair` far more often
    (8.3, 13.17).
  - Collisions (take outputs of a run with empty lists, then pass some as a training negative, a real
    query, a P string and a sanity query): each is removed, counted under its kind, 100 still returned,
    none of the removed strings in the result, the counts or any message (11.25).
  - A candidate repeating an earlier one is counted, not kept (patch a shape to repeat).
  - A shape patched to emit one constant string raises `PoolExhausted` with the count reached, after at
    most 2,000 candidates, and the message holds no candidate text (11.25).
  - Outputs are ASCII and normalised (11.4); empty vocabulary raises `ValueError`.
  - `generate_part_numbers` output unchanged (criterion 1).
- **Depends on:** M2.T9, M2.T11, M2.T12.
- **Status:** validated (opus, 2026-11-07). Implementation `0ebc041`, corrections `908b955`, wrap
  `a5fc42f`. Mutants: 19 in the first pass (14 killed by a failing test, 2 only by a hang, 3
  survived); the three survivors were re-run against `908b955` and are now caught, and the
  drop-list order mutant stays accepted as equivalent. `a5fc42f` only wraps
  `PART_NUMBER_REMOVAL_KINDS` one kind per line, value and order unchanged;
  `awk 'length > 100' bqc/negatives.py` prints nothing; `test_negatives`: 44 tests OK.
- **Validation (`opus`, 2026-11-07, `0ebc041`): CHANGES REQUESTED.** Full suite: 863 tests OK. The
  `generate_part_numbers` digest was recomputed on the pre-change tree (`git archive 0ebc041~1`):
  `dec1b942...ba128`, equal to the pinned value. Criteria 1, 2, 4, 5 (code), 6 (code) and 7 are met;
  the committed `bqc/negatives.py` is coherent after the disclosed re-apply (one definition, the
  docstring matches the code, the tree was clean). Mutation pass in a scratch copy, 19 mutants: 14
  killed by a failing test, 2 killed only by a hang, 3 survived (two need tests, one equivalent):
  1. `tests/test_negatives.py` (no test; criterion 3): nothing pins "one `random.Random(seed)`". The
     mutant that seeds each shape with `random.Random(f"{seed}/{shape_id}")` passes all tests. Add a
     test that fails for it, for example patch `bqc.negatives.random.Random` with
     `wraps=random.Random` and assert exactly one call, with `seed`; or show that shape 2's outputs
     change when a patched shape 1 consumes extra draws from the shared `rng`.
  2. `tests/test_negatives.py:403`: `test_outputs_are_ascii_and_normalised` uses the lowercase `VOCAB`,
     whose renders are already normalised, so the mutant that drops `normalise()` on the output passes
     (criterion 6). Run it, too, with a vocabulary of words that change under `normalise()` (for
     example `["Leather", "SOFA", "Oak"]`) and assert every `words_part_number` output equals its own
     `normalise()` and holds no upper case.
  3. `tests/test_negatives.py:390`: `test_cap_counts_candidates_over_the_whole_call` stays under the
     cap, so moving the accounting to count survivors instead of candidates (`drawn += 1` per kept
     row) is caught only because the constant-shape test then loops forever. Add a case that fails
     fast: every shape yields a new string only once every 30 draws (so about 750 draws per shape and
     more than 2,000 over the call, but under 2,000 in any one shape); assert `PoolExhausted`, exactly
     2,000 render calls, and `reached < 100`. With the mutant the call returns 100 rows.
  4. `tests/test_negatives.py:275-277` and `tests/test_synth_partnumbers.py:141-143`: three blank lines
     before a top-level definition (two is the file's convention), and both files lost their final
     newline (`:419`, `:158`). Restore them.

  Not requested: the mutant that moves `training_negatives` after `sanity` in the drop lists survives;
  it only changes which kind counts a string present in several lists, which `drop_exact` defines,
  and no criterion fixes it. `drawn > cap` and the survivor-accounting mutant hang rather than fail;
  item 3 makes the latter fail fast; the former can't fail fast without a timeout and is accepted.

  Rulings: (a) `PartNumberNegative(text, source, template_id, seed)` instead of `Sample` is accepted: it
  carries the metadata 3.3 step 4 names and mirrors `Span`, the other training-only record of this
  module; M7.T3 maps it to the family tag `"part_number"` and the group `("part_number", shape)` when
  it joins the training side. (b) The count keys are accepted as they are: `training_negatives`,
  `real_queries` and `part_numbers` are the rule-4 drop-list names already in `bqc/evaluate.py:281` and
  `:300`, and `real_queries`, `part_numbers`, `sanity` match `span_removals`; `earlier` follows the 3.3
  wording ("an earlier candidate"; `build_spans` says `drawn` for its own step). M7.T3 puts the dict
  verbatim under `part_number_negatives` (see M7.T3 criterion 6) and names the new rule-4 drop list
  `part_number_negatives`, never `part_numbers`, which already means the P strings. (c) No ASCII guard
  in the builder: both vocabularies are ASCII by construction (spans are folded in 3.2 step 1, WANDS
  queries are ASCII per section 3 and 13.1), so a guard would check a property no caller can violate
  (YAGNI). M7.T3 asserts it on its outputs in both arms instead (added to its tests). (d) The "Next
  task" move to M7.T3 was premature while M7.T2 is under review; it now reads M7.T2 (corrections),
  then M7.T3.
- **Corrections (`sonnet`, 2026-11-07):** items 1 to 4 above are done in a follow-up `test(M7.T2)`
  commit.
- **Re-validation (`opus`, 2026-11-07, `908b955`): CHANGES REQUESTED.** Full suite: 865 tests OK
  (141 s). `908b955` changes tests, blank lines and final newlines only, and items 1 to 4 are done:
  `test_one_generator_is_built_and_shared_by_the_shapes` wraps `random.Random` and asserts one call
  with `SEED_PN`; `test_outputs_are_ascii_and_normalised` also runs `["Leather", "SOFA", "Oak"]` and
  checks lower case and the `words_part_number` words;
  `test_cap_fails_fast_when_survivors_are_slower_than_candidates` asserts `PoolExhausted`, exactly
  2,000 render calls and `reached < 100`; the blank lines and both final newlines are fixed. The
  three surviving mutants, each run alone in a scratch copy under `timeout` and then reverted, are
  now caught: per-shape `random.Random(f"{seed}/{shape_id}")` fails
  `test_one_generator_is_built_and_shared_by_the_shapes`; dropping `normalise()` on the output fails
  `test_outputs_are_ascii_and_normalised`; counting survivors fails
  `test_cap_fails_fast_when_survivors_are_slower_than_candidates` in 0.04 s ("PoolExhausted not
  raised"). The module run still hangs on the constant-shape test, which was accepted. One item is
  left, from `0ebc041` and missed in the first review:
  1. `bqc/negatives.py:30` is 103 columns (`PART_NUMBER_REMOVAL_KINDS = (...)` on one line); every
     other line of the four task files is within 100. Wrap the tuple over several lines, one kind
     per line or as the file wraps other long tuples, with no change to its value or order.

  Process deviations, as at M3.T9: the correction commit carries the `fable` co-author trailer, and
  it adds a line to `PLAN.md`, which `opus` owns (rewrapped here to 100 columns). Neither changes
  the code.

### M7.T3: Part-number hard negatives in the outer folds

- **Goal:** every training side built by `iter_fold_data` gets its 100 part-number hard negatives, in
  both arms and in the LOFO and catalogue-gap passes, and the run report counts them.
- **Design refs:** `DESIGN.md` `cbc2169`: 3.2, 3.3 (steps 1-4; rulings PQ-7a, PQ-8a to PQ-8d), 4 (rule 3,
  rule 4, the 1:1 rule), 8.1, 8.2, 8.3, 8.6 items 1 and 5, 9 (no clustering use), 10 (seeds), 11.7,
  11.9, 11.16, 11.18, 11.25.
- **Files:** `bqc/evaluate.py` (`iter_fold_data`, `_side` or its caller, `build_run_report`
  parameters), `tests/test_folds.py`, `tests/test_lofo.py`, `tests/test_catalogue_gap.py`,
  `tests/test_runner_comparison.py`, `tests/test_run_report.py`.
- **Acceptance criteria:**
  1. Per repeat and arm, the M7.T2 builder runs with seed `seed + PART_NUMBER_SEED_OFFSET + repeat` and
     the arm's training vocabulary: the spans' capped occurrences (the list positives use) in the
     primary arm, the training-side queries' occurrences in the comparison arm. Where that vocabulary
     changes per fold (comparison arm; catalogue-gap pass) the builder runs per fold with the repeat's
     seed, so the folds of one repeat share the seed but not the vocabulary (PQ-8d). Drop lists: the
     arm's own training negatives (the spans in the primary arm, the training-side real queries in the
     comparison arm), all 480 real queries, the P strings, the sanity queries, and earlier candidates
     (PQ-8c).
  2. The 100 rows join the training side only, label 0, with family tag `"part_number"` (never `None`)
     and their shape as group; they are never in `threshold_texts`, the test side or E, in either arm.
     In the comparison arm the out-of-fold threshold uses the real-query rows only (as
     `bqc/evaluate.py:535` does for `None`), and in the inner `StratifiedGroupKFold` the rows of one
     shape share one group, `("part_number", shape)`, like a template's rows (PQ-8a); never one group
     per row.
  3. Positives render from the arm's span or query occurrences only; no part-number negative's token
     reaches a training positive's vocabulary (3.3 step 3). Rule 4 (PQ-8b): a training positive equal,
     after `normalise()`, to one of the arm's part-number negatives is dropped through `drop_exact` and
     counted under its own kind in the side's `dropped` counts, as for the other training negatives; so
     the part-number negatives are built before the positives are screened. The test side is not
     screened against them and stays identical across arms.
  4. Quota (PQ-7a): the 100 count as negatives wherever the negative count sets the quota. Primary main
     run and primary LOFO: `(n + 100) // n_training_templates` (LOFO: over the remaining templates, today
     `n // n_remaining` at `bqc/evaluate.py:294`); comparison LOFO: `(384 + 100) // n_remaining`, with 384
     the training-side query count (`len` of the fold's training queries, not a literal); comparison
     main run: the fixed `QUOTA` 20, unchanged.
  5. LOFO (`leave_out`) and catalogue-gap (`query_classes`) passes include them in both arms.
  6. Fold report: a key `part_number_negatives` next to `span_removals` with the count, the count per
     shape and the drop counts per kind; `train.negatives` keeps the span or query count. `parameters`
     gains the count (100) and `part_number_seed_offset`. No part-number text in the report.
     Key names (M7.T2 validation ruling): the drop counts are the builder's dict verbatim, kinds
     `training_negatives, real_queries, part_numbers, sanity, earlier`; the rule-4 drop list of
     criterion 3 is named `part_number_negatives`.
  7. `PoolExhausted` from the builder propagates; `evaluate` exits non-zero with the count, as for spans
     (M4.T4 criterion 2), naming the part-number builder.
  8. The clustering experiment is untouched (section 9). The 11.18 determinism tests
     (`tests/test_run_report.py`, `tests/test_cli_evaluate.py` `DeterminismTest`) pass unchanged.
- **Tests** (stub templates, fakes):
  - Every primary and comparison training side holds exactly 100 rows tagged `"part_number"`, 25 per
    shape, label 0; none in `threshold_texts`, `test_texts` or E (both arms, LOFO and gap passes).
  - Same within a repeat and different across repeats in the primary arm; seed is
    `SEED + 100_000 + repeat` (recorded on the outputs).
  - A sentinel word only in a test-side query never reaches a part-number negative (both arms); a
    sentinel only in a training-side query reaches one only in the comparison arm (8.3, 11.9).
  - A builder patched to emit a sentinel token: that token is in no training positive.
  - `requested_per_template`, hand-computed (PQ-7a): `(n + 100) // n_training_templates` in the primary
    main run; `(n + 100) // n_remaining` in the primary LOFO pass; `(len(training queries) + 100) //
    n_remaining` in the comparison LOFO pass (stub sizes where the `+ 100` changes the result, so the old
    formula fails); `QUOTA` in the comparison main run.
  - Comparison arm and gap pass: two folds of one repeat get part-number negatives with the same seed
    but built from their own vocabularies (a sentinel only in fold 1's training queries appears only in
    fold 1's negatives).
  - Inner split groups (recording `StratifiedGroupKFold` or the groups passed to `_inner_out_of_fold`):
    the 100 rows carry 4 groups, `("part_number", shape)`, 25 rows each; real queries keep singleton
    groups.
  - Rule 4 (PQ-8b): a template patched to emit one of that side's part-number negatives drops it from
    the training positives and counts it under the part-number kind (both arms); the test side is
    unchanged, element for element, against a run without the patch.
  - Drop lists (PQ-8c): a part-number candidate equal to a training-side query (comparison arm), a span
    (primary arm), a test-side real query, a P string or a sanity query is dropped and counted.
  - Comparison arm: a fake scoring the part-number rows 1.0 doesn't change the threshold.
  - Fold report and parameters carry the counts; no candidate text anywhere in the JSON.
  - Every part-number negative is ASCII in both arms (the builder doesn't guard it; M7.T2 ruling).
  - Builder exhaustion stops `iter_fold_data` with `PoolExhausted` (11.25) and `evaluate` exits
    non-zero with the count.
  - Existing degenerate-fold, sanity-collision and determinism tests still pass (11.7, 11.16, 11.18).
- **Depends on:** M7.T2.
- **Status:** validated (opus, 2026-11-07). Implementation `6fe85a6`. Full suite 908 OK; 28 mutants,
  all killed.
- **Validation (`opus`, 2026-11-07, `6fe85a6`): PASS.** `.venv/bin/python -m unittest discover -s
  tests`: 908 tests OK (211 s). Criteria 1 to 8 are met and pinned:
  `iter_fold_data` builds once per repeat in the primary arm from the spans' capped occurrences and
  per fold in the comparison arm and the gap pass, all with `seed + 100_000 + repeat`. The drop
  lists are the arm's negatives, all real queries, the P strings and the sanity queries. The rows
  sit between the negatives and the positives with label 0, family `"part_number"` and group
  `("part_number", shape)`. They never reach `threshold_texts`, the test side or E. The comparison
  threshold still takes `f is None` rows only (`bqc/evaluate.py:574`). The training positives are
  screened under `part_number_negatives` and the test side is not. The quota follows PQ-7a. The
  fold report gains `part_number_negatives`, and `parameters` gains `part_number_count` and
  `part_number_seed_offset`. `bqc/cluster.py` and `bqc/cli.py` are unchanged (`git diff
  6fe85a6~1 6fe85a6`; no `part_number` in `cluster.py`). The determinism tests were not edited.

  Mutation pass: each mutant was applied alone in a scratch copy (`git archive 6fe85a6`) and run
  against `test_folds`, `test_runner_comparison`, `test_catalogue_gap`, `test_run_report` and
  `test_lofo` (`test_cli_evaluate.FailureTest` for the message mutant). All 28 were killed by
  failing tests:
  - Screening: the positives not screened against the rows (built after); the drop kind renamed;
    the test side screened against the rows; the rows put on the test side.
  - Build frequency: the primary arm built per fold; the gap pass's primary arm built once per
    repeat.
  - Vocabulary: the comparison vocabulary taken from all 480 queries; the primary builder given an
    uncapped vocabulary; part-number tokens added to the positives' vocabulary.
  - Report: `removals` zeroed; `per_shape` miscounted; `train.negatives` counting the 100; one
    `parameters` key missing.
  - Tags and threshold: family `None`; one group per row; the rows in the comparison threshold
    set; the rows in `threshold_texts`.
  - LOFO: the builder skipped in the comparison arm and, separately, in the primary arm.
  - Seed: the offset dropped; the repeat dropped.
  - Quota: the primary arm on the old formula; the comparison LOFO pass on the old formula; the
    comparison main run on the new formula.
  - Drop lists: the comparison arm dropping against the spans; against the training queries only
    instead of all 480; sanity omitted; the primary arm dropping without the P strings.
  - Exit message: the builder's message without "part-number".

  Rulings: (a) the `NoRemovedTextTest` fixture moving to `n_splits=5` is accepted. Reverted
  alone, the test errors with `FoldError: training fold lacks families ['A']`, the stub effect 13.24
  describes. At five folds it still runs both arms and asserts `dropped > 0`, so it loses no
  coverage. No other test changed `n_splits`. The inner 11.16 guard is still pinned by
  `tests/test_runner_comparison.py:170` (`lacks families ['D']`), which this commit did not touch.
  (b) The sub-keys `{count, per_shape, removals}` stand. `removals` is the builder's dict verbatim
  (`PART_NUMBER_REMOVAL_KINDS`) and parallels `span_removals`. (c) The primary quota divides by
  `len(train_templates)`, which is correct: 3.3 says `n_training_templates`, and the literal 24
  belongs only to the final model (M7.T4). (d) The two lines over 100 columns
  (`tests/test_folds.py:425`, `tests/test_runner_comparison.py:131`) date from `9c7eb85`/`8ca9124`
  and were not touched; noted only. (e) `cli.py` is unchanged and correct: `PoolExhausted` is
  already caught at `bqc/cli.py:481`. The builder's message reads `reached N of 100 part-number
  negatives after M candidates` (`bqc/negatives.py:197`), and
  `test_exhausted_part_number_pool_exits_non_zero_with_its_count` pins "part-number", the count and
  the absence of the candidate text.

  Notes, not corrections:
  - The primary builder runs even in a comparison-only main run. This matches `build_spans` there.
  - `tests/test_folds.py` restates the removal kinds as `REMOVAL_KINDS_PN` where `test_run_report`
    imports `PART_NUMBER_REMOVAL_KINDS`. This is acceptable as a pin of the M7.T2 key ruling.
  - A stray blank line was added after `class DegenerateFoldTest` (`tests/test_folds.py:712`).
  - 13.24 gives "Python 3.13" without the patch version.

### M7.T4: Part-number hard negatives in the final models

- **Goal:** the final-model fits of 8.7 include the 100 part-number hard negatives, never in the
  threshold set.
- **Design refs:** `DESIGN.md` `cbc2169`: 3.3 (steps 2-4; rulings PQ-7a, PQ-7b, PQ-8a to PQ-8c), 8.2, 8.6
  item 4, 8.7, 11.25.
- **Files:** `bqc/evaluate.py` (`fit_final_model`), `tests/test_final_model.py`.
- **Acceptance criteria:**
  1. Both arms: the builder runs once on the final fit's vocabulary (spans' capped occurrences; all real
     queries' occurrences) with seed `SEED + PART_NUMBER_SEED_OFFSET` (PQ-7b), and the 100 rows join the
     training side with the `"part_number"` tag. Drop lists (PQ-8c): the arm's own training negatives
     (the final spans; all 480 real queries), all 480 real queries, the P strings, the sanity queries and
     earlier candidates. A training positive equal to one of them is dropped and counted (PQ-8b, as in
     M7.T3). In the comparison arm's inner split the 100 rows form 4 groups, `("part_number", shape)`
     (PQ-8a).
  2. The threshold uses the real queries' scores only: all 480 in the primary arm; in the comparison arm
     the out-of-fold scores of the real-query rows only (today `[:len(negatives)]`,
     `bqc/evaluate.py:1081`, must not take part-number rows).
  3. Quota (PQ-7a): primary arm `(n + 100) // 24`; comparison arm the fixed `QUOTA` 20, unchanged.
     `FinalModel.report` carries the `part_number_negatives` counts as in the fold report, no text.
  4. The part-number diagnostic (8.6 item 4) and sanity scoring are unchanged in shape; the diagnostic
     strings are disjoint from the hard negatives by construction (they are in the drop list).
- **Tests:**
  - Both arms: 100 `"part_number"` rows in the fit (recording fake), none among the threshold scores (a
    fake scoring them extreme leaves `t` unchanged).
  - Primary arm threshold over exactly the 480 real queries; comparison arm over exactly the 480
    out-of-fold real-query scores.
  - Report counts present, no text; quota hand-computed: `(n + 100) // 24` in the primary arm (an `n`
    where it differs from `n // 24`), 20 in the comparison arm.
  - Seed: the outputs' `seed` is `SEED + 100_000` in both arms, and the 100 equal the builder's output
    for that seed and vocabulary.
  - No final-model hard negative equals a P string, a real query or a sanity query.
  - Comparison arm: the inner groups of the 100 rows are the 4 `("part_number", shape)` groups.
  - Rule 4: a template patched to emit a final-model part-number negative has that positive dropped and
    counted.
- **Depends on:** M7.T2, M7.T3.
- **Status:** validated (`opus`, 2026-11-07, `801b08f`). Corrections 1 and 2 verified; see the
  re-validation note below.
- **Validation (`opus`, 2026-11-07, `9fba29c`): CHANGES REQUESTED.** `.venv/bin/python -m unittest
  discover -s tests`: 917 tests OK (240 s). The code meets criteria 1 to 4: one builder call per fit
  with `SEED + PART_NUMBER_SEED_OFFSET` on the fit's vocabulary and the PQ-8c drop lists; the rows
  tagged `"part_number"` and grouped by shape through `_side`; the comparison threshold taken from
  the `None`-family out-of-fold rows (`bqc/evaluate.py:1127`), the primary one from the 480 real
  queries; the primary quota `(len(negatives) + len(hard)) // 24`, `QUOTA` in the comparison arm;
  rule 4 under `part_number_negatives`; `report["part_number_negatives"]` from
  `_part_number_report`. Only `tests/test_final_model.py` changed among the tests, so the
  M3.T11/M3.T12 diagnostic and sanity tests pass unedited.

  Mutation pass: each mutant was applied alone to a `git archive 9fba29c` copy and run against
  `test_final_model` (29 tests). 21 of 24 were killed: drop lists without the sanity queries, the P
  strings or the real queries; the primary builder given the real queries as training negatives;
  the comparison threshold taken by position (`[:len(negatives)]`) after `_side` puts the rows
  first; the slice `[:len(negatives) + len(hard)]`; the filter `f != "part_number"`; `per_shape`
  keyed by source; `per_shape` over 99 rows; the block under `part_numbers`; the block nested in
  `dropped`; `removals` zeroed; a second builder call, used or discarded; the seed without `SEED`;
  the rule-4 list missing; the rule-4 kind renamed; part-number tokens in the positives'
  vocabulary; the rows left out of the primary fit. Survivors: (i) **the primary builder given the
  uncapped span vocabulary**, and likewise the comparison builder given a capped one: the fixture
  never reaches the cap (120 spans against `VOCABULARY_CAP` 10 000), so capped and uncapped are
  equal (item 1). (ii) The `per_shape` counts swapped between shapes is an equivalent mutant (all
  25). (iii) `(n + 99) // 24` equals `(n + 100) // 24` at `n = 299`, so it is not distinguishable
  there. That is acceptable: the criterion asks only for an `n` where the result differs from
  `n // 24`, and the code uses `len(hard)`. The `earlier` mutant lives in the builder, which the
  test and its oracle (`expected_hard`) share, so it was skipped here. `test_negatives` (M7.T2)
  pins it.

  Corrections:
  1. `tests/test_final_model.py:239-252` (`test_the_builder_runs_on_the_fits_vocabulary`) and the
     `fit` helper at `:59`: the cap never binds, so the builder could receive the uncapped span
     vocabulary (criterion 1, "spans' capped occurrences") without a failing test. Required fix:
     let `fit` pass `vocabulary_cap`, run this test with a cap smaller than the spans' token count
     (for example 6), assert that the capped and uncapped span occurrences differ (the cap binds),
     and assert `primary.args[0] == token_occurrences(span_texts, cap=6)` and
     `comparison.args[0] == token_occurrences(REAL)` (uncapped).
  2. `bqc/evaluate.py:1092`: the edited docstring line is 113 columns, and the project wraps at 100
     (see the M7.T2 correction `a5fc42f`). Required fix: rewrap the docstring.

  Rulings: (a) F1, process: the TDD order slipped (implementation before tests). The mutation
  pass is not enough evidence on its own, because of item 1. With item 1 fixed it will be, and no
  rework beyond item 1 is needed. (b) Comparison-arm real-query collision: forcing a collision
  there would need a candidate equal to a real query, which is also the comparison vocabulary. The
  drop-list assertion `list(call.args[3]) == REAL` for both calls is enough, together with the
  builder's own drop tests (M7.T2). The "no real queries" mutant is killed. (c) "Next task" now
  reads M7.T4 (corrections), then M7.T5.
  Addressed in the `test(M7.T4)` correction commit (items 1 and 2).
- **Re-validation (`opus`, 2026-11-07, `801b08f`): PASS.** `.venv/bin/python -m unittest discover
  -s tests`: 917 tests OK (182 s). Item 1: `fit` forwards `**kwargs` (so `vocabulary_cap`) to
  `fit_final_model`. `test_the_builder_runs_on_the_fits_vocabulary` runs both arms with cap 6,
  asserts that the capped and uncapped span occurrences differ, and pins the primary call to
  `token_occurrences(span_texts, cap=6)` and the comparison call to `token_occurrences(REAL)`
  (uncapped). Item 2: no line over 100 columns in `bqc/evaluate.py` or
  `tests/test_final_model.py`. The `bqc/evaluate.py` diff is the docstring rewrap only. Survivors
  from (i), each applied alone to a `git archive 801b08f` copy and run against `test_final_model`:
  the primary builder given the uncapped span vocabulary (`bqc/evaluate.py:1103`) and the
  comparison builder given a capped one (`:1107`). Both are killed by
  `test_the_builder_runs_on_the_fits_vocabulary` (29 tests, 1 failure each).

### M7.T5: `lr_hand` adapter

- **Goal:** the fourth `Scorer`: logistic regression on the TF-IDF matrix plus the three standardised
  hand features, registered and on by default.
- **Design refs:** `DESIGN.md` `cafa701`: 5, 6.2, 6.3, 6.4, 6.5, 10 (default `--models`), 11.1, 11.2,
  11.3, 11.12, 13.23 (M7.T1).
- **Files:** `bqc/models.py`, `bqc/evaluate.py` (the `LR_HAND = "lr_hand"` name next to `LR` and `MLP`),
  `bqc/cli.py` (`MODEL_FACTORIES`), `tests/test_models_lr_hand.py` (new test file),
  `tests/test_cli_evaluate.py` (model enumerations: `ALL_MODELS` at `:28`, the default parse at
  `:469`).
- **Acceptance criteria:**
  1. A module-level class on `_VectorisedScorer`, no constructor arguments: `LogisticRegression(C=1.0,
     max_iter=1000)` on the M3.T1 TF-IDF output horizontally stacked (sparse, per 13.23) with the three
     `_hand_features` columns standardised by a `StandardScaler` fitted in `fit` only. `_hand_features`
     is reused, not copied; the composition is the one M7.T1 found to stay sparse.
  2. `normalise` once per text in `fit` and `score`; scores are P(label 1) in `[0, 1]`;
     `NotFittedError` before fit on the empty and non-empty paths; a fitted instance pickles and scores
     identically after the round trip.
  3. Full `get_params()` pinned: the classifier against `LogisticRegression(C=1.0, max_iter=1000)`, the
     TF-IDF part against `make_vectoriser()`, the scaler against `StandardScaler()`.
  4. `MODEL_FACTORIES` gains `lr_hand`; default `--models` is `lr,mlp,trivial,lr_hand` (section 10
     order). The MLP-only rules (comparison-arm LOFO, the 5-minute message) don't apply to it.
  5. Any probe-derived fact the code relies on is cited in a comment or docstring (13.23).
- **Tests:**
  - Configuration: the three `get_params()` equalities.
  - The fitted classifier has `len(vocabulary_) + 3` coefficients (fails if the hand columns are
    dropped); the transformed matrix is sparse.
  - Scaler statistics equal the training rows' hand features (normalised text), and scoring another
    batch leaves them unchanged; a text scores the same alone and inside a batch.
  - Hand features from normalised text: `"sofa ​ bed"` scores as `"sofa bed"` (the M4.T3 survivor).
  - `normalise` called once per text in fit and in score (patch count, as `tests/test_models_lr.py`).
  - `""` and an all-unknown string give finite scores without raising (11.1, 11.3); a 10 kB input
    scores (11.2).
  - `NotFittedError` on both paths; pickle round trip; refit determinism; scores in `[0, 1]`.
  - CLI: default parse lists four models; `--models lr_hand` is valid; an unknown name is still a usage
    error.
- **Depends on:** M7.T1, M4.T1, M4.T3, M4.T4.
- **Status:** validated
- **Validation (`opus`, 2026-11-07, `d359c6d`): PASS.** `.venv/bin/python -m unittest discover -s
  tests`: 936 tests OK (238 s, on a host shared with the mutation runs). Criterion 1:
  `LRHandScorer` is module-level on `_VectorisedScorer`, no constructor arguments; the
  `FeatureUnion` of `make_vectoriser()` and `make_pipeline(FunctionTransformer(_hand_features),
  StandardScaler())` is 13.23's sparse composition, and the hand branch gets the base class's
  normalised text. Criterion 2: base-class `fit` / `score` reused (normalise once, P(1),
  `NotFittedError` on both paths), pinned with pickle, refit, range, `""`, `"☃☃"` and 10 kB tests.
  Criterion 3: the three `get_params()` equalities plus `func is _hand_features`. Criterion 4:
  `LR_HAND` next to `LR` / `MLP`, last in `MODEL_FACTORIES`, so the default is
  `lr,mlp,trivial,lr_hand`; the comparison-arm LOFO skip and the fit timer key on `MLP` only, and
  the extended LOFO test asserts `lr_hand` runs A-D in the comparison arm. Criterion 5: the
  docstring cites 13.23 with date and versions. 11.12: no warning suppression.
  Mutants, each alone on a `git archive d359c6d` copy, all killed: `StandardScaler(with_mean=False)`,
  no scaler, hand branch dropped, `C=0.5`, `max_iter=500` (config tests); the hand branch on raw
  text through a `_hand_features` wrapper (config test, and behaviourally by the `"sofa ​ bed"`
  test; the scaler-statistics test alone wouldn't catch it, as `TEXTS` is already normalised);
  `_hand_features` wrapped in a lambda (config and pickle tests); branch order swapped (caught by
  the positional config and coefficient tests; the scores would be equivalent up to lbfgs
  rounding, since L2 logistic regression is invariant to column order); `LR_HAND = "lr_hnad"` (15
  errors and 3 failures in `test_cli_evaluate`); `lr_hand` left out of the default but registered,
  and `lr_hand` placed second in `MODEL_FACTORIES` (`test_defaults`,
  `test_bare_flags_cover_all_of_models`); the MLP fit timer applied to `lr_hand` (the
  perf-counter test runs out of its two ticks); the comparison-arm LOFO skip applied to `lr_hand`
  (the extended LOFO test).
  Rulings: (a) real `lr_hand` in the name-based CLI tests: a note, not a correction. Per-test
  timing against `d359c6d^`: 14 s of the increase is `DeterminismTest`, which keeps the real models
  (11.18, as in the M4.T4 ruling); the comparison-arm LOFO test adds 3.8 s and every other test 1.1 s
  or less. The M4.T4 corrections faked tests costing 6-14 s each, so a fake here would save little.
  (b) `bqc/cli.py:89` is 104 columns, unchanged by this task (it was `:87` before): a note for the
  next task that edits `bqc/cli.py`. (c) The extended LOFO comparison-arm test is good: the
  `lofo_skip_lrhand` mutant shows it pins the MLP-only rule.

### M7.T6: `model_comparison_hand` and section coverage for `lr_hand`

- **Goal:** the run report compares `lr_hand` with `lr` under each arm, and every section that covers
  `lr` covers `lr_hand`.
- **Design refs:** `DESIGN.md` `cafa701`: 6.5, 7 (verdict rule, `model_comparison_hand`), 8.2, 8.6 items
  1, 3, 4 and 5, 10 (PQ-5 section models).
- **Files:** `bqc/evaluate.py` (`build_run_report`), `tests/test_paired_comparison.py` or
  `tests/test_run_report.py`, `tests/test_cli_evaluate.py`.
- **Acceptance criteria:**
  1. With `lr` and `lr_hand` both in `models`, each arm gets `model_comparison_hand` =
     `paired_comparison(lr_hand folds, lr folds)`: `lr_hand` minus `lr`, the shape and verdict rule of
     section 7 (no new comparison code). Absent if either model didn't run. `model_comparison` (MLP
     minus LR) unchanged.
  2. Sanity section, part-number diagnostic, LOFO (both arms, including the comparison arm, where only
     the MLP is never run) and catalogue gap include `lr_hand` wherever they include `lr`, and the
     `--lofo` / `--catalogue-gap` model lists accept it (PQ-5).
  3. `arm_comparison` covers `lr_hand` like every model.
- **Tests** (fakes, hand-computed):
  - Per-fold differences, mean, sd, se, wins and `justified` / `worse` for a hand-built pair of fold
    lists, with the sign `lr_hand` minus `lr` (swapping the arguments flips it).
  - Present under each arm with both models; absent with only `lr`, only `lr_hand`, or neither.
  - Body key sets updated where `tests/test_run_report.py` pins them.
  - CLI with a fake under `lr_hand`: rows in sanity, part-number diagnostic, LOFO in both arms and
    catalogue gap; `--lofo lr,trivial` reports `lr_hand`'s LOFO rows as not run with the reason, while
    its main fit and final model are present.
- **Depends on:** M7.T5, M4.T8.
- **Status:** todo

### M7.T7: Re-run on real data and report

- **Goal:** the M6.T1 procedure at design v2.8, with all four models and the part-number hard
  negatives, reported to `fable`.
- **Design refs:** `DESIGN.md` `cafa701`: 3.3, 6.5, 7, 8.2, 8.5, 8.6 (items 1-5), 8.6.1, 9, 10, 12.1
  (items 18-20), 13.9.
- **Files:** none committed (`results/` is gitignored), apart from this task's status and report.
- **Acceptance criteria:**
  1. From a clean clone at the commit after M7.T6 with a fresh `.venv`: `fetch`, `generate`, one
     `evaluate --negatives product,real --lofo --catalogue-gap` with the default (all four) models, and
     `cluster` complete. Wall time per command, the MLP fit time (5-minute rule, section 10) and the
     run JSON paths are reported.
  2. Every table of M6.T1's report, now with `lr_hand`: headline per arm and model, per-family and
     per-template recall, LOFO, comparisons, cluster, sanity (both views), part-number diagnostic,
     one-word false positives, false positives with the eye review (13.9), catalogue gap.
  3. `model_comparison_hand` under each arm (recall, FPR gate, ROC-AUC, `justified`, `worse`), next to
     `model_comparison` and `arm_comparison`.
  4. LOFO rows of `lr_hand` against `lr`, per family and arm, with the difference in held-out recall;
     whether the transfer on B and D improves (6.5).
  5. Part-number diagnostic before and after: M6's 85% for `lr` (and the other models' M6 figures)
     next to this run's fraction flagged, overall and per shape, per model and arm; sanity rows 16, 39
     and 43 per model and arm.
  6. The part-number negatives' realised counts and drop counts from the fold reports (per arm,
     summed, any non-zero kind named).
  7. The recommendation under section 7: `lr_hand` replaces `lr` only if `model_comparison_hand` is
     `justified` in the primary arm; otherwise `lr` stays. M6-to-M7 differences in `lr`'s own numbers
     are shown as context, labelled as not a paired comparison (8.2). Nothing is acted on: the user
     decides.
- **Tests:** none (no code). The determinism tests cover reproducibility.
- **Depends on:** M7.T1 to M7.T6.
- **Status:** todo
