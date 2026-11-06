# PLAN.md: bot-query-classifier

Owner: `opus`. Plans `DESIGN.md` v2.7 (commit `b025fde`). Created 2026-11-05.

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
| M3 | Features and evaluation harness against a fake `Scorer`, both arms, catalogue-gap check, SMOTE dependency removal | 14 |
| M4 | Real models, `evaluate` command, paired comparisons | 5 |
| M5 | HDBSCAN experiment (unchanged by v2.0) | 3 |
| M6 | Full run on real data | 1 |

Task ids stay stable: `M3.T6` and `M4.T5` were probe tasks, and `M4.T6` and `M4.T7` were the SMOTE tasks
(dropped by the user, 12.1 item 17). All four are retired and aren't reused.
`M1.T7` was also a dropped probe; the id now names the fetch fix below, and the old probe
appears only in earlier versions of this plan. Tasks run in file order, which isn't always id
order: M2.T9, M2.T10, M2.T11 and M2.T12 come before M2.T8, M3.T13 comes after M3.T5, M3.T15 comes
after M3.T13, and M3.T14 comes after M3.T9.

v2.0 doesn't reopen validated work: changes to validated code are new tasks (M1.T8, M2.T10,
M2.T11), and earlier entries keep their statuses and the design version they were validated against.

**Next task:** M3.T8.

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
| Wilson reference values | 13.13 | M3.T7 |
| Part-number shapes in WANDS | 13.15 | M2.T9, M3.T12 |
| `product.csv` format and coverage | 13.18 | M1.T8, M2.T12 |
| Product-name spans as training negatives | 13.19 | M2.T12, M3.T5, M3.T14 |

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

### Where the user acts

- **Sanity labels: done.** All 50 rows are labelled (`7332c6b`), so real runs score the sanity
  section. No agent modifies `data/sanity.csv`. A template that can emit a sanity query
  changes; the file doesn't (rule 4).
- **Q12: after M6.T1.** M6.T1 reports the part-number diagnostic so the user can decide
  whether to add synthetic part-number negatives (12.2). No task waits for the answer, and
  this plan schedules no work for it.
- **LR or MLP: after M6.T1.** M6.T1 carries a recommendation; the user decides (12.1, item 13).
- **Catalogue gap: only if the check fails in M6.T1.** What to do then is a user decision
  (8.6 item 5).

12.1 items 1 to 11 and 13 to 17 are closed; embeddings were abandoned for latency (item 15), truncated SVD
on the section 5 features was rejected (item 16), and SMOTE was dropped (item 17). Off-domain queries stay outside both classes, so no task
exists for them. SKUs and part numbers are legitimate (Q11, 2.2).

### Open questions for `fable`

PQ-1 to PQ-4 are answered in `DESIGN.md` 12.3 and PQ-5 in section 10, and all are applied below; their
text is in `530de4f` and `a3cf42e`. PQ-6 is answered in section 7, and its text is in `f0c12cd`. None is
open.

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

Sign-off: pending.

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
- **Status:** done
- **Status note:** implemented in `3d6f709`, with F7. Real run by `opus`, all 15 folds of both arms, scratch
  logistic regression, twice in separate processes with different `PYTHONHASHSEED`: the two files are
  byte-identical, and their names differ only in the timestamp. In the file: no sanity or part-number string,
  no span or positive text (the only matches are schema keys and the model name `lr`), and every false
  positive is a WANDS query. The realised mix sits next to the target (mean 3.46-3.49 against 3.38), as in
  13.20. The seed is recorded by `write_report` from the same `SEED`, so criterion 1 holds with no argument.
  `config_hash` changes with each of the nine parameters. Process, disclosed: an empty heredoc against
  `bqc/evaluate.py` that changed nothing, and the commit message through a heredoc.
- **Changes requested:**
  1. `bqc/evaluate.py:675`: `iter_fold_data` is called with 13 positional arguments, and the tests don't notice
     `sanity` or `part_numbers` not being passed, `vocabulary_cap` replaced, or `test_per_template` and
     `eval_per_template` swapped. The first would let a sanity query into a fit with no error (rule 4, 8.6.1).
     The real run is right today. Pass them by keyword, and add a test through `build_run_report` next to
     `tests/test_run_report.py:222` that each reaches `iter_fold_data`, including that a positive equal to a
     sanity query raises `FoldError`.
  2. `tests/test_run_report.py:242`: dropping `now` in `run_report` passes. Assert the given stamp in the name.
  3. `tests/test_run_report.py:123`: every fold warns alike, so reading the wrong fold's warnings passes. Add
     a scorer that warns in one fold only.

### M3.T9: Leave-one-family-out

- **Goal:** report recall and achieved FPR on each family at a threshold chosen without it.
- **Design refs:** `DESIGN.md` `6cec6c6`: 8.2, 8.6 (item 1), 11.16; `a8b8600`: 11.30.
- **Files:** `bqc/evaluate.py`, `tests/test_lofo.py`.
- **Acceptance criteria:**
  1. For each F in A-D (E has no LOFO row), per outer fold and per arm: the arm's training negatives
     stay unchanged, and positives come from the training-side templates of the other three families
     only, at `n // n_remaining_training_templates` each, so the training side stays 1:1.
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
- **Status:** todo

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
- **Status:** todo

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
- **Status:** todo

### M3.T11: Sanity section of the run

- **Goal:** score the sanity set with the final models when it's fully labelled, and skip
  it otherwise.
- **Design refs:** 8.6.1, 11.20, 11.21, 11.30.
- **Files:** `bqc/evaluate.py`, `tests/test_sanity_section.py`.
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
- **Status:** todo

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
- **Status:** todo

---

## M4: Real models and paired comparisons

Sign-off: pending.

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

- **Goal:** run the harness with the real models from the command line, in either arm.
- **Design refs:** `DESIGN.md` `0386ca8`: 8.2, 8.4, 8.5, 8.6, 10, 13.9, 13.22.
- **Files:** `bqc/cli.py`, `tests/test_cli_evaluate.py`, `DESIGN.md` (section 13.9, the measurement only).
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
     is valid with either arm.
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
- **Depends on:** M2.T8, M3.T9, M3.T11, M3.T12, M3.T14, M4.T1, M4.T2, M4.T3.
- **Status:** todo

### M4.T8: MLP-vs-LR and between-arm paired comparisons

- **Goal:** compute the paired differences M6 reports: MLP against LR, and primary against comparison
  arm.
- **Design refs:** `DESIGN.md` `b025fde`: 6.2, 7, 8.2, 12.1 (items 13, 17).
- **Files:** `bqc/evaluate.py`, `tests/test_paired_comparison.py`.
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
- **Status:** todo

---

## M6: Full run

Sign-off: pending.

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
- **Status:** todo
