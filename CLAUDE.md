# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Message format

Start every message to the user with:

`aye, aye Captain; TL;DR: <one-sentence summary of what follows>`

Reports from one agent to another (subagent results) are exempt.

## Project

...

## Workflow

Work moves through the roles in a fixed loop. The agent files hold the details.

1. **Design** (`fable`): decisions and their rationale go into `DESIGN.md`. Only `fable` edits it, except that `sonnet` records probe findings in section 13.
2. **Plan** (`opus`): `PLAN.md` holds milestones and tasks with ids like `M1.T3`, each with design refs, acceptance criteria, and tests.
3. **Implement** (`sonnet`): one task per delegation, one commit per task. Corrections after review are separate commits (see Commits).
4. **Validate** (`opus`): verdict is PASS or CHANGES REQUESTED with `file:line` items.
5. **Sign off** (`fable`): per milestone, after running the suite and reading the diff.

Task status: `todo` -> `in progress` -> `done` (sonnet) -> `validated` (opus) -> `signed off` (fable).

Subagents start with a fresh context. Every delegation names the `PLAN.md` task id and the `DESIGN.md` sections to read.

## Architecture

...

## Principles

- **Simple, working solutions first.** Choose the most direct implementation that meets the requirement, and make it work before making it elegant.
- **DRY.** Every rule, constant, schema, and piece of knowledge has exactly one home. This applies to knowledge, not to any two lines that happen to look alike: don't extract an abstraction from a single occurrence.
- **YAGNI.** Build what the current task needs. No speculative options, extension points, or configuration for hypothetical futures. If something seems useful later, note it under "Later" in `DESIGN.md` instead of building it.
- **SOLID.** Single responsibility per class and module. Stages depend on ports (interfaces), never on vendor SDKs. New models are new adapters, not edits to stages.
- **Standard library first.** Add a dependency only when the standard library can't reasonably do the job. Justify each new dependency in the commit message and add it to `pyproject.toml`.
- **Don't assume API or runtime behavior.** When unsure how a library, API, or Python feature behaves, probe it: write a minimal script in `scratch/` (gitignored), run it, and record the finding with its date and version in `DESIGN.md` section 13 or in a comment next to the code that depends on it.
- **TDD, in this order:**
  1. Skeleton: modules, classes, signatures, types, docstrings, bodies raising `NotImplementedError`.
  2. Unit tests, including edge cases and failure paths.
  3. Run the tests and confirm they fail for the expected reason.
  4. Implement until the tests pass.
  5. Refactor with the tests green, then run the full suite.
- **Think about what can go wrong.** Before implementing, list the edge cases and failure modes (bad input, partial failure, provider errors, concurrency, encoding) and cover them with tests or explicit handling. `DESIGN.md` section 11 is the starting list.

## Commands

...

## Commits

The user has authorized agents to commit without asking (2026-11-05). This doesn't cover pushing.

- Commit directly on `master`. Don't create branches.
- Format: Conventional Commits, `<type>(<scope>): <summary>`. Pick the type from the content: `feat`, `fix`, `test`, `refactor`, `docs`, `build`, or `chore`.
- For work on one `PLAN.md` task, the scope is the task id, for example `feat(M1.T3): add canonical JSON hashing`. Otherwise the scope is optional.
- One commit per implemented task.
- Corrections requested by the reviewer go in a new commit with the same scope. Never amend or squash them into the implementation commit.
- Each role commits its own changes before handing back, so the working tree is clean at every handoff.

## Guardrails

...
