# Agent Instructions

Read `AGENTS-LOCAL.md` when present. It contains repository-machine-specific
operating rules and must remain local only; do not stage or commit it.

Use `$comfyui-custom-node-development` for all development, testing, review, and packaging work in this repository.

If the skill is unavailable, stop and tell the user that it must be installed before continuing.

## Local ComfyUI Context

<!-- comfyui-custom-node-context:start -->
comfyui_root: C:\Users\ishim\Tools\ComfyUI
custom_nodes_root: C:\Users\ishim\Tools\ComfyUI\custom_nodes
repository_root: C:\Users\ishim\Tools\ComfyUI\custom_nodes\ComfyUI-UtilsCollection
virtual_environment_root: C:\Users\ishim\Tools\ComfyUI\.venv
python_executable: C:\Users\ishim\Tools\ComfyUI\.venv\Scripts\python.exe
<!-- comfyui-custom-node-context:end -->

## Repository-Specific Instructions

### Evidence and claim discipline

- Do not present assumptions, plausible explanations, remembered behavior, or
  pattern-based guesses as established facts.
- Before making a causal, behavioral, compatibility, or architectural claim,
  verify it against direct evidence covering the relevant path. Inspect the
  actual source, configuration, inputs, consumers, connections, and runtime
  assembly needed to support the claim.
- Distinguish verified facts, supported inferences, and unknowns explicitly.
  When required evidence is unavailable, state that the conclusion cannot yet
  be determined instead of filling the gap with a likely explanation.
- Do not infer an end-to-end result from an isolated helper, preset, schema,
  unit test, string-construction test, or passing test suite. Verify the actual
  integration and consumer behavior before generalizing beyond what was tested.
- Do not invent workflow topology, data flow, precedence, model behavior, or
  user intent. Inspect the relevant artifact or ask for the specific missing
  evidence when it cannot be discovered locally.
- Do not generalize a model or prompt protocol from a shared tokenizer, helper,
  socket type, or similarly named node. Verify the target model, node contract,
  template assembly, and consuming path. A node-specific prefix or label does
  not establish generic encoder behavior.
- When correcting an unsupported claim, identify exactly which parts were
  verified and which parts were assumed. Do not replace one unsupported
  explanation with another.
- Accept failure and request guidance or advice on the approach when continued
  attempts are unlikely to work and are being made only to force the task toward
  apparent completion; do not continue speculative trial-and-error to avoid
  acknowledging that the current approach has failed.

### Python module structure

- Keep `*_nodes.py` modules limited to node classes, schemas, registration
  collections, constants, and thin execution orchestration.
- Do not add module-level helper functions to `*_nodes.py`. Put algorithms,
  tensor operations, parsing, geometry, model handling, and reusable execution
  logic in a focused domain `*_helpers.py` module.
- Extend an existing helper module when its domain matches. Do not create
  catch-all helper modules or move unrelated domains into an existing helper.
- Do not extract trivial forwarding or single-expression functions unless they
  are reused or required by an external interface.
- Treat the allowlist in `tests/test_node_module_structure.py` as frozen legacy
  debt. Do not expand it. Move relevant logic out of a node module instead.
- Before adding behavior to a node module that already contains standalone
  helpers, refactor the affected helper logic into the appropriate helper
  module rather than increasing the node module's standalone-function count.

### Test selection

- Use `tests/run_tests.py` as the repository-owned test selector.
- During iteration, run an exact test or one explicit `--group`; do not run the
  complete suite after every edit.
- Before handoff, run `tests/run_tests.py --changed` once for the accumulated
  worktree changes.
- Use `tests/run_tests.py --final` only as the deliberate broader gate. Do not
  repeat it when no relevant files changed after a successful run.
- An unmapped production source file is an error. Add or correct its entry in
  `tests/test_groups.toml` instead of silently skipping coverage.
- Local untracked artifacts are intentionally excluded. Stage a new production
  source file before relying on `--changed`, or run its intended `--group`
  explicitly while creating it.

### Managed VLM presets

- Treat serialized values in `vlm_presets.py` as managed runtime data. Never
  read, git diff, print to console or hand-edit a serialized VLM preset literal.
- Treat whitespace inside serialized or triple-quoted preset values as content.
  A `git diff --check` trailing-whitespace warning may originate inside a preset
  string; report it, but never trim, normalize, format, or otherwise alter that
  whitespace without explicit user authorization.
- Before changing a managed VLM preset, inspect the matching tracked `*_vars.py`
  authority identified by `CONFIG` in
  `scripts/manage_vlm_preset_authorities.py`. Also inspect any narrowly relevant
  `local/sync_vlm_*.py` utility to understand its preset-specific relationship.
  Local utilities are focused validators, not synchronization authorities: run
  them without `--apply` and do not use their `--apply` paths during the normal
  managed workflow.
- Edit the matching readable authority first. Run every relevant local validator
  without `--apply`, then run `scripts/manage_vlm_preset_authorities.py` without
  `--apply`. After both validations succeed, apply synchronization only with
  `scripts/manage_vlm_preset_authorities.py --apply`. Rerun the manager without
  `--apply` until its `changed` list is empty, rerun relevant local validators
  without `--apply` until they report synchronized state, and inspect the bounded
  generated diff. Do not begin test edits or test execution before this sequence
  is complete.
- When `CONFIG` in `scripts/manage_vlm_preset_authorities.py` does not cover the
  requested readable/runtime pair, extend that tracked manager or add compatible
  deterministic tooling before changing runtime data. Do not fall back to
  ambiguous dictionary-boundary patches or manual replacement of generated
  literals.
- Inspect the resulting bounded runtime diff and test both readable/runtime
  synchronization and the consuming node's exposed preset options.
- Treat `{user_query}` and `{system_query}` as legacy injection markers unless
  the currently verified assembly path proves otherwise. Do not use either as
  a substitute for the regular user message when assigning image roles,
  timestamps, segment structure, or mixed-media intent.
- Audit the complete assembled prompt seen by the VLM: system instruction,
  system-query prefix, regular user request, suffix, legacy injected values,
  and downstream consumer. Rules in named sections are not isolated; repeated
  or contradictory wording affects the whole response.
- Phrase-presence and snapshot tests do not establish prompt behavior. Add an
  assembled-context contract test for field allocation, media-label scope,
  timestamp preservation, final-subject continuity, and forbidden competing
  instructions. Treat direct generation output supplied by the user as
  first-level evidence of remaining prompt defects.
- Do not perform repeated speculative repair attempts. After one synchronization,
  syntax, or escaping failure, stop that command shape, inspect the exact scoped
  diff and established utility contract, correct the deterministic workflow,
  and validate it in dry-run mode before another mutation.
- Never restore a dirty managed preset from `HEAD` merely because generated
  syntax is damaged. First establish whether the file contains other legitimate
  changes and use its readable authority plus the tracked manager for recovery.
  Ask before discarding any change whose ownership or recoverability is
  uncertain.

### Session continuity

- Treat an automatic context summary as a non-authoritative index. Let the
  compact-scoped `resume-state-lock` hook reload applicable rules and required
  skills, end its turn, and wait for explicit user confirmation. After that
  confirmation, do not repeat the hook's reload; verify scoped status, the
  current user contract, affected files, and the intended deterministic tooling
  from direct evidence before any mutation.
- Do not claim recovered state from a preserved session unless the preservation
  result identifies the intended source. Read only its generated delta, and use
  historical messages to recover execution constraints and known failures, not
  as authority over a newer user request.
- A brief user correction is steering to apply while continuing. Do not stop,
  produce another plan, repeat what the user said, or substitute discussion for
  the requested repository work.
