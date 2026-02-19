# Handover: claude-desktop-versions

Last updated: 2026-02-18

## What This Project Does

Compares releases of the Claude Desktop Electron app as repackaged by
[aaddrick/claude-desktop-debian](https://github.com/aaddrick/claude-desktop-debian).
The main script `scripts/compare-releases.py` downloads two AppImage releases,
extracts them, beautifies the JS with prettier, diffs them using difftastic
(or difflib as fallback), and then drives Claude CLI (`claude -p`) to analyze
the diffs and produce a human-readable summary.

## Current State (as of 2026-02-18)

The script works end-to-end. The most recent run compared
`v1.3.11+claude1.1.3363` vs `v1.3.11+claude1.1.3541`. Output lives in
`compare-work/summary.md`. That summary correctly identified four real changes:

1. DXT auto-updates toggle bug fix — missing IPC argument in `mainWindow.js`
2. Linux/KWin window geometry fix — `frame-fix-wrapper.js` monkey-patches
   `getContentBounds()` to bypass Chromium's stale layout cache
3. Log path co-location with custom data directory — `index.pre.js` now
   redirects `app.setPath("logs", ...)` when `CLAUDE_USER_DATA_DIR` is set
4. Agent SDK bump (0.2.41 → 0.2.45) plus a new "future" experimental SDK alias

The pipeline reduced 157 candidate hunks down to 96 for analysis and then
filtered to 14 that were meaningfully different. The Claude summary was
high quality.

## File Structure

```
scripts/compare-releases.py   — main script, 1266 lines
compare-work/                 — working directory (gitignored)
  old/                        — extracted old release (beautified JS)
  new/                        — extracted new release (beautified JS)
  old_work/                   — AppImage + squashfs for old release
  new_work/                   — AppImage + squashfs for new release
  report.md                   — raw diff report (markdown)
  report.json                 — machine-readable diff report
  analysis-progress.json      — per-hunk analysis tracking (resume support)
  filter_results.txt          — debug log of noise filtering decisions
  summary.md                  — Claude-generated summary (final output)
```

There is no `.gitignore` yet. The `compare-work/` directory should be
gitignored — it contains large extracted AppImage contents and downloaded
binaries. Add one before committing anything.

## Pipeline Steps (in order)

1. Resolve release tags (auto-detects latest two with different Claude versions)
2. Download AppImages via `gh release download`
3. Extract AppImages (`./AppImage --appimage-extract` into squashfs)
4. Extract `app.asar` via `npx asar extract`
5. Beautify JS via `npx prettier`
6. Compare file trees, match Vite hash-renamed files with `VITE_HASH_RE`
7. Generate `report.md` and `report.json`
8. Claude-powered analysis — per-hunk with Sonnet, summary with Opus

## CLI Flags

```
--old TAG      Old release tag (default: auto-detect)
--new TAG      New release tag (default: auto-detect)
--keep         Preserve extracted files after run
--no-analyze   Skip Claude analysis, produce only raw reports
--model MODEL  Override summary model (default: opus)
--workdir DIR  Working directory (default: ./compare-work)
```

## Noise Filtering Pipeline

This was the hard part. Minified JS diffs produce enormous amounts of
build artifact noise. The pipeline filters it in layers:

**1. Binary skip** — `.node` files and anything starting with "Binary" in
the diff header are skipped entirely.

**2. DFT limit skip** — difftastic falls back to text mode for files that
exceed `DFT_BYTE_LIMIT` or `DFT_GRAPH_LIMIT`. Those hunks contain the
`"exceeded DFT_"` string in their header and get skipped. The file is still
kept if string-level changes look interesting.

**3. File-level string noise filter** — `are_string_changes_noise_only()`
at line 745. If every new/removed string in a file matches
`STRING_NOISE_PATTERNS` (line 716), the whole file is skipped. Patterns cover
UUIDs, hex hashes, version strings, ISO timestamps, Sentry IDs, IPC namespace
UUIDs (`$eipc_message$_`), embedded build info JSON blobs, Vite hashed import
paths.

**4. Minified variable rename detection** — `_normalize_minified_names()` at
line 734. Replaces standalone 1-2 letter identifiers with `_`, then compares
the sorted normalized lists. If they match in count and content, all remaining
"changes" are just the minifier picking different short variable names.
`are_string_changes_noise_only()` calls this as a second pass.

**5. Per-hunk noise filter** — `is_build_noise_only()` at line 610. Only
works for difflib output (unified diff with `+/-` prefixes). For difftastic
output, all non-empty hunks pass through because the inline format doesn't
distinguish changed lines from context.

## Claude CLI Integration Patterns

These came from studying the flyspacea orchestrator. All four are in the script:

- `_claude_env()` (line 786) — strips `CLAUDECODE` env var so nested
  invocations work when running inside a Claude Code session
- `_run_claude_cli()` (line 815) — rate limit detection + exponential
  backoff retry (3 attempts, 60s base wait)
- `_extract_structured_output()` (line 844) — handles both `{"result": ...}`
  and `{"structured_output": ...}` response wrappers
- Per-hunk analysis uses `--output-format json --json-schema HUNK_ANALYSIS_SCHEMA`
- Both analysis calls use `--dangerously-skip-permissions`

Progress tracking via `analysis-progress.json` supports resume on failure —
if the script is killed mid-analysis, re-running it picks up from the last
completed hunk.

## The Open Problem: index.js

`index.js` is the main application bundle — 6.6 MiB, 225K beautified lines.
difftastic can't handle it (exceeds `DFT_BYTE_LIMIT`), so all 49 of its hunks
get the DFT-limit skip treatment. The file has 10,159 new strings and 9,974
removed strings, so there ARE real changes in there. They just aren't reaching
the analysis step.

`mainView.js` has the same problem — all 12 hunks hit `DFT_GRAPH_LIMIT`.

Three approaches worth considering:

**Option A — difflib fallback for DFT-limit files.** When a file's difftastic
output shows only DFT-limit hunks, re-diff it with Python's `difflib` instead.
difflib has no size limit. The result is unified diff (with `+/-` prefixes),
which the existing `is_build_noise_only()` filter already handles correctly.
This is probably the right first step — it's a contained change to
`analyze_changes()` and reuses all the existing filtering.

**Option B — string-diff-only analysis.** Skip hunks entirely for these files
and just send the string changes (new vs removed) to Claude. Strings contain
error messages, API endpoints, feature flags, and other semantically rich
content. The per-hunk deobfuscation won't work, but the summary model can
still make useful inferences.

**Option C — humanify deobfuscation.** Run
[humanify](https://github.com/jehna/humanify) on both old and new `index.js`
before diffing. This gives Claude actual readable names to work with. The
output would be much more accurate. The tradeoff is complexity — humanify
requires its own setup, probably adds 5-10 minutes to the run, and you'd need
to decide whether to deobfuscate before or after beautifying.

Option A is the lowest-effort path. Option C is the gold standard.

## Dependencies

- `gh` — GitHub CLI (release download, asset listing)
- `npx` — runs `prettier` and `asar` without global install
- `difft` — difftastic, optional but recommended for AST-aware diffs
- `claude` — Claude CLI for the analysis step

## Known Quirks

- `frame-fix-wrapper.js` is the debian repackager's own patch, not Anthropic's
  code. The summary correctly identified it as such. Future runs will show
  changes to it whenever the patch is updated.
- The `filter_results.txt` file in `compare-work/` is a useful debug artifact
  showing exactly which files were kept vs skipped and why. It's generated
  during the analysis step.
- Vite hash renaming means the same logical file may appear under different
  names across releases. The `VITE_HASH_RE` pattern strips these hashes for
  matching purposes. The match is done in `compare_file_trees()`.
- The script is designed to be run from any directory — it resolves `workdir`
  to an absolute path immediately.
