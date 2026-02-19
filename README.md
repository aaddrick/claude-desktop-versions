# claude-desktop-versions

A tool for comparing Claude Desktop releases by diffing extracted and beautified AppImage contents. Automatically identifies functional code changes between versions while filtering out build noise like hash rotations, Sentry IDs, and minified identifier renames.

## How It Works

1. **Downloads** AppImages for two releases from [claude-desktop-debian](https://github.com/aaddrick/claude-desktop-debian) via `gh`
2. **Extracts** the AppImage and unpacks `app.asar` contents
3. **Beautifies** minified JavaScript with Prettier for readable diffs
4. **Compares** file trees, matching files across Vite content-hash renames
5. **Diffs** changed files using difftastic (AST-aware) or difflib (line-based fallback)
6. **Filters** build artifact noise (UUIDs, debug IDs, minified name shuffles)
7. **Generates** structured Markdown and JSON reports
8. **Analyzes** changes with Claude CLI for per-file and overall summaries (optional)

## Prerequisites

- Python 3.8+
- [GitHub CLI](https://cli.github.com/) (`gh`) — authenticated with access to the release repo
- Node.js / npm — for `npx prettier` and `npx asar`
- [difftastic](https://difftastic.wilfred.hughes.name/) (`difft`) — optional, enables AST-aware structural diffs
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude`) — optional, enables AI-powered change analysis

## Usage

```bash
# Compare the two most recent releases with different Claude versions (auto-detect)
python scripts/compare-releases.py

# Compare specific release tags
python scripts/compare-releases.py --old v1.3.10+claude1.1.3500 --new v1.3.11+claude1.1.3541

# Skip Claude analysis, produce only raw diff reports
python scripts/compare-releases.py --no-analyze

# Keep extracted files for manual inspection
python scripts/compare-releases.py --keep

# Use a different model for the Claude summary step
python scripts/compare-releases.py --model sonnet
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--old TAG` | auto-detect | Old release tag to compare |
| `--new TAG` | auto-detect | New release tag to compare |
| `--workdir DIR` | `./compare-work` | Working directory for downloads, extractions, and reports |
| `--keep` | off | Preserve extracted/beautified files after the run |
| `--no-analyze` | off | Skip the Claude-powered analysis step |
| `--model MODEL` | `opus` | Model to use for the summary generation step |

## Output

Reports are written to the working directory (`compare-work/` by default):

- **`report.md`** — Human-readable Markdown with file-level diffs, new/removed strings, and code hunks
- **`report.json`** — Machine-readable JSON with the same data
- **`analysis/`** — Per-file Claude analyses and an overall summary (when Claude CLI is available)

## How Noise Filtering Works

Minified JS produces enormous diffs dominated by non-functional changes. The script uses a multi-layer filtering strategy:

- **Vite hash matching** — Files renamed due to content hash changes (e.g. `main-DW5TxSpY.js` → `main-Xk9mP2Qa.js`) are paired and diffed as modifications
- **Build artifact patterns** — UUIDs, hex hashes, Sentry DSNs, debug IDs, and version strings are recognized and excluded
- **Minified rename detection** — Hunks where every changed line differs only by short identifier names (minifier output) are filtered out
- **String extraction** — Stable tokens (string literals, URLs, imports, property names) are compared independently of code structure
- **DFT limit fallback** — When difftastic exceeds its graph/byte limits on large files, the script falls back to difflib automatically
