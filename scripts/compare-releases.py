#!/usr/bin/env python3
"""Compare two releases of claude-desktop-debian to identify functional changes.

Extracts AppImages, beautifies JS, and produces structured diff reports
suitable for feeding to Claude for comprehension.

Dependencies: gh CLI, npx (prettier, asar)
Optional: difftastic (difft) for structural AST-aware diffs
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import urllib.request
from pathlib import Path


REPO = "aaddrick/claude-desktop-debian"


class TokenTracker:
    """Thread-safe tracker for Claude API token usage per model."""

    def __init__(self):
        self._lock = threading.Lock()
        self._usage = {}  # model_id -> {inputTokens, outputTokens, cacheRead, cacheCreation, costUSD, calls}

    def record(self, response_json):
        """Extract and record token usage from a Claude CLI JSON response."""
        model_usage = response_json.get("modelUsage", {})
        if not model_usage:
            return
        with self._lock:
            for model_id, usage in model_usage.items():
                if model_id not in self._usage:
                    self._usage[model_id] = {
                        "inputTokens": 0,
                        "outputTokens": 0,
                        "cacheReadInputTokens": 0,
                        "cacheCreationInputTokens": 0,
                        "costUSD": 0.0,
                        "calls": 0,
                    }
                entry = self._usage[model_id]
                entry["inputTokens"] += usage.get("inputTokens", 0)
                entry["outputTokens"] += usage.get("outputTokens", 0)
                entry["cacheReadInputTokens"] += usage.get("cacheReadInputTokens", 0)
                entry["cacheCreationInputTokens"] += usage.get("cacheCreationInputTokens", 0)
                entry["costUSD"] += usage.get("costUSD", 0.0)
                entry["calls"] += 1

    def summary(self):
        """Return a formatted summary string of token usage per model."""
        with self._lock:
            if not self._usage:
                return "  No token usage recorded."
            lines = []
            total_cost = 0.0
            for model_id in sorted(self._usage):
                u = self._usage[model_id]
                total_cost += u["costUSD"]
                lines.append(
                    f"  {model_id}: "
                    f"{u['calls']} calls, "
                    f"{u['inputTokens']:,} input + "
                    f"{u['cacheReadInputTokens']:,} cache-read + "
                    f"{u['cacheCreationInputTokens']:,} cache-write + "
                    f"{u['outputTokens']:,} output tokens "
                    f"(${u['costUSD']:.4f})"
                )
            lines.append(f"  Total cost: ${total_cost:.4f}")
            return "\n".join(lines)

    def summary_markdown(self):
        """Return a markdown-formatted summary for inclusion in reports."""
        with self._lock:
            if not self._usage:
                return ""
            lines = ["\n---\n", "### Analysis Cost"]
            total_cost = 0.0
            for model_id in sorted(self._usage):
                u = self._usage[model_id]
                total_cost += u["costUSD"]
                lines.append(
                    f"- **{model_id}**: {u['calls']} calls — "
                    f"{u['inputTokens']:,} input, "
                    f"{u['cacheReadInputTokens']:,} cache-read, "
                    f"{u['cacheCreationInputTokens']:,} cache-write, "
                    f"{u['outputTokens']:,} output "
                    f"(${u['costUSD']:.4f})"
                )
            lines.append(f"\n**Total cost: ${total_cost:.4f}**")
            return "\n".join(lines)


# Global token tracker instance
token_tracker = TokenTracker()

# Regex to strip Vite content hashes from filenames
# Matches patterns like "main-DW5TxSpY.js" -> "main.js"
VITE_HASH_RE = re.compile(r"-[A-Za-z0-9_-]{6,12}(\.\w+)$")


def has_difftastic():
    """Check if difftastic (difft) is available."""
    try:
        subprocess.run(["difft", "--version"], capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False


def has_claude_cli():
    """Check if the claude CLI is available."""
    try:
        subprocess.run(["claude", "--version"], capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False


HAS_DIFFT = None  # lazy-initialized
HAS_CLAUDE = None  # lazy-initialized


def fetch_voice_profile(url):
    """Fetch a voice profile from a URL. Returns the content as a string, or None on failure."""
    try:
        print(f"  Fetching voice profile from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "compare-releases/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        print(f"  Voice profile fetched ({len(content)} chars)")
        return content
    except Exception as e:
        print(f"  Warning: Failed to fetch voice profile: {e}")
        return None


def get_has_difft():
    global HAS_DIFFT
    if HAS_DIFFT is None:
        HAS_DIFFT = has_difftastic()
    return HAS_DIFFT


def get_has_claude():
    global HAS_CLAUDE
    if HAS_CLAUDE is None:
        HAS_CLAUDE = has_claude_cli()
    return HAS_CLAUDE


def strip_vite_hash(filename):
    """Strip Vite content hash from a filename.

    'main-DW5TxSpY.js' -> 'main.js'
    'AboutWindow-Bo92QBBQ.js' -> 'AboutWindow.js'
    'index.js' -> 'index.js' (unchanged)
    """
    return VITE_HASH_RE.sub(r"\1", filename)


def run(cmd, **kwargs):
    """Run a command, return stdout. Raise on failure."""
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        stderr = result.stderr if result.stderr else ""
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{stderr}")
    return result.stdout


def parse_tag(tag):
    """Parse tag like 'v1.3.11+claude1.1.3541' into (repo_ver, claude_ver)."""
    m = re.match(r"v([^+]+)\+claude(.+)", tag)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def resolve_releases(old_tag, new_tag):
    """Find the two most recent releases with different Claude versions.

    If --old/--new are provided, use those directly.
    Otherwise, scan the release list for the latest two distinct claude versions.
    """
    if old_tag and new_tag:
        return old_tag, new_tag

    output = run(["gh", "release", "list", "--limit", "50", "-R", REPO])
    releases = []
    for line in output.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            tag = parts[2].strip()
            repo_ver, claude_ver = parse_tag(tag)
            if claude_ver:
                releases.append((tag, repo_ver, claude_ver))

    if len(releases) < 2:
        print("Error: Need at least 2 releases to compare.", file=sys.stderr)
        sys.exit(1)

    if old_tag:
        # Find the old tag in releases, use latest as new
        return old_tag, releases[0][0]
    if new_tag:
        # Find next different claude version as old
        _, _, new_cv = parse_tag(new_tag)
        for tag, _, cv in releases:
            if cv != new_cv:
                return tag, new_tag
        print("Error: Cannot find an older release with a different Claude version.", file=sys.stderr)
        sys.exit(1)

    # Auto-detect: latest two with different claude versions
    new = releases[0]
    for tag, rv, cv in releases[1:]:
        if cv != new[2]:
            return tag, new[0]

    print("Error: All recent releases have the same Claude version.", file=sys.stderr)
    sys.exit(1)


def download_appimage(tag, dest_dir):
    """Download the amd64 AppImage for a given release tag."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # List assets to find the AppImage
    output = run(["gh", "release", "view", tag, "--json", "assets", "-R", REPO])
    assets = json.loads(output).get("assets", [])
    appimage_asset = None
    for a in assets:
        name = a.get("name", "")
        if name.endswith(".AppImage") and "amd64" in name:
            appimage_asset = name
            break

    if not appimage_asset:
        # Try x86_64 pattern
        for a in assets:
            name = a.get("name", "")
            if name.endswith(".AppImage"):
                appimage_asset = name
                break

    if not appimage_asset:
        raise RuntimeError(f"No AppImage found in release {tag}")

    appimage_path = dest_dir / appimage_asset
    if not appimage_path.exists():
        print(f"  Downloading {appimage_asset}...")
        run(["gh", "release", "download", tag,
             "--pattern", appimage_asset,
             "--dir", str(dest_dir),
             "-R", REPO])
    else:
        print(f"  {appimage_asset} already downloaded.")

    return appimage_path


def extract_appimage(appimage_path, dest_dir):
    """Extract an AppImage to dest_dir using --appimage-extract."""
    dest_dir = Path(dest_dir)
    appimage_path = Path(appimage_path).resolve()

    # --appimage-extract always creates squashfs-root in cwd
    os.chmod(appimage_path, 0o755)
    squashfs_dir = dest_dir / "squashfs-root"
    if squashfs_dir.exists():
        print(f"  squashfs-root already exists, skipping extraction.")
        return squashfs_dir

    print(f"  Extracting AppImage...")
    run([str(appimage_path), "--appimage-extract"], cwd=str(dest_dir),
        capture_output=True, text=True)
    return squashfs_dir


def extract_asar(squashfs_dir, dest_dir):
    """Extract app.asar from the squashfs root."""
    squashfs_dir = Path(squashfs_dir)
    dest_dir = Path(dest_dir)

    # Search for app.asar - location varies by packaging
    candidates = [
        squashfs_dir / "resources" / "app.asar",
        squashfs_dir / "usr" / "lib" / "node_modules" / "electron" / "dist" / "resources" / "app.asar",
    ]
    asar_path = None
    for c in candidates:
        if c.exists():
            asar_path = c
            break
    if asar_path is None:
        # Fallback: search for it
        found = list(squashfs_dir.rglob("app.asar"))
        found = [f for f in found if f.name == "app.asar" and "default_app" not in str(f)]
        if found:
            asar_path = found[0]
        else:
            raise RuntimeError(f"app.asar not found in {squashfs_dir}")

    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"  asar already extracted, skipping.")
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting app.asar...")
    run(["npx", "asar", "extract", str(asar_path), str(dest_dir)])
    return dest_dir


def beautify_js(source_dir):
    """Run prettier on all JS files in the directory."""
    source_dir = Path(source_dir)
    js_files = list(source_dir.rglob("*.js"))
    if not js_files:
        print(f"  No JS files found to beautify.")
        return

    # Check if already beautified by sampling the largest JS file (most likely minified)
    sample = max(js_files, key=lambda f: f.stat().st_size)
    with open(sample, "r", errors="replace") as f:
        first_lines = f.read(4000)
    lines = first_lines.splitlines()
    if lines:
        max_line = max(len(l) for l in lines[:20])
        if max_line < 200:
            print(f"  JS appears already beautified (checked {sample.name}), skipping.")
            return

    print(f"  Beautifying {len(js_files)} JS files with prettier...")
    # Run prettier in batches to avoid arg-list-too-long
    batch_size = 50
    for i in range(0, len(js_files), batch_size):
        batch = js_files[i:i + batch_size]
        try:
            run(["npx", "prettier", "--write", "--parser", "babel"] +
                [str(f) for f in batch])
        except RuntimeError:
            # Some files may fail to parse; try individually
            for f in batch:
                try:
                    run(["npx", "prettier", "--write", "--parser", "babel", str(f)])
                except RuntimeError:
                    pass  # Skip files that can't be parsed


def file_hash(path):
    """SHA256 hash of a file's content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def match_renamed_files(added, removed):
    """Match files that were renamed due to Vite content hash changes.

    Returns: (matched_pairs, true_added, true_removed)
    where matched_pairs is list of (old_rel, new_rel) tuples.
    """
    # Build lookup: (parent_dir, dehashed_name) -> path
    removed_by_key = {}
    for rel in removed:
        key = (str(rel.parent), strip_vite_hash(rel.name))
        removed_by_key[key] = rel

    matched = []
    true_added = set()
    matched_old = set()

    for rel in added:
        key = (str(rel.parent), strip_vite_hash(rel.name))
        if key in removed_by_key:
            old_rel = removed_by_key[key]
            matched.append((old_rel, rel))
            matched_old.add(old_rel)
        else:
            true_added.add(rel)

    true_removed = {r for r in removed if r not in matched_old}
    return matched, true_added, true_removed


def compare_file_trees(old_dir, new_dir):
    """Compare two directory trees, matching Vite hash-renamed files.

    Returns (added, removed, modified, unchanged, renamed) where renamed
    is a list of (old_rel, new_rel) tuples treated as modifications.
    """
    old_dir = Path(old_dir)
    new_dir = Path(new_dir)

    old_files = {p.relative_to(old_dir) for p in old_dir.rglob("*") if p.is_file()}
    new_files = {p.relative_to(new_dir) for p in new_dir.rglob("*") if p.is_file()}

    added = new_files - old_files
    removed = old_files - new_files
    common = old_files & new_files

    # Match hash-renamed files
    renamed, added, removed = match_renamed_files(added, removed)

    modified = set()
    unchanged = set()
    for rel in common:
        if file_hash(old_dir / rel) != file_hash(new_dir / rel):
            modified.add(rel)
        else:
            unchanged.add(rel)

    return added, removed, modified, unchanged, renamed


def extract_strings(text):
    """Extract stable tokens from JS source: string literals, URLs, error messages, imports."""
    strings = set()

    # Double-quoted strings
    for m in re.finditer(r'"((?:[^"\\]|\\.){3,})"', text):
        strings.add(m.group(1))

    # Single-quoted strings
    for m in re.finditer(r"'((?:[^'\\]|\\.){3,})'", text):
        strings.add(m.group(1))

    # Template literal fragments (between ${ })
    for m in re.finditer(r'`((?:[^`\\]|\\.)*?)`', text):
        content = m.group(1)
        # Split on template expressions to get literal parts
        parts = re.split(r'\$\{[^}]*\}', content)
        for p in parts:
            p = p.strip()
            if len(p) >= 3:
                strings.add(p)

    # URLs
    for m in re.finditer(r'https?://[^\s"\'`<>)\]]+', text):
        strings.add(m.group(0))

    # require() and import paths
    for m in re.finditer(r'require\(["\']([^"\']+)["\']\)', text):
        strings.add(f"require:{m.group(1)}")
    for m in re.finditer(r'from\s+["\']([^"\']+)["\']', text):
        strings.add(f"import:{m.group(1)}")

    # Property names in object literals (key: value patterns)
    for m in re.finditer(r'(?:^|\n)\s{2,}(\w{3,}):', text):
        strings.add(f"prop:{m.group(1)}")

    return strings


def extract_changed_hunks_difflib(old_text, new_text, context_lines=5):
    """Generate unified diff hunks using Python's difflib."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile="old", tofile="new",
        n=context_lines
    ))
    return "".join(diff)


def extract_changed_hunks_difft(old_path, new_path):
    """Generate structural diff using difftastic."""
    result = subprocess.run(
        ["difft", "--display", "inline", "--color", "never",
         "--context", "5", str(old_path), str(new_path)],
        capture_output=True, text=True
    )
    # difft returns 0 for no changes, 1 for changes — both are success
    return result.stdout


def extract_changed_hunks(old_text, new_text, context_lines=5,
                          old_path=None, new_path=None):
    """Generate diff hunks. Uses difftastic if available, falls back to difflib."""
    if get_has_difft() and old_path and new_path:
        return extract_changed_hunks_difft(old_path, new_path)
    return extract_changed_hunks_difflib(old_text, new_text, context_lines)


def _is_all_dft_limit(hunks_text):
    """Check if all difftastic hunks hit DFT_BYTE_LIMIT or DFT_GRAPH_LIMIT."""
    if not hunks_text or not hunks_text.strip():
        return False
    # Match difftastic hunk headers (filename --- N/M --- Language lines)
    pattern = re.compile(
        r'^(\S.+\s---\s(?:\d+/\d+\s---\s)?[A-Z]\S*(?:\s\(.+\))?)$', re.MULTILINE
    )
    headers = pattern.findall(hunks_text)
    if not headers:
        return False
    return all("exceeded DFT_" in h for h in headers)


def _analyze_file_pair(old_path, new_path, rel_label, suffix):
    """Analyze a single old/new file pair for changes."""
    report = {
        "path": str(rel_label),
        "status": "modified",
        "new_strings": [],
        "removed_strings": [],
        "hunks": "",
    }

    if suffix == ".js":
        try:
            old_text = old_path.read_text(errors="replace")
            new_text = new_path.read_text(errors="replace")

            old_strings = extract_strings(old_text)
            new_strings = extract_strings(new_text)
            report["new_strings"] = sorted(new_strings - old_strings)
            report["removed_strings"] = sorted(old_strings - new_strings)

            hunks = extract_changed_hunks(old_text, new_text,
                                          old_path=old_path, new_path=new_path)
            # Detect DFT-limit-only difftastic output and fall back to difflib
            used_difflib_fallback = False
            if get_has_difft() and old_path and new_path and _is_all_dft_limit(hunks):
                print(f"    {rel_label}: difftastic exceeded limits, falling back to difflib")
                hunks = extract_changed_hunks_difflib(old_text, new_text, context_lines=5)
                used_difflib_fallback = True
            report["diff_engine"] = "difflib" if (used_difflib_fallback or not get_has_difft()) else "difftastic"
            report["hunks"] = hunks
        except Exception as e:
            report["hunks"] = f"Error analyzing: {e}"
    else:
        try:
            old_text = old_path.read_text(errors="replace")
            new_text = new_path.read_text(errors="replace")
            hunks = extract_changed_hunks(old_text, new_text, context_lines=3,
                                          old_path=old_path, new_path=new_path)
            # Detect DFT-limit-only difftastic output and fall back to difflib
            used_difflib_fallback = False
            if get_has_difft() and old_path and new_path and _is_all_dft_limit(hunks):
                print(f"    {rel_label}: difftastic exceeded limits, falling back to difflib")
                hunks = extract_changed_hunks_difflib(old_text, new_text, context_lines=3)
                used_difflib_fallback = True
            report["diff_engine"] = "difflib" if (used_difflib_fallback or not get_has_difft()) else "difftastic"
            report["hunks"] = hunks
        except Exception:
            pass

    return report


def analyze_changes(old_dir, new_dir, added, removed, modified, renamed):
    """Analyze modified and renamed files for string changes and code hunks."""
    old_dir = Path(old_dir)
    new_dir = Path(new_dir)
    file_reports = []

    for rel in sorted(added):
        file_reports.append({
            "path": str(rel),
            "status": "added",
            "strings": [],
            "hunks": "",
        })

    for rel in sorted(removed):
        file_reports.append({
            "path": str(rel),
            "status": "removed",
            "strings": [],
            "hunks": "",
        })

    # Same-name modified files
    for rel in sorted(modified):
        report = _analyze_file_pair(
            old_dir / rel, new_dir / rel, rel, rel.suffix)
        file_reports.append(report)

    # Renamed (hash-busted) files — treat as modified
    for old_rel, new_rel in sorted(renamed, key=lambda p: str(p[1])):
        report = _analyze_file_pair(
            old_dir / old_rel, new_dir / new_rel,
            f"{new_rel} (was {old_rel.name})", new_rel.suffix)
        report["status"] = "renamed"
        report["old_path"] = str(old_rel)
        file_reports.append(report)

    return file_reports


def _render_file_report_md(fr, lines):
    """Render a single file report entry into markdown lines."""
    new_strings = fr.get("new_strings", [])
    removed_strings = fr.get("removed_strings", [])

    if new_strings:
        lines.append("**New strings/tokens:**")
        for s in new_strings[:50]:
            lines.append(f"- `{s}`")
        if len(new_strings) > 50:
            lines.append(f"- ... and {len(new_strings) - 50} more")
        lines.append("")

    if removed_strings:
        lines.append("**Removed strings/tokens:**")
        for s in removed_strings[:50]:
            lines.append(f"- `{s}`")
        if len(removed_strings) > 50:
            lines.append(f"- ... and {len(removed_strings) - 50} more")
        lines.append("")

    hunks = fr.get("hunks", "")
    if hunks:
        display_hunks = hunks
        if len(hunks) > 50000:
            display_hunks = hunks[:50000] + "\n... [truncated, diff too large] ...\n"
        lines.append("**Changed code hunks:**")
        lines.append("```diff")
        lines.append(display_hunks.rstrip())
        lines.append("```")
        lines.append("")


def generate_report_md(old_tag, new_tag, added, removed, modified, unchanged, renamed, file_reports):
    """Generate a human-readable markdown report."""
    old_rv, old_cv = parse_tag(old_tag)
    new_rv, new_cv = parse_tag(new_tag)

    diff_engine = "difftastic (AST-aware)" if get_has_difft() else "difflib (line-based)"

    lines = [
        f"# Release Comparison Report",
        f"",
        f"## Summary",
        f"- **Old release**: `{old_tag}` (repo: {old_rv}, claude: {old_cv})",
        f"- **New release**: `{new_tag}` (repo: {new_rv}, claude: {new_cv})",
        f"- **Files added**: {len(added)}",
        f"- **Files removed**: {len(removed)}",
        f"- **Files modified**: {len(modified)}",
        f"- **Files renamed (hash change)**: {len(renamed)}",
        f"- **Files unchanged**: {len(unchanged)}",
        f"- **Diff engine**: {diff_engine}",
        f"",
    ]

    if added:
        lines.append("## Added Files")
        for p in sorted(added):
            lines.append(f"- `{p}`")
        lines.append("")

    if removed:
        lines.append("## Removed Files")
        for p in sorted(removed):
            lines.append(f"- `{p}`")
        lines.append("")

    changed_reports = [fr for fr in file_reports if fr["status"] in ("modified", "renamed")]
    if changed_reports:
        lines.append("## Modified Files")
        lines.append("")

        for fr in changed_reports:
            if fr["status"] == "renamed":
                lines.append(f"### `{fr['path']}`")
            else:
                lines.append(f"### `{fr['path']}`")
            lines.append("")
            _render_file_report_md(fr, lines)

    return "\n".join(lines)


def generate_report_json(old_tag, new_tag, added, removed, modified, unchanged, renamed, file_reports):
    """Generate a machine-readable JSON report."""
    old_rv, old_cv = parse_tag(old_tag)
    new_rv, new_cv = parse_tag(new_tag)

    return {
        "summary": {
            "old_tag": old_tag,
            "new_tag": new_tag,
            "old_repo_version": old_rv,
            "old_claude_version": old_cv,
            "new_repo_version": new_rv,
            "new_claude_version": new_cv,
            "files_added": len(added),
            "files_removed": len(removed),
            "files_modified": len(modified),
            "files_renamed": len(renamed),
            "files_unchanged": len(unchanged),
            "diff_engine": "difftastic" if get_has_difft() else "difflib",
        },
        "added_files": sorted(str(p) for p in added),
        "removed_files": sorted(str(p) for p in removed),
        "modified_files": sorted(str(p) for p in modified),
        "renamed_files": [{"old": str(o), "new": str(n)} for o, n in sorted(renamed, key=lambda p: str(p[1]))],
        "file_reports": file_reports,
    }


# Regex patterns for build artifact noise
BUILD_NOISE_PATTERNS = [
    re.compile(r'^[-+].*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.MULTILINE),  # UUIDs
    re.compile(r'^[-+].*"?sentryDsn"?\s*[:=]', re.MULTILINE),
    re.compile(r'^[-+].*"?debugId"?\s*[:=]', re.MULTILINE),
    re.compile(r'^[-+].*"?debug_id"?\s*[:=]', re.MULTILINE),
]

# Only lines starting with + or - (actual changes) matter for noise detection
DIFF_CHANGE_LINE_RE = re.compile(r'^[-+](?![-+])', re.MULTILINE)


def is_build_noise_only(hunk_text, use_difft=False):
    """Check if a hunk only contains build artifact changes (hashes, UUIDs, sentry IDs).

    Returns True if the hunk has no substantive code changes.

    For difftastic output (inline format without +/- prefixes), we cannot
    reliably distinguish changed vs context lines, so we never filter —
    all difftastic hunks are considered substantive. The noise filtering
    happens at the file level via string_changes instead.

    For difflib output (unified diff with +/- prefixes), we only check
    actual changed lines.
    """
    if use_difft:
        # Difftastic uses an inline format without +/- prefixes.
        # We can't distinguish changed lines from context, so pass all hunks through.
        # Empty hunks are still filtered.
        return not hunk_text.strip()

    # Difflib unified diff format — check +/- lines only
    noise_patterns = [
        re.compile(r'^["\']?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}["\']?[,;]?$'),  # UUID
        re.compile(r'^["\']?[0-9a-f]{32,64}["\']?[,;]?$'),  # hex hashes
        re.compile(r'^.*sentryDsn.*$', re.IGNORECASE),
        re.compile(r'^.*debugId.*$', re.IGNORECASE),
        re.compile(r'^.*debug_id.*$', re.IGNORECASE),
        re.compile(r'^["\']?\d+\.\d+\.\d+["\']?[,;]?$'),  # version strings like "1.2.3"
        re.compile(r'^["\']?[0-9a-f]{7,40}["\']?[,;]?$'),  # commit hashes
    ]

    # Separate added and removed lines
    adds = []
    removes = []
    for line in hunk_text.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("+"):
            adds.append(line[1:].strip())
        elif line.startswith("-"):
            removes.append(line[1:].strip())

    changes = adds + removes
    if not changes or all(not c for c in changes):
        return True

    # Check 1: all changed lines match explicit noise patterns
    all_pattern_noise = True
    for change in changes:
        if not change:
            continue
        if not any(p.match(change) for p in noise_patterns):
            all_pattern_noise = False
            break
    if all_pattern_noise:
        return True

    # Check 2: minified identifier renames.
    # If every +/- line, after normalizing short identifiers, either
    # matches a noise pattern or has a corresponding line on the other side,
    # then the hunk is just minified name shuffling.
    if adds and removes and len(adds) == len(removes):
        is_rename_only = True
        for a, r in zip(adds, removes):
            if not a and not r:
                continue
            # If both match noise patterns individually, that's fine
            if a and any(p.match(a) for p in noise_patterns) and \
               r and any(p.match(r) for p in noise_patterns):
                continue
            # Normalize minified names and compare
            if _normalize_minified_names(a) != _normalize_minified_names(r):
                is_rename_only = False
                break
        if is_rename_only:
            return True

    return False


def split_hunks(hunks_text, use_difft):
    """Split a diff string into individual hunk chunks.

    For difftastic output, splits on 'filename --- N/M --- Language' headers.
    For difflib output, splits on '@@ ... @@' hunk headers.

    Returns list of dicts: [{"header": str, "content": str}, ...]
    """
    if not hunks_text or not hunks_text.strip():
        return []

    results = []

    if use_difft:
        # Difftastic headers come in two forms:
        #   Multi-hunk: "filename --- 1/15 --- JavaScript"
        #   Single-hunk: "filename --- JavaScript"
        #   With notes: "filename --- 1/3022 --- Text (exceeded DFT_BYTE_LIMIT)"
        # The header always ends with " --- Language..." where Language is a word
        # optionally preceded by " --- N/M"
        pattern = re.compile(
            r'^(\S.+\s---\s(?:\d+/\d+\s---\s)?[A-Z]\S*(?:\s\(.+\))?)$', re.MULTILINE
        )
        parts = pattern.split(hunks_text)
        # parts alternates: [pre-text, header1, content1, header2, content2, ...]
        i = 1  # skip any preamble before first header
        while i < len(parts) - 1:
            header = parts[i].strip()
            content = parts[i + 1]
            if content.strip():
                results.append({"header": header, "content": content.strip()})
            i += 2
        # If no headers matched but there's content, treat the whole thing as one hunk
        if not results and hunks_text.strip():
            results.append({"header": "unknown", "content": hunks_text.strip()})
    else:
        # Difflib format: split on @@ ... @@ headers
        pattern = re.compile(r'^(@@\s.*?\s@@.*)$', re.MULTILINE)
        parts = pattern.split(hunks_text)
        # Skip the file header (--- / +++ lines before first @@)
        i = 1
        while i < len(parts) - 1:
            header = parts[i].strip()
            content = parts[i + 1]
            if content.strip():
                results.append({"header": header, "content": header + "\n" + content.strip()})
            i += 2

    return results


# Patterns that match build-artifact string changes (UUIDs, hashes, sentry, IPC namespaces, timestamps, versions)
STRING_NOISE_PATTERNS = [
    re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),  # UUID
    re.compile(r'^[0-9a-f]{7,64}$'),  # hex hashes (commit SHAs, content hashes)
    re.compile(r'^\d+\.\d+\.\d+$'),  # version strings
    re.compile(r'^\d{4}-\d{2}-\d{2}T'),  # ISO timestamps
    re.compile(r'.*sentryDsn.*', re.IGNORECASE),
    re.compile(r'.*debugId.*', re.IGNORECASE),
    re.compile(r'.*debug_id.*', re.IGNORECASE),
    re.compile(r'.*sentry-dbid-[0-9a-f-]+'),  # sentry debug ID strings
    re.compile(r'.*SENTRY_RELEASE.*'),
    re.compile(r'^\$eipc_message\$_[0-9a-f-]+'),  # IPC channel namespace UUIDs
    re.compile(r'^sentry-dbid-'),
    re.compile(r'^\{"commitHash":"[0-9a-f]+"'),  # embedded build info JSON blobs
    re.compile(r'^import:\./\S+-[A-Za-z0-9_-]{6,12}\.\w+$'),  # Vite hashed import paths
    re.compile(r'^\./\S+-[A-Za-z0-9_-]{6,12}\.\w+$'),  # Vite hashed module paths
]


def _normalize_minified_names(s):
    """Normalize minified identifiers to detect renames.

    Replaces short identifiers with '_' so that strings differing only by
    minified variable names compare as equal.

    Handles patterns like:
    - 1-3 letter names: v, ba, iPe, AEe
    - $-prefixed: $2e, $$
    - _-prefixed short names: _O, _2, _sentryDebugIds
    - Capitalized short suffixes: vt, Bt (used in minified class/function names)
    - IPC namespace UUIDs embedded in strings: $eipc_message$_UUID_$_...
    """
    # Step 1: Normalize UUIDs (must run before $ and hex hash normalization)
    s = re.sub(
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        'UUID', s
    )
    # Step 2: Normalize embedded hex hashes (commit SHAs, content hashes)
    s = re.sub(r'["\']?[0-9a-f]{7,64}["\']?', 'HASH', s)
    # Step 3: Replace $-prefixed short identifiers ($2e, $$, $t)
    s = re.sub(r'\$[a-zA-Z0-9]{0,2}\b', '_', s)
    # Step 4: Replace _-prefixed short identifiers (_O, _2, _e)
    s = re.sub(r'\b_[a-zA-Z0-9]{0,2}\b', '_', s)
    # Step 5: Replace minified identifiers — short alphanumeric names
    # Catches: v, ba, iPe, AEe, u4e, f4e, qMe, HMe, X_, ab, cn, etc.
    s = re.sub(r'\b[a-zA-Z]{1,3}[0-9]*[a-zA-Z_]?\b', '_', s)
    return s


def are_string_changes_noise_only(new_strings, removed_strings):
    """Check if all string changes for a file are build artifacts.

    Returns True if every new/removed string matches a noise pattern
    (UUIDs, commit hashes, version strings, sentry IDs, IPC namespace rotations),
    OR if all changes are just minified variable name differences (e.g., 'v' vs 'b').
    """
    all_strings = list(new_strings) + list(removed_strings)
    if not all_strings:
        return True

    # First pass: check explicit noise patterns
    non_noise = []
    for s in all_strings:
        if any(p.match(s) for p in STRING_NOISE_PATTERNS):
            continue
        non_noise.append(s)

    if not non_noise:
        return True

    # Second pass: check if remaining "non-noise" strings are just minified
    # variable renames. Filter out noise-matched strings first, then normalize
    # and compare the rest.
    non_noise_new = [s for s in new_strings if not any(p.match(s) for p in STRING_NOISE_PATTERNS)]
    non_noise_rem = [s for s in removed_strings if not any(p.match(s) for p in STRING_NOISE_PATTERNS)]

    if len(non_noise_new) == len(non_noise_rem) and len(non_noise_new) > 0:
        new_normalized = sorted(_normalize_minified_names(s) for s in non_noise_new)
        removed_normalized = sorted(_normalize_minified_names(s) for s in non_noise_rem)
        if new_normalized == removed_normalized:
            return True

    return False


# Hunk size thresholds for tiered analysis
HUNK_SIZE_THRESHOLD = 40_000  # bytes — triggers Tier 2/3 preprocessing
LONG_LINE_THRESHOLD = 2_000  # chars — indicates minified JS (sub-line diffing vs raw chunking)
MAX_STRING_CONTEXT = 5_000  # max chars of string context in prompts
MAX_STRING_LENGTH = 200  # truncate individual strings in context


def compute_subline_diff(removed_lines, added_lines):
    """Compute a statement-level diff for minified JS with long lines.

    Splits removed/added lines on semicolons into individual statements,
    normalizes minified identifiers, and uses SequenceMatcher to find
    semantically meaningful changes (filtering out pure renames).

    Args:
        removed_lines: List of removed line strings (without - prefix)
        added_lines: List of added line strings (without + prefix)

    Returns:
        (diff_text, stats) where diff_text is a formatted diff string
        and stats is a dict with counts.
    """
    def split_statements(lines):
        """Split lines on semicolons into individual statements."""
        text = "\n".join(lines)
        # Split on ; but keep the ; attached to the preceding statement
        raw = re.split(r'(;)', text)
        statements = []
        for i in range(0, len(raw) - 1, 2):
            stmt = raw[i] + (raw[i + 1] if i + 1 < len(raw) else "")
            stmt = stmt.strip()
            if stmt:
                statements.append(stmt)
        # Handle trailing content after last ;
        if len(raw) % 2 == 1 and raw[-1].strip():
            statements.append(raw[-1].strip())
        return statements

    # If too few statements, return original lines directly
    removed_stmts = split_statements(removed_lines)
    added_stmts = split_statements(added_lines)

    if len(removed_stmts) < 5 and len(added_stmts) < 5:
        diff_lines = []
        for line in removed_lines:
            diff_lines.append(f"- {line}")
        for line in added_lines:
            diff_lines.append(f"+ {line}")
        return "\n".join(diff_lines), {
            "removed_stmts": len(removed_stmts),
            "added_stmts": len(added_stmts),
            "changed_stmts": len(removed_stmts) + len(added_stmts),
            "method": "raw_lines",
        }

    # Normalize for comparison
    norm_removed = [_normalize_minified_names(s) for s in removed_stmts]
    norm_added = [_normalize_minified_names(s) for s in added_stmts]

    matcher = difflib.SequenceMatcher(None, norm_removed, norm_added, autojunk=False)
    diff_lines = []
    changed_count = 0

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        # Add 1 statement of context before and after
        ctx_before_r = max(0, i1 - 1)
        ctx_after_r = min(len(removed_stmts), i2 + 1)
        ctx_before_a = max(0, j1 - 1)
        ctx_after_a = min(len(added_stmts), j2 + 1)

        if op == "replace":
            # Check if this is just a minified rename
            if norm_removed[i1:i2] == norm_added[j1:j2]:
                continue  # Pure rename, skip
            changed_count += (i2 - i1) + (j2 - j1)
            # Context before
            if ctx_before_r < i1:
                diff_lines.append(f"  {removed_stmts[ctx_before_r]}")
            for s in removed_stmts[i1:i2]:
                diff_lines.append(f"- {s}")
            for s in added_stmts[j1:j2]:
                diff_lines.append(f"+ {s}")
            # Context after
            if i2 < ctx_after_r:
                diff_lines.append(f"  {removed_stmts[i2]}")
            diff_lines.append("---")
        elif op == "delete":
            changed_count += i2 - i1
            if ctx_before_r < i1:
                diff_lines.append(f"  {removed_stmts[ctx_before_r]}")
            for s in removed_stmts[i1:i2]:
                diff_lines.append(f"- {s}")
            if i2 < ctx_after_r:
                diff_lines.append(f"  {removed_stmts[i2]}")
            diff_lines.append("---")
        elif op == "insert":
            changed_count += j2 - j1
            if ctx_before_a < j1:
                diff_lines.append(f"  {added_stmts[ctx_before_a]}")
            for s in added_stmts[j1:j2]:
                diff_lines.append(f"+ {s}")
            if j2 < ctx_after_a:
                diff_lines.append(f"  {added_stmts[j2]}")
            diff_lines.append("---")

    diff_text = "\n".join(diff_lines)
    stats = {
        "removed_stmts": len(removed_stmts),
        "added_stmts": len(added_stmts),
        "changed_stmts": changed_count,
        "method": "subline_diff",
    }
    return diff_text, stats


def chunk_subline_diff(diff_text, max_size=HUNK_SIZE_THRESHOLD):
    """Split a subline diff into chunks that fit within max_size.

    Splits on '---' separator lines between change regions, greedily
    accumulating into chunks.

    Args:
        diff_text: The formatted subline diff text
        max_size: Maximum size per chunk in chars

    Returns:
        List of chunk strings.
    """
    if len(diff_text) <= max_size:
        return [diff_text]

    regions = diff_text.split("\n---\n")
    chunks = []
    current = ""

    for region in regions:
        region = region.strip()
        if not region:
            continue
        candidate = (current + "\n---\n" + region).strip() if current else region
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If single region exceeds max_size, include it anyway (will be truncated by Claude)
            current = region

    if current:
        chunks.append(current)

    return chunks if chunks else [diff_text[:max_size]]


def build_string_fallback_content(hunk_content, new_strings, removed_strings):
    """Build a string-change summary as Tier 3 fallback for oversized hunks.

    Extracts strings from the hunk's added/removed lines, computes the
    hunk-local string diff, and filters noise.

    Args:
        hunk_content: The raw diff hunk text
        new_strings: File-level new strings list
        removed_strings: File-level removed strings list

    Returns:
        Formatted string-change summary text.
    """
    # Extract +/- lines from the hunk
    added_text = []
    removed_text = []
    for line in hunk_content.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added_text.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed_text.append(line[1:])

    # Extract strings from hunk-local added/removed content
    hunk_added_strings = extract_strings("\n".join(added_text))
    hunk_removed_strings = extract_strings("\n".join(removed_text))

    # Compute hunk-local new/removed strings
    hunk_new = hunk_added_strings - hunk_removed_strings
    hunk_removed = hunk_removed_strings - hunk_added_strings

    # Filter noise
    def filter_noise(strings):
        return [s for s in sorted(strings)
                if not any(p.match(s) for p in STRING_NOISE_PATTERNS)]

    hunk_new_filtered = filter_noise(hunk_new)
    hunk_removed_filtered = filter_noise(hunk_removed)

    lines = []
    lines.append(f"## Hunk String Changes (extracted from {len(hunk_content)} char hunk)")
    lines.append(f"Total hunk-local: +{len(hunk_new_filtered)} -{len(hunk_removed_filtered)} strings (after noise filter)")
    lines.append("")

    if hunk_new_filtered:
        lines.append("### New strings/tokens in this hunk:")
        for s in hunk_new_filtered[:100]:
            truncated = s[:MAX_STRING_LENGTH] + "..." if len(s) > MAX_STRING_LENGTH else s
            lines.append(f"+ {truncated}")
        if len(hunk_new_filtered) > 100:
            lines.append(f"  ... and {len(hunk_new_filtered) - 100} more")
    lines.append("")

    if hunk_removed_filtered:
        lines.append("### Removed strings/tokens in this hunk:")
        for s in hunk_removed_filtered[:100]:
            truncated = s[:MAX_STRING_LENGTH] + "..." if len(s) > MAX_STRING_LENGTH else s
            lines.append(f"- {truncated}")
        if len(hunk_removed_filtered) > 100:
            lines.append(f"  ... and {len(hunk_removed_filtered) - 100} more")

    return "\n".join(lines)


def chunk_raw_hunk(content, max_size=HUNK_SIZE_THRESHOLD):
    """Split a non-minified large hunk at change-cluster boundaries.

    For unified diffs, groups consecutive change lines (+/-) with their
    surrounding context into clusters, then greedily combines clusters
    into chunks within max_size.

    Args:
        content: The unified diff hunk text
        max_size: Maximum size per chunk in chars

    Returns:
        List of chunk strings.
    """
    if len(content) <= max_size:
        return [content]

    lines = content.splitlines(keepends=True)
    # Find clusters of change lines with context
    clusters = []
    current_cluster = []
    context_before = []

    for line in lines:
        is_change = line.startswith("+") or line.startswith("-")
        is_header = line.startswith("@@") or line.startswith("---") or line.startswith("+++")

        if is_header:
            if current_cluster:
                clusters.append("".join(current_cluster))
                current_cluster = []
            context_before = []
            continue

        if is_change:
            if not current_cluster and context_before:
                # Add up to 3 lines of context before
                current_cluster.extend(context_before[-3:])
            current_cluster.append(line)
        else:
            if current_cluster:
                # Context line after changes — add up to 3, then break cluster
                current_cluster.append(line)
                if len([l for l in current_cluster if not (l.startswith("+") or l.startswith("-"))]) >= 3:
                    clusters.append("".join(current_cluster))
                    current_cluster = []
                    context_before = []
                    continue
            context_before.append(line)

    if current_cluster:
        clusters.append("".join(current_cluster))

    if not clusters:
        # Fallback: just split by size
        return [content[i:i + max_size] for i in range(0, len(content), max_size)]

    # Greedily combine clusters into chunks
    chunks = []
    current = ""
    for cluster in clusters:
        candidate = current + "\n...\n" + cluster if current else cluster
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = cluster if len(cluster) <= max_size else cluster[:max_size]

    if current:
        chunks.append(current)

    return chunks if chunks else [content[:max_size]]


def preprocess_hunk(hunk_dict, string_changes):
    """Preprocess a hunk for analysis, routing to the appropriate tier.

    Tier 1: Hunk < HUNK_SIZE_THRESHOLD → direct analysis (unchanged path)
    Tier 2: Oversized hunk with long lines → sub-line statement diffing
    Tier 3: Build string fallback content for anything still too large

    Args:
        hunk_dict: Dict with 'content', 'header', etc. from split_hunks()
        string_changes: Dict with 'new_strings' and 'removed_strings'

    Returns:
        List of dicts: [{"content", "tier", "original_size", "header", ...}]
    """
    content = hunk_dict["content"]
    header = hunk_dict.get("header", "")
    original_size = len(content)

    # Tier 1: small enough for direct analysis
    if original_size <= HUNK_SIZE_THRESHOLD:
        return [{
            "content": content,
            "tier": "direct",
            "original_size": original_size,
            "header": header,
        }]

    # Check if lines are long (minified JS)
    lines = content.splitlines()
    max_line_len = max((len(l) for l in lines), default=0)

    if max_line_len >= LONG_LINE_THRESHOLD:
        # Tier 2: sub-line statement diffing for minified JS
        removed_lines = []
        added_lines = []
        for line in lines:
            if line.startswith("-") and not line.startswith("---"):
                removed_lines.append(line[1:])
            elif line.startswith("+") and not line.startswith("+++"):
                added_lines.append(line[1:])

        if not removed_lines and not added_lines:
            # No actual changes, just context
            return [{
                "content": content,
                "tier": "direct",
                "original_size": original_size,
                "header": header,
            }]

        diff_text, stats = compute_subline_diff(removed_lines, added_lines)

        if not diff_text.strip():
            # All changes were pure renames
            print(f"      Tier 2: all changes are minified renames ({original_size} chars → noise)")
            return [{
                "content": "",
                "tier": "noise",
                "original_size": original_size,
                "header": header,
                "stats": stats,
            }]

        reduction = (1 - len(diff_text) / original_size) * 100
        print(f"      Tier 2: subline diff {original_size} → {len(diff_text)} chars ({reduction:.0f}% reduction, {stats['changed_stmts']} changed stmts)")

        # Chunk if still too large
        chunks = chunk_subline_diff(diff_text)
        results = []
        for i, chunk in enumerate(chunks):
            results.append({
                "content": chunk,
                "tier": "subline",
                "original_size": original_size,
                "header": header,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "stats": stats,
            })
        return results
    else:
        # Large hunk but not minified — chunk at change-cluster boundaries
        chunks = chunk_raw_hunk(content)
        print(f"      Chunked large hunk: {original_size} chars → {len(chunks)} chunks")
        results = []
        for i, chunk in enumerate(chunks):
            results.append({
                "content": chunk,
                "tier": "direct",
                "original_size": original_size,
                "header": header,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
        return results


# Rate limit retry settings
RATE_LIMIT_MAX_RETRIES = 3
RATE_LIMIT_BASE_WAIT = 60  # seconds


def _claude_env():
    """Build environment for Claude CLI subprocess calls.

    Removes CLAUDECODE env var to allow nested invocations when run
    from within a Claude Code session.
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    return env


def _is_rate_limited(result):
    """Check if a subprocess result indicates a rate limit error."""
    combined = (result.stdout or "") + (result.stderr or "")
    return bool(re.search(r'rate.limit|429|too many requests|quota.exceeded', combined, re.IGNORECASE))


def _extract_wait_time(result):
    """Try to extract a retry-after wait time from Claude CLI output."""
    combined = (result.stdout or "") + (result.stderr or "")
    m = re.search(r'retry.after[^0-9]*(\d+)', combined, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'wait[^0-9]*(\d+)[^0-9]*min', combined, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 60
    return RATE_LIMIT_BASE_WAIT


def _run_claude_cli(cmd, input_text, timeout_secs):
    """Run a Claude CLI command with rate limit retry.

    Args:
        cmd: Command list for subprocess
        input_text: Text to send to stdin
        timeout_secs: Timeout in seconds

    Returns:
        subprocess.CompletedProcess result

    Raises:
        subprocess.TimeoutExpired if all attempts time out.
    """
    env = _claude_env()
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        result = subprocess.run(
            cmd, input=input_text, capture_output=True, text=True,
            timeout=timeout_secs, env=env
        )
        if result.returncode == 0 or not _is_rate_limited(result):
            # Try to record token usage from JSON responses
            if result.returncode == 0 and result.stdout.strip():
                try:
                    resp = json.loads(result.stdout)
                    if isinstance(resp, dict):
                        token_tracker.record(resp)
                except (json.JSONDecodeError, ValueError):
                    pass  # non-JSON output, skip tracking
            return result
        wait = _extract_wait_time(result) + 10  # small buffer
        if attempt < RATE_LIMIT_MAX_RETRIES:
            print(f"    Rate limited. Waiting {wait}s before retry ({attempt + 1}/{RATE_LIMIT_MAX_RETRIES})...")
            time.sleep(wait)
    return result  # return last attempt's result


def _extract_structured_output(response_json):
    """Extract the actual structured output from Claude CLI JSON response.

    Claude CLI --output-format json may wrap output in different ways:
    - {"result": ...} — direct result
    - {"structured_output": ...} — structured output wrapper
    """
    if "structured_output" in response_json:
        inner = response_json["structured_output"]
        if isinstance(inner, str):
            return json.loads(inner)
        return inner
    if "result" in response_json:
        inner = response_json["result"]
        if isinstance(inner, str):
            return json.loads(inner)
        return inner
    return response_json


# JSON schema for per-hunk Claude analysis
HUNK_ANALYSIS_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "What the code in this hunk does, including the purpose of key functions and logic"
        },
        "change_summary": {
            "type": "string",
            "description": "What functionally changed (not just what's different — the semantic delta)"
        },
        "key_identifiers": {
            "type": "object",
            "description": "Map of minified identifier names to guessed meaningful names, e.g. {\"e\": \"event\", \"t\": \"timeout\"}",
            "additionalProperties": {"type": "string"}
        }
    },
    "required": ["description", "change_summary", "key_identifiers"]
})


def analyze_hunk_with_claude(hunk_content, file_path, string_changes, model="haiku", tier="direct"):
    """Analyze a single diff hunk using Claude CLI.

    Args:
        hunk_content: The diff hunk text
        file_path: Path of the file being analyzed
        string_changes: Dict with 'new_strings' and 'removed_strings' for context
        model: Claude model to use (default: sonnet)
        tier: Analysis tier - "direct", "subline", or "string_fallback"

    Returns:
        Parsed JSON dict with analysis, or error dict on failure.
    """
    new_strings = string_changes.get("new_strings", [])
    removed_strings = string_changes.get("removed_strings", [])

    # Cap string context to avoid bloating the prompt
    def _cap_strings(strings, max_total=MAX_STRING_CONTEXT):
        result = []
        total = 0
        for s in strings:
            truncated = s[:MAX_STRING_LENGTH] + "..." if len(s) > MAX_STRING_LENGTH else s
            entry = f"- {truncated}"
            if total + len(entry) > max_total:
                result.append(f"- ... ({len(strings) - len(result)} more truncated)")
                break
            result.append(entry)
            total += len(entry)
        return result

    context_section = ""
    if new_strings or removed_strings:
        context_section = "\n\n## String Changes in This File\n"
        if new_strings:
            context_section += "New strings:\n" + "\n".join(_cap_strings(new_strings)) + "\n"
        if removed_strings:
            context_section += "Removed strings:\n" + "\n".join(_cap_strings(removed_strings)) + "\n"

    if tier == "subline":
        prompt = f"""You are analyzing statement-level changes extracted from a minified/bundled Electron desktop application (Claude Desktop).
The diff below shows individual JavaScript statements that changed between versions.
Minified identifier renames have already been filtered out — what remains are semantic/functional changes.
Lines prefixed with '-' were removed, '+' were added, unprefixed lines are context.

## File: {file_path}
{context_section}
## Statement-Level Changes
```
{hunk_content}
```

## Instructions
1. Describe what the changed statements do.
2. Describe what *changed* — the functional/behavioral delta, not just syntactic differences.
3. Provide a key_identifiers map of important minified names to guessed meaningful names.

Return your analysis as JSON."""
    elif tier == "string_fallback":
        prompt = f"""You are analyzing string and API changes extracted from an oversized diff hunk in a minified/bundled Electron desktop application (Claude Desktop).
The full code diff was too large to analyze directly, so only the string literals, URLs, imports, and property names that changed are shown below.

## File: {file_path}

## String/API Changes
{hunk_content}

## Instructions
1. Based on the string changes, infer what new features, APIs, or behaviors were added or removed.
2. Describe the likely functional changes.
3. Note any significant new URLs, error messages, feature flags, or configuration keys.

Return your analysis as JSON."""
    else:
        prompt = f"""You are analyzing a diff hunk from a minified/bundled Electron desktop application (Claude Desktop).
The code has been beautified but identifiers are still minified (single letters, short meaningless names).

## File: {file_path}
{context_section}
## Diff Hunk
```
{hunk_content}
```

## Instructions
1. Describe what the code in this hunk does.
2. Describe what *changed* — not just what's different, but the functional/behavioral delta.
3. Provide a key_identifiers map of important minified names to guessed meaningful names.

Return your analysis as JSON."""

    error_stub = {"description": "", "change_summary": "", "key_identifiers": {}}
    try:
        result = _run_claude_cli(
            ["claude", "-p", "--model", model,
             "--output-format", "json", "--json-schema", HUNK_ANALYSIS_SCHEMA,
             "--dangerously-skip-permissions"],
            input_text=prompt, timeout_secs=1200
        )
        if result.returncode != 0:
            return {"error": f"Claude CLI failed: {result.stderr[:500]}", **error_stub}

        response = json.loads(result.stdout)
        return _extract_structured_output(response)

    except subprocess.TimeoutExpired:
        return {"error": "Claude CLI timed out", **error_stub}
    except (json.JSONDecodeError, KeyError) as e:
        return {"error": f"Failed to parse Claude response: {e}", **error_stub}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", **error_stub}


# JSON schema for batched hunk analysis (array of per-hunk results)
HUNK_BATCH_SCHEMA = json.dumps({
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "hunk_index": {
                "type": "integer",
                "description": "The 1-based hunk number from the prompt"
            },
            "description": {
                "type": "string",
                "description": "What the code in this hunk does, including the purpose of key functions and logic"
            },
            "change_summary": {
                "type": "string",
                "description": "What functionally changed (not just what's different — the semantic delta)"
            },
            "key_identifiers": {
                "type": "object",
                "description": "Map of minified identifier names to guessed meaningful names",
                "additionalProperties": {"type": "string"}
            }
        },
        "required": ["hunk_index", "description", "change_summary", "key_identifiers"]
    }
})


def _batch_hunk_tasks(tasks, max_batch_chars=30000):
    """Group consecutive same-file hunk tasks into batches.

    Args:
        tasks: List of hunk task dicts (must share the same file within a batch)
        max_batch_chars: Max total content chars per batch

    Returns:
        List of lists, where each inner list is a batch of tasks to analyze together.
    """
    batches = []
    current_batch = []
    current_file = None
    current_chars = 0

    for task in tasks:
        task_chars = len(task["content"])
        # Start new batch if: different file, or would exceed limit, or task is non-direct tier
        if (task["file"] != current_file or
                current_chars + task_chars > max_batch_chars or
                task["tier"] != "direct"):
            if current_batch:
                batches.append(current_batch)
            current_batch = [task]
            current_file = task["file"]
            current_chars = task_chars
        else:
            current_batch.append(task)
            current_chars += task_chars

    if current_batch:
        batches.append(current_batch)
    return batches


def analyze_hunk_batch_with_claude(batch, model="haiku"):
    """Analyze multiple hunks from the same file in a single Claude call.

    Args:
        batch: List of hunk task dicts (all same file)
        model: Claude model to use

    Returns:
        List of (task, result) tuples.
    """
    file_path = batch[0]["file"]
    string_changes = batch[0]["string_changes"]
    new_strings = string_changes.get("new_strings", [])
    removed_strings = string_changes.get("removed_strings", [])

    # Build string context once for the batch
    def _cap_strings(strings, max_total=MAX_STRING_CONTEXT):
        result = []
        total = 0
        for s in strings:
            truncated = s[:MAX_STRING_LENGTH] + "..." if len(s) > MAX_STRING_LENGTH else s
            entry = f"- {truncated}"
            if total + len(entry) > max_total:
                result.append(f"- ... ({len(strings) - len(result)} more truncated)")
                break
            result.append(entry)
            total += len(entry)
        return result

    context_section = ""
    if new_strings or removed_strings:
        context_section = "\n\n## String Changes in This File\n"
        if new_strings:
            context_section += "New strings:\n" + "\n".join(_cap_strings(new_strings)) + "\n"
        if removed_strings:
            context_section += "Removed strings:\n" + "\n".join(_cap_strings(removed_strings)) + "\n"

    # Build the multi-hunk prompt
    hunks_section = ""
    for i, task in enumerate(batch):
        hunks_section += f"\n### Hunk {i + 1} (progress key: {task['progress_key']})\n```\n{task['content']}\n```\n"

    prompt = f"""You are analyzing {len(batch)} diff hunks from a minified/bundled Electron desktop application (Claude Desktop).
The code has been beautified but identifiers are still minified (single letters, short meaningless names).

## File: {file_path}
{context_section}
{hunks_section}

## Instructions
For EACH hunk above, provide:
1. A description of what the code does.
2. What *changed* — not just what's different, but the functional/behavioral delta.
3. A key_identifiers map of important minified names to guessed meaningful names.

Return a JSON array with one object per hunk, in the same order. Each object must include `hunk_index` (1-based)."""

    error_stub = {"description": "", "change_summary": "", "key_identifiers": {}}

    try:
        result = _run_claude_cli(
            ["claude", "-p", "--model", model,
             "--output-format", "json", "--json-schema", HUNK_BATCH_SCHEMA,
             "--dangerously-skip-permissions"],
            input_text=prompt, timeout_secs=180
        )
        if result.returncode != 0:
            return [(task, {"error": f"Claude CLI failed: {result.stderr[:500]}", **error_stub}) for task in batch]

        response = json.loads(result.stdout)
        parsed = _extract_structured_output(response)

        # Map results back to tasks by position
        results = []
        if isinstance(parsed, list):
            for i, task in enumerate(batch):
                if i < len(parsed):
                    results.append((task, parsed[i]))
                else:
                    results.append((task, {"error": "Missing from batch response", **error_stub}))
        else:
            # Unexpected format — treat as single result for first task, error for rest
            results.append((batch[0], parsed if isinstance(parsed, dict) else {"error": "Unexpected response format", **error_stub}))
            for task in batch[1:]:
                results.append((task, {"error": "Batch response was not an array", **error_stub}))
        return results

    except subprocess.TimeoutExpired:
        return [(task, {"error": "Claude CLI timed out", **error_stub}) for task in batch]
    except (json.JSONDecodeError, KeyError) as e:
        return [(task, {"error": f"Failed to parse Claude response: {e}", **error_stub}) for task in batch]
    except Exception as e:
        return [(task, {"error": f"Unexpected error: {e}", **error_stub}) for task in batch]


def _build_file_analysis_text(hunks, include_code=True):
    """Build analysis text for a list of hunk results."""
    text = ""
    for i, hunk in enumerate(hunks):
        if "error" in hunk and hunk["error"]:
            text += f"\n### Hunk {i + 1} (analysis failed: {hunk['error']})\n"
            continue
        text += f"\n### Hunk {i + 1}\n"
        text += f"**Description:** {hunk.get('description', 'N/A')}\n"
        text += f"**Change summary:** {hunk.get('change_summary', 'N/A')}\n"
        key_ids = hunk.get("key_identifiers")
        if include_code and key_ids:
            mappings = ", ".join(f"`{k}` = {v}" for k, v in key_ids.items())
            text += f"**Key identifiers:** {mappings}\n"
    return text


def _summarize_file_with_claude(file_path, hunks, old_tag, new_tag, model):
    """Summarize changes for a single file using Claude."""
    analysis_text = _build_file_analysis_text(hunks, include_code=True)
    # If still too large without code, strip it
    if len(analysis_text) > 80000:
        analysis_text = _build_file_analysis_text(hunks, include_code=False)

    prompt = f"""Summarize the changes in this file between two releases of Claude Desktop (Electron app).

**Old release:** {old_tag}
**New release:** {new_tag}
**File:** {file_path}
**Hunks analyzed:** {len(hunks)}

{analysis_text}

Write a concise summary (1-5 bullet points) of the substantive changes in this file.
Focus on functional/behavioral changes, not minified name shuffles. Omit hunks that failed."""

    try:
        result = _run_claude_cli(
            ["claude", "-p", "--model", model, "--output-format", "json",
             "--dangerously-skip-permissions"],
            input_text=prompt, timeout_secs=180
        )
        if result.returncode != 0:
            return f"- {file_path}: summary failed ({result.stderr[:200]})"
        try:
            resp = json.loads(result.stdout)
            return resp.get("result", result.stdout)
        except (json.JSONDecodeError, ValueError):
            return result.stdout
    except subprocess.TimeoutExpired:
        return f"- {file_path}: summary timed out"
    except Exception as e:
        return f"- {file_path}: summary error ({e})"


# Max chars of analysis text that fits comfortably in a single summary call
_SUMMARY_SINGLE_STAGE_LIMIT = 100000


def generate_summary_with_claude(all_analyses, old_tag, new_tag, model="opus", voice_profile=None):
    """Generate a cohesive summary from all per-hunk analyses using Claude.

    For small diffs, sends all analyses in one prompt. For large diffs
    (e.g., difflib fallback on big bundles), uses a two-stage approach:
    first summarize each file independently, then combine.

    Args:
        all_analyses: List of dicts with file path and hunk analyses
        old_tag: Old release tag
        new_tag: New release tag
        model: Claude model to use (default: opus)
        voice_profile: Optional voice profile text for styling the summary

    Returns:
        Summary markdown string.
    """
    # Build the analysis content grouped by file
    analysis_text = ""
    for file_analysis in all_analyses:
        file_path = file_analysis["file"]
        hunks = file_analysis["hunks"]
        if not hunks:
            continue
        analysis_text += f"\n## {file_path}\n"
        analysis_text += _build_file_analysis_text(hunks, include_code=True)

    # If the analysis text is small enough, use single-stage summary
    if len(analysis_text) <= _SUMMARY_SINGLE_STAGE_LIMIT:
        return _generate_final_summary(analysis_text, old_tag, new_tag, model, voice_profile)

    # Two-stage summary: per-file first, then combine
    print("    Analysis too large for single prompt, using two-stage summary...")
    file_summaries = []
    for file_analysis in all_analyses:
        file_path = file_analysis["file"]
        hunks = file_analysis["hunks"]
        if not hunks:
            continue
        ok_hunks = [h for h in hunks if not ("error" in h and h["error"])]
        if not ok_hunks:
            continue

        file_text = _build_file_analysis_text(ok_hunks, include_code=False)
        if len(file_text) > _SUMMARY_SINGLE_STAGE_LIMIT:
            # Very large file — needs its own summarization pass
            print(f"    Summarizing {file_path} ({len(ok_hunks)} hunks)...")
            summary = _summarize_file_with_claude(
                file_path, ok_hunks, old_tag, new_tag, model)
            file_summaries.append(f"\n## {file_path}\n{summary}")
        else:
            file_summaries.append(f"\n## {file_path}\n{file_text}")

    combined = "\n".join(file_summaries)
    # If combined per-file summaries still too large, truncate
    if len(combined) > _SUMMARY_SINGLE_STAGE_LIMIT:
        combined = combined[:_SUMMARY_SINGLE_STAGE_LIMIT] + "\n\n... [remaining file summaries truncated] ...\n"

    return _generate_final_summary(combined, old_tag, new_tag, model, voice_profile)


def _generate_final_summary(analysis_text, old_tag, new_tag, model, voice_profile=None):
    """Generate the final summary from (possibly pre-summarized) analysis text."""
    voice_section = ""
    if voice_profile:
        voice_section = f"""
## Writing Style

Write the summary in the voice described by this profile. Match the tone, sentence rhythm,
and structural patterns. The voice is friendly, direct, and technically competent.

<voice-profile>
{voice_profile}
</voice-profile>
"""

    prompt = f"""You are summarizing the changes between two releases of Claude Desktop, an Electron application.

**Old release:** {old_tag}
**New release:** {new_tag}

Below are per-hunk analyses of every changed code section. Each hunk has been individually analyzed
to deobfuscate minified names and describe what changed.

{analysis_text}

## Instructions
Write a concise release notes summary. Focus on changes that matter to users and developers.

**Skip mundane/routine items entirely** — do NOT include:
- IPC channel UUID rotations (these happen every build)
- Minifier variable renames or re-identifier passes
- Vite content hash rotations
- Sentry release SHA updates
- Build timestamp changes
- Any change that is purely a mechanical side-effect of the build process

**Do include:**
- New features and capabilities
- Bug fixes
- Meaningful dependency additions or major version bumps
- Behavioral changes, API changes, or refactors that affect functionality
- CSS/UI changes that affect what users see

Structure the summary with:
1. A brief overview (2-3 sentences) of the highlights
2. Sections for notable changes, grouped naturally (features, fixes, dependency updates, etc.)
3. Keep descriptions clear and practical — explain what changed and why it matters

Do NOT include a confidence assessment table. If you're unsure about something, note it inline.
{voice_section}
Write in markdown format."""

    try:
        result = _run_claude_cli(
            ["claude", "-p", "--model", model, "--output-format", "json",
             "--dangerously-skip-permissions"],
            input_text=prompt, timeout_secs=300
        )
        if result.returncode != 0:
            return f"# Summary Generation Failed\n\nClaude CLI error: {result.stderr[:1000]}"
        try:
            resp = json.loads(result.stdout)
            return resp.get("result", result.stdout)
        except (json.JSONDecodeError, ValueError):
            return result.stdout

    except subprocess.TimeoutExpired:
        return "# Summary Generation Failed\n\nClaude CLI timed out during summary generation."
    except Exception as e:
        return f"# Summary Generation Failed\n\nUnexpected error: {e}"


def run_claude_analysis(file_reports, workdir, old_tag, new_tag, summary_model="sonnet", voice_profile=None):
    """Orchestrate Claude analysis of all changed hunks.

    Manages progress tracking, per-hunk analysis, and final summary generation.

    Args:
        file_reports: List of file report dicts from analyze_changes()
        workdir: Path to working directory
        old_tag: Old release tag
        new_tag: New release tag
        summary_model: Model to use for final summary (default: sonnet)
        voice_profile: Optional voice profile text for styling the summary
    """
    workdir = Path(workdir)
    progress_path = workdir / "analysis-progress.json"
    summary_path = workdir / "summary.md"
    use_difft = get_has_difft()

    # Load existing progress for resume support
    progress = {"status": "in_progress", "total_hunks": 0, "completed_hunks": 0, "hunks": []}
    completed_keys = set()
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text())
            failed_count = 0
            for h in progress.get("hunks", []):
                if h.get("status") == "completed":
                    # Only skip hunks that succeeded — retry failed ones
                    if h.get("result", {}).get("error"):
                        failed_count += 1
                        continue
                    completed_keys.add((h["file"], h["hunk_index"]))
                    # Also support new progress_key format
                    if "progress_key" in h:
                        completed_keys.add((h["file"], h["progress_key"]))
            ok_count = len(completed_keys)
            print(f"  Resuming from progress file: {ok_count} hunks succeeded, {failed_count} failed (will retry)")
        except (json.JSONDecodeError, KeyError):
            progress = {"status": "in_progress", "total_hunks": 0, "completed_hunks": 0, "hunks": []}

    # Build list of all hunks to analyze, with preprocessing for oversized hunks
    all_hunk_tasks = []
    skipped_files = []
    noise_hunks = 0
    for fr in file_reports:
        if fr["status"] not in ("modified", "renamed"):
            continue
        hunks_text = fr.get("hunks", "")
        if not hunks_text or not hunks_text.strip():
            continue

        # Skip binary files
        file_path = fr["path"]
        first_line = hunks_text.strip().splitlines()[0] if hunks_text.strip() else ""
        if "Binary" in first_line:
            skipped_files.append((file_path, "binary"))
            continue

        # File-level noise filter: if all string changes are just UUIDs,
        # hashes, sentry IDs, IPC namespace rotations, etc., skip the file.
        # But don't skip files with no string changes — they may have real
        # structural changes not captured by string extraction (e.g., JSON files).
        new_strings = fr.get("new_strings", [])
        removed_strings = fr.get("removed_strings", [])
        if (new_strings or removed_strings) and are_string_changes_noise_only(new_strings, removed_strings):
            skipped_files.append((file_path, "build noise only"))
            continue

        file_used_difft = fr.get("diff_engine", "difftastic" if use_difft else "difflib") == "difftastic"
        hunks = split_hunks(hunks_text, file_used_difft)
        string_changes = {
            "new_strings": new_strings,
            "removed_strings": removed_strings,
        }

        for idx, hunk in enumerate(hunks):
            # Skip hunks where difftastic fell back to text mode (not useful for analysis)
            if file_used_difft and "exceeded DFT_" in hunk.get("header", ""):
                continue
            if is_build_noise_only(hunk["content"], use_difft=file_used_difft):
                continue

            # Preprocess oversized hunks
            preprocessed = preprocess_hunk(hunk, string_changes)
            for sub_idx, item in enumerate(preprocessed):
                if item["tier"] == "noise":
                    noise_hunks += 1
                    continue
                if not item["content"].strip():
                    continue

                # Build a unique key for progress tracking
                # Use (file, hunk_index, sub_index) for chunked hunks
                sub_key = f"{idx}" if len(preprocessed) == 1 else f"{idx}.{sub_idx}"
                all_hunk_tasks.append({
                    "file": file_path,
                    "hunk_index": idx,
                    "sub_index": sub_idx,
                    "progress_key": sub_key,
                    "hunk_header": item.get("header", hunk["header"]),
                    "content": item["content"],
                    "tier": item["tier"],
                    "original_size": item.get("original_size", len(hunk["content"])),
                    "string_changes": string_changes,
                })

    if skipped_files:
        print(f"  Skipped {len(skipped_files)} files: {', '.join(f'{f} ({r})' for f, r in skipped_files)}")
    if noise_hunks:
        print(f"  Filtered {noise_hunks} noise-only hunks (pure minified renames)")

    total = len(all_hunk_tasks)
    progress["total_hunks"] = total
    print(f"  Found {total} non-trivial hunks to analyze")

    if total == 0:
        print("  No hunks to analyze — all changes appear to be build artifacts.")
        progress["status"] = "completed"
        progress_path.write_text(json.dumps(progress, indent=2))
        summary_path.write_text("# No Substantive Changes\n\nAll changes between releases appear to be build artifacts (hashes, UUIDs, version strings).\n")
        return

    # Analyze each hunk — count how many need work vs already done
    tasks_to_run = []
    already_done = 0
    for task in all_hunk_tasks:
        key = (task["file"], task["hunk_index"])
        new_key = (task["file"], task["progress_key"])
        if key in completed_keys or new_key in completed_keys:
            already_done += 1
        else:
            tasks_to_run.append(task)

    if already_done:
        print(f"  Skipping {already_done} already-completed hunks")

    # Group tasks into batches (consecutive same-file hunks)
    batches = _batch_hunk_tasks(tasks_to_run)
    completed = 0
    remaining = len(tasks_to_run)
    progress_lock = threading.Lock()
    max_workers = 4

    def _run_batch(batch, batch_idx):
        """Execute a single batch and return (batch_idx, results)."""
        if len(batch) == 1:
            task = batch[0]
            result = analyze_hunk_with_claude(
                task["content"], task["file"], task["string_changes"],
                tier=task["tier"]
            )
            return batch_idx, [(task, result)]
        else:
            return batch_idx, analyze_hunk_batch_with_claude(batch)

    print(f"  Processing {len(batches)} batches with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for batch_idx, batch in enumerate(batches):
            if len(batch) == 1:
                task = batch[0]
                tier_label = f" [{task['tier']}]" if task["tier"] != "direct" else ""
                size_label = f" (preprocessed from {task['original_size']} chars)" if task["original_size"] > HUNK_SIZE_THRESHOLD else ""
                print(f"  Queued: {task['file']} hunk {task['progress_key']}{tier_label}{size_label}")
            else:
                keys = ", ".join(t["progress_key"] for t in batch)
                print(f"  Queued batch: {batch[0]['file']} hunks {keys}")
            future = executor.submit(_run_batch, batch, batch_idx)
            futures[future] = batch

        for future in as_completed(futures):
            batch = futures[future]
            try:
                _batch_idx, batch_results = future.result()
            except Exception as e:
                # If the entire batch call raised, record errors for all tasks
                error_stub = {"description": "", "change_summary": "", "key_identifiers": {}}
                batch_results = [(t, {"error": f"Batch execution failed: {e}", **error_stub}) for t in batch]

            with progress_lock:
                for task, result in batch_results:
                    hunk_entry = {
                        "file": task["file"],
                        "hunk_index": task["hunk_index"],
                        "progress_key": task["progress_key"],
                        "hunk_header": task["hunk_header"],
                        "tier": task["tier"],
                        "original_size": task["original_size"],
                        "status": "completed",
                        "result": result,
                    }
                    progress["hunks"].append(hunk_entry)
                    completed += 1

                progress["completed_hunks"] = already_done + completed
                progress_path.write_text(json.dumps(progress, indent=2))
                print(f"  [{completed}/{remaining}] hunks complete")

    progress["status"] = "analysis_complete"
    progress_path.write_text(json.dumps(progress, indent=2))

    # Group results by file for summary, using latest result per hunk key
    # (handles retries where multiple entries exist for the same hunk)
    latest_by_key = {}
    for h in progress["hunks"]:
        if h["status"] != "completed":
            continue
        # Use progress_key if available, fall back to hunk_index
        key = (h["file"], h.get("progress_key", h["hunk_index"]))
        latest_by_key[key] = h  # later entries overwrite earlier (failed) ones

    file_analyses = {}
    for h in latest_by_key.values():
        # Skip entries that still have errors after retries
        if h.get("result", {}).get("error"):
            continue
        f = h["file"]
        if f not in file_analyses:
            file_analyses[f] = {"file": f, "hunks": []}
        file_analyses[f]["hunks"].append(h["result"])

    all_analyses = sorted(file_analyses.values(), key=lambda x: x["file"])

    # Generate summary
    print(f"\n  Generating summary with {summary_model}...")
    summary = generate_summary_with_claude(all_analyses, old_tag, new_tag, model=summary_model, voice_profile=voice_profile)

    cost_section = token_tracker.summary_markdown()
    summary_path.write_text(summary + cost_section)
    progress["status"] = "completed"
    progress_path.write_text(json.dumps(progress, indent=2))

    print(f"  Summary written to {summary_path}")

    # Print token usage summary
    print("\n  === Token Usage by Model ===")
    print(token_tracker.summary())


def main():
    parser = argparse.ArgumentParser(
        description="Compare two claude-desktop-debian releases to identify functional changes."
    )
    parser.add_argument("--old", help="Old release tag (default: auto-detect)")
    parser.add_argument("--new", help="New release tag (default: auto-detect)")
    parser.add_argument("--workdir", default="./compare-work",
                        help="Working directory (default: ./compare-work)")
    parser.add_argument("--keep", action="store_true",
                        help="Keep extracted/beautified files after run")
    parser.add_argument("--new-appimage",
                        help="Path to a local AppImage for the new version (skip download)")
    parser.add_argument("--no-analyze", action="store_true",
                        help="Skip Claude-powered analysis step (produce only raw reports)")
    parser.add_argument("--model", default="sonnet",
                        help="Model for summary generation (default: sonnet)")
    parser.add_argument("--voice-profile-url",
                        help="URL to a voice profile .md file for styling the summary")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # Fetch voice profile if requested
    voice_profile = None
    if args.voice_profile_url:
        voice_profile = fetch_voice_profile(args.voice_profile_url)

    # Step 1: Resolve releases
    print("=== Step 1: Resolving releases ===")
    old_tag, new_tag = resolve_releases(args.old, args.new)
    print(f"  Old: {old_tag}")
    print(f"  New: {new_tag}")

    old_work = workdir / "old_work"
    new_work = workdir / "new_work"
    old_app = workdir / "old"
    new_app = workdir / "new"
    old_work.mkdir(parents=True, exist_ok=True)
    new_work.mkdir(parents=True, exist_ok=True)

    # Step 2: Download AppImages
    print("\n=== Step 2: Downloading AppImages ===")
    old_appimage = download_appimage(old_tag, old_work)
    if args.new_appimage:
        new_appimage = Path(args.new_appimage).resolve()
        if not new_appimage.exists():
            raise FileNotFoundError(f"Local AppImage not found: {new_appimage}")
        print(f"  Using local AppImage: {new_appimage}")
    else:
        new_appimage = download_appimage(new_tag, new_work)

    # Step 3: Extract AppImages
    print("\n=== Step 3: Extracting AppImages ===")
    old_squashfs = extract_appimage(old_appimage, old_work)
    new_squashfs = extract_appimage(new_appimage, new_work)

    # Step 4: Extract app.asar
    print("\n=== Step 4: Extracting app.asar ===")
    old_app = extract_asar(old_squashfs, old_app)
    new_app = extract_asar(new_squashfs, new_app)

    # Step 5: Beautify JS
    print("\n=== Step 5: Beautifying JS ===")
    beautify_js(old_app)
    beautify_js(new_app)

    # Step 6: Compare and analyze
    print("\n=== Step 6: Comparing files ===")
    added, removed, modified, unchanged, renamed = compare_file_trees(old_app, new_app)
    print(f"  Added: {len(added)}, Removed: {len(removed)}, Modified: {len(modified)}, "
          f"Renamed: {len(renamed)}, Unchanged: {len(unchanged)}")

    if get_has_difft():
        print("  Using difftastic for structural diffs")
    else:
        print("  difftastic not found, using difflib (install difft for AST-aware diffs)")

    print("\n=== Step 6b: Analyzing changes ===")
    file_reports = analyze_changes(old_app, new_app, added, removed, modified, renamed)

    # Step 7: Generate reports
    print("\n=== Step 7: Generating reports ===")
    report_md = generate_report_md(old_tag, new_tag, added, removed, modified, unchanged, renamed, file_reports)
    report_json = generate_report_json(old_tag, new_tag, added, removed, modified, unchanged, renamed, file_reports)

    md_path = workdir / "report.md"
    json_path = workdir / "report.json"

    md_path.write_text(report_md)
    json_path.write_text(json.dumps(report_json, indent=2))

    print(f"\n  Reports written to:")
    print(f"    {md_path}")
    print(f"    {json_path}")

    # Step 8: Claude-powered analysis
    if not args.no_analyze:
        if get_has_claude():
            print("\n=== Step 8: Claude-powered analysis ===")
            run_claude_analysis(file_reports, workdir, old_tag, new_tag, summary_model=args.model, voice_profile=voice_profile)
        else:
            print("\n=== Step 8: Skipped (claude CLI not found) ===")
            print("  Install the claude CLI to enable AI-powered change analysis.")
    else:
        print("\n=== Step 8: Skipped (--no-analyze) ===")

    # Cleanup if not keeping
    if not args.keep:
        print("\n=== Cleanup ===")
        shutil.rmtree(old_work, ignore_errors=True)
        shutil.rmtree(new_work, ignore_errors=True)
        shutil.rmtree(old_app, ignore_errors=True)
        shutil.rmtree(new_app, ignore_errors=True)
        print("  Removed extracted files (use --keep to preserve them)")
    else:
        print(f"\n  Extracted files preserved in {workdir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
