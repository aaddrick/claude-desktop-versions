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
import resource
import shutil
import tarfile
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import urllib.request
from pathlib import Path


REPO = "aaddrick/claude-desktop-debian"

# Anthropic SDK (optional — falls back to claude CLI if unavailable)
try:
    import anthropic
    HAS_SDK = True
except ImportError:
    HAS_SDK = False

# Tree-sitter (optional — falls back to text diff if unavailable)
try:
    import tree_sitter_javascript as _tsjs
    from tree_sitter import Language as _TsLanguage, Parser as _TsParser
    _ts_parser = _TsParser(_TsLanguage(_tsjs.language()))
    HAS_TREESITTER = True
except ImportError:
    HAS_TREESITTER = False

# Model pricing for cost calculation (USD per million tokens)
# SDK doesn't provide costUSD like CLI does, so we compute it
MODEL_PRICING = {
    # Standard pricing
    "claude-opus-4-6": {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75, "batch_discount": 0.5},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75, "batch_discount": 0.5},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25, "batch_discount": 0.5},
    # Aliases used in CLI (short names map to latest)
    "opus": {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75, "batch_discount": 0.5},
    "sonnet": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75, "batch_discount": 0.5},
    "haiku": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25, "batch_discount": 0.5},
}

# Map short model names to full SDK model IDs
MODEL_ID_MAP = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}


def _log_memory(label):
    """Log peak RSS for both the Python process and its waited-for children."""
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # On Linux, ru_maxrss is in KB; on macOS, it's in bytes
    if sys.platform == "darwin":
        self_mb = self_rss / (1024 * 1024)
        child_mb = child_rss / (1024 * 1024)
    else:
        self_mb = self_rss / 1024
        child_mb = child_rss / 1024
    print(f"  [Memory] {label}: self={self_mb:.1f} MB, children_peak={child_mb:.1f} MB")


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

    def summary_markdown(self, duration_secs=None):
        """Return a markdown-formatted table of token usage for inclusion in reports."""
        with self._lock:
            if not self._usage:
                return ""
            lines = ["\n\n---\n", "### Analysis Cost", ""]
            if duration_secs is not None:
                mins, secs = divmod(int(duration_secs), 60)
                lines.append(f"**Duration:** {mins}m {secs}s\n")
            lines.append("| Model | Calls | Input | Cache Read | Cache Write | Output | Cost |")
            lines.append("|-------|------:|------:|-----------:|------------:|-------:|-----:|")
            total_cost = 0.0
            total_calls = 0
            total_input = 0
            total_cache_read = 0
            total_cache_write = 0
            total_output = 0
            for model_id in sorted(self._usage):
                u = self._usage[model_id]
                total_cost += u["costUSD"]
                total_calls += u["calls"]
                total_input += u["inputTokens"]
                total_cache_read += u["cacheReadInputTokens"]
                total_cache_write += u["cacheCreationInputTokens"]
                total_output += u["outputTokens"]
                lines.append(
                    f"| {model_id} | {u['calls']} | "
                    f"{u['inputTokens']:,} | "
                    f"{u['cacheReadInputTokens']:,} | "
                    f"{u['cacheCreationInputTokens']:,} | "
                    f"{u['outputTokens']:,} | "
                    f"${u['costUSD']:.4f} |"
                )
            lines.append(
                f"| **Total** | **{total_calls}** | "
                f"**{total_input:,}** | "
                f"**{total_cache_read:,}** | "
                f"**{total_cache_write:,}** | "
                f"**{total_output:,}** | "
                f"**${total_cost:.4f}** |"
            )
            lines.append("")
            lines.append("*Like this project? [Consider sponsoring!](https://github.com/sponsors/aaddrick)*")
            return "\n".join(lines)


def _install_instructions_markdown():
    """Return markdown install instructions for inclusion in release notes."""
    return """

---

### Installation

#### APT (Debian/Ubuntu - Recommended)

```bash
# Add the GPG key
curl -fsSL https://aaddrick.github.io/claude-desktop-debian/KEY.gpg | sudo gpg --dearmor -o /usr/share/keyrings/claude-desktop.gpg

# Add the repository
echo "deb [signed-by=/usr/share/keyrings/claude-desktop.gpg arch=amd64,arm64] https://aaddrick.github.io/claude-desktop-debian stable main" | sudo tee /etc/apt/sources.list.d/claude-desktop.list

# Update and install
sudo apt update
sudo apt install claude-desktop
```

#### DNF (Fedora/RHEL - Recommended)

```bash
# Add the repository
sudo curl -fsSL https://aaddrick.github.io/claude-desktop-debian/rpm/claude-desktop.repo -o /etc/yum.repos.d/claude-desktop.repo

# Install
sudo dnf install claude-desktop
```

#### AUR (Arch Linux)

```bash
# Using yay
yay -S claude-desktop-appimage

# Or using paru
paru -S claude-desktop-appimage
```

#### Pre-built Releases

Download the latest `.deb`, `.rpm`, or `.AppImage` from the [Releases page](https://github.com/aaddrick/claude-desktop-debian/releases)."""


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


def download_reference_source(tag, dest_dir):
    """Download and extract reference-source.tar.gz for a given release tag.

    Returns the path to the extracted app directory, or None if the tarball
    is not available for this release.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if this release has a reference-source.tar.gz asset
    output = run(["gh", "release", "view", tag, "--json", "assets", "-R", REPO])
    assets = json.loads(output).get("assets", [])
    tarball_name = None
    for a in assets:
        if a.get("name", "") == "reference-source.tar.gz":
            tarball_name = "reference-source.tar.gz"
            break

    if not tarball_name:
        return None

    tarball_path = dest_dir / tarball_name
    if not tarball_path.exists():
        print(f"  Downloading {tarball_name}...")
        run(["gh", "release", "download", tag,
             "--pattern", tarball_name,
             "--dir", str(dest_dir),
             "-R", REPO])
    else:
        print(f"  {tarball_name} already downloaded.")

    # Extract tarball — contains app-extracted/ directory
    app_dir = dest_dir / "app-extracted"
    if app_dir.exists() and any(app_dir.iterdir()):
        print(f"  Reference source already extracted, skipping.")
        return app_dir

    print(f"  Extracting reference source...")
    with tarfile.open(tarball_path, "r:gz") as tf:
        tf.extractall(path=str(dest_dir), filter="data")

    if not app_dir.exists():
        raise RuntimeError(f"Expected app-extracted/ in tarball, not found in {dest_dir}")

    return app_dir


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

    # Log file inventory for memory profiling
    file_sizes = [f.stat().st_size for f in js_files]
    total_size = sum(file_sizes)
    largest_size = max(file_sizes)
    largest_file = js_files[file_sizes.index(largest_size)]
    print(f"  JS file inventory: {len(js_files)} files, "
          f"total={total_size / (1024 * 1024):.1f} MB, "
          f"largest={largest_file.name} ({largest_size / (1024 * 1024):.1f} MB)")

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
    # Cap Node heap to avoid OOM on large files (confirmed: prettier spikes to 1.8GB uncapped)
    prettier_env = os.environ.copy()
    prettier_env["NODE_OPTIONS"] = "--max-old-space-size=512"
    # Run prettier in batches to avoid arg-list-too-long
    batch_size = 50
    for i in range(0, len(js_files), batch_size):
        batch = js_files[i:i + batch_size]
        try:
            run(["npx", "prettier", "--write", "--parser", "babel"] +
                [str(f) for f in batch], env=prettier_env)
        except RuntimeError:
            # Some files may fail to parse; try individually
            for f in batch:
                try:
                    run(["npx", "prettier", "--write", "--parser", "babel", str(f)],
                        env=prettier_env)
                except RuntimeError:
                    print(f"  Warning: prettier failed on {f.name} ({f.stat().st_size} bytes)")


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


# Threshold above which we use external diff instead of difflib (avoids O(n*m) memory)
DIFFLIB_SIZE_THRESHOLD = 200_000  # bytes — combined old+new text size


def _extract_hunks_external_diff(old_text, new_text, context_lines=5):
    """Generate unified diff using external diff command (memory-efficient for large files)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".old", delete=False) as f_old, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".new", delete=False) as f_new:
        f_old.write(old_text)
        f_new.write(new_text)
        f_old_path, f_new_path = f_old.name, f_new.name
    try:
        result = subprocess.run(
            ["diff", f"-U{context_lines}", "--label", "old", "--label", "new",
             f_old_path, f_new_path],
            capture_output=True, text=True
        )
        # diff returns 0=identical, 1=different, 2=error
        if result.returncode == 2:
            return f"Error running diff: {result.stderr[:500]}"
        return result.stdout
    finally:
        os.unlink(f_old_path)
        os.unlink(f_new_path)


def extract_changed_hunks_difflib(old_text, new_text, context_lines=5):
    """Generate unified diff hunks. Uses external diff for large files to avoid O(n*m) memory."""
    combined_size = len(old_text) + len(new_text)
    if combined_size > DIFFLIB_SIZE_THRESHOLD:
        print(f"    Using external diff for large file ({combined_size / 1024:.0f} KB combined)")
        return _extract_hunks_external_diff(old_text, new_text, context_lines)

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


def _ts_analyze_file(old_path, new_path, rel_label):
    """Analyze a JS file pair using tree-sitter structural comparison.

    Returns a file_report dict with declaration-level change information.
    Falls back to text diff if tree-sitter can't handle the file.
    """
    old_path = Path(old_path)
    new_path = Path(new_path)

    try:
        old_source = old_path.read_bytes()
        new_source = new_path.read_bytes()
    except Exception as e:
        return _analyze_file_pair(old_path, new_path, rel_label, ".js")

    old_decls, old_err_rate = _ts_fingerprint_declarations(old_source)
    new_decls, new_err_rate = _ts_fingerprint_declarations(new_source)

    # Robustness: fall back to text diff if parse quality is poor
    if old_err_rate > 0.05 or new_err_rate > 0.05:
        print(f"    {rel_label}: tree-sitter error rate too high "
              f"(old={old_err_rate:.1%}, new={new_err_rate:.1%}), falling back to text diff")
        return _analyze_file_pair(old_path, new_path, rel_label, ".js")

    # Robustness: fall back if too few declarations (e.g., giant IIFE wrapper)
    if len(old_decls) < 10 and len(new_decls) < 10:
        print(f"    {rel_label}: too few declarations "
              f"(old={len(old_decls)}, new={len(new_decls)}), falling back to text diff")
        return _analyze_file_pair(old_path, new_path, rel_label, ".js")

    match_result = _ts_match_declarations(old_decls, new_decls)

    # Also extract strings for context (reuse existing function)
    old_text = old_source.decode("utf-8", errors="replace")
    new_text = new_source.decode("utf-8", errors="replace")
    old_strings = extract_strings(old_text)
    new_strings = extract_strings(new_text)

    # Build text previews for added/removed declarations (cap at 2KB each)
    def _decl_preview(decl, source):
        raw = source[decl["byte_range"][0]:decl["byte_range"][1]]
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        if len(text) > 2048:
            text = text[:2048] + "\n... (truncated)"
        return text

    added_previews = [
        {"type": d["type"], "name": d.get("name"), "size": d["size"],
         "text": _decl_preview(d, new_source)}
        for d in match_result["added"]
    ]
    removed_previews = [
        {"type": d["type"], "name": d.get("name"), "size": d["size"],
         "text": _decl_preview(d, old_source)}
        for d in match_result["removed"]
    ]

    print(f"    {rel_label}: {match_result['unchanged']} unchanged, "
          f"{len(match_result['content_modified'])} content-modified, "
          f"{len(match_result['added'])} added, {len(match_result['removed'])} removed")

    return {
        "path": str(rel_label),
        "status": "modified",
        "analysis_mode": "treesitter",
        "unchanged_count": match_result["unchanged"],
        "content_modified": [
            {**cm, "old_strings": sorted(cm["old_strings"]), "new_strings": sorted(cm["new_strings"])}
            for cm in match_result["content_modified"]
        ],
        "added_declarations": added_previews,
        "removed_declarations": removed_previews,
        "new_strings": sorted(new_strings - old_strings),
        "removed_strings": sorted(old_strings - new_strings),
        "hunks": "",  # no text hunks in tree-sitter mode
    }


# Module-level flag for tree-sitter mode (set by main())
_use_treesitter = False


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
        if _use_treesitter and rel.suffix == ".js":
            report = _ts_analyze_file(old_dir / rel, new_dir / rel, rel)
        else:
            report = _analyze_file_pair(
                old_dir / rel, new_dir / rel, rel, rel.suffix)
        file_reports.append(report)

    # Renamed (hash-busted) files — treat as modified
    for old_rel, new_rel in sorted(renamed, key=lambda p: str(p[1])):
        if _use_treesitter and new_rel.suffix == ".js":
            report = _ts_analyze_file(old_dir / old_rel, new_dir / new_rel,
                                       f"{new_rel} (was {old_rel.name})")
            report["status"] = "renamed"
            report["old_path"] = str(old_rel)
        else:
            report = _analyze_file_pair(
                old_dir / old_rel, new_dir / new_rel,
                f"{new_rel} (was {old_rel.name})", new_rel.suffix)
            report["status"] = "renamed"
            report["old_path"] = str(old_rel)
        file_reports.append(report)

    return file_reports


def _render_file_report_md(fr, lines):
    """Render a single file report entry into markdown lines."""
    # Tree-sitter structural summary
    if fr.get("analysis_mode") == "treesitter":
        unchanged = fr.get("unchanged_count", 0)
        added_decls = fr.get("added_declarations", [])
        removed_decls = fr.get("removed_declarations", [])
        content_mod = fr.get("content_modified", [])
        lines.append(f"**Analysis:** tree-sitter structural comparison")
        lines.append(f"- Unchanged declarations: {unchanged}")
        if content_mod:
            lines.append(f"- Content-modified: {len(content_mod)} groups")
        if added_decls:
            lines.append(f"- Added: {len(added_decls)} declarations ({sum(d['size'] for d in added_decls) / 1024:.0f} KB)")
        if removed_decls:
            lines.append(f"- Removed: {len(removed_decls)} declarations ({sum(d['size'] for d in removed_decls) / 1024:.0f} KB)")
        lines.append("")

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

    diff_engine = "tree-sitter (structural)" if _use_treesitter else ("difftastic (AST-aware)" if get_has_difft() else "difflib (line-based)")

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
            "diff_engine": "tree-sitter" if _use_treesitter else ("difftastic" if get_has_difft() else "difflib"),
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
# --- Tree-sitter structural analysis functions ---

def _ts_structural_hash(node):
    """Recursively hash AST node types, ignoring leaf text (identifiers, literals).

    Two declarations that differ only in minified variable names will
    produce identical structural hashes.
    """
    if node.child_count == 0:
        return hashlib.sha256(node.type.encode()).digest()
    h = hashlib.sha256(node.type.encode())
    for child in node.children:
        h.update(_ts_structural_hash(child))
    return h.digest()


def _ts_get_decl_name(node, source):
    """Extract the identifier name from a top-level declaration node."""
    if node.type in ("function_declaration", "generator_function_declaration", "class_declaration"):
        for child in node.children:
            if child.type in ("identifier", "property_identifier"):
                return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    elif node.type in ("variable_declaration", "lexical_declaration"):
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                if name_node:
                    return source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
    return None


def _ts_extract_string_literals(node, source):
    """Extract string literal values from a node subtree (for content verification)."""
    strings = set()
    if node.type == "string" and node.child_count > 0:
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        if len(text) > 5:
            strings.add(text)
    for child in node.children:
        strings.update(_ts_extract_string_literals(child, source))
    return strings


def _ts_fingerprint_declarations(source_bytes):
    """Parse a JS file and fingerprint each top-level declaration.

    Returns list of dicts:
        [{index, type, name, struct_hash, content_hash, size, byte_range, strings}]
    """
    tree = _ts_parser.parse(source_bytes)
    root = tree.root_node

    # Check for excessive parse errors
    error_count = sum(1 for child in root.children if child.type == "ERROR")
    error_rate = error_count / max(root.child_count, 1)

    decls = []
    for i, node in enumerate(root.children):
        s_hash = _ts_structural_hash(node).hex()[:16]
        c_hash = hashlib.sha256(source_bytes[node.start_byte:node.end_byte]).hexdigest()[:16]
        name = _ts_get_decl_name(node, source_bytes)
        strings = _ts_extract_string_literals(node, source_bytes)
        decls.append({
            "index": i,
            "type": node.type,
            "name": name,
            "struct_hash": s_hash,
            "content_hash": c_hash,
            "size": node.end_byte - node.start_byte,
            "byte_range": (node.start_byte, node.end_byte),
            "strings": strings,
        })

    return decls, error_rate


def _ts_match_declarations(old_decls, new_decls):
    """Match declarations between two versions using structural hashes.

    Returns dict with:
        unchanged: count of structurally identical declarations (skip)
        content_modified: list of struct_hashes where structure matches but strings differ
        added: list of new declarations with no structural match
        removed: list of old declarations with no structural match
    """
    from collections import defaultdict

    # Group by structural hash
    old_by_struct = defaultdict(list)
    for d in old_decls:
        old_by_struct[d["struct_hash"]].append(d)

    new_by_struct = defaultdict(list)
    for d in new_decls:
        new_by_struct[d["struct_hash"]].append(d)

    unchanged_count = 0
    content_modified = []  # struct_hashes where strings differ
    added = []
    removed = []

    shared_hashes = old_by_struct.keys() & new_by_struct.keys()
    only_old_hashes = old_by_struct.keys() - new_by_struct.keys()
    only_new_hashes = new_by_struct.keys() - old_by_struct.keys()

    # Process shared structural hashes
    for sh in shared_hashes:
        old_group = old_by_struct[sh]
        new_group = new_by_struct[sh]
        matched = min(len(old_group), len(new_group))

        # Check if string content differs across the group
        old_strings = set()
        for d in old_group:
            old_strings.update(d["strings"])
        new_strings = set()
        for d in new_group:
            new_strings.update(d["strings"])

        if old_strings != new_strings and (old_strings or new_strings):
            content_modified.append({
                "struct_hash": sh,
                "count": matched,
                "old_strings": old_strings - new_strings,
                "new_strings": new_strings - old_strings,
                "type": old_group[0]["type"],
            })
        else:
            unchanged_count += matched

        # Handle count differences
        if len(new_group) > len(old_group):
            added.extend(new_group[len(old_group):])
        elif len(old_group) > len(new_group):
            removed.extend(old_group[len(new_group):])

    # Declarations with no structural match
    for sh in only_new_hashes:
        added.extend(new_by_struct[sh])
    for sh in only_old_hashes:
        removed.extend(old_by_struct[sh])

    return {
        "unchanged": unchanged_count,
        "content_modified": content_modified,
        "added": added,
        "removed": removed,
    }


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


def _cli_log(workdir, msg):
    """Append a timestamped message to the CLI debug log."""
    log_path = Path(workdir) / "cli-debug.log" if workdir else Path("cli-debug.log")
    timestamp = time.strftime("%H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")


# Module-level workdir reference for logging (set by run_claude_analysis)
_cli_log_workdir = None
# Module-level flag for SDK vs CLI mode (set by main() based on args)
_use_sdk = False
_use_batch = False


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
    # Extract model from cmd for logging
    model = "unknown"
    for i, arg in enumerate(cmd):
        if arg == "--model" and i + 1 < len(cmd):
            model = cmd[i + 1]
            break

    input_len = len(input_text) if input_text else 0
    _cli_log(_cli_log_workdir,
             f"START model={model} input_chars={input_len} timeout={timeout_secs}s")

    env = _claude_env()
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        t0 = time.time()
        try:
            result = subprocess.run(
                cmd, input=input_text, capture_output=True, text=True,
                timeout=timeout_secs, env=env
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            _cli_log(_cli_log_workdir,
                     f"TIMEOUT model={model} input_chars={input_len} "
                     f"elapsed={elapsed:.1f}s timeout={timeout_secs}s attempt={attempt + 1}")
            if attempt >= RATE_LIMIT_MAX_RETRIES:
                raise
            continue
        elapsed = time.time() - t0
        if result.returncode == 0 or not _is_rate_limited(result):
            # Try to record token usage from JSON responses
            if result.returncode == 0 and result.stdout.strip():
                try:
                    resp = json.loads(result.stdout)
                    if isinstance(resp, dict):
                        token_tracker.record(resp)
                        api_ms = resp.get("duration_api_ms", "?")
                        _cli_log(_cli_log_workdir,
                                 f"OK model={model} elapsed={elapsed:.1f}s "
                                 f"api_ms={api_ms} input_chars={input_len} "
                                 f"stdout_len={len(result.stdout)}")
                except (json.JSONDecodeError, ValueError):
                    _cli_log(_cli_log_workdir,
                             f"OK model={model} elapsed={elapsed:.1f}s "
                             f"input_chars={input_len} (non-JSON response)")
            if result.returncode != 0:
                _cli_log(_cli_log_workdir,
                         f"FAIL model={model} rc={result.returncode} "
                         f"elapsed={elapsed:.1f}s input_chars={input_len} "
                         f"stderr={result.stderr[:300]}")
                print(f"    CLI failed (rc={result.returncode}) after {elapsed:.1f}s: {result.stderr[:200]}")
            return result
        wait = _extract_wait_time(result) + 10  # small buffer
        _cli_log(_cli_log_workdir,
                 f"RATE_LIMITED model={model} elapsed={elapsed:.1f}s "
                 f"attempt={attempt + 1} wait={wait}s")
        if attempt < RATE_LIMIT_MAX_RETRIES:
            print(f"    Rate limited after {elapsed:.1f}s. Waiting {wait}s before retry ({attempt + 1}/{RATE_LIMIT_MAX_RETRIES})...")
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


# --- Anthropic SDK call functions ---

# JSON schema for per-hunk analysis (used by both CLI and SDK paths)
HUNK_ANALYSIS_TOOL = {
    "name": "analysis",
    "description": "Structured analysis of code changes",
    "input_schema": {
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
    }
}


def _resolve_model_id(model_name):
    """Resolve short model names (opus, sonnet, haiku) to full SDK model IDs."""
    return MODEL_ID_MAP.get(model_name, model_name)


def _compute_cost(model_id, usage, is_batch=False):
    """Compute cost in USD from token counts and model pricing."""
    pricing = MODEL_PRICING.get(model_id)
    if not pricing:
        # Try without version suffix
        for key, val in MODEL_PRICING.items():
            if key in model_id:
                pricing = val
                break
    if not pricing:
        return 0.0

    discount = pricing.get("batch_discount", 1.0) if is_batch else 1.0
    cost = (
        usage.input_tokens * pricing["input"] / 1_000_000 * discount
        + usage.output_tokens * pricing["output"] / 1_000_000 * discount
        + getattr(usage, "cache_read_input_tokens", 0) * pricing["cache_read"] / 1_000_000 * discount
        + getattr(usage, "cache_creation_input_tokens", 0) * pricing["cache_write"] / 1_000_000 * discount
    )
    return cost


def _api_call(prompt, model="sonnet", tool=None, timeout_secs=1200, is_batch=False):
    """Make a single Anthropic SDK call.

    Args:
        prompt: The user message text
        model: Model name (short or full ID)
        tool: Optional tool dict for structured output (forces tool_choice)
        timeout_secs: Request timeout in seconds
        is_batch: Whether this call is part of a batch (affects cost calculation)

    Returns:
        dict with keys: "result" (parsed output), "usage" (token counts), "error" (if failed)
    """
    model_id = _resolve_model_id(model)
    client = anthropic.Anthropic(
        max_retries=3,
        timeout=timeout_secs,
    )

    kwargs = {
        "model": model_id,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if tool:
        kwargs["tools"] = [tool]
        kwargs["tool_choice"] = {"type": "tool", "name": tool["name"]}

    t0 = time.time()
    try:
        response = client.messages.create(**kwargs)
    except anthropic.APITimeoutError:
        elapsed = time.time() - t0
        _cli_log(_cli_log_workdir, f"TIMEOUT model={model_id} elapsed={elapsed:.1f}s")
        return {"result": None, "error": "API timeout", "usage": None}
    except anthropic.APIError as e:
        elapsed = time.time() - t0
        _cli_log(_cli_log_workdir, f"API_ERROR model={model_id} elapsed={elapsed:.1f}s error={e}")
        return {"result": None, "error": f"API error: {e}", "usage": None}

    elapsed = time.time() - t0

    # Record token usage
    if response.usage:
        cost = _compute_cost(model_id, response.usage, is_batch=is_batch)
        with token_tracker._lock:
            if model_id not in token_tracker._usage:
                token_tracker._usage[model_id] = {
                    "inputTokens": 0, "outputTokens": 0,
                    "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0,
                    "costUSD": 0.0, "calls": 0,
                }
            entry = token_tracker._usage[model_id]
            entry["inputTokens"] += response.usage.input_tokens
            entry["outputTokens"] += response.usage.output_tokens
            entry["cacheReadInputTokens"] += getattr(response.usage, "cache_read_input_tokens", 0)
            entry["cacheCreationInputTokens"] += getattr(response.usage, "cache_creation_input_tokens", 0)
            entry["costUSD"] += cost
            entry["calls"] += 1

    _cli_log(_cli_log_workdir,
             f"OK model={model_id} elapsed={elapsed:.1f}s "
             f"input={response.usage.input_tokens} output={response.usage.output_tokens}")

    # Extract result
    if tool:
        # Structured output — find the tool_use block
        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_block:
            return {"result": tool_block.input, "error": None, "usage": response.usage}
        return {"result": None, "error": "No tool_use block in response", "usage": response.usage}
    else:
        # Free-form text output
        text_block = next((b for b in response.content if b.type == "text"), None)
        if text_block:
            return {"result": text_block.text, "error": None, "usage": response.usage}
        return {"result": None, "error": "No text block in response", "usage": response.usage}


def _api_build_request(custom_id, prompt, model="sonnet", tool=None):
    """Build a single request dict for the Batch API.

    Returns a dict suitable for client.messages.batches.create(requests=[...]).
    """
    model_id = _resolve_model_id(model)
    params = {
        "model": model_id,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if tool:
        params["tools"] = [tool]
        params["tool_choice"] = {"type": "tool", "name": tool["name"]}

    return {"custom_id": custom_id, "params": params}


def _batch_submit_and_poll(requests, workdir, progress_data=None):
    """Submit a Message Batch, poll for completion, return results.

    Args:
        requests: List of request dicts from _api_build_request()
        workdir: Working directory for progress file
        progress_data: Optional existing progress dict to update

    Returns:
        dict mapping custom_id -> result dict
    """
    client = anthropic.Anthropic()
    progress_path = Path(workdir) / "analysis-progress.json"

    # Submit batch
    print(f"  Submitting batch of {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch ID: {batch.id}")

    # Persist batch_id for resumability
    if progress_data is not None:
        progress_data["batch_id"] = batch.id
        progress_data["batch_status"] = "submitted"
        progress_path.write_text(json.dumps(progress_data, indent=2))

    # Poll for completion
    while True:
        status = client.messages.batches.retrieve(batch.id)
        counts = status.request_counts
        done = counts.succeeded + counts.errored + counts.expired + counts.canceled
        total = done + counts.processing
        print(f"  Batch {batch.id}: {done}/{total} complete "
              f"({counts.succeeded} ok, {counts.errored} errors, {counts.expired} expired)")

        if status.processing_status == "ended":
            break
        if status.processing_status in ("canceling",):
            print(f"  Batch is canceling, waiting...")
        time.sleep(30)

    # Collect results
    results = {}
    errors = 0
    for result in client.messages.batches.results(batch.id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            msg = result.result.message
            # Record token usage
            if msg.usage:
                cost = _compute_cost(msg.model, msg.usage, is_batch=True)
                with token_tracker._lock:
                    model_id = msg.model
                    if model_id not in token_tracker._usage:
                        token_tracker._usage[model_id] = {
                            "inputTokens": 0, "outputTokens": 0,
                            "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0,
                            "costUSD": 0.0, "calls": 0,
                        }
                    entry = token_tracker._usage[model_id]
                    entry["inputTokens"] += msg.usage.input_tokens
                    entry["outputTokens"] += msg.usage.output_tokens
                    entry["cacheReadInputTokens"] += getattr(msg.usage, "cache_read_input_tokens", 0)
                    entry["cacheCreationInputTokens"] += getattr(msg.usage, "cache_creation_input_tokens", 0)
                    entry["costUSD"] += cost
                    entry["calls"] += 1

            # Extract structured or text result
            tool_block = next((b for b in msg.content if b.type == "tool_use"), None)
            if tool_block:
                results[cid] = {"result": tool_block.input, "error": None}
            else:
                text_block = next((b for b in msg.content if b.type == "text"), None)
                results[cid] = {"result": text_block.text if text_block else None, "error": None}
        elif result.result.type == "errored":
            results[cid] = {"result": None, "error": f"Batch error: {result.result.error}"}
            errors += 1
        elif result.result.type == "expired":
            results[cid] = {"result": None, "error": "Request expired (24h batch limit)"}
            errors += 1
        else:
            results[cid] = {"result": None, "error": f"Unknown status: {result.result.type}"}
            errors += 1

    if errors:
        print(f"  Batch completed with {errors} errors out of {len(requests)} requests")

    # Update progress
    if progress_data is not None:
        progress_data["batch_status"] = "completed"
        progress_path.write_text(json.dumps(progress_data, indent=2))

    return results


# JSON schema for per-hunk Claude analysis (CLI format — kept for --legacy-cli)
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


def analyze_hunk_with_claude(hunk_content, file_path, string_changes, model="sonnet", tier="direct"):
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
    _cli_log(_cli_log_workdir,
             f"HUNK_SINGLE file={file_path} tier={tier} "
             f"hunk_chars={len(hunk_content)} prompt_chars={len(prompt)}")

    # SDK path
    if _use_sdk:
        try:
            resp = _api_call(prompt, model=model, tool=HUNK_ANALYSIS_TOOL, timeout_secs=1200)
            if resp["error"]:
                return {"error": resp["error"], **error_stub}
            return resp["result"]
        except Exception as e:
            return {"error": f"SDK error: {e}", **error_stub}

    # Legacy CLI path
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
        _cli_log(_cli_log_workdir,
                 f"HUNK_TIMEOUT file={file_path} tier={tier} "
                 f"hunk_chars={len(hunk_content)} prompt_chars={len(prompt)}")
        return {"error": "Claude CLI timed out", **error_stub}
    except (json.JSONDecodeError, KeyError) as e:
        return {"error": f"Failed to parse Claude response: {e}", **error_stub}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", **error_stub}


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

    # SDK path
    if _use_sdk:
        try:
            resp = _api_call(prompt, model=model, timeout_secs=180)
            if resp["error"]:
                return f"- {file_path}: summary failed ({resp['error']})"
            return resp["result"]
        except Exception as e:
            return f"- {file_path}: summary error ({e})"

    # Legacy CLI path
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


def fetch_previous_release_notes(tag):
    """Fetch release notes for a previous tag from GitHub.

    Returns the release body text with the cost/analysis section stripped,
    or None if the release doesn't exist or gh is unavailable.
    """
    try:
        result = subprocess.run(
            ["gh", "release", "view", tag, "--repo", REPO, "--json", "body", "--jq", ".body"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None
        body = result.stdout.strip()
        if not body:
            return None
        # Strip the "Analysis Cost" section (not useful for dedup)
        body = re.split(r'\n---\n\n\*\*Analysis Cost', body, maxsplit=1)[0]
        return body.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None


def generate_summary_with_claude(all_analyses, old_tag, new_tag, model="opus", voice_profile=None, previous_notes=None):
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
        previous_notes: Optional previous release notes for dedup

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
        return _generate_final_summary(analysis_text, old_tag, new_tag, model, voice_profile, previous_notes)

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

    return _generate_final_summary(combined, old_tag, new_tag, model, voice_profile, previous_notes)


def _generate_final_summary(analysis_text, old_tag, new_tag, model, voice_profile=None, previous_notes=None):
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

    previous_notes_section = ""
    if previous_notes:
        previous_notes_section = f"""
## Previous Release Notes
The following changes were already announced in the previous release ({old_tag}).
Do NOT repeat any of these items. Only describe changes that are NEW in this release.

<previous-release-notes>
{previous_notes}
</previous-release-notes>

"""

    prompt = f"""You are summarizing the changes between two releases of Claude Desktop, an Electron application.

**Old release:** {old_tag}
**New release:** {new_tag}

Below are per-hunk analyses of every changed code section. Each hunk has been individually analyzed
to deobfuscate minified names and describe what changed.

{analysis_text}
{previous_notes_section}## Instructions
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
1. Start with a H2 heading: ## {old_tag} → {new_tag}
2. A brief overview (2-3 sentences) of the highlights
3. Sections for notable changes, grouped naturally (features, fixes, dependency updates, etc.)
4. Keep descriptions clear and practical — explain what changed and why it matters

Do NOT include a confidence assessment table. If you're unsure about something, note it inline.
{voice_section}
Write in markdown format."""

    # SDK path
    if _use_sdk:
        try:
            resp = _api_call(prompt, model=model, timeout_secs=300)
            if resp["error"]:
                return f"# Summary Generation Failed\n\nSDK error: {resp['error']}"
            return resp["result"]
        except Exception as e:
            return f"# Summary Generation Failed\n\nUnexpected error: {e}"

    # Legacy CLI path
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


def run_claude_analysis(file_reports, workdir, old_tag, new_tag, summary_model="opus", voice_profile=None, max_workers=4):
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
    global _cli_log_workdir
    workdir = Path(workdir)
    _cli_log_workdir = workdir
    analysis_start_time = time.time()
    progress_path = workdir / "analysis-progress.json"
    summary_path = workdir / "summary.md"
    use_difft = get_has_difft()

    # Clear previous debug log
    log_path = workdir / "cli-debug.log"
    if log_path.exists():
        log_path.unlink()
    _cli_log(workdir, "=== Analysis started ===")

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

    # Build list of all tasks to analyze
    all_hunk_tasks = []
    skipped_files = []
    noise_hunks = 0
    ts_file_count = 0

    for fr in file_reports:
        if fr["status"] not in ("modified", "renamed"):
            continue

        file_path = fr["path"]

        # Tree-sitter path: build declaration-level tasks
        if fr.get("analysis_mode") == "treesitter":
            added_decls = fr.get("added_declarations", [])
            removed_decls = fr.get("removed_declarations", [])
            content_modified = fr.get("content_modified", [])
            unchanged_count = fr.get("unchanged_count", 0)

            # Skip if nothing to analyze
            if not added_decls and not removed_decls and not content_modified:
                skipped_files.append((file_path, f"no structural changes ({unchanged_count} unchanged)"))
                continue

            ts_file_count += 1

            # Build prompt chunks from declarations (cap at 40KB per chunk)
            new_strings = fr.get("new_strings", [])
            removed_strings = fr.get("removed_strings", [])
            string_section = ""
            if new_strings or removed_strings:
                string_section = "\n## String Changes (for context)\n"
                if new_strings:
                    string_section += "New strings:\n" + "\n".join(f"- {s}" for s in new_strings[:30]) + "\n"
                if removed_strings:
                    string_section += "Removed strings:\n" + "\n".join(f"- {s}" for s in removed_strings[:30]) + "\n"

            # Build declaration text blocks
            decl_blocks = []
            if content_modified:
                for cm in content_modified:
                    block = f"### Content-Modified ({cm['type']}, {cm['count']} declarations)\n"
                    if cm.get("new_strings"):
                        block += "New strings: " + ", ".join(f'`{s}`' for s in list(cm["new_strings"])[:10]) + "\n"
                    if cm.get("old_strings"):
                        block += "Removed strings: " + ", ".join(f'`{s}`' for s in list(cm["old_strings"])[:10]) + "\n"
                    decl_blocks.append(block)

            for d in added_decls:
                block = f"### Added: {d['type']}"
                if d.get("name"):
                    block += f" `{d['name']}`"
                block += f" ({d['size']} bytes)\n```javascript\n{d['text']}\n```\n"
                decl_blocks.append(block)

            for d in removed_decls:
                block = f"### Removed: {d['type']}"
                if d.get("name"):
                    block += f" `{d['name']}`"
                block += f" ({d['size']} bytes)\n```javascript\n{d['text']}\n```\n"
                decl_blocks.append(block)

            # Chunk into prompts at ~40KB
            chunks = []
            current_chunk = []
            current_size = 0
            for block in decl_blocks:
                if current_size + len(block) > 38000 and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                current_chunk.append(block)
                current_size += len(block)
            if current_chunk:
                chunks.append("\n".join(current_chunk))

            for chunk_idx, chunk_text in enumerate(chunks):
                header = f"""You are analyzing structural changes in a file from Claude Desktop (Electron app).
Tree-sitter AST comparison identified genuine structural changes — declarations
with identical structure (minified renames only) have been filtered out.

## File: {file_path}
## Stats: {unchanged_count} unchanged, {len(added_decls)} added, {len(removed_decls)} removed, {len(content_modified)} content-modified
## Chunk: {chunk_idx + 1}/{len(chunks)}
{string_section}
## Declarations
"""
                prompt = header + chunk_text + """

## Instructions
1. Describe what the changed/added/removed code does.
2. Describe the functional/behavioral delta — what's new, what's gone.
3. Provide a key_identifiers map of important minified names to guessed meaningful names.

Return your analysis as JSON."""

                sub_key = f"ts-{chunk_idx}"
                all_hunk_tasks.append({
                    "file": file_path,
                    "hunk_index": chunk_idx,
                    "sub_index": 0,
                    "progress_key": sub_key,
                    "hunk_header": f"tree-sitter chunk {chunk_idx + 1}/{len(chunks)}",
                    "content": prompt,
                    "tier": "treesitter",
                    "original_size": len(chunk_text),
                    "string_changes": {"new_strings": new_strings, "removed_strings": removed_strings},
                })

            continue

        # Text diff path (existing): build hunk-level tasks
        hunks_text = fr.get("hunks", "")
        if not hunks_text or not hunks_text.strip():
            continue

        # Skip binary files
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

    if ts_file_count:
        print(f"  Tree-sitter analyzed {ts_file_count} JS files")
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

    completed = 0
    remaining = len(tasks_to_run)
    progress_lock = threading.Lock()

    def _run_single(task, task_idx):
        """Execute a single hunk analysis and return (task_idx, task, result)."""
        if task["tier"] == "treesitter":
            # Tree-sitter tasks have pre-built prompts — send directly
            error_stub = {"description": "", "change_summary": "", "key_identifiers": {}}
            if _use_sdk:
                try:
                    resp = _api_call(task["content"], model="sonnet", tool=HUNK_ANALYSIS_TOOL, timeout_secs=1200)
                    if resp["error"]:
                        result = {"error": resp["error"], **error_stub}
                    else:
                        result = resp["result"]
                except Exception as e:
                    result = {"error": f"SDK error: {e}", **error_stub}
            else:
                try:
                    cli_result = _run_claude_cli(
                        ["claude", "-p", "--model", "sonnet",
                         "--output-format", "json", "--json-schema", HUNK_ANALYSIS_SCHEMA,
                         "--dangerously-skip-permissions"],
                        input_text=task["content"], timeout_secs=1200
                    )
                    if cli_result.returncode != 0:
                        result = {"error": f"CLI failed: {cli_result.stderr[:500]}", **error_stub}
                    else:
                        response = json.loads(cli_result.stdout)
                        result = _extract_structured_output(response)
                except Exception as e:
                    result = {"error": f"Error: {e}", **error_stub}
        else:
            result = analyze_hunk_with_claude(
                task["content"], task["file"], task["string_changes"],
                tier=task["tier"]
            )
        return task_idx, task, result

    print(f"  Processing {remaining} hunks with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for task_idx, task in enumerate(tasks_to_run):
            tier_label = f" [{task['tier']}]" if task["tier"] != "direct" else ""
            size_label = f" (preprocessed from {task['original_size']} chars)" if task["original_size"] > HUNK_SIZE_THRESHOLD else ""
            print(f"  Queued: {task['file']} hunk {task['progress_key']}{tier_label}{size_label}")
            future = executor.submit(_run_single, task, task_idx)
            futures[future] = task

        for future in as_completed(futures):
            task = futures[future]
            try:
                _task_idx, task, result = future.result()
            except Exception as e:
                error_stub = {"description": "", "change_summary": "", "key_identifiers": {}}
                result = {"error": f"Execution failed: {e}", **error_stub}

            with progress_lock:
                for task, result in [(task, result)]:
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
                # Log memory at each hunk completion including accumulated data size
                progress_json_size = len(json.dumps(progress)) if completed % 5 == 0 else 0
                if progress_json_size:
                    print(f"  [{completed}/{remaining}] hunks complete"
                          f" | progress_json={progress_json_size / 1024:.0f} KB")
                    _log_memory(f"Hunk {completed}/{remaining}")
                else:
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

    # Fetch previous release notes for dedup
    previous_notes = fetch_previous_release_notes(old_tag)
    if previous_notes:
        print(f"  Fetched previous release notes for dedup ({len(previous_notes)} chars)")
    else:
        print("  No previous release notes found (new items won't be deduped)")

    # Generate summary
    print(f"\n  Generating summary with {summary_model}...")
    summary = generate_summary_with_claude(all_analyses, old_tag, new_tag, model=summary_model, voice_profile=voice_profile, previous_notes=previous_notes)

    analysis_duration = time.time() - analysis_start_time
    install_section = _install_instructions_markdown()
    cost_section = token_tracker.summary_markdown(duration_secs=analysis_duration)
    summary_path.write_text(summary + install_section + cost_section)
    progress["status"] = "completed"
    progress_path.write_text(json.dumps(progress, indent=2))

    print(f"  Summary written to {summary_path}")

    # Print token usage summary
    print("\n  === Token Usage by Model ===")
    print(token_tracker.summary())


def main():
    # Force unbuffered stdout so progress lines appear before any SIGTERM
    sys.stdout.reconfigure(line_buffering=True)
    print("[compare-releases] Starting up...")
    _log_memory("startup")
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
    parser.add_argument("--model", default="opus",
                        help="Model for summary generation (default: opus)")
    parser.add_argument("--voice-profile-url",
                        help="URL to a voice profile .md file for styling the summary")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel workers (default: 4)")
    parser.add_argument("--batch", action="store_true",
                        help="Use Anthropic Batch API (default in CI, 50%% cheaper)")
    parser.add_argument("--legacy-cli", action="store_true",
                        help="Use claude CLI subprocess instead of SDK")
    parser.add_argument("--no-treesitter", action="store_true",
                        help="Disable tree-sitter structural analysis, use text diff instead")
    args = parser.parse_args()

    # Auto-detect worker count (OOM is fixed, 4 workers is safe)
    if args.max_workers is None:
        args.max_workers = 4

    # Auto-detect analysis mode
    if not args.legacy_cli and not HAS_SDK:
        if get_has_claude():
            print("  Note: anthropic SDK not installed, using claude CLI (install with: pip install anthropic)")
            args.legacy_cli = True
        else:
            print("Error: Neither ANTHROPIC_API_KEY (for SDK) nor claude CLI found.", file=sys.stderr)
            print("  Install SDK: pip install anthropic && export ANTHROPIC_API_KEY=...", file=sys.stderr)
            print("  Or install CLI: npm install -g @anthropic-ai/claude-code", file=sys.stderr)
            sys.exit(1)

    if not args.legacy_cli and HAS_SDK and not os.environ.get("ANTHROPIC_API_KEY"):
        if get_has_claude():
            print("  Note: ANTHROPIC_API_KEY not set, falling back to claude CLI")
            args.legacy_cli = True
        else:
            print("Error: ANTHROPIC_API_KEY environment variable required for SDK mode.", file=sys.stderr)
            sys.exit(1)

    # Default to batch mode in CI
    if not args.legacy_cli and not args.batch and os.environ.get("CI"):
        args.batch = True

    # Set module-level mode flags
    global _use_sdk, _use_batch, _use_treesitter
    _use_sdk = not args.legacy_cli
    _use_batch = args.batch
    _use_treesitter = HAS_TREESITTER and not args.no_treesitter
    if _use_sdk:
        mode_desc = "Batch API" if _use_batch else "SDK (sync)"
    else:
        mode_desc = "CLI (legacy)"
    ts_desc = "tree-sitter" if _use_treesitter else "text diff"
    print(f"  Analysis mode: {mode_desc} | Diff mode: {ts_desc}")

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

    # Step 2: Get app source (prefer reference-source tarball, fall back to AppImage)
    print("\n=== Step 2: Acquiring app source ===")

    # Try reference-source.tar.gz first (lightweight, avoids AppImage OOM on CI)
    old_ref = download_reference_source(old_tag, old_work)
    if old_ref:
        old_app = old_ref
        print(f"  Old: using reference source tarball")
    if args.new_appimage:
        new_ref = None  # Local AppImage explicitly provided, skip tarball
    else:
        new_ref = download_reference_source(new_tag, new_work)
    if new_ref:
        new_app = new_ref
        print(f"  New: using reference source tarball")

    # Fall back to AppImage extraction for releases without tarballs
    if not old_ref:
        print(f"  Old: no reference source, falling back to AppImage")
        old_appimage = download_appimage(old_tag, old_work)
        old_squashfs = extract_appimage(old_appimage, old_work)
        old_app = extract_asar(old_squashfs, old_app)
        if not args.keep:
            print(f"  Removing {old_squashfs} to free disk")
            shutil.rmtree(old_squashfs)
    _log_memory("After old source acquired")

    if not new_ref:
        if args.new_appimage:
            new_appimage = Path(args.new_appimage).resolve()
            if not new_appimage.exists():
                raise FileNotFoundError(f"Local AppImage not found: {new_appimage}")
            print(f"  New: using local AppImage: {new_appimage}")
        else:
            print(f"  New: no reference source, falling back to AppImage")
            new_appimage = download_appimage(new_tag, new_work)
        new_squashfs = extract_appimage(new_appimage, new_work)
        new_app = extract_asar(new_squashfs, new_app)
        if not args.keep:
            print(f"  Removing {new_squashfs} to free disk")
            shutil.rmtree(new_squashfs)
    _log_memory("After new source acquired")

    # Step 5: Beautify JS
    if _use_treesitter:
        print("\n=== Step 5: Skipped (tree-sitter handles minified JS directly) ===")
    else:
        print("\n=== Step 5: Beautifying JS ===")
        beautify_js(old_app)
        beautify_js(new_app)
    _log_memory("After step 5")

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
    _log_memory("After step 6 (compare + analyze)")

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
    _log_memory("After step 7 (generate reports)")

    # Step 8: Claude-powered analysis
    if not args.no_analyze:
        can_analyze = _use_sdk or get_has_claude()
        if can_analyze:
            print("\n=== Step 8: Claude-powered analysis ===")
            run_claude_analysis(file_reports, workdir, old_tag, new_tag, summary_model=args.model, voice_profile=voice_profile, max_workers=args.max_workers)
            _log_memory("After step 8 (Claude analysis)")
        else:
            print("\n=== Step 8: Skipped (no SDK or CLI available) ===")
            print("  Install anthropic SDK or claude CLI to enable AI-powered change analysis.")
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
