# scripts/tools/mt4_indicator_importer.py
"""
MT4 → Project importer (interactive, safe-writes, AI-assisted summary matching)

Per file:
  1) Convert .mq4 to a calc() function (pure pandas/numpy) → NEW file in scripts/generated_imports/
  2) Read entire indicators_summary.txt and AI-pick the best-matching block → infer Role + notes
  3) Generate a signal() function + a paste-ready config snippet → NEW files in scripts/generated_imports/

Key behaviors:
- Run with NO args. It scans for *.mq4 under common folders and processes one file at a time.
- NEVER overwrites: outputs are timestamped.
- If it cannot infer a valid role, it SKIPS (no default to Momentum).
- After each file, press Enter to continue / 's' to skip / 'q' to quit.

Requires: pip install openai (>=1.0)
"""

import difflib
import glob
import json
import os
import pathlib
import re
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# 0) OPENAI API KEY — inline default (env var still supported)
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = json.loads("resources/openapi_key.json")["key"]
OPENAI_MODEL = "gpt-5"

try:
    from openai import OpenAI  # openai>=1.0.0
except Exception:
    print("ERROR: `openai` package not found. Install with: pip install openai")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# 1) Paths & constants (tweak as needed)
# ──────────────────────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve()
# Repo root assumed to be parent of 'scripts'
REPO_ROOT = HERE.parents[2] if len(HERE.parents) >= 2 else HERE.parent
REPO_ROOT = REPO_ROOT / "Algo Trading"
SCRIPTS_DIR = REPO_ROOT / "scripts"

SUMMARY_FILE_CANDIDATES = [
    REPO_ROOT / "resources/text_files/indicators_summary.txt",
]

# Likely spots for MT4 source files
SCAN_DIRS = [
    REPO_ROOT / "resources/mq4",
]

OUT_DIR = SCRIPTS_DIR / "generated_indicators"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Valid roles we recognize (NO default to Momentum)
VALID_ROLES = {
    "ATR",
    "Baseline",
    "C1",
    "C2",
    "Confirmation",
    "VolumeIndicator",
    "Exit",
    # also allow class-style group names (mapped below)
    "Trend",
    "Momentum",
    "Volatility",
    "Volume",
}

# Tag list (mirrors scripts.config constants that signal_functions uses)
TAG_LIST = [
    "BULLISH_SIGNAL",
    "BEARISH_SIGNAL",
    "BULLISH_TREND",
    "BEARISH_TREND",
    "NEUTRAL_TREND",
    "OVERBOUGHT",
    "OVERSOLD",
    "NO_SIGNAL",
    "HIGH_VOLUME",
    "LOW_VOLUME",
    "HIGH_VOLATILITY",
    "LOW_VOLATILITY",
    "INCREASING_VOLATILITY",
    "DECREASING_VOLATILITY",
    "STABLE_VOLATILITY",
    "INCONCLUSIVE",
]


# ──────────────────────────────────────────────────────────────────────────────
# 2) Utility
# ──────────────────────────────────────────────────────────────────────────────
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_api_key() -> OpenAI:
    # accept OPENAI_API_KEY or OPENAI_OPENAI_API_KEY for safety
    key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_OPENAI_API_KEY")
        or OPENAI_API_KEY
    )
    if not key or not key.startswith("sk-"):
        print(
            "ERROR: OpenAI API key missing. Set OPENAI_API_KEY or edit OPENAI_API_KEY in this file."
        )
        sys.exit(1)
    os.environ["OPENAI_API_KEY"] = key
    return OpenAI()


def _smart_decode_mq4(data: bytes) -> str:
    """
    Robustly decode MT4/MT5 source bytes to text.
    - Detects BOM for UTF-16/UTF-8.
    - Heuristic for UTF-16LE/BE if no BOM (based on NUL-byte ratio & position).
    - Falls back through common encodings; never returns interleaved \\x00.
    """
    # BOM-based fast paths
    if data.startswith(b"\xff\xfe"):  # UTF-16 LE BOM
        return data.decode("utf-16-le", errors="strict")
    if data.startswith(b"\xfe\xff"):  # UTF-16 BE BOM
        return data.decode("utf-16-be", errors="strict")
    if data.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
        return data.decode("utf-8-sig", errors="strict")

    # Heuristic: many NULs => UTF-16 without BOM
    nul_ratio = data.count(b"\x00") / max(1, len(data))
    if nul_ratio > 0.2:
        # Check which side tends to be NUL to guess endianness
        even_nuls = sum(1 for i in range(0, len(data), 2) if data[i] == 0)
        odd_nuls = sum(1 for i in range(1, len(data), 2) if data[i] == 0)
        try_order = (
            ["utf-16-le", "utf-16-be"]
            if even_nuls > odd_nuls
            else ["utf-16-be", "utf-16-le"]
        )
        for enc in try_order:
            try:
                text = data.decode(enc, errors="strict")
                # Sanity: decoded text should not contain NULs
                if "\x00" not in text:
                    return text
            except Exception:
                pass

    # Try a robust list of single-byte encodings and UTF-16 (auto) last
    for enc in ["utf-8", "cp1252", "latin-1", "iso-8859-1", "utf-16"]:
        try:
            text = data.decode(enc, errors="strict")
            if "\x00" not in text:
                return text
        except Exception:
            continue

    # Final fallback: replace errors; ensure no NULs remain
    text = data.decode("utf-8", errors="replace")
    return text.replace("\x00", "")


def clean_code_for_ai(code: str) -> str:
    """
    Remove null bytes and normalize line endings before sending to the AI.
    """
    # Remove null characters and excessive whitespace
    code = code.replace("\x00", "")
    # Normalize line endings
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return code.strip()


import pathlib
from typing import Optional


def read_text(p: pathlib.Path, txt_out_dir: Optional[pathlib.Path] = None) -> str:
    """
    If given an .mq4 (or other non-.txt) file, create a UTF-8 .txt copy, read it, then delete the copy.
    If given a .txt file, read it directly without deleting.
    """
    p = pathlib.Path(p)
    is_temp_copy = p.suffix.lower() != ".txt"

    if is_temp_copy:
        raw = p.read_bytes()
        text = _smart_decode_mq4(raw)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "".join(ch for ch in text if ch == "\t" or ch == "\n" or ord(ch) >= 32)

        if txt_out_dir:
            txt_out_dir = pathlib.Path(txt_out_dir)
            txt_out_dir.mkdir(parents=True, exist_ok=True)
            txt_path = txt_out_dir / (p.stem + ".txt")
        else:
            txt_path = p.with_suffix(".txt")

        txt_path.write_text(text, encoding="utf-8")
    else:
        # Already a .txt — just read it
        txt_path = p
        text = txt_path.read_text(encoding="utf-8", errors="ignore")

    try:
        return text
    finally:
        if is_temp_copy and txt_path.exists():
            txt_path.unlink()


def write_new(name_prefix: str, ext: str, content: str) -> pathlib.Path:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name_prefix).strip("_")
    out = OUT_DIR / f"{safe}.{ext.lstrip('.')}"
    out.write_text(content, encoding="utf-8")
    return out


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_").lower()


def camelcase(s: str) -> str:
    # Split into alphanumeric word chunks
    words = re.findall(r"[A-Za-z0-9]+", s)
    # Capitalize first letter of each word, except keep all-uppercase parts as-is
    return "".join(w if w.isupper() else w.capitalize() for w in words)


def list_mq4_files() -> List[pathlib.Path]:
    files = []
    for d in SCAN_DIRS:
        if d.exists():
            files += [
                pathlib.Path(p)
                for p in glob.glob(str(d / "**" / "*.mq4"), recursive=True)
            ]
    # Dedup preserve order
    seen = set()
    unique = []
    for p in files:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _split_blocks(summary_text: str) -> List[str]:
    return re.split(r"\n\s*\n", summary_text)


def _name_words(name: str) -> List[str]:
    # very simple word extraction; keep digits/letters
    return re.findall(r"[A-Za-z0-9]+", name.lower())


def simple_presence_block(summary_text: str, indicator_name: str) -> Optional[str]:
    blocks = _split_blocks(summary_text)
    if not blocks:
        return None
    words = _name_words(indicator_name)
    if not words:
        return None

    best_score, best_block = -1.0, None
    for b in blocks:
        bl = b.lower()
        matches = sum(1 for w in words if w in bl)
        if matches > best_score:
            best_score, best_block = matches, b

    return best_block if (best_block and best_score > 0) else None


# ──────────────────────────────────────────────────────────────────────────────
# 3) AI helpers
# ──────────────────────────────────────────────────────────────────────────────
def chat(
    client: OpenAI,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> str:
    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (r.choices[0].message.content or "").strip()


# --- Full-text AI fallback: send ENTIRE summaries file to extract best block ---
def ai_extract_best_block_full(
    client: OpenAI, summary_text: str, indicator_name: str
) -> Optional[str]:
    prompt = f"""You will be given the full text of an indicator summaries file.

Task: Find the SINGLE best-matching description for the indicator name: "{indicator_name}".
• If multiple candidates exist (aliases/variants), choose the most relevant by name and usage.
• RETURN the exact matching block VERBATIM (no commentary). 
• If you cannot find any match, return exactly: NONE

FULL SUMMARIES:
{summary_text}
"""
    extracted = chat(
        client,
        system="You extract verbatim text blocks from long documents. Output ONLY the block, or NONE.",
        user=prompt,
        temperature=1,
    ).strip()

    if not extracted or extracted.upper() == "NONE" or len(extracted) < 20:
        return None
    return extracted


def prompt_convert_mq4_to_python(ind_name: str, mq4_code: str) -> str:
    fn = f"{camelcase(ind_name)}"
    # raw = f"raw_{camelcase(ind_name)}"
    return f"""Convert the following MQL4 indicator into pure Python using ONLY numpy/pandas (no TA-Lib).

Data:
- Input: a pandas DataFrame 'df' with columns ['Timestamp','Open','High','Low','Close','Volume'] (index arbitrary)
- Output:
    import numpy as np
    import pandas as pd

If the output is a single line or buffer:
    def {fn}(df: pd.DataFrame, (other params)) -> pd.Series:
        \"\"\"Return the primary indicator line(s) aligned to df.index; vectorized, handle NaNs, stable defaults. For the other parameters, use what is specifically required for this specific indicator.\"\"\"

If multiple buffers/lines exist for the output:
    def {fn}(df: pd.DataFrame, (other params)) -> pd.DataFrame:
        \"\"\"Return ALL lines as columns with clear denoted names; aligned to df.index. Parameters handled same as above.\"\"\"

Rules:
- Assume the input is a DataFrame with columns ['Timestamp','Open','High','Low','Close','Volume'] (index arbitrary).
- Vectorize (rolling/ewm, no per-row loops unless necessary).
- Be explicit with defaults in 'params'.
- Do NOT print or plot; no side effects.
- Make sure the function name is capitalized (e.g. 'schaff trend' -> 'SchaffTrend)
- Use clear column names if returning a DataFrame.
- Align outputs with df.index and preserve length with NaNs at warmup.
- Do NOT include unused parameters.
- Do NOT include Pandas Series as a parameter.

ONLY return valid Python code (imports + functions). No commentary, no backticks.

MQL4 SOURCE:
{mq4_code}
"""


def prompt_signal_from_summary(
    ind_name: str, calc_function: str, role: str, summary_block: str
) -> str:
    fn = f"signal_{slugify(ind_name)}"
    calc = f"{camelcase(ind_name)}"
    return f"""Write a Python signal function for '{ind_name}' based on the following calculation function using NNFX-style semantics:

Calculation function:
{calc_function}

\"\"\"The output from the calc function {calc}(...) will be the INPUT PARAMETER of this signal function, used to generate signals. Return a list of tags (choose from provided constants).\"\"\"

Constraints:
- import pandas as pd
- from scripts.config import {", ".join(TAG_LIST)}
- Signature:

    If the calculation function outputs a DataFrame:
    def {fn}(df: pd.DataFrame) -> list[str]:

    If the calculation function outputs a Series:
    def {fn}(series: pd.Series) -> list[str]:
        
- Always:
  • Consider the last two bars for cross/threshold decisions.
  • Robust to NaNs (no exceptions when the series is too short; fall back to [NEUTRAL_TREND] or [NO_SIGNAL, NEUTRAL_TREND]).
  • Append exactly one final trend tag (BULLISH_TREND, BEARISH_TREND, or NEUTRAL_TREND).

- Allowed tags: {TAG_LIST}
- Role for this indicator: '{role}'
- Use the interpretation below (verbatim block from summaries) to craft the logic:

- Do NOT use if statements to interpret either DataFrame OR Series, only one based on the calculation function's output.

{summary_block}

ONLY return valid Python code (imports + function). No commentary, no backticks.
"""


def prompt_config_snippet(
    ind_name: str, calc_function: str, role: str, description: str
) -> str:
    slug = slugify(ind_name)
    camel = camelcase(ind_name)
    # We'll emit a paste-ready snippet (including suggested imports path)
    return f"""# Create this dict in Python code based on this calculation function:

    Calculation function:
    {calc_function}

{{
  "name": "{ind_name}",
  "function": {camel},
  "signal_function": signal_{slug},
  "raw_function": {camel},
  "description": {json.dumps(description or (ind_name + " indicator."))},
  "parameters": {{"period": 14}},  # adjust keys and values based on the calculation function's defaults, using all params
  "parameter_space": {{"period": [7, 14, 21, 30]}}, # adjust this based on the calculation function, using all params in a reasonable range. Don't include values that'd cause errors, for parameters like offset, alpha, things like this.
  "role": "{role}"  # Valid pools: Baseline, Confirmation, VolumeIndicator, ATR/Volatility, Exit, Trend, Momentum
}}

- Ensure the function name is capitalized (e.g. 'schaff trend' -> 'SchaffTrend').
- Ensure the raw_function is the same as the function name.
- Ensure the parameters exactly match the defaults of the 'function'.
- Ensure the parameter_space includes only values in a reasonable range and only for parameters that are typically tweaked without issue.
"""


# ──────────────────────────────────────────────────────────────────────────────
# 4) Summary matching (local fuzzy + AI fallback with chunking)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SummaryMatch:
    role: Optional[str]
    description: str
    block: str


ROLE_NORMALIZE = {
    "baseline": "Baseline",
    "atr": "ATR",
    "atr/volatility": "ATR",
    "volatility": "ATR",
    "vol": "ATR",
    "volume": "VolumeIndicator",
    "volumeindicator": "VolumeIndicator",
    "exit": "Exit",
    "c1": "C1",
    "c2": "C2",
    "confirmation": "Confirmation",
    "trend": "Trend",
    "momentum": "Momentum",
    # alias cleanups
    "volind": "VolumeIndicator",
    "vol-indicator": "VolumeIndicator",
}

ROLE_PATTERN = re.compile(r"(?i)\brole\s*[:\-]\s*([A-Za-z0-9_/ ]+)")
DESC_PATTERN = re.compile(r"(?i)\bdescription\s*[:\-]\s*(.+)")


def normalize_role(r: Optional[str]) -> Optional[str]:
    if not r:
        return None
    key = r.strip().lower()
    norm = ROLE_NORMALIZE.get(key, r.strip())
    # unify common tokens
    if norm.lower() == "volatility":
        norm = "ATR"  # NNFX-volatility slot acts as ATR/vol
    if norm.lower() == "volume":
        norm = "VolumeIndicator"
    return norm


def local_fuzzy_block(summary_text: str, indicator_name: str) -> Optional[str]:
    # Split into blocks separated by a blank line
    blocks = re.split(r"\n\s*\n", summary_text)
    if not blocks:
        return None

    # Tokenize the indicator name into individual words (lowercased)
    words = re.findall(r"\w+", indicator_name.lower())

    candidates = []
    for b in blocks:
        block_lower = b.lower()
        # Count how many of the words appear in this block
        matches = sum(1 for w in words if w in block_lower)
        # Simple scoring: fraction of words found
        score = matches / len(words) if words else 0
        candidates.append((score, b))

    # Pick the block with the highest match score
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_block = candidates[0]

    return best_block if best_score > 0 else None


def ai_pick_block(
    client: OpenAI, summary_text: str, indicator_name: str
) -> Optional[str]:
    # Chunk large summaries; pick best chunk, then extract best-matching block
    chunks: List[str] = []
    step = 12000  # chars per chunk (safe for 4o-mini)
    for i in range(0, len(summary_text), step):
        chunks.append(summary_text[i : i + step])

    ratings = []
    for idx, chunk in enumerate(chunks):
        q = f"""You will be given an indicator name and a chunk of a summaries file.
Return a single integer 0-10: how likely this chunk contains the best matching description
for the indicator (consider common synonyms / alternate names).

Indicator: {indicator_name}

Chunk:
{chunk[:6000]}
"""
        score_txt = chat(
            client,
            "You rate relevance only. Output ONLY an integer 0-10.",
            q,
            temperature=1,
        ).strip()
        try:
            score = int(re.findall(r"\d+", score_txt)[0])
        except Exception:
            score = 0
        ratings.append((score, idx))

    if not ratings:
        return None
    ratings.sort(reverse=True)
    best_idx = ratings[0][1]
    best_chunk = chunks[best_idx]

    extract = chat(
        client,
        "You extract verbatim text blocks.",
        f"""From the text below, extract the single best matching indicator description for: "{indicator_name}".
- If multiple candidates, choose the most relevant by name/alias/usage.
- Return the extracted block verbatim (no commentary). If none, return exactly: NONE

TEXT:
{best_chunk}
""",
        temperature=1,
    )
    block = extract.strip()
    if block.upper() == "NONE" or len(block) < 20:
        return None
    return block


def parse_role_and_description(block: str) -> Tuple[Optional[str], str]:
    role = None
    m = ROLE_PATTERN.search(block)
    if m:
        role = normalize_role(m.group(1))
    desc = ""
    d = DESC_PATTERN.search(block)
    if d:
        desc = d.group(1).strip()

    # Heuristic role by keywords if not explicitly labeled
    if not role:
        low = block.lower()
        for k, v in ROLE_NORMALIZE.items():
            if re.search(rf"\b{k}\b", low):
                role = v
                break
        if not role:
            # try common words
            if "baseline" in low:
                role = "Baseline"
            elif any(t in low for t in ["atr", "volatility", "volatil"]):
                role = "ATR"
            elif "volume" in low:
                role = "VolumeIndicator"
            elif re.search(r"\bc1\b", low):
                role = "C1"
            elif re.search(r"\bc2\b", low):
                role = "C2"
            elif "exit" in low:
                role = "Exit"

    return role, desc


# --- Replace your get_indicator_summary(...) with this version ---
def get_indicator_summary(
    client: OpenAI, indicator_name: str, summary_text: str
) -> Optional[SummaryMatch]:
    # 1) Simple “word presence” hardcoded search
    block = simple_presence_block(summary_text, indicator_name)

    # 2) If nothing found, send the FULL summaries text to AI to extract best block
    if not block:
        print("   • No block found by word presence, AI extracting best block...")
        block = ai_extract_best_block_full(client, summary_text, indicator_name)

    if not block:
        return None

    role, desc = parse_role_and_description(block)
    return SummaryMatch(role=role, description=desc, block=block)


# ──────────────────────────────────────────────────────────────────────────────
# 5) Main per-file pipeline
# ──────────────────────────────────────────────────────────────────────────────
def process_file(
    client: OpenAI, mq4_path: pathlib.Path, summary_text: Optional[str]
) -> None:
    print(f"\n➡️  Processing: {mq4_path}")
    ind_name = mq4_path.stem
    mq4_code = read_text(mq4_path)
    mq4_code = clean_code_for_ai(mq4_code)

    # 1) Generate calc function file via AI
    calc_module_basename = f"{camelcase(ind_name)}_calc"

    # Check if we've already generated this file
    if (OUT_DIR / f"{calc_module_basename}.py").is_file():
        print("   • SKIP calc function: already exists.")
        # Read python code
        calc_function = read_text(OUT_DIR / f"{calc_module_basename}.py")
    else:
        calc_function = chat(
            client,
            "You are an expert quant developer. You write correct, vectorized pandas code.",
            prompt_convert_mq4_to_python(ind_name, mq4_code),
            temperature=1,
        )
        if not calc_function.strip().startswith("import"):
            # Defensive: sometimes models preface text; keep only last code block-ish
            calc_function = calc_function.strip()

        calc_out = write_new(calc_module_basename, "py", calc_function)
        print(f"   • Calc function → {calc_out}")

    # 2) Find matching summary (AI + local fuzzy). If none/invalid role → SKIP (no defaults)
    if not summary_text:
        print("   • SKIP signal/config: indicators_summary.txt not found.")
        return

    match = get_indicator_summary(client, ind_name, summary_text)
    if not match:
        print("   • SKIP signal/config: no matching summary found by AI/local search.")
        return

    if not match.role or match.role not in VALID_ROLES:
        print(
            f"   • SKIP signal/config: inferred role invalid or missing ({match.role!r})."
        )
        return

    # 3) Generate signal function
    sig_module_basename = f"{slugify(ind_name)}_signal"
    if (OUT_DIR / f"{sig_module_basename}.py").is_file():
        print("   • SKIP signal function: already exists.")
        # Read python code
        signal_function = read_text(OUT_DIR / f"{sig_module_basename}.py")
    else:
        signal_function = chat(
            client,
            "You write pragmatic signal functions compatible with an NNFX-style backtester.",
            prompt_signal_from_summary(
                ind_name, calc_function, match.role, match.block
            ),
            temperature=1,
        )
        sig_out = write_new(sig_module_basename, "py", signal_function)
        print(f"   • Signal function → {sig_out}")

    # 4) Generate paste-ready config snippet
    cfg_basename = f"{slugify(ind_name)}_config"
    if (OUT_DIR / f"{cfg_basename}.py").is_file():
        print("   • SKIP config: already exists.")
        # Read python code
        cfg_code = read_text(OUT_DIR / f"{cfg_basename}.py")
    else:
        cfg_code = chat(
            client,
            "You emit small, correct Python dicts/snippets for configuration blocks.",
            prompt_config_snippet(
                ind_name, calc_function, match.role, match.description
            ),
            temperature=1,
        )
        cfg_out = write_new(cfg_basename, "py", cfg_code)
        print(f"   • Config snippet → {cfg_out}")

    print("   ✅ Done for this file.")


# ──────────────────────────────────────────────────────────────────────────────
# 6) Entrypoint (interactive: one file at a time)
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("MT4 Indicator Importer  —  interactive & safe-writes\n")
    client = set_api_key()

    # Load entire summaries file (we handle chunking internally)
    summary_path = next((p for p in SUMMARY_FILE_CANDIDATES if p.exists()), None)
    summary_text = read_text(summary_path) if summary_path else None
    if summary_text:
        print(f"Found summaries at: {summary_path}")
    else:
        print("WARNING: indicators_summary.txt not found; will generate calc only.\n")

    # Locate .mq4 files
    mq4_files = list_mq4_files()
    if not mq4_files:
        print("No .mq4 files were found. Place your MT4 indicators under one of:")
        for d in SCAN_DIRS:
            print("  -", d)
        print("\nThen re-run this script.")
        return

    # Process one-at-a-time with user control
    total = len(mq4_files)
    for i, path in enumerate(mq4_files, start=1):
        print(f"\n[{i}/{total}] Next file: {path}")
        # choice = (
        #     input("Press Enter to process, 's' to skip, 'q' to quit: ").strip().lower()
        # )
        choice = "run"
        if choice == "q":
            print("Exiting.")
            break
        if choice == "s":
            print("   • Skipped.")
            continue

        try:
            process_file(client, path, summary_text)
        except Exception as e:
            print(f"   • ERROR while processing this file: {e}")

        # Small pause to keep outputs readable
        time.sleep(0.2)

    print("\nAll done. Check outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
