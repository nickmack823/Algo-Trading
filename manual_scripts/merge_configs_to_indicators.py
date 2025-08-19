import json
import pathlib
import re
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
ind_py = ROOT / "scripts/indicators.py"
cfg_path = ROOT / "scripts/generated_indicators/combined_configs.json"

# ---- Config: role -> list name in indicators.py
ROLE_TO_LIST = {
    "Trend": "additional_trend",
    "Momentum": "additional_momentum",
    "Baseline": "baseline_indicators",
    "ATR/Volatility": "additional_volatility",  # or "atr_indicators" if desired per-indicator
    # "Confirmation": decide per-block below (default Momentum):
    "Confirmation": "additional_trend",
}

# ---- Read files
text = cfg_path.read_text(encoding="utf-8", errors="ignore")
ind_text = ind_py.read_text(encoding="utf-8")


# ---- Extract existing names per list (to avoid dups)
def list_block(pattern, src):
    m = re.search(pattern, src, flags=re.S)
    return (m.group(0), m.start(), m.end()) if m else (None, -1, -1)


def grab_names(list_src):
    return set(re.findall(r'"name"\s*:\s*"([^"]+)"', list_src))


lists_patterns = {
    "additional_trend": r"additional_trend\s*=\s*\[\s*(.*?)\n\]",
    "additional_momentum": r"additional_momentum\s*=\s*\[\s*(.*?)\n\]",
    "baseline_indicators": r"baseline_indicators\s*=\s*\[\s*(.*?)\n\]",
    "additional_volatility": r"additional_volatility\s*=\s*\[\s*(.*?)\n\]",
    "atr_indicators": r"atr_indicators\s*=\s*\[\s*(.*?)\n\]",
}

list_src_cache = {}
list_names_cache = {}
for k, pat in lists_patterns.items():
    src, a, b = list_block(pat, ind_text)
    if src:
        list_src_cache[k] = src
        list_names_cache[k] = grab_names(src)
    else:
        list_src_cache[k] = None
        list_names_cache[k] = set()

# ---- Find { ... } blocks in the combined file (brace balancing)
blocks = []
brace = 0
start = None
for i, ch in enumerate(text):
    if ch == "{":
        if brace == 0:
            start = i
        brace += 1
    elif ch == "}":
        brace -= 1
        if brace == 0 and start is not None:
            blocks.append(text[start : i + 1])
            start = None


def clean_dict_literal(s):
    # Drop trailing comma after the last field if present and remove "role"
    # Turn "signal_function": "signal_x" into bare identifier: signal_x
    # Also same for "function"/"raw_function" if they were quoted by mistake
    # We’ll do simple regex rewrites that keep Python-literal validity.
    # 1) Remove "role": "..." entries safely
    s = re.sub(r',?\s*"role"\s*:\s*"(?:[^"]*)"\s*', "", s)
    # 2) Replace quoted function identifiers for signal_function/function/raw_function
    for key in ("signal_function", "function", "raw_function"):
        s = re.sub(rf'("{key}"\s*:\s*)"([A-Za-z_][A-Za-z0-9_\.]*)"', rf"\1\2", s)
    return s


def decide_destination(block_text):
    m = re.search(r'"role"\s*:\s*"([^"]+)"', block_text)
    role = m.group(1) if m else None
    dest = ROLE_TO_LIST.get(role or "", "additional_momentum")
    # Example tweaks for known items:
    if role == "Trend" and re.search(r'"name"\s*:\s*"SuperTrend"', block_text, re.I):
        dest = "additional_trend"
    if role == "ATR/Volatility" and re.search(
        r'"name"\s*:\s*"(ATR|NATR|FilteredATR)"', block_text
    ):
        dest = "atr_indicators"
    return dest, role


patches = {k: [] for k in lists_patterns.keys()}

for raw in blocks:
    dest, role = decide_destination(raw)
    if list_src_cache.get(dest) is None:
        # list not present in file; skip
        continue
    name_m = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    name = name_m.group(1) if name_m else "<NONAME>"
    if name in list_names_cache[dest]:
        continue  # skip duplicates
    cleaned = clean_dict_literal(raw).strip()
    patches[dest].append(cleaned)

if not any(patches.values()):
    print("No new indicators to insert.")
    raise SystemExit(0)

# ---- Make a backup
shutil.copy2(ind_py, ind_py.with_suffix(".py.bak"))

# ---- Insert before the closing ] of each target list
new_text = ind_text
for dest, pat in lists_patterns.items():
    if patches[dest]:
        # Find the exact block we matched earlier and append “, { ... }” entries before the final closing bracket
        m = re.search(dest + r"\s*=\s*\[\s*", new_text)
        if not m:
            continue
        # Find the matching closing bracket for this list
        # naive approach: find the first ']' after the start of the list definition line
        start_pos = m.end()
        end_pos = new_text.find("]\n", start_pos)
        if end_pos == -1:
            continue
        insertion = ""
        for entry in patches[dest]:
            insertion += ",\n    " + entry.replace("\n", "\n    ")
        new_text = new_text[:end_pos] + insertion + new_text[end_pos:]

ind_py.write_text(new_text, encoding="utf-8")
print("Done. Patched indicators into:", [k for k, v in patches.items() if v])
