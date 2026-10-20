"""Turn result files into the tables, numbers, figures and supplementary files of the manuscript.

    python -m twws.cli report --config configs/paper_reduced_ext.yaml --out work/report

Nothing in the manuscript is typed by hand: every reported value is written here from the JSON
files under ``<work_dir>/results`` and ``<work_dir>/corpus``:

    generated/numbers.tex          one macro per value, used in the text as \\R{key}
    generated/numbers.json         the same values (used to build the response letter)
    generated/tables/*.tex         complete table environments
    generated/figures/*.pdf        figures (each with a CSV of the plotted values)
    generated/supplementary/*.csv  numerical values behind every figure and table
    generated/report_manifest.json SHA-256 of every result file that was read

A table whose experiment has not been run is replaced by a visible placeholder, so a missing
result can never be mistaken for a reported one.
"""
from __future__ import annotations

import csv
import inspect
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from ..metrics import clopper_pearson
from ..stages import workspace
from ..utils import ensure_dir, get_logger, read_json, sha256_file, write_json

log = get_logger("twws.report")

ORACLE_SHORT = {"stat": "Statistical", "signature": "Signature", "cluster": "Clustering", "transfer": "Transfer",
                "domain": "Domain", "rate": "Rate", "beacon": "Regularity"}
ORACLE_LABEL = {"stat": "Statistical thresholds", "signature": "Signature (Snort)", "cluster": "Density clustering",
                "transfer": "Transfer classifier", "domain": "Domain heuristics",
                "rate": "Repeated connections", "beacon": "Regular repetition"}
FAMILY_ORDER = ["BruteForce", "DoS", "DDoS", "Web", "Infiltration", "Bot"]
FAMILY_LABEL = {"BruteForce": "Brute force", "DoS": "DoS", "DDoS": "DDoS", "Web": "Web attacks",
                "Infiltration": "Infiltration", "Bot": "Botnet", "Benign": "Benign", "PortScan": "Port scan",
                "Ambiguous": "Ambiguous"}
# the same names as they read in running text ("the ... oracles", "recall is highest for ...")
ORACLE_TEXT = {"stat": "statistical-threshold", "signature": "signature", "cluster": "clustering", "transfer": "transfer",
               "domain": "domain-heuristic", "rate": "repeated-connection", "beacon": "regularity"}
ORACLE_NOUN = {"stat": "statistical thresholds", "signature": "signatures", "cluster": "density clustering",
               "transfer": "transfer classifier", "domain": "domain heuristics",
               "rate": "repeated-connection oracle", "beacon": "regularity oracle"}


def _source_name(name: str, table: dict) -> str:
    """Name of a label-model source; merged oracles ('stat+transfer') are joined with ' + '."""
    return " + ".join(table.get(p, p) for p in str(name).split("+"))
FAMILY_TEXT = {"BruteForce": "brute-force attacks", "DoS": "DoS attacks", "DDoS": "DDoS attacks", "Web": "web attacks",
               "Infiltration": "infiltration", "Bot": "botnet traffic", "PortScan": "port scans"}


ATTACK_TEXT = {"FTP-BruteForce": "FTP brute force", "SSH-Bruteforce": "SSH brute force", "DoS-GoldenEye": "DoS GoldenEye",
               "DoS-Slowloris": "DoS Slowloris", "DoS-SlowHTTPTest": "DoS SlowHTTPTest", "DoS-Hulk": "DoS Hulk",
               "DDoS-LOIC-HTTP": "DDoS LOIC-HTTP", "DDoS-LOIC-UDP": "DDoS LOIC-UDP", "DDoS-HOIC": "DDoS HOIC",
               "BruteForce-Web": "web brute force", "BruteForce-XSS": "cross-site scripting", "SQL-Injection": "SQL injection",
               "Infiltration": "infiltration", "Bot": "botnet (Zeus/Ares)",
               # CICIDS2017 (configs/attack_schedule_ids2017.yaml)
               "FTP-Patator": "FTP-Patator", "SSH-Patator": "SSH-Patator", "DoS-slowloris": "DoS slowloris",
               "DoS-Slowhttptest": "DoS Slowhttptest", "DDoS-LOIT": "DDoS LOIT", "Web-BruteForce": "web brute force",
               "Web-XSS": "cross-site scripting", "Web-SQLi": "SQL injection",
               "Infiltration-CoolDisk": "infiltration (Cool Disk)", "Infiltration-Meta": "infiltration (Metasploit)",
               "Botnet-ARES": "botnet (Ares)", "PortScan": "port scan", "Heartbleed": "Heartbleed"}


def join_words(items, last: str = "and") -> str:
    items = [str(x) for x in items]
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} {last} {items[1]}"
    return ", ".join(items[:-1]) + f", {last} " + items[-1]


SUBSPACE_LABEL = {"statistical": "Per-flow and host statistics", "temporal": "Windowed host aggregates",
                  "latent": "Latent (autoencoder)"}
# categorical palette (validated: adjacent and all-pairs CVD separation on a light surface)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ============================================================================ formatting helpers
def _flatten(d, prefix: str = "") -> dict:
    """Nested dictionaries as {dotted key: value}; lists are kept as values."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


MINUS = "$-$"     # typographic minus; unlike a hyphen, LaTeX does not break a line after it


def signed(x: float, nd: int = 3) -> str:
    """A difference with its sign; a difference that rounds to zero is written without one (0.000, not -0.000)."""
    r = round(float(x), nd)
    if r == 0:
        return f"{0.0:.{nd}f}"
    return f"{r:+.{nd}f}".replace("-", MINUS)


def tex(s) -> str:
    """Escape plain text for LaTeX."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def tt(s) -> str:
    return r"\texttt{" + tex(s) + "}"


def isnum(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not (isinstance(x, float) and math.isnan(x))


def f(x, nd: int = 3) -> str:
    if not isnum(x):
        return "--"
    t = f"{float(x):.{nd}f}"
    if t.startswith("-"):
        return t[1:] if float(t) == 0 else MINUS + t[1:]               # no "-0.000" from rounding
    return t


def pct(x, nd: int = 1) -> str:
    return f(100 * float(x), nd) if isnum(x) else "--"


def integer(x) -> str:
    return f"{int(round(float(x))):,}".replace(",", "{,}") if isnum(x) else "--"


def pm(s: dict | None, nd: int = 3) -> str:
    """mean +- standard deviation over seeds."""
    if not s or not isnum(s.get("mean")):
        return "--"
    if s.get("n", 1) < 2 or not isnum(s.get("std")):
        return f(s["mean"], nd)
    sd = float(s["std"])
    # a standard deviation that rounds to zero but is not zero is written as "< 0.001" (not as "0.000", which would
    # claim that all seeds agree to that precision); an exact zero stays "0.000"
    sd_txt = f"$<${10 ** -nd:.{nd}f}" if 0 < sd < 0.5 * 10 ** -nd else f(sd, nd)
    return f"{f(s['mean'], nd)} \\ensuremath{{\\pm}} {sd_txt}"


def pm_pct(s: dict | None, nd: int = 2) -> str:
    """mean +- standard deviation of a proportion, in percent, with the '<' rule of pm() for the deviation."""
    if not s or not isnum(s.get("mean")):
        return "--"
    if s.get("n", 1) < 2 or not isnum(s.get("std")):
        return pct(s["mean"], nd)
    sd = 100 * float(s["std"])
    sd_txt = f"$<${10 ** -nd:.{nd}f}" if 0 < sd < 0.5 * 10 ** -nd else f"{sd:.{nd}f}"
    return f"{pct(s['mean'], nd)} $\\pm$ {sd_txt}"


def ties(values: dict, best: bool = True, nd: int = 3) -> list[str]:
    """Keys whose value equals the largest (or smallest) value at the printed precision, in family order."""
    if not values:
        return []
    target = round(float((max if best else min)(values.values())), nd)
    return [k for k in fam_sorted(values) if round(float(values[k]), nd) == target]


def ci(s: dict | None, nd: int = 3) -> str:
    """95% t-interval of the seed mean."""
    if not s or not isnum(s.get("mean")) or not isnum(s.get("ci")):
        return "--"
    lo, hi = _ci_ends(s)
    return f"[{f(lo, nd)}, {f(hi, nd)}]"


def _ci_ends(s: dict) -> tuple[float, float]:
    """Ends of the 95% t-interval of the seed mean; for a proportion (mean in [0, 1]) clipped to [0, 1]."""
    m, c = float(s["mean"]), float(s["ci"])
    lo, hi = m - c, m + c
    if 0.0 <= m <= 1.0:
        lo, hi = max(lo, 0.0), min(hi, 1.0)
    return lo, hi


def sci(x) -> str:
    if not isnum(x):
        return "--"
    x = float(x)
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    if -3 <= e <= 3:
        return f"{x:.3g}"
    m = x / 10 ** e
    if abs(m - 1.0) < 1e-9:
        return f"\\ensuremath{{10^{{{e}}}}}"
    return f"\\ensuremath{{{m:.1f}\\times10^{{{e}}}}}"


def param(v) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return tex(v)


class Numbers:
    """Collects the values used in running text."""

    def __init__(self) -> None:
        self.text: dict[str, str] = {}
        self.raw: dict[str, object] = {}

    def put(self, key: str, value, text: str | None = None) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        self.raw[key] = value.item() if isinstance(value, (np.integer, np.floating)) else value
        self.text[key] = str(value) if text is None else text

    def num(self, key, x, nd=3):
        if isnum(x):
            self.put(key, float(x), f(x, nd))

    def pct(self, key, x, nd=1):
        if isnum(x):
            self.put(key, float(x), pct(x, nd))

    def int(self, key, x):
        if isnum(x):
            self.put(key, int(round(float(x))), integer(x))

    def summary(self, prefix: str, s: dict | None, nd=3):
        """mean, std, interval ends and 'mean +- std' of one seed summary."""
        if not s or not isnum(s.get("mean")):
            return
        self.num(prefix, s["mean"], nd)
        if isnum(s.get("std")):
            sd = float(s["std"])
            # same rule as pm(): a deviation that rounds to zero but is not zero is "<0.001", an exact zero "0.000"
            self.put(prefix + ".std", sd, f"$<${10 ** -nd:.{nd}f}" if 0 < sd < 0.5 * 10 ** -nd else f(sd, nd))
            self.put(prefix + ".pm", float(s["mean"]), pm(s, nd))
        if isnum(s.get("ci")):
            lo, hi = _ci_ends(s)
            self.num(prefix + ".lo", lo, nd)
            self.num(prefix + ".hi", hi, nd)
            self.put(prefix + ".ci", float(s["ci"]), ci(s, nd))
        # lowest and highest seed (kept if a builder has set these keys with its own rule)
        for e in ("min", "max"):
            if isnum(s.get(e)) and f"{prefix}.{e}" not in self.text:
                self.num(f"{prefix}.{e}", s[e], nd)

    def write(self, out: Path) -> None:
        lines = ["% Generated by twws.report.build -- do not edit.",
                 r"\makeatletter"]
        for k in sorted(self.text):
            lines.append(r"\@namedef{twws@" + k + "}{" + self.text[k] + "}")
        lines.append(r"\makeatother")
        (out / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(out / "numbers.json", {"text": self.text, "raw": self.raw})


# numbers of the labels of the compiled manuscript (read from its .aux by build_report), so that the plain-text copies
# of the tables can print "Table 16" where the LaTeX source has \ref{tab:main}
REF_NUMBERS: dict[str, str] = {}

GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ", "epsilon": "ε", "eta": "η",
         "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
         "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "pm": "±", "times": "×", "le": "≤", "ge": "≥",
         "leq": "≤", "geq": "≥", "approx": "≈", "infty": "∞", "cdot": "·", "ldots": "…", "dots": "…"}


def _brace_arg(s: str, i: int) -> tuple[str, int]:
    """The balanced {...} group that starts at s[i] (after optional spaces): its content and the index after it."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s) or s[i] != "{":
        return "", i
    depth, j = 0, i
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def latex_to_text(s: str) -> str:
    """Plain text of a table cell (for the comma-separated copies of the tables)."""
    s = str(s).strip()
    for a, b in ((MINUS, "-"), ("{,}", ","), ("\\%", "%"), ("\\&", "&"), ("\\_", "_"), ("\\#", "#"),
                 ("\\{", "\x01"), ("\\}", "\x02"), ("\\,", " "), ("\\;", " "), ("\\ ", " "), ("~", " "),
                 ("--", "–"), ("``", "“"), ("''", "”")):
        s = s.replace(a, b)
    out, i = [], 0
    while i < len(s):                        # unwrap every \command{argument}; keep the text of its last argument
        if s[i] == "\\":
            m = re.match(r"\\([A-Za-z]+)\*?", s[i:])
            if not m:
                i += 1
                continue
            name, i = m.group(1), i + m.end()
            if name in GREEK:
                out.append(GREEK[name])
                continue
            args = []
            while i < len(s) and s[i] == "{":
                a, i = _brace_arg(s, i)
                args.append(a)
            if name == "ref" and args:
                out.append(REF_NUMBERS.get(args[-1], "?"))
            elif name in ("cite", "label", "tabtag", "tabcolor", "hspace", "vspace", "rule", "phantom"):
                pass
            elif args:
                out.append(latex_to_text(args[-1]))
            continue
        out.append(s[i])
        i += 1
    s = "".join(out)
    s = re.sub(r"_\{([^{}]*)\}", r"_\1", s)
    s = re.sub(r"\b([A-Za-z])_(\d)\b", r"\1\2", s)          # L_2 -> L2
    s = s.replace("$", "").replace("{", "").replace("}", "").replace("\x01", "{").replace("\x02", "}")
    return re.sub(r"\s+", " ", s).strip()


def latex_row_cells(row: str) -> list[str]:
    """Split one LaTeX table row (without the trailing double backslash) into its cells; a multicolumn cell over n
    columns becomes its text followed by n - 1 empty cells, so that every row has one cell per column."""
    cells, cur, depth, i = [], [], 0, 0
    while i < len(row):
        ch = row[i]
        if ch == "\\" and i + 1 < len(row):
            cur.append(row[i:i + 2])
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    cells.append("".join(cur))
    out = []
    for c in cells:
        c = c.strip()
        m = re.match(r"\\multicolumn\s*\{(\d+)\}", c)
        out.append(latex_to_text(c))
        if m:
            out.extend([""] * (int(m.group(1)) - 1))
    return out


# Wording of the published tables.  The submitted manuscript of this work is called "the first version" in the
# variant names of the result files; the article itself calls that configuration the "initial design".
_DISPLAY = [
    (r"Five oracles of the first version \(signature and fan-out rules revised\); equal-vote majority; refinement: "
     r"the method of the first version, rebuilt \((\d+) seeds\)",
     r"Initial design: five oracles (signature and fan-out rules revised), equal-vote majority, refinement (\1 seeds)"),
    (r"Five oracles of the first version \(rules revised\); label model of this revision, single start, no merging; "
     r"refinement", "Five oracles (rules revised); label model, single start, no merging; refinement"),
    (r"From the oracles of the first version to the selected configuration",
     "From the initial design to the selected configuration"),
    (r"Rows in italics: the definitions of the first version", "Rows in italics: the initial definitions of two rules"),
    (r"with the semi-supervised refinement of the first version", "with the refinement of the initial design"),
    (r"what the refinement of the first version would do to these labels is given in Step~2 of "
     r"Section~\\ref\{sec:method\} and in Section~\\ref\{sec:oracles\}\.",
     r"what the refinement of the initial design would do to these labels is given in the text."),
    (r"the two settings of Step~5 that", "the two settings of the selection stage that"),
    (r", first version\)", ", initial rule)"),
    (r" \(first version\)", " (initial rule)"),
    (r"\(as in the first version\)", "(as in the initial design)"),
    (r"label model of this revision", "label model"),
    (r"(?i)the first version", "the initial design"),
    (r"first version", "initial design"),
]


def display_text(s: str) -> str:
    """Table text as printed: no reference to the submitted manuscript of this work (see _DISPLAY)."""
    for a, b in _DISPLAY:
        s = re.sub(a, b, s)
    return s


class Table:
    """A complete LaTeX table environment written to ``tables/<name>.tex``."""

    def __init__(self, name: str, caption: str, colspec: str, header: list[str] | list[list[str]],
                 size: str = r"\small", notes: str | None = None, star: bool = False, resize: bool = False,
                 long: bool = False) -> None:
        self.name, self.caption, self.colspec, self.size = name, caption, colspec, size
        self.header = header if header and isinstance(header[0], list) else [header]
        self.rows: list[str] = []
        self.notes, self.star, self.resize = notes, star, resize
        self.long = long          # a table that can exceed one page is set as a longtable

    def row(self, *cells) -> "Table":
        self.rows.append(" & ".join(str(c) for c in cells) + r" \\")
        return self

    def rule(self) -> "Table":
        self.rows.append(r"\midrule")
        return self

    def raw(self, line: str) -> "Table":
        self.rows.append(line)
        return self

    def _render_long(self) -> str:
        ncol = len(self.header[0]) if not isinstance(self.header[0], str) else self.header[0].count("&") + 1
        head = [h if isinstance(h, str) else " & ".join(h) + r" \\" for h in self.header]
        body = [r"\begingroup", self.size, rf"\tabcolor{{{self.name}}}",
                rf"\begin{{longtable}}{{{self.colspec}}}",
                rf"\caption{{\tabtag{{{self.name}}}{self.caption}}}\label{{tab:{self.name}}}\\",
                r"\toprule", *head, r"\midrule", r"\endfirsthead",
                rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\tablename\ \thetable\ (continued)}}\\",
                r"\toprule", *head, r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
        body += self.rows
        body += [r"\end{longtable}"]
        if self.notes:
            body.append(r"\par\noindent\begin{minipage}{\linewidth}\footnotesize " + self.notes + r"\end{minipage}\par")
        body.append(r"\endgroup")
        return "\n".join(body) + "\n"

    def render(self) -> str:
        if self.long:
            return self._render_long()
        env = "table*" if self.star else "table"
        body = [rf"\begin{{{env}}}[!htbp]", r"\centering", self.size,
                rf"\caption{{\tabtag{{{self.name}}}{self.caption}}}\label{{tab:{self.name}}}",
                rf"\tabcolor{{{self.name}}}"]
        body.append(r"\begin{adjustbox}{max width=\linewidth}")
        body.append(rf"\begin{{tabular}}{{{self.colspec}}}")
        body.append(r"\toprule")
        for h in self.header:
            body.append(h if isinstance(h, str) else " & ".join(h) + r" \\")
        body.append(r"\midrule")
        body += self.rows
        body += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
        if self.notes:
            body.append(r"\par\vspace{2pt}\begin{minipage}{\linewidth}\footnotesize " + self.notes + r"\end{minipage}")
        body.append(rf"\end{{{env}}}")
        return "\n".join(body) + "\n"

    def write(self, out: Path) -> None:
        (out / "tables" / f"{self.name}.tex").write_text("% Generated by twws.report.build -- do not edit.\n"
                                                        + display_text(self.render()), encoding="utf-8")
        self.write_csv(out)

    def write_csv(self, out: Path) -> None:
        """The cells of the table as plain text (supplementary/tables/<name>.csv), one line per row of the table,
        and its caption in supplementary/tables/index.csv.  Every row must have as many cells as the header."""
        import csv
        d = out / "supplementary" / "tables"
        d.mkdir(parents=True, exist_ok=True)
        end = "\\" + "\\"                    # the end of a table row
        lines = [display_text(h if isinstance(h, str) else " & ".join(h) + " " + end) for h in self.header] + \
            [display_text(r) for r in self.rows]
        rows = []
        for ln in lines:
            s = ln.rstrip()
            if not s.endswith(end) or s.lstrip().startswith(("\\midrule", "\\cmidrule", "\\addlinespace")):
                continue
            rows.append(latex_row_cells(s[:-2]))
        width = max((len(r) for r in rows[:len(self.header)]), default=0)
        bad = [r for r in rows if len(r) != width]
        if bad:
            raise ValueError(f"table {self.name}: {len(bad)} row(s) with a number of cells other than {width}: {bad[0]}")
        with open(d / f"{self.name}.csv", "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
        idx = d / "index.csv"
        head = ["file", "table", "caption"]
        old = list(csv.reader(open(idx, encoding="utf-8"))) if idx.exists() else [head]
        old = [r for r in old[1:] if r and r[0] != f"{self.name}.csv"]
        num = REF_NUMBERS.get(f"tab:{self.name}")
        old.append([f"{self.name}.csv", f"Table {num}" if num else "", latex_to_text(display_text(self.caption))])
        with open(idx, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([head] + sorted(old))


# What each table of the manuscript contains.  A table whose experiment has not been run is written as a placeholder
# with this caption, so that a draft never shows a caption that belongs to another table.
TABLE_PURPOSE = {
    "manifest": "Retained captures per day (file-level manifest in the code repository).",
    "corpus": "Composition of the flow stream extracted from the retained captures and of the evaluation corpus drawn from it.",
    "search": "Search space of the detector and selected configuration.",
    "oracle_stats": "Behavior of each oracle on the training split, measured against ground truth.",
    "oracle_family": "Share of the flows of each class that an oracle votes malicious.",
    "oracle_pairs": "Pairwise behavior of the oracles on training flows where both vote.",
    "oracle_joint": "Number of oracles that are wrong on the same training flow, against the distribution implied by independent errors.",
    "aggregation": "Quality of the aggregated pseudo-labels against ground truth for each aggregation rule.",
    "trust": "Trust score against the accuracy of the pseudo-label.",
    "label_model": "Oracle error rates estimated by the label model without labels, against the rates measured with ground truth.",
    "oracle_ablation": "Detector trained on pseudo-labels from different oracle sets and aggregation rules.",
    "abl_summary": "Summary of the ablation study: one variant per stage, with the supervised references.",
    "pseudo_labels": "Quality of the pseudo-labels against ground truth: after aggregation, after refinement, and on the test split.",
    "pl_attacks": "Share of the malicious training flows of each attack that the pseudo-labels and the two oracles that carry them label malicious.",
    "main": "Detection performance of the complete pipeline on the test split, with seed-to-seed and bootstrap intervals.",
    "main_seeds": "Per-seed detection performance of the complete pipeline on the test split.",
    "confusion": "Test-set confusion matrix of the complete pipeline.",
    "family": "Detection coverage per attack family.",
    "abl_objective": "Step-by-step ablation of the training objective.",
    "abl_selection": "Ablation of the feature-selection stage.",
    "abl_refinement": "Ablation of the semi-supervised refinement and of its thresholds.",
    "abl_other": "Ablation of preprocessing and feature subspaces, and sensitivity to the oracle settings.",
    "fs": "Feature-selection methods under the same classifier, hyperparameters, and partition.",
    "loao": "Leave-one-attack-out: recall on the held-out family.",
    "protocols": "Partition protocols, each evaluated with and without the host-level features.",
    "protocol_splits": "Size and composition of the splits under each partition protocol.",
    "baselines": "Proposed detector and end-to-end baselines under one protocol.",
    "baselines_extra": "Threshold-free metrics, false-positive rate, and accuracy of the compared models.",
    "baseline_cfg": "Configurations of the end-to-end baselines selected by the shared search, model size, and training time.",
    "cross": "Cross-dataset evaluation without adaptation to the target dataset.",
    "cross_family": "Recall per attack family on the target dataset.",
    "second_dataset": "The method applied to the second dataset without its labels: recall per family of the pseudo-labels "
                      "and of the detector.",
    "main_steps": "From the oracles of the first version to the selected configuration: training labels and detector "
                  "of each configuration.",
    "adversarial": "Recall on attacker-sent malicious test flows under traffic-level evasion, without and with augmentation of the training set.",
    "adversarial_family": "Recall per attack family under traffic-level evasion, without and with augmentation of the training set.",
    "adv_oracles": "Votes of the two repetition oracles on the traffic of attacker-controlled endpoints under connection spacing.",
    "abl_selection_extra": "Feature selection with the size of the mutual-information filter and the number of folds changed.",
    "shap": "Global feature attribution of the detector (mean absolute TreeSHAP contribution).",
    "shap_subspace": "Share of the attribution carried by each feature subspace.",
    "shap_family": "Operator-facing explanation per class.",
    "latent": "Reading of the latent features used by the detector through named features.",
    "online": "Service time of the online path for one micro-batch of finalized flow records on a single worker.",
    "load": "Load test of the online path.",
    "qos": "QoS under enforcement, by trace-driven emulation on every flow record of the retained captures.",
    "qos_hold": "QoS under enforcement for different hold-down periods.",
    "normalization": "Detection quality with training-frozen and with sliding-window scaling statistics.",
    "runtime": "Measured wall-clock time per pipeline stage.",
    "memory": "Memory measured during the experiments.",
    "selected_features": "Features retained by the stability-aware selection.",
    "supplementary_files": "Supplementary files with the numerical values behind the tables and figures.",
}


def placeholder(out: Path, name: str, what: str = "") -> None:
    what = TABLE_PURPOSE.get(name, what)
    txt = ("% Generated by twws.report.build -- do not edit.\n"
           r"\begin{table}[!htbp]\centering" "\n"
           rf"\caption{{\tabtag{{{name}}}{tex(what)}}}\label{{tab:{name}}}" "\n"
           r"\fbox{\parbox{0.9\linewidth}{\centering\textbf{RESULT NOT AVAILABLE.} The experiment that fills this table "
           r"has not been run; no value is reported.}}" "\n"
           r"\end{table}" "\n")
    (out / "tables" / f"{name}.tex").write_text(txt, encoding="utf-8")


def save_csv(out: Path, name: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(out / "supplementary" / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def fam_sorted(names) -> list[str]:
    names = list(names)
    return [x for x in FAMILY_ORDER if x in names] + sorted(x for x in names if x not in FAMILY_ORDER and x != "Benign")


def metric_cells(s: dict, keys=("precision", "recall", "f1"), nd=3) -> list[str]:
    return [pm(s.get(k), nd) for k in keys]


# Every attack family is listed in every per-family table, also a family with a handful of test flows, so that the
# totals add up (reviewer comment R2.5).  The sampling variation of a recall over the flows of one family is stated in
# the note under the table as an exact binomial interval: families with at least LARGE_FAMILY flows are summarized by
# their widest half-interval, every smaller family is named with its interval.
LARGE_FAMILY = 1000


def binomial_note(items, count_text: str) -> tuple[str, float | None, dict]:
    """Note under a per-family table: exact 95% (Clopper-Pearson) interval of each recall over the evaluated flows.

    ``items`` are (key, text label, flows, detected flows).  Returns the note, the widest half-interval among the
    families with at least LARGE_FAMILY flows (None if there is none), and {key: (lo, hi)} for every family."""
    large, small, ends = [], [], {}
    for key, lab, n, k in items:
        if not isnum(n) or int(n) <= 0 or not isnum(k):
            continue
        n, k = int(n), int(round(float(k)))
        lo, hi = clopper_pearson(k, n)
        ends[key] = (lo, hi)
        (large if n >= LARGE_FAMILY else small).append((lab, n, k, lo, hi))
    parts, hw = [], None
    if large:
        hw = max(max(k / n - lo, hi - k / n) for _, n, k, lo, hi in large)
        parts.append(f"at most $\\pm${f(hw)} for the families with at least {integer(LARGE_FAMILY)} flows")
    for lab, n, k, lo, hi in small:
        parts.append(f"from {f(lo)} to {f(hi)} for {lab} ({integer(k)} of {integer(n)} flows detected)")
    if not parts:
        return "", hw, ends
    note = ("Sampling variation over the test flows (exact 95\\% binomial interval of the recall, " + count_text
            + "; Section~\\ref{sec:metrics}): " + "; ".join(parts) + ".")
    return note, hw, ends


def _family_text(fm: str) -> str:
    return FAMILY_TEXT.get(fm, FAMILY_LABEL.get(fm, fm))


# ============================================================================ figures
def _mpl():
    import logging

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6, "lines.markersize": 5,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
    })
    return plt


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / "figures" / f"{name}.pdf")
    fig.savefig(out / "figures" / f"{name}.png", dpi=200)
    import matplotlib.pyplot as plt

    plt.close(fig)


# ============================================================================ builders
def b_setup(cfg, ws, R, N: Numbers, out: Path) -> None:
    """Configuration constants quoted in the text, software versions and hardware."""
    c = cfg
    N.put("cfg.corpus.seed", int(c.corpus.seed))
    N.put("cfg.corpus.sample_seed", int(c.corpus.get("sample_seed", 42)))
    bh = c.corpus.background_hosts_per_day
    N.put("cfg.corpus.background_hosts", bh if isinstance(bh, str) else int(bh))
    mb = c.corpus.get("max_background_member_bytes")
    N.put("cfg.corpus.background_size_clause", int(mb or 0),
          f" whose uncompressed size does not exceed {float(mb) / 2 ** 30:g}~GiB" if mb else "")
    N.int("cfg.corpus.family_cap", c.corpus.family_cap)
    if c.corpus.get("benign_share") is not None:      # None = keep all benign flows (not used for the paper)
        N.put("cfg.corpus.benign_share", float(c.corpus.benign_share), pct(c.corpus.benign_share, 0))
    N.int("cfg.corpus.max_rows", c.corpus.max_rows)
    N.put("cfg.corpus.n_days", len(c.corpus.days))
    N.put("cfg.corpus.dedup_tol", float(c.corpus.dedup_tolerance_s), f"{float(c.corpus.dedup_tolerance_s):g}")
    N.put("cfg.extract.entropy_bytes", int(c.extract.entropy_max_bytes), integer(c.extract.entropy_max_bytes))
    N.put("cfg.extract.entropy_bytes_raw", int(c.extract.entropy_max_bytes))        # as typed on the command line
    # constants of the code that the text quotes
    import inspect

    from ..baselines import DL_ES_MAX_ROWS, GRAPH_SCORE_CHUNK
    from ..labels import load_schedule
    from ..preprocess import RobustImputerScaler
    from ..weak import dawid_skene

    N.int("cfg.features.impute_fit_rows", inspect.signature(RobustImputerScaler.__init__).parameters["fit_rows"].default)
    N.int("cfg.refine.max_fit_rows", c.refinement.get("max_fit_rows", 500_000))
    N.int("cfg.exp.dl_es_rows", DL_ES_MAX_ROWS)
    N.int("cfg.exp.graph_chunk", GRAPH_SCORE_CHUNK)
    N.put("cfg.label_model.max_iter", int(inspect.signature(dawid_skene).parameters["n_iter"].default))
    slack = float(load_schedule(c.corpus.schedule).get("window_slack_seconds", 300))
    N.put("cfg.label.window_slack_min", slack / 60.0, f"{slack / 60.0:g}")
    N.put("cfg.snort.alert_slack", float(c.snort.get("alert_slack_s", 1.0)), f"{float(c.snort.get('alert_slack_s', 1.0)):g}")
    o = c.oracles
    N.put("cfg.stat.syn_ratio", float(o.stat.syn_ratio), f"{float(o.stat.syn_ratio):g}")
    N.put("cfg.stat.bpp_low", float(o.stat.bpp_low), f"{float(o.stat.bpp_low):g}")
    N.put("cfg.stat.bpp_high", float(o.stat.bpp_high), integer(o.stat.bpp_high))
    N.put("cfg.stat.guard", float(o.stat.guard), pct(o.stat.guard, 0))
    N.put("cfg.cluster.eps", float(o.cluster.eps), f"{float(o.cluster.eps):g}")
    N.put("cfg.cluster.min_samples", int(o.cluster.min_samples))
    N.put("cfg.cluster.pca", int(o.cluster.pca_components))
    N.put("cfg.transfer.hi", float(o.transfer.hi), f"{float(o.transfer.hi):g}")
    N.put("cfg.transfer.lo", float(o.transfer.lo), f"{float(o.transfer.lo):g}")
    N.put("cfg.transfer.trees", int(o.transfer.n_estimators))
    N.put("cfg.transfer.depth", int(o.transfer.max_depth))
    N.put("cfg.domain.tls_port_min", int(o.domain.tls_port_min))
    N.put("cfg.domain.fanout_min", int(o.domain.fanout_min))
    fq = o.domain.get("fanout_quantile")
    if fq is not None:
        N.put("cfg.domain.fanout_quantile", float(fq), f"{float(fq):g}")
        N.put("cfg.domain.fanout_quantile_pct", float(fq) * 100, f"{float(fq) * 100:g}")
    classes = c.snort.get("attack_classtypes")
    N.put("cfg.snort.n_attack_classes", len(classes) if classes is not None else 0)
    if classes is not None:
        N.put("cfg.snort.attack_classes", ", ".join(classes), ", ".join(tex(x) for x in classes))   # roman: breaks at hyphens
    N.put("cfg.refine.high_trust", float(c.refinement.high_trust), f"{float(c.refinement.high_trust):g}")
    N.put("cfg.refine.p_flip", float(c.refinement.p_flip), f"{float(c.refinement.p_flip):g}")
    N.put("cfg.refine.trust", float(c.refinement.refined_trust), f"{float(c.refinement.refined_trust):g}")
    s = c.selection
    N.put("cfg.sel.mi_top", int(s.mi_top))
    N.put("cfg.sel.k", int(s.k))
    N.put("cfg.sel.folds", int(s.n_folds))
    N.put("cfg.sel.theta", float(s.theta), f"{float(s.theta):g}")
    N.put("cfg.sel.borderline_low", float(s.borderline_low), f"{float(s.borderline_low):g}")
    N.int("cfg.sel.max_rows", s.max_rows)
    N.int("cfg.sel.rf_max_samples", s.rf_max_samples)
    # feature dimensions follow from the code and the configuration (b_main repeats them from the result file)
    from ..features import HOST_FEATURES, N_AGG, STAT_FEATURES

    d1, d2, d3 = len(STAT_FEATURES) + len(HOST_FEATURES), N_AGG * len(c.features.windows), int(c.features.latent_dim)
    N.put("feat.d_stat", d1)
    N.put("feat.d_temporal", d2)
    N.put("feat.d_latent", d3)
    N.put("feat.d_total", d1 + d2 + d3)
    N.put("cfg.features.latent_dim", int(c.features.latent_dim))
    N.put("cfg.features.ae_hidden", int(c.features.ae_hidden))
    N.put("cfg.features.ae_epochs", int(c.features.ae_epochs))
    N.int("cfg.features.ae_max_rows", c.features.ae_max_rows)
    N.put("cfg.features.impute_clusters", int(c.features.impute_clusters))
    N.put("cfg.search.trials", int(c.search.n_trials))
    N.put("cfg.search.random", int(c.search.n_random))
    N.put("cfg.search.tpe", int(c.search.n_trials) - int(c.search.n_random))
    N.put("cfg.search.seed", int(c.search.seed))
    N.int("cfg.search.rows", c.search.search_rows)
    N.int("cfg.search.val_rows", c.search.val_rows)
    N.put("cfg.search.rounds", int(c.search.num_boost_round))
    N.put("cfg.search.patience", int(c.search.early_stopping_rounds))
    d = c.model.defaults
    N.put("cfg.model.rounds", int(d.num_boost_round))
    N.put("cfg.model.patience", int(d.early_stopping_rounds))
    N.put("cfg.model.warmup", int(d.warmup_rounds))
    N.put("cfg.model.hess_floor", float(d.hess_floor), sci(d.hess_floor))
    N.int("cfg.model.es_rows", c.model.es_max_rows)
    e = c.experiments
    N.put("cfg.exp.seeds", ", ".join(str(x) for x in e.seeds))
    N.put("cfg.exp.n_seeds", len(e.seeds))
    N.put("cfg.exp.ablation_seeds", ", ".join(str(x) for x in e.ablation_seeds))
    N.put("cfg.exp.n_ablation_seeds", len(e.ablation_seeds))
    sec = e.get("secondary_seeds") or e.seeds
    N.put("cfg.exp.secondary_seeds", ", ".join(str(x) for x in sec))
    N.put("cfg.exp.n_secondary_seeds", len(sec))
    # the configuration profile that produced the results; for a profile that modifies another one ("base:"), the
    # sentence of the manuscript lists every value in which the two differ
    cp, bp = c.get("_config_path"), c.get("_base_config_path")
    pf = Path(cp).as_posix().split("/")[-1] if cp else "unknown"
    N.put("cfg.profile_file", pf, tex(pf))
    if bp:
        from ..config import load_config as _load_base
        b = _load_base(bp)
        while b.get("_base_config_path"):      # compare with the full profile at the root of the chain of profiles
            bp = b.get("_base_config_path")
            b = _load_base(bp)
        N.put("cfg.base_file", Path(bp).name, tex(Path(bp).name))
        known = {  # dotted key -> (description, formatter)
            "corpus.max_rows": ("at most {v} instead of {b} flows in the corpus", integer),
            "corpus.family_cap": ("at most {v} instead of {b} flows per attack family", integer),
            "selection.max_rows": ("{v} instead of {b} training flows for the selection stage", integer),
            "selection.rf_max_samples": ("{v} instead of {b} flows per tree of its random forests", integer),
            "experiments.dl_max_train_rows": ("{v} instead of {b} training flows for the neural baselines", integer),
            "experiments.secondary_seeds": ("{v} instead of {b} seeds for the secondary experiments", len),
            "oracles.enabled": ("{v} instead of {b} oracles, the added ones for repeated and for regular connections",
                                len),
            "oracles.aggregation": ("the label model with dependent oracles merged", str),
            "refinement.enabled": ("no refinement of low-trust flows", str),
        }
        diffs, other, method_diffs = [], [], []
        method_keys = ("oracles.enabled", "oracles.aggregation", "refinement.enabled")
        flat_c, flat_b = _flatten(dict(c)), _flatten(dict(b))
        for k in sorted(set(flat_c) | set(flat_b)):
            last = k.split(".")[-1]
            if k.startswith(("_", "report.")) or last.endswith(("_dir", "_path")) or last in ("work_dir", "capture_dir"):
                continue                          # file locations are not settings of the method or the protocol
            if k == "refinement.enabled" and flat_c.get(k, True) == flat_b.get(k, True):
                continue                          # absent means enabled
            if k == "oracles.group_jaccard":
                continue                          # part of the aggregation rule, described with it
            if flat_c.get(k) != flat_b.get(k):
                if k in known:
                    text_, fmt = known[k]
                    vv_, bb_ = flat_c.get(k), flat_b.get(k)
                    txt_ = text_.format(v=fmt(vv_) if vv_ is not None else "none", b=fmt(bb_) if bb_ is not None else "none")
                    if k in method_keys:
                        method_diffs.append((method_keys.index(k), txt_))
                    else:
                        diffs.append(txt_)
                else:
                    other.append(k)
        clause = join_words(diffs) if diffs else "no setting"
        if other:
            clause += f"; {len(other)} further setting{'s differ' if len(other) > 1 else ' differs'} (see the configuration file)"
        method_diffs = [t_ for _, t_ in sorted(method_diffs)]
        if method_diffs:
            N.put("cfg.profile_sentence", pf,
                  f"The results were produced with the released configuration \\texttt{{{tex(pf)}}}. It runs the "
                  f"method of this article, which differs from the five-oracle configuration kept in the full profile "
                  f"(\\texttt{{{tex(Path(bp).name)}}}; second row of Table~\\ref{{tab:main_steps}}) in three respects: "
                  f"{'; '.join(method_diffs[:-1]) + '; and ' + method_diffs[-1] if len(method_diffs) > 1 else method_diffs[0]}. Its scale "
                  f"differs from that of the full profile only as follows: {clause}.")
        else:
            N.put("cfg.profile_sentence", pf,
                  f"The results were produced with the reduced profile of the released configuration "
                  f"(\\texttt{{{tex(pf)}}}), which differs from the full profile (\\texttt{{{tex(Path(bp).name)}}}) "
                  f"only as follows: {clause}.")
        N.put("cfg.profile_n_other", len(other))
    else:
        N.put("cfg.profile_sentence", pf,
              f"The results were produced with the full profile of the released configuration (\\texttt{{{tex(pf)}}}).")
    N.int("cfg.exp.bootstrap", e.bootstrap)
    N.put("cfg.exp.bootstrap_seed", int(c.get_path("experiments.bootstrap_seed", 0)))
    N.put("cfg.exp.baseline_trials", int(e.baseline_trials))
    N.int("cfg.exp.dl_rows", e.dl_max_train_rows)
    N.put("cfg.exp.dl_epochs", int(e.dl_epochs))
    N.put("cfg.exp.graph_steps", max(10, 20 * int(e.dl_epochs)))   # full-batch steps of the E-GraphSAGE baseline
    N.put("cfg.exp.seq_window", int(e.seq_window))
    N.put("cfg.online.batch", int(e.online.batch_size))
    N.int("cfg.online.window", e.online.window)
    N.put("cfg.online.seconds", float(e.online.seconds_per_rate), f"{float(e.online.seconds_per_rate):g}")
    N.put("cfg.adv.pad", join_words(e.adversarial.pad_bytes, "or"))
    N.put("cfg.adv.dummy", join_words(e.adversarial.dummy_packets, "or"))
    N.put("cfg.adv.stretch", join_words([f"{float(x):g}" for x in e.adversarial.stretch], "or"))
    N.put("cfg.adv.throttle", join_words([f"{float(x):g}" for x in e.adversarial.throttle], "or"))
    N.put("cfg.exp.n_pipeline_seeds", len(e.get("pipeline_seeds", [])))
    N.put("cfg.online.hold", float(e.online.get("hold_down_s", 5.0)), f"{float(e.online.get('hold_down_s', 5.0)):g}")
    N.int("cfg.adv.max_rows", e.adversarial.max_rows)
    N.int("cfg.shap.rows", e.shap_rows)
    N.put("cfg.online.rates", ", ".join(integer(x) for x in e.online.rates))
    N.put("cfg.online.workers", ", ".join(str(x) for x in e.online.workers))
    N.put("cfg.exp.pipeline_seeds", ", ".join(str(x) for x in e.get("pipeline_seeds", [])))
    N.put("cfg.exp.n_baselines", len(e.baselines))
    # seeds of the partition and of the learned preprocessing (PrepOptions / Ctx.base_options)
    from ..pipeline import PrepOptions

    N.put("cfg.split.seed", int(PrepOptions().split_seed))
    N.put("cfg.pipeline.seed", int(e.seeds[0]))
    fw = e.get("forward") or {}
    for k in ("train", "val", "test"):          # forward-in-time protocol: days from the configuration
        if fw.get(k):
            N.put(f"proto.forward.days.{k}", ", ".join(fw[k]), ", ".join(tex(x) for x in fw[k]))
    # environment of the main run
    env = None
    for name in ("main", "search", "oracles"):
        if R.get(name) and R[name].get("environment"):
            env = R[name]["environment"]
            break
    if env:
        N.put("env.python", env.get("python"))
        N.put("env.cpu", env.get("cpu_model"), tex(env.get("cpu_model") or "not recorded"))
        N.put("env.cpus", env.get("logical_cpus"))
        N.put("env.memory_gb", env.get("memory_gb"), f"{env.get('memory_gb')}")
        N.put("env.platform", env.get("platform"), tex(env.get("platform") or ""))
        for k, v in (env.get("packages") or {}).items():
            if v:
                N.put(f"env.pkg.{k}", v, tex(v))
        for k, v in (env.get("tools") or {}).items():
            if v:
                N.put(f"env.tool.{k}", v, tex(v))
    synthetic = any(bool(v.get("synthetic")) for v in R.values() if isinstance(v, dict))
    N.put("synthetic", int(synthetic), "1" if synthetic else "0")


def b_manifest(cfg, ws, R, N, out) -> None:
    data = ws.capture_root / "data"
    man_p, mem_p = data / "manifest.csv", data / "archive_members.csv"
    einfo_p, rep_p = ws.capture_root / "extract" / "extract_info.json", ws.capture_root / "extract" / "extract_report.csv"
    if not man_p.exists():
        placeholder(out, "manifest", "Retained captures per day (file-level manifest in the code repository).")
        return
    man = pd.read_csv(man_p)
    members = pd.read_csv(mem_p) if mem_p.exists() else None
    rep = pd.read_csv(rep_p) if rep_p.exists() else None
    shutil.copy(man_p, out / "supplementary" / "manifest.csv")
    if mem_p.exists():
        shutil.copy(mem_p, out / "supplementary" / "archive_members.csv")
    t = Table("manifest",
              "Capture days of CSE-CIC-IDS2018 and retained per-host captures. ``Victim'' captures are those of the "
              "hosts attacked on that day; ``background'' captures are drawn at random with the stated seed. The "
              "file-level manifest (archive, member name, size, CRC-32, SHA-256) is in the file "
              r"\texttt{results/manifest.csv} of the code repository.",
              "llrrrrr", [["Day", "Scenario", "In archive", "Victim", "Background", "Retained (GB)", "Flows"]],
              size=r"\footnotesize")
    from ..labels import load_schedule

    sched = load_schedule(cfg.corpus.schedule)
    tot = {"in": 0, "v": 0, "b": 0, "gb": 0.0, "fl": 0}
    for day in cfg.corpus.days:
        m = man[man["day"] == day]
        n_in = int((members["day"] == day).sum()) if members is not None and "day" in members else float("nan")
        nv, nb = int((m["role"] == "victim").sum()), int((m["role"] == "background").sum())
        gb = float(m["file_size"].sum()) / 1e9
        fl = float("nan")
        if rep is not None and "day" in rep:
            fl = int(rep.loc[rep["day"] == day, "flows"].sum())
        fam = sched["days"].get(day, {}).get("family", "")
        t.row(tex(day), FAMILY_LABEL.get(fam, fam or "benign only"), integer(n_in), nv, nb, f(gb, 2), integer(fl))
        tot["in"] += 0 if not isnum(n_in) else n_in
        tot["v"] += nv
        tot["b"] += nb
        tot["gb"] += gb
        tot["fl"] += 0 if not isnum(fl) else fl
    t.rule().row("Total", "", integer(tot["in"]), tot["v"], tot["b"], f(tot["gb"], 2), integer(tot["fl"]))
    excl = dict(cfg.corpus.get("excluded_days") or {})
    if excl:
        t.notes = "Not used: " + "; ".join(f"{tex(k)}, {tex(v)}" for k, v in excl.items()) + "."
    t.write(out)
    N.put("manifest.captures", int(len(man)), integer(len(man)))
    N.put("manifest.victim", int((man["role"] == "victim").sum()))
    N.put("manifest.background", int((man["role"] == "background").sum()))
    N.put("manifest.gb", float(man["file_size"].sum()) / 1e9, f(float(man["file_size"].sum()) / 1e9, 1))
    N.put("manifest.days", int(man["day"].nunique()))
    if einfo_p.exists():
        ei = read_json(einfo_p)
        zv = re.search(r"\d+\.\d+(?:\.\d+)?", ei.get("zeek_version") or "")
        N.put("extract.zeek_version", ei.get("zeek_version"), f"version {zv.group(0)}" if zv else tex(ei.get("zeek_version") or ""))
        sn = ei.get("snort") or {}
        if sn:
            sv = re.search(r"\d+\.\d+(?:\.\d+)*", sn.get("snort_version") or "")
            N.put("extract.snort_version", sn.get("snort_version"),
                  f"version {sv.group(0)}" if sv else tex(sn.get("snort_version") or "not recorded"))
            N.int("extract.snort_rules", sn.get("active_rules"))
            N.put("extract.snort_rule_files", len(sn.get("rule_files") or []))
            src_txt = {"distribution default": "the default rule set of the system's Snort package",
                       "configured": "a rule set given in the configuration"}.get(sn.get("rule_source"), tex(sn.get("rule_source") or ""))
            N.put("extract.snort_rule_source", sn.get("rule_source"), src_txt)
            save_csv(out, "snort_rule_files", [{"file": Path(x["file"]).name, "rules": x["rules"], "sha256": x["sha256"]}
                                               for x in (sn.get("rule_files") or [])])
    chk_p = ws.capture_root / "extract" / "capture_checks.csv"
    if chk_p.exists():                       # structural check and repair of the captures (twws.pcapcheck)
        chk = pd.read_csv(chk_p)
        chk = chk[chk["day"].astype(str).isin([str(d) for d in cfg.corpus.days])]
        rep_ = chk[chk["repaired"].astype(bool)]
        N.int("extract.captures_checked", int(len(chk)))
        N.int("extract.captures_repaired", int(len(rep_)))
        N.int("extract.captures_stray", int((rep_["stray_regions"] > 0).sum()))
        N.int("extract.captures_tail", int((rep_["stray_regions"] == 0).sum()))
        N.int("extract.repair_regions", int(rep_["stray_regions"].sum()))
        kb = float(rep_["removed_bytes"].sum()) / 1e3
        N.put("extract.repair_kb", kb, f(kb, 1))
        save_csv(out, "capture_checks", chk.to_dict("records"))


def b_corpus(cfg, ws, R, N, out) -> None:
    ci_p, si_p = ws.corpus_dir / "corpus_info.json", ws.corpus_dir / "stream_info.json"
    if not ci_p.exists():
        placeholder(out, "corpus", "Composition of the stream and of the evaluation corpus.")
        return
    cinfo = read_json(ci_p)
    sinfo = read_json(si_p) if si_p.exists() else {}
    split = None
    if R.get("main"):
        split = {r["split"]: r for r in R["main"]["pipeline"]["split_table"]}
    fams = fam_sorted(cinfo["selected"].keys())
    hdr = ["Class", "In stream", "Selected"] + (["Train", "Validation", "Test"] if split else [])
    t = Table("corpus",
              "Composition of the flow stream extracted from the retained captures and of the evaluation corpus drawn "
              "from it (unidirectional flows). ``Ambiguous'' flows are excluded from the corpus but remain in the host "
              "streams from which temporal features are computed."
              + (" The partition is the class-proportional 60/20/20 split by flow." if split else ""),
              "l" + "r" * (len(hdr) - 1), [hdr])

    def line(name, key):
        cells = [name, integer(cinfo["available"].get(key, 0)), integer(cinfo["selected"].get(key, 0))]
        if split:
            cells += [integer(split[s]["families"].get(key, 0)) for s in ("train", "validation", "test")]
        t.row(*cells)

    line("Benign", "Benign")
    for fm in fams:
        line(FAMILY_LABEL.get(fm, fm), fm)
    if cinfo["available"].get("Ambiguous"):
        cells = ["Ambiguous (excluded)", integer(cinfo["available"]["Ambiguous"]), "0"] + (["0"] * 3 if split else [])
        t.row(*cells)
    t.rule()
    cells = ["Total", integer(sum(cinfo["available"].values())), integer(cinfo["n_selected"])]
    if split:
        cells += [integer(split[s]["flows"]) for s in ("train", "validation", "test")]
    t.row(*cells)
    cells = ["Malicious share (\\%)", "", pct(cinfo["malicious_share"], 2)]
    if split:
        cells += [pct(split[s]["malicious_share"], 2) for s in ("train", "validation", "test")]
    t.row(*cells)
    t.write(out)
    N.int("corpus.n", cinfo["n_selected"])
    N.pct("corpus.malicious_share", cinfo["malicious_share"], 2)
    n_mal = sum(v for k, v in cinfo["selected"].items() if k != "Benign")
    N.int("corpus.n_malicious", n_mal)
    N.int("corpus.n_benign", cinfo["selected"].get("Benign", 0))
    N.put("corpus.n_families", len(fams))
    N.int("stream.n", sum(cinfo["available"].values()))
    N.int("stream.n_ambiguous", cinfo["available"].get("Ambiguous", 0))
    for k, v in cinfo["available"].items():
        N.int(f"stream.family.{k}", v)
    for k, v in cinfo["selected"].items():
        N.int(f"corpus.family.{k}", v)
    try:                                   # descriptive shares of the corpus
        meta = ws.meta
        N.pct("corpus.tls_share", float((meta["tls"].to_numpy() > 0).mean()), 1)
        N.pct("corpus.entropy_missing_share", float(meta["entropy"].isna().mean()), 1)
        N.pct("corpus.tcp_share", float((meta["proto"].astype(str) == "tcp").mean()), 1)
        N.int("corpus.n_connections", int(meta["conn"].nunique()))
        N.int("corpus.n_sources", int(meta["src_ip"].astype(str).nunique()))
        N.put("corpus.n_days", int(meta["day"].astype(str).nunique()))
        both = meta.groupby("conn")["dir"].nunique()
        N.pct("corpus.conn_both_directions", float((both == 2).mean()), 1)
    except Exception as exc:  # pragma: no cover
        log.warning("corpus descriptive shares not computed: %s", exc)
    capped = [FAMILY_LABEL.get(k, k) for k in fams if cinfo["available"].get(k, 0) > cinfo["selected"].get(k, 0)]
    N.put("corpus.capped_families", ", ".join(capped) if capped else "none")
    if cinfo.get("thinning_rate"):
        N.num("corpus.thinning_rate", cinfo["thinning_rate"], 3)
    if split:
        for s in ("train", "validation", "test"):
            N.int(f"split.{s}", split[s]["flows"])
            N.pct(f"split.{s}.malicious_share", split[s]["malicious_share"], 2)
    if sinfo:
        N.int("stream.extracted", sinfo.get("flows_extracted"))
        d = sinfo.get("deduplication", {})
        N.int("stream.dedup_removed", d.get("flows_removed"))
        N.int("stream.connections", sinfo.get("connections_stream"))
        N.put("stream.utc_offset", sinfo.get("utc_offset_hours"),
              "\\ensuremath{" + f"{int(sinfo.get('utc_offset_hours', 0)):+d}" + "}")   # typeset minus sign
        N.int("stream.snort_alert_flows", sinfo.get("snort_alert_flows"))
        N.int("stream.snort_alerts", sinfo.get("snort_alerts_total"))
        diag = sinfo.get("utc_offset_diagnostic") or []
        best = max(diag, key=lambda r: r.get("share_inside") or 0) if diag else None
        if best:
            N.pct("stream.offset_share_inside", best.get("share_inside"), 1)
        save_csv(out, "label_summary_by_day", sinfo.get("label_summary") or [])
        save_csv(out, "utc_offset_diagnostic", diag)
        attack_flows, family_flows = {}, {}
        for row in sinfo.get("label_summary") or []:
            a, fm, n = str(row.get("attack") or ""), str(row.get("family") or ""), int(row.get("flows") or 0)
            family_flows[fm] = family_flows.get(fm, 0) + n
            if a and fm not in ("Benign", "Ambiguous"):
                attack_flows[(fm, a)] = attack_flows.get((fm, a), 0) + n
        for (fm, a), n in attack_flows.items():       # e.g. stream.attack.DDoS-HOIC, share within its family
            N.int(f"stream.attack.{a}", n)
            if family_flows.get(fm):
                N.pct(f"stream.attack_share.{a}", n / family_flows[fm], 1)
        if sinfo.get("snort_top_signatures"):
            save_csv(out, "snort_top_signatures", sinfo["snort_top_signatures"])
        N.int("stream.snort_alerts_attack_classes", sinfo.get("snort_alerts_attack_classes"))
        N.int("stream.snort_alert_flows_any", sinfo.get("snort_alert_flows_any"))
        if isnum(sinfo.get("snort_alerts_total")) and sinfo.get("snort_alerts_total"):
            N.pct("stream.snort_attack_class_share", (sinfo.get("snort_alerts_attack_classes") or 0)
                  / sinfo["snort_alerts_total"], 1)
        if sinfo.get("snort_alerts_by_class"):
            save_csv(out, "snort_alerts_by_class", [{"class": k, "alerts": v} for k, v in sinfo["snort_alerts_by_class"].items()])


def b_source(cfg, ws, R, N, out) -> None:
    """Second dataset (CICIDS2017): flows, clock offset and labelling, read from its stream and corpus files."""
    from ..stages import _abs

    wd = cfg.get_path("transfer.work_dir")
    if not wd:
        return
    root = _abs(wd)
    sp, cp = root / "corpus" / "stream_info.json", root / "corpus" / "corpus_info.json"
    if not sp.exists():
        return
    sinfo = read_json(sp)
    N.int("src.flows", sinfo.get("flows_extracted"))
    N.int("src.dedup_removed", (sinfo.get("deduplication") or {}).get("flows_removed"))
    if sinfo.get("utc_offset_hours") is not None:
        N.put("src.utc_offset", sinfo["utc_offset_hours"], "\\ensuremath{" + f"{int(sinfo['utc_offset_hours']):+d}" + "}")
    diag = sinfo.get("utc_offset_diagnostic") or []
    best = max(diag, key=lambda r: r.get("share_inside") or 0) if diag else None
    if best:
        N.pct("src.offset_share_inside", best.get("share_inside"), 1)
        N.int("src.attacker_flows", best.get("attacker_flows"))
    fam_rows = {}
    for row in sinfo.get("label_summary") or []:
        fm = str(row.get("family") or "")
        n, ins = int(row.get("flows") or 0), int(row.get("inside_window") or 0)
        a, b = fam_rows.get(fm, (0, 0))
        fam_rows[fm] = (a + n, b + ins)
    for fm, (n, ins) in fam_rows.items():
        N.int(f"src.family.{fm}", n)
        if fm not in ("Benign", "Ambiguous") and n:
            N.pct(f"src.family_inside.{fm}", ins / n, 1)
    save_csv(out, "label_summary_by_day_ids2017", sinfo.get("label_summary") or [])
    save_csv(out, "utc_offset_diagnostic_ids2017", diag)
    if cp.exists():
        N.int("src.corpus.n", read_json(cp).get("n_selected"))


def b_pl_attacks(cfg, ws, R, N, out) -> None:
    """Coverage of the pseudo-labels per attack, from the label check (``twws.cli label-check``), which computes the
    votes and both aggregation rules exactly as the pipeline does on the training subset of the primary partition."""
    p = ws.root / "calibration" / "label_check.json"
    lc = read_json(p) if p.exists() else None
    if not lc or "pseudo_label_recall_by_attack_train" not in lc:
        placeholder(out, "pl_attacks")
        return
    att = lc["attacks_train"]
    ds = lc["pseudo_label_recall_by_attack_train"]["ds"]
    maj = lc["pseudo_label_recall_by_attack_train"]["majority"]
    vote = lc["malicious_vote_share_by_attack_train"]
    ref = R.get("_ref_label_check") or {}
    ds_ref = (ref.get("pseudo_label_recall_by_attack_train") or {}).get("ds") or {}
    order = sorted(att, key=lambda a: (FAMILY_ORDER.index(att[a]["family"]) if att[a]["family"] in FAMILY_ORDER else 99, a))
    if ds_ref and "rate" in vote:
        # extended oracle set: the five oracles of the first version (reference) next to all oracles
        show = [n for n in ("rate", "beacon") if n in vote]
        t = Table("pl_attacks",
                  "Share of the malicious training flows of each attack (primary partition) that the pseudo-labels label "
                  "malicious: label model with the five oracles of the first version, label model with all "
                  f"{len(lc['oracles'])} oracles (dependent oracles merged), and equal-vote majority of all oracles; and share "
                  "that the two repetition oracles vote malicious.",
                  "llrrrrrr", [["", "", "", "Five", "All", "", "", ""],
                               ["Family", "Attack", "Flows", "oracles", "oracles", "Majority",
                                *[ORACLE_SHORT.get(n, n) for n in show]]], size=r"\footnotesize")
        for a in order:
            fm = att[a]["family"]
            t.row(FAMILY_LABEL.get(fm, fm), tex(ATTACK_TEXT.get(a, a)), integer(att[a]["flows"]), f(ds_ref.get(a)),
                  f(ds[a]), f(maj[a]), *[f(vote[n][a]) for n in show])
    else:
        show = [n for n in ("stat", "transfer") if n in vote]
        t = Table("pl_attacks",
                  "Share of the malicious training flows of each attack (primary partition) that the pseudo-labels label "
                  "malicious, for both aggregation rules, and share that the statistical-threshold and transfer oracles vote "
                  "malicious. The other three oracles vote malicious on at most "
                  f"{integer(max([round(vote[n][a] * att[a]['flows']) for n in vote if n not in show for a in att] or [0]))} "
                  "training flows of any attack.",
                  "llrrrrr", [["Family", "Attack", "Flows", "Label model", "Majority",
                               *[ORACLE_SHORT.get(n, n) for n in show]]], size=r"\footnotesize")
        for a in order:
            fm = att[a]["family"]
            t.row(FAMILY_LABEL.get(fm, fm), tex(ATTACK_TEXT.get(a, a)), integer(att[a]["flows"]), f(ds[a]), f(maj[a]),
                  *[f(vote[n][a]) for n in show])
    rows_csv = []
    for a in order:
        fm = att[a]["family"]
        rows_csv.append({"family": fm, "attack": a, "flows": att[a]["flows"], "label_model": ds[a], "majority": maj[a],
                         **({"label_model_five_oracles": ds_ref.get(a)} if ds_ref else {}),
                         **{f"vote_{n}": vote[n][a] for n in vote}})
        N.num(f"lc.ds.attack.{a}.recall", ds[a])
        if ds_ref and ds_ref.get(a) is not None:
            N.num(f"lc.ref.attack.{a}.recall", ds_ref[a])
        N.int(f"lc.attack.{a}.flows", att[a]["flows"])
        for n in vote:
            N.num(f"lc.vote.{n}.{a}", vote[n][a])
    t.write(out)
    if ds_ref:
        full_r = [a for a in order if (ds_ref.get(a) or 0) >= 0.99]
        none_r = [a for a in order if (ds_ref.get(a) or 0) == 0]
        N.put("lc.ref.n_covered", len(full_r))
        N.put("lc.ref.covered_attacks", join_words([ATTACK_TEXT.get(a, a) for a in full_r]) or "none")
        N.put("lc.ref.n_uncovered", len(none_r))
        N.put("lc.ref.uncovered_attacks", join_words([ATTACK_TEXT.get(a, a) for a in none_r]) or "none")
    save_csv(out, "pseudo_label_recall_by_attack", rows_csv)
    full = [a for a in order if ds[a] >= 0.99]
    part = [a for a in order if 0 < ds[a] < 0.99]
    none_ = [a for a in order if ds[a] == 0]
    N.put("lc.covered_attacks", join_words([ATTACK_TEXT.get(a, a) for a in full]) or "none")
    N.put("lc.n_covered", len(full))
    N.put("lc.partial_attacks", join_words([ATTACK_TEXT.get(a, a) for a in part]) or "none")
    N.put("lc.n_partial", len(part))
    N.put("lc.n_uncovered", len(none_))
    N.put("lc.uncovered_attacks", join_words([ATTACK_TEXT.get(a, a) for a in none_]) or "none")
    N.put("lc.n_attacks", len(order))
    N.int("lc.covered_flows", sum(att[a]["flows"] for a in full))
    N.put("lc.covered_threshold_pct", 99)
    if "transfer" in vote and none_:      # how often the transfer classifier flags an attack that the pseudo-labels miss
        N.pct("lc.transfer.max_vote_uncovered", max(vote["transfer"][a] for a in none_), 1)
        N.put("lc.transfer.max_vote_uncovered_attack", tex(ATTACK_TEXT.get(max(none_, key=lambda a: vote["transfer"][a]), "")))
    # which vote patterns the label model labels malicious (statistical thresholds and transfer classifier)
    st = lc.get("ds_stat_transfer_train")
    if st:
        N.int("lc.ds.n_mal", st["ds_malicious"])
        N.int("lc.agree.rows", st["agree_rows"])
        N.int("lc.ds.n_mal_agree", st["ds_malicious_agree"])
        N.pct("lc.ds.mal_agree_share", st["ds_malicious_agree"] / max(1, st["ds_malicious"]), 1)
        N.int("lc.ds.n_mal_one_abstains", st["ds_malicious_one_abstains"])
        N.int("lc.ds.n_mal_other", st["ds_malicious_other"])
        N.int("lc.ds.n_mal_not_agree", st["ds_malicious"] - st["ds_malicious_agree"])
    # the iteration limit of EM: the same fit with ten times as many iterations
    it = lc.get("ds_iteration_check")
    if it:
        N.int("lc.iter.limit", it["limit_default"])
        N.int("lc.iter.used", it["iterations_default"])
        N.put("lc.iter.at_limit", "yes" if it["iterations_default"] >= it["limit_default"] else "no")
        N.int("lc.iter.conv", it["iterations_10x"])
        N.put("lc.iter.converged_10x", "yes" if it["iterations_10x"] < it["limit_10x"] else "no")
        N.pct("lc.iter.pi_default", it["pi_default"], 2)
        N.pct("lc.iter.pi_conv", it["pi_10x"], 2)
        if "to_malicious_train" in it:
            N.int("lc.iter.changes", it["label_changes_train"])
            N.int("lc.iter.to_mal", it["to_malicious_train"])
            N.int("lc.iter.to_mal_benign", it["to_malicious_benign_train"])
            N.int("lc.iter.to_ben", it["to_benign_train"])
            N.num("lc.iter.precision", it["precision_10x_train"])
            N.num("lc.iter.recall", it["recall_10x_train"])
            rc = it.get("recall_by_attack_10x_train") or {}
            full10 = [a for a in order if rc.get(a, 0) >= 0.99]
            N.put("lc.iter.covered_attacks", join_words([ATTACK_TEXT.get(a, a) for a in full10]) or "none")
            N.put("lc.iter.same_covered", "yes" if full10 == full else "no")
        if "stat_flagged_train" in it:
            N.int("lc.iter.stat_flagged", it["stat_flagged_train"])
            N.int("lc.iter.n_mal", it["malicious_10x_train"])
            N.int("lc.iter.mal_not_stat", it["malicious_10x_not_stat_flagged_train"])
            N.int("lc.iter.stat_not_mal", it["stat_flagged_not_malicious_10x_train"])
    vv = lc.get("ds_iteration_check_variants") or []
    if vv:
        ORL = {"stat": "the statistical thresholds", "signature": "the signature oracle", "cluster": "the clustering oracle",
               "transfer": "the transfer classifier", "domain": "the domain heuristics",
               "rate": "the repeated-connection oracle", "beacon": "the regularity oracle"}
        nice = lambda v: ("all oracles" if v == "all oracles" else "without " + ORL.get(v.replace("without ", ""), v))
        at_lim = [x["variant"] for x in vv if x["at_limit"]]
        N.put("lc.iter.variants_at_limit", len(at_lim), join_words([nice(v) for v in at_lim]) or "none")
        N.put("lc.iter.other_at_limit", len([v for v in at_lim if v != "all oracles"]),
              join_words([nice(v) for v in at_lim if v != "all oracles"]) or "none")
        same = [x["variant"] for x in vv if x["labels_equal_converged_all_oracles"] and not x["at_limit"]
                and x["variant"] != "all oracles"]
        N.put("lc.iter.equal_variant", len(same), join_words([nice(v) for v in same]) or "none")
        for x in vv:
            key = x["variant"].replace("without ", "without_").replace(" ", "_")
            N.int(f"lc.iter.{key}.iterations", x["iterations"])
            N.int(f"lc.iter.{key}.malicious", x["malicious"])
            N.int(f"lc.iter.{key}.tp", x["tp"])


def b_oracles(cfg, ws, R, N, out) -> None:
    r = R.get("oracles")
    names_cfg = ["oracle_stats", "oracle_pairs", "aggregation", "trust", "oracle_ablation"]
    if not r:
        for n in names_cfg:
            placeholder(out, n, "Weak-supervision oracle analysis.")
        return
    names = r["oracles"]
    N.put("oracles.n", len(names))
    N.put("oracles.names", ", ".join(ORACLE_LABEL.get(n, n).lower() for n in names))
    # ---- per-oracle statistics
    t = Table("oracle_stats",
              "Behavior of each oracle on the training split, measured against ground truth (which no oracle sees). "
              "Coverage: share of flows on which the oracle votes; ``malicious votes'': share of flows voted malicious; "
              "precision: correct share of its malicious votes; sensitivity and false-alarm rate are taken over the "
              "flows on which it votes; the last column is the share of all malicious flows it votes malicious.",
              "lrrrrrr", [["", "Coverage", "Malicious", "", "Sensi-", "False", "Recall"],
                          ["Oracle", "(\\%)", "votes (\\%)", "Precision", "tivity", "alarm", "(all)"]], size=r"\footnotesize")
    rows_csv = []
    for n in names:
        s = r["per_oracle"][n]["train"]
        t.row(ORACLE_LABEL.get(n, n), pct(s["coverage"]), pct(s["malicious_vote_share"], 2), f(s["precision"]),
              f(s["sensitivity_voted"]), f(s["false_alarm_voted"], 4), f(s["recall_all_malicious"]))
        rows_csv.append({"oracle": n, "split": "train", **s})
        rows_csv.append({"oracle": n, "split": "test", **r["per_oracle"][n]["test"]})
        for k in ("coverage", "malicious_vote_share", "abstain_share"):
            N.pct(f"oracle.{n}.{k}", s[k], 1 if k != "malicious_vote_share" else 2)
        for k in ("precision", "sensitivity_voted", "recall_all_malicious"):
            N.num(f"oracle.{n}.{k}", s[k])
        N.num(f"oracle.{n}.false_alarm_voted", s["false_alarm_voted"], 4)
        if isnum(s.get("benign_flagged_share")):
            N.pct(f"oracle.{n}.benign_flagged", s["benign_flagged_share"], 2)
    first = r.get("first_version_oracles") or {}
    if first:                                    # definitions of the first version, measured but not used
        t.rule()
        for label, v in first.items():
            s = v["train"]
            t.row(r"\emph{" + tex(label[0].upper() + label[1:]) + "}", pct(s["coverage"]), pct(s["malicious_vote_share"], 2),
                  f(s["precision"]), f(s["sensitivity_voted"]), f(s["false_alarm_voted"], 4), f(s["recall_all_malicious"]))
            rows_csv.append({"oracle": label, "split": "train", **s})
            short = "signature" if label.startswith("signature") else "domain" if label.startswith("domain") else None
            if short:
                N.pct(f"oracle.first.{short}.malicious_vote_share", s["malicious_vote_share"], 2)
                N.num(f"oracle.first.{short}.precision", s["precision"])
                if isnum(s.get("benign_flagged_share")):
                    N.pct(f"oracle.first.{short}.benign_flagged", s["benign_flagged_share"], 1)
        t.notes = ("Rows in italics: the definitions of the first version (every Snort alert; fixed fan-out threshold), "
                   "measured in the same way but not used for the pseudo-labels.")
    t.write(out)
    save_csv(out, "oracle_statistics", rows_csv)
    # per-family vote shares (supplementary)
    fam_rows = []
    for n in names:
        for fm, v in r["per_oracle"][n]["malicious_vote_share_by_family"].items():
            fam_rows.append({"oracle": n, "family": fm, "malicious_vote_share": v,
                             "abstain_share": r["per_oracle"][n]["abstain_share_by_family"].get(fm)})
    save_csv(out, "oracle_votes_by_family", fam_rows)
    fams = fam_sorted({x["family"] for x in fam_rows})
    t = Table("oracle_family",
              "Share of the flows of each class (\\%) that an oracle votes malicious (whole corpus). The benign row is "
              "the oracle's false-alarm share.",
              "l" + "r" * len(names), [["Class"] + [ORACLE_SHORT.get(n, n) for n in names]], size=r"\footnotesize")
    look = {(x["oracle"], x["family"]): x["malicious_vote_share"] for x in fam_rows}
    for fm in ["Benign"] + fams:
        t.row(FAMILY_LABEL.get(fm, fm), *[pct(look.get((n, fm)), 1) for n in names])
    t.write(out)
    # ---- pairwise agreement and error correlation
    t = Table("oracle_pairs",
              "Pairwise behavior of the oracles on training flows where both vote. $\\kappa$: Cohen's kappa of the "
              "votes; $\\phi$: correlation of the error indicators (not defined when one oracle makes no error on the "
              "co-voted flows); the last two columns give the observed share of flows on which both are wrong and the "
              "share expected if their errors were independent.",
              "llrrrrr", [["", "", "Co-voted", "", "", "Both wrong", "If independent"],
                          ["Oracle A", "Oracle B", "(\\%)", "$\\kappa$", "$\\phi$", "(\\%)", "(\\%)"]], size=r"\footnotesize")
    worst = None
    for p in r["pairwise_train"]:
        t.row(ORACLE_SHORT.get(p["a"], p["a"]), ORACLE_SHORT.get(p["b"], p["b"]), pct(p["co_voted_share"]),
              f(p["kappa"], 2), f(p["error_correlation"], 2), pct(p["both_wrong_share"], 2),
              pct(p["both_wrong_if_independent"], 2))
        if isnum(p.get("error_correlation")) and (worst is None or p["error_correlation"] > worst["error_correlation"]):
            worst = p
    t.write(out)
    save_csv(out, "oracle_pairwise", r["pairwise_train"])
    if worst:
        N.num("oracles.max_phi", worst["error_correlation"], 2)
        N.put("oracles.max_phi_pair", f"{ORACLE_TEXT.get(worst['a'], worst['a'])} and "
                                      f"{ORACLE_TEXT.get(worst['b'], worst['b'])}")
    je, jm = r["joint_errors_train"], r["joint_errors_train_malicious"]
    for key, j in (("all", je), ("mal", jm)):
        N.pct(f"oracles.joint.{key}.ge2", j["share_ge2_wrong"], 2)
        N.pct(f"oracles.joint.{key}.ge2_indep", j["share_ge2_wrong_independent"], 2)
        N.pct(f"oracles.joint.{key}.ge3", j["share_ge3_wrong"], 2)
        N.pct(f"oracles.joint.{key}.ge3_indep", j["share_ge3_wrong_independent"], 2)
        if j["share_ge2_wrong_independent"] > 0:
            N.num(f"oracles.joint.{key}.ge2_ratio", j["share_ge2_wrong"] / j["share_ge2_wrong_independent"], 1)
        for k_, (ob, ind) in enumerate(zip(j.get("observed", []), j.get("independent", []))):   # exactly k wrong
            N.pct(f"oracles.joint.{key}.eq{k_}", ob, 2)
            N.pct(f"oracles.joint.{key}.eq{k_}_indep", ind, 2)
    K = len(names)
    t = Table("oracle_joint",
              "Number of oracles that are wrong on the same training flow: observed distribution against the "
              "Poisson-binomial distribution implied by independent errors with the same marginal rates (\\%).",
              "l" + "r" * (K + 1), [["Flows"] + [f"{i} wrong" for i in range(K + 1)]], size=r"\footnotesize")
    t.row("All, observed", *[pct(x, 2) for x in je["observed"]])
    t.row("All, independent", *[pct(x, 2) for x in je["independent"]])
    t.row("Malicious, observed", *[pct(x, 2) for x in jm["observed"]])
    t.row("Malicious, independent", *[pct(x, 2) for x in jm["independent"]])
    t.write(out)
    save_csv(out, "oracle_joint_errors", [{"flows": "all", "kind": k, **{f"wrong_{i}": v for i, v in enumerate(je[k])}}
                                          for k in ("observed", "independent")]
             + [{"flows": "malicious", "kind": k, **{f"wrong_{i}": v for i, v in enumerate(jm[k])}}
                for k in ("observed", "independent")])
    # ---- aggregation rules
    agg = r["aggregation"]
    t = Table("aggregation",
              "Quality of the aggregated pseudo-labels against ground truth (training split) for each aggregation rule"
              + (", before refinement" if bool(cfg.get_path("refinement.enabled", True)) else "") +
              ". ``Labelled'': share of flows that receive a label. Precision, recall, F1 and "
              "accuracy refer to the flows that receive a label; ``recall (all)'' counts a malicious flow without a "
              "label as missed. ``High trust'': share with trust of at least the high-trust threshold; the last column "
              "counts high-trust flows labelled malicious.",
              "lrrrrrrrr", [["", "Labelled", "", "", "Recall", "", "", "High trust", "High-trust"],
                            ["Aggregation", "(\\%)", "Precision", "Recall", "(all)", "F1", "Accuracy", "(\\%)", "malicious"]],
              size=r"\footnotesize")
    rows_agg = [("majority", "Equal-vote majority"), ("ds", "Label model")]
    if "ds_ungrouped" in agg:   # extended oracle set: the label model with and without merging dependent oracles
        rows_agg = [("majority", "Equal-vote majority"), ("ds_ungrouped", "Label model, oracles not merged"),
                    ("ds", "Label model, dependent oracles merged")]
    for key, label in rows_agg:
        a = agg[key]
        q = a["train"]
        t.row(label, pct(q["labelled_share"]), f(q["precision"]), f(q["recall"]), f(q.get("recall_all_malicious")),
              f(q["f1"]), f(q["accuracy"]), pct(a["high_trust_share_train"]), integer(a["high_trust_malicious_train"]))
        for k in ("precision", "recall", "f1", "accuracy"):
            N.num(f"agg.{key}.{k}", q[k])
        N.num(f"agg.{key}.recall_all", q.get("recall_all_malicious"))
        N.pct(f"agg.{key}.labelled", q["labelled_share"])
        N.pct(f"agg.{key}.pseudo_mal_share", q["pseudo_malicious_share"], 2)
        N.pct(f"agg.{key}.high_trust_share", a["high_trust_share_train"])
        N.int(f"agg.{key}.high_trust_malicious", a["high_trust_malicious_train"])
        tq = agg[key]["test"]
        for k in ("precision", "recall", "f1", "accuracy"):
            N.num(f"agg.{key}.test.{k}", tq[k])
    t.write(out)
    N.pct("agg.true_mal_share", agg["ds"]["train"]["true_malicious_share"], 2)
    # trust calibration.  Code versions before 25 binned the majority trust at the five-oracle shares 0.2, ..., 1.0,
    # which miss the shares k/7 of seven oracles; results/trust.json (experiment 'trust') recomputes the table with
    # the corrected binning and is used when its pseudo-labels are those of the oracle experiment.
    tsrc, n_or = agg, len(names)
    tr_res = (R.get("trust") or {}).get("aggregation") or {}
    if tr_res:
        same = all(tr_res.get(k, {}).get("train", {}).get(c) == agg[k]["train"].get(c)
                   for k in ("majority", "ds") for c in ("tp", "fp", "fn", "tn"))
        N.put("trust.same_labels", int(same), "yes" if same else "NO")
        if same:
            tsrc, n_or = tr_res, int(R["trust"].get("n_oracles") or n_or)
        else:
            log.error("results/trust.json: pseudo-labels differ from those of the oracle experiment; not used")
    N.put("trust.recomputed", int(tsrc is not agg), "yes" if tsrc is not agg else "no")
    complete = {k: sum(r["share"] for r in tsrc[k]["trust_table_train"] if r["trust"] != "not binned") > 0.999
                for k in ("majority", "ds")}
    t = Table("trust",
              f"Trust score against the accuracy of the pseudo-label (training split). For equal-vote majority the "
              f"trust is the share $k/{n_or}$ of the {n_or} oracles that vote for the label (rows: $k$ of {n_or}); for "
              f"the label model it is the posterior probability of the assigned label. "
              + ("Every training flow falls into one row." if all(complete.values()) else
                 "RESULT INCOMPLETE: the rows of a block cover only part of the training flows (binning of code "
                 "versions before 25); the corrected table is computed by the experiment \\texttt{trust}."),
              "llrrr", [["Aggregation", "Trust", "Flows (\\%)", "Label accuracy", "Truly malicious (\\%)"]],
              size=r"\footnotesize")
    rows_csv = []
    bin_key = {"[0.5, 0.6)": "b50", "[0.6, 0.8)": "b60", "[0.8, 0.95)": "b80", "[0.95, 0.99)": "b95",
               "[0.99, 1]": "b99", "no majority": "none"}
    for key, label in (("majority", "Equal-vote majority"), ("ds", "Label model")):
        first = True
        rows_k = tsrc[key]["trust_table_train"]
        for row in rows_k:
            if not row["n"]:
                continue
            t.row(label if first else "", tex(row["trust"]), pct(row["share"], 2), f(row["accuracy"]),
                  pct(row["malicious_true_share"], 1))
            rows_csv.append({"aggregation": key, **row})
            first = False
            bk = bin_key.get(row["trust"]) or ("k" + str(row["trust"]).split("/")[0] if "/" in str(row["trust"]) else None)
            if bk and complete[key]:          # no value is written from a table whose rows miss flows
                N.pct(f"trust.{key}.{bk}.share", row["share"], 2)
                N.num(f"trust.{key}.{bk}.accuracy", row["accuracy"])
                N.pct(f"trust.{key}.{bk}.mal_share", row["malicious_true_share"], 1)
        N.pct(f"trust.{key}.binned_share", sum(r["share"] for r in rows_k if r["trust"] != "not binned"), 2)
        labelled = [r for r in rows_k if r["n"] and r["trust"] not in ("no majority", "not binned")
                    and isnum(r["accuracy"])]
        if labelled and complete[key]:   # the trust range whose labels are least accurate (calibration check)
            w = min(labelled, key=lambda r: r["accuracy"])
            N.put(f"trust.{key}.worst.range", w["trust"], tex(w["trust"]))
            N.pct(f"trust.{key}.worst.share", w["share"], 2)
            N.num(f"trust.{key}.worst.accuracy", w["accuracy"])
            N.pct(f"trust.{key}.worst.mal_share", w["malicious_true_share"], 1)
        if key == "majority":
            t.rule()
    t.write(out)
    save_csv(out, "trust_calibration", rows_csv)
    m = agg["ds"].get("model")
    if m:
        t = Table("label_model",
                  "Oracle error rates estimated by the label model without labels, against the rates measured with "
                  "ground truth (whole corpus). Sensitivity: P(vote malicious $|$ malicious, votes); false alarm: "
                  "P(vote malicious $|$ benign, votes).",
                  "lrrrr", [["", "\\multicolumn{2}{c}{Sensitivity}", "\\multicolumn{2}{c}{False alarm}"],
                            ["Oracle", "estimated", "measured", "estimated", "measured"]], size=r"\footnotesize")
        for n in list(m["oracles"]):          # oracles, or label-model sources when dependent oracles are merged
            o = m["oracles"][n]
            t.row(_source_name(n, ORACLE_SHORT), f(o["sensitivity_estimated"]), f(o["sensitivity_true"]),
                  f(o["false_alarm_estimated"], 4), f(o["false_alarm_true"], 4))
            N.num(f"lm.{n}.sens_est", o["sensitivity_estimated"])
            N.num(f"lm.{n}.sens_true", o["sensitivity_true"])
            N.num(f"lm.{n}.fa_est", o["false_alarm_estimated"], 4)
            N.num(f"lm.{n}.fa_true", o["false_alarm_true"], 4)
        from ..weak import dawid_skene, dawid_skene_multistart
        grouped = r.get("label_model_method") == "ds_grouped"
        lim = int(inspect.signature(dawid_skene_multistart if grouped else dawid_skene).parameters["n_iter"].default)
        t.notes = (f"Estimated prior of the malicious class: {pct(m['pi_estimated'], 2)}\\% "
                   f"(true share {pct(m['pi_true'], 2)}\\%); EM iterations: {m['iterations']} (limit {lim}).")
        if grouped:
            t.notes += (" The statistical thresholds and the transfer classifier form one source of the label model "
                        "(their malicious votes coincide; Section~\\ref{sec:method}).") if "stat+transfer" in m["oracles"] else ""
        t.write(out)
        N.pct("agg.ds.pi_est", m["pi_estimated"], 2)
        N.pct("agg.ds.pi_true", m["pi_true"], 2)
        N.put("agg.ds.iterations", int(m["iterations"]))
    # ---- detector-level oracle ablation
    abl = r["detector_ablation"]
    t = Table("oracle_ablation",
              f"Detector trained on pseudo-labels from different oracle sets and aggregation rules (test split, "
              f"mean $\\pm$ standard deviation over {len(r.get('ablation_seeds', []))} model seeds). ``Label accuracy'' is "
              "the accuracy of the training pseudo-labels.",
              "lrrrrr", [["Oracles / aggregation", "Label accuracy", "Precision", "Recall", "F1", "Features"]],
              size=r"\footnotesize")
    rows_csv = []
    ref = None
    for i, v in enumerate(abl):
        label = v["variant"]
        for n in names:
            label = label.replace(f"without {n}", f"without {ORACLE_LABEL.get(n, n).lower()}") \
                         .replace(f"only {n}", f"only {ORACLE_LABEL.get(n, n).lower()}")
        if i in (2, 2 + len(names)):
            t.rule()
        plt_ = v.get("pseudo_labels_train") or {}
        if "tp" in plt_:                                      # how many training flows the variant labels malicious
            slug_ = re.sub(r"[^a-z0-9]+", "_", v["variant"].lower()).strip("_")
            N.int(f"oabl.{slug_}.label_tp", plt_["tp"])
            N.int(f"oabl.{slug_}.label_pos", plt_["tp"] + plt_["fp"])
        if "summary" not in v:
            t.row(tex(label), f(v.get("pseudo_labels_train", {}).get("accuracy")), r"\multicolumn{4}{l}{" + tex(v.get("status", "not run")) + "}")
            continue
        s = v["summary"]
        t.row(tex(label), f(v["pseudo_labels_train"]["accuracy"]), *metric_cells(s), v.get("n_features", "--"))
        rows_csv.append({"variant": v["variant"], "label_accuracy": v["pseudo_labels_train"]["accuracy"],
                         "label_malicious": v["pseudo_labels_train"].get("tp", 0) + v["pseudo_labels_train"].get("fp", 0),
                         "label_true_malicious": v["pseudo_labels_train"].get("tp"),
                         **{f"{k}_{st}": s[k][st] for k in ("precision", "recall", "f1") for st in ("mean", "std")}})
        slug = re.sub(r"[^a-z0-9]+", "_", v["variant"].lower()).strip("_")
        N.summary(f"oabl.{slug}.f1", s["f1"])
        N.summary(f"oabl.{slug}.recall", s["recall"])
        N.summary(f"oabl.{slug}.precision", s["precision"])
        N.num(f"oabl.{slug}.label_acc", v["pseudo_labels_train"]["accuracy"])
        for k in ("precision", "recall", "f1"):               # final training pseudo-labels (after refinement)
            N.num(f"oabl.{slug}.label_{k}", v["pseudo_labels_train"][k])
        if "fpr" in v["pseudo_labels_train"]:
            N.pct(f"oabl.{slug}.label_benign_flagged", v["pseudo_labels_train"]["fpr"], 1)
        for fm, fv in ((s.get("families") or {}) if isinstance(s.get("families"), dict) else {}).items():
            N.num(f"oabl.{slug}.family.{fm}", fv.get("mean"))     # recall of the detector per family (test split)
        if i == 0:
            ref = s
    lcp = ws.root / "calibration" / "label_check.json"
    lcv = (read_json(lcp).get("ds_iteration_check_variants") if lcp.exists() else None) or []
    # the check concerns the label model with a single start and the first iteration limit; with merged oracles the
    # rows of this table use several starts and a higher limit, so the note would describe another model
    if str(cfg.get_path("oracles.aggregation", "ds")) == "ds_grouped":
        lcv = []
    if lcv:                                   # which label-model rows stop at the iteration limit of EM
        lim_names = []
        for x in lcv:
            if x["at_limit"]:
                nm = x["variant"]
                for n in names:
                    nm = nm.replace(f"without {n}", f"without {ORACLE_LABEL.get(n, n).lower()}")
                lim_names.append(nm)
        t.notes = ("EM of the label model stops at its iteration limit for the rows " + join_words(lim_names) +
                   r"; the other rows of the label model converge (Section~\ref{sec:oracles}).") if lim_names else None
    t.write(out)
    save_csv(out, "oracle_ablation", rows_csv)
    untrainable = []
    for v in abl:
        if "summary" not in v:
            lab_ = v["variant"]
            for n in names:
                lab_ = lab_.replace(f"without {n}", f"without the {ORACLE_NOUN.get(n, n)}") \
                           .replace(f"only {n}", f"with the {ORACLE_NOUN.get(n, n)} alone")
            untrainable.append(lab_)
    N.put("oabl.not_trainable", join_words(untrainable) or "none")
    N.put("oabl.n_not_trainable", len(untrainable))
    # ---- figure: error-correlation matrix
    try:
        plt = _mpl()
        K = len(names)
        M = np.full((K, K), np.nan)
        for p in r["pairwise_train"]:
            i, j = names.index(p["a"]), names.index(p["b"])
            if isnum(p.get("error_correlation")):
                M[i, j] = M[j, i] = p["error_correlation"]
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
        cmap.set_bad("#ffffff")
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        im = ax.imshow(M, vmin=-1, vmax=1, cmap=cmap)
        ax.set_xticks(range(K), [ORACLE_SHORT.get(n, n) for n in names], fontsize=7, rotation=35, ha="right")
        ax.set_yticks(range(K), [ORACLE_SHORT.get(n, n) for n in names], fontsize=7)
        ax.grid(False)
        for i in range(K):
            for j in range(K):
                if i != j:
                    ax.text(j, i, "n/a" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}".replace("-", "\N{MINUS SIGN}"), ha="center", va="center",
                            fontsize=7.5, color=INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Error correlation $\\phi$", fontsize=8)
        cb.outline.set_visible(False)
        _save(fig, out, "oracle_error_correlation")
    except Exception as exc:  # pragma: no cover
        log.warning("figure oracle_error_correlation failed: %s", exc)


def b_search(cfg, ws, R, N, out) -> None:
    r = R.get("search")
    if not r:
        placeholder(out, "search", "Hyperparameter search space and selected values.")
        return
    best = r["best_params"]
    names = {"eta": "Learning rate $\\eta$", "max_depth": "Maximum tree depth", "min_child_weight": "Minimum child weight",
             "gamma_split": "Minimum split gain $\\gamma_{\\mathrm{split}}$", "reg_lambda": "$L_2$ leaf penalty $\\lambda$",
             "colsample_bytree": "Column sampling ratio", "subsample": "Row sampling ratio",
             "focal_gamma": "Focal focusing parameter $\\gamma_f$", "delta": "Gradient clip $\\delta$",
             "class_weight": "Class weighting $\\alpha$"}
    t = Table("search",
              f"Search space of the detector and selected configuration ({r['n_trials']} trials: {r['n_random']} random, "
              f"{r['n_trials'] - r['n_random']} TPE; criterion: pseudo-label F1 on validation rows, so no ground-truth "
              "label enters the search).",
              "lll", [["Hyperparameter", "Range", "Selected"]], size=r"\footnotesize")
    for k, spec in r["space"].items():
        if spec["type"] == "categorical":
            rng = "\\{" + ", ".join(param(None if str(c).lower() == "none" else c) for c in spec["choices"]) + "\\}"
        else:
            rng = f"[{param(spec['low'])}, {param(spec['high'])}]" + (" (log)" if spec["type"] == "loguniform" else "") \
                  + (" (integer)" if spec["type"] == "int" else "")
        t.row(names.get(k, tex(k)), rng, param(best.get(k)))
        N.put(f"best.{k}", best.get(k) if best.get(k) is not None else "none", param(best.get(k)))
    t.write(out)
    N.num("search.best_value", r["best_value"], 4)
    vals = [t.get("value") for t in r.get("trials", []) if t.get("value") is not None]
    if vals:
        N.num("search.worst_value", float(min(vals)), 4)
        N.num("search.median_value", float(np.median(vals)), 4)
        N.put("search.n_trials", len(vals))
    N.put("search.seconds", float(r["seconds"]), f(r["seconds"] / 60.0, 1))
    N.int("search.rows", r["search_rows"])
    N.int("search.val_rows", r["val_rows"])
    save_csv(out, "search_trials", r["trials"])
    try:
        plt = _mpl()
        tr = pd.DataFrame(r["trials"]).sort_values("trial")
        fig, ax = plt.subplots(figsize=(3.6, 2.3))
        rnd = tr["phase"] == "random"
        ax.plot(tr.loc[rnd, "trial"] + 1, tr.loc[rnd, "value"], "o", color=C1, label="Random trials",
                markeredgecolor="white", markeredgewidth=0.8)
        ax.plot(tr.loc[~rnd, "trial"] + 1, tr.loc[~rnd, "value"], "s", color=C2, label="TPE trials",
                markeredgecolor="white", markeredgewidth=0.8)
        ax.plot(tr["trial"] + 1, tr["value"].cummax(), "-", color=MUTED, linewidth=1.2, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Pseudo-label F1 (validation)")
        ax.legend(loc="lower right")
        _save(fig, out, "search_trials")
    except Exception as exc:  # pragma: no cover
        log.warning("figure search_trials failed: %s", exc)


def b_main(cfg, ws, R, N, out) -> None:
    r = R.get("main")
    if not r:
        for n in ("main", "main_seeds", "confusion", "family", "pseudo_labels", "selected_features"):
            placeholder(out, n, "Detection performance of the complete pipeline.")
        return
    s = r["summary"]
    seeds = r["seeds"]
    keys = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("accuracy", "Accuracy"),
            ("fpr", "FPR"), ("auc_roc", "AUC-ROC"), ("pr_auc", "PR-AUC")]
    b = r.get("bootstrap_ci_representative") or {}
    nd = lambda k: 4 if k == "fpr" else 3
    t = Table("main",
              f"Detection performance of the complete pipeline on the test split, evaluated against ground truth "
              f"(decision threshold calibrated on pseudo-labels). Mean and sample standard deviation over the "
              f"{len(seeds)} model seeds, 95\\% $t$-interval of the seed mean, and 95\\% bootstrap interval over test "
              f"flows for seed {r['representative_seed']}. Per-seed values are in the file "
              r"\texttt{results/main\_per\_seed.csv} of the code repository.",
              "lrrrr", [["Metric", "Mean", "Std. dev.", "95\\% CI (seeds)", "95\\% CI (test flows)"]])
    for k, lab in keys:
        boot = f"[{f(b[k]['lo'], nd(k))}, {f(b[k]['hi'], nd(k))}]" if k in b else "--"
        t.row(lab, f(s[k]["mean"], nd(k)), f(s[k]["std"], 4), ci(s[k], nd(k)), boot)
    t.write(out)
    t = Table("main_seeds",
              "Per-seed detection performance of the complete pipeline on the test split (the seed changes the row "
              "subsampling of the objective, the column sampling of the trees and the early-stopping sample; the "
              "prepared data are identical).",
              "l" + "r" * (len(keys) + 2), [["Seed"] + [k[1] for k in keys] + ["Threshold", "Trees"]], size=r"\footnotesize")
    for run in r["runs"]:
        t.row(run["seed"], *[f(run[k], nd(k)) for k, _ in keys], f(run["threshold"], 3), run["n_trees"])
    t.write(out)
    save_csv(out, "main_per_seed", [{k: v for k, v in run.items() if not isinstance(v, (dict, list))} for run in r["runs"]])
    for k, _ in keys:
        N.summary(f"main.{k}", s[k], 4 if k == "fpr" else 3)
        if k in b:
            N.num(f"main.{k}.boot_lo", b[k]["lo"], 4 if k == "fpr" else 3)
            N.num(f"main.{k}.boot_hi", b[k]["hi"], 4 if k == "fpr" else 3)
    N.pct("main.fpr.pct", s["fpr"]["mean"], 2)
    N.summary("main.macro_recall", s.get("macro_recall"))
    N.put("main.rep_seed", int(r["representative_seed"]))
    N.put("main.n_seeds", len(seeds))
    # the paired bootstrap against the baselines uses the first seed: how that seed ranks among the seeds
    f1s = [float(x["f1"]) for x in r["runs"]]
    if f1s:
        N.num("main.f1.first_seed", f1s[0], 4)
        N.num("main.f1.mean4", float(np.mean(f1s)), 4)
        rank = 1 + sum(v > f1s[0] for v in f1s)
        N.put("main.f1.first_seed_rank", rank,
              {1: "the highest", 2: "the second highest"}.get(rank, f"rank {rank}") + f" of the {len(f1s)} seeds")
        # seed-to-seed spread against the sampling error of the test set (standard error implied by the 95% bootstrap
        # interval of the representative seed, i.e. its width divided by 2 x 1.96)
        if len(f1s) > 1 and "f1" in b:
            sd = float(np.std(f1s, ddof=1))
            se = (b["f1"]["hi"] - b["f1"]["lo"]) / (2 * 1.959964)
            N.num("main.f1.std4", sd, 4)
            N.num("main.f1.min4", min(f1s), 4)
            N.num("main.f1.max4", max(f1s), 4)
            N.num("main.f1.boot_se", se, 4)
            N.num("main.f1.boot_width", b["f1"]["hi"] - b["f1"]["lo"], 4)
            if se > 0:
                N.num("main.f1.sd_over_se", sd / se, 1)
    N.num("main.trees", float(np.mean([x["n_trees"] for x in r["runs"]])), 0)
    N.num("main.threshold", float(np.mean([x["threshold"] for x in r["runs"]])), 3)
    N.num("main.fit_seconds", float(np.mean([x["fit_seconds"] for x in r["runs"]])), 1)
    obj = r["runs"][0].get("objective") or {}
    N.num("main.alpha_mal", obj.get("alpha_malicious_mean"), 2)
    N.num("main.alpha_ben", obj.get("alpha_benign_mean"), 3)
    N.pct("main.clipped_share", obj.get("clipped_share_mean"), 1)
    N.pct("main.neg_hess_share", obj.get("negative_hessian_share_max"), 2)
    pl_test = r["runs"][0].get("pseudo_label_test") or {}
    for k in ("precision", "recall", "f1"):
        N.num(f"main.vs_pseudo.{k}", pl_test.get(k))
    # ---- confusion matrix and per-family table of the same seed (they add up by construction)
    rep = r["representative"]
    t = Table("confusion",
              f"Test-set confusion matrix of the complete pipeline (seed {r['representative_seed']}, the seed whose F1 "
              "is closest to the seed mean).",
              "lrrr", [["", "Predicted malicious", "Predicted benign", "Total"]])
    t.row("Actual malicious", integer(rep["tp"]), integer(rep["fn"]), integer(rep["tp"] + rep["fn"]))
    t.row("Actual benign", integer(rep["fp"]), integer(rep["tn"]), integer(rep["fp"] + rep["tn"]))
    t.rule().row("Total", integer(rep["tp"] + rep["fp"]), integer(rep["fn"] + rep["tn"]),
                 integer(rep["tp"] + rep["fn"] + rep["fp"] + rep["tn"]))
    t.write(out)
    for k in ("tp", "fn", "fp", "tn"):
        N.int(f"conf.{k}", rep[k])
    N.int("conf.pos", rep["tp"] + rep["fn"])
    N.int("conf.neg", rep["fp"] + rep["tn"])
    for k in ("precision", "recall", "f1", "accuracy"):
        N.num(f"conf.{k}", rep[k])
    fam = {x["family"]: x for x in rep["families"]}
    order = fam_sorted(fam.keys())
    cp_note, cp_hw, cp_ends = binomial_note(
        [(fm, _family_text(fm), fam[fm]["support"], fam[fm]["tp"]) for fm in order],
        f"counts of seed {r['representative_seed']}")
    t = Table("family",
              f"Detection coverage per attack family for the same model and seed as Table~\\ref{{tab:confusion}} "
              f"(seed {r['representative_seed']}); every family of the test split is listed, so the totals equal the "
              f"malicious row of the confusion matrix. The last column is the mean $\\pm$ standard deviation of the "
              f"recall over the {len(seeds)} seeds.",
              "lrrrrr", [["Family", "Test flows", "Detected (TP)", "Missed (FN)", "Recall", "Recall, all seeds"]],
              notes=cp_note or None)
    for fm, (lo_, hi_) in cp_ends.items():
        N.num(f"family.{fm}.cp_lo", lo_)
        N.num(f"family.{fm}.cp_hi", hi_)
    if cp_hw is not None:
        N.num("family.cp_halfwidth_large", cp_hw)
    N.int("rule.large_family", LARGE_FAMILY)
    ttp = tfn = tsup = 0
    for fm in order:
        x = fam[fm]
        t.row(FAMILY_LABEL.get(fm, fm), integer(x["support"]), integer(x["tp"]), integer(x["fn"]), f(x["recall"]),
              pm(s["families"].get(fm)))
        ttp, tfn, tsup = ttp + x["tp"], tfn + x["fn"], tsup + x["support"]
        N.int(f"family.{fm}.support", x["support"])
        N.int(f"family.{fm}.tp", x["tp"])
        N.int(f"family.{fm}.fn", x["fn"])
        N.num(f"family.{fm}.recall", x["recall"])
        N.summary(f"family.{fm}.recall_seeds", s["families"].get(fm))
    t.rule().row("All malicious", integer(tsup), integer(ttp), integer(tfn), f(ttp / max(1, tsup)), pm(s["recall"]))
    t.write(out)
    N.put("family.total_matches_confusion", int(ttp == rep["tp"] and tfn == rep["fn"]),
          "yes" if (ttp == rep["tp"] and tfn == rep["fn"]) else "NO")
    rec = {fm: fam[fm]["recall"] for fm in order}
    if rec:
        lo, hi = min(rec, key=rec.get), max(rec, key=rec.get)
        # every family that ties at the printed precision is named, so that a tie is not hidden
        N.put("family.best", join_words([FAMILY_TEXT.get(x, x) for x in ties(rec, True)]))
        N.put("family.worst", join_words([FAMILY_TEXT.get(x, x) for x in ties(rec, False)]))
        N.num("family.best.recall", rec[hi])
        N.num("family.worst.recall", rec[lo])
        # families of which no test flow is detected (named in the text, so that a tie at zero is not hidden)
        zero = [fm for fm in fam_sorted(rec) if rec[fm] == 0]
        N.put("family.zero", join_words([FAMILY_TEXT.get(z, z) for z in zero]) or "none")
        N.put("family.n_zero", len(zero))
        N.put("family.n", len(rec))
    # ---- pseudo-label quality
    pl = r["pseudo_labels"]
    refine_on = bool((r["pipeline"].get("refinement") or {}).get("enabled", cfg.get_path("refinement.enabled")))
    if refine_on:
        cap = ("Quality of the pseudo-labels against ground truth (never used for training): after aggregation, after "
               "refinement, and on the test split.")
        rows_pl = (("train_before_refinement", "Training, after aggregation"),
                   ("train_after_refinement", "Training, after refinement"),
                   ("test_after_refinement", "Test, after refinement"))
    else:                                   # the selected configuration has no refinement: one row per split
        cap = ("Quality of the pseudo-labels against ground truth (never used for training) on the training and on the "
               "test split. The selected configuration applies no refinement; what the refinement of the first version "
               "would do to these labels is given in Step~2 of Section~\\ref{sec:method} and in "
               "Section~\\ref{sec:oracles}.")
        rows_pl = (("train_after_refinement", "Training"), ("test_after_refinement", "Test"))
    t = Table("pseudo_labels", cap,
              "lrrrrr", [["Pseudo-labels", "Labelled (\\%)", "Precision", "Recall", "F1", "Accuracy"]])
    for key, label in rows_pl:
        q = pl[key]
        t.row(label, pct(q["labelled_share"]), f(q["precision"]), f(q["recall"]), f(q["f1"]), f(q["accuracy"]))
    for key in ("train_before_refinement", "train_after_refinement", "test_after_refinement"):
        q = pl[key]
        short = {"train_before_refinement": "pl.agg", "train_after_refinement": "pl.train", "test_after_refinement": "pl.test"}[key]
        for k in ("precision", "recall", "f1", "accuracy"):
            N.num(f"{short}.{k}", q[k])
        N.pct(f"{short}.error", 1.0 - q["accuracy"], 2)
        N.pct(f"{short}.labelled", q["labelled_share"])
        N.int(f"{short}.fp", q["fp"])
        N.int(f"{short}.fn", q["fn"])
    t.write(out)
    fam_pl = {x["family"]: x for x in pl["train_after_refinement"].get("families", [])}
    for fm, x in fam_pl.items():
        N.num(f"pl.family.{fm}.recall", x["recall"])
    save_csv(out, "pseudo_label_recall_by_family", [{"family": k, **v} for k, v in fam_pl.items()])
    # ---- pipeline facts
    P = r["pipeline"]
    fp = P["feature_pipeline"]
    N.put("feat.d_stat", int(fp["d_statistical"]))
    N.put("feat.d_temporal", int(fp["d_temporal"]))
    N.put("feat.d_latent", int(fp["d_latent"]))
    N.put("feat.d_total", int(fp["d_total"]))
    ae = fp.get("autoencoder") or {}
    if ae:
        N.put("ae.epochs_run", ae.get("epochs_run"))
        N.num("ae.best_val_mse", ae.get("best_val_mse"), 4)
        N.int("ae.train_rows", ae.get("train_rows"))
        N.int("ae.val_rows", ae.get("val_rows"))
        if ae.get("architecture"):
            N.put("ae.architecture", "-".join(str(x) for x in ae["architecture"]),
                  "--".join(str(x) for x in ae["architecture"]))
    pre = fp.get("preprocessing") or {}
    N.put("pre.n_clusters", pre.get("n_clusters"))
    N.put("pre.features_with_missing", pre.get("features_with_missing"))
    N.pct("pre.max_missing_share", pre.get("max_missing_share"), 1)
    sel = P["selection"]
    for k in ("n_stable", "n_borderline", "n_borderline_kept", "n_selected", "rows_used"):
        if isnum(sel.get(k)):
            N.int(f"sel.{k}", sel[k])
    N.num("sel.seconds", sel.get("seconds"), 0)
    ref = P["refinement"]
    for k in ("n_low", "n_no_majority", "n_relabelled", "n_no_majority_assigned", "train_low_trust", "train_relabelled",
              "fit_rows", "n_high"):
        if isnum(ref.get(k)):
            N.int(f"refine.{k}", ref[k])
    if isnum(ref.get("n_low")) and ref.get("n_total"):
        N.pct("refine.low_share", ref["n_low"] / ref["n_total"], 2)
    names_sel = r["selected_features"]
    groups = {"statistical": [n for n in names_sel if n.startswith(("s_", "h_"))],
              "temporal": [n for n in names_sel if n.startswith("t_")],
              "latent": [n for n in names_sel if n.startswith("z_")]}
    for g, v in groups.items():
        N.put(f"sel.count.{g}", len(v))
    stab = sel.get("stability_scores") or {}
    mi = sel.get("mutual_information") or {}
    save_csv(out, "selected_features", [{"feature": n, "selected": int(n in names_sel), "stability": stab.get(n, 0.0),
                                         "mutual_information_nats": mi.get(n)} for n in r["feature_names"]])
    # the experiments run after the default pipeline was fitted again (see Section "Implementation details") may
    # work with another selection; it is listed below the main one
    refit = (R.get("interpret") or {}).get("selected_features")
    if refit:                       # whether the later fit of the default pipeline selected the same features
        N.put("sel.refit_same", int(set(refit) == set(names_sel)), "yes" if set(refit) == set(names_sel) else "no")
        N.put("sel.n_refit_checked", len(refit))
    refit = list(refit) if refit and set(refit) != set(names_sel) else None
    cap = (f"Features retained by the stability-aware selection in the main experiment ({len(names_sel)} of "
           f"{len(r['feature_names'])})")
    if refit:
        cap += (f"; below, the {len(refit)} features of the pipeline that was fitted again during the ablations and used "
                r"by the later experiments (Section~\ref{sec:impl}); features that only one of the two selections "
                "contains are marked with an asterisk")
    t = Table("selected_features",
              cap + ". The complete list with stability scores and mutual information is in the file "
              r"\texttt{results/selected\_features.csv} of the code repository.",
              "p{0.20\\linewidth}p{0.74\\linewidth}", [["Subspace", "Selected features"]], size=r"\scriptsize")
    mark = lambda n, other: tt(n) + ("$^{*}$" if other is not None and n not in other else "")
    for g in ("statistical", "temporal", "latent"):
        t.row(f"{SUBSPACE_LABEL[g]} ({len(groups[g])})", ", ".join(mark(n, refit) for n in groups[g]) or "--")
    N.put("sel.n_main", len(names_sel))
    if refit:
        g2 = {"statistical": [n for n in refit if n.startswith(("s_", "h_"))],
              "temporal": [n for n in refit if n.startswith("t_")],
              "latent": [n for n in refit if n.startswith("z_")]}
        t.rule()
        t.row(r"\multicolumn{2}{l}{\emph{Pipeline fitted again during the ablations}}")
        for g in ("statistical", "temporal", "latent"):
            t.row(f"{SUBSPACE_LABEL[g]} ({len(g2[g])})", ", ".join(mark(n, names_sel) for n in g2[g]) or "--")
            N.put(f"sel.refit.count.{g}", len(g2[g]))
        N.put("sel.n_refit", len(refit))
        N.put("sel.n_common", len(set(refit) & set(names_sel)))
    t.write(out)
    # timing of the default pipeline
    tm = P.get("timing") or {}
    for k in ("stage_a", "stage_b", "stage_c"):
        N.num(f"time.{k}", tm.get(k), 0)
    for k, v in (tm.get("oracles") or {}).items():
        N.num(f"time.oracle.{k}", v, 1)
    od = P.get("oracle_describe") or {}
    cl = od.get("cluster") or {}
    for k, short in (("train_core_share", "core"), ("train_border_share", "border"), ("train_noise_share", "noise"),
                     ("pca_explained_variance", "pca_var")):
        if isnum(cl.get(k)):
            N.pct(f"cluster.{short}", cl[k], 1)
    N.int("cluster.fit_rows", cl.get("fit_rows"))
    tr = od.get("transfer") or {}
    for k in ("source_rows", "source_holdout_accuracy", "source_holdout_f1", "source_malicious_share"):
        if isnum(tr.get(k)):
            N.put(f"transfer.{k}", tr[k], integer(tr[k]) if k == "source_rows" else f(tr[k], 3))
    dm = od.get("domain") or {}
    for k, v in dm.items():
        if isnum(v):
            N.put(f"domain.{k}", v, integer(v) if float(v).is_integer() else f(v, 3))
    # ---- robustness to the pipeline seed
    if r.get("pipeline_seed_summary"):
        ps = r["pipeline_seed_summary"]
        for k in ("precision", "recall", "f1"):
            N.summary(f"pipe.{k}", ps[k])
        N.put("pipe.n", len(r["pipeline_seed_runs"]))
        ns = [x.get("n_selected") for x in r["pipeline_seed_runs"] if isnum(x.get("n_selected"))]
        if ns:
            N.put("pipe.n_selected_min", int(min(ns)))
            N.put("pipe.n_selected_max", int(max(ns)))
        save_csv(out, "pipeline_seed_runs", [{k: v for k, v in x.items() if not isinstance(v, (dict, list))}
                                             for x in r["pipeline_seed_runs"]])


ABL_TABLES = {
    "objective": ("abl_objective", "Ablation of the training objective"),
    "selection": ("abl_selection", "Ablation of the feature-selection stage"),
    "refinement": ("abl_refinement", "Ablation of the semi-supervised refinement"),
    "other": ("abl_other", "Ablation of preprocessing, feature subspaces and oracle settings"),
}


def b_ablations(cfg, ws, R, N, out) -> None:
    r = R.get("ablations")
    if not r or not r.get("variants"):
        for name, what in ABL_TABLES.values():
            placeholder(out, name, what + ".")
        return
    V = r["variants"]
    ref = next((v for v in V if v["group"] == "reference" and v["name"].startswith("full method") and "summary" in v), None)
    sup = next((v for v in V if v["name"].startswith("supervised reference") and "summary" in v), None)
    fsup = next((v for v in V if v["name"].startswith("fully supervised reference") and "summary" in v), None)
    n_seeds = len(r.get("seeds", []))
    rows_csv = []
    for v in V:
        if "summary" in v:
            s = v["summary"]
            rows_csv.append({"group": v["group"], "variant": v["name"], "n_features": v.get("n_features"),
                             "mean_trees": v.get("mean_trees"), "clipped_share": v.get("clipped_share"),
                             **{f"{k}_{st}": s[k][st] for k in ("precision", "recall", "f1", "fpr", "pr_auc")
                                for st in ("mean", "std")}})
            slug = re.sub(r"[^a-z0-9]+", "_", (v["group"] + " " + v["name"]).lower()).strip("_")
            N.summary(f"abl.{slug}.f1", s["f1"])
            N.summary(f"abl.{slug}.precision", s["precision"])
            N.summary(f"abl.{slug}.recall", s["recall"])
            for k_ in ("pr_auc", "auc_roc"):                # ranking quality (threshold-free)
                if isnum((s.get(k_) or {}).get("mean")):
                    N.summary(f"abl.{slug}.{k_}", s[k_])
            if isnum((s.get("fpr") or {}).get("mean")):
                N.pct(f"abl.{slug}.fpr", s["fpr"]["mean"], 1)
            if ref is not None:
                d = s["f1"]["mean"] - ref["summary"]["f1"]["mean"]
                N.put(f"abl.{slug}.df1", float(d), signed(d))
                N.put(f"abl.{slug}.df1_4d", float(d), signed(d, 4))
                # does every seed of the variant lie below (above) every seed of the full method?
                vs, rs = s["f1"], ref["summary"]["f1"]
                if all(isnum(x) for x in (vs.get("max"), vs.get("min"), rs.get("min"), rs.get("max"))) and v is not ref:
                    N.put(f"abl.{slug}.all_seeds_below", "yes" if vs["max"] < rs["min"] else "no")
                    N.put(f"abl.{slug}.all_seeds_above", "yes" if vs["min"] > rs["max"] else "no")
            if isnum(v.get("clipped_share")):
                N.pct(f"abl.{slug}.clipped", v["clipped_share"], 1)
            pl = v.get("pseudo_labels_train") or {}
            if "tp" in pl:                        # training pseudo-labels of the variant (after refinement)
                N.int(f"abl.{slug}.label_tp", pl["tp"])
                N.int(f"abl.{slug}.label_fp", pl["fp"])
                N.num(f"abl.{slug}.label_precision", pl["precision"])
    save_csv(out, "ablations", rows_csv)
    if ref and "tp" in (ref.get("pseudo_labels_train") or {}):
        rp = ref["pseudo_labels_train"]
        same = lambda v: (v.get("pseudo_labels_train") or {}).get("tp") == rp["tp"] and \
            (v.get("pseudo_labels_train") or {}).get("fp") == rp["fp"]
        var = [v for v in V if "summary" in v and v is not ref and "tp" in (v.get("pseudo_labels_train") or {})]
        N.put("abl.n_same_labels", sum(1 for v in var if same(v)))
        N.put("abl.n_with_labels", len(var))
        fv = [v for v in var if "first version" in v["name"]]
        if fv:
            N.put("abl.first_version_same_labels", "yes" if all(same(v) for v in fv) else "no")
        N.int("abl.ref.label_tp", rp["tp"])
        N.int("abl.ref.label_fp", rp["fp"])
    if ref:
        N.summary("abl.ref.f1", ref["summary"]["f1"])
        # largest change of the F1-score among all variants of the method (the supervised references excluded)
        ds_ = [abs(v["summary"]["f1"]["mean"] - ref["summary"]["f1"]["mean"]) for v in V
               if "summary" in v and v is not ref and not v["name"].startswith(("supervised reference", "fully supervised"))]
        if ds_:
            N.num("abl.max_abs_df1", max(ds_))
            N.put("abl.n_variants", len(ds_))
        for g in sorted({v["group"] for v in V}):
            dg = [abs(v["summary"]["f1"]["mean"] - ref["summary"]["f1"]["mean"]) for v in V
                  if v["group"] == g and "summary" in v and v is not ref
                  and not v["name"].startswith(("supervised reference", "fully supervised"))]
            if dg:
                gs = re.sub(r"[^a-z0-9]+", "_", g.lower()).strip("_")
                N.num(f"abl.max_abs_df1.{gs}", max(dg))
                # an upper bound at four decimals (rounded up): "at most 0.001" would hold only after rounding
                N.put(f"abl.max_abs_df1_bound.{gs}", max(dg), f(math.ceil(max(dg) * 1e4 - 1e-9) / 1e4, 4))
                N.put(f"abl.n_variants.{gs}", len(dg))
        # the stages that act on fixed pseudo-labels, and those that change the pseudo-labels
        for nm, gr in (("detector_stages", ("objective", "selection", "preprocessing", "subspaces")),
                       ("training_stages", ("objective", "selection", "preprocessing")),   # feature subspaces excluded
                       ("labelling", ("refinement", "oracle settings"))):
            dg = [abs(v["summary"]["f1"]["mean"] - ref["summary"]["f1"]["mean"]) for v in V
                  if v["group"] in gr and "summary" in v and v is not ref
                  and not v["name"].startswith(("supervised reference", "fully supervised"))]
            if dg:
                N.num(f"abl.max_abs_df1.{nm}", max(dg))
                N.put(f"abl.max_abs_df1_bound.{nm}", max(dg), f(math.ceil(max(dg) * 1e4 - 1e-9) / 1e4, 4))
                N.put(f"abl.n_variants.{nm}", len(dg))
        nt = [v for v in V if "summary" not in v]
        N.put("abl.n_not_run", len(nt))
    if sup:
        for k in ("precision", "recall", "f1"):
            N.summary(f"abl.supervised.{k}", sup["summary"][k])
        if ref:
            d = sup["summary"]["f1"]["mean"] - ref["summary"]["f1"]["mean"]
            N.put("abl.supervised.gap", float(d), signed(d))
    if fsup:
        for k in ("precision", "recall", "f1"):
            N.summary(f"abl.fully_supervised.{k}", fsup["summary"][k])
        if ref:
            d = fsup["summary"]["f1"]["mean"] - ref["summary"]["f1"]["mean"]
            N.put("abl.fully_supervised.gap", float(d), signed(d))

    def delta(s):
        if ref is None:
            return "--"
        d = s["f1"]["mean"] - ref["summary"]["f1"]["mean"]
        return signed(d)

    def table(name, caption, groups, extra_cols=(), with_ref=True):
        hdr = ["Variant", "Precision", "Recall", "F1", "$\\Delta$F1"] + [c[0] for c in extra_cols]
        t = Table(name, caption + f" Test split, mean $\\pm$ standard deviation over {n_seeds} model seeds; "
                  "$\\Delta$F1 is the difference to the full method.",
                  "p{0.33\\linewidth}rrrr" + "r" * len(extra_cols), [hdr], size=r"\scriptsize")
        if with_ref and ref:
            t.row("Full method (selected configuration)", *metric_cells(ref["summary"]), "--", *[c[1](ref) for c in extra_cols])
            t.rule()
        any_row = False
        for v in V:
            if v["group"] not in groups:
                continue
            if "summary" not in v:
                t.row(tex(v["name"]), r"\multicolumn{" + str(4 + len(extra_cols)) + "}{l}{" + tex(v.get("status", "not run")) + "}")
                continue
            t.row(tex(v["name"]).replace("gamma = ", "$\\gamma_f$ = ").replace("delta = ", "$\\delta$ = "),
                  *metric_cells(v["summary"]), delta(v["summary"]), *[c[1](v) for c in extra_cols])
            any_row = True
        if name == "abl_objective" and (sup or fsup):
            t.rule()
            if sup:
                t.row("Supervised reference: ground-truth training labels", *metric_cells(sup["summary"]),
                      delta(sup["summary"]), *["--" for _ in extra_cols])
            if fsup:
                t.row("Fully supervised reference: ground truth also for early stopping and threshold",
                      *metric_cells(fsup["summary"]), delta(fsup["summary"]), *["--" for _ in extra_cols])
        if any_row:
            t.write(out)
        else:
            placeholder(out, name, caption)

    table("abl_objective",
          "Step-by-step ablation of the training objective, from the plain logistic loss to the full objective, and "
          "variants of the selected configuration. PR-AUC is threshold-free (mean $\\pm$ standard deviation over the seeds). "
          "``Clipped'' is the mean share of drawn rows whose weighted gradient is clipped. The supervised references train the same detector on ground-truth labels and bound what "
          "better pseudo-labels could deliver.",
          ("objective",),
          extra_cols=(("PR-AUC", lambda v: pm((v.get("summary") or {}).get("pr_auc"))),
                      ("Clipped (\\%)", lambda v: pct(v.get("clipped_share"), 1)), ("Trees", lambda v: f(v.get("mean_trees"), 0))))
    table("abl_selection",
          "Ablation of the feature-selection stage: no selection, single-criterion rankings, stability threshold "
          "$\\theta$, ranking depth $k$ and the borderline rule. PR-AUC is threshold-free (mean $\\pm$ standard "
          "deviation over the seeds).",
          ("selection",), extra_cols=(("Features", lambda v: str(v.get("n_features", "--"))),
                                      ("PR-AUC", lambda v: pm((v.get("summary") or {}).get("pr_auc")))))
    refine_added = any(v["group"] == "refinement" and v["name"].startswith("with") for v in V)
    table("abl_refinement",
          ("The semi-supervised refinement, which the selected configuration omits (Section~\\ref{sec:method}), added with "
           "the thresholds of the first version and with two other thresholds. ``Label precision'' and ``Label accuracy'' "
           "refer to the training pseudo-labels after the refinement." if refine_added else
           "Ablation of the semi-supervised refinement and of its thresholds."),
          ("refinement",), extra_cols=((("Label precision", lambda v: f((v.get("pseudo_labels_train") or {}).get("precision"))),)
                                       if refine_added else ()) +
                                      (("Label accuracy", lambda v: f((v.get("pseudo_labels_train") or {}).get("accuracy"))),))
    # summary for the main text: one or two variants per stage (all rows also in the four detailed tables)
    pick = [("Training objective", "objective", "logistic loss, no weights", "logistic loss without weights"),
            ("", "objective", "+ trust weights", "+ trust weights"),
            ("", "objective", "+ class weights (reciprocal)", "+ reciprocal class weights (selected objective)"),
            ("", "objective", "+ focal term (gamma = 2)", "+ focal term ($\\gamma_f=2$)"),
            ("", "objective", "+ gradient clipping (delta = 0.1)", "+ focal term and gradient clipping ($\\delta=0.1$)"),
            ("", "objective", "selected configuration without trust weights", "selected objective without trust weights"),
            ("", "objective", "selected configuration, class weighting none", "selected objective without class weights"),
            ("Feature selection", "selection", "no selection (all features)", "no selection (all features)"),
            ("", "selection", "mutual information only", "mutual information only"),
            ("", "selection", "random-forest ranking only", "random-forest ranking only"),
            ("", "selection", "stability threshold 0.6", "stability threshold $\\theta_{\\mathrm{stab}}=0.6$"),
            ("", "selection", "stability threshold 1.0", "stability threshold $\\theta_{\\mathrm{stab}}=1.0$"),
            ("", "selection", "k = 20", "ranking depth $k=20$"),
            ("", "selection", "k = 60", "ranking depth $k=60$"),
            ("Features", "preprocessing", "without imputation and scaling", "without imputation and scaling"),
            ("", "subspaces", "without latent features", "without latent features"),
            ("", "subspaces", "without host-level temporal features", "without host-level features"),
            ("Labelling stage", "refinement", "with the semi-supervised refinement of the first version",
             "with the refinement of the initial design"),
            ("", "refinement", "with refinement, threshold 0.95", "with refinement, relabelling threshold 0.95"),
            ("", "refinement", "with refinement, high-trust threshold 0.6", "with refinement, high-trust threshold 0.6"),
            ("", "oracle settings", "label model without merging dependent oracles", "label model without merging"),
            ("", "oracle settings", "signature oracle counts every Snort alert (first version)",
             "signature oracle counts every alert (initial rule)"),
            ("", "oracle settings", "fan-out threshold fixed at 15 (first version)", "fan-out threshold fixed at 15 (initial rule)"),
            ("", "oracle settings", "repeated-connection threshold at the 0.99 quantile",
             "repeated-connection threshold at the 99th percentile"),
            ("", "oracle settings", "repeated-connection threshold at the 0.9999 quantile",
             "repeated-connection threshold at the 99.99th percentile")]
    if ref:
        ts = Table("abl_summary",
                   "Summary of the ablation study: each row changes one element of the selected configuration "
                   f"(test split, mean $\\pm$ standard deviation over {n_seeds} model seeds; $\\Delta$F1 is the difference "
                   "to the full method). The supervised references train the same detector on ground-truth labels. "
                   "All variants, with precision, recall, and further settings, are in the file "
                   r"\texttt{results/ablations.csv} of the code repository.",
                   "llrrr", [["Stage", "Variant", "F1", "$\\Delta$F1", "PR-AUC"]], size=r"\footnotesize")
        ts.row("--", "Full method (selected configuration)", pm(ref["summary"]["f1"]), "--", pm(ref["summary"].get("pr_auc")))
        last = None
        for stage, grp, name, label in pick:
            v = next((x for x in V if x["group"] == grp and x["name"].replace("$\\gamma_f$", "gamma").replace("$\\delta$", "delta")
                      .startswith(name) and "summary" in x), None)
            if v is None:
                v = next((x for x in V if x["group"] == grp and x["name"].startswith(name) and "summary" in x), None)
            if v is None:
                continue
            if stage and stage != last:
                ts.rule()
                last = stage
            ts.row(stage, label, pm(v["summary"]["f1"]), delta(v["summary"]), pm(v["summary"].get("pr_auc")))
        if sup or fsup:
            ts.rule()
            if sup:
                ts.row("References", "ground-truth training labels", pm(sup["summary"]["f1"]), delta(sup["summary"]),
                       pm(sup["summary"].get("pr_auc")))
            if fsup:
                ts.row("", "ground truth also for early stopping and threshold", pm(fsup["summary"]["f1"]),
                       delta(fsup["summary"]), pm(fsup["summary"].get("pr_auc")))
        ts.write(out)
    else:
        placeholder(out, "abl_summary", TABLE_PURPOSE["abl_summary"])
    table("abl_other",
          "Ablation of preprocessing and feature subspaces, and sensitivity to the oracle settings. PR-AUC is "
          "threshold-free (mean $\\pm$ standard deviation over the seeds).",
          ("preprocessing", "subspaces", "oracle settings"),
          extra_cols=(("PR-AUC", lambda v: pm((v.get("summary") or {}).get("pr_auc"))),
                      ("Label accuracy", lambda v: f((v.get("pseudo_labels_train") or {}).get("accuracy")))))


def b_fs(cfg, ws, R, N, out) -> None:
    r = R.get("fs_compare")
    if not r:
        placeholder(out, "fs", "Comparison of feature-selection methods under a fixed classifier.")
        return
    t = Table("fs",
              f"Feature-selection methods under the same classifier, hyperparameters and partition (test split, mean "
              f"$\\pm$ standard deviation over {len(r['seeds'])} model seeds). ``Time'' is the wall-clock time of the "
              "selection step on the row sample each method uses (see text).",
              "p{0.24\\linewidth}rrrrr", [["Method", "Features", "Precision", "Recall", "F1", "Time (s)"]], size=r"\footnotesize")
    rows_csv = []
    for row in r["rows"]:
        if "summary" not in row:
            t.row(tex(row["method"]), r"\multicolumn{5}{l}{" + tex(row.get("status", "not run")) + "}")
            continue
        s = row["summary"]
        t.row(tex(row["method"]), row["n_features"], *metric_cells(s), f(row["selection_seconds"], 1))
        rows_csv.append({"method": row["method"], "n_features": row["n_features"], "selection_seconds": row["selection_seconds"],
                         "detail": row.get("selector_info"),
                         **{f"{k}_{st}": s[k][st] for k in ("precision", "recall", "f1") for st in ("mean", "std")}})
        N.summary(f"fs.{row['key']}.f1", s["f1"])
        N.summary(f"fs.{row['key']}.precision", s["precision"])
        N.summary(f"fs.{row['key']}.recall", s["recall"])
        N.num(f"fs.{row['key']}.seconds", row["selection_seconds"], 1)
        N.put(f"fs.{row['key']}.n", int(row["n_features"]))
    t.write(out)
    save_csv(out, "feature_selection_comparison", rows_csv)
    done = [x for x in r["rows"] if "summary" in x]
    if done:
        best = max(done, key=lambda x: x["summary"]["f1"]["mean"])
        N.put("fs.best", best["method"], tex(best["method"]))
        N.put("fs.n_methods", len(done))
        N.num("fs.best.f1", best["summary"]["f1"]["mean"])
        f1s = [x["summary"]["f1"]["mean"] for x in done]
        N.num("fs.f1_range", max(f1s) - min(f1s))


def b_loao(cfg, ws, R, N, out) -> None:
    r = R.get("loao")
    if not r or not r.get("rows"):
        placeholder(out, "loao", "Leave-one-attack-out generalization.")
        return
    def _pl_heldout(row):
        """Recall of the pseudo-labels (oracles and label model fitted without the family) on its test flows."""
        fams = (row.get("pseudo_labels_test") or {}).get("families") or []
        fams = fams if isinstance(fams, list) else [{"family": k, **v} for k, v in fams.items()]
        x = next((x for x in fams if x.get("family") == row["family"]), None)
        return None if x is None else x.get("recall")

    has_pl = any(_pl_heldout(row) is not None for row in r["rows"])
    cp_note, cp_hw, cp_ends = binomial_note(
        [(row["family"], _family_text(row["family"]), row["support"],
          (row["held_out_recall"] or {}).get("mean", float("nan")) * row["support"]) for row in r["rows"]
         if isnum((row.get("held_out_recall") or {}).get("mean"))],
        "for the held-out family at the mean number of detected flows over the seeds; the interval column gives the "
        "seed-to-seed variation")
    for fm, (lo_, hi_) in cp_ends.items():
        N.num(f"loao.{fm}.cp_hi", hi_)
    if cp_hw is not None:
        N.num("loao.cp_halfwidth_large", cp_hw)
    t = Table("loao",
              f"Leave-one-attack-out: each family is removed from training and validation and appears only in the "
              f"test split. Recall on the held-out family"
              + (" of the pseudo-labels (oracles and label model fitted without the family) and of the detector"
                 if has_pl else "")
              + f", benign false-positive rate and mean recall on the other families (mean $\\pm$ standard deviation "
              f"over {len(r['seeds'])} seeds; interval: 95\\% $t$-interval).",
              "lrrrrrr" if has_pl else "lrrrrr",
              [["Held-out", "Test"] + (["Pseudo-"] if has_pl else []) + ["Recall", "", "Benign", "Recall"],
               ["family", "flows"] + (["labels"] if has_pl else []) + ["(held-out)", "95\\% CI", "FPR (\\%)",
                                                                        "(others)"]],
              size=r"\footnotesize", notes=cp_note or None)
    rows_csv = []
    pl_large = []
    for row in r["rows"]:
        h = row["held_out_recall"]
        fpr = row["benign_fpr"]
        plh = _pl_heldout(row)
        t.row(FAMILY_LABEL.get(row["family"], row["family"]), integer(row["support"]),
              *([f(plh)] if has_pl else []), pm(h), ci(h),
              pm_pct(fpr, 2), pm(row["recall_other_families"]))
        N.summary(f"loao.{row['family']}.recall", h)
        N.pct(f"loao.{row['family']}.fpr", fpr["mean"], 2)
        if plh is not None:
            N.num(f"loao.{row['family']}.pl_recall", plh)
            if row["support"] >= 1000:
                pl_large.append(plh)
        thr = ((row.get("oracle_describe") or {}).get("rate") or {}).get("threshold")
        if isnum(thr):
            N.num(f"loao.{row['family']}.rate_threshold", float(thr), 1)
        # coverage of the botnet test flows by the pseudo-labels of this run (the threshold moves with the withheld family)
        fams_pl = (row.get("pseudo_labels_test") or {}).get("families") or []
        fams_pl = fams_pl if isinstance(fams_pl, list) else [{"family": k, **v} for k, v in fams_pl.items()]
        xb = next((x for x in fams_pl if x.get("family") == "Bot"), None)
        if xb is not None:
            N.num(f"loao.{row['family']}.pl_bot", xb.get("recall"))
        for x in row["runs"]:
            rows_csv.append({"family": row["family"], **x})
    t.write(out)
    save_csv(out, "loao_per_seed", rows_csv)
    rec = {row["family"]: row["held_out_recall"]["mean"] for row in r["rows"]}
    lo, hi = min(rec, key=rec.get), max(rec, key=rec.get)
    N.put("loao.best", join_words([FAMILY_TEXT.get(x, x) for x in ties(rec, True)]))
    N.put("loao.worst", join_words([FAMILY_TEXT.get(x, x) for x in ties(rec, False)]))
    N.num("loao.best.recall", rec[hi])
    N.num("loao.worst.recall", rec[lo])
    N.put("loao.n_below_half", sum(v < 0.5 for v in rec.values()))
    fprs = [row["benign_fpr"]["mean"] for row in r["rows"] if isnum((row.get("benign_fpr") or {}).get("mean"))]
    if fprs:
        N.pct("loao.fpr_min", min(fprs), 2)
        N.pct("loao.fpr_max", max(fprs), 2)
    N.put("loao.n", len(rec))
    if pl_large:                               # families with enough test flows for a recall to mean something
        N.num("loao.pl_min_large", min(pl_large))
        N.num("loao.pl_max_large", max(pl_large))
    zero = [fm for fm in fam_sorted(rec) if rec[fm] == 0]
    N.put("loao.zero", join_words([FAMILY_TEXT.get(z, z) for z in zero]) or "none")
    N.put("loao.n_zero", len(zero))
    # held-out recall next to the recall of the same family when it is part of training (main experiment)
    fam_main = {x["family"]: x["recall"] for x in ((R.get("main") or {}).get("representative") or {}).get("families", [])}
    if fam_main:
        diffs = [abs(rec[fm] - fam_main[fm]) for fm in rec if fm in fam_main]
        if diffs:
            N.num("loao.max_abs_diff_to_main", max(diffs))


def _proto_label(label: str) -> str:
    """Row label of a partition protocol as it reads in the manuscript (result files name the primary partition
    after the first submission)."""
    return str(label).replace(" (first submission)", " (primary partition)")


def b_protocols(cfg, ws, R, N, out) -> None:
    r = R.get("protocols")
    if not r or not r.get("rows"):
        placeholder(out, "protocols", "Partition protocols and sensitivity to the host-level temporal features.")
        placeholder(out, "protocol_splits", "Composition of the splits under each partition protocol.")
        return
    t = Table("protocols",
              f"Partition protocols, each evaluated with and without the host-level features (windowed aggregates and "
              f"host statistics). Test split, mean $\\pm$ standard deviation over {len(r['seeds'])} model seeds. The "
              "size of the subsets under each protocol is in the file \\texttt{results/partition\\_protocols.csv} of the code "
              "repository.",
              "p{0.30\\linewidth}lrrr", [["Protocol", "Host features", "Precision", "Recall", "F1"]],
              size=r"\footnotesize")
    t2 = Table("protocol_splits",
               "Size of the training and test splits under each partition protocol, share of malicious flows in the "
               "test split and accuracy of the test pseudo-labels.",
               "p{0.34\\linewidth}rrrr", [["", "Training", "Test", "Malicious in", "Pseudo-label"],
                                           ["Protocol", "flows", "flows", "test (\\%)", "accuracy (test)"]],
               size=r"\footnotesize")
    rows_csv = []
    last = None
    for row in r["rows"]:
        lab = tex(_proto_label(row["label"])) if row["protocol"] != last else ""
        if row["protocol"] != last and last is not None:
            t.rule()
        first_of_protocol = row["protocol"] != last
        last = row["protocol"]
        hf = "with" if row["temporal"] == "global" else "without"
        if "summary" not in row:
            t.row(lab, hf, r"\multicolumn{3}{l}{" + tex(row.get("status", "not run")) + "}")
            continue
        st = {x["split"]: x for x in row["split_table"]}
        s = row["summary"]
        t.row(lab, hf, *metric_cells(s))
        if first_of_protocol:
            t2.row(tex(_proto_label(row["label"])), integer(st["train"]["flows"]), integer(st["test"]["flows"]),
                   pct(st["test"]["malicious_share"], 1), f((row.get("pseudo_labels_test") or {}).get("accuracy")))
        rows_csv.append({"protocol": row["protocol"], "host_features": hf, "train_flows": st["train"]["flows"],
                         "test_flows": st["test"]["flows"], "test_malicious_share": st["test"]["malicious_share"],
                         **{f"{k}_{sx}": s[k][sx] for k in ("precision", "recall", "f1", "fpr", "pr_auc") for sx in ("mean", "std")}})
        key = f"proto.{row['protocol']}.{'with' if row['temporal'] == 'global' else 'without'}"
        for k in ("precision", "recall", "f1"):
            N.summary(f"{key}.{k}", s[k])
        for k in ("auc_roc", "pr_auc"):                      # threshold-free: separates ranking from calibration
            if isinstance(s.get(k), dict):
                N.summary(f"{key}.{k}", s[k])
        if isinstance(s.get("recall"), dict) and isnum(s["recall"].get("min")):
            N.num(f"{key}.recall.min", s["recall"]["min"])
            N.num(f"{key}.recall.max", s["recall"]["max"])
        N.pct(f"{key}.fpr", s["fpr"]["mean"], 2)
        N.int(f"{key}.test_flows", st["test"]["flows"])
        N.int(f"{key}.train_flows", st["train"]["flows"])
        N.pct(f"{key}.test_malicious", st["test"]["malicious_share"], 1)
        plt_ = row.get("pseudo_labels_test") or {}
        for k in ("precision", "recall", "f1"):              # quality of the test pseudo-labels under this protocol
            if isnum(plt_.get(k)):
                N.num(f"{key}.pl_{k}", plt_[k])
        N.summary(f"{key}.macro_recall", s.get("macro_recall"))
        for fm, v in (s.get("families") or {}).items():
            N.num(f"{key}.family.{fm}", v["mean"])
        od = row.get("oracle_describe") or {}
        if (od.get("rate") or {}).get("threshold") is not None:     # threshold of the repeated-connection oracle
            N.num(f"{key}.rate_threshold", float(od["rate"]["threshold"]), 1)
        for nm_, plq in (("test", row.get("pseudo_labels_test")), ("train", row.get("pseudo_labels_train"))):
            for x in ((plq or {}).get("families") or []):
                if isinstance(x, dict) and isnum(x.get("recall")):
                    N.num(f"{key}.pl_{nm_}.family.{x['family']}", x["recall"])
    t.write(out)
    t2.write(out)
    save_csv(out, "partition_protocols", rows_csv)
    # differences that the text discusses
    get = lambda p, tflag: next((x["summary"]["f1"]["mean"] for x in r["rows"]
                                 if x["protocol"] == p and x["temporal"] == tflag and "summary" in x), None)
    base = get("random", "global")
    for p in ("conn", "chrono", "forward"):
        v = get(p, "global")
        if base is not None and v is not None:
            N.put(f"proto.{p}.df1", float(v - base), signed(v - base))
    gains = []
    for p in ("random", "conn", "chrono", "forward"):
        a, b = get(p, "global"), get(p, "none")
        if a is not None and b is not None:
            N.put(f"proto.{p}.host_gain", float(a - b), signed(a - b))
            gains.append(abs(a - b))
    if gains:                                   # largest effect of the host-level features under any protocol
        N.num("proto.max_abs_host_gain", max(gains))
    # the F1-score that the chronological protocol would reach with the botnet recall of the random partition (same
    # test counts and false-positive rate): how much of its higher score the coverage of the botnet explains
    rows_g = {x["protocol"]: x for x in r["rows"] if x.get("temporal") == "global" and "summary" in x}
    if "chrono" in rows_g and "random" in rows_g:
        sc, sr = rows_g["chrono"]["summary"], rows_g["random"]["summary"]
        fc, fr = sc.get("families") or {}, sr.get("families") or {}
        stc = {x["split"]: x for x in rows_g["chrono"].get("split_table") or []}
        if "Bot" in fc and "Bot" in fr and "test" in stc:
            pos = sum(v["support"] for v in fc.values())
            tp = sum(v["support"] * v["mean"] for v in fc.values())
            fp = sc["fpr"]["mean"] * (stc["test"]["flows"] - pos)
            tp2 = tp - fc["Bot"]["support"] * (fc["Bot"]["mean"] - fr["Bot"]["mean"])
            if tp2 > 0:
                pr, rc = tp2 / (tp2 + fp), tp2 / pos
                N.num("proto.chrono.f1_with_random_bot", 2 * pr * rc / (pr + rc))
    if r.get("forward_days"):
        for k, v in r["forward_days"].items():
            N.put(f"proto.forward.days.{k}", ", ".join(v), ", ".join(tex(x) for x in v))
    # which attack families the forward-in-time test split holds that its training split does not
    fwd = next((x for x in r["rows"] if x["protocol"] == "forward" and x.get("split_table")), None)
    if fwd:
        fams = {x["split"]: {k for k, n in (x.get("families") or {}).items() if k != "Benign" and n > 0}
                for x in fwd["split_table"]}

        def names(keys) -> str:
            return join_words([FAMILY_TEXT.get(k, k) for k in fam_sorted(keys)]) or "none"

        for split in ("train", "validation", "test"):
            N.put(f"proto.forward.families.{split}", names(fams.get(split, set())))
        unseen = fams.get("test", set()) - fams.get("train", set())
        N.put("proto.forward.unseen_families", names(unseen))
        N.put("proto.forward.n_unseen_families", len(unseen))
        N.put("proto.forward.n_test_families", len(fams.get("test", set())))
    # malicious flows of every attack per subset, from the corpus with the partition function and seed of the pipeline
    try:
        from ..corpus import TEST, TRAIN, VAL_A, VAL_B, make_split
        from ..pipeline import PrepOptions

        meta = ws.meta
        mal = meta["label"].to_numpy() == 1
        fam = meta["family"].astype(str).to_numpy()[mal]
        att = meta["attack"].astype(str).to_numpy()[mal]
        fwd = cfg.get_path("experiments.forward")
        part = {TRAIN: "train", VAL_A: "validation", VAL_B: "validation", TEST: "test"}
        rows_att = []
        for p in ("random", "conn", "chrono", "forward"):
            sp = make_split(meta, p, PrepOptions().split_seed, dict(fwd) if p == "forward" and fwd else None)[mal]
            names_sp = np.array([part.get(int(x), "unused") for x in sp], dtype=object)
            tab = pd.crosstab([pd.Series(fam, name="family"), pd.Series(att, name="attack")],
                              pd.Series(names_sp, name="split"))
            for (fm, a), cnt in tab.iterrows():
                row = {"protocol": p, "family": fm, "attack": a}
                for split in ("train", "validation", "test"):
                    n = int(cnt.get(split, 0))
                    row[split] = n
                    N.int(f"proto.{p}.{split}.attack.{a}", n)
                rows_att.append(row)
            for split in ("train", "validation", "test"):     # malicious flows per subset (all attacks)
                N.int(f"proto.{p}.{split}.malicious", int(tab[split].sum()) if split in tab.columns else 0)
        save_csv(out, "protocol_attacks_by_split", rows_att)
    except Exception as exc:  # pragma: no cover
        log.warning("attacks per split not computed: %s", exc)


def _first_seed_rank_text(main: dict | None) -> str:
    """'the best of the 5 seeds' etc. for the first seed of the proposed detector."""
    f1s = [float(x["f1"]) for x in ((main or {}).get("runs") or [])]
    if not f1s:
        return "one of the seeds"
    rank = 1 + sum(v > f1s[0] for v in f1s)
    return {1: "the best", 2: "the second best"}.get(rank, f"rank {rank}") + f" of the {len(f1s)} seeds"


def _seed_matched_diff(main: dict | None, m: dict, metric: str = "f1") -> dict | None:
    """metric(proposed) - metric(baseline) for the models trained with the same seed: mean and t-interval over the
    seeds, and on how many seeds the baseline is higher ('n_baseline_higher' of 'n')."""
    from ..metrics import seed_summary
    a = {int(x["seed"]): float(x[metric]) for x in ((main or {}).get("runs") or []) if "seed" in x and isnum(x.get(metric))}
    b = {int(x["seed"]): float(x[metric]) for x in (m.get("runs") or []) if "seed" in x and isnum(x.get(metric))}
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    s = seed_summary([a[k] - b[k] for k in common])
    s["n_baseline_higher"] = sum(1 for k in common if b[k] > a[k])
    return s if isnum(s.get("ci")) else None


def _ranking_seed_clause(done: dict, hi: list, auc_hi: list, diffs: dict) -> tuple[str, list, list, list]:
    """Seed-matched reading of the threshold-free metrics of the baselines with a higher mean than the proposed detector
    (Table 'baselines_extra'): for which of them the interval of the PR-AUC difference excludes zero (and whether they
    rank better on every seed), for which it contains zero, and for which the AUC-ROC interval excludes zero."""
    def excl(key, mk):
        x = diffs.get((key, mk))
        return bool(x) and abs(x["mean"]) > x["ci"]
    ex = [k for k in hi if excl(k, "pr_auc")]
    zero = [k for k in hi if diffs.get((k, "pr_auc")) and not excl(k, "pr_auc")]
    ex_auc = [k for k in auc_hi if excl(k, "auc_roc")]
    names = lambda ks: join_words([tex(done[k]["name"]) for k in ks])
    every = bool(ex) and all(diffs[(k, "pr_auc")]["n_baseline_higher"] == diffs[(k, "pr_auc")]["n"] for k in ex)
    parts = []
    if ex:
        parts.append(f"the 95\\% interval of the difference in PR-AUC excludes zero for {names(ex)}"
                     + ((", which rank" if len(ex) > 1 else ", which ranks") + " the test flows better than the proposed "
                        "detector on every seed" if every else ""))
    if zero:
        parts.append(("and it contains zero for " if ex else "the 95\\% interval of the difference in PR-AUC contains "
                      "zero for ") + names(zero))
    s = ", ".join(parts)
    if auc_hi:
        words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
        scope = (f"of these {words.get(len(hi), len(hi))} baselines" if set(auc_hi) == set(hi) and len(hi) > 1
                 else "of the baselines with a higher mean AUC-ROC")
        if not ex_auc:
            s += f"; {scope}, the interval of the difference in AUC-ROC contains zero for every one"
        elif len(ex_auc) == len(auc_hi):
            s += f"; {scope}, the interval of the difference in AUC-ROC excludes zero for every one"
        else:
            s += f"; {scope}, the interval of the difference in AUC-ROC excludes zero only for {names(ex_auc)}"
    return s, ex, zero, ex_auc


def b_baselines(cfg, ws, R, N, out) -> None:
    r = R.get("baselines")
    if not r or not r.get("models"):
        placeholder(out, "baselines", "Comparison with end-to-end detection models.")
        placeholder(out, "baselines_extra", "Threshold-free metrics of the end-to-end baselines.")
        placeholder(out, "baseline_cfg", "Configurations of the end-to-end baselines.")
        return
    main = R.get("main")
    seeds = r.get("seeds", [])
    P = r.get("protocol", {})
    # table order of the configuration, whatever the training order; a configured model without a result is listed
    order = list(cfg.experiments.baselines)
    models = dict(sorted(r["models"].items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)))
    try:
        from ..baselines import BASELINES as _BL
    except Exception:                      # the names are cosmetic
        _BL = {}
    for k in order:
        if k not in models:
            models[k] = {"name": getattr(_BL.get(k), "name", k), "status": "not completed in the time available"}
    r = {**r, "models": models}
    t = Table("baselines",
              f"Proposed detector and end-to-end baselines under one protocol: same partition, same pseudo-labels and "
              f"trust weights for training (KitNET, an anomaly detector, is trained on the flows pseudo-labelled "
              f"benign), same search budget ({P.get('trials', '--')} trials), ground truth for evaluation only. Test split, "
              f"mean $\\pm$ standard deviation over {len(seeds)} seeds. $\\Delta$F1 = F1(proposed) $-$ F1(baseline). First "
              "seed: paired bootstrap over the test flows of the first seed, with its 95\\% interval; it describes the "
              "sampling of the test flows only, and the first seed is " + _first_seed_rank_text(main) + " of the "
              "proposed detector. Seed-matched: mean over the seeds of the difference between the two models trained "
              "with the same seed, with its 95\\% $t$-interval over the seeds.",
              "lrrrll", [["Model", "Precision", "Recall", "F1", "$\\Delta$F1, first seed [95\\% CI]",
                          "$\\Delta$F1, seed-matched [95\\% CI]"]], size=r"\footnotesize")
    t2 = Table("baselines_extra",
               f"Threshold-free metrics, false-positive rate and accuracy of the models of "
               f"Table~\\ref{{tab:baselines}} (seed means). Seed-matched: mean over the seeds of the difference "
               f"between the proposed detector and the baseline trained with the same seed (proposed minus baseline), "
               f"with its 95\\% $t$-interval over the seeds.",
               "lrrrrll", [["Model", "PR-AUC", "AUC-ROC", "FPR (\\%)", "Accuracy", "$\\Delta$PR-AUC, seed-matched",
                            "$\\Delta$AUC-ROC, seed-matched"]], size=r"\footnotesize")
    rank_diffs = {}                                # (model, metric) -> seed-matched difference (proposed - baseline)
    rows_csv = []
    if main:
        s = main["summary"]
        t.row("Proposed", *metric_cells(s), "--", "--")
        t.rule()
        t2.row("Proposed", f(s["pr_auc"]["mean"]), f(s["auc_roc"]["mean"]), pct(s["fpr"]["mean"], 2), f(s["accuracy"]["mean"]),
               "--", "--")
        t2.rule()
    best = None
    for key, m in r["models"].items():
        if "summary" not in m:
            t.row(tex(m.get("name", key)), r"\multicolumn{5}{l}{" + tex(m.get("status", "not run")) + "}")
            continue
        s = m["summary"]
        pb = m["runs"][0].get("paired_vs_proposed") or {}
        d = f"{signed(pb['diff_mean'], 4)} [{signed(pb['lo'], 4)}, {signed(pb['hi'], 4)}]" if pb else "--"
        sd_ = _seed_matched_diff(main, m)
        dm = (f"{signed(sd_['mean'], 4)} [{signed(sd_['mean'] - sd_['ci'], 4)}, {signed(sd_['mean'] + sd_['ci'], 4)}]"
              if sd_ else "--")
        if sd_:
            N.put(f"base.{key}.seed_diff", sd_["mean"], signed(sd_["mean"], 4))
            N.put(f"base.{key}.seed_diff_ci", sd_["ci"],
                  f"[{signed(sd_['mean'] - sd_['ci'], 4)}, {signed(sd_['mean'] + sd_['ci'], 4)}]")
            N.put(f"base.{key}.seed_diff_contains_zero", "yes" if abs(sd_["mean"]) <= sd_["ci"] else "no")
        t.row(tex(m["name"]), *metric_cells(s), d, dm)
        cells = []
        for mk in ("pr_auc", "auc_roc"):           # seed-matched differences of the threshold-free metrics
            x = _seed_matched_diff(main, m, mk)
            if x:
                rank_diffs[(key, mk)] = x
                N.put(f"base.{key}.{mk}_seed_diff", x["mean"], signed(x["mean"], 4))
                N.put(f"base.{key}.{mk}_seed_diff_ci", x["ci"],
                      f"[{signed(x['mean'] - x['ci'], 4)}, {signed(x['mean'] + x['ci'], 4)}]")
                N.put(f"base.{key}.{mk}_seed_higher", x["n_baseline_higher"], f"{x['n_baseline_higher']} of {x['n']}")
            cells.append(f"{signed(x['mean'], 4)} [{signed(x['mean'] - x['ci'], 4)}, {signed(x['mean'] + x['ci'], 4)}]"
                         if x else "--")
        t2.row(tex(m["name"]), f(s["pr_auc"]["mean"]), f(s["auc_roc"]["mean"]), pct(s["fpr"]["mean"], 2), f(s["accuracy"]["mean"]),
               *cells)
        rows_csv.append({"model": m["name"], **{f"{k}_{sx}": s[k][sx] for k in ("precision", "recall", "f1", "accuracy", "fpr", "auc_roc", "pr_auc")
                                                for sx in ("mean", "std", "ci")},
                         "paired_diff_f1": pb.get("diff_mean"), "paired_lo": pb.get("lo"), "paired_hi": pb.get("hi"),
                         "paired_p": pb.get("p_two_sided")})
        for k in ("precision", "recall", "f1", "pr_auc", "auc_roc"):
            N.summary(f"base.{key}.{k}", s[k])
        N.pct(f"base.{key}.fpr", s["fpr"]["mean"], 2)
        if pb:
            N.put(f"base.{key}.diff", float(pb["diff_mean"]), signed(pb['diff_mean'], 4))     # differences are small:
            N.put(f"base.{key}.diff_ci", float(pb["lo"]), f"[{signed(pb['lo'], 4)}, {signed(pb['hi'], 4)}]")  # 4 decimals
            N.put(f"base.{key}.significant", int(pb["lo"] > 0 or pb["hi"] < 0), "yes" if (pb["lo"] > 0 or pb["hi"] < 0) else "no")
        if best is None or s["f1"]["mean"] > best[1]:
            best = (m["name"], s["f1"]["mean"], key)
    t.write(out)
    t2.write(out)
    save_csv(out, "baselines", rows_csv)
    # threshold-free metrics: here the completed learners differ more than in F1 (ranking of the test flows)
    done = {k: m for k, m in r["models"].items() if "summary" in m}
    his = {}                                       # baselines with a higher mean, per threshold-free metric
    for k in ("pr_auc", "auc_roc"):
        vals = {key: m["summary"][k]["mean"] for key, m in done.items() if isnum((m["summary"].get(k) or {}).get("mean"))}
        if vals:
            N.num(f"base.{k}_min", min(vals.values()))
            N.num(f"base.{k}_max", max(vals.values()))
            if main and isnum((main["summary"].get(k) or {}).get("mean")):
                ref_v = main["summary"][k]["mean"]
                hi = [key for key, x in vals.items() if round(x, 3) > round(ref_v, 3)]
                his[k] = hi
                N.put(f"base.n_{k}_higher", len(hi))
                N.put(f"base.{k}_higher_names", join_words([tex(done[key]["name"]) for key in hi]) or "none")
                if hi:
                    N.num(f"base.{k}_higher_min", min(vals[key] for key in hi))
                    N.num(f"base.{k}_higher_max", max(vals[key] for key in hi))
                lo_ = [key for key, x in vals.items() if round(x, 3) < round(ref_v, 3)]
                N.put(f"base.n_{k}_lower", len(lo_))
                N.put(f"base.{k}_lower_names", join_words([tex(done[key]["name"]) for key in lo_]) or "none")
    # seed-matched reading of the ranking metrics (Section 6.7 and the letter, R2.4)
    if main and his.get("pr_auc") is not None:
        clause, ex, zero, ex_auc = _ranking_seed_clause(done, his.get("pr_auc", []), his.get("auc_roc", []), rank_diffs)
        N.put("base.rank_seed_clause", clause)
        N.put("base.pr_auc_seed_excl_names", join_words([tex(done[k]["name"]) for k in ex]) or "none")
        N.put("base.pr_auc_seed_excl_any", "yes" if ex else "no")     # flag: 'not the best' rests on these models
        N.put("base.pr_auc_seed_zero_names", join_words([tex(done[k]["name"]) for k in zero]) or "none")
        N.put("base.auc_roc_seed_excl_names", join_words([tex(done[k]["name"]) for k in ex_auc]) or "none")
        rngs = {key: (m["summary"].get("pr_auc") or {}) for key, m in done.items()}
        rngs = {key: v["max"] - v["min"] for key, v in rngs.items() if isnum(v.get("max")) and isnum(v.get("min"))}
        ms = main["summary"].get("pr_auc") or {}
        if rngs and isnum(ms.get("max")) and isnum(ms.get("min")):   # flag: its PR-AUC range exceeds every baseline's
            N.put("main.pr_auc.range_above_all_baselines", "yes" if ms["max"] - ms["min"] > max(rngs.values()) else "no")
    # flags for sentences of the text that hold only for some outcomes (checked when a baseline is added)
    # the raw value of base.<key>.diff_ci is the lower end of the paired bootstrap interval (set above)
    above = [bool(isnum(N.raw.get(f"base.{key}.diff_ci")) and N.raw.get(f"base.{key}.diff_ci") > 0) for key in done]
    if done:
        N.put("base.all_paired_above_zero", "yes" if above and all(above) else "no")
    if main and isnum(((main["summary"].get("pr_auc") or {}).get("std"))):
        sds = {key: (m["summary"].get("pr_auc") or {}).get("std") for key, m in done.items()}
        sds = {key: v for key, v in sds.items() if isnum(v)}
        if sds:
            kmax = max(sds, key=sds.get)
            N.num("base.pr_auc_std_max", sds[kmax], 4)
            N.put("base.pr_auc_std_max_name", tex(done[kmax]["name"]))
            N.put("main.pr_auc.std_above_all_baselines", "yes" if main["summary"]["pr_auc"]["std"] > sds[kmax] else "no")
    for fm in ("Web", "Infiltration"):            # attacks that only KitNET detects in part (so far)
        oth = [(done[key]["summary"].get("families") or {}).get(fm, {}).get("mean") for key in done if key != "kitnet"]
        oth = [v for v in oth if isnum(v)]
        if oth:
            N.num(f"base.{fm}.recall_max_other_than_kitnet", max(oth))
            N.put(f"base.{fm}.detected_only_by_kitnet", "yes" if max(oth) < 0.0005 else "no")
    # which kinds of model were completed (the comparison says nothing about kinds that were not)
    # kinds of model: tree ensembles and neural networks learn from the pseudo-labels with a supervised loss; KitNET,
    # an anomaly detector, is trained on the flows pseudo-labelled benign only
    TREE = {"rf", "lightgbm", "xgb_logloss"}
    ANOMALY = {"kitnet"}
    tree = [key for key in done if key in TREE]
    nn = [key for key in done if key not in TREE and key not in ANOMALY]
    anomaly = [key for key in done if key in ANOMALY]
    sup = tree + nn
    other = nn + anomaly
    N.put("base.n_tree", len(tree))
    N.put("base.n_nn", len(nn))
    N.put("base.n_sup", len(sup))
    N.put("base.n_other", len(other))
    N.put("base.nn_names", join_words([tex(done[k]["name"]) for k in nn]) or "none")
    N.put("base.sup_names", join_words([tex(done[k]["name"]) for k in sup]) or "none")
    fam_ref = (main or {}).get("summary", {}).get("families") or {}
    for grp, keys in (("tree", tree), ("nn", nn), ("sup", sup), ("other", other)):   # spread per kind of model
        f1g = [done[k]["summary"]["f1"]["mean"] for k in keys]
        if not f1g:
            continue
        N.num(f"base.{grp}.f1_min", min(f1g), 4)
        N.num(f"base.{grp}.f1_max", max(f1g), 4)
        up4 = lambda x: f(math.ceil(x * 1e4 - 1e-9) / 1e4, 4)     # upper bounds are rounded up
        if main:
            mx = max(abs(x - main["summary"]["f1"]["mean"]) for x in f1g)
            N.put(f"base.{grp}.max_abs_diff_to_proposed", mx, up4(mx))
        dg = [abs(v["mean"] - fam_ref[fm]["mean"]) for k in keys
              for fm, v in ((done[k]["summary"].get("families") or {}).items())
              if fm in fam_ref and isnum(v.get("mean")) and isnum(fam_ref[fm].get("mean"))]
        if dg:
            N.put(f"base.{grp}.max_family_recall_diff", max(dg), up4(max(dg)))
        smk = [(k, _seed_matched_diff(main, done[k])) for k in keys]
        smk = [(k, x) for k, x in smk if x]
        sm = [x for _, x in smk]
        if sm:                                     # seed-matched differences whose t-interval contains zero
            nz = sum(1 for x in sm if abs(x["mean"]) <= x["ci"])
            N.put(f"base.{grp}.n_seed_diff_zero", nz)
            N.put(f"base.{grp}.seed_diff_all_zero", "yes" if nz == len(sm) else "no")
            # the models whose seed-matched interval excludes zero, with the difference and its interval
            ex = [f"{tex(done[k]['name'])} ({signed(x['mean'], 4)} [{signed(x['mean'] - x['ci'], 4)}, "
                  f"{signed(x['mean'] + x['ci'], 4)}])" for k, x in smk if abs(x["mean"]) > x["ci"]]
            N.put(f"base.{grp}.seed_diff_nonzero_list", join_words(ex) or "none")
            N.put(f"base.{grp}.n_seed_diff_nonzero", len(ex))
            if grp == "sup":                       # the sentence of Section 6.7 on the seed-matched differences
                tab = "(last column of Table~\\ref{tab:baselines})"
                pos = [(k, x) for k, x in smk if x["mean"] - x["ci"] > 0]     # proposed detector better
                neg = [(k, x) for k, x in smk if x["mean"] + x["ci"] < 0]     # baseline better
                if not pos and not neg:
                    sent = ("Matched by seed, the difference to each of these baselines has a 95\\% interval that "
                            f"contains zero {tab}. We therefore do not read the paired differences as an advantage of "
                            "the proposed detector.")
                else:
                    lst = lambda kx: join_words([f"{tex(done[k]['name'])} ({signed(x['mean'], 4)} "
                                                 f"[{signed(x['mean'] - x['ci'], 4)}, {signed(x['mean'] + x['ci'], 4)}])"
                                                 for k, x in kx])
                    sent = f"Matched by seed {tab}, the difference has a 95\\% interval that contains zero for every " \
                           f"one of these baselines except {lst(pos + neg)}."
                    if pos:
                        one = len(pos) == 1
                        seeds_all = all(x["n_baseline_higher"] == 0 for _, x in pos)
                        mx = max(x["mean"] for _, x in pos)
                        sent += (f" The proposed detector is better than {'this baseline' if one else 'these baselines'}"
                                 + (" on every seed" if seeds_all else "") + f", by {signed(mx, 4)[1:]} in F1-score "
                                 + ("" if one else "at most ") + "on average; among these baselines, we read the paired "
                                 "differences as an advantage of the proposed detector over "
                                 + join_words([tex(done[k]["name"]) for k, _ in pos]) + " only, and a small one.")
                    if neg:
                        sent += (f" {'This baseline is' if len(neg) == 1 else 'These baselines are'} better than the "
                                 "proposed detector.")
                N.put("base.sup.seed_sentence", sent)
                N.put("base.sup.seed_letter_clause",          # the same outcome in one clause of the letter (R2.4)
                      "matched by seed, the F1-score of none of them differs from that of the proposed detector with a "
                      "95\\% interval that excludes zero" if not pos and not neg else
                      "matched by seed, the F1-score differs from that of the proposed detector with a 95\\% interval "
                      "that excludes zero only for " + lst(pos + neg))
        if main:                                   # ranking metrics of this kind of model against the detector
            for mk in ("pr_auc", "auc_roc"):
                ref_v = (main["summary"].get(mk) or {}).get("mean")
                vals = {k: (done[k]["summary"].get(mk) or {}).get("mean") for k in keys}
                vals = {k: v for k, v in vals.items() if isnum(v)}
                if isnum(ref_v) and vals:
                    N.put(f"base.{grp}.{mk}_higher_names", join_words([tex(done[k]["name"]) for k, v in vals.items()
                                                                      if round(v, 3) > round(ref_v, 3)]) or "none")
                    N.put(f"base.{grp}.{mk}_lower_names", join_words([tex(done[k]["name"]) for k, v in vals.items()
                                                                     if round(v, 3) < round(ref_v, 3)]) or "none")
        if main and len(f1g) > 1:                  # spread of the mean F1-scores against the seed range of the detector
            sp_ = max(f1g) - min(f1g)
            rs = [float(x["f1"]) for x in (main.get("runs") or []) if isnum(x.get("f1"))]
            if rs:
                N.put(f"base.{grp}.spread", sp_, f(math.ceil(sp_ * 1e4 - 1e-9) / 1e4, 4))
                N.put(f"base.{grp}.spread_below_seed_range", "yes" if sp_ < max(rs) - min(rs) else "no")
                if grp == "sup":                   # the clause of Section 6.7 on the spread of the learners
                    rng = f"({f(min(rs), 4)} to {f(max(rs), 4)})"
                    spv = f(math.ceil(sp_ * 1e4 - 1e-9) / 1e4, 4)
                    if sp_ < max(rs) - min(rs):
                        cl = (f"the spread of their mean F1-scores ({spv}) is smaller than the range of the proposed "
                              f"detector over its own seeds {rng}, so on this corpus the choice among these learners "
                              "changes the mean F1-score by less than the choice of the seed changes that of the proposed "
                              "detector")
                    else:
                        cl = (f"the spread of their mean F1-scores ({spv}) is "
                              + ("about as large as" if sp_ <= 1.5 * (max(rs) - min(rs)) else "larger than")
                              + f" the range of the proposed detector over its own seeds {rng}, so on this corpus the "
                              f"choice among these learners changes the mean F1-score by at most {spv}")
                    N.put("base.sup.spread_clause", cl)
    for key, m in done.items():                    # recall per family of every completed baseline
        for fm, v in (m["summary"].get("families") or {}).items():
            if isnum(v.get("mean")):
                N.num(f"base.{key}.family.{fm}", v["mean"])
    if done:
        parts = []
        if tree:
            parts.append(f"{len(tree)} tree ensemble{'s' if len(tree) != 1 else ''}")
        if nn:
            parts.append(f"{len(nn)} neural network{'s' if len(nn) != 1 else ''} trained with a supervised loss ("
                         + join_words([tex(done[k]["name"]) for k in nn]) + ")")
        if anomaly:
            parts.append("the autoencoder-based anomaly detector " + join_words([tex(done[k]["name"]) for k in anomaly]))
        kinds = ("the completed baselines are all tree ensembles, as is the proposed detector"
                 if not other else "the completed baselines comprise " + join_words(parts))
        N.put("base.kinds_clause", kinds)
    else:
        kinds = ""
    # the paired differences of every completed baseline, in table order (first seed; Table 'baselines')
    dl = []
    for key, m in done.items():
        pb = (m.get("runs") or [{}])[0].get("paired_vs_proposed") or {}
        if pb:
            dl.append(f"{tex(m['name'])} {signed(pb['diff_mean'], 4)} [{signed(pb['lo'], 4)}, {signed(pb['hi'], 4)}]")
    if dl:
        N.put("base.diff_list", len(dl), "; ".join(dl))
        n_pos = sum(1 for m in done.values()
                    if ((m.get("runs") or [{}])[0].get("paired_vs_proposed") or {}).get("lo", -1) > 0)
        N.put("base.n_diff_above_zero", n_pos)
        N.put("base.diff_all_above_zero", "yes" if n_pos == len(dl) else "no")
    # how many baselines have a higher mean F1-score than the proposed detector, named
    if main:
        ref_f1 = main["summary"]["f1"]["mean"]
        hi_ = [(k, m) for k, m in done.items() if m["summary"]["f1"]["mean"] > ref_f1]
        if not hi_:
            txt = f"none of the {len(done)} baselines has a higher mean F1-score than the proposed detector"
        else:
            names = join_words([tex(m["name"]) for _, m in hi_])
            txt = (f"{len(hi_)} of the {len(done)} baselines ({names}) "
                   + ("has" if len(hi_) == 1 else "have") + " a higher mean F1-score than the proposed detector")
        N.put("base.n_better_clause", len(hi_), txt)
    N.put("base.completed", join_words([tex(m["name"]) for m in done.values()]) or "none")
    per_seed = []
    for key, m in r["models"].items():
        for x in m.get("runs", []):
            per_seed.append({"model": m.get("name", key), **{k: v for k, v in x.items() if not isinstance(v, (dict, list))}})
    save_csv(out, "baselines_per_seed", per_seed)
    if best:
        # baselines whose mean F1 equals the highest one at four decimals are named together (a margin of 0.00001
        # decides nothing)
        tied = [m["name"] for m in r["models"].values()
                if "summary" in m and round(m["summary"]["f1"]["mean"], 4) == round(best[1], 4)]
        N.put("base.best", best[0], join_words([tex(x) for x in tied]))
        N.put("base.best_tied", len(tied))
        N.num("base.best.f1", best[1])
        N.num("base.best.f1_4d", best[1], 4)
        if main:
            d = main["summary"]["f1"]["mean"] - best[1]
            N.put("base.best.gap", float(d), signed(d, 5))
            n_better = sum(1 for m in r["models"].values() if "summary" in m and m["summary"]["f1"]["mean"] > main["summary"]["f1"]["mean"])
            N.put("base.n_better", int(n_better))
    N.put("base.n", sum(1 for m in r["models"].values() if "summary" in m))
    N.put("base.n_configured", len(order))
    missing = [m.get("name", k) for k, m in r["models"].items() if "summary" not in m]
    N.put("base.n_missing", len(missing))
    N.put("base.missing", join_words([tex(x) for x in missing]) or "none")
    N.put("base.missing_clause", ("every configured baseline was completed" if not missing else
                                  join_words([tex(x) for x in missing]) + (" was" if len(missing) == 1 else " were")
                                  + " not completed in the time available"))
    # the first sentence of limitation A.lim7 and the open item of the letter, for both cases
    n_done = sum(1 for m in r["models"].values() if "summary" in m)
    done_names = join_words([tex(m["name"]) for m in r["models"].values() if "summary" in m]) or "none"
    if missing:
        verb = "was" if len(missing) == 1 else "were"
        them = "it" if len(missing) == 1 else "them"
        N.put("base.completion_sentence",
              f"Of the {len(order)} baselines of the comparison protocol, {n_done} were completed in the time available "
              f"for this revision ({done_names}), and {join_words([tex(x) for x in missing])} {verb} not "
              "(Section~\\ref{sec:e2e}).")
        N.put("base.setup_sentence",            # last sentence of the protocol paragraph (Section 5)
              f"Of the {len(order)} models, {n_done} were completed in the time available for this revision "
              f"({done_names}); Table~\\ref{{tab:baselines}} lists {join_words([tex(x) for x in missing])} as not "
              "completed.")
        N.put("base.open_item",
              f"Baselines: {n_done} of the {len(order)} baselines of the comparison protocol were completed in the time "
              f"available ({done_names}); {join_words([tex(x) for x in missing])} {verb} not, and the manuscript and "
              f"this letter say so. Decide whether to complete {them} before submission (the run on your computer "
              f"continues with {them}, experiment baselines of configs/paper_reduced_ext.yaml); a completed baseline "
              "enters the tables when the report is regenerated, and the sentences on the baselines must then be read "
              "again.")
        # the opening of the comparison (Section 6) and two sentences of the letter
        N.put("base.compare_clause", f"{n_done} of the {len(order)} baselines")
        N.put("base.status_kinds_clause", N.text["base.missing_clause"] + (f"; {kinds}" if kinds else ""))
        N.put("base.letter_missing_sentence",
              f"Of the {len(order)} baselines of the protocol, {N.text['base.missing_clause']}; the manuscript names "
              "any baseline that is missing.")
        N.put("base.letter_constraint_clause",
              f"of the {len(order)} baselines of the comparison protocol, {N.text['base.missing_clause']}, and the "
              "protocol gives the neural baselines and KitNET a smaller training budget than the tree models")
    else:
        N.put("base.completion_sentence", "")     # nothing to limit: Sections 5.5 and 6.7 say that all were completed
        N.put("base.setup_sentence", f"All {len(order)} models were completed.")
        N.put("base.open_item", "")            # nothing to decide: the letter drops an empty open item
        N.put("base.compare_clause", f"all {len(order)} baselines")
        N.put("base.status_kinds_clause", kinds.replace("the completed baselines", "the baselines", 1))
        N.put("base.letter_missing_sentence", f"All {len(order)} baselines of the protocol were completed.")
        N.put("base.letter_constraint_clause",
              "the comparison protocol gives the neural baselines and KitNET a smaller training budget than the tree "
              "models")
    f1s = [m["summary"]["f1"]["mean"] for m in r["models"].values() if "summary" in m]
    if f1s:                                    # spread of the baselines' F1-scores
        N.num("base.f1_min", min(f1s), 4)
        N.num("base.f1_max", max(f1s), 4)
        if main:
            ref_f1 = main["summary"]["f1"]["mean"]
            N.num("base.max_abs_diff_to_proposed", max(abs(x - ref_f1) for x in f1s), 4)
        # do the baselines detect the same families as the proposed detector? (largest per-family recall difference)
        fam_ref = (main or {}).get("summary", {}).get("families") or {}
        diffs = []
        for m in r["models"].values():
            fs = (m.get("summary") or {}).get("families") or {}
            for fm, v in fs.items():
                if fm in fam_ref and isnum(v.get("mean")) and isnum(fam_ref[fm].get("mean")):
                    diffs.append(abs(v["mean"] - fam_ref[fm]["mean"]))
        if diffs:
            N.num("base.max_family_recall_diff", max(diffs))
    for k in ("trials", "random_trials", "search_rows", "search_val_rows", "dl_max_train_rows", "dl_epochs", "seq_window"):
        if isnum(P.get(k)):
            N.int(f"base.protocol.{k}", P[k])
    t = Table("baseline_cfg",
              "Configurations of the end-to-end baselines selected by the shared search, size of the model of the first "
              "seed, and mean training time of one final fit over the seeds.",
              "p{0.17\\linewidth}p{0.48\\linewidth}rr", [["Model", "Selected configuration", "Size", "Fit (s)"]],
              size=r"\scriptsize")
    for key, m in r["models"].items():
        d = m.get("describe")
        if not d:
            continue
        cfg_s = ", ".join(f"{tex(k)} = {param(v)}" for k, v in d.get("params", {}).items())
        size = d.get("n_parameters")
        size = integer(size) + " param." if isnum(size) else tex(size)
        fit = np.mean([x.get("fit_seconds", np.nan) for x in m.get("runs", [])]) if m.get("runs") else float("nan")
        t.row(tex(m["name"]), cfg_s, size, f(fit, 1))
    t.write(out)


def _adversarial_family(r: dict, h: dict, N: Numbers, out: Path) -> None:
    """Recall per attack family under every transformation, standard and augmented training.

    The aggregate recall of Table 'adversarial' mixes families that one transformation affects in opposite directions
    (padding raises the botnet recall and removes the brute-force recall), so the text reads the families from here.
    Keys: adv.[aug.]f.<kind>.<level>.<family>, adv.[aug.]f.clean.<family>, adv.[aug.]f.worst.<family>, and per
    transformation the largest change of a family with at least LARGE_FAMILY flows (adv.[aug.]f.<kind>.max_change)
    and the lowest recall over the levels for each family (adv.[aug.]f.<kind>.min.<family>, with its level)."""
    clean = r["clean"].get("families") or {}
    fams = fam_sorted(clean.keys())
    if not fams:
        return
    hdr = [f"{FAMILY_LABEL.get(fm, fm)} ({integer(clean[fm]['support'])})" for fm in fams]
    t = Table("adversarial_family",
              "Recall per attack family (number of evaluated flows in parentheses) under the transformations of "
              "Section~\\ref{sec:expsetup}, (a) with standard training and (b) with augmented training; the last row "
              "of each part is the per-flow worst case over the grid. For a family with fewer than "
              f"{integer(LARGE_FAMILY)} flows the recall rests on a small sample (Section~\\ref{{sec:metrics}}).",
              "ll" + "r" * len(fams), [["Transformation", "Level", *hdr]], size=r"\footnotesize")
    short = {"padding": "Padding (bytes/packet)", "dummy": "Dummy packets per flow", "stretch": "Duration stretched ($\\times$)",
             "throttle": "Connection spacing ($\\times$)"}
    parts = [("(a) Standard training", r, "")]
    if h and "clean" in h:
        parts.append(("(b) Augmented training", h, "aug."))
    for i, (title, src, tag) in enumerate(parts):
        if i:
            t.rule()
        t.raw(rf"\multicolumn{{{2 + len(fams)}}}{{l}}{{\emph{{{title}}}}} \\")
        cf = src["clean"].get("families") or {}
        t.row("None (clean)", "--", *[f((cf.get(fm) or {}).get("recall")) for fm in fams])
        for fm in fams:
            if isnum((cf.get(fm) or {}).get("recall")):
                N.num(f"adv.{tag}f.clean.{fm}", cf[fm]["recall"])
        last = None
        low: dict = {}
        change: dict = {}
        for a in src.get("attacks", []):
            af = a.get("families") or {}
            t.row(short[a["kind"]] if a["kind"] != last else "", f"{a['level']:g}",
                  *[f((af.get(fm) or {}).get("recall")) for fm in fams])
            last = a["kind"]
            for fm in fams:
                rc = (af.get(fm) or {}).get("recall")
                if not isnum(rc):
                    continue
                N.num(f"adv.{tag}f.{a['kind']}.{a['level']:g}.{fm}", rc)
                if (a["kind"], fm) not in low or rc < low[(a["kind"], fm)][1]:
                    low[(a["kind"], fm)] = (a["level"], rc)
                if clean.get(fm, {}).get("support", 0) >= LARGE_FAMILY and isnum((cf.get(fm) or {}).get("recall")):
                    d = abs(rc - cf[fm]["recall"])
                    change[a["kind"]] = max(change.get(a["kind"], 0.0), d)
        for (kind, fm), (lv, rc) in low.items():       # a lower bound, so rounded down ("at least 0.996")
            N.put(f"adv.{tag}f.{kind}.min.{fm}", rc, f(math.floor(rc * 1e3 + 1e-9) / 1e3))
            N.put(f"adv.{tag}f.{kind}.min_level.{fm}", lv, f"{lv:g}")
        for kind, d in change.items():
            N.num(f"adv.{tag}f.{kind}.max_change", d)
        wf = (src.get("best_of_grid") or {}).get("families") or {}
        t.row("Per-flow worst case", "--", *[f((wf.get(fm) or {}).get("recall")) for fm in fams])
        for fm in fams:
            if isnum((wf.get(fm) or {}).get("recall")):
                N.num(f"adv.{tag}f.worst.{fm}", wf[fm]["recall"])
    t.write(out)


def b_adversarial(cfg, ws, R, N, out) -> None:
    r = R.get("adversarial")
    if not r or "attacks" not in r:
        placeholder(out, "adversarial", "Recall under traffic-level evasion transformations.")
        placeholder(out, "adversarial_family")
        return
    h = r.get("hardened") or {}
    hat = {(a["kind"], a["level"]): a for a in h.get("attacks", [])}
    label = {"padding": "Payload padding (bytes/packet)", "dummy": "Dummy packets per flow",
             "stretch": "Flow duration stretched ($\\times$)", "throttle": "Connection spacing ($\\times$)"}
    t = Table("adversarial",
              f"Recall on {integer(r['evaluated_flows'])} malicious test flows sent by attacker-controlled endpoints when "
              "the attacker transforms all of its traffic, without and with augmentation of the training set. Host-level "
              "features of every flow are recomputed from the transformed stream; decision thresholds are unchanged.",
              "llrr", [["Transformation", "Level", "Recall", "Recall, augmented training"]], size=r"\footnotesize")
    t.row("None (clean)", "--", f(r["clean"]["recall"]), f((h.get("clean") or {}).get("recall")))
    t.rule()
    rows_csv = [{"kind": "clean", "level": 0, "recall": r["clean"]["recall"], "recall_augmented": (h.get("clean") or {}).get("recall")}]
    last = None
    worst = {}
    for a in r["attacks"]:
        ha = hat.get((a["kind"], a["level"]), {})
        t.row(label[a["kind"]] if a["kind"] != last else "", f"{a['level']:g}", f(a["recall"]), f(ha.get("recall")))
        last = a["kind"]
        rows_csv.append({"kind": a["kind"], "level": a["level"], "recall": a["recall"], "recall_augmented": ha.get("recall")})
        N.num(f"adv.{a['kind']}.{a['level']:g}", a["recall"])
        if ha:
            N.num(f"adv.aug.{a['kind']}.{a['level']:g}", ha["recall"])
        if a["kind"] not in worst or a["recall"] < worst[a["kind"]][1]:
            worst[a["kind"]] = (a["level"], a["recall"])
    t.rule()
    t.row("Per-flow worst case over the grid", "--", f(r["best_of_grid"]["recall"]), f((h.get("best_of_grid") or {}).get("recall")))
    t.write(out)
    save_csv(out, "adversarial", rows_csv)
    fam_rows = []
    for a in [{"kind": "clean", "level": 0, **r["clean"]}] + r["attacks"]:
        for fm, v in (a.get("families") or {}).items():
            fam_rows.append({"kind": a["kind"], "level": a["level"], "family": fm, **v})
    save_csv(out, "adversarial_by_family", fam_rows)
    _adversarial_family(r, h, N, out)
    N.int("adv.n", r["evaluated_flows"])
    N.num("adv.clean", r["clean"]["recall"])
    N.num("adv.worst_grid", r["best_of_grid"]["recall"])
    for fm, v in (r["clean"].get("families") or {}).items():        # where the recall without transformation comes from
        N.num(f"adv.fam.{fm}.recall", v["recall"])
        N.int(f"adv.fam.{fm}.n", v["support"])
    for k, (lv, rc) in worst.items():
        N.num(f"adv.{k}.worst", rc)
        N.put(f"adv.{k}.worst_level", lv, f"{lv:g}")
    # per transformation: smallest level of the grid and the recall there, highest recall over the levels,
    # and the same for the detector with augmented training
    for k in worst:
        ak = [a for a in r["attacks"] if a["kind"] == k]
        a0 = min(ak, key=lambda a: a["level"])
        N.put(f"adv.{k}.min_level", a0["level"], f"{a0['level']:g}")
        N.num(f"adv.{k}.at_min_level", a0["recall"])
        N.num(f"adv.{k}.best", max(a["recall"] for a in ak))
        hk = [hat[(a["kind"], a["level"])]["recall"] for a in ak if (a["kind"], a["level"]) in hat]
        if hk:
            N.num(f"adv.aug.{k}.worst", min(hk))
            N.num(f"adv.aug.{k}.best", max(hk))
    if worst:
        kmin = min(worst, key=lambda k: worst[k][1])
        N.put("adv.most_effective", kmin, {"padding": "payload padding", "dummy": "dummy packets",
                                           "stretch": "duration stretching", "throttle": "connection spacing"}[kmin])
        N.num("adv.most_effective.recall", worst[kmin][1])
    if h:
        N.num("adv.aug.clean", h["clean"]["recall"])
        N.num("adv.aug.worst_grid", h["best_of_grid"]["recall"])
        N.int("adv.aug.rows", h.get("augmented_rows"))
        for k in ("precision", "recall", "f1"):
            N.num(f"adv.aug.test.{k}", h["test_metrics_clean"][k])
            N.num(f"adv.base.test.{k}", r["baseline_test_metrics_clean"][k])
        N.pct("adv.aug.test.fpr", h["test_metrics_clean"]["fpr"], 2)
        N.pct("adv.base.test.fpr", r["baseline_test_metrics_clean"]["fpr"], 2)
    N.num("adv.check", r.get("check_clean_matches_cached"), 6)
    try:
        plt = _mpl()
        kinds = [k for k in ("padding", "dummy", "stretch", "throttle") if any(a["kind"] == k for a in r["attacks"])]
        fig, axes = plt.subplots(1, len(kinds), figsize=(1.75 * len(kinds), 2.1), sharey=True)
        axes = np.atleast_1d(axes)
        short = {"padding": "Padding (bytes/packet)", "dummy": "Dummy packets", "stretch": "Duration ($\\times$)",
                 "throttle": "Spacing ($\\times$)"}
        for ax, k in zip(axes, kinds):
            xs = [a["level"] for a in r["attacks"] if a["kind"] == k]
            ys = [a["recall"] for a in r["attacks"] if a["kind"] == k]
            ax.plot(range(len(xs)), ys, "-o", color=C1, label="Standard training", markeredgecolor="white", markeredgewidth=0.8)
            if hat:
                yh = [hat.get((k, x), {}).get("recall", np.nan) for x in xs]
                ax.plot(range(len(xs)), yh, "--s", color=C2, label="Augmented training", markeredgecolor="white", markeredgewidth=0.8)
            ax.axhline(r["clean"]["recall"], color=MUTED, linewidth=0.8, linestyle=":")
            ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
            ax.set_xlabel(short[k])
            ax.set_ylim(-0.03, 1.03)
        axes[0].set_ylabel("Recall on attacker flows")
        axes[0].text(0.02, r["clean"]["recall"] + 0.02, "clean", color=MUTED, fontsize=7, transform=axes[0].get_yaxis_transform())
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.08))
        _save(fig, out, "evasion")
    except Exception as exc:  # pragma: no cover
        log.warning("figure evasion failed: %s", exc)


def b_interpret(cfg, ws, R, N, out) -> None:
    r = R.get("interpret")
    if not r or "global" not in r:
        for n in ("shap", "shap_family", "latent"):
            placeholder(out, n, "Feature attribution of the detector.")
        return
    g = r["global"]
    top = g[:12]
    t = Table("shap",
              f"Global feature attribution of the detector: mean absolute TreeSHAP contribution to the raw score on "
              f"{integer(r['rows'])} test flows ({integer((r.get('per_family') or {}).get('Benign', {}).get('rows'))} benign; "
              f"{integer(r['rows'] - ((r.get('per_family') or {}).get('Benign', {}).get('rows') or 0))} malicious, the same number "
              "per attack family or all test flows of a smaller family), and share of the total.",
              "rllrr", [["Rank", "Feature", "Subspace", "Mean $|\\mathrm{SHAP}|$", "Share (\\%)"]], size=r"\footnotesize")
    for i, x in enumerate(top, 1):
        t.row(i, tt(x["feature"]), SUBSPACE_LABEL.get(x["subspace"], x["subspace"]), f(x["mean_abs_shap"]), pct(x["share"]))
    t.write(out)
    save_csv(out, "shap_global", g)
    for i, x in enumerate(g[:5], 1):
        N.put(f"shap.top{i}", x["feature"], tt(x["feature"]))
        N.pct(f"shap.top{i}.share", x["share"])
    N.pct("shap.top5_share", sum(x["share"] for x in g[:5]))
    N.pct("shap.top10_share", sum(x["share"] for x in g[:10]))
    for k in ("statistical", "temporal", "latent"):
        N.pct(f"shap.subspace.{k}", r["subspace_share_shap"].get(k, 0.0))
        N.pct(f"gain.subspace.{k}", r["subspace_share_gain"].get(k, 0.0))
    for k, v in r["subspace_count"].items():
        N.put(f"shap.count.{k}", int(v))
    N.put("shap.additivity", float(r["additivity_max_abs_error"]), sci(r["additivity_max_abs_error"]))
    N.int("shap.rows", r["rows"])
    pf_ = r.get("per_family") or {}
    if "Benign" in pf_:
        N.int("shap.n_benign", pf_["Benign"]["rows"])
        N.int("shap.n_malicious", r["rows"] - pf_["Benign"]["rows"])
        mal_counts = [v["rows"] for k, v in pf_.items() if k != "Benign"]
        if mal_counts:
            N.int("shap.n_per_family", max(mal_counts))
    t = Table("shap_subspace",
              "Share of the attribution carried by each feature subspace: TreeSHAP (mean absolute contribution) and "
              "total split gain.",
              "lrrr", [["", "Selected", "TreeSHAP", "Split gain"],
                       ["Subspace", "features", "share (\\%)", "share (\\%)"]], size=r"\footnotesize")
    for k in ("statistical", "temporal", "latent"):
        if k in r["subspace_share_shap"]:
            t.row(SUBSPACE_LABEL[k], r["subspace_count"].get(k, 0), pct(r["subspace_share_shap"][k]), pct(r["subspace_share_gain"].get(k)))
    t.write(out)
    # per family
    t = Table("shap_family",
              "Operator-facing explanation per class: the three features with the largest mean signed TreeSHAP "
              "contribution in magnitude (positive values push the score towards ``malicious'') and the share of the "
              "sampled flows of the class that the detector flags.",
              "lrp{0.62\\linewidth}", [["Class", "Flagged (\\%)", "Leading contributions (mean SHAP)"]], size=r"\footnotesize")
    fam_rows = []
    for fm in ["Benign"] + fam_sorted(r["per_family"].keys()):
        x = r["per_family"].get(fm)
        if not x:
            continue
        lead = "; ".join(f"{tt(c['feature'])} ({signed(c['mean_shap'], 2)})" for c in x["top"][:3])
        t.row(FAMILY_LABEL.get(fm, fm), pct(x["detected_share"]), lead)
        for c in x["top"]:
            fam_rows.append({"class": fm, "rows": x["rows"], "detected_share": x["detected_share"], **c})
    t.write(out)
    save_csv(out, "shap_by_class", fam_rows)
    probes = r.get("latent_probes") or []
    if probes:
        t = Table("latent",
                  "Reading of the latent features used by the detector: attribution share, the three named inputs "
                  "each latent dimension correlates with most strongly (Pearson $r$ on training flows) and the share "
                  "of its variance explained by a linear function of the named features ($R^2$).",
                  "lrp{0.56\\linewidth}r", [["Latent", "Share (\\%)", "Most correlated named features ($r$)", "$R^2$"]],
                  size=r"\footnotesize")
        for p in probes:
            t.row(tt(p["latent"]), pct(p["share"]), "; ".join(f"{tt(c['feature'])} ({signed(c['r'], 2)})" for c in p["top_correlates"]),
                  f(p["linear_r2"], 2))
        t.write(out)
        save_csv(out, "latent_probes", [{"latent": p["latent"], "share": p["share"], "linear_r2": p["linear_r2"],
                                         **{f"corr{i + 1}_{k}": c[k] for i, c in enumerate(p["top_correlates"]) for k in ("feature", "r")}}
                                        for p in probes])
        N.put("latent.n", len(probes))
        r2 = [p["linear_r2"] for p in probes]
        N.num("latent.r2_min", min(r2), 2)
        N.num("latent.r2_median", float(np.median(r2)), 2)
        N.put("latent.n_r2_ge_09", sum(x >= 0.9 for x in r2))
    else:
        placeholder(out, "latent", "No latent feature was selected, so no latent probe is reported.")
        N.put("latent.n", 0)
    try:
        plt = _mpl()
        topn = g[:15][::-1]
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        col = {"statistical": C1, "temporal": C2, "latent": C3}
        hatch = {"statistical": "", "temporal": "///", "latent": "..."}
        for i, x in enumerate(topn):
            ax.barh(i, x["mean_abs_shap"], height=0.62, color=col[x["subspace"]], hatch=hatch[x["subspace"]],
                    edgecolor="white", linewidth=0.6)
        ax.set_yticks(range(len(topn)), [x["feature"] for x in topn], fontsize=7, family="DejaVu Sans Mono")
        ax.set_xlabel("Mean |SHAP| (raw-score units)")
        ax.grid(axis="y", visible=False)
        from matplotlib.patches import Patch

        present = [k for k in ("statistical", "temporal", "latent") if any(x["subspace"] == k for x in topn)]
        ax.legend(handles=[Patch(facecolor=col[k], hatch=hatch[k], edgecolor="white", label=SUBSPACE_LABEL[k]) for k in present],
                  loc="lower right", fontsize=7)
        _save(fig, out, "shap_global")
    except Exception as exc:  # pragma: no cover
        log.warning("figure shap_global failed: %s", exc)


def b_online(cfg, ws, R, N, out) -> None:
    r = R.get("online")
    if not r or "service_time" not in r:
        for n in ("online", "load", "normalization"):
            placeholder(out, n, "Online operation of the detector.")
        return
    st = r["service_time"]
    pb = st["per_batch_ms"]
    t = Table("online",
              f"Service time of the online path for one micro-batch of {st['batch_size']} finalized flow records on a "
              f"single worker (one thread), measured over {integer(st['records'])} records processed back to back. "
              "Times in milliseconds, measured from the availability of a finalized flow record; the lifetime of "
              "the flow and Zeek's connection time-outs are not included.",
              "lrrr", [["Stage", "Median", "P95", "Mean"]], size=r"\footnotesize")
    for k, lab in (("features", "Host-state update and feature computation"), ("normalise", "Imputation, scaling, encoder"),
                   ("model", "Tree-ensemble inference"), ("total", "Total per micro-batch")):
        if k == "total":
            t.rule()
        t.row(lab, f(pb[k]["p50"], 2), f(pb[k]["p95"], 2), f(pb[k]["mean"], 2))
        N.num(f"online.batch.{k}.p50", pb[k]["p50"], 2)
        N.num(f"online.batch.{k}.p95", pb[k]["p95"], 2)
        N.num(f"online.batch.{k}.mean", pb[k]["mean"], 2)
    t.notes = (f"Mean service time per flow: {f(st['per_flow_ms_mean'], 3)} ms; single-worker capacity: "
               f"{integer(st['single_worker_capacity_flows_per_s'])} flow records/s.")
    t.write(out)
    N.num("online.per_flow_ms", st["per_flow_ms_mean"], 3)
    N.int("online.capacity", st["single_worker_capacity_flows_per_s"])
    N.put("online.batch_size", int(st["batch_size"]))
    N.int("online.records", st["records"])
    N.num("online.model_mb", r.get("model_megabytes"), 2)
    N.num("online.bundle_mb", r.get("bundle_megabytes"), 2)
    N.put("online.trees", int(r.get("n_trees", 0)))
    eq = r.get("engine_equivalence") or {}
    N.put("online.equiv.max_diff", eq.get("max_abs_difference"), sci(eq.get("max_abs_difference")))
    N.int("online.equiv.records", eq.get("records"))
    N.num("online.equiv.us_per_record", eq.get("engine_microseconds_per_record"), 1)
    N.put("online.equiv.nan_mismatch", eq.get("nan_mismatches"))
    load = r.get("load_test") or []
    if load:
        t = Table("load",
                  "Load test of the online path: flow records are replayed at a fixed arrival rate; latency is the "
                  "time from the arrival of a finalized record to its verdict (queueing, batching and processing). "
                  "A row is marked saturated when the achieved rate falls below 97\\% of the offered rate or the "
                  "95th-percentile latency exceeds one second.",
                  "rrrrrrrl", [["Workers", "Offered (flows/s)", "Achieved (flows/s)", "Mean", "Median", "P95", "P99", "State"],
                               ["", "", "", "\\multicolumn{4}{c}{Latency (ms)}", ""]], size=r"\footnotesize")
        for row in load:
            L = row["latency_ms"]
            t.row(row["workers"], integer(row["offered_rate"]), integer(row["achieved_rate"]), f(L["mean"], 1), f(L["p50"], 1),
                  f(L["p95"], 1), f(L["p99"], 1), "saturated" if row["saturated"] else "stable")
        t.write(out)
        save_csv(out, "latency_load_test", [{"workers": x["workers"], "offered_rate": x["offered_rate"], "achieved_rate": x["achieved_rate"],
                                             "records": x["records"], "saturated": x["saturated"],
                                             **{f"latency_ms_{k}": v for k, v in x["latency_ms"].items()},
                                             "rss_mb_per_worker": x["rss_mb_per_worker"]} for x in load])
        stable = [x for x in load if not x["saturated"]]
        for w in sorted({x["workers"] for x in load}):
            sw = [x for x in stable if x["workers"] == w]
            if sw:
                top = max(sw, key=lambda x: x["offered_rate"])
                N.int(f"online.load.w{w}.max_rate", top["offered_rate"])
                N.num(f"online.load.w{w}.p95", top["latency_ms"]["p95"], 1)
                N.num(f"online.load.w{w}.p50", top["latency_ms"]["p50"], 1)
                N.num(f"online.load.w{w}.p99", top["latency_ms"]["p99"], 1)
        if stable:
            top = max(stable, key=lambda x: x["offered_rate"])
            N.int("online.load.max_rate", top["offered_rate"])
            N.put("online.load.max_rate_workers", int(top["workers"]))
            nw = int(top["workers"])
            words = {1: "one", 2: "two", 3: "three", 4: "four", 6: "six", 8: "eight"}
            N.put("online.load.max_rate_workers_text", nw, f"{words.get(nw, str(nw))} worker{'s' if nw > 1 else ''}")
            N.num("online.load.max_rate_p95", top["latency_ms"]["p95"], 1)
            N.num("online.load.max_rate_p50", top["latency_ms"]["p50"], 1)
        # resident memory of a worker: only separate worker processes measure it; with one worker the replay runs in
        # the experiment process, whose memory also holds the corpus
        sep = [x["rss_mb_per_worker"] for x in load if x["workers"] > 1]
        N.num("online.rss_mb", float(np.mean(sep)) if sep else float(np.mean([x["rss_mb_per_worker"] for x in load])), 0)
        N.put("online.rss_from_separate_workers", "yes" if sep else "no")
        lo_rate = min(x["offered_rate"] for x in load)
        for w in sorted({x["workers"] for x in load}):
            r0 = [x for x in load if x["workers"] == w and x["offered_rate"] == lo_rate]
            if r0:
                N.num(f"online.load.w{w}.p95_low", r0[0]["latency_ms"]["p95"], 1)
        N.int("online.load.low_rate", lo_rate)
        N.put("online.load.max_workers", int(max(x["workers"] for x in load)))
        N.put("online.load.n_points", len(load))
        try:
            plt = _mpl()
            fig, ax = plt.subplots(figsize=(3.8, 2.6))
            cols = [C1, C2, C3]
            marks = ["o", "s", "^"]
            styles = ["-", "--", "-."]
            for i, w in enumerate(sorted({x["workers"] for x in load})[:3]):
                rows = sorted([x for x in load if x["workers"] == w], key=lambda x: x["offered_rate"])
                ax.plot([x["offered_rate"] for x in rows], [x["latency_ms"]["p95"] for x in rows], styles[i] + marks[i],
                        color=cols[i], label=f"{w} worker{'s' if w > 1 else ''}", markeredgecolor="white", markeredgewidth=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Offered load (flow records/s)")
            ax.set_ylabel("P95 latency (ms)")
            ax.legend(loc="best")
            _save(fig, out, "latency_load")
        except Exception as exc:  # pragma: no cover
            log.warning("figure latency_load failed: %s", exc)
    else:
        placeholder(out, "load", "Load test of the online path.")
    nc = r.get("normalization_comparison") or []
    if nc:
        t = Table("normalization",
                  "Detection quality of the same model on the test split (in order of record availability) with the "
                  "training-frozen scaling statistics and with sliding-window statistics re-estimated online.",
                  "lrrrrrr", [["Scaling statistics", "Precision", "Recall", "F1", "FPR (\\%)", "PR-AUC", "AUC-ROC"]])
        for row in nc:
            t.row(re.sub(r"W = (\d+)", lambda m_: "$W$ = " + integer(int(m_.group(1))), tex(row["normalization"])),
                  f(row["precision"]), f(row["recall"]), f(row["f1"]),
                  pct(row["fpr"], 2), f(row["pr_auc"]), f(row.get("auc_roc")))
        t.write(out)
        save_csv(out, "normalization_comparison", [{k: v for k, v in x.items() if k != "families"} for x in nc])
        for i, row in enumerate(nc):
            tag = "frozen" if i == 0 else f"sliding{i}"
            for k in ("f1", "precision", "recall", "pr_auc", "auc_roc"):
                N.num(f"online.norm.{tag}.{k}", row.get(k))
            N.pct(f"online.norm.{tag}.fpr", row.get("fpr"), 2)
        for i, row in enumerate(nc[1:], 1):
            N.put(f"online.norm.sliding{i}.df1", float(row["f1"] - nc[0]["f1"]), signed(row['f1'] - nc[0]['f1']))
            m = re.search(r"(\d+)", row["normalization"])
            if m:
                N.int(f"online.norm.sliding{i}.W", int(m.group(1)))
    else:
        placeholder(out, "normalization", "Frozen against sliding-window scaling statistics.")
    hd = r.get("hold_down")
    if hd:
        N.put("online.hold.seconds", float(hd["hold_seconds"]), f"{hd['hold_seconds']:g}")
        N.pct("online.hold.benign_blocked", hd["benign_flows_blocked_share"], 2)
        N.pct("online.hold.malicious_blocked", hd["malicious_flows_blocked_share"], 1)


def b_cross(cfg, ws, R, N, out) -> None:
    r = R.get("cross_dataset")
    if not r or "target_frozen_statistics" not in r:
        placeholder(out, "cross", "Cross-dataset evaluation" + (f" ({r.get('status')})" if r and r.get("status") else "") + ".")
        N.put("cross.available", 0, "0")
        return
    N.put("cross.available", 1, "1")
    seeds = r.get("seeds", [])
    t = Table("cross",
              f"Cross-dataset evaluation: the detector is trained on {tex(r['source'])} (without the transfer oracle, so no "
              f"label of the target dataset is used anywhere) and applied unchanged to {integer(r['target_rows'])} flows "
              f"of {tex(r['target'])} extracted with the same Zeek script. Mean $\\pm$ standard deviation over {len(seeds)} seeds.",
              "p{0.36\\linewidth}rrrr", [["Evaluation", "Precision", "Recall", "F1", "FPR (\\%)"]], size=r"\footnotesize")
    for key, lab in (("source_test", f"Source test split ({tex(r['source'])})"),
                     ("target_frozen_statistics", "Target, scaling statistics frozen from the source"),
                     ("target_recalibrated_statistics", "Target, scaling statistics re-estimated on unlabelled target traffic")):
        s = r[key]
        t.row(lab, *metric_cells(s), pct(s["fpr"]["mean"], 2))
        short = {"source_test": "cross.source", "target_frozen_statistics": "cross.frozen",
                 "target_recalibrated_statistics": "cross.recal"}[key]
        for k in ("precision", "recall", "f1", "auc_roc", "pr_auc"):
            N.summary(f"{short}.{k}", s[k])
        N.pct(f"{short}.fpr", s["fpr"]["mean"], 2)
        for fm, v in (s.get("families") or {}).items():
            N.num(f"{short}.family.{fm}", v["mean"])
    t.write(out)
    N.int("cross.target_rows", r["target_rows"])
    N.pct("cross.target_malicious_share", r["target_malicious_share"], 1)
    N.put("cross.target", r["target"], tex(r["target"]))
    N.put("cross.oracles", join_words([ORACLE_NOUN.get(n, n) for n in r.get("oracles_used", [])]))
    d = r["target_frozen_statistics"]["f1"]["mean"] - r["source_test"]["f1"]["mean"]
    N.put("cross.df1", float(d), signed(d))
    fams = r["target_frozen_statistics"].get("families") or {}
    cp_note, cp_hw, cp_ends = binomial_note(
        [(fm, _family_text(fm), fams[fm]["support"], fams[fm]["mean"] * fams[fm]["support"])
         for fm in fam_sorted(fams.keys()) if isnum(fams[fm].get("mean")) and isnum(fams[fm].get("support"))],
        "at the mean number of detected flows over the seeds")
    for fm, (lo_, hi_) in cp_ends.items():
        N.num(f"cross.frozen.family.{fm}.cp_hi", hi_)
    t = Table("cross_family",
              f"Recall per attack family on the target dataset ({tex(r['target'])}), scaling statistics frozen from the "
              f"source; every family of the target dataset is listed (mean $\\pm$ standard deviation over the seeds).",
              "lrr", [["Family", "Target flows", "Recall"]], notes=(cp_note.replace("over the test flows", "over the target flows") or None))
    for fm in fam_sorted(fams.keys()):
        t.row(FAMILY_LABEL.get(fm, fm), integer(fams[fm]["support"]), pm(fams[fm]))
    t.write(out)
    save_csv(out, "cross_dataset_family", [{"family": k, **v} for k, v in fams.items()])


def b_second(cfg, ws, R, N, out) -> None:
    """The method applied to CICIDS2017 without its labels (docs/ORACLES_EXTENDED.md, protocol step 5;
    configs/ids2017_ext.yaml), read from ``report.second_work_dir``.  Keys ``ds2.*``."""
    sd = R.get("_second") or {}
    m = sd.get("main")
    lc = sd.get("label_check") or {}
    if not m:
        placeholder(out, "second_dataset", TABLE_PURPOSE["second_dataset"])
        N.put("ds2.available", 0, "0")
        return
    N.put("ds2.available", 1, "1")
    s = m["summary"]
    for k in ("precision", "recall", "f1", "auc_roc", "pr_auc", "macro_recall"):
        N.summary(f"ds2.main.{k}", s.get(k))
    N.pct("ds2.main.fpr.pct", s["fpr"]["mean"], 2)
    N.put("ds2.main.n_seeds", len(m.get("seeds", [])))
    N.int("ds2.n", (m.get("pipeline") or {}).get("n_rows"))
    pl = m.get("pseudo_labels") or {}
    ptr, pte = pl.get("train_before_refinement") or {}, pl.get("test_after_refinement") or {}
    N.int("ds2.n_train", ptr.get("n"))
    N.int("ds2.n_test", pte.get("n"))
    N.num("ds2.pl.train.precision", ptr.get("precision"))
    N.num("ds2.pl.train.recall", ptr.get("recall_all_malicious"))
    N.num("ds2.pl.test.precision", pte.get("precision"))
    N.num("ds2.pl.test.recall", pte.get("recall_all_malicious"))
    N.num("ds2.pl.test.f1", pte.get("f1"))
    N.pct("ds2.malicious_share", pte.get("true_malicious_share"), 1)
    od = (m.get("pipeline") or {}).get("oracle_describe") or {}
    if "rate" in od:
        N.num("ds2.rate.threshold", float(od["rate"]["threshold"]), 1)
    names = list((m.get("pipeline") or {}).get("oracles") or lc.get("oracles") or [])
    N.put("ds2.oracles", join_words([ORACLE_NOUN.get(n, n) for n in names]))
    srcs = (((lc.get("aggregation_train") or {}).get("ds") or {}).get("model") or {}).get("sources")
    if srcs:
        merged = [x for x in srcs if "+" in x]
        groups = [" and the ".join(ORACLE_NOUN.get(n, n) for n in x.split("+")) for x in merged]
        N.put("ds2.merged", join_words(groups) or "none")
        N.put("ds2.merged_text", ("merges the " + "; it also merges the ".join(groups)) if groups
              else "merges no two oracles there")
    if m.get("params_source"):
        N.put("ds2.params_source", m["params_source"], tex(m["params_source"]))
    # per family: detector (representative seed) and pseudo-labels of the test flows
    fam_det = {x["family"]: x for x in (m.get("representative") or {}).get("families", [])}
    fam_pl = {x["family"]: x for x in (pte.get("families") or [])} if isinstance(pte.get("families"), list) else \
        {k: v for k, v in (pte.get("families") or {}).items()}
    fams = fam_sorted(set(fam_det) | set(fam_pl))
    ds_name = str(cfg.get_path("report.second_dataset_name") or "the second dataset")
    # the configuration fixed before testing (refinement on), applied to the same dataset (configs/ids2017_ext_frozen.yaml)
    mf = (R.get("_second_frozen") or {}).get("main")
    fam_fr = {x["family"]: x for x in (mf.get("representative") or {}).get("families", [])} if mf else {}
    if mf:
        sf = mf["summary"]
        for k in ("precision", "recall", "f1", "macro_recall"):
            N.summary(f"ds2f.main.{k}", sf.get(k))
        N.pct("ds2f.main.fpr.pct", sf["fpr"]["mean"], 2)
        plf = (mf.get("pseudo_labels") or {}).get("train_after_refinement") or {}
        N.num("ds2f.pl.train.precision", plf.get("precision"))
        N.num("ds2f.pl.train.recall", plf.get("recall_all_malicious"))
        if mf.get("params_source"):
            N.put("ds2f.params_source", mf["params_source"], tex(mf["params_source"]))
        for fm, x in fam_fr.items():
            N.num(f"ds2f.family.{fm}.recall", x.get("recall"))
    N.put("ds2f.available", 1 if mf else 0, "1" if mf else "0")
    head = ["Family", "Test flows", "Pseudo-labels", "Detector"] + (["Detector, fixed before testing"] if mf else [])
    t = Table("second_dataset",
              "The method applied to " + tex(ds_name) + " without any of its labels (specified before the "
              "experiment, Section~\\ref{sec:cross}): the oracles other than the transfer classifier, every threshold "
              "recomputed from the unlabelled training flows, the hyperparameters selected on CSE-CIC-IDS2018. Recall per "
              f"family on the test split: pseudo-labels and detector of the selected configuration (seed "
              f"{m.get('representative_seed', '--')})"
              + ("; last column: detector of the configuration fixed before testing, with the refinement of the first "
                 f"version (seed {mf.get('representative_seed', '--')})" if mf else "")
              + f"; last rows: precision, recall, and F1-score of the pseudo-labels and, as mean $\\pm$ standard "
                f"deviation over {len(m.get('seeds', []))} seeds, of the detectors (the pseudo-labels do not depend on "
                "the seed).",
              "lrrr" + ("r" if mf else ""), [head], size=r"\footnotesize")
    rows_csv = []
    for fm in fams:
        d, p_ = fam_det.get(fm, {}), fam_pl.get(fm, {})
        sup = d.get("support", p_.get("support"))
        extra = [f(fam_fr.get(fm, {}).get("recall"))] if mf else []
        t.row(FAMILY_LABEL.get(fm, fm), integer(sup), f(p_.get("recall")), f(d.get("recall")), *extra)
        N.num(f"ds2.family.{fm}.recall", d.get("recall"))
        N.num(f"ds2.pl.family.{fm}.recall", p_.get("recall"))
        N.int(f"ds2.family.{fm}.support", sup)
        row = {"family": fm, "test_flows": sup, "pseudo_label_recall_test": p_.get("recall"),
               "detector_recall_test": d.get("recall")}
        if mf:
            row["detector_fixed_before_testing_recall_test"] = fam_fr.get(fm, {}).get("recall")
        rows_csv.append(row)
    t.rule()
    for lab_, key, plk in (("Precision", "precision", "precision"), ("Recall", "recall", "recall_all_malicious"),
                           ("F1", "f1", "f1")):
        extra = [pm(mf["summary"][key])] if mf else []
        t.row(lab_, "", f(pte.get(plk)), pm(s[key]), *extra)
    # exact binomial interval of the detector's recall per family (as under every per-family table; R2.5)
    cp_note, cp_hw, cp_ends = binomial_note(
        [(fm, _family_text(fm), fam_det[fm].get("support"), fam_det[fm].get("tp")) for fm in fams
         if fm in fam_det and isnum(fam_det[fm].get("tp"))],
        f"detector of seed {m.get('representative_seed', '--')}")
    if cp_note:
        t.notes = cp_note
    for fm, (lo_, hi_) in cp_ends.items():
        N.num(f"ds2.family.{fm}.cp_hi", hi_)
    t.write(out)
    save_csv(out, "second_dataset_family", rows_csv)
    # coverage of the training flows by attack (label check of the second dataset)
    rec = (lc.get("pseudo_label_recall_by_attack_train") or {}).get("ds") or {}
    att = lc.get("attacks_train") or {}
    vote = lc.get("malicious_vote_share_by_attack_train") or {}
    if rec and att:
        forder = {x: i for i, x in enumerate(fam_sorted({v["family"] for v in att.values()}))}
        order = sorted(att, key=lambda a: (forder.get(att[a]["family"], len(forder)), a))
        full = [a for a in order if rec.get(a, 0) >= 0.99]
        none_ = [a for a in order if rec.get(a, 0) == 0]
        part = [a for a in order if 0 < rec.get(a, 0) < 0.99]
        N.put("ds2.lc.n_attacks", len(order))
        N.put("ds2.lc.n_covered", len(full))
        nm = lambda a: tex(ATTACK_TEXT.get(a, a))
        N.put("ds2.lc.covered_attacks", join_words([nm(a) for a in full]) or "none")
        N.put("ds2.lc.n_partial", len(part))
        N.put("ds2.lc.partial_attacks", join_words([nm(a) for a in part]) or "none")
        N.put("ds2.lc.n_uncovered", len(none_))
        N.put("ds2.lc.uncovered_attacks", join_words([nm(a) for a in none_]) or "none")
        save_csv(out, "second_dataset_attacks", [
            {"family": att[a]["family"], "attack": a, "training_flows": att[a]["flows"], "pseudo_label_recall": rec.get(a),
             **{f"vote_{n}": (vote.get(n) or {}).get(a) for n in vote}} for a in order])
        for a in order:
            key = re.sub(r"[^A-Za-z0-9]+", "-", a).strip("-")
            N.num(f"ds2.lc.attack.{key}.recall", rec.get(a))
            for n in vote:                  # every oracle (the text names the votes behind each coverage)
                N.num(f"ds2.lc.vote.{n}.{key}", vote[n].get(a))
        # malicious votes on the brute-force training flows, counted (rounded shares read 0.000 although a single
        # signature alert exists on CICIDS2017; Sections 6.8 and 7.7)
        bf = [a for a in order if att[a]["family"] == "BruteForce"]
        if bf:
            nbf = sum(int(att[a]["flows"]) for a in bf)
            nv = {n: sum(int(round(float((vote[n] or {}).get(a) or 0.0) * int(att[a]["flows"]))) for a in bf) for n in vote}
            tot = sum(nv.values())
            N.int("ds2.lc.brute.n_train", nbf)
            N.put("ds2.lc.brute.n_mal_votes", tot)
            who = [f"{ORACLE_TEXT.get(n, n)} oracle" + (f": {c}" if len([x for x in nv.values() if x]) > 1 else "")
                   for n, c in nv.items() if c]
            N.put("ds2.lc.brute.vote_clause",
                  "on whose flows no oracle votes malicious" if tot == 0 else
                  f"on whose {nbf:,} training flows the oracles cast ".replace(",", "{,}")
                  + ("a single malicious vote" if tot == 1 else f"only {tot} malicious votes")
                  + f" ({'from the ' if tot == 1 else ''}{join_words(who)})")
    agg = (lc.get("aggregation_train") or {})
    for k in ("ds", "ds_ungrouped", "majority"):
        if k in agg:
            N.num(f"ds2.lc.{k}.precision", agg[k].get("precision"))
            N.num(f"ds2.lc.{k}.recall", agg[k].get("recall_all_malicious"))
    # quality of each oracle on the training flows (measured) and as the label model estimates it (without labels):
    # an oracle can be precise on the flows it flags and still carry little weight in the label model
    for n, o in (lc.get("per_oracle_train") or {}).items():
        N.num(f"ds2.oracle.{n}.precision", o.get("precision"))
        N.pct(f"ds2.oracle.{n}.benign_flagged", o.get("benign_flagged_share"), 2)
        N.num(f"ds2.oracle.{n}.false_alarm_voted", o.get("false_alarm_voted"))
    mdl = ((agg.get("ds") or {}).get("model") or {})
    for i, src in enumerate(mdl.get("sources") or []):
        sens, fa = (mdl.get("sensitivity") or [None] * 99)[i], (mdl.get("false_alarm") or [None] * 99)[i]
        N.num(f"ds2.lm.{src}.sens_est", sens)
        N.num(f"ds2.lm.{src}.fa_est", fa)
        if isnum(sens) and isnum(fa) and fa > 0:
            N.num(f"ds2.lm.{src}.lr_malicious", sens / fa, 1)     # likelihood ratio of a malicious vote


def b_qos(cfg, ws, R, N, out) -> None:
    """QoS under enforcement by trace-driven emulation (experiment ``qos``, twws/qos.py): table, hold-down table,
    figure with the metrics of the first version's QoS figures, numbers ``qos.*``."""
    r = R.get("qos")
    if not r or not r.get("rows"):
        placeholder(out, "qos", "QoS under enforcement, trace-driven emulation.")
        placeholder(out, "qos_hold", "QoS under enforcement for different hold-down periods.")
        return
    ok = [x for x in r["rows"] if "throughput_preservation" in x]
    cond = lambda a: "under attack" if a else "no attack"
    t = Table("qos",
              "QoS under enforcement, by trace-driven emulation on every flow record of the retained captures "
              f"({integer(r['verdicts']['records'])} records): a malicious verdict blocks its source for "
              f"{r['hold_s']:g}~s; the trace is replayed with all times divided by the speed factor, and the verdict "
              "follows the availability of the record by the 95th-percentile latency that the load test measured at the "
              "smallest sustained arrival rate at or above the emulated one, with the number of workers that gave the "
              "lowest latency there (Table~\\ref{tab:load}). Delay and jitter are those of the decision path "
              "(95th percentile, and 95th minus 50th percentile). Preservation: delivered/offered benign bytes; "
              "availability: share of benign records of which no part is dropped; loss: dropped/offered benign "
              "packets; SLA: share of one-second intervals in which less than 99\\% of the offered benign traffic is "
              "delivered; attack dropped: share of the attack bytes dropped.",
              "llrrrrrrrrr",
              [["Condition", "Speed", "Benign load", "Delay", "Jitter", "Goodput", "Preserv.", "Avail.", "Loss", "SLA",
                "Attack"],
               ["", "", "(Mbit/s)", "(ms)", "(ms)", "(Mbit/s)", "(\\%)", "(\\%)", "(\\%)", "(\\%)", "dropped (\\%)"]],
              size=r"\scriptsize")
    rows_csv = []
    for a in (False, True):
        for x in sorted([y for y in r["rows"] if bool(y["attack"]) == a], key=lambda y: y["speed"]):
            if "throughput_preservation" not in x:
                t.row(cond(a), f"{x['speed']:g}", "", r"\multicolumn{8}{l}{" + tex(x.get("status", "not emulated")) + "}")
                continue
            L = x.get("latency") or {}
            p95, p50 = L.get("p95_ms"), L.get("p50_ms")
            jit = (p95 - p50) if isnum(p95) and isnum(p50) else None
            t.row(cond(a), f"{x['speed']:g}", f(x["offered_benign_mbps"], 2), f(p95, 1), f(jit, 1),
                  f(x["goodput_mbps"], 2), pct(x["throughput_preservation"], 3), pct(x["availability"], 3),
                  pct(x["packet_loss_ratio"], 3), pct(x["sla_violation_share"], 2),
                  pct(x["attack_bytes_dropped_share"], 1) if a else "--")
            rows_csv.append({"condition": cond(a), "speed": x["speed"], "record_rate": x.get("record_rate"),
                             "offered_benign_mbps": x["offered_benign_mbps"], "delay_p95_ms": p95, "jitter_ms": jit,
                             "goodput_mbps": x["goodput_mbps"], "throughput_preservation": x["throughput_preservation"],
                             "availability": x["availability"], "packet_loss_ratio": x["packet_loss_ratio"],
                             "sla_violation_share": x["sla_violation_share"],
                             "attack_bytes_dropped_share": x["attack_bytes_dropped_share"],
                             "benign_sources_hit_share": x.get("benign_sources_hit_share"), "hold_s": x["hold_s"]})
        if not a:
            t.rule()
    t.write(out)
    save_csv(out, "qos_emulation", rows_csv)
    # hold-down sensitivity at speed 1
    hr = r.get("hold_rows") or []
    if hr:
        t2 = Table("qos_hold",
                   "QoS under enforcement at the original time scale of the trace (speed 1) for different hold-down "
                   "periods; definitions as in Table~\\ref{tab:qos}.",
                   "lrrrrrr", [["Condition", "Hold-down (s)", "Preserv. (\\%)", "Avail. (\\%)", "Loss (\\%)",
                                "SLA (\\%)", "Attack dropped (\\%)"]], size=r"\footnotesize")
        for a in (False, True):
            for x in sorted([y for y in hr if bool(y["attack"]) == a], key=lambda y: y["hold_s"]):
                t2.row(cond(a), f"{x['hold_s']:g}", pct(x["throughput_preservation"], 3), pct(x["availability"], 3),
                       pct(x["packet_loss_ratio"], 3), pct(x["sla_violation_share"], 2),
                       pct(x["attack_bytes_dropped_share"], 1) if a else "--")
                key = f"qos.hold{int(round(x['hold_s']))}.{'att' if a else 'na'}"
                N.pct(f"{key}.preservation", x["throughput_preservation"], 3)
                N.pct(f"{key}.sla", x["sla_violation_share"], 2)
                if a:
                    N.pct(f"{key}.attack_dropped", x["attack_bytes_dropped_share"], 1)
            if not a:
                t2.rule()
        t2.write(out)
        save_csv(out, "qos_hold_down", [{k: v for k, v in x.items() if not isinstance(v, dict)} for x in hr])
    else:
        placeholder(out, "qos_hold", "QoS under enforcement for different hold-down periods.")
    # numbers
    v = r["verdicts"]
    N.int("qos.records", v["records"])
    N.num("qos.verdict.precision", v["precision"])
    N.num("qos.verdict.recall", v["recall"])
    N.pct("qos.verdict.fpr", v["fpr"], 2)
    N.put("qos.hold", float(r["hold_s"]), f"{r['hold_s']:g}")
    N.num("qos.record_rate", r["record_rate"], 1)
    N.put("qos.n_days", len(r.get("per_day", [])))
    N.put("qos.seed", int(r["seed"]))
    sp_ok = sorted({x["speed"] for x in ok})
    if sp_ok:
        N.put("qos.speed_max", sp_ok[-1], f"{sp_ok[-1]:g}")
        N.put("qos.speeds", ", ".join(f"{z:g}" for z in sorted({x['speed'] for x in r['rows']})))
    beyond = sorted({x["speed"] for x in r["rows"] if "throughput_preservation" not in x})
    N.put("qos.n_beyond", len(beyond))
    N.put("qos.beyond_text", ("speed " + ", ".join(f"{z:g}" for z in beyond)) if beyond else "none")
    for a in (False, True):
        c = "att" if a else "na"
        xs = sorted([x for x in ok if bool(x["attack"]) == a], key=lambda y: y["speed"])
        if not xs:
            continue
        lo, hi = xs[0], xs[-1]
        N.num(f"qos.{c}.load_min", lo["offered_benign_mbps"], 2)
        N.num(f"qos.{c}.load_max", hi["offered_benign_mbps"], 1)
        N.int(f"qos.{c}.rate_max", hi.get("record_rate"))
        for tag, x in (("s1", lo), ("smax", hi)):
            N.pct(f"qos.{c}.{tag}.preservation", x["throughput_preservation"], 3)
            N.pct(f"qos.{c}.{tag}.availability", x["availability"], 3)
            N.pct(f"qos.{c}.{tag}.loss", x["packet_loss_ratio"], 3)
            N.pct(f"qos.{c}.{tag}.sla", x["sla_violation_share"], 2)
            N.pct(f"qos.{c}.{tag}.sources_hit", x.get("benign_sources_hit_share"), 2)
            if a:
                N.pct(f"qos.{c}.{tag}.attack_dropped", x["attack_bytes_dropped_share"], 1)
            L = x.get("latency") or {}
            if isnum(L.get("p95_ms")):
                N.num(f"qos.{c}.{tag}.delay_p95", L["p95_ms"], 1)
                N.num(f"qos.{c}.{tag}.jitter", L["p95_ms"] - L.get("p50_ms", L["p95_ms"]), 1)
    # figure: the metrics of the first version's QoS figures against the offered benign load
    try:
        plt = _mpl()
        panels = [("delay", "(a) Delay, P95 (ms)"), ("jitter", "(b) Jitter, P95$-$P50 (ms)"),
                  ("goodput", "(c) Goodput (Mbit/s)"), ("loss", "(d) Packet loss (%)"),
                  ("avail", "(e) Availability (%)"), ("sla", "(f) SLA violations (%)"),
                  ("pres", "(g) Throughput preserved (%)"), ("att", "(h) Attack bytes dropped (%)")]
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
        for (key, label), ax in zip(panels, axes.ravel()):
            for a, col, mk, ls in ((False, C1, "o", "-"), (True, C2, "s", "--")):
                if key == "att" and not a:
                    continue
                if key in ("delay", "jitter") and not a:     # one decision path for both conditions
                    continue
                xs = sorted([x for x in ok if bool(x["attack"]) == a], key=lambda y: y["speed"])
                if not xs:
                    continue
                X = [x["offered_benign_mbps"] for x in xs]
                L = [x.get("latency") or {} for x in xs]
                Y = {"delay": [l_.get("p95_ms") for l_ in L],
                     "jitter": [(l_.get("p95_ms", np.nan) - l_.get("p50_ms", np.nan)) for l_ in L],
                     "goodput": [x["goodput_mbps"] for x in xs],
                     "loss": [100 * x["packet_loss_ratio"] for x in xs],
                     "avail": [100 * x["availability"] for x in xs],
                     "sla": [100 * x["sla_violation_share"] for x in xs],
                     "pres": [100 * x["throughput_preservation"] for x in xs],
                     "att": [100 * (x["attack_bytes_dropped_share"] or 0.0) for x in xs]}[key]
                lab_ = "decision path" if key in ("delay", "jitter") else cond(a)
                ax.plot(X, Y, ls + mk, color=(C3 if key in ("delay", "jitter") else col), label=lab_,
                        markeredgecolor="white", markeredgewidth=0.7, markersize=4, linewidth=1.3)
            ax.set_xscale("log")
            if key == "goodput":
                ax.set_yscale("log")
            if key == "att":            # a share that barely changes: the full scale avoids exaggerating it
                ax.set_ylim(0, 100)
            ax.set_title(label, fontsize=7.5, loc="left")
            ax.tick_params(labelsize=6.5)
        for ax in axes[1]:
            ax.set_xlabel("Offered benign load (Mbit/s)", fontsize=7)
        axes[0, 3].legend(loc="best", fontsize=6.5)
        fig.tight_layout(pad=0.6, w_pad=0.8, h_pad=0.9)
        _save(fig, out, "qos_emulation")
    except Exception as exc:  # pragma: no cover
        log.warning("figure qos_emulation failed: %s", exc)


def b_memory(cfg, ws, R, N, out) -> None:
    """Memory measured during the experiments (replaces the memory table of the first version)."""
    rows = []
    ci_p = ws.corpus_dir / "corpus_info.json"
    if ci_p.exists():
        for x in read_json(ci_p).get("timing", []):
            if isnum(x.get("rss_end_mb")):
                rows.append(("Offline", f"Corpus: {x['stage']}", x.get("rss_end_mb"), x.get("peak_rss_mb_process")))
    o = R.get("online") or {}
    st = o.get("service_time") or {}
    if st:
        rows.append(("Online", "Model of the detector (serialized booster)", o.get("model_megabytes"), None))
        rows.append(("Online", "Deployed bundle (preprocessing, selection, model)", o.get("bundle_megabytes"), None))
        rows.append(("Online", "Experiment process before streaming (corpus and model loaded)", st.get("rss_mb_before"), None))
        nrec = st.get("records")       # plain digits with a comma: the item text is escaped by tex() below
        rows.append(("Online", f"Experiment process after streaming {int(nrec):,} records" if isnum(nrec)
                     else "Experiment process after streaming",
                     st.get("rss_mb_after"), st.get("process_peak_rss_mb")))
        load = o.get("load_test") or []
        # with one worker the load test runs in the experiment process itself; spawned workers show the footprint
        # of a deployed worker (preprocessing, selection, model, streaming state)
        wk = [x for x in load if x.get("workers", 1) > 1 and isnum(x.get("rss_mb_per_worker"))]
        if wk:
            rows.append(("Online", "Worker process of the load test (spawned workers, mean)",
                         float(np.mean([x["rss_mb_per_worker"] for x in wk])), None))
            N.num("mem.worker_mb", float(np.mean([x["rss_mb_per_worker"] for x in wk])), 0)
        N.num("mem.model_mb", o.get("model_megabytes"), 2)
        N.num("mem.bundle_mb", o.get("bundle_megabytes"), 1)
    rows = [r for r in rows if isnum(r[2]) or isnum(r[3])]
    if not rows:
        placeholder(out, "memory", "Memory measured during the experiments.")
        return
    t = Table("memory",
              "Memory measured during the experiments on the machine of Table~\\ref{tab:runtime}: resident set size "
              "(RSS) at the end of each stage and the peak of the process up to that point, in MiB; sizes of the "
              "serialized model and bundle in MB.",
              "llrr", [["Phase", "Item", "Size / RSS", "Peak RSS"]], size=r"\footnotesize")
    for ph, item, a, b in rows:
        t.row(ph, tex(item), f(a, 1) if isnum(a) else "--", f(b, 0) if isnum(b) else "--")
    t.write(out)
    save_csv(out, "memory", [{"phase": ph, "item": item, "size_or_rss": a, "peak_rss": b} for ph, item, a, b in rows])
    peaks = [r[3] for r in rows if r[0] == "Offline" and isnum(r[3])]
    if peaks:
        N.num("mem.offline_peak_mb", max(peaks), 0)


def b_runtime(cfg, ws, R, N, out) -> None:
    """Wall-clock time of the stages for which a measurement exists."""
    rows = []
    ci_p = ws.corpus_dir / "corpus_info.json"
    einfo_p = ws.capture_root / "extract" / "extract_info.json"
    # extraction runs day by day and in parallel: the time of each tool is summed over all retained captures
    zrep_p = ws.capture_root / "extract" / "extract_report.csv"
    if zrep_p.exists():
        z = pd.read_csv(zrep_p)
        z = z[z["day"].astype(str).isin([str(d) for d in cfg.corpus.days])]
        rows.append(("Flow extraction with Zeek, summed over captures", float(z["seconds"].sum())))
    snort_secs = [float(p.read_text().strip() or 0) for d in cfg.corpus.days
                  for p in (ws.capture_root / "snort" / "snort" / str(d)).glob("*/snort.seconds")]
    if snort_secs:                               # one file per capture, written by every Snort run
        rows.append(("Signature oracle: Snort replay, summed over captures", float(sum(snort_secs))))
    if ci_p.exists():
        for x in read_json(ci_p).get("timing", []):
            rows.append((f"Corpus: {x['stage']}", x["seconds"]))
    pairs_p = ws.corpus_dir / "pairs_info.json"    # inputs of the oracles 'rate' and 'beacon' (twws.cli pairs)
    if pairs_p.exists():
        pi = read_json(pairs_p)
        rows.append((f"Pair statistics over the stream ({int(pi.get('stream_records', 0)):,} records)", pi.get("seconds")))
    m = R.get("main")
    if m:
        tm = m["pipeline"].get("timing", {})
        ae = (m["pipeline"]["feature_pipeline"].get("autoencoder") or {}).get("seconds")
        rows.append(("Partition, imputation, scaling, autoencoder (stage A)", tm.get("stage_a")))
        if isnum(ae):
            rows.append((r"\quad of which autoencoder training", ae))
        for k, v in (tm.get("oracles") or {}).items():
            rows.append((f"Oracle: {ORACLE_LABEL.get(k, k).lower()}", v))
        rows.append(("Feature selection", m["pipeline"]["selection"].get("seconds")))
        if (m["pipeline"].get("refinement") or {}).get("enabled", True):     # not timed when it is switched off
            rows.append(("Refinement", m["pipeline"]["refinement"].get("seconds")))
        rows.append(("Detector training, one seed (mean)", float(np.mean([x["fit_seconds"] for x in m["runs"]]))))
    s = R.get("search")
    if s:
        rows.append((f"Hyperparameter search ({s['n_trials']} trials on {int(s['search_rows']):,} rows)", s["seconds"]))
    rows = [(a, b) for a, b in rows if isnum(b)]
    if not rows:
        placeholder(out, "runtime", "Measured wall-clock time per pipeline stage.")
        return
    env = (m or s or {}).get("environment", {})
    has_oracles = any(a.startswith("Oracle: ") for a, _ in rows)
    t = Table("runtime",
              "Measured wall-clock time per pipeline stage on the machine used for the experiments "
              f"({tex(env.get('cpu_model') or 'CPU not recorded')}, {env.get('logical_cpus', '--')} logical CPUs, "
              f"{env.get('memory_gb', '--')} GB memory).",
              "lr", [["Stage", "Time"]],
              notes=("The oracle rows give the time needed to compute the votes of the corpus flows from inputs that "
                     "earlier stages store: the Snort alerts of the replay and the pair statistics.") if has_oracles else None)

    def _fmt(b: float) -> str:
        if b < 0.05:
            return r"$<$0.1 s"
        return f"{b:.1f} s" if b < 120 else f"{b / 60:.1f} min"

    for a, b in rows:
        t.row(a if a.startswith("\\") else tex(a), _fmt(b))
    t.write(out)
    save_csv(out, "runtime", [{"stage": a.replace("\\quad ", ""), "seconds": b} for a, b in rows])


SUPPLEMENTARY_FILES = {
    "memory": "Memory measured during the experiments (RSS per stage, model and bundle sizes)",
    "qos_emulation": "QoS under enforcement (trace-driven emulation): every condition and speed factor with offered load, "
                     "delay, jitter, goodput, preservation, availability, packet loss, SLA violations, attack dropped",
    "qos_hold_down": "QoS under enforcement at speed 1 for every hold-down period",
    "second_dataset_family": "The method applied to CICIDS2017 without its labels: recall per family of the pseudo-labels and "
                             "of the detector on the test split",
    "second_dataset_attacks": "The method applied to CICIDS2017 without its labels: share of the training flows of each "
                              "attack that the pseudo-labels and each oracle label malicious",
    "manifest": "Retained captures: archive, member name, size, CRC-32 stored in the archive, SHA-256 of the capture as distributed",
    "capture_checks": "Structural check of every retained capture: complete packet records, damaged regions skipped and bytes "
                      "removed by the repair, size and SHA-256 as distributed",
    "archive_members": "Every member of every daily archive with its role (victim, background, not selected)",
    "snort_rule_files": "Rule files of the signature oracle with the number of active rules and SHA-256 digests",
    "snort_top_signatures": "Most frequent signatures raised on the retained captures",
    "snort_alerts_by_class": "Snort alerts per rule class (classtype); only the attack classes count for the signature oracle",
    "label_summary_by_day": "Flows per capture day and class after schedule-based labelling",
    "utc_offset_diagnostic": "Share of attacker flows inside the scheduled attack windows for every candidate clock offset",
    "label_summary_by_day_ids2017": "CICIDS2017: flows per capture day and class after schedule-based labelling",
    "utc_offset_diagnostic_ids2017": "CICIDS2017: share of flows with an attacker address inside the scheduled attack windows "
                                     "for every candidate clock offset",
    "oracle_statistics": "Coverage, abstention, precision, and recall of each oracle against ground truth",
    "oracle_votes_by_family": "Votes of each oracle per class",
    "oracle_pairwise": "Agreement, error correlation, and joint error rate of every pair of oracles",
    "oracle_joint_errors": "Number of oracles that are wrong on the same flow, observed and under independence",
    "trust_calibration": "Accuracy of the pseudo-label per trust-score bin for both aggregation rules",
    "oracle_ablation": "Detector trained on pseudo-labels from different oracle sets and aggregation rules",
    "search_trials": "Configuration and score of every trial of the hyperparameter search",
    "main_per_seed": "Detection metrics of every seed",
    "pseudo_label_recall_by_family": "Recall of the pseudo-labels per attack family",
    "pseudo_label_recall_by_attack": "Share of the malicious training flows of each attack labelled malicious by each aggregation rule and voted malicious by each oracle",
    "selected_features": "Every feature with its stability score and whether it was selected",
    "pipeline_seed_runs": "Complete pipeline re-fitted with different seeds",
    "ablations": "Every ablation variant: mean and standard deviation over seeds, share of clipped rows",
    "feature_selection_comparison": "Feature-selection methods: metrics, selection time, and row sample used",
    "loao_per_seed": "Leave-one-attack-out recall of every held-out family and seed",
    "partition_protocols": "Partition protocols with and without host-level features",
    "protocol_attacks_by_split": "Malicious flows of every attack in the training, validation and test subsets of each partition protocol",
    "baselines": "Baselines: mean and standard deviation over seeds, paired bootstrap interval",
    "baselines_per_seed": "Baselines: metrics of every seed",
    "adversarial": "Recall under every evasion transformation and strength, standard and augmented training",
    "adversarial_by_family": "Recall under evasion per attack family",
    "shap_global": "Mean absolute TreeSHAP contribution of every selected feature",
    "shap_by_class": "Leading TreeSHAP contributions per class with direction",
    "latent_probes": "Latent features: most correlated named features and linear explained variance",
    "latency_load_test": "Load test: latency statistics for every worker count and arrival rate",
    "normalization_comparison": "Frozen and sliding-window normalization at deployment",
    "cross_dataset_family": "Cross-dataset recall per attack family",
    "runtime": "Wall-clock time per pipeline stage",
}


def b_supplementary(cfg, ws, R, N: Numbers, out: Path) -> None:
    """Table of the supplementary files that this report actually wrote (never a hand-typed list)."""
    files = sorted(p.stem for p in (out / "supplementary").glob("*.csv"))
    N.put("supp.n_files", len(files))
    if not files:
        placeholder(out, "supplementary_files", "Supplementary files with the numerical values behind the tables and figures.")
        return
    order = [k for k in SUPPLEMENTARY_FILES if k in files] + [k for k in files if k not in SUPPLEMENTARY_FILES]
    t = Table("supplementary_files",
              "Supplementary files (comma-separated values) with the numerical values behind the tables and figures.",
              r"@{}p{0.36\linewidth}p{0.60\linewidth}@{}", [["File", "Content"]], size=r"\footnotesize", long=True)
    for k in order:
        if k not in SUPPLEMENTARY_FILES:
            log.warning("supplementary file %s.csv has no description", k)
        t.row(r"\path{" + k + ".csv}", tex(SUPPLEMENTARY_FILES.get(k, "(no description)")))
    t.row(r"\path{tables/<label>.csv}", "The cells of every table produced by the released code, one file per table "
          r"(for example \path{tables/main.csv} for Table~\ref{tab:main}); \path{tables/index.csv} gives the caption "
          "of the table that each file holds")
    t.write(out)


def b_diagnostics(cfg, ws, R, N, out) -> None:
    """Diagnostics of the labelling rules computed after the results (twws/experiments/diagnostics.py): which oracle
    covers the botnet flows of the forward-in-time test days, the votes of the two repetition oracles under the
    connection spacing of the evasion experiment, and the repeated-connection threshold on the second dataset."""
    D = R.get("_diag") or {}
    fo = D.get("diag_oracles")
    if fo:
        N.num("diag.fwd.rate_threshold", (fo.get("describe") or {}).get("rate", {}).get("threshold"), 1)
        tb = fo.get("test_botnet") or {}
        if tb:
            N.int("diag.fwd.bot.n", tb["n"])
            for k in ("rate_malicious", "beacon_malicious", "rate_only", "beacon_only", "both", "neither",
                      "other_oracles_malicious", "pseudo_label_malicious", "pseudo_label_given_rate",
                      "pseudo_label_given_beacon_only"):
                N.num(f"diag.fwd.bot.{k}", tb.get(k))
            # counts behind the shares: flows that only the regularity oracle flags, and how many of them the
            # pseudo-labels call malicious
            if isnum(tb.get("beacon_only")) and isnum(tb.get("pseudo_label_given_beacon_only")):
                nb = int(round(tb["beacon_only"] * tb["n"]))
                N.int("diag.fwd.bot.n_beacon_only", nb)
                N.int("diag.fwd.bot.n_pl_beacon_only", int(round(tb["pseudo_label_given_beacon_only"] * nb)))
        val = (fo.get("subsets") or {}).get("validation") or {}
        att = {a: r for a, r in val.items() if a != "Benign"}
        if att:
            N.int("diag.fwd.val.malicious", sum(r["n"] for r in att.values()))
            N.num("diag.fwd.val.pl_max", max(r["pseudo_label_malicious"] for r in att.values()))
    sp = D.get("diag_spacing")
    if sp:
        N.int("diag.space.n", sp["controlled_records"])
        N.num("diag.space.threshold", sp["threshold"], 1)
        fams = fam_sorted(sp.get("families") or [])
        levels = [row["level"] for row in sp["levels"]]
        for row in sp["levels"]:
            lv = f"{row['level']:g}"
            for fm, r in row["families"].items():
                N.num(f"diag.space.{lv}.{fm}.rate", (r["rate"] or {}).get("malicious"))
                N.num(f"diag.space.{lv}.{fm}.rate_benign", (r["rate"] or {}).get("benign"))
                N.num(f"diag.space.{lv}.{fm}.rate_abstain", (r["rate"] or {}).get("abstain"))
                N.num(f"diag.space.{lv}.{fm}.beacon", (r["beacon"] or {}).get("malicious"))
                ri = r.get("regularity_inputs") or {}
                if "iat_med_below_min" in ri:     # which condition of the regularity rule the flows miss
                    N.num(f"diag.space.{lv}.{fm}.iat_below_min", ri["iat_med_below_min"])
                    N.num(f"diag.space.{lv}.{fm}.disp_ok", ri["disp_at_most_max"])
            N.num(f"diag.space.{lv}.all.rate", row["all"]["rate"].get("malicious"))
            N.num(f"diag.space.{lv}.all.beacon", row["all"]["beacon"].get("malicious"))
        # the regularity oracle on the botnet flows over all spacing factors above 1 (lowest value)
        bb = [row["families"]["Bot"]["beacon"]["malicious"] for row in sp["levels"]
              if row["level"] > 1 and "Bot" in row["families"]]
        if bb:
            N.put("diag.space.bot.beacon_min_spaced", min(bb), f(math.floor(min(bb) * 1e3 + 1e-9) / 1e3))
        hdr = ["Oracle", "Family (flows)"] + [("none" if lv == 1 else f"$\\times${lv:g}") for lv in levels]
        t = Table("adv_oracles",
                  "Share of the malicious flows sent by attacker-controlled endpoints (all "
                  f"{integer(sp['controlled_records'])} such flows of the stream, not only the test flows) on which the "
                  "two repetition oracles vote malicious, and (last rows) on which the repeated-connection oracle "
                  "abstains because the rate lies in the guard band around its threshold, without spacing (none) and "
                  "after the connection spacing of "
                  "Table~\\ref{tab:adversarial}; threshold of the repeated-connection oracle as in the selected "
                  f"configuration ({f(sp['threshold'], 1)} connections per pair and minute). Computed after the results "
                  "(Section~\\ref{sec:adv}); no label model and no detector are involved.",
                  "ll" + "r" * len(levels), [hdr], size=r"\footnotesize")
        for oi, (okey, olab, vk) in enumerate((("rate", "Repeated connections", "malicious"),
                                                ("beacon", "Regular repetition", "malicious"),
                                                ("rate", "Repeated connections: abstains", "abstain"))):
            if oi:
                t.rule()
            for j, fm in enumerate(fams):
                n_ = sp["levels"][0]["families"][fm]["n"]
                t.row(olab if j == 0 else "", f"{FAMILY_LABEL.get(fm, fm)} ({integer(n_)})",
                      *[f(row["families"][fm][okey].get(vk)) for row in sp["levels"]])
        t.write(out)
    else:
        placeholder(out, "adv_oracles")
    ds_ = D.get("diag_selection")
    vs = [v for v in ((ds_ or {}).get("variants") or []) if "summary" in v]
    ref_v = next((v for v in vs if v["name"].startswith("reference")), None)
    if ref_v and len(vs) > 1:
        rs = ref_v["summary"]
        rf1 = {int(r["seed"]): float(r["f1"]) for r in ref_v.get("runs", [])}
        refsel = set(ref_v.get("selected") or [])
        t = Table("abl_selection_extra",
                  "Feature selection with the size $d'$ of the mutual-information filter and the number $F$ of folds "
                  "changed one at a time (the two settings of Step~5 that Table~\\ref{tab:abl_selection} keeps fixed; "
                  f"reference: $d'={ds_['reference']['mi_top']}$, $F={ds_['reference']['n_folds']}$). Computed after the "
                  "results in the development container, on the ablation seeds and with the hyperparameters of the "
                  "selected configuration; the reference row is the selected configuration run in the same environment, "
                  "where its selection keeps " + str(ref_v["n_features"]) + " features. $\\Delta$F1: mean over the seeds "
                  "of the difference to the reference trained with the same seed. Overlap: Jaccard index of the selected "
                  "features and those of the reference.",
                  "lrrrrrrr", [["Variant", "Precision", "Recall", "F1", "$\\Delta$F1", "Features", "Overlap", "PR-AUC"]],
                  size=r"\footnotesize")
        dfs, jac = [], []
        for v in vs:
            sm = v["summary"]
            vf1 = {int(r["seed"]): float(r["f1"]) for r in v.get("runs", [])}
            common = sorted(set(vf1) & set(rf1))
            is_ref = v is ref_v
            d = (sum(vf1[k] - rf1[k] for k in common) / len(common)) if common and not is_ref else None
            sel = set(v.get("selected") or [])
            j = (len(sel & refsel) / len(sel | refsel)) if (sel or refsel) and not is_ref else None
            if d is not None:
                dfs.append(d)
            if j is not None:
                jac.append(j)
            lab = re.sub(r"\bF = (\d+)", r"$F=\1$", re.sub(r"d' = (\d+)", r"$d'=\1$", tex(v["name"])))
            lab = lab[:1].upper() + lab[1:]
            t.row(lab, *metric_cells(sm), "--" if d is None else signed(d, 4), str(v["n_features"]),
                  "--" if j is None else f(j, 2), pm(sm.get("pr_auc")))
            if is_ref:
                t.rule()
        t.write(out)
        N.summary("diagsel.ref.f1", rs["f1"])
        N.put("diagsel.ref.n_features", int(ref_v["n_features"]))
        N.put("diagsel.n_variants", len(vs) - 1)
        if dfs:
            mx = max(abs(x) for x in dfs)
            N.put("diagsel.max_abs_df1", mx, f(math.ceil(mx * 1e4 - 1e-9) / 1e4, 4))
        if jac:
            N.num("diagsel.min_overlap", min(jac), 2)
        nf = [int(v["n_features"]) for v in vs]
        N.put("diagsel.n_features_min", min(nf))
        N.put("diagsel.n_features_max", max(nf))
    else:
        placeholder(out, "abl_selection_extra")
    rt = D.get("diag_rate_threshold_ids2017")
    if rt:
        N.num("diag.ds2rate.threshold_own", rt["threshold_own"], 1)
        N.int("diag.ds2rate.pairs", rt.get("pairs_train"))
        N.int("diag.ds2rate.pairs_above", rt.get("pairs_above_own"))
        N.int("diag.ds2rate.pairs_above_attack", rt.get("pairs_above_own_with_attack"))
        if isinstance(rt.get("pairs_above_own"), int) and isinstance(rt.get("pairs_above_own_with_attack"), int):
            N.int("diag.ds2rate.pairs_above_benign", rt["pairs_above_own"] - rt["pairs_above_own_with_attack"])
        for a, r in (rt.get("attacks") or {}).items():
            key = a
            q = r.get("n60_quantiles") or {}
            if "0.5" in q:
                N.num(f"diag.ds2rate.{key}.n60_median", q["0.5"], 0)
            N.num(f"diag.ds2rate.{key}.n60_max", r.get("n60_max"), 0)
            for thr_s, v in (r.get("votes") or {}).items():
                N.num(f"diag.ds2rate.{key}.at.{thr_s}", v.get("malicious"))
        for thr_s, v in ((rt.get("benign_train") or {}).get("votes") or {}).items():
            N.pct(f"diag.ds2rate.benign.at.{thr_s}", v.get("malicious"), 2)


def b_main_steps(cfg, ws, R, N, out) -> None:
    """From the five oracles of the first version to the selected configuration (docs/ORACLES_EXTENDED.md): the
    training labels and the detector of each configuration, and the pseudo-labels of the test flows without detector."""
    main = R.get("main")
    ref = (R.get("_ref") or {}).get("main")
    fr = (R.get("_frozen") or {}).get("main")
    if not main or not cfg.get("corpus_work_dir"):
        placeholder(out, "main_steps", TABLE_PURPOSE["main_steps"])
        return
    ff = (R.get("_frozen_first") or {}).get("runs") or []
    first_note = ""
    if ff and fr:
        from ..metrics import seed_summary
        bp = ((R.get("_frozen_first") or {}).get("search") or {}).get("best_params")
        bq = ((R.get("_frozen") or {}).get("search") or {}).get("best_params")
        other = ("its own hyperparameter search selected other settings and " if bp and bq and bp != bq else "")
        first_note = (" The configuration fixed before testing was first run on another machine, where "
                      f"{other}the detector reached an F1-score of "
                      f"{pm(seed_summary([x['f1'] for x in ff]))} over the same {len(ff)} seeds (log released with the "
                      "code); the refinement was removed after that run. Its row reports the run on the machine of all "
                      "other results.")
    t = Table("main_steps",
              "From the oracles of the first version to the selected configuration: quality of the training labels "
              "that the detector is trained on (after the refinement where it is used; training split) and of the "
              "detector (test split, mean $\\pm$ standard deviation over the seeds of each run). The last row applies "
              "the oracles and the label model of the selected configuration to the test flows directly, without "
              "detector. All rows are evaluated against ground truth of CSE-CIC-IDS2018; the first row comes from the "
              "oracle experiment of the five-oracle configuration." + first_note,
              ">{\\raggedright\\arraybackslash}p{0.40\\linewidth}rrrrr",
              [["", "\\multicolumn{2}{c}{Training labels}", "\\multicolumn{3}{c}{Test split}"],
                                          ["Configuration", "Precision", "Recall", "Precision", "Recall", "F1"]],
              size=r"\footnotesize")

    def lab(r):
        pl = (r.get("pseudo_labels") or {}).get("train_after_refinement") or {}
        return f(pl.get("precision")), f(pl.get("recall_all_malicious"))

    # the method of the first version (equal-vote majority of the five oracles, then refinement), rebuilt: the row of
    # the oracle experiment of the five-oracle configuration (its ablation seeds)
    ref_or = (R.get("_ref") or {}).get("oracles") or {}
    maj = next((v for v in ref_or.get("detector_ablation", []) if v.get("variant") == "all oracles, equal-vote majority"),
               None)
    if maj and maj.get("summary"):
        pl = maj.get("pseudo_labels_train") or {}
        s = maj["summary"]
        # a precision is undefined when nothing is flagged
        pl_prec = f(pl.get("precision")) if (pl.get("tp") or 0) + (pl.get("fp") or 0) > 0 else "--"
        def _mean(k):
            v = s.get(k)
            return v.get("mean") if isinstance(v, dict) else v
        nothing_flagged = _mean("recall") == 0 and _mean("fpr") == 0      # no test flow is flagged by any seed
        det_prec = "--" if nothing_flagged else pm(s["precision"])
        t.row("Five oracles of the first version (signature and fan-out rules revised); equal-vote majority; "
              f"refinement: the method of the first version, rebuilt ({len(ref_or.get('ablation_seeds', []))} seeds)",
              pl_prec, f(pl.get("recall_all_malicious")), det_prec, pm(s["recall"]), pm(s["f1"]))
        N.num("ref.majority.f1", s["f1"]["mean"] if isinstance(s["f1"], dict) else s["f1"])
        N.num("ref.majority.pl.precision", pl.get("precision"))
        N.num("ref.majority.pl.recall", pl.get("recall_all_malicious"))
        N.int("ref.majority.pl.tp", pl.get("tp"))
        N.int("ref.majority.pl.pos", (pl.get("tp") or 0) + (pl.get("fp") or 0))
        N.int("ref.majority.pl.n", pl.get("n"))

    rows = []
    if ref:
        rows.append(("Five oracles of the first version (rules revised); label model of this revision, single start, "
                     "no merging; refinement", ref))
    frozen_label = "Seven oracles; dependent oracles merged; refinement (configuration fixed before testing)"
    if fr:
        rows.append((frozen_label, fr))
    rows.append(("Seven oracles; dependent oracles merged; no refinement (selected configuration)", main))
    for name, r in rows:
        if name.startswith("Seven oracles; dependent oracles merged; no refinement") and not fr \
                and cfg.get_path("report.frozen_work_dir"):
            t.row(frozen_label, "[pending]", "[pending]", "[pending]", "[pending]", "[pending]")
        s = r["summary"]
        t.row(name, *lab(r), pm(s["precision"]), pm(s["recall"]), pm(s["f1"]))
    pte = (main.get("pseudo_labels") or {}).get("test_after_refinement") or {}
    if pte:
        t.rule()
        t.row("Pseudo-labels of the selected configuration applied to the test flows (no detector)", "--", "--",
              f(pte.get("precision")), f(pte.get("recall_all_malicious")), f(pte.get("f1")))
    t.write(out)
    if ref:
        pl = (ref.get("pseudo_labels") or {}).get("train_after_refinement") or {}
        N.num("ref.pl.train_used.precision", pl.get("precision"))
        N.num("ref.pl.train_used.recall", pl.get("recall_all_malicious"))
    if fr:
        pl = (fr.get("pseudo_labels") or {}).get("train_after_refinement") or {}
        N.num("ext.frozen.pl.train.precision", pl.get("precision"))
        N.num("ext.frozen.pl.train.recall", pl.get("recall_all_malicious"))
        N.summary("ext.frozen.fpr", (fr.get("summary") or {}).get("fpr"), 4)
        N.put("ext.frozen.n_seeds", len(fr.get("seeds", [])))


def b_extended(cfg, ws, R, N, out) -> None:
    """Values of the extended oracle set (docs/ORACLES_EXTENDED.md) and of the five-oracle reference.

    Only for a configuration that reads the corpus of another workspace (``corpus_work_dir``): the results of that
    workspace are the five-oracle reference (keys ``ref.*``)."""
    ref, lc_ref = R.get("_ref") or {}, R.get("_ref_label_check") or {}
    enabled = list(cfg.get_path("oracles.enabled") or [])
    N.put("ext.n_oracles", len(enabled), {5: "five", 6: "six", 7: "seven"}.get(len(enabled), str(len(enabled))))
    N.int("ext.k", len(enabled))
    from ..weak import dawid_skene_multistart
    N.int("cfg.label_model.max_iter_grouped", int(inspect.signature(dawid_skene_multistart).parameters["n_iter"].default))
    rc = (R.get("main") or {}).get("refinement_check_train")
    if rc:
        N.int("ext.refine.low_trust", rc["low_trust"])
        N.pct("ext.refine.low_trust_non_tcp_pct", rc["low_trust_non_tcp"] / max(1, rc["low_trust"]), 1)
        N.int("ext.refine.to_mal", rc["to_malicious"])
        N.int("ext.refine.to_mal_benign", rc["to_malicious_benign"])
        N.int("ext.refine.to_mal_non_tcp", rc["to_malicious_non_tcp"])
        N.num("ext.refine.precision_before", rc["before"]["precision"])
        N.num("ext.refine.precision_after", rc["after"]["precision"])
        N.num("ext.refine.recall_after", rc["after"]["recall_all_malicious"])
        N.pct("ext.refine.benign_non_tcp_flagged", rc["benign_non_tcp_flagged_after"], 1)
    fr = R.get("_frozen") or {}
    if fr.get("main"):
        for met in ("precision", "recall", "f1"):
            N.summary(f"ext.frozen.{met}", (fr["main"].get("summary") or {}).get(met))
        if isnum(((fr["main"].get("pipeline") or {}).get("n_selected"))):
            N.int("ext.frozen.n_features", fr["main"]["pipeline"]["n_selected"])
    ff = R.get("_frozen_first") or {}
    if ff.get("runs"):           # first run of the frozen configuration (log of another environment; same seeds)
        from ..metrics import seed_summary
        for met in ("precision", "recall", "f1"):
            N.summary(f"ext.frozen_first.{met}", seed_summary([x[met] for x in ff["runs"]]))
        N.put("ext.frozen_first.n_seeds", len(ff["runs"]))
        if isnum(ff.get("n_selected")):
            N.int("ext.frozen_first.n_features", ff["n_selected"])
        env_ = (ff.get("search") or {}).get("environment") or {}
        if env_:                                   # the machine of the first run (not the machine of the results)
            N.put("ext.frozen_first.env.cpu", env_.get("cpu_model", ""), tex(env_.get("cpu_model") or "not recorded"))
            N.int("ext.frozen_first.env.cpus", env_.get("logical_cpus"))
            N.num("ext.frozen_first.env.memory_gb", env_.get("memory_gb"), 1)
        rf_ = ff.get("refine") or {}
        if rf_:
            N.int("ext.frozen_first.refine.to_mal", rf_.get("relabelled_to_malicious_train"))
            N.int("ext.frozen_first.refine.to_mal_benign", rf_.get("relabelled_to_malicious_train_benign"))
            N.num("ext.frozen_first.refine.precision_after", (rf_.get("train_after_refinement") or {}).get("precision"))
            N.num("ext.frozen_first.refine.recall_after", (rf_.get("train_after_refinement") or {}).get("recall"))
        bp = (ff.get("search") or {}).get("best_params") or {}
        bq = ((fr.get("search") or {}).get("best_params")) or {}
        if bp and bq:            # does the search of the re-run select the same configuration?
            diff = sorted(k for k in set(bp) | set(bq) if bp.get(k) != bq.get(k))
            N.put("ext.frozen_first.search_same", "yes" if not diff else "no")
            cw = lambda v: "none" if v is None else tex(v)
            N.put("ext.frozen_first.class_weight", cw(bp.get("class_weight")))
            N.put("ext.frozen.class_weight", cw(bq.get("class_weight")))
    if True:                                    # settings of the two further oracles (also when they are not used)
        q = float(cfg.get_path("oracles.rate.quantile", 0.999))
        N.put("cfg.rate.quantile", q, f"{q:g}")
        N.put("cfg.rate.quantile_pct", 100 * q, f"{100 * q:g}")
        g = float(cfg.get_path("oracles.rate.guard", 0.05))
        N.put("cfg.rate.guard_pct", 100 * g, f"{100 * g:g}")
    if True:
        N.int("cfg.beacon.min_conn", int(cfg.get_path("oracles.beacon.min_conn", 10)))
        d = float(cfg.get_path("oracles.beacon.disp_max", 0.1))
        N.put("cfg.beacon.disp_max", d, f"{d:g}")
        N.put("cfg.beacon.disp_benign", 2 * d, f"{2 * d:g}")
        N.put("cfg.beacon.iat_min", float(cfg.get_path("oracles.beacon.iat_min", 1.0)),
              f"{float(cfg.get_path('oracles.beacon.iat_min', 1.0)):g}")
    gj = float(cfg.get_path("oracles.group_jaccard", 0.5))
    N.put("cfg.group_jaccard", gj, f"{gj:g}")
    main = R.get("main") or {}
    od = (main.get("pipeline") or {}).get("oracle_describe") or {}
    if "rate" in od and od["rate"].get("threshold") is not None:
        N.num("rate.threshold", float(od["rate"]["threshold"]), 1)
    lc = R.get("_label_check") or {}
    m = ((lc.get("aggregation_train") or {}).get("ds") or {}).get("model") or {}
    jac = m.get("jaccard") or {}
    if jac:
        st = jac.get("stat|transfer")
        if st is not None:
            N.num("ext.jaccard.stat_transfer", st)
        others = [v for k, v in jac.items() if k != "stat|transfer"]
        if others:
            N.num("ext.jaccard.max_other", max(others))
    if m.get("sources"):
        N.put("ext.sources", len(m["sources"]), join_words([_source_name(x, ORACLE_NOUN) for x in m["sources"]]))
    for key in ("ds", "ds_ungrouped", "majority"):
        r = (lc.get("aggregation_train") or {}).get(key)
        if r:
            N.num(f"lc.{key}.precision", r["precision"])
            N.num(f"lc.{key}.recall", r["recall_all_malicious"])
    # label model without merging, fitted with the EM starts and limit of the merged model
    ms = lc.get("ds_ungrouped_multistart") or {}
    if ms:
        N.num("lc.ds_ungrouped_ms.precision", ms.get("precision"))
        N.num("lc.ds_ungrouped_ms.recall", ms.get("recall_all_malicious"))
        N.put("lc.ds_ungrouped_ms.same_labels", "yes" if ms.get("labels_equal_ds_ungrouped_train") else "no")
        N.int("lc.ds_ungrouped_ms.label_changes", ms.get("label_changes_vs_ds_ungrouped_train"))
    # the five oracles of the first version under the same merging rule
    g5 = lc.get("ds_grouped_first_version_oracles") or {}
    if g5:
        N.num("lc.first_grouped.precision", g5.get("precision"))
        N.num("lc.first_grouped.recall", g5.get("recall_all_malicious"))
        N.num("lc.first_grouped.f1", g5.get("f1"))
        if isnum(g5.get("tp")) and isnum(g5.get("fp")):        # how many training flows the merged model labels malicious
            N.int("lc.first_grouped.n_malicious", g5["tp"] + g5["fp"])
            N.pct("lc.first_grouped.malicious_pct", g5.get("pseudo_malicious_share"), 2)
        N.put("lc.first_grouped.n_sources", len(g5.get("sources") or []))
    # the label model against the repeated-connection oracle (training flows)
    vr = lc.get("label_model_vs_rate_train") or {}
    if vr:
        N.int("lc.vs_rate.malicious_flows", vr["malicious_flows"])
        N.int("lc.vs_rate.malicious_differ", vr["malicious_flows"] - vr["malicious_flows_same_decision"])
        N.pct("lc.vs_rate.malicious_same_pct", vr["malicious_flows_same_decision"] / max(1, vr["malicious_flows"]), 1)
        N.int("lc.vs_rate.extra", vr["labelled_not_rate"])
        N.int("lc.vs_rate.extra_benign", vr["labelled_not_rate_benign"])
        N.int("lc.vs_rate.extra_rate_abstains", vr["labelled_where_rate_abstains"])
        # flows labelled malicious although the repeated-connection oracle votes benign on them
        N.int("lc.vs_rate.extra_rate_benign", vr["labelled_not_rate"] - vr["labelled_where_rate_abstains"])
        ra = vr.get("rate_alone") or {}
        N.num("lc.rate_alone.precision", ra.get("precision"))
        N.num("lc.rate_alone.recall", ra.get("recall_all_malicious"))
        N.num("lc.rate_alone.f1", ra.get("f1"))
    # five-oracle reference
    if ref.get("main"):
        for met in ("precision", "recall", "f1", "accuracy", "auc_roc", "pr_auc", "macro_recall"):
            N.summary(f"ref.main.{met}", (ref["main"].get("summary") or {}).get(met))
        fam_ref = ((ref["main"].get("representative") or {}).get("families")) or []
        for v in (fam_ref if isinstance(fam_ref, list) else []):
            if isinstance(v, dict) and isnum(v.get("recall")):
                N.num(f"ref.family.{v['family']}.recall", v["recall"])
        pl = (ref["main"].get("pseudo_labels") or {}).get("train_before_refinement") or {}
        if pl:
            N.num("ref.pl.train.precision", pl["precision"])
            N.num("ref.pl.train.recall", pl["recall_all_malicious"])
    if lc_ref:
        it = lc_ref.get("ds_iteration_check") or {}
        if it:
            N.int("ref.lc.iter.limit", it["limit_default"])
            N.int("ref.lc.iter.used", it["iterations_default"])
            if "precision_10x_train" in it:
                N.num("ref.lc.iter.precision", it["precision_10x_train"])
                N.num("ref.lc.iter.recall", it["recall_10x_train"])
        rec = (lc_ref.get("pseudo_label_recall_by_attack_train") or {}).get("ds") or {}
        att = lc_ref.get("attacks_train") or {}
        tot = sum(v["flows"] for v in att.values())
        unc = [a for a, v in rec.items() if v == 0.0 and a in att]
        if tot:
            N.pct("ext.uncovered_share", sum(att[a]["flows"] for a in unc) / tot, 1)
            N.put("ext.uncovered_attacks", len(unc), join_words([ATTACK_TEXT.get(a, a) for a in unc]))


BUILDERS = [b_setup, b_manifest, b_corpus, b_source, b_pl_attacks, b_oracles, b_search, b_main, b_ablations, b_fs, b_loao, b_protocols,
            b_baselines, b_adversarial, b_interpret, b_online, b_qos, b_cross, b_second, b_runtime, b_memory, b_supplementary,
            b_extended, b_main_steps, b_diagnostics]
RESULT_NAMES = ["search", "main", "oracles", "ablations", "fs_compare", "loao", "protocols", "baselines", "adversarial",
                "interpret", "online", "qos", "cross_dataset"]


def _abs_path(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def build_report(cfg, out: str | Path) -> dict:
    ws = workspace(cfg)
    out = Path(out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    for sub in ("tables", "figures", "supplementary"):
        ensure_dir(out / sub)
        for stale in (out / sub).iterdir():  # a file of an earlier report must never survive into this one
            if stale.is_file():
                stale.unlink()
    for stale in ("numbers.tex", "numbers.json", "report_manifest.json"):
        (out / stale).unlink(missing_ok=True)
    if (out / "supplementary" / "tables").is_dir():          # per-table copies and their index of an earlier report
        shutil.rmtree(out / "supplementary" / "tables")
    REF_NUMBERS.clear()                                      # label numbers of the last compiled manuscript, if any
    aux = _abs_path(cfg.get_path("report.aux_path") or str(REPO_ROOT / "paper" / "build" / "manuscript_clean.aux"))
    if aux.exists():
        for m in re.finditer(r"\\newlabel\{([^{}]+)\}\{\{([^{}]*)\}", aux.read_text(encoding="utf-8", errors="replace")):
            REF_NUMBERS[m.group(1)] = m.group(2)
    R, manifest = {}, {}
    for n in RESULT_NAMES:
        p = ws.result_path(n)
        if p.exists():
            R[n] = read_json(p)
            manifest[f"results/{n}.json"] = sha256_file(p)
    for extra in ("corpus/corpus_info.json", "corpus/stream_info.json", "data/manifest.csv", "extract/extract_info.json",
                  "extract/capture_checks.csv", "transfer/transfer_info.json", "calibration/label_check.json"):
        base = (ws.capture_root if extra.startswith(("data/", "extract/"))
                else ws.corpus_dir.parent if extra.startswith("corpus/") else ws.root)
        p = base / extra
        if extra.startswith("transfer/") and not p.exists() and cfg.get_path("transfer.model_path"):
            p = _abs_path(cfg.get_path("transfer.model_path")).parent / "transfer_info.json"
        if p.exists():
            manifest[extra] = sha256_file(p)
    from ..stages import _abs
    if cfg.get_path("transfer.work_dir"):
        src_root = _abs(cfg.get_path("transfer.work_dir"))
        for extra in ("corpus/corpus_info.json", "corpus/stream_info.json"):   # second dataset (CICIDS2017)
            p = src_root / extra
            if p.exists():
                manifest[f"second_dataset/{extra}"] = sha256_file(p)
    p = ws.result_path("trust")
    if p.exists():       # the trust table recomputed with the corrected binning (code version 25); see b_oracles
        R["trust"] = read_json(p)
        manifest["results/trust.json"] = sha256_file(p)
    lcp = ws.root / "calibration" / "label_check.json"
    if lcp.exists():
        R["_label_check"] = read_json(lcp)
    if cfg.get("corpus_work_dir"):          # the five-oracle reference of an extended configuration
        ref_root = _abs_path(cfg.get("corpus_work_dir"))
        R["_ref"] = {n: read_json(ref_root / "results" / f"{n}.json") for n in RESULT_NAMES
                     if (ref_root / "results" / f"{n}.json").exists()}
        for n in R["_ref"]:
            manifest[f"five_oracle_reference/results/{n}.json"] = sha256_file(ref_root / "results" / f"{n}.json")
        if (ref_root / "calibration" / "label_check.json").exists():
            R["_ref_label_check"] = read_json(ref_root / "calibration" / "label_check.json")
            manifest["five_oracle_reference/calibration/label_check.json"] = sha256_file(
                ref_root / "calibration" / "label_check.json")
    if cfg.get_path("report.frozen_work_dir"):   # the first frozen extended configuration (with refinement)
        fr_root = _abs_path(cfg.get_path("report.frozen_work_dir"))
        R["_frozen"] = {n: read_json(fr_root / "results" / f"{n}.json") for n in ("search", "main")
                        if (fr_root / "results" / f"{n}.json").exists()}
        for n in R["_frozen"]:
            manifest[f"frozen_configuration/results/{n}.json"] = sha256_file(fr_root / "results" / f"{n}.json")
        # the first run of that configuration, after which the refinement was removed; it ran in another environment
        # and only its log and search result were kept (docs/runs/frozen_with_refinement, docs/ORACLES_EXTENDED.md)
        first = REPO_ROOT / "docs" / "runs" / "frozen_with_refinement"
        if (first / "run_search_main.log").exists():
            seeds = [dict(seed=int(m.group(1)), precision=float(m.group(2)), recall=float(m.group(3)), f1=float(m.group(4)))
                     for m in re.finditer(r"seed (\d+): P ([0-9.]+) R ([0-9.]+) F1 ([0-9.]+)",
                                          (first / "run_search_main.log").read_text(encoding="utf-8"))]
            R["_frozen_first"] = {"runs": seeds}
            manifest["docs/runs/frozen_with_refinement/run_search_main.log"] = sha256_file(first / "run_search_main.log")
            if (first / "search.json").exists():
                R["_frozen_first"]["search"] = read_json(first / "search.json")
                manifest["docs/runs/frozen_with_refinement/search.json"] = sha256_file(first / "search.json")
            if (first / "refine_check.json").exists():      # what the refinement did to the training labels there
                R["_frozen_first"]["refine"] = read_json(first / "refine_check.json")
                manifest["docs/runs/frozen_with_refinement/refine_check.json"] = sha256_file(first / "refine_check.json")
            m_sel = re.search(r"(\d+) selected features", (first / "run_search_main.log").read_text(encoding="utf-8"))
            if m_sel:
                R["_frozen_first"]["n_selected"] = int(m_sel.group(1))
    if cfg.get_path("report.diag_dir"):          # diagnostics of the labelling rules added after the results
        dd = _abs_path(cfg.get_path("report.diag_dir"))
        R["_diag"] = {}
        for n in ("diag_oracles", "diag_spacing", "diag_rate_threshold_ids2017", "diag_selection"):
            if (dd / f"{n}.json").exists():
                R["_diag"][n] = read_json(dd / f"{n}.json")
                manifest[f"diagnostics/{n}.json"] = sha256_file(dd / f"{n}.json")
    if cfg.get_path("report.second_work_dir"):   # the method applied to CICIDS2017 without its labels
        sd_root = _abs_path(cfg.get_path("report.second_work_dir"))
        R["_second"] = {n: read_json(sd_root / "results" / f"{n}.json") for n in ("main",)
                        if (sd_root / "results" / f"{n}.json").exists()}
        if (sd_root / "calibration" / "label_check.json").exists():
            R["_second"]["label_check"] = read_json(sd_root / "calibration" / "label_check.json")
        for rel in ("results/main.json", "calibration/label_check.json"):
            if (sd_root / rel).exists():
                manifest[f"second_dataset_method/{rel}"] = sha256_file(sd_root / rel)
    if cfg.get_path("report.second_frozen_work_dir"):   # the configuration fixed before testing, applied to CICIDS2017
        sf_root = _abs_path(cfg.get_path("report.second_frozen_work_dir"))
        R["_second_frozen"] = {n: read_json(sf_root / "results" / f"{n}.json") for n in ("main",)
                               if (sf_root / "results" / f"{n}.json").exists()}
        if (sf_root / "results" / "main.json").exists():
            manifest["second_dataset_frozen/results/main.json"] = sha256_file(sf_root / "results" / "main.json")
    N = Numbers()
    status = {}
    for b in BUILDERS:
        try:
            b(cfg, ws, R, N, out)
            status[b.__name__] = "ok"
        except Exception as exc:  # a failing builder must not hide the others
            log.exception("report builder %s failed", b.__name__)
            status[b.__name__] = f"FAILED: {exc}"
    for name in TABLE_PURPOSE:     # every table of the manuscript exists, as a table or as a named placeholder
        if not (out / "tables" / f"{name}.tex").exists():
            placeholder(out, name)
    N.put("results.available", ", ".join(sorted(k for k in R if not k.startswith("_"))))
    N.put("results.missing", ", ".join(n for n in RESULT_NAMES if n not in R) or "none")
    N.put("results.failed_builders", ", ".join(k for k, v in status.items() if v != "ok") or "none")
    N.write(out)
    write_json(out / "report_manifest.json", {"work_dir": str(ws.root), "capture_dir": str(ws.capture_root), "config": cfg.get("_config_path"),
                                              "sources_sha256": manifest, "builders": status,
                                              "missing_results": [n for n in RESULT_NAMES if n not in R]})
    log.info("report written to %s (%d values, %d result files, missing: %s)", out, len(N.text), len(R),
             [n for n in RESULT_NAMES if n not in R] or "none")
    return status
