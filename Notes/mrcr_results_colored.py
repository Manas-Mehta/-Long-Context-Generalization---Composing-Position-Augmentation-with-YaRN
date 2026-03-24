"""Generate colored HTML tables of MRCR results for pasting into Google Docs."""

HTML_HEAD = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body { font-family: Arial, sans-serif; font-size: 11pt; max-width: 1800px; margin: 20px auto; }
h1 { font-size: 16pt; }
h2 { font-size: 14pt; margin-top: 30px; }
h3 { font-size: 12pt; margin-top: 20px; }
table { border-collapse: collapse; margin: 10px 0 20px 0; }
th, td { border: 1px solid #999; padding: 4px 8px; text-align: center; font-size: 10pt; }
th { background: #2c3e50; color: white; font-weight: bold; }
th.group { background: #1a252f; }
td.label { text-align: left; background: #f8f8f8 !important; font-weight: 500; }
.note { font-size: 9pt; color: #666; margin-bottom: 20px; }
.sep { border-left: 3px solid #2c3e50 !important; }
.cat-header td { background: #34495e !important; color: white; font-weight: bold; text-align: left; font-size: 10pt; }
</style></head><body>
"""

HTML_TAIL = "</body></html>"


def score_to_color(val):
    if val is None:
        return "#f8f8f8"
    v = max(0.0, min(1.0, val))
    if v < 0.5:
        r = 255
        g = int(200 * (v / 0.5))
        b = int(60 * (v / 0.5))
    else:
        t = (v - 0.5) / 0.5
        r = int(255 * (1 - t) + 76 * t)
        g = int(200 * (1 - t) + 175 * t)
        b = int(60 * (1 - t) + 80 * t)
    return f"rgb({r},{g},{b})"


def text_color(val):
    if val is None:
        return "#333"
    if val > 0.75:
        return "white"
    if val < 0.15:
        return "white"
    return "#111"


def fmt(val):
    if val is None:
        return "—"
    return f"{val:.3f}"


def make_row(cells, score_cols, sep_cols=None, is_label=None):
    if sep_cols is None:
        sep_cols = set()
    if is_label is None:
        is_label = set()
    parts = []
    for i, c in enumerate(cells):
        sep = " sep" if i in sep_cols else ""
        lbl = " label" if i in is_label else ""
        cls = f"{sep.strip()} {lbl.strip()}".strip()
        cls_attr = f' class="{cls}"' if cls else ""
        if i in score_cols and isinstance(c, (int, float)):
            bg = score_to_color(c)
            fg = text_color(c)
            parts.append(f'<td{cls_attr} style="background:{bg};color:{fg}">{fmt(c)}</td>')
        else:
            parts.append(f'<td{cls_attr}>{c if c is not None else "—"}</td>')
    return "<tr>" + "".join(parts) + "</tr>\n"


def make_table(headers, rows, score_cols=None, sep_cols=None, label_cols=None, header_groups=None):
    if score_cols is None:
        score_cols = set(range(len(headers)))
    if sep_cols is None:
        sep_cols = set()
    if label_cols is None:
        label_cols = {0, 1}
    html = ""
    if header_groups:
        html += "<table>\n<tr>"
        for text, span, is_sep in header_groups:
            cls = ' class="group sep"' if is_sep else ' class="group"'
            html += f"<th{cls} colspan=\"{span}\">{text}</th>"
        html += "</tr>\n<tr>"
    else:
        html += "<table>\n<tr>"
    for i, h in enumerate(headers):
        sep = ' class="sep"' if i in sep_cols else ""
        html += f"<th{sep}>{h}</th>"
    html += "</tr>\n"
    for row in rows:
        if row == "CAT_HEADER":
            continue
        html += make_row(row, score_cols, sep_cols, label_cols)
    html += "</table>\n"
    return html


def make_cat_table(headers, sections, score_cols, sep_cols=None, label_cols=None, header_groups=None):
    """Table with category header rows separating sections."""
    if sep_cols is None:
        sep_cols = set()
    if label_cols is None:
        label_cols = {0, 1}
    ncols = len(headers)
    html = ""
    if header_groups:
        html += "<table>\n<tr>"
        for text, span, is_sep in header_groups:
            cls = ' class="group sep"' if is_sep else ' class="group"'
            html += f"<th{cls} colspan=\"{span}\">{text}</th>"
        html += "</tr>\n<tr>"
    else:
        html += "<table>\n<tr>"
    for i, h in enumerate(headers):
        sep = ' class="sep"' if i in sep_cols else ""
        html += f"<th{sep}>{h}</th>"
    html += "</tr>\n"
    for cat_name, rows in sections:
        html += f'<tr class="cat-header"><td colspan="{ncols}">{cat_name}</td></tr>\n'
        for row in rows:
            html += make_row(row, score_cols, sep_cols, label_cols)
    html += "</table>\n"
    return html


def ood(b1, b2, b3, b4):
    return round((b1 + b2 + b3 + b4) / 4, 3)

def slight_ood(b1, b2):
    return round((b1 + b2) / 2, 3)

def very_ood(b3, b4):
    return round((b3 + b4) / 2, 3)

def avg5(b0, b1, b2, b3, b4):
    return round((b0 + b1 + b2 + b3 + b4) / 5, 3)


# ============================================================================
# ALL DATA: (name, train_yarn, pos_method, L_TL, eval_yarn, b0, b1, b2, b3, b4)
# ============================================================================
ALL_DATA = [
    # Baselines
    ("Vanilla", "—", "—", "—", "None", 0.389, 0.365, 0.465, 0.165, 0.056),
    ("YaRN inf-only", "—", "—", "—", "f=4", 0.346, 0.302, 0.319, 0.242, 0.114),
    ("LoRA baseline", "None", "—", "—", "None", 0.998, 0.966, 0.746, 0.545, 0.317),

    # RPE-only (no YaRN train)
    ("RPE fixed L=32K", "None", "RPE", "32K", "None", 0.636, 0.504, 0.503, 0.473, 0.265),
    ("RPE fixed L=32K", "None", "RPE", "32K", "f=4", 0.531, 0.499, 0.641, 0.590, 0.551),
    ("RPE fixed v2 L=32K", "None", "RPE", "32K", "None", 0.744, 0.506, 0.654, 0.446, 0.247),
    ("RPE cur L=32K", "None", "RPE", "32K", "None", 0.922, 0.691, 0.749, 0.563, 0.255),
    ("RPE cur L=32K", "None", "RPE", "32K", "f=4", 0.817, 0.592, 0.677, 0.646, 0.528),
    ("RPE cur v2 L=65K", "None", "RPE", "65K", "None", 0.891, 0.628, 0.560, 0.251, 0.169),
    ("RPE fixed L=16K", "None", "RPE", "16K", "None", 0.924, 0.537, 0.686, 0.470, 0.221),
    ("RPE fixed L=64K", "None", "RPE", "64K", "None", 0.486, 0.541, 0.592, 0.272, 0.234),
    ("RPE fixed L=128K", "None", "RPE", "128K", "None", 0.523, 0.537, 0.554, 0.430, 0.136),
    ("RPE cur L=16K", "None", "RPE", "16K", "None", 0.924, 0.939, 0.874, 0.688, 0.330),
    ("RPE cur L=64K", "None", "RPE", "64K", "None", 0.778, 0.528, 0.611, 0.341, 0.273),
    ("RPE cur L=128K", "None", "RPE", "128K", "None", 0.889, 0.573, 0.678, 0.606, 0.334),
    ("RPE cur L=128K", "None", "RPE", "128K", "f=4", 0.783, 0.599, 0.731, 0.585, 0.537),

    # PoSE-only (no YaRN train)
    ("PoSE fixed TL=32K", "None", "PoSE", "32K", "None", 0.961, 0.876, 0.937, 0.602, 0.274),
    ("PoSE cur TL=32K", "None", "PoSE", "32K", "None", 0.961, 0.844, 0.808, 0.474, 0.266),

    # YaRN-only
    ("Y4", "f=4", "—", "—", "f=4", 0.891, 0.692, 0.619, 0.619, 0.441),
    ("Y2", "f=2", "—", "—", "None", 0.997, 0.758, 0.714, 0.543, 0.211),
    ("Y2", "f=2", "—", "—", "f=2", 0.998, 0.782, 0.858, 0.598, 0.491),
    ("Y2", "f=2", "—", "—", "f=4", 0.998, 0.718, 0.789, 0.734, 0.556),
    ("Y3", "f=3", "—", "—", "None", 1.000, 0.725, 0.748, 0.366, 0.238),
    ("Y3", "f=3", "—", "—", "f=3", 0.962, 0.788, 0.843, 0.734, 0.527),
    ("Y3", "f=3", "—", "—", "f=4", 0.962, 0.633, 0.780, 0.654, 0.604),

    # YaRN f=2 + RPE (L=4K/8K removed — no RPE effect when L <= seq_length)
    ("Y2-Rc16", "f=2", "RPE cur", "16K", "None", 0.998, 0.815, 0.778, 0.619, 0.304),
    ("Y2-Rc16", "f=2", "RPE cur", "16K", "f=2", 0.998, 0.783, 0.936, 0.752, 0.584),
    ("Y2-Rc16", "f=2", "RPE cur", "16K", "f=4", 0.960, 0.693, 0.856, 0.833, 0.753),
    ("Y2-Rc32", "f=2", "RPE cur", "32K", "None", 0.998, 0.790, 0.649, 0.578, 0.229),
    ("Y2-Rc32", "f=2", "RPE cur", "32K", "f=2", 0.998, 0.910, 0.872, 0.624, 0.490),
    ("Y2-Rc32", "f=2", "RPE cur", "32K", "f=4", 0.960, 0.719, 0.809, 0.866, 0.528),
    ("Y2-Rc64", "f=2", "RPE cur", "64K", "None", 0.688, 0.520, 0.642, 0.513, 0.084),
    ("Y2-Rc64", "f=2", "RPE cur", "64K", "f=2", 0.881, 0.685, 0.680, 0.581, 0.422),
    ("Y2-Rc64", "f=2", "RPE cur", "64K", "f=4", 0.920, 0.658, 0.741, 0.676, 0.648),
    ("Y2-Rc128", "f=2", "RPE cur", "128K", "None", 0.958, 0.634, 0.713, 0.526, 0.212),
    ("Y2-Rc128", "f=2", "RPE cur", "128K", "f=2", 0.993, 0.693, 0.618, 0.592, 0.427),
    ("Y2-Rc128", "f=2", "RPE cur", "128K", "f=4", 0.916, 0.658, 0.808, 0.641, 0.499),

    # YaRN f=3 + RPE
    ("Y3-Rc128", "f=3", "RPE cur", "128K", "None", 0.854, 0.572, 0.547, 0.558, 0.233),
    ("Y3-Rc128", "f=3", "RPE cur", "128K", "f=3", 0.963, 0.625, 0.743, 0.737, 0.493),
    ("Y3-Rc128", "f=3", "RPE cur", "128K", "f=4", 0.927, 0.594, 0.744, 0.618, 0.502),

    # YaRN f=4 + RPE (L=4K/8K removed — no RPE effect when L <= seq_length)
    ("Y4-Rc16", "f=4", "RPE cur", "16K", "None", 0.964, 0.719, 0.841, 0.714, 0.269),
    ("Y4-Rc16", "f=4", "RPE cur", "16K", "f=4", 0.965, 0.688, 0.764, 0.874, 0.744),
    ("Y4-Rc64", "f=4", "RPE cur", "64K", "None", 0.737, 0.597, 0.614, 0.517, 0.223),
    ("Y4-Rc64", "f=4", "RPE cur", "64K", "f=4", 0.564, 0.627, 0.674, 0.847, 0.568),
    ("Y4-Rc128", "f=4", "RPE cur", "128K", "None", 0.598, 0.536, 0.610, 0.484, 0.173),
    ("Y4-Rc128", "f=4", "RPE cur", "128K", "f=4", 0.569, 0.502, 0.635, 0.659, 0.498),

    # YaRN f=2 + PoSE
    ("Y2-P16", "f=2", "PoSE", "16K", "None", 0.998, 0.750, 0.744, 0.481, 0.247),
    ("Y2-P16", "f=2", "PoSE", "16K", "f=2", 0.998, 0.755, 0.936, 0.669, 0.525),
    ("Y2-P16", "f=2", "PoSE", "16K", "f=4", 0.961, 0.632, 0.821, 0.662, 0.493),
    ("Y2-P32", "f=2", "PoSE", "32K", "None", 0.998, 0.841, 0.840, 0.654, 0.248),
    ("Y2-P32", "f=2", "PoSE", "32K", "f=2", 0.998, 0.789, 0.873, 0.707, 0.562),
    ("Y2-P32", "f=2", "PoSE", "32K", "f=4", 0.998, 0.780, 0.880, 0.857, 0.777),

    # YaRN f=4 + PoSE
    ("Y4-P16", "f=4", "PoSE", "16K", "None", 0.855, 0.625, 0.715, 0.354, 0.232),
    ("Y4-P16", "f=4", "PoSE", "16K", "f=4", 0.815, 0.572, 0.497, 0.624, 0.523),
    ("Y4-P32", "f=4", "PoSE", "32K", "None", 0.889, 0.631, 0.745, 0.444, 0.283),
    ("Y4-P32", "f=4", "PoSE", "32K", "f=4", 0.890, 0.547, 0.703, 0.597, 0.591),
]


def get_unique_methods():
    """Get unique training methods (name + train_yarn + pos + L)."""
    seen = {}
    for name, ty, pos, ltl, ey, b0, b1, b2, b3, b4 in ALL_DATA:
        key = (name, ty, pos, ltl)
        if key not in seen:
            seen[key] = {"no_yarn": None, "best_yarn": None, "best_yarn_f": None}
        if ey == "None":
            seen[key]["no_yarn"] = (b0, b1, b2, b3, b4)
        else:
            score = b4  # rank by bin4
            if seen[key]["best_yarn"] is None or score > seen[key]["best_yarn"][4]:
                seen[key]["best_yarn"] = (b0, b1, b2, b3, b4)
                seen[key]["best_yarn_f"] = ey
    return seen


def main():
    out = HTML_HEAD
    out += "<h1>MRCR Results — Methods Summary & Leaderboard</h1>\n"
    out += '<p class="note">Color scale: <span style="background:rgb(255,0,0);color:white;padding:2px 6px">0.0</span> '
    out += '<span style="background:rgb(255,200,60);padding:2px 6px">0.5</span> '
    out += '<span style="background:rgb(76,175,80);color:white;padding:2px 6px">1.0</span>. '
    out += "Open in browser &rarr; Select all &rarr; Copy &rarr; Paste into Google Docs.</p>\n"

    # =====================================================================
    # TABLE 0: PLAIN METHODS LIST — NO SCORES
    # =====================================================================
    out += "<h1>Methods Tested — Settings Reference</h1>\n"
    out += '<p class="note">All methods use Qwen2.5-7B-Instruct, LoRA rank 16, trained on bin 0 (4K-8K). Eval YaRN shows which factors were tested at inference.</p>\n'

    # Collect unique training configs
    method_list = []
    seen_keys = set()
    for name, ty, pos, ltl, ey, *_ in ALL_DATA:
        key = (name, ty, pos, ltl)
        if key not in seen_keys:
            seen_keys.add(key)
            method_list.append(key)

    # Collect eval yarns per training config
    eval_yarns = {}
    for name, ty, pos, ltl, ey, *_ in ALL_DATA:
        key = (name, ty, pos, ltl)
        eval_yarns.setdefault(key, [])
        if ey not in eval_yarns[key]:
            eval_yarns[key].append(ey)

    m_cat_order = [
        ("Baselines", lambda k: k[0] in ("Vanilla", "YaRN inf-only", "LoRA baseline")),
        ("RPE-only (no YaRN train)", lambda k: k[1] == "None" and "RPE" in k[2]),
        ("PoSE-only (no YaRN train)", lambda k: k[1] == "None" and "PoSE" in k[2]),
        ("YaRN-only", lambda k: k[2] == "—" and k[1] not in ("—", "None")),
        ("YaRN + RPE", lambda k: k[1] not in ("—", "None") and "RPE" in k[2]),
        ("YaRN + PoSE", lambda k: k[1] not in ("—", "None") and "PoSE" in k[2]),
    ]

    m_headers = ["Method", "Train YaRN", "Pos. Method", "L / TL", "Eval YaRN tested"]
    m_sections = []
    for cat_name, predicate in m_cat_order:
        rows = []
        for key in method_list:
            if not predicate(key):
                continue
            name, ty, pos, ltl = key
            ey_list = ", ".join(eval_yarns[key])
            rows.append([name, ty, pos, ltl, ey_list])
        if rows:
            m_sections.append((cat_name, rows))

    out += make_cat_table(m_headers, m_sections, set(), set(), {0, 1, 2, 3, 4})

    # =====================================================================
    # TABLE 1: ALL METHODS WITH SETTINGS — NO YARN vs BEST YARN sub-columns
    # =====================================================================
    out += "<h1>All Methods Tested — Settings & Results</h1>\n"
    out += '<p class="note">Each row = one unique training configuration. '
    out += '"No YaRN" = eval without frequency scaling. '
    out += '"Best YaRN" = best eval-time YaRN factor for that model (by bin 4 score).</p>\n'

    methods = get_unique_methods()

    # Define category order for the methods table
    cat_order = [
        ("Baselines (No Training)", lambda k: k[2] == "—" and k[1] == "—"),
        ("LoRA Baseline", lambda k: k[0] == "LoRA baseline"),
        ("RPE-only (no YaRN train)", lambda k: k[1] == "None" and k[2] == "RPE"),
        ("PoSE-only (no YaRN train)", lambda k: k[1] == "None" and k[2] == "PoSE"),
        ("YaRN-only", lambda k: k[2] == "—" and k[1] not in ("—", "None")),
        ("YaRN + RPE (train-time YaRN)", lambda k: k[1] not in ("—", "None") and "RPE" in k[2]),
        ("YaRN + PoSE (train-time YaRN)", lambda k: k[1] not in ("—", "None") and "PoSE" in k[2]),
    ]

    headers_methods = [
        "Method", "Train YaRN", "Pos. Method", "L / TL",
        "Bin 0", "Bin 1", "Bin 2", "Bin 3", "Bin 4",
        "Bin 0", "Bin 1", "Bin 2", "Bin 3", "Bin 4", "Eval f"
    ]
    header_groups = [
        ("Training Settings", 4, False),
        ("Eval: No YaRN", 5, True),
        ("Eval: Best YaRN", 6, True),
    ]
    score_cols = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
    sep_cols = {4, 9}
    label_cols = {0, 1, 2, 3, 14}

    sections = []
    for cat_name, predicate in cat_order:
        rows = []
        for key, vals in methods.items():
            if not predicate(key):
                continue
            name, ty, pos, ltl = key
            no = vals["no_yarn"] or (None, None, None, None, None)
            by = vals["best_yarn"] or (None, None, None, None, None)
            bf = vals["best_yarn_f"] or "—"
            rows.append([name, ty, pos, ltl,
                         no[0], no[1], no[2], no[3], no[4],
                         by[0], by[1], by[2], by[3], by[4], bf])
        # Sort rows by bin4 of best yarn (descending), fallback to no-yarn bin4
        def sort_key(r):
            yarn_b4 = r[13] if isinstance(r[13], (int, float)) else -1
            no_b4 = r[8] if isinstance(r[8], (int, float)) else -1
            return max(yarn_b4, no_b4)
        rows.sort(key=sort_key, reverse=True)
        if rows:
            sections.append((cat_name, rows))

    out += make_cat_table(headers_methods, sections, score_cols, sep_cols, label_cols, header_groups)

    # =====================================================================
    # TABLE 2: CATEGORY LEADERBOARD — TOP 2 PER CATEGORY
    # =====================================================================
    out += "<h1>Category Leaderboard — Top 2 Per Category</h1>\n"
    out += '<p class="note">Ranked by best bin 4 score per unique training config. Shows both no-YaRN and best-YaRN eval.</p>\n'

    # Group unique training configs into categories
    cat_configs = {
        "Baselines": [],
        "RPE-only": [],
        "PoSE-only": [],
        "YaRN-only": [],
        "YaRN + RPE": [],
        "YaRN + PoSE": [],
    }

    for key, vals in methods.items():
        name, ty, pos, ltl = key
        no = vals["no_yarn"] or (None, None, None, None, None)
        by = vals["best_yarn"] or (None, None, None, None, None)
        bf = vals["best_yarn_f"] or "—"
        # Best bin4 across both eval conditions
        no_b4 = no[4] if no[4] is not None else -1
        by_b4 = by[4] if by[4] is not None else -1
        best_b4 = max(no_b4, by_b4)
        entry = (name, ty, pos, ltl, no, by, bf, best_b4)

        if name in ("Vanilla", "YaRN inf-only", "LoRA baseline"):
            cat_configs["Baselines"].append(entry)
        elif ty == "None" and "RPE" in pos:
            cat_configs["RPE-only"].append(entry)
        elif ty == "None" and "PoSE" in pos:
            cat_configs["PoSE-only"].append(entry)
        elif pos == "—" and ty not in ("—", "None"):
            cat_configs["YaRN-only"].append(entry)
        elif ty not in ("—", "None") and "RPE" in pos:
            cat_configs["YaRN + RPE"].append(entry)
        elif ty not in ("—", "None") and "PoSE" in pos:
            cat_configs["YaRN + PoSE"].append(entry)

    lb_headers = [
        "#", "Method", "Train YaRN", "Pos", "L/TL",
        "Bin 0", "Bin 1", "Bin 2", "Bin 3", "Bin 4",
        "Bin 0", "Bin 1", "Bin 2", "Bin 3", "Bin 4", "f"
    ]
    lb_score_cols = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    lb_sep_cols = {5, 10}
    lb_label_cols = {0, 1, 2, 3, 4, 15}

    lb_header_groups = [
        ("", 5, False),
        ("Eval: No YaRN", 5, True),
        ("Eval: Best YaRN", 6, True),
    ]

    cat_order_lb = ["Baselines", "RPE-only", "PoSE-only", "YaRN-only", "YaRN + RPE", "YaRN + PoSE"]
    lb_sections = []
    for cat in cat_order_lb:
        entries = cat_configs[cat]
        entries.sort(key=lambda x: -x[7])  # sort by best_b4
        top = entries[:2]
        rows = []
        for i, (nm, ty, pos, ltl, no, by, bf, _) in enumerate(top, 1):
            rows.append([i, nm, ty, pos, ltl,
                         no[0], no[1], no[2], no[3], no[4],
                         by[0], by[1], by[2], by[3], by[4], bf])
        lb_sections.append((cat, rows))

    out += make_cat_table(lb_headers, lb_sections, lb_score_cols, lb_sep_cols, lb_label_cols, lb_header_groups)

    # =====================================================================
    # TABLE 3: OVERALL TOP 10 LEADERBOARD (bin 4) — unique training configs
    # =====================================================================
    out += "<h1>Overall Top 10 — Ranked by Bin 4</h1>\n"
    out += '<p class="note">Unique training configs, ranked by best bin 4 across all eval conditions.</p>\n'

    all_configs = []
    for key, vals in methods.items():
        name, ty, pos, ltl = key
        no = vals["no_yarn"] or (None, None, None, None, None)
        by = vals["best_yarn"] or (None, None, None, None, None)
        bf = vals["best_yarn_f"] or "—"
        no_b4 = no[4] if no[4] is not None else -1
        by_b4 = by[4] if by[4] is not None else -1
        best_b4 = max(no_b4, by_b4)
        all_configs.append((name, ty, pos, ltl, no, by, bf, best_b4))

    all_configs.sort(key=lambda x: -x[7])
    top10_rows = []
    for i, (nm, ty, pos, ltl, no, by, bf, _) in enumerate(all_configs[:10], 1):
        top10_rows.append([i, nm, ty, pos, ltl,
                           no[0], no[1], no[2], no[3], no[4],
                           by[0], by[1], by[2], by[3], by[4], bf])

    out += make_table(lb_headers, top10_rows, lb_score_cols, lb_sep_cols, lb_label_cols, lb_header_groups)

    # =====================================================================
    # TABLE 4: OVERALL TOP 10 — Ranked by Avg
    # =====================================================================
    out += "<h1>Overall Top 10 — Ranked by Average</h1>\n"
    out += '<p class="note">Unique training configs, ranked by best avg(bins 0-4) across all eval conditions.</p>\n'

    def best_avg(entry):
        no = entry[4]
        by = entry[5]
        no_avg = avg5(*no) if no[0] is not None else -1
        by_avg = avg5(*by) if by[0] is not None else -1
        return max(no_avg, by_avg)

    all_configs_avg = sorted(all_configs, key=lambda x: -best_avg(x))
    top10_avg_rows = []
    for i, (nm, ty, pos, ltl, no, by, bf, _) in enumerate(all_configs_avg[:10], 1):
        top10_avg_rows.append([i, nm, ty, pos, ltl,
                               no[0], no[1], no[2], no[3], no[4],
                               by[0], by[1], by[2], by[3], by[4], bf])

    out += make_table(lb_headers, top10_avg_rows, lb_score_cols, lb_sep_cols, lb_label_cols, lb_header_groups)

    out += HTML_TAIL

    outpath = "Notes/mrcr_results_colored.html"
    with open(outpath, "w") as f:
        f.write(out)
    print(f"Written to {outpath}")
    print("Open in browser -> Cmd+A -> Cmd+C -> Paste into Google Docs")


if __name__ == "__main__":
    main()
