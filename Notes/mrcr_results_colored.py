"""Generate colored HTML tables of MRCR results for pasting into Google Docs."""

HTML_HEAD = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body { font-family: Arial, sans-serif; font-size: 11pt; max-width: 1200px; margin: 20px auto; }
h1 { font-size: 16pt; }
h2 { font-size: 14pt; margin-top: 30px; }
h3 { font-size: 12pt; margin-top: 20px; }
table { border-collapse: collapse; margin: 10px 0 20px 0; }
th, td { border: 1px solid #999; padding: 4px 8px; text-align: center; font-size: 10pt; }
th { background: #2c3e50; color: white; font-weight: bold; }
td:first-child, td:nth-child(2) { text-align: left; background: #f8f8f8 !important; font-weight: 500; }
.note { font-size: 9pt; color: #666; margin-bottom: 20px; }
.sep { border-left: 3px solid #2c3e50 !important; }
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


def make_row(cells, score_cols, sep_cols=None):
    if sep_cols is None:
        sep_cols = set()
    parts = []
    for i, c in enumerate(cells):
        sep = " sep" if i in sep_cols else ""
        if i in score_cols and isinstance(c, (int, float)):
            bg = score_to_color(c)
            fg = text_color(c)
            parts.append(f'<td class="{sep.strip()}" style="background:{bg};color:{fg}">{fmt(c)}</td>')
        else:
            parts.append(f'<td class="{sep.strip()}">{c if c is not None else "—"}</td>')
    return "<tr>" + "".join(parts) + "</tr>\n"


def make_table(headers, rows, score_cols=None, sep_cols=None):
    if score_cols is None:
        score_cols = set(range(len(headers)))
    if sep_cols is None:
        sep_cols = set()
    html = "<table>\n<tr>"
    for i, h in enumerate(headers):
        sep = ' class="sep"' if i in sep_cols else ""
        html += f"<th{sep}>{h}</th>"
    html += "</tr>\n"
    for row in rows:
        html += make_row(row, score_cols, sep_cols)
    html += "</table>\n"
    return html


# Helper to compute merged scores
def ood(b1, b2, b3, b4):
    return round((b1 + b2 + b3 + b4) / 4, 3)

def slight_ood(b1, b2):
    return round((b1 + b2) / 2, 3)

def very_ood(b3, b4):
    return round((b3 + b4) / 2, 3)

def avg5(b0, b1, b2, b3, b4):
    return round((b0 + b1 + b2 + b3 + b4) / 5, 3)


def main():
    out = HTML_HEAD
    out += "<h1>MRCR Complete Results — All Phases (Colored)</h1>\n"
    out += '<p class="note">Color scale: <span style="background:rgb(255,0,0);color:white;padding:2px 6px">0.0</span> '
    out += '<span style="background:rgb(255,200,60);padding:2px 6px">0.5</span> '
    out += '<span style="background:rgb(76,175,80);color:white;padding:2px 6px">1.0</span>. '
    out += "Open in browser → Select all → Copy → Paste into Google Docs.</p>\n"

    # ===================== VIEW 1: BY METHOD FAMILY =====================
    out += "<h1>View 1: By Method Family</h1>\n"

    headers = ["Condition", "Eval YaRN", "4K-8K", "8K-16K", "16K-32K", "32K-64K", "64K-128K"]
    sc = {2, 3, 4, 5, 6}

    # 1A No training
    out += "<h2>1A. No Training (Phase 2 baselines)</h2>\n"
    out += make_table(headers, [
        ["Vanilla", "None", 0.389, 0.365, 0.465, 0.165, 0.056],
        ["YaRN inference-only", "f=4", 0.346, 0.302, 0.319, 0.242, 0.114],
    ], sc)

    # 1B LoRA only
    out += "<h2>1B. LoRA-only (Phase 3)</h2>\n"
    out += make_table(headers, [
        ["LoRA baseline", "None", 0.998, 0.966, 0.746, 0.545, 0.317],
    ], sc)

    # 1C RPE-only
    out += "<h2>1C. RPE-only — no YaRN during training</h2>\n"
    out += make_table(headers, [
        ["RPE fixed L=32K", "None", 0.636, 0.504, 0.503, 0.473, 0.265],
        ["RPE fixed L=32K", "f=4", 0.531, 0.499, 0.641, 0.590, 0.551],
        ["RPE fixed v2 L=32K", "None", 0.744, 0.506, 0.654, 0.446, 0.247],
        ["RPE cur L=32K", "None", 0.922, 0.691, 0.749, 0.563, 0.255],
        ["RPE cur L=32K", "f=4", 0.817, 0.592, 0.677, 0.646, 0.528],
        ["RPE cur v2 L=65K", "None", 0.891, 0.628, 0.560, 0.251, 0.169],
        ["RPE fixed L=16K", "None", 0.924, 0.537, 0.686, 0.470, 0.221],
        ["RPE fixed L=64K", "None", 0.486, 0.541, 0.592, 0.272, 0.234],
        ["RPE fixed L=128K", "None", 0.523, 0.537, 0.554, 0.430, 0.136],
        ["RPE cur L=16K", "None", 0.924, 0.939, 0.874, 0.688, 0.330],
        ["RPE cur L=64K", "None", 0.778, 0.528, 0.611, 0.341, 0.273],
        ["RPE cur L=128K", "None", 0.889, 0.573, 0.678, 0.606, 0.334],
        ["RPE cur L=128K", "f=4", 0.783, 0.599, 0.731, 0.585, 0.537],
        ["RPE cur L=4K (P6)", "None", 0.997, 0.783, 0.746, 0.445, 0.268],
        ["RPE cur L=4K (P6)", "f=4", 0.846, 0.682, 0.674, 0.633, 0.456],
        ["RPE cur L=8K (P6)", "None", 0.961, 0.908, 0.843, 0.558, 0.296],
        ["RPE cur L=8K (P6)", "f=4", 0.782, 0.407, 0.182, 0.035, 0.038],
    ], sc)

    # 1D PoSE only
    out += "<h2>1D. PoSE-only — no YaRN during training (Phase 4)</h2>\n"
    out += make_table(headers, [
        ["PoSE fixed TL=32K", "None", 0.961, 0.876, 0.937, 0.602, 0.274],
        ["PoSE cur TL=32K", "None", 0.961, 0.844, 0.808, 0.474, 0.266],
    ], sc)

    # 1E YaRN only
    out += "<h2>1E. YaRN-only — no RPE/PoSE</h2>\n"
    out += make_table(headers, [
        ["Y4 (train f=4)", "f=4", 0.891, 0.692, 0.619, 0.619, 0.441],
        ["Y2 (train f=2)", "None", 0.997, 0.758, 0.714, 0.543, 0.211],
        ["Y2 (train f=2)", "f=2", 0.998, 0.782, 0.858, 0.598, 0.491],
        ["Y2 (train f=2)", "f=4", 0.998, 0.718, 0.789, 0.734, 0.556],
        ["Y3 (train f=3)", "None", 1.000, 0.725, 0.748, 0.366, 0.238],
        ["Y3 (train f=3)", "f=3", 0.962, 0.788, 0.843, 0.734, 0.527],
        ["Y3 (train f=3)", "f=4", 0.962, 0.633, 0.780, 0.654, 0.604],
    ], sc)

    # 1F YaRN + RPE
    out += "<h2>1F. YaRN + RPE</h2>\n"
    out += "<h3>Train YaRN f=2 + RPE</h3>\n"
    out += make_table(headers, [
        ["Y2-Rc4", "None", 0.998, 0.667, 0.745, 0.446, 0.245],
        ["Y2-Rc4", "f=2", 0.998, 0.749, 0.891, 0.537, 0.492],
        ["Y2-Rc4", "f=4", 0.998, 0.724, 0.817, 0.658, 0.418],
        ["Y2-Rc8", "None", 0.964, 0.694, 0.873, 0.571, 0.248],
        ["Y2-Rc8", "f=2", 0.927, 0.726, 0.876, 0.715, 0.557],
        ["Y2-Rc8", "f=4", 0.890, 0.686, 0.785, 0.767, 0.570],
        ["Y2-Rc16", "None", 0.998, 0.815, 0.778, 0.619, 0.304],
        ["Y2-Rc16", "f=2", 0.998, 0.783, 0.936, 0.752, 0.584],
        ["Y2-Rc16", "f=4", 0.960, 0.693, 0.856, 0.833, 0.753],
        ["Y2-Rc32", "None", 0.998, 0.790, 0.649, 0.578, 0.229],
        ["Y2-Rc32", "f=2", 0.998, 0.910, 0.872, 0.624, 0.490],
        ["Y2-Rc32", "f=4", 0.960, 0.719, 0.809, 0.866, 0.528],
        ["Y2-Rc64", "None", 0.688, 0.520, 0.642, 0.513, 0.084],
        ["Y2-Rc64", "f=2", 0.881, 0.685, 0.680, 0.581, 0.422],
        ["Y2-Rc64", "f=4", 0.920, 0.658, 0.741, 0.676, 0.648],
        ["Y2-Rc128 (Expt 3)", "None", 0.958, 0.634, 0.713, 0.526, 0.212],
        ["Y2-Rc128 (Expt 3)", "f=2", 0.993, 0.693, 0.618, 0.592, 0.427],
        ["Y2-Rc128 (Expt 3)", "f=4", 0.916, 0.658, 0.808, 0.641, 0.499],
    ], sc)

    out += "<h3>Train YaRN f=3 + RPE</h3>\n"
    out += make_table(headers, [
        ["Y3-Rc128 (Expt 3)", "None", 0.854, 0.572, 0.547, 0.558, 0.233],
        ["Y3-Rc128 (Expt 3)", "f=3", 0.963, 0.625, 0.743, 0.737, 0.493],
        ["Y3-Rc128 (Expt 3)", "f=4", 0.927, 0.594, 0.744, 0.618, 0.502],
    ], sc)

    out += "<h3>Train YaRN f=4 + RPE</h3>\n"
    out += make_table(headers, [
        ["Y4-Rc4", "None", 0.858, 0.719, 0.777, 0.410, 0.201],
        ["Y4-Rc4", "f=4", 0.778, 0.598, 0.701, 0.528, 0.526],
        ["Y4-Rc8", "None", 0.926, 0.687, 0.745, 0.564, 0.276],
        ["Y4-Rc8", "f=4", 0.960, 0.689, 0.609, 0.541, 0.633],
        ["Y4-Rc16", "None", 0.964, 0.719, 0.841, 0.714, 0.269],
        ["Y4-Rc16", "f=4", 0.965, 0.688, 0.764, 0.874, 0.744],
        ["Y4-Rc64", "None", 0.737, 0.597, 0.614, 0.517, 0.223],
        ["Y4-Rc64", "f=4", 0.564, 0.627, 0.674, 0.847, 0.568],
        ["Y4-Rc128", "None", 0.598, 0.536, 0.610, 0.484, 0.173],
        ["Y4-Rc128", "f=4", 0.569, 0.502, 0.635, 0.659, 0.498],
    ], sc)

    # 1G YaRN + PoSE
    out += "<h2>1G. YaRN + PoSE</h2>\n"
    out += "<h3>Train YaRN f=2 + PoSE</h3>\n"
    out += make_table(headers, [
        ["Y2-P16", "None", 0.998, 0.750, 0.744, 0.481, 0.247],
        ["Y2-P16", "f=2", 0.998, 0.755, 0.936, 0.669, 0.525],
        ["Y2-P16", "f=4", 0.961, 0.632, 0.821, 0.662, 0.493],
        ["Y2-P32", "None", 0.998, 0.841, 0.840, 0.654, 0.248],
        ["Y2-P32", "f=2", 0.998, 0.789, 0.873, 0.707, 0.562],
        ["Y2-P32", "f=4", 0.998, 0.780, 0.880, 0.857, 0.777],
    ], sc)

    out += "<h3>Train YaRN f=4 + PoSE</h3>\n"
    out += make_table(headers, [
        ["Y4-P16", "None", 0.855, 0.625, 0.715, 0.354, 0.232],
        ["Y4-P16", "f=4", 0.815, 0.572, 0.497, 0.624, 0.523],
        ["Y4-P32", "None", 0.889, 0.631, 0.745, 0.444, 0.283],
        ["Y4-P32", "f=4", 0.890, 0.547, 0.703, 0.597, 0.591],
    ], sc)

    # ===================== VIEW 2: BY EVAL YARN SETTING =====================
    out += "<h1>View 2: By Eval YaRN Setting</h1>\n"

    headers2 = ["Condition", "Train YaRN", "Method", "L/TL", "4K-8K", "8K-16K", "16K-32K", "32K-64K", "64K-128K"]
    sc2 = {4, 5, 6, 7, 8}

    out += "<h2>2A. Eval with YaRN f=4</h2>\n"
    out += make_table(headers2, [
        ["YaRN inf-only", "—", "—", "—", 0.346, 0.302, 0.319, 0.242, 0.114],
        ["Y4 (Pure-Y4)", "f=4", "—", "—", 0.891, 0.692, 0.619, 0.619, 0.441],
        ["Y2", "f=2", "—", "—", 0.998, 0.718, 0.789, 0.734, 0.556],
        ["Y3", "f=3", "—", "—", 0.962, 0.633, 0.780, 0.654, 0.604],
        ["RPE fixed L=32K", "None", "RPE", "32K", 0.531, 0.499, 0.641, 0.590, 0.551],
        ["RPE cur L=32K", "None", "RPE", "32K", 0.817, 0.592, 0.677, 0.646, 0.528],
        ["RPE cur L=128K", "None", "RPE", "128K", 0.783, 0.599, 0.731, 0.585, 0.537],
        ["RPE cur L=4K", "None", "RPE", "4K", 0.846, 0.682, 0.674, 0.633, 0.456],
        ["RPE cur L=8K", "None", "RPE", "8K", 0.782, 0.407, 0.182, 0.035, 0.038],
        ["Y2-Rc4", "f=2", "RPE", "4K", 0.998, 0.724, 0.817, 0.658, 0.418],
        ["Y2-Rc8", "f=2", "RPE", "8K", 0.890, 0.686, 0.785, 0.767, 0.570],
        ["Y2-Rc16", "f=2", "RPE", "16K", 0.960, 0.693, 0.856, 0.833, 0.753],
        ["Y2-Rc32", "f=2", "RPE", "32K", 0.960, 0.719, 0.809, 0.866, 0.528],
        ["Y2-Rc64", "f=2", "RPE", "64K", 0.920, 0.658, 0.741, 0.676, 0.648],
        ["Y2-Rc128", "f=2", "RPE", "128K", 0.916, 0.658, 0.808, 0.641, 0.499],
        ["Y3-Rc128", "f=3", "RPE", "128K", 0.927, 0.594, 0.744, 0.618, 0.502],
        ["Y4-Rc4", "f=4", "RPE", "4K", 0.778, 0.598, 0.701, 0.528, 0.526],
        ["Y4-Rc8", "f=4", "RPE", "8K", 0.960, 0.689, 0.609, 0.541, 0.633],
        ["Y4-Rc16", "f=4", "RPE", "16K", 0.965, 0.688, 0.764, 0.874, 0.744],
        ["Y4-Rc64", "f=4", "RPE", "64K", 0.564, 0.627, 0.674, 0.847, 0.568],
        ["Y4-Rc128", "f=4", "RPE", "128K", 0.569, 0.502, 0.635, 0.659, 0.498],
        ["Y2-P16", "f=2", "PoSE", "16K", 0.961, 0.632, 0.821, 0.662, 0.493],
        ["Y2-P32", "f=2", "PoSE", "32K", 0.998, 0.780, 0.880, 0.857, 0.777],
        ["Y4-P16", "f=4", "PoSE", "16K", 0.815, 0.572, 0.497, 0.624, 0.523],
        ["Y4-P32", "f=4", "PoSE", "32K", 0.890, 0.547, 0.703, 0.597, 0.591],
    ], sc2)

    out += "<h2>2B. No YaRN at eval</h2>\n"
    out += make_table(headers2, [
        ["Vanilla", "—", "—", "—", 0.389, 0.365, 0.465, 0.165, 0.056],
        ["LoRA baseline", "None", "—", "—", 0.998, 0.966, 0.746, 0.545, 0.317],
        ["RPE cur L=16K", "None", "RPE", "16K", 0.924, 0.939, 0.874, 0.688, 0.330],
        ["RPE cur L=32K", "None", "RPE", "32K", 0.922, 0.691, 0.749, 0.563, 0.255],
        ["RPE cur L=64K", "None", "RPE", "64K", 0.778, 0.528, 0.611, 0.341, 0.273],
        ["RPE cur L=128K", "None", "RPE", "128K", 0.889, 0.573, 0.678, 0.606, 0.334],
        ["RPE cur L=4K", "None", "RPE", "4K", 0.997, 0.783, 0.746, 0.445, 0.268],
        ["RPE cur L=8K", "None", "RPE", "8K", 0.961, 0.908, 0.843, 0.558, 0.296],
        ["RPE fixed L=16K", "None", "RPE", "16K", 0.924, 0.537, 0.686, 0.470, 0.221],
        ["RPE fixed L=32K", "None", "RPE", "32K", 0.636, 0.504, 0.503, 0.473, 0.265],
        ["RPE fixed L=64K", "None", "RPE", "64K", 0.486, 0.541, 0.592, 0.272, 0.234],
        ["RPE fixed L=128K", "None", "RPE", "128K", 0.523, 0.537, 0.554, 0.430, 0.136],
        ["PoSE fixed", "None", "PoSE", "32K", 0.961, 0.876, 0.937, 0.602, 0.274],
        ["PoSE cur", "None", "PoSE", "32K", 0.961, 0.844, 0.808, 0.474, 0.266],
        ["Y2", "f=2", "—", "—", 0.997, 0.758, 0.714, 0.543, 0.211],
        ["Y3", "f=3", "—", "—", 1.000, 0.725, 0.748, 0.366, 0.238],
        ["Y2-Rc4", "f=2", "RPE", "4K", 0.998, 0.667, 0.745, 0.446, 0.245],
        ["Y2-Rc8", "f=2", "RPE", "8K", 0.964, 0.694, 0.873, 0.571, 0.248],
        ["Y2-Rc16", "f=2", "RPE", "16K", 0.998, 0.815, 0.778, 0.619, 0.304],
        ["Y2-Rc32", "f=2", "RPE", "32K", 0.998, 0.790, 0.649, 0.578, 0.229],
        ["Y2-Rc64", "f=2", "RPE", "64K", 0.688, 0.520, 0.642, 0.513, 0.084],
        ["Y2-Rc128", "f=2", "RPE", "128K", 0.958, 0.634, 0.713, 0.526, 0.212],
        ["Y4-Rc4", "f=4", "RPE", "4K", 0.858, 0.719, 0.777, 0.410, 0.201],
        ["Y4-Rc8", "f=4", "RPE", "8K", 0.926, 0.687, 0.745, 0.564, 0.276],
        ["Y4-Rc16", "f=4", "RPE", "16K", 0.964, 0.719, 0.841, 0.714, 0.269],
        ["Y4-Rc64", "f=4", "RPE", "64K", 0.737, 0.597, 0.614, 0.517, 0.223],
        ["Y4-Rc128", "f=4", "RPE", "128K", 0.598, 0.536, 0.610, 0.484, 0.173],
        ["Y3-Rc128", "f=3", "RPE", "128K", 0.854, 0.572, 0.547, 0.558, 0.233],
        ["Y2-P16", "f=2", "PoSE", "16K", 0.998, 0.750, 0.744, 0.481, 0.247],
        ["Y2-P32", "f=2", "PoSE", "32K", 0.998, 0.841, 0.840, 0.654, 0.248],
        ["Y4-P16", "f=4", "PoSE", "16K", 0.855, 0.625, 0.715, 0.354, 0.232],
        ["Y4-P32", "f=4", "PoSE", "32K", 0.889, 0.631, 0.745, 0.444, 0.283],
    ], sc2)

    out += "<h2>2C. Eval with matching YaRN f=2</h2>\n"
    out += make_table(headers2, [
        ["Y2", "f=2", "—", "—", 0.998, 0.782, 0.858, 0.598, 0.491],
        ["Y2-Rc4", "f=2", "RPE", "4K", 0.998, 0.749, 0.891, 0.537, 0.492],
        ["Y2-Rc8", "f=2", "RPE", "8K", 0.927, 0.726, 0.876, 0.715, 0.557],
        ["Y2-Rc16", "f=2", "RPE", "16K", 0.998, 0.783, 0.936, 0.752, 0.584],
        ["Y2-Rc32", "f=2", "RPE", "32K", 0.998, 0.910, 0.872, 0.624, 0.490],
        ["Y2-Rc64", "f=2", "RPE", "64K", 0.881, 0.685, 0.680, 0.581, 0.422],
        ["Y2-Rc128", "f=2", "RPE", "128K", 0.993, 0.693, 0.618, 0.592, 0.427],
        ["Y2-P16", "f=2", "PoSE", "16K", 0.998, 0.755, 0.936, 0.669, 0.525],
        ["Y2-P32", "f=2", "PoSE", "32K", 0.998, 0.789, 0.873, 0.707, 0.562],
    ], sc2)

    # ===================== VIEW 3: LEADERBOARDS =====================
    out += "<h1>View 3: Ranked Leaderboards</h1>\n"
    out += '<p class="note">ID = bin 0 (4K-8K, training range). OOD = avg(bins 1-4). Slight OOD = avg(bins 1-2). Very OOD = avg(bins 3-4).</p>\n'

    # Leaderboard data: (name, eval_yarn, b0, b1, b2, b3, b4)
    all_conditions = [
        ("Y2-P32", "f=4", 0.998, 0.780, 0.880, 0.857, 0.777),
        ("Y2-Rc16", "f=4", 0.960, 0.693, 0.856, 0.833, 0.753),
        ("Y4-Rc16", "f=4", 0.965, 0.688, 0.764, 0.874, 0.744),
        ("Y2-Rc64", "f=4", 0.920, 0.658, 0.741, 0.676, 0.648),
        ("Y4-Rc8", "f=4", 0.960, 0.689, 0.609, 0.541, 0.633),
        ("Y3", "f=4", 0.962, 0.633, 0.780, 0.654, 0.604),
        ("Y4-P32", "f=4", 0.890, 0.547, 0.703, 0.597, 0.591),
        ("Y2-Rc16", "f=2", 0.998, 0.783, 0.936, 0.752, 0.584),
        ("Y2-Rc8", "f=4", 0.890, 0.686, 0.785, 0.767, 0.570),
        ("Y4-Rc64", "f=4", 0.564, 0.627, 0.674, 0.847, 0.568),
        ("Y2-P32", "f=2", 0.998, 0.789, 0.873, 0.707, 0.562),
        ("Y2-Rc8", "f=2", 0.927, 0.726, 0.876, 0.715, 0.557),
        ("Y2", "f=4", 0.998, 0.718, 0.789, 0.734, 0.556),
        ("RPE fixed L=32K", "f=4", 0.531, 0.499, 0.641, 0.590, 0.551),
        ("RPE cur L=128K", "f=4", 0.783, 0.599, 0.731, 0.585, 0.537),
        ("Y2-Rc32", "f=4", 0.960, 0.719, 0.809, 0.866, 0.528),
        ("RPE cur L=32K", "f=4", 0.817, 0.592, 0.677, 0.646, 0.528),
        ("Y3", "f=3", 0.962, 0.788, 0.843, 0.734, 0.527),
        ("Y4-Rc4", "f=4", 0.778, 0.598, 0.701, 0.528, 0.526),
        ("Y2-P16", "f=2", 0.998, 0.755, 0.936, 0.669, 0.525),
        ("Y4-P16", "f=4", 0.815, 0.572, 0.497, 0.624, 0.523),
        ("Y3-Rc128", "f=4", 0.927, 0.594, 0.744, 0.618, 0.502),
        ("Y2-Rc128", "f=4", 0.916, 0.658, 0.808, 0.641, 0.499),
        ("Y4-Rc128", "f=4", 0.569, 0.502, 0.635, 0.659, 0.498),
        ("Y3-Rc128", "f=3", 0.963, 0.625, 0.743, 0.737, 0.493),
        ("Y2-P16", "f=4", 0.961, 0.632, 0.821, 0.662, 0.493),
        ("Y2-Rc4", "f=2", 0.998, 0.749, 0.891, 0.537, 0.492),
        ("Y2", "f=2", 0.998, 0.782, 0.858, 0.598, 0.491),
        ("Y2-Rc32", "f=2", 0.998, 0.910, 0.872, 0.624, 0.490),
        ("RPE cur L=4K", "f=4", 0.846, 0.682, 0.674, 0.633, 0.456),
        ("Y4 (Pure-Y4)", "f=4", 0.891, 0.692, 0.619, 0.619, 0.441),
        ("Y2-Rc128", "f=2", 0.993, 0.693, 0.618, 0.592, 0.427),
        ("Y2-Rc64", "f=2", 0.881, 0.685, 0.680, 0.581, 0.422),
        ("Y2-Rc4", "f=4", 0.998, 0.724, 0.817, 0.658, 0.418),
        ("RPE cur L=16K", "None", 0.924, 0.939, 0.874, 0.688, 0.330),
        ("RPE cur L=128K", "None", 0.889, 0.573, 0.678, 0.606, 0.334),
        ("LoRA baseline", "None", 0.998, 0.966, 0.746, 0.545, 0.317),
        ("Y2-Rc16", "None", 0.998, 0.815, 0.778, 0.619, 0.304),
        ("RPE cur L=8K", "None", 0.961, 0.908, 0.843, 0.558, 0.296),
        ("Y4-P32", "None", 0.889, 0.631, 0.745, 0.444, 0.283),
        ("Y4-Rc8", "None", 0.926, 0.687, 0.745, 0.564, 0.276),
        ("PoSE fixed", "None", 0.961, 0.876, 0.937, 0.602, 0.274),
        ("RPE cur L=64K", "None", 0.778, 0.528, 0.611, 0.341, 0.273),
        ("Y4-Rc16", "None", 0.964, 0.719, 0.841, 0.714, 0.269),
        ("RPE cur L=4K", "None", 0.997, 0.783, 0.746, 0.445, 0.268),
        ("PoSE cur", "None", 0.961, 0.844, 0.808, 0.474, 0.266),
        ("RPE fixed L=32K", "None", 0.636, 0.504, 0.503, 0.473, 0.265),
        ("RPE cur L=32K", "None", 0.922, 0.691, 0.749, 0.563, 0.255),
        ("Y2-P32", "None", 0.998, 0.841, 0.840, 0.654, 0.248),
        ("Y2-Rc8", "None", 0.964, 0.694, 0.873, 0.571, 0.248),
        ("RPE fixed v2", "None", 0.744, 0.506, 0.654, 0.446, 0.247),
        ("Y2-P16", "None", 0.998, 0.750, 0.744, 0.481, 0.247),
        ("Y2-Rc4", "None", 0.998, 0.667, 0.745, 0.446, 0.245),
        ("Y3", "None", 1.000, 0.725, 0.748, 0.366, 0.238),
        ("RPE fixed L=64K", "None", 0.486, 0.541, 0.592, 0.272, 0.234),
        ("Y3-Rc128", "None", 0.854, 0.572, 0.547, 0.558, 0.233),
        ("Y4-P16", "None", 0.855, 0.625, 0.715, 0.354, 0.232),
        ("Y2-Rc32", "None", 0.998, 0.790, 0.649, 0.578, 0.229),
        ("Y4-Rc64", "None", 0.737, 0.597, 0.614, 0.517, 0.223),
        ("RPE fixed L=16K", "None", 0.924, 0.537, 0.686, 0.470, 0.221),
        ("Y2-Rc128", "None", 0.958, 0.634, 0.713, 0.526, 0.212),
        ("Y2", "None", 0.997, 0.758, 0.714, 0.543, 0.211),
        ("Y4-Rc4", "None", 0.858, 0.719, 0.777, 0.410, 0.201),
        ("RPE cur v2", "None", 0.891, 0.628, 0.560, 0.251, 0.169),
        ("Y4-Rc128", "None", 0.598, 0.536, 0.610, 0.484, 0.173),
        ("RPE fixed L=128K", "None", 0.523, 0.537, 0.554, 0.430, 0.136),
        ("YaRN inf-only", "f=4", 0.346, 0.302, 0.319, 0.242, 0.114),
        ("Y2-Rc64", "None", 0.688, 0.520, 0.642, 0.513, 0.084),
        ("Vanilla", "—", 0.389, 0.365, 0.465, 0.165, 0.056),
        ("RPE cur L=8K", "f=4", 0.782, 0.407, 0.182, 0.035, 0.038),
    ]

    def build_lb_row(rank, name, ey, b0, b1, b2, b3, b4):
        return [rank, name, ey, b0, b1, b2, b3, b4,
                ood(b1, b2, b3, b4), slight_ood(b1, b2), very_ood(b3, b4), avg5(b0, b1, b2, b3, b4)]

    h_lb = ["#", "Condition", "Eval YaRN",
            "Bin 0 (ID)", "Bin 1", "Bin 2", "Bin 3", "Bin 4",
            "OOD (1-4)", "Slight OOD (1-2)", "Very OOD (3-4)", "Avg (0-4)"]
    sc_lb = {3, 4, 5, 6, 7, 8, 9, 10, 11}
    sep_lb = {8}  # separator before merged columns

    # 3A: by bin 4
    out += "<h2>3A. Ranked by bin 4 (64K-128K) — top 25</h2>\n"
    sorted_b4 = sorted(all_conditions, key=lambda x: -x[6])[:25]
    rows_b4 = []
    for i, (name, ey, b0, b1, b2, b3, b4) in enumerate(sorted_b4, 1):
        rows_b4.append(build_lb_row(i, name, ey, b0, b1, b2, b3, b4))
    out += make_table(h_lb, rows_b4, sc_lb, sep_lb)

    # 3B: by avg
    out += "<h2>3B. Ranked by average score — top 25</h2>\n"
    sorted_avg = sorted(all_conditions, key=lambda x: -avg5(*x[2:]))[:25]
    rows_avg = []
    for i, (name, ey, b0, b1, b2, b3, b4) in enumerate(sorted_avg, 1):
        rows_avg.append(build_lb_row(i, name, ey, b0, b1, b2, b3, b4))
    out += make_table(h_lb, rows_avg, sc_lb, sep_lb)

    # 3C: by very OOD
    out += "<h2>3C. Ranked by Very OOD (avg bins 3-4) — top 25</h2>\n"
    sorted_vood = sorted(all_conditions, key=lambda x: -very_ood(x[5], x[6]))[:25]
    rows_vood = []
    for i, (name, ey, b0, b1, b2, b3, b4) in enumerate(sorted_vood, 1):
        rows_vood.append(build_lb_row(i, name, ey, b0, b1, b2, b3, b4))
    out += make_table(h_lb, rows_vood, sc_lb, sep_lb)

    out += HTML_TAIL

    outpath = "Notes/mrcr_results_colored.html"
    with open(outpath, "w") as f:
        f.write(out)
    print(f"Written to {outpath}")
    print("Open in browser -> Cmd+A -> Cmd+C -> Paste into Google Docs")


if __name__ == "__main__":
    main()
