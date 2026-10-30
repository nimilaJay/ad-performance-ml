"""Build the final report (report/269122F_Jayawardena_ML_Report.pdf) from the saved results.

Every number in the text is read from models/ and reports/, so the PDF stays consistent with the
code after retraining. The interpretation (Sections 5-6) was written for the current results:
after retraining on new data, re-read those sections and update any statement that no longer holds.

Usage:  python report/build_report.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, ListFlowable, ListItem,
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config, explain  # noqa: E402

FIGS = config.FIGURES_DIR
SHOTS = ROOT / "report" / "screenshots"
OUT = ROOT / "report" / "269122F_Jayawardena_ML_Report.pdf"
REPO_URL = "https://github.com/nimilaJay/ad-performance-ml"
LIVE_URL = "https://ctr-dashboard.purplefield-4d57f34d.eastasia.azurecontainerapps.io"

# ---------------------------------------------------------------- fonts and styles

FONT_DIRS = [Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"), Path("C:/Windows/Fonts")]
FONT_FILES = {
    "Body": ["Times New Roman.ttf", "times.ttf"], "Body-Bold": ["Times New Roman Bold.ttf", "timesbd.ttf"],
    "Body-Italic": ["Times New Roman Italic.ttf", "timesi.ttf"],
    "Body-BoldItalic": ["Times New Roman Bold Italic.ttf", "timesbi.ttf"],
    "Sans": ["Arial.ttf", "arial.ttf"], "Sans-Bold": ["Arial Bold.ttf", "arialbd.ttf"],
}


def register_fonts():
    found = {}
    for name, files in FONT_FILES.items():
        for d in FONT_DIRS:
            hit = next((d / f for f in files if (d / f).exists()), None)
            if hit:
                pdfmetrics.registerFont(TTFont(name, str(hit)))
                found[name] = True
                break
    if len(found) < len(FONT_FILES):
        raise SystemExit("Times New Roman / Arial TTF fonts not found; add their folder to FONT_DIRS.")
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                                  boldItalic="Body-BoldItalic")
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold")


register_fonts()
INK, GREY, RULE, HEAD_BG = colors.HexColor("#1a1a1a"), colors.HexColor("#5a5a5a"), colors.HexColor("#b8b8b8"), \
    colors.HexColor("#eeeeee")
ACCENT = colors.HexColor("#1f4e79")

S = {
    "body": ParagraphStyle("body", fontName="Body", fontSize=11, leading=15.2, alignment=TA_JUSTIFY,
                           spaceAfter=7, textColor=INK),
    "H1": ParagraphStyle("H1", fontName="Sans-Bold", fontSize=15, leading=19, spaceBefore=16, spaceAfter=8,
                         textColor=ACCENT, keepWithNext=1),
    "H2": ParagraphStyle("H2", fontName="Sans-Bold", fontSize=11.5, leading=15, spaceBefore=10, spaceAfter=5,
                         textColor=INK, keepWithNext=1),
    "caption": ParagraphStyle("caption", fontName="Body-Italic", fontSize=9.5, leading=12.5, textColor=GREY,
                              spaceBefore=4, spaceAfter=12, alignment=TA_LEFT),
    "tcaption": ParagraphStyle("tcaption", fontName="Body-Italic", fontSize=9.5, leading=12.5, textColor=GREY,
                               spaceBefore=6, spaceAfter=4, keepWithNext=1),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=9, leading=11.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=9, leading=11.5, textColor=INK),
    "ref": ParagraphStyle("ref", fontName="Body", fontSize=10, leading=13.5, leftIndent=18, firstLineIndent=-18,
                          spaceAfter=5, alignment=TA_LEFT),
    "eq": ParagraphStyle("eq", fontName="Body", fontSize=11.5, leading=16, alignment=TA_CENTER, spaceBefore=4,
                         spaceAfter=10),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.8, leading=11.5, leftIndent=12, spaceAfter=7,
                           textColor=INK),
}
# Same look as H1/H2 but kept out of the table of contents.
S["H1n"] = ParagraphStyle("H1n", parent=S["H1"])
S["H2n"] = ParagraphStyle("H2n", parent=S["H2"])

PLACEHOLDER = '<font backColor="#fff1a8">{}</font>'


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="body"):
    return ListFlowable([ListItem(P(t, style), leftIndent=14) for t in items], bulletType="bullet",
                        start="•", leftIndent=14, bulletFontName="Body", bulletFontSize=9, spaceAfter=6)


def numbered(items):
    return ListFlowable([ListItem(P(t), leftIndent=16) for t in items], bulletType="1", leftIndent=16,
                        bulletFontName="Body", bulletFontSize=11, spaceAfter=6)


def table(rows, widths, caption=None, bold_first_col=False):
    data = [[Paragraph(str(c), S["cellb"]) for c in rows[0]]]
    for r in rows[1:]:
        data.append([Paragraph(str(c), S["cellb" if (bold_first_col and i == 0) else "cell"]) for i, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    parts = [P(caption, "tcaption")] if caption else []
    return KeepTogether(parts + [t, Spacer(1, 8)])


def img(path, width):
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * h / w)


def figure(paths, caption, width=None, cols=1, gap=8):
    width = width or TEXT_W
    if cols == 1:
        body = [img(p, width) for p in paths]
    else:
        cells = [img(p, width / cols - gap) for p in paths]
        rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
        t = Table(rows, colWidths=[width / cols] * cols, hAlign="CENTER")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), gap)]))
        body = [t]
    for b in body:
        b.hAlign = "CENTER"
    return KeepTogether(body + [P(caption, "caption")])


# ---------------------------------------------------------------- page layout

PAGE_W, PAGE_H = A4
MARGIN_X, MARGIN_Y = 25 * mm, 22 * mm
TEXT_W = PAGE_W - 2 * MARGIN_X
SHORT_TITLE = "Explainable ML for Campaign Performance Prediction"


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 8)
    canvas.setFillColor(GREY)
    canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 13 * mm, SHORT_TITLE)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN_X, PAGE_H - 14.5 * mm, PAGE_W - MARGIN_X, PAGE_H - 14.5 * mm)
    canvas.drawCentredString(PAGE_W / 2, 12 * mm, str(doc.page))
    canvas.restoreState()


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X, topMargin=MARGIN_Y,
                         bottomMargin=MARGIN_Y, title="Explainable Machine Learning for Digital Marketing Campaign "
                         "Performance Prediction", author="Jayawardena J.D.N.T.", subject="MSc in Artificial Intelligence")
        frame = Frame(MARGIN_X, MARGIN_Y, TEXT_W, PAGE_H - 2 * MARGIN_Y, id="f", leftPadding=0, rightPadding=0)
        self.addPageTemplates([PageTemplate("title", [frame]), PageTemplate("body", [frame], onPage=decorate)])
        self._heading_no = 0

    def beforeDocument(self):
        # multiBuild lays the document out several times; bookmark keys must repeat on every pass.
        self._heading_no = 0

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("H1", "H2"):
            level = 0 if flowable.style.name == "H1" else 1
            text = flowable.getPlainText()
            key = f"h{self._heading_no}"
            self._heading_no += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)
            self.notify("TOCEntry", (level, text, self.page, key))


# ---------------------------------------------------------------- facts from the saved results

meta = json.loads((config.MODELS_DIR / "metadata.json").read_text())
metrics = json.loads((config.REPORTS_DIR / "metrics.json").read_text())
expl = json.loads((config.REPORTS_DIR / "explainability_summary.json").read_text())
tuning = pd.read_csv(config.REPORTS_DIR / "tuning_results.csv")
segments = pd.read_csv(config.REPORTS_DIR / "segment_performance.csv")
preds = pd.read_csv(config.REPORTS_DIR / "test_predictions.csv")
ds = pd.read_csv(config.DATA_PROCESSED / "model_dataset.csv", parse_dates=["date", "target_date"])
importance = pd.read_csv(config.REPORTS_DIR / "ebm_term_importance.csv")
ebm = joblib.load(config.MODELS_DIR / "ebm.joblib")
logreg = joblib.load(config.MODELS_DIR / "logreg.joblib")
catboost = joblib.load(config.MODELS_DIR / "catboost.joblib")

M = {m: v["test"] for m, v in metrics.items()}
V = {m: v["validation"] for m, v in metrics.items()}
EBM, LR, CAT, PER = "EBM", "Logistic Regression", "CatBoost", "Persistence baseline"


def f3(x):
    return f"{x:.3f}"


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def paired_auc_diffs(n_boot=1000, seed=0):
    """Cluster bootstrap of ROC-AUC differences (EBM minus each other model) on the test set."""
    y, g = preds[config.TARGET].to_numpy(), preds["campaign_id"].to_numpy()
    rows = [np.flatnonzero(g == c) for c in np.unique(g)]
    rng = np.random.default_rng(seed)
    out = {m: [] for m in (PER, CAT, LR)}
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        if y[idx].min() == y[idx].max():
            continue
        base = roc_auc_score(y[idx], preds[f"p_{EBM}"].to_numpy()[idx])
        for m in out:
            out[m].append(base - roc_auc_score(y[idx], preds[f"p_{m}"].to_numpy()[idx]))
    return {m: (float(np.mean(v)), *np.percentile(v, [2.5, 97.5]), float(np.mean(np.array(v) > 0)))
            for m, v in out.items()}


DIFF = paired_auc_diffs()
splits = {s: ds[ds.split == s] for s in ["train", "val", "test"]}
n_rows, n_campaigns = len(ds), ds.campaign_id.nunique()
n_test_campaigns = splits["test"].campaign_id.nunique()
campaigns_by_platform = ds.groupby("platform").campaign_id.nunique().to_dict()
log = meta["cleaning_log"]
bench = meta["benchmark_ctr_pct"]
lab = ds.assign(next_index=ds.next_ctr / ds.benchmark_ctr)
rho1 = lab[["ctr_index_1d", "next_index"]].corr(method="spearman").iloc[0, 1]
rho7 = lab[["ctr_index_7d", "next_index"]].corr(method="spearman").iloc[0, 1]
pr = meta["positive_rate"]

ebm_tune = tuning[tuning.model == EBM]
auc_by_inter = ebm_tune.groupby("param_interactions").val_roc_auc.max()
fit_secs = tuning.groupby("model").fit_seconds.mean()
best = meta["best_params"]

imp = dict(zip(importance.term, importance.importance))
n_terms = len(importance)
n_inter = int((importance.kind == "interaction").sum())
max_inter_imp = importance.loc[importance.kind == "interaction", "importance"].max()
shapes = expl["shape_summaries"]
dow = dict(zip(shapes["day_of_week"]["category"], shapes["day_of_week"]["score"]))
plat = dict(zip(shapes["platform"]["category"], shapes["platform"]["score"]))
age_shape = explain.ebm_shape(ebm, "campaign_day")
age_end_day, age_end_score = int(age_shape.x_left.iloc[-1]), age_shape.score.iloc[-1]
age_start_score = age_shape.score.iloc[0]
idx7 = explain.ebm_shape(ebm, "ctr_index_7d")
cross7 = float(idx7.loc[idx7.score >= 0, "x_left"].iloc[0])

seg_ebm = segments[segments.model == EBM].set_index("segment")
objs = seg_ebm[seg_ebm.segment_type == "objective"]

X_test = splits["test"][config.FEATURES].reset_index(drop=True)
age_grid = explain.feature_grid(X_test, "campaign_day", n=25)
pdp_age = {name: explain.partial_dependence(m, X_test, "campaign_day", age_grid)
           for name, m in [(EBM, ebm), (LR, logreg), (CAT, catboost)]}

perm = pd.read_csv(config.REPORTS_DIR / "permutation_importance.csv")
perm_lr = perm[perm.model == LR].set_index("feature").importance

cases = {c["case"]: c for c in expl["local_cases"]}
below, wrong = cases["local_confident_below"], cases["local_misclassified"]


def case_row(c):
    return ds[(ds.campaign_id == c["campaign_id"]) & (ds.target_date == pd.Timestamp(c["target_date"]))].iloc[0]


below_row, wrong_row = case_row(below), case_row(wrong)

data_note = "simulated" if meta["is_synthetic"] else "real"
if not meta["is_synthetic"]:
    print("NOTE: models were trained on real data. Sections 2.1 and 5-8 were written for the simulated "
          "dataset; revise the wording before submitting.")

# ---------------------------------------------------------------- pipeline diagram


def pipeline_diagram():
    w, h = TEXT_W, 165
    d = Drawing(w, h)
    steps = [
        ["Daily reports", "Google Ads, Meta Ads"], ["Cleaning and", "validation"],
        ["Features and", "benchmark"], ["Chronological split", "70 / 15 / 15"],
        ["Grid search,", "validation ROC-AUC"], ["Decision threshold", "(Youden's J)"],
        ["Refit on train+val,", "single test run"], ["Explain: EBM terms,", "PDP, permutation"],
    ]
    bw, bh, gap_x = (w - 3 * 22) / 4, 44, 22
    rows_y = [h - bh - 4, h - 2 * bh - 38]
    centres = []
    for i, lines in enumerate(steps):
        r, c = divmod(i, 4)
        x, y = c * (bw + gap_x), rows_y[r]
        fill = colors.HexColor("#e8f0f8") if i < 4 else colors.HexColor("#f3f3f3")
        d.add(Rect(x, y, bw, bh, rx=5, ry=5, fillColor=fill, strokeColor=colors.HexColor("#7f9cbc"),
                   strokeWidth=0.7))
        for k, line in enumerate(lines):
            d.add(String(x + bw / 2, y + bh / 2 + 3 - 11 * k, line, fontName="Sans", fontSize=8,
                         textAnchor="middle", fillColor=INK))
        centres.append((x, y))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=GREY, strokeWidth=0.9))
        if x2 > x1:
            d.add(Polygon([x2, y2, x2 - 5, y2 + 3, x2 - 5, y2 - 3], fillColor=GREY, strokeColor=GREY))
        else:
            d.add(Polygon([x2, y2, x2 - 3, y2 + 5, x2 + 3, y2 + 5], fillColor=GREY, strokeColor=GREY))

    for i in range(7):
        if i == 3:
            continue
        x, y = centres[i]
        arrow(x + bw, y + bh / 2, x + bw + gap_x, y + bh / 2)
    # wrap from step 4 (end of row 1) to step 5 (start of row 2)
    x4, y4 = centres[3]
    x5, y5 = centres[4]
    mid_y = (y4 + y5 + bh) / 2
    d.add(Line(x4 + bw / 2, y4, x4 + bw / 2, mid_y, strokeColor=GREY, strokeWidth=0.9))
    d.add(Line(x4 + bw / 2, mid_y, x5 + bw / 2, mid_y, strokeColor=GREY, strokeWidth=0.9))
    arrow(x5 + bw / 2, mid_y, x5 + bw / 2, y5 + bh)
    d.add(String(w / 2, 12, "Trained EBM, Logistic Regression and CatBoost models are served in a Streamlit dashboard",
                 fontName="Body-Italic", fontSize=8.5, textAnchor="middle", fillColor=GREY))
    return d


def mlops_diagram():
    w, h = TEXT_W, 178
    d = Drawing(w, h)
    gap_x = 16
    bw, bh = (w - 4 * gap_x) / 5, 46
    top_y, bottom_y = h - bh - 4, h - 2 * bh - 54
    x_of = [c * (bw + gap_x) for c in range(5)]
    blue, grey = colors.HexColor("#e8f0f8"), colors.HexColor("#f3f3f3")

    def box(c, y, lines, fill):
        x = x_of[c]
        d.add(Rect(x, y, bw, bh, rx=5, ry=5, fillColor=fill, strokeColor=colors.HexColor("#7f9cbc"),
                   strokeWidth=0.7))
        for k, line in enumerate(lines):
            d.add(String(x + bw / 2, y + bh / 2 + 9 - 11 * k, line, fontName="Sans-Bold" if k == 0 else "Sans",
                         fontSize=8, textAnchor="middle", fillColor=INK))

    def head(x, y, direction):
        pts = {"right": [x, y, x - 5, y + 3, x - 5, y - 3], "up": [x, y, x - 3, y - 5, x + 3, y - 5]}[direction]
        d.add(Polygon(pts, fillColor=GREY, strokeColor=GREY))

    def line(*xy):
        for (x1, y1), (x2, y2) in zip(xy, xy[1:]):
            d.add(Line(x1, y1, x2, y2, strokeColor=GREY, strokeWidth=0.9))

    top = [["Push to main", "code or merged", "new models"], ["1. Test", "20 tests and", "quality gate"],
           ["2. Build", "Docker image,", "smoke test"], ["3. Deploy", "registry, new", "app revision"],
           ["Live dashboard", "Azure Container", "Apps (HTTPS)"]]
    for c, lines in enumerate(top):
        box(c, top_y, lines, blue if c < 4 else grey)
        if c < 4:
            line((x_of[c] + bw, top_y + bh / 2), (x_of[c + 1], top_y + bh / 2))
            head(x_of[c + 1], top_y + bh / 2, "right")
    bottom = {0: ["Retrain", "Run workflow", "button"], 1: ["Train and test", "tuning, explain,", "quality gate"],
              2: ["Pull request", "metrics compared", "with production"], 4: ["Health check", "daily: does the", "app answer?"]}
    for c, lines in bottom.items():
        box(c, bottom_y, lines, grey)
    for c in (0, 1):
        line((x_of[c] + bw, bottom_y + bh / 2), (x_of[c + 1], bottom_y + bh / 2))
        head(x_of[c + 1], bottom_y + bh / 2, "right")
    # A person reviews and merges the pull request, which is a push to main.
    mid_y = (top_y + bottom_y + bh) / 2
    cx2, cx0 = x_of[2] + bw / 2, x_of[0] + bw / 2
    line((cx2, bottom_y + bh), (cx2, mid_y), (cx0, mid_y), (cx0, top_y))
    head(cx0, top_y, "up")
    d.add(String((cx0 + cx2) / 2, mid_y + 4, "review and merge", fontName="Body-Italic", fontSize=8,
                 textAnchor="middle", fillColor=GREY))
    cx4 = x_of[4] + bw / 2
    line((cx4, bottom_y + bh), (cx4, top_y))
    head(cx4, top_y, "up")
    d.add(String(w / 2, 10, "All steps run in GitHub Actions; only the initial Azure setup is run once by hand",
                 fontName="Body-Italic", fontSize=8.5, textAnchor="middle", fillColor=GREY))
    return d


# ---------------------------------------------------------------- content

def build():
    st = []
    ph = PLACEHOLDER.format

    # Title page
    st += [Spacer(1, 55 * mm),
           Paragraph("Explainable Machine Learning for Digital Marketing Campaign Performance Prediction",
                     ParagraphStyle("t", fontName="Sans-Bold", fontSize=22, leading=28, textColor=ACCENT)),
           Spacer(1, 6 * mm),
           Paragraph("Predicting whether a campaign's next-day click-through rate will beat its benchmark, "
                     "using an Explainable Boosting Machine",
                     ParagraphStyle("st", fontName="Body-Italic", fontSize=13, leading=18, textColor=GREY)),
           Spacer(1, 30 * mm)]
    info = [["Programme", "MSc in Artificial Intelligence"], ["Module", "Machine Learning (assignment report)"],
            ["Student name", "Jayawardena J.D.N.T."], ["Student ID", "269122F"],
            ["Submission date", "29 October 2026"], ["Source code", f'<a href="{REPO_URL}" color="{ACCENT.hexval()}">{REPO_URL}</a>'],
            ["Live demo", f'<a href="{LIVE_URL}" color="{ACCENT.hexval()}">{LIVE_URL}</a>']]
    t = Table([[P(a, "cellb"), P(b, "cell")] for a, b in info], colWidths=[38 * mm, 100 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    st += [t, NextPageTemplate("body"), PageBreak()]

    # Abstract + contents
    d_per = DIFF[PER]
    st += [P("Abstract", "H1n"), P(
        "Digital marketing teams review campaign performance every day and decide which campaigns need attention, "
        "but platform dashboards only describe what has already happened. This project builds a model that predicts "
        "whether a campaign's click-through rate (CTR) tomorrow will beat a benchmark for comparable campaigns, and "
        "explains each prediction. The main model is an Explainable Boosting Machine (EBM), a generalised additive "
        "model with pairwise interactions that is fitted by gradient boosting but remains fully interpretable. It is "
        "compared with logistic regression, CatBoost and a naive persistence rule on a chronologically held-out "
        "test period. Real advertising account exports were not yet available when this report was written, so the "
        f"pipeline was developed and evaluated on a {data_note} dataset of {n_rows:,} campaign-days from "
        f"{n_campaigns} campaigns; it accepts Google Ads and Meta Ads exports without code changes. On the test "
        f"period the EBM reached a ROC-AUC of {f3(M[EBM]['roc_auc'])} (95% CI {f3(M[EBM]['roc_auc_ci'][0])}–"
        f"{f3(M[EBM]['roc_auc_ci'][1])}). This was indistinguishable from CatBoost ({f3(M[CAT]['roc_auc'])}) and "
        f"significantly better than assuming tomorrow will look like today (difference {d_per[0]:+.3f}, 95% CI "
        f"{d_per[1]:.3f} to {d_per[2]:.3f}). The EBM's learned shape functions recovered the structure built into "
        "the data, including CTR persistence, creative fatigue and a weekend dip, and also exposed an effect that "
        "the model averages away. A Streamlit dashboard, deployed on Microsoft Azure through a tested GitHub Actions "
        "pipeline, makes the predictions and explanations available to non-technical users."),
        PageBreak(), P("Contents", "H1n")]
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("toc0", fontName="Body-Bold", fontSize=10.5, leading=12, spaceBefore=2, leftIndent=0),
        ParagraphStyle("toc1", fontName="Body", fontSize=10, leading=11, leftIndent=14)]
    toc.tableStyle = TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 1)])
    toc.dotsMinLevel = 0
    st += [toc, PageBreak()]

    # 1 Introduction
    st += [P("1. Introduction", "H1"), P("1.1 Background and motivation", "H2"), P(
        "Online advertising budgets are now spread across several platforms, and most of that spend is managed day "
        "by day. A marketing team running dozens of campaigns on Google Ads and Meta Ads has to decide each morning "
        "which campaigns to leave alone, which to refresh with new creative, and which to pause or rebalance. "
        "Click-through rate is one of the first signals it looks at. CTR reacts quickly, it is available for every "
        "campaign type, and a falling CTR is often the earliest visible sign that an advert has worn out or is "
        "reaching the wrong audience."), P(
        "The difficulty is that daily CTR is noisy. It moves with the day of the week, with auction competition, "
        "with budget changes and with the age of the creative. Platform dashboards report these movements after the "
        "fact but do not say whether yesterday's dip is likely to continue. A forward-looking warning would help a "
        "team prioritise its attention, but only if it comes with a reason. A score saying that a campaign is "
        "at risk is of little use unless the team can see whether the cause is a long decline in engagement, an "
        "old creative or simply that tomorrow is a Saturday. Most published work on click prediction targets "
        "individual ad impressions at very large scale (Richardson, Dominowska and Ragno, 2007; McMahan et al., "
        "2013). This project works one level up, at the level of the campaign-day, where the user is a marketer "
        "rather than an ad server."),
        P("1.2 Problem statement", "H2"), P(
        "The task is framed as binary classification. For each campaign at the end of day <i>t</i>, using only data "
        "available at that point, the model estimates the probability that the campaign's CTR on day <i>t</i>+1 "
        "will exceed a benchmark for campaigns of the same platform and objective. A positive prediction means the "
        "campaign is expected to beat the benchmark; a negative prediction flags it as at risk of underperforming. "
        "The output is meant as decision support. Its explanations describe statistical associations in historical "
        "data, not causal effects."),
        P("1.3 Objectives", "H2"), numbered([
            "Assemble a campaign-day dataset containing the features listed in the project brief, and define a fair "
            "target that leaks no future information.",
            "Identify the factors associated with next-day CTR through exploratory analysis.",
            "Train an Explainable Boosting Machine and compare it with logistic regression, CatBoost and a "
            "persistence baseline.",
            "Evaluate the models with precision, recall, F1-score and ROC-AUC on a chronologically held-out test "
            "period, with confidence intervals.",
            "Explain the EBM globally and for individual predictions, and check its behaviour against domain "
            "knowledge.",
            "Build an interactive dashboard in which marketing users can enter campaign data and see predictions "
            "and explanations."]),
        P("Section 2 describes the dataset and its preparation, Section 3 the choice of algorithm and Section 4 the "
          "method. Results are reported in Section 5 and interpreted in Section 6. Section 7 presents the dashboard, "
          "Section 8 discusses limitations and ethical issues, and Section 9 concludes.")]

    # 2 Dataset
    st += [P("2. Dataset", "H1"), P("2.1 Data source", "H2"), P(
        "The project brief planned to use daily reports exported from Google Ads, Meta Ads or authorised local "
        "campaign reports. Access to such an account had not yet been secured when this report was written, so the "
        "data used here is simulated. It was generated by <font face='Courier' size='9.5'>src/generate_synthetic.py"
        "</font> and is labelled as synthetic throughout the code and in the dashboard. The results in Sections 5 "
        "and 6 therefore show that the method works end to end and behaves sensibly; they are not evidence about "
        "the performance of real campaigns. The pipeline was written around the column layout of real platform "
        "exports, and Section 2.6 explains how real data replaces the simulated file without changing any code."),
        P(f"The simulation was designed to reproduce patterns that are widely discussed in digital advertising, not "
          f"to make the problem easy. It contains {n_campaigns} campaigns running between "
          f"{pd.Timestamp(meta['date_range'][0]):%d %B %Y} and {pd.Timestamp(meta['date_range'][1]):%d %B %Y}, "
          f"{campaigns_by_platform.get('Google Ads', 0)} on Google Ads and "
          f"{campaigns_by_platform.get('Meta Ads', 0)} on Meta Ads, each with one of five objectives (Awareness, "
          "Traffic, Engagement, Leads and Sales). Every campaign has a typical CTR and cost per thousand impressions "
          "(CPM) for its platform and objective, multiplied by a campaign-specific quality factor. On top of this "
          "the generator adds:"),
        bullets([
            "creative fatigue, in which CTR decays exponentially with the number of days since the last creative "
            "refresh, faster on Meta than on Google, with occasional random refreshes;",
            "a short learning phase in which new campaigns deliver weaker and more volatile CTR;",
            "budget changes, where spending well above a campaign's usual level raises CPM and lowers CTR as "
            "delivery reaches broader audiences;",
            "day-of-week effects, with Google campaigns stronger on weekdays and Meta campaigns stronger at weekends;",
            "persistent day-to-day drift (a first-order autoregressive process) plus a one-day random shock;",
            "clicks drawn from a binomial distribution given impressions and CTR, occasional paused days, and a small "
            "number of duplicated rows of the kind produced by overlapping exports."]),
        P("Because the data-generating process is known, the simulated data has one genuine advantage: it allows a "
          "check of whether the explanations in Section 6 recover real structure rather than artefacts of the model."),
        P("2.2 Unit of analysis and target variable", "H2"), P(
        "Each row describes one campaign at the end of one day (<i>t</i>) and is built only from data up to and "
        "including that day. The target, <i>above_benchmark_next_day</i>, equals 1 when the campaign's CTR on day "
        "<i>t</i>+1 is higher than its benchmark and 0 otherwise. Two choices shape this definition."), P(
        "First, the benchmark is set per platform and objective. CTR levels differ by almost an order of magnitude "
        "between, for example, Google lead-generation campaigns and Meta awareness campaigns (Table 1), so a single "
        "benchmark would mostly measure campaign type rather than performance. Each benchmark is the median daily "
        "CTR of all campaign-days in that group during the training period only. Fixing it before any validation or "
        "test data is seen keeps future information out of the labels, and using the median gives roughly balanced "
        "classes in training."), P(
        f"Second, a day is only labelled if day <i>t</i>+1 has at least {meta['min_impressions']} impressions. Below "
        "that, a handful of clicks can move CTR by several percentage points and the label would mostly reflect "
        "noise.")]
    groups = sorted(bench)
    bench_rows = [["Platform", "Awareness", "Engagement", "Leads", "Sales", "Traffic"]]
    for platform in sorted({g.split(" | ")[0] for g in groups}):
        bench_rows.append([platform] + [f"{bench.get(f'{platform} | {o}', float('nan')):.2f}%" for o in
                                        ["Awareness", "Engagement", "Leads", "Sales", "Traffic"]])
    st.append(table(bench_rows, [32 * mm] + [24 * mm] * 5,
                    "Table 1. Benchmark CTR per platform and objective (median daily CTR in the training period).",
                    bold_first_col=True))

    st += [P("2.3 Features", "H2"), P(
        f"Table 2 lists the {len(config.FEATURES)} features. They cover the factors named in the project brief: "
        "platform, objective, campaign duration, historical spend, previous impressions and clicks, and previous "
        "CTR. CTR history is expressed as a <i>CTR index</i>, the campaign's CTR divided by its benchmark, so that "
        "a value of 1.0 means \"at benchmark\" for any campaign type. This puts Google and Meta campaigns on a "
        "common scale and makes the learned effects easier to read. The index is computed over the last day, the "
        "last three days and the last seven days, together with a trend term. Rolling windows are measured in "
        "calendar days, so a paused day counts as a day without delivery rather than being skipped. The day of week "
        "refers to the day being predicted, which is known in advance.")]
    feat_rows = [["Group", "Feature", "Definition"],
                 ["Context", "Advertising platform", "Google Ads or Meta Ads"],
                 ["", "Campaign objective", "Awareness, Traffic, Engagement, Leads or Sales"],
                 ["", "Day of week", "Day of the week being predicted (t+1)"],
                 ["", "Campaign age", "Days since the campaign's first day (campaign duration)"],
                 ["Spend", "Spend, last day", "Spend on day t"],
                 ["", "Average daily spend, 7 days", "Mean daily spend over days t−6 to t"],
                 ["", "Cumulative spend", "Total spend since the campaign started"],
                 ["", "CPM, 7 days", "Cost per 1,000 impressions over the last 7 days"],
                 ["Delivery", "Impressions and clicks, last day", "Impressions and clicks on day t"],
                 ["", "Average daily impressions and clicks, 7 days", "Means over days t−6 to t"],
                 ["CTR history", "CTR index, 1, 3 and 7 days", "CTR over the window divided by the benchmark"],
                 ["", "CTR trend", "One-day index minus seven-day index"]]
    st.append(table(feat_rows, [26 * mm, 58 * mm, TEXT_W - 84 * mm],
                    "Table 2. Model features. Impressions and clicks each contribute two features, giving "
                    f"{len(config.FEATURES)} in total."))

    model_rows = n_rows
    st += [P("2.4 Preprocessing", "H2"), P(
        "Cleaning (<font face='Courier' size='9.5'>src/data.py</font>) parses dates and numbers, removes total rows "
        "from exports, drops impossible records (negative values, or more clicks than impressions), merges "
        "duplicate campaign-days and fills gaps inside each campaign's date range with zero-delivery days. On the "
        f"simulated data the raw file had {log['rows_in']:,} rows; cleaning merged {log['duplicates_merged']} "
        f"duplicates, found {log['invalid_values']} invalid values and filled {log['gap_days_filled']} gap days, "
        f"giving a daily panel of {log['rows_out']:,} campaign-days. Removing days without a next-day label (the "
        "last day of each campaign and days followed by fewer than 100 impressions) and days with no delivery in "
        f"the previous week left {model_rows:,} labelled rows."), P(
        "Encoding and scaling depend on the model. The EBM handles platform, objective and day of week as nominal "
        "features and bins the numeric features, so it needs neither one-hot encoding nor scaling. CatBoost also "
        "handles categorical features natively, using ordered target statistics. For logistic regression the "
        "categorical features are one-hot encoded, the heavy-tailed volume features (spend, impressions, clicks, "
        "campaign age and CPM) are log-transformed, and all numeric features are standardised, with median "
        "imputation and a missing-value indicator. On a day with no delivery the one-day CTR is undefined, so the "
        "one-day index falls back to the seven-day index."),
        P("2.5 Dataset size and split", "H2"), P(
        f"The final dataset has {n_rows:,} labelled campaign-days, well above the target of 1,000 in the project "
        "brief. Table 3 shows how it divides into the chronological training, validation and test sets described in "
        "Section 4.3.")]
    split_rows = [["Split", "Days predicted", "Rows", "Campaigns", "Above benchmark"]]
    for s, name in [("train", "Training"), ("val", "Validation"), ("test", "Test")]:
        d = splits[s]
        split_rows.append([name, f"{d.target_date.min():%d %b %Y} – {d.target_date.max():%d %b %Y}", f"{len(d):,}",
                           d.campaign_id.nunique(), pct(d[config.TARGET].mean(), 1)])
    st.append(table(split_rows, [26 * mm, 52 * mm, 22 * mm, 24 * mm, 30 * mm], "Table 3. Chronological split.",
                    bold_first_col=True))

    st += [P("2.6 Ethics and collecting real data", "H2"), P(
        "The project uses only aggregated, campaign-level statistics: daily spend, impressions and clicks. It does "
        "not use user-level data, demographics or any other personal information, so no individual can be "
        "identified from the dataset. Real data should come only from accounts the author is authorised to access, "
        "with the account owner's permission. Campaign names can be commercially sensitive, so the conversion tool "
        "replaces them with one-way hashed identifiers, and real exports are excluded from version control."), P(
        "To use real data, daily campaign reports are exported from Google Ads (a campaign report segmented by day, "
        "with cost, impressions and clicks) and from Meta Ads Manager (the campaign table broken down by day, with "
        "amount spent, impressions and link clicks), then converted with:"),
        P('python -m src.data --export google.csv "Google Ads" --export meta.csv "Meta Ads"', "code"),
        P("The converter copes with the title lines that Google Ads places above the header, total rows, currency "
          "symbols and thousands separators. Retraining then needs no code changes.")]

    # 3 Algorithm
    st += [P("3. Algorithm selection: Explainable Boosting Machine", "H1"), P("3.1 How EBM works", "H2"), P(
        "An EBM is a generalised additive model with optional pairwise interactions, sometimes called a GA<super>2"
        "</super>M (Lou, Caruana and Gehrke, 2012; Lou et al., 2013). For binary classification it models the "
        "log-odds of the positive class as"),
        P("log[ <i>p</i> / (1 − <i>p</i>) ] = β<sub>0</sub> + Σ<sub><i>j</i></sub> <i>f<sub>j</sub></i>"
          "(<i>x<sub>j</sub></i>) + Σ<sub><i>i</i>&lt;<i>j</i></sub> <i>f<sub>ij</sub></i>(<i>x<sub>i</sub></i>, "
          "<i>x<sub>j</sub></i>)", "eq"),
        P("where each <i>f<sub>j</sub></i> is an arbitrary learned function of one feature and each "
          "<i>f<sub>ij</sub></i> a function of a pair of features. Because the model is a sum, any prediction can be "
          "split exactly into one contribution per term."), P(
        "What separates EBM from classical GAMs, which fit smooth splines (Hastie and Tibshirani, 1990), is how the "
        "functions are learned. The InterpretML implementation (Nori et al., 2019) uses cyclic gradient boosting. "
        "It visits the features one at a time in a round-robin, fits a very small tree to the current residuals "
        "using only that feature, and adds it to that feature's function with a low learning rate. Because each "
        "step sees a single feature, the effect of correlated features is shared out gradually instead of being "
        "claimed by whichever feature a tree happens to split on first. After thousands of rounds each function is "
        "stored as a lookup table over bins of the feature. Pairwise interactions are then chosen with the FAST "
        "heuristic and fitted to the remaining residuals in the same way. Finally, the whole procedure is repeated "
        "on bootstrap samples of the training data (14 \"outer bags\" by default) and averaged, which lowers "
        "variance and gives an uncertainty band for every learned function."),
        P("3.2 Why EBM was selected", "H2"), P(
        "The task needs a model that can capture non-linear effects. The relationship between recent and future CTR "
        "levels off at high values, fatigue only sets in after the first week or so, and day-of-week effects are "
        "categorical rather than linear. At the same time the output is meant to be acted on by marketers, who "
        "need to know why a campaign is flagged. Post-hoc methods such as SHAP or LIME can describe a black-box "
        "model, but they are approximations whose output depends on the explainer and its settings, and Rudin "
        "(2019) argues that where an interpretable model performs as well, it should be preferred. EBMs have been "
        "shown to come close to the accuracy of gradient-boosted trees on tabular data (Caruana et al., 2015; Nori "
        "et al., 2019), and their explanations are the model itself rather than a description of it. The dataset "
        "also suits the method: a few thousand rows of mixed numeric and categorical features, on which an EBM "
        "trains in about a second."),
        P("3.3 How EBM differs from standard models", "H2"), P(
        "Table 4 compares EBM with the standard models named in the assignment guidelines and with CatBoost, which "
        "is used here as the accuracy reference.")]
    cmp_rows = [["Model", "How it predicts", "Non-linear effects", "How it is explained", "Weakness for this task"],
                ["Decision tree", "A sequence of if-then splits", "Yes, as steps", "Read the path to a leaf",
                 "Unstable; deep trees become unreadable, shallow ones are inaccurate"],
                ["Logistic regression", "Weighted sum of features", "No (unless engineered)",
                 "One coefficient per feature", "Cannot follow curved or saturating effects; unstable when "
                 "features are correlated"],
                ["k-nearest neighbours", "Vote of the most similar past cases", "Yes", "Only by showing neighbours",
                 "No model of feature effects; sensitive to scaling"],
                ["CatBoost (gradient-boosted trees)", "Sum of hundreds of deep trees", "Yes, with complex interactions",
                 "Needs post-hoc tools", "Black box: explanations are approximations"],
                ["EBM", "Sum of learned per-feature and pairwise functions", "Yes, any shape per feature",
                 "Exact: plot each function; add up contributions", "Interactions limited to pairs"]]
    st.append(table(cmp_rows, [27 * mm, 34 * mm, 27 * mm, 33 * mm, TEXT_W - 121 * mm],
                    "Table 4. EBM compared with standard models.", bold_first_col=True))
    st += [P(
        "Compared with logistic regression, EBM keeps the additive structure but replaces each coefficient with a "
        "learned curve, so a feature can matter a great deal in one range and not at all in another. Compared with a "
        "decision tree, it does not split the data into ever smaller regions; each feature's effect is estimated "
        "from all of the data and stays readable however many features there are. k-NN stores no model at all, so "
        "it cannot say which features drove a prediction. Compared with a general gradient-boosting model such as "
        "CatBoost (Prokhorenkova et al., 2018), EBM restricts itself to single-feature and pairwise terms. That "
        "restriction is what makes it interpretable, and Section 5 tests whether it costs accuracy."),
        P("3.4 Eligibility", "H2"), P(
        "The assignment requires an algorithm that was not covered in lectures. EBM was not among the algorithms "
        "taught in the module " + ph("[confirm with the module leader before submission]") + ". Logistic "
        "regression appears in this project only as a comparison model.")]

    # 4 Methodology
    st += [P("4. Methodology", "H1"), P("4.1 Overview", "H2"), P(
        "Figure 1 summarises the workflow, which follows the steps set out in the project brief. Everything is "
        "scripted in Python using pandas, scikit-learn (Pedregosa et al., 2011), InterpretML and CatBoost, and a "
        "full run from the raw file to the final figures takes under a minute on a laptop."),
        KeepTogether([pipeline_diagram(), P("Figure 1. Modelling workflow. Blue steps prepare the data; grey steps "
                                            "train, evaluate and explain the models.", "caption")]),
        P("4.2 Exploratory analysis", "H2"), P(
        "Four findings from the exploratory analysis (<font face='Courier' size='9.5'>notebooks/01_eda.ipynb</font>) "
        "shaped the modelling. First, CTR levels differ strongly by objective and platform (Figure 2a), which "
        f"motivated the per-group benchmark. Second, CTR is persistent: the Spearman correlation between a "
        f"campaign's CTR index on one day and on the next is {rho1:.2f}, and {rho7:.2f} for the seven-day index "
        "(Figure 2b), so a rule that simply assumes tomorrow will look like today is a demanding baseline. Third, "
        "CTR rises over the first week of a campaign and then declines with age on both platforms (Figure 2c), the "
        "expected signature of a learning phase followed by creative fatigue. Fourth, the weekly pattern runs in "
        "opposite directions on the two platforms (Figure 2d): Google campaigns are stronger on weekdays and Meta "
        "campaigns at weekends."), P(
        f"The share of campaign-days above benchmark also drifts upwards over time, from {pct(pr['train'])} in "
        f"training to {pct(pr['val'])} in validation and {pct(pr['test'])} in test. This is one reason for testing "
        "on a later period rather than a random sample."),
        figure([FIGS / "eda_ctr_by_group.png", FIGS / "eda_persistence.png", FIGS / "eda_fatigue.png",
                FIGS / "eda_day_of_week.png"],
               "Figure 2. Exploratory analysis: (a, top left) daily CTR by objective and platform; (b, top right) "
               "CTR index today against tomorrow; (c, bottom left) median next-day CTR index by campaign age; "
               "(d, bottom right) median CTR index by day of week.",
               cols=2),
        P("4.3 Chronological split", "H2"), P(
        "Rows are ordered by the date being predicted. The oldest 70% form the training set, the next 15% the "
        "validation set and the newest 15% the test set (Table 3). Nothing is shuffled, so every evaluation is on "
        "days later than anything the model was fitted on, which is how the model would be used in practice. A "
        "random split would let the model see the same campaign on the days either side of a test day and would "
        "overstate performance. As a check, the features for 60 randomly chosen rows were rebuilt from data cut off "
        "at day <i>t</i>; they were identical to the stored features, confirming that no future information "
        "reaches the inputs."),
        P("4.4 Models and hyperparameter search", "H2"), P(
        "Each model was tuned by grid search: every configuration was trained on the training set and scored by "
        "ROC-AUC on the validation set (Table 5). For EBM the search covered the number of pairwise interactions "
        "(0, 5 or 10, where 0 gives a purely additive model), the maximum number of bins per feature and the "
        "minimum number of samples per leaf, with the learning rate kept at the library default. For logistic "
        "regression it covered the inverse regularisation strength <i>C</i> and class weighting, and for CatBoost "
        "the tree depth, learning rate and L2 regularisation, with the number of trees set by early stopping on "
        "the validation set.")]
    grid_rows = [["Model", "Values searched", "Selected"],
                 ["EBM", "interactions: 0, 5, 10; max_bins: 32, 128; min_samples_leaf: 4, 20; "
                  "learning_rate = 0.015; 14 outer bags",
                  f"interactions = {best[EBM]['interactions']}, max_bins = {best[EBM]['max_bins']}, "
                  f"min_samples_leaf = {best[EBM]['min_samples_leaf']}"],
                 ["Logistic regression", "C: 0.01, 0.1, 1, 10; class_weight: none, balanced",
                  f"C = {best[LR]['C']:g}, class_weight = {best[LR]['class_weight'] or 'none'}"],
                 ["CatBoost", "depth: 4, 6; learning_rate: 0.03, 0.1; l2_leaf_reg: 3, 10; up to 2,000 "
                  "trees with early stopping (100 rounds)",
                  f"depth = {best[CAT]['depth']}, learning_rate = {best[CAT]['learning_rate']}, "
                  f"l2_leaf_reg = {best[CAT]['l2_leaf_reg']}"]]
    st.append(table(grid_rows, [30 * mm, 78 * mm, TEXT_W - 108 * mm],
                    "Table 5. Hyperparameter grids (all combinations were tried) and the selected values.",
                    bold_first_col=True))
    st += [P(
        "The decision threshold for each model was chosen on the validation set by maximising Youden's <i>J</i> "
        "statistic, sensitivity plus specificity minus one (Youden, 1950). An earlier version maximised F1-score "
        "instead, which pushed every threshold down to about 0.25 and meant the models almost never flagged an "
        "underperforming campaign, the main purpose of the tool. After tuning, each model was refitted on the "
        "training and validation sets together and evaluated once on the test set."),
        P("4.5 Baseline", "H2"), P(
        "A persistence baseline predicts that a campaign will beat its benchmark tomorrow if it beat it today "
        "(one-day CTR index above 1), and uses the one-day index as its ranking score. Given the persistence seen "
        "in the exploratory analysis, a model is only worth having if it beats this rule."),
        P("4.6 Evaluation metrics", "H2"), P(
        "Precision, recall and F1-score are reported with \"above benchmark\" as the positive class, as specified in "
        "the project brief. Because the practical aim is to catch underperforming campaigns, specificity is also "
        "reported, labelled <i>at-risk recall</i>: the share of below-benchmark days that the model flags. ROC-AUC "
        "and PR-AUC summarise ranking quality across all thresholds, and the Brier score measures the accuracy of "
        "the predicted probabilities. Days from the same campaign are correlated, so 95% confidence intervals were "
        "obtained by resampling whole campaigns rather than individual days (a cluster bootstrap; Efron and "
        f"Tibshirani, 1993). The test set contains only {n_test_campaigns} campaigns, so the intervals are fairly "
        "wide. Differences between models were assessed on paired resamples of the same campaigns.")]

    # 5 Results
    st += [P("5. Results", "H1"), P("5.1 Overall test performance", "H2"), P(
        "Table 6 summarises the test-set results. The three trained models ranked campaign-days almost equally "
        f"well: ROC-AUC was {f3(M[EBM]['roc_auc'])} for EBM, {f3(M[CAT]['roc_auc'])} for CatBoost and "
        f"{f3(M[LR]['roc_auc'])} for logistic regression, against {f3(M[PER]['roc_auc'])} for the persistence "
        "baseline. The confidence intervals overlap heavily, which is expected with "
        f"{n_test_campaigns} test campaigns, so the paired comparisons in Table 7 are more informative.")]
    res_rows = [["Model", "Thr.", "Precision", "Recall", "F1", "At-risk recall", "ROC-AUC (95% CI)", "PR-AUC",
                 "Brier"]]
    for m in [EBM, CAT, LR, PER]:
        r = M[m]
        res_rows.append([m.replace("Persistence baseline", "Persistence"), f"{r['threshold']:.2f}",
                         f3(r["precision"]), f3(r["recall"]), f3(r["f1"]), f3(r["specificity"]),
                         f"{f3(r['roc_auc'])} ({r['roc_auc_ci'][0]:.3f}–{r['roc_auc_ci'][1]:.3f})", f3(r["pr_auc"]),
                         "–" if r["brier"] is None or np.isnan(r["brier"]) else f3(r["brier"])])
    st.append(table(res_rows, [27 * mm, 11 * mm, 16 * mm, 14 * mm, 13 * mm, 17 * mm, 31 * mm, 15 * mm,
                               TEXT_W - 144 * mm],
                    f"Table 6. Test-set results ({len(splits['test'])} campaign-days, {n_test_campaigns} campaigns). "
                    "Thresholds were fixed on the validation set; the persistence rule uses a CTR index of 1.",
                    bold_first_col=True))
    diff_rows = [["Comparison", "Difference in ROC-AUC", "95% CI", "Resamples favouring EBM"]]
    for m in [PER, CAT, LR]:
        mean, lo, hi, share = DIFF[m]
        diff_rows.append([f"EBM − {m.replace('Persistence baseline', 'persistence')}", f"{mean:+.3f}",
                          f"{lo:+.3f} to {hi:+.3f}", pct(share, 1)])
    st.append(table(diff_rows, [52 * mm, 36 * mm, 36 * mm, 40 * mm],
                    "Table 7. Paired differences in test ROC-AUC (cluster bootstrap, 1,000 resamples).",
                    bold_first_col=True))
    st += [P(
        f"The EBM's advantage over persistence ({DIFF[PER][0]:+.3f}) has an interval that excludes zero, so the "
        "model adds real information beyond yesterday's CTR. Its differences from CatBoost "
        f"({DIFF[CAT][0]:+.3f}) and logistic regression ({DIFF[LR][0]:+.3f}) do not; in other words, the "
        "glass-box model gave up no measurable accuracy relative to the black-box reference."), P(
        "At the chosen thresholds the models made different trade-offs. EBM had the highest recall of the three "
        f"trained models ({f3(M[EBM]['recall'])}) and the highest F1-score ({f3(M[EBM]['f1'])}), and caught "
        f"{pct(M[EBM]['specificity'])} of below-benchmark days. Logistic regression and CatBoost settled on higher "
        f"thresholds and flagged more of the at-risk days ({pct(M[LR]['specificity'])} and "
        f"{pct(M[CAT]['specificity'])}) at the cost of recall on the positive class. The persistence rule's F1-score "
        f"({f3(M[PER]['f1'])}) is close to the models', which deserves to be said plainly: for a simple yes/no "
        "decision the models add only a little over assuming that tomorrow will resemble today. Their larger "
        "contribution lies in ranking, in well-calibrated probabilities, which persistence cannot provide, and in "
        "the explanations."),
        figure([FIGS / "roc_curves.png", FIGS / "calibration.png"],
               "Figure 3. Test-set ROC curves (left) and calibration curves (right).", cols=2), P(
        "Figure 3 shows that the curves of the three trained models lie almost on top of one another and above the "
        f"persistence curve across most of the range. The calibration curves follow the diagonal closely (Brier "
        f"scores {f3(min(M[m]['brier'] for m in (EBM, CAT, LR)))}–{f3(max(M[m]['brier'] for m in (EBM, CAT, LR)))}"
        "), so a predicted 70% chance of beating the benchmark can be read roughly at face value. Figure 4 shows "
        "the confusion matrices at the chosen thresholds."),
        figure([FIGS / "confusion_matrices.png"], "Figure 4. Test-set confusion matrices at the validation-chosen "
               "thresholds."), P(
        f"Validation ROC-AUC was lower than test ROC-AUC for every model (EBM {f3(V[EBM]['roc_auc'])} against "
        f"{f3(M[EBM]['roc_auc'])}). Part of this comes from refitting on more recent data, but the persistence "
        f"rule, which involves no training at all, improved by a similar amount ({f3(V[PER]['roc_auc'])} to "
        f"{f3(M[PER]['roc_auc'])}). Most of the gap therefore reflects a test period that happened to be easier to "
        "predict. The test figures should be read together with their intervals, and the validation figures give a "
        "fair idea of performance in a less favourable period."),
        P("5.2 Effect of hyperparameters", "H2"), P(
        "Hyperparameters mattered little. The best EBM used "
        f"{best[EBM]['interactions']} interactions, but the purely additive model scored only "
        f"{auc_by_inter.max() - auc_by_inter.get(0, auc_by_inter.min()):.4f} lower in validation ROC-AUC "
        f"({auc_by_inter.get(0, np.nan):.4f} against {auc_by_inter.max():.4f}). Most of the signal is carried by "
        "main effects, which is convenient, because main effects are the easiest terms to explain. Logistic "
        f"regression preferred weak regularisation (<i>C</i> = {best[LR]['C']:g}) and CatBoost a depth of "
        f"{best[CAT]['depth']} with a learning rate of {best[CAT]['learning_rate']}. All models trained quickly, "
        f"taking on average {fit_secs[EBM]:.1f} s per EBM fit and {fit_secs[CAT]:.1f} s per CatBoost fit."),
        P("5.3 Performance across segments", "H2"), P(
        f"Performance was broadly similar on the two platforms, with an EBM ROC-AUC of "
        f"{f3(seg_ebm.loc['Google Ads', 'roc_auc'])} on Google Ads and {f3(seg_ebm.loc['Meta Ads', 'roc_auc'])} on "
        f"Meta Ads (Table 8). It varied more by objective, from {f3(objs.roc_auc.min())} to "
        f"{f3(objs.roc_auc.max())}, although the smallest group has only {int(objs.n.min())} test rows and its "
        f"figure is noisy. The weak spot is at-risk recall for Sales campaigns ({f3(seg_ebm.loc['Sales', 'specificity'])}"
        f"): {pct(seg_ebm.loc['Sales', 'share_above'])} of Sales days in the test period were above benchmark, and "
        "a single threshold shared by all groups flags fewer of the minority of bad days. A threshold per segment, "
        "or a benchmark that is updated over time, would address this.")]
    seg_rows = [["Segment", "Test rows", "Above benchmark", "ROC-AUC", "F1", "At-risk recall"]]
    for seg_type in ["platform", "objective"]:
        for seg, r in seg_ebm[seg_ebm.segment_type == seg_type].iterrows():
            seg_rows.append([seg, int(r.n), pct(r.share_above), f3(r.roc_auc), f3(r.f1), f3(r.specificity)])
    st.append(table(seg_rows, [34 * mm, 24 * mm, 30 * mm, 24 * mm, 22 * mm, 30 * mm],
                    "Table 8. EBM test performance by platform and by objective.", bold_first_col=True))

    # 6 Explainability
    i7, i1, i3 = imp["ctr_index_7d"], imp["ctr_index_1d"], imp["ctr_index_3d"]
    st += [P("6. Explainability and interpretation", "H1"), P("6.1 Global feature importance", "H2"), P(
        "EBM term importance is the mean absolute contribution of a term to the log-odds over the training data "
        f"(Figure 5). The three CTR-index features dominate (seven-day {i7:.3f}, one-day {i1:.3f}, three-day "
        f"{i3:.3f}), followed at a distance by campaign age ({imp['campaign_day']:.3f}), day of week "
        f"({imp['day_of_week']:.3f}) and the CTR trend ({imp['ctr_trend']:.3f}). Clicks and spend carry small "
        f"effects. Platform has almost no main effect ({plat.get('Google Ads', 0):+.3f} and "
        f"{plat.get('Meta Ads', 0):+.3f} log-odds for Google and Meta), which is exactly what should happen: the "
        "per-group benchmark already absorbs platform differences, so the model only has to learn how a campaign "
        f"moves relative to its peers. Of the {n_terms} terms in the final model, {n_inter} are pairwise "
        f"interactions, none with an importance above {max_inter_imp:.3f}."),
        figure([FIGS / "ebm_global_importance.png"], "Figure 5. EBM term importance (15 most important terms).",
               width=TEXT_W * 0.78),
        P("6.2 Learned shape functions", "H2"), P(
        "Figure 6 shows what the EBM learned for its most important features. Each panel plots the contribution to "
        "the log-odds against the feature value; the shaded band shows the variation across the bagged models."),
        figure([FIGS / "ebm_shape_functions.png"], "Figure 6. EBM shape functions for the nine most important main "
               "effects. Vertical lines mark a CTR index of 1 (at benchmark)."),
        bullets([
            "<b>CTR index.</b> For all three windows the contribution rises steadily and crosses zero at an index of "
            f"about {cross7:.2f} for the seven-day window, close to the point at which a campaign is exactly at its "
            "benchmark. That point was not given to the model; it was learned from the data. The curves flatten "
            "above an index of roughly 1.6 to 2, so a campaign far above benchmark gains little from being further "
            "above it.",
            f"<b>Campaign age.</b> Young campaigns receive a small positive contribution ({age_start_score:+.2f}), "
            f"after which the effect declines almost linearly to about {age_end_score:+.2f} by day {age_end_day}. "
            "This is the shape of creative fatigue.",
            f"<b>Day of week.</b> Wednesday ({dow['Wed']:+.2f}) and Thursday ({dow['Thu']:+.2f}) are the strongest "
            f"days and Saturday ({dow['Sat']:+.2f}) and Sunday ({dow['Sun']:+.2f}) the weakest.",
            "<b>CTR trend.</b> A sharp recent rise above the seven-day level raises the probability and a modest dip "
            "lowers it. Very large drops receive a slightly positive contribution, which hints at a rebound after "
            "one-off bad days, but this part of the curve rests on few observations and has a wide band.",
            "<b>Cumulative spend</b> has a falling curve, largely because it is correlated with campaign age. It "
            "should not be read as \"spending more makes CTR worse\"."]),
        P("6.3 Interactions", "H2"), P(
        "The strongest interaction pairs campaign age with the seven-day CTR index (Figure 7). A weak recent CTR is "
        "penalised more heavily in an older campaign than in a young one, while young campaigns with a low index "
        "receive a small upward correction. This fits the idea that a new campaign's poor start partly reflects the "
        "platform's learning phase and tends to recover, whereas poor CTR in an old campaign reflects fatigue and "
        "persists. The effect is real but small, in line with the small gain from interactions in Section 5.2."),
        figure([FIGS / "ebm_top_interaction.png"], "Figure 7. Strongest EBM interaction: campaign age × seven-day "
               "CTR index. Blue raises and red lowers the log-odds. Axes are in bins of the features.",
               width=TEXT_W * 0.55),
        P("6.4 Local explanations", "H2"), P(
        "Because the EBM is additive, each prediction breaks down exactly into an intercept plus one contribution "
        "per term, and the code checks that these add up to the predicted log-odds for every test row. Figure 8 "
        f"shows two test cases. In the first, campaign {below['campaign_id']} on "
        f"{pd.Timestamp(below['target_date']):%d %B %Y} (predicted probability {pct(below['probability'], 1)}), all "
        "three CTR-index terms are strongly negative because the campaign had been running at about "
        f"{below_row.ctr_index_7d:.2f} times its benchmark over the previous week; campaign age and the coming "
        f"{pd.Timestamp(below['target_date']):%A} add further negative weight. The campaign did fall below its "
        "benchmark."), P(
        f"The second case is the most confident error in the test set. Campaign {wrong['campaign_id']} was given a "
        f"{pct(wrong['probability'])} chance of beating its benchmark on "
        f"{pd.Timestamp(wrong['target_date']):%d %B %Y}, because its CTR had been running at "
        f"{wrong_row.ctr_index_7d:.2f} times the benchmark over the previous week and {wrong_row.ctr_index_1d:.2f} "
        f"times on the last day. The next day its CTR fell to {wrong_row.next_ctr:.3f}%, just below the benchmark of "
        f"{wrong_row.benchmark_ctr:.3f}% and {1 - wrong_row.next_ctr / wrong_row.ctr_7d:.0%} lower than its own "
        "seven-day average. The explanation makes the error easy to understand: nothing in the "
        "campaign's history pointed to the drop, so it was most likely a one-off shock that no model based on past "
        "performance could foresee."),
        P("6.5 Partial dependence", "H2"), P(
        "Partial dependence plots (Friedman, 2001) show the average predicted probability as one feature is varied "
        "while the others keep their observed values, and they can be computed for any model. Figure 9 compares "
        "the three models. EBM and CatBoost produce almost the same curves, including the flattening at high CTR "
        "index, which suggests both have found the same underlying relationship. Logistic regression is limited "
        "to smooth monotone curves. For campaign age this exaggerates the effect: its partial dependence falls from "
        f"{pdp_age[LR][0]:.2f} at the start of a campaign to {pdp_age[LR][-1]:.2f} by day {int(age_grid[-1])}, "
        f"whereas the EBM's falls only from {pdp_age[EBM][0]:.2f} to {pdp_age[EBM][-1]:.2f} and CatBoost's from "
        f"{pdp_age[CAT][0]:.2f} to {pdp_age[CAT][-1]:.2f}."),
        figure([FIGS / "local_confident_below.png", FIGS / "local_misclassified.png"],
               "Figure 8. Local explanations for two test cases: a correctly flagged at-risk campaign (top) and the "
               "most confident misclassification (bottom). Blue bars raise and red bars lower the probability."),
        P("6.6 Permutation importance", "H2"), P(
        "Permutation importance measures how much test ROC-AUC drops when one feature is shuffled (Breiman, 2001). "
        "For EBM and CatBoost it confirms the global picture: the CTR-index features account for almost all of the "
        "drop and everything else is close to zero (Figure 10). Logistic regression behaves differently. It relies "
        f"heavily on last-day spend ({perm_lr['spend_1d']:.3f}) and campaign age ({perm_lr['campaign_day']:.3f}), "
        "features the other two models largely ignore. The volume features are strongly correlated with each "
        "other, and a linear model can only approximate the curved CTR effects by leaning on correlated proxies. A "
        "marketer reading the logistic regression could easily conclude that spend drives CTR, which the other "
        "models do not support. This is a practical argument for EBM: its explanations are not only exact but also "
        "more stable."),
        figure([FIGS / "pdp_model_comparison.png"], "Figure 9. Partial dependence on the test set for the four most "
               "important numeric features, for all three models."),
        figure([FIGS / "permutation_importance.png"], "Figure 10. Permutation importance on the test set (mean and "
               "standard deviation over 10 shuffles).", width=TEXT_W * 0.8),
        P("6.7 Does the model agree with domain knowledge?", "H2"), P(
        "Broadly, yes. CTR persistence reflects the fact that an advert's relevance and audience fit change slowly. "
        "The decline with campaign age matches the long-documented wear-out of advertising under repeated exposure "
        "(Pechmann and Stewart, 1988). A weekday peak is plausible for search-led campaigns aimed at people at work, "
        "though it is the kind of pattern that should be checked against real account data. Because the simulated "
        "data was generated with these mechanisms, this section doubles as a recovery test, and the EBM recovered "
        "persistence, fatigue and the learning-phase effect without being told about them."), P(
        "Two discrepancies are worth stating. First, the exploratory analysis showed opposite weekly patterns on "
        "the two platforms, but the EBM learned a single day-of-week curve, and none of its selected interactions "
        "involves platform and day of week. Its weekday effect is therefore an average that is slightly wrong for "
        "both platforms. EBM lets interactions be specified explicitly, so a platform × day-of-week term can be "
        "forced into the model. Second, the generator lowers CTR when a campaign spends well above its usual "
        "budget, but no feature expresses spend relative to the campaign's own norm, so the model cannot isolate "
        "this effect. A ratio of last-day spend to seven-day average spend would be a natural addition. Both points "
        "illustrate the value of a glass-box model: the gaps show up in the explanations, whereas in a black box "
        "they would stay hidden.")]

    # 7 Dashboard
    st += [P("7. Front-end dashboard and deployment", "H1"), P("7.1 Dashboard", "H2"), P(
        "The trained models are served through a Streamlit dashboard (<font face='Courier' size='9.5'>"
        "app/streamlit_app.py</font>) with five tabs."),
        bullets([
            "<b>Predict.</b> The user either picks a campaign and date from the dataset or starts a blank campaign, "
            "then enters or edits the platform, objective, campaign age, total spend and the last seven days of "
            "spend, impressions and clicks. The dashboard returns the probability that tomorrow's CTR beats the "
            "benchmark, an \"on track\" or \"at risk\" verdict, the EBM's contribution chart, a short plain-language "
            "summary of what raises and lowers the chance, and the probabilities from the two comparison models "
            "(Figure 11). Editing the table turns this into a what-if tool, for example to see how a further drop in "
            "clicks would change the outlook.",
            "<b>Campaign watchlist.</b> Scores every campaign active on a chosen date, or every campaign in an "
            "uploaded CSV file, ranks them from most to least at risk and names the main negative factor for each "
            "(Figure 12). The list can be downloaded.",
            "<b>What the model learned.</b> Interactive versions of the importance chart, shape functions, "
            "interactions and partial dependence plots.",
            "<b>Model performance.</b> Test metrics with confidence intervals, ROC and precision-recall curves, "
            "confusion matrices and the full hyperparameter search.",
            "<b>Data overview.</b> Dataset size, CTR trends by platform and the CTR distribution by objective."]),
        figure([SHOTS / "predict_result.png"], "Figure 11. Dashboard prediction for an at-risk Google Ads Leads "
               "campaign, with the EBM's explanation."),
        figure([SHOTS / "watchlist.png"], "Figure 12. Campaign watchlist ranked by the chance of beating the "
               "benchmark tomorrow."),
        P("The dashboard calls the same feature-engineering code as the training pipeline. A test confirmed that, "
          "for 60 random campaign-days, the inputs it builds from a seven-day history are identical to those used "
          "in training. It also shows a warning banner whenever the loaded models were trained on simulated data.")]

    code = "<font face='Courier' size='9.5'>{}</font>".format
    mlops = [["Practice", "Implementation"],
             ["Versioning", "Code, data, models and metrics are kept in git. Each Docker image and each Container Apps "
              "revision is tagged with its commit, which the dashboard shows as \u201cApp build\u201d."],
             ["Automated testing", "20 tests on data cleaning, the chronological split, look-ahead leakage, "
              "reproduction of the reported results, EBM additivity and the dashboard."],
             ["Quality gate", "A release is blocked unless the EBM reaches a test ROC-AUC of at least 0.80 and beats "
              "the persistence baseline."],
             ["Continuous delivery", "Test, then build and smoke-test the container, push it to Azure Container "
              "Registry, roll out a new revision and check the live URL."],
             ["Continuous training", "A retraining workflow repeats tuning, evaluation and the quality gate, then "
              "proposes the models in a pull request that compares their metrics with production."],
             ["Monitoring", "A daily health check of the live app, plus container logs and metrics in Azure Log "
              "Analytics."],
             ["Rollback", "Re-running an earlier deploy job puts that commit's image back into service."],
             ["Security", "GitHub signs in to Azure through OpenID Connect without a stored password, as an identity "
              "limited to one resource group; the app pulls images with a separate read-only identity."]]
    st += [P("7.2 Deployment and MLOps", "H2"), P(
        "The dashboard runs on Microsoft Azure as a Docker container on Azure Container Apps, which scales to zero "
        "when nobody is using it, so apart from about US$5 a month for the container registry it stays within the "
        "free allowance of a student subscription. It is publicly available at:"),
        Paragraph(f'<a href="{LIVE_URL}" color="{ACCENT.hexval()}">{LIVE_URL}</a>',
                  ParagraphStyle("url", parent=S["body"], alignment=TA_CENTER)),
        P(f"After a one-time setup script ({code('deploy/azure_setup.sh')}), everything is automated with three GitHub Actions workflows that "
        "follow common MLOps practice (Kreuzberger, Kühl and Hirschl, 2023), shown in Figure 13 and Table 9."),
        KeepTogether([mlops_diagram(), P("Figure 13. Deployment and MLOps workflow. Blue steps run on every push "
                                         "to the main branch; grey steps run on demand, daily or in production.",
                                         "caption")]),
        P(f"Every push to the main branch runs {code('ci-cd.yml')}. The tests treat the model as part of the "
          "software: besides checking the code, they confirm that the saved models still reproduce the test "
          "predictions in this report, that each EBM explanation adds up exactly to its prediction, and that "
          "features computed for a day do not change when all later days are removed. A quality gate then stops "
          "the release if the EBM falls below a test ROC-AUC of 0.80 or no longer beats the persistence "
          "baseline. Only an image that has passed these tests and rendered the dashboard inside its container is "
          "deployed (Figure 14). The retraining workflow ({}) runs the full training pipeline on the data in the "
          "repository and proposes the new models in a pull request, so a person reviews the metric changes "
          "before anything reaches production; merging the pull request triggers the normal CI/CD run. The first "
          "retraining run, on GitHub's Linux servers rather than the machine the models were first trained on, "
          "produced slightly different models from the same data and seeds: the EBM's test ROC-AUC moved from "
          "0.893 to 0.894 and CatBoost's from 0.895 to 0.892. Small platform-dependent changes like these are why "
          "new models are compared and reviewed before they are deployed rather than replacing the old ones "
          "automatically.".format(code("retrain.yml"))),
        table(mlops, [36 * mm, TEXT_W - 36 * mm], "Table 9. MLOps practices in the deployment.",
              bold_first_col=True),
        figure([SHOTS / "github_actions.png"], "Figure 14. A CI/CD run in GitHub Actions: tests, then build and "
               "smoke test of the container image, then deployment to the production environment on Azure."),
        P("The set-up has clear limits, which are typical of ML systems in their first deployment (Sculley et al., "
          "2015). Retraining is started by hand rather than by a schedule or by detected data drift, and the "
          "health check only confirms that the app answers, not that its predictions remain accurate. The "
          "training data is stored in the repository, which is acceptable for simulated data but not for "
          "confidential account exports; with real data, the retraining workflow would read them from protected "
          "cloud storage instead.")]

    # 8 Critical discussion
    st += [P("8. Critical discussion", "H1"), P("8.1 Limitations of the model", "H2"), P(
        "The model looks one day ahead and answers a yes/no question. It says nothing about how far above or below "
        "the benchmark a campaign will land, and a campaign hovering around its benchmark will flip between classes "
        "for reasons that are mostly noise. Its inputs are limited to a campaign's own recent delivery: creative "
        "content, audience targeting, placements, competitor activity, promotions and seasonality are all missing, "
        "and in practice they drive much of the variation in CTR. The misclassified case in Section 6.4 is typical "
        "of what the model cannot see. The gain over the persistence rule is real but modest, especially for hard "
        "yes/no labels, so the model is best seen as a way to rank and explain campaigns rather than as a precise "
        "forecast. Finally, a single threshold serves some segments worse than others (Section 5.3), and the model "
        "has been evaluated on one test period only."),
        P("8.2 Data quality", "H2"), P(
        "The most important limitation is that the results come from simulated data. The simulation encodes "
        "plausible mechanisms, but its noise is cleaner than real data and every effect the model found was put "
        "there by design. Real exports bring further problems: tracking and attribution settings that change over "
        "time, different click definitions (Meta's link clicks against all clicks), reporting time zones and "
        "currencies, platforms restating recent days after the fact, and small campaigns whose daily CTR is "
        "dominated by noise. The cleaning code handles duplicates, gaps, total rows and impossible values, and the "
        "100-impression rule limits noisy labels, but the other issues need care when the data is collected."),
        P("8.3 Risks of bias and unfairness", "H2"), P(
        "The dataset contains no personal data, so the usual concerns about protected characteristics do not apply "
        "directly. Bias can still arise at the level of campaigns. Small-budget campaigns have fewer impressions and "
        "noisier CTR, so their predictions are less reliable and they may be flagged more often for random reasons. "
        "Long-running campaigns contribute more campaign-days and therefore carry more weight in the median "
        "benchmark. There is also a feedback risk: if flagged campaigns are routinely paused, they stop producing "
        "data, and the model never learns whether they would have recovered. Performance also differs by objective "
        "(Table 8), so a team relying on the tool for Sales campaigns would miss more problems than one running "
        "Traffic campaigns. Reporting results by segment, as done here, is a first safeguard."),
        P("8.4 Real-world impact and ethical considerations", "H2"), P(
        "Used well, the tool helps a small team decide where to look first each morning and gives a reason that "
        "can be checked against what the team knows about the campaign. Used badly, it could replace judgement: a "
        "confident \"on track\" label might lead someone to ignore a campaign whose client brief has just changed. "
        "The predictions describe associations in historical data, not causes, so they should not be used to "
        "justify actions such as cutting a budget on the grounds that the model \"shows\" spend lowers CTR. The "
        "interpretable model helps with accountability here, because any decision based on it can be traced to "
        "specific contributions. On the data side, real use requires the account owner's permission, "
        "confidentiality for client names and figures, and compliance with the advertising platforms' terms of "
        "service. Because conditions on the platforms change constantly, the model would need regular retraining "
        "and its performance would have to be monitored over time.")]

    # 9 Conclusion
    st += [P("9. Conclusion and future work", "H1"), P(
        "This project built an end-to-end pipeline that predicts whether a digital marketing campaign's CTR will "
        f"beat its benchmark tomorrow and explains each prediction. On a {data_note} dataset of {n_rows:,} "
        "campaign-days, an Explainable Boosting Machine matched the accuracy of CatBoost and logistic regression, "
        "outperformed a persistence rule, and produced explanations that recovered the mechanisms built into the "
        "data while exposing two gaps in the feature set. These results show that the method works; they do not "
        "show how well it would work on real campaigns."), P(
        "The next steps follow directly from the discussion. The pipeline should first be retrained on real "
        "Google Ads and Meta Ads exports, which needs no code changes. The feature set should then gain a "
        "platform × day-of-week interaction and a relative-spend feature. Per-segment thresholds, and a regression "
        "or multi-day target, would make the predictions more useful in practice. Finally, the deployed dashboard "
        "already has automated tests, a quality gate, on-demand retraining and a daily health check; data-drift "
        "monitoring and a retraining schedule are the remaining steps towards production use.")]

    # References
    refs = [
        "Breiman, L. (2001) 'Random forests', <i>Machine Learning</i>, 45(1), pp. 5–32.",
        "Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M. and Elhadad, N. (2015) 'Intelligible models for "
        "healthcare: predicting pneumonia risk and hospital 30-day readmission', <i>Proceedings of the 21st ACM "
        "SIGKDD International Conference on Knowledge Discovery and Data Mining</i>, pp. 1721–1730.",
        "Efron, B. and Tibshirani, R.J. (1993) <i>An Introduction to the Bootstrap</i>. New York: Chapman &amp; Hall.",
        "Friedman, J.H. (2001) 'Greedy function approximation: a gradient boosting machine', <i>Annals of "
        "Statistics</i>, 29(5), pp. 1189–1232.",
        "Hastie, T. and Tibshirani, R. (1990) <i>Generalized Additive Models</i>. London: Chapman &amp; Hall.",
        "Kreuzberger, D., Kühl, N. and Hirschl, S. (2023) 'Machine learning operations (MLOps): overview, "
        "definition, and architecture', <i>IEEE Access</i>, 11, pp. 31866–31879.",
        "Lou, Y., Caruana, R. and Gehrke, J. (2012) 'Intelligible models for classification and regression', "
        "<i>Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i>, "
        "pp. 150–158.",
        "Lou, Y., Caruana, R., Gehrke, J. and Hooker, G. (2013) 'Accurate intelligible models with pairwise "
        "interactions', <i>Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and "
        "Data Mining</i>, pp. 623–631.",
        "McMahan, H.B. et al. (2013) 'Ad click prediction: a view from the trenches', <i>Proceedings of the 19th "
        "ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i>, pp. 1222–1230.",
        "Nori, H., Jenkins, S., Koch, P. and Caruana, R. (2019) 'InterpretML: a unified framework for machine "
        "learning interpretability', <i>arXiv preprint</i> arXiv:1909.09223.",
        "Pechmann, C. and Stewart, D.W. (1988) 'Advertising repetition: a critical review of wearin and wearout', "
        "<i>Current Issues and Research in Advertising</i>, 11(1–2), pp. 285–329.",
        "Pedregosa, F. et al. (2011) 'Scikit-learn: machine learning in Python', <i>Journal of Machine Learning "
        "Research</i>, 12, pp. 2825–2830.",
        "Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V. and Gulin, A. (2018) 'CatBoost: unbiased "
        "boosting with categorical features', <i>Advances in Neural Information Processing Systems</i>, 31.",
        "Richardson, M., Dominowska, E. and Ragno, R. (2007) 'Predicting clicks: estimating the click-through rate "
        "for new ads', <i>Proceedings of the 16th International Conference on World Wide Web</i>, pp. 521–530.",
        "Rudin, C. (2019) 'Stop explaining black box machine learning models for high stakes decisions and use "
        "interpretable models instead', <i>Nature Machine Intelligence</i>, 1(5), pp. 206–215.",
        "Sculley, D. et al. (2015) 'Hidden technical debt in machine learning systems', <i>Advances in Neural "
        "Information Processing Systems</i>, 28, pp. 2503–2511.",
        "Youden, W.J. (1950) 'Index for rating diagnostic tests', <i>Cancer</i>, 3(1), pp. 32–35."]
    st += [P("References", "H1")] + [P(r, "ref") for r in refs]

    # Appendix
    import catboost as cb
    import interpret
    import sklearn
    import streamlit
    versions = (f"Python {sys.version.split()[0]}, interpret {interpret.__version__}, scikit-learn "
                f"{sklearn.__version__}, CatBoost {cb.__version__}, pandas {pd.__version__}, Streamlit "
                f"{streamlit.__version__}")
    st += [P("Appendix A. Reproducing the results", "H1"), P(
        f"All results can be regenerated from the source code. Software used: {versions}. From the project folder:"),
        P("pip install -r requirements.txt", "code"),
        P("python -m src.generate_synthetic&nbsp;&nbsp;&nbsp;# only needed for the simulated data", "code"),
        P("python -m src.train&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
          "&nbsp;&nbsp;# cleaning, features, tuning, evaluation", "code"),
        P("python -m src.explain&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
          "# explainability figures and tables", "code"),
        P("streamlit run app/streamlit_app.py&nbsp;# dashboard", "code"),
        P("pytest&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
          "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# tests and quality gate", "code"),
        P("bash deploy/azure_setup.sh&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# one-time Azure setup; "
          "later deployments run in GitHub Actions", "code"),
        P("Random seeds are fixed, and two consecutive runs produced identical data and metrics.")]
    files = [["File", "Purpose"],
             ["src/data.py", "Loading, validation and cleaning; converter for platform exports"],
             ["src/generate_synthetic.py", "Simulated campaign data (development only)"],
             ["src/features.py", "Features, benchmark, target and chronological split"],
             ["src/models.py", "Model definitions and hyperparameter grids"],
             ["src/train.py", "Tuning, threshold selection, refitting and test evaluation"],
             ["src/evaluate.py", "Metrics, cluster bootstrap, segment analysis and evaluation plots"],
             ["src/explain.py", "EBM explanations, partial dependence and permutation importance"],
             ["notebooks/01_eda.ipynb", "Exploratory data analysis"],
             ["app/streamlit_app.py", "Interactive dashboard"],
             ["tests/", "Automated tests, including the deployment quality gate"],
             ["Dockerfile", "Container image of the dashboard"],
             [".github/workflows/", "GitHub Actions: CI/CD to Azure, retraining and daily health check"],
             ["deploy/azure_setup.sh", "One-time Azure setup"],
             ["report/build_report.py", "Builds this report from the saved results"]]
    st.append(table(files, [52 * mm, TEXT_W - 52 * mm], "Table A1. Main source files."))
    return st


def main():
    doc = ReportDoc(str(OUT))
    doc.multiBuild(build())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
