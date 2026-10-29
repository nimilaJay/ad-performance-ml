"""Plotly charts for the dashboard, styled with the same palette as the report figures."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

THEMES = {
    "light": {"ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"], "pos": "#2a78d6", "neg": "#e34948",
              "neutral": "#f0efec", "band": "rgba(42,120,214,0.12)",
              "seq": ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"]},
    "dark": {"ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70"], "pos": "#3987e5", "neg": "#e66767",
             "neutral": "#383835", "band": "rgba(57,135,229,0.15)",
             "seq": ["#184f95", "#2a78d6", "#6da7ec", "#cde2fb"]},
}
MODEL_ORDER = ["EBM", "Logistic Regression", "CatBoost"]


class Charts:
    def __init__(self, mode: str = "light"):
        self.t = THEMES.get(mode, THEMES["light"])
        self.model_color = {m: c for m, c in zip(MODEL_ORDER, self.t["series"])}
        self.model_color["Persistence baseline"] = self.t["muted"]

    def _layout(self, fig, height=380, **kw):
        """Shared styling. Titles live in the page (st.markdown) so they never collide with legends."""
        t = self.t
        # automargin is set explicitly: Streamlit's default Plotly template does not grow margins for labels.
        axis = dict(gridcolor=t["grid"], linecolor=t["axis"], zerolinecolor=t["axis"], automargin=True,
                    tickfont=dict(color=t["ink2"]), title_font=dict(color=t["ink2"]))
        fig.update_layout(
            height=height, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=t["ink"], size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(font_size=12), **kw)
        fig.update_xaxes(**axis)
        fig.update_yaxes(**axis)
        return fig

    # -------------------------------------------------------------- explanations

    def contributions(self, contrib: pd.DataFrame, top=10):
        d = contrib.head(top).iloc[::-1]
        labels = [f"{lab} = {val}" for lab, val in zip(d["label"], d["value"])]
        colors = [self.t["pos"] if v >= 0 else self.t["neg"] for v in d["contribution"]]
        fig = go.Figure(go.Bar(
            x=d["contribution"], y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
            width=0.6, hovertemplate="%{y}<br>contribution %{x:+.3f} log-odds<extra></extra>"))
        fig.add_vline(x=0, line_color=self.t["axis"], line_width=1)
        fig.update_xaxes(title="Contribution to log-odds")
        fig.update_yaxes(showgrid=False)
        return self._layout(fig, height=60 + 34 * len(d))

    def importance(self, imp: pd.DataFrame, top=15):
        d = imp.head(top).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=d["importance"], y=d["label"], orientation="h", width=0.6,
            marker=dict(color=self.t["series"][0], cornerradius=4),
            text=[f"{v:.3f}" for v in d["importance"]], textposition="outside",
            textfont=dict(color=self.t["ink2"], size=11),
            hovertemplate="%{y}<br>mean |contribution| %{x:.3f}<extra></extra>"))
        # Headroom so the value labels outside the longest bar are not clipped.
        fig.update_xaxes(title="Mean |contribution| (log-odds)", range=[0, d["importance"].max() * 1.2])
        fig.update_yaxes(showgrid=False)
        return self._layout(fig, height=60 + 30 * len(d))

    def shape(self, shape: pd.DataFrame, feature_label: str, x_range=None):
        t = self.t
        fig = go.Figure()
        if "category" in shape:
            colors = [t["pos"] if v >= 0 else t["neg"] for v in shape["score"]]
            fig.add_bar(x=shape["category"].astype(str), y=shape["score"], width=0.6,
                        marker=dict(color=colors, cornerradius=4),
                        error_y=dict(type="data", symmetric=False, array=shape["upper"] - shape["score"],
                                     arrayminus=shape["score"] - shape["lower"], color=t["ink2"], thickness=1),
                        hovertemplate="%{x}: %{y:+.3f} log-odds<extra></extra>")
            fig.update_xaxes(showgrid=False)
        else:
            x = np.r_[shape["x_left"], shape["x_right"].iloc[-1]]
            y, lo, hi = (np.r_[shape[c], shape[c].iloc[-1]] for c in ("score", "lower", "upper"))
            fig.add_scatter(x=x, y=hi, line=dict(width=0, shape="hv"), hoverinfo="skip", showlegend=False)
            fig.add_scatter(x=x, y=lo, line=dict(width=0, shape="hv"), fill="tonexty", fillcolor=t["band"],
                            hoverinfo="skip", showlegend=False)
            fig.add_scatter(x=x, y=y, line=dict(color=t["series"][0], width=2, shape="hv"), showlegend=False,
                            hovertemplate="%{x:.3g}: %{y:+.3f} log-odds<extra></extra>")
            if x_range:
                fig.update_xaxes(range=x_range)
        fig.add_hline(y=0, line_color=t["axis"], line_width=1)
        fig.update_xaxes(title=feature_label)
        fig.update_yaxes(title="Contribution (log-odds)")
        return self._layout(fig, height=360)

    def interaction(self, left, right, scores, x_label, y_label):
        lim = float(np.abs(scores).max())
        xc = [f"{a:.3g}–{b:.3g}" for a, b in zip(left[:-1], left[1:])]
        yc = [f"{a:.3g}–{b:.3g}" for a, b in zip(right[:-1], right[1:])]
        fig = go.Figure(go.Heatmap(
            z=scores.T, x=xc, y=yc, zmin=-lim, zmax=lim,
            colorscale=[[0, self.t["neg"]], [0.5, self.t["neutral"]], [1, self.t["pos"]]],
            colorbar=dict(title="log-odds", tickfont=dict(color=self.t["ink2"])),
            hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>contribution %{{z:+.3f}}<extra></extra>"))
        fig.update_xaxes(title=x_label, showticklabels=False, showgrid=False)
        fig.update_yaxes(title=y_label, showticklabels=False, showgrid=False)
        return self._layout(fig, height=420)

    def pdp(self, grid, curves: dict, feature_label: str, ref_line=None):
        fig = go.Figure()
        for name, y in curves.items():
            fig.add_scatter(x=grid, y=y, name=name, mode="lines", line=dict(color=self.model_color[name], width=2),
                            hovertemplate=f"{name}<br>%{{x:.3g}} → %{{y:.1%}}<extra></extra>")
        if ref_line is not None:
            fig.add_vline(x=ref_line, line_color=self.t["axis"], line_width=1)
        fig.update_xaxes(title=feature_label)
        fig.update_yaxes(title="Average P(above benchmark)", range=[0, 1], tickformat=".0%")
        return self._layout(fig, height=380, hovermode="x unified")

    # -------------------------------------------------------------- performance

    def roc(self, y, probs: dict):
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Chance", mode="lines", line=dict(color=self.t["axis"], width=1),
                        hoverinfo="skip")
        for name, p in probs.items():
            fpr, tpr, thr = roc_curve(y, p)
            fig.add_scatter(x=fpr, y=tpr, name=name, mode="lines", line=dict(color=self.model_color[name], width=2),
                            customdata=thr, hovertemplate=f"{name}<br>FPR %{{x:.2f}}, TPR %{{y:.2f}}"
                                                          "<br>threshold %{customdata:.2f}<extra></extra>")
        fig.update_xaxes(title="False positive rate", range=[0, 1])
        fig.update_yaxes(title="True positive rate", range=[0, 1.02])
        return self._layout(fig, height=420)

    def pr(self, y, probs: dict):
        fig = go.Figure()
        fig.add_hline(y=float(np.mean(y)), line_color=self.t["axis"], line_width=1)
        for name, p in probs.items():
            prec, rec, _ = precision_recall_curve(y, p)
            fig.add_scatter(x=rec, y=prec, name=name, mode="lines", line=dict(color=self.model_color[name], width=2),
                            hovertemplate=f"{name}<br>recall %{{x:.2f}}, precision %{{y:.2f}}<extra></extra>")
        fig.update_xaxes(title="Recall", range=[0, 1])
        fig.update_yaxes(title="Precision", range=[0, 1.02])
        return self._layout(fig, height=420)

    def confusion(self, y, pred):
        cm = confusion_matrix(y, pred, labels=[0, 1])
        share = cm / cm.sum()
        labels = ["Below benchmark", "Above benchmark"]
        fig = go.Figure(go.Heatmap(
            z=share, x=labels, y=labels, zmin=0, zmax=1, showscale=False,
            colorscale=[[i / (len(self.t["seq"]) - 1), c] for i, c in enumerate(self.t["seq"])],
            text=[[f"{cm[i, j]}<br>{share[i, j]:.0%}" for j in range(2)] for i in range(2)],
            texttemplate="%{text}", textfont=dict(size=14),
            hovertemplate="actual %{y}<br>predicted %{x}<br>%{text}<extra></extra>"))
        fig.update_xaxes(title="Predicted", showgrid=False)
        fig.update_yaxes(title="Actual", showgrid=False, autorange="reversed")
        return self._layout(fig, height=360)

    # -------------------------------------------------------------- data overview

    def ctr_index_over_time(self, df: pd.DataFrame, split_dates: dict):
        weekly = (df.assign(week=df["target_date"].dt.to_period("W").dt.start_time)
                    .groupby(["week", "platform"])["ctr_index_1d"].median().reset_index())
        fig = go.Figure()
        for i, (platform, g) in enumerate(weekly.groupby("platform")):
            fig.add_scatter(x=g["week"], y=g["ctr_index_1d"], name=platform, mode="lines",
                            line=dict(color=self.t["series"][i], width=2),
                            hovertemplate=f"{platform}<br>week of %{{x|%d %b %Y}}<br>median index %{{y:.2f}}<extra></extra>")
        fig.add_hline(y=1, line_color=self.t["axis"], line_width=1)
        for label, date in split_dates.items():
            fig.add_vline(x=pd.Timestamp(date).timestamp() * 1000, line_color=self.t["muted"], line_width=1,
                          annotation_text=label, annotation_font_color=self.t["ink2"], annotation_position="top left")
        fig.update_yaxes(title="Median daily CTR index (1 = benchmark)")
        return self._layout(fig, height=400, hovermode="x unified")

    def ctr_by_group(self, df: pd.DataFrame):
        fig = go.Figure()
        for i, (platform, g) in enumerate(df.groupby("platform")):
            fig.add_box(x=g["objective"], y=g["ctr_1d"], name=platform, marker_color=self.t["series"][i],
                        boxpoints=False, line=dict(width=1.5))
        fig.update_layout(boxmode="group")
        fig.update_yaxes(title="Daily CTR (%)", type="log", tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10],
                         ticktext=["0.1", "0.2", "0.5", "1", "2", "5", "10"])
        return self._layout(fig, height=400)
