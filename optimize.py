"""
optimize.py
Hybrid Surrogate NSGA-II Optimizer (Strict 0.7mm Airgap Edition)
- Dynamically selects best models (XGBoost/CatBoost)
- Strictly forces AI input to 0.0 to prevent extrapolation hallucinations
- Strictly forces Output Geometry to Stator Inner - 1.4 (0.7mm radial airgap)
- Includes Hyper-Resilient JSON parser to prevent bounds TypeErrors
"""

import argparse
import json
import warnings
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

TARGET_MAP = {"torque": 0, "efficiency": 1, "power_factor": 2, "ripple": 3}
Y_KEYS = {0: "y1", 1: "y2", 2: "y3", 3: "y4"}

# ── 1. Smart Name Cleaner ────────────────────────────────────────────────────

def clean_var_name(name):
    """Cleans Altair variable names while preserving Stator/Rotor distinctions."""
    name_str = str(name)
    parts = name_str.split("::")
    
    if len(parts) >= 2 and parts[-2] in ["Stator", "Rotor"]:
        return f"{parts[-2]} {parts[-1]}".strip()
    
    return parts[-1].strip()


def manifold_distance(x, mean, std, nn_index, k):
    """
    Vectorized distance of each row in x to its k-th nearest REAL neighbor in
    standardized variable space. Follows the actual (possibly curved,
    non-convex) shape of the feasible design region, instead of approximating
    it with a single smooth ellipsoid -- so it correctly rejects candidates
    that sit in a 'gap' between real clusters even if every individual
    variable is in range and a covariance-based check would let them through.
    """
    x_std = (x - mean) / std
    dists, _ = nn_index.kneighbors(x_std, n_neighbors=k)
    return dists[:, -1]


def enforce_airgap(x, stator_idx, rotor_idx, airgap_info):
    """
    Forces the Rotor::Outer column to respect the airgap that was ACTUALLY
    observed in the training data, instead of a hardcoded 1.4mm.

    - airgap_info is None / missing       -> no relationship known, leave x untouched.
    - airgap_info['is_fixed'] is True      -> the AI has only ever seen ~one gap
                                               (e.g. 0.7mm radial today). Lock onto the
                                               measured mean gap. Retrain with a
                                               different fixed gap later and this
                                               adapts automatically, no code change.
    - airgap_info['is_fixed'] is False     -> the data spans a real range of gaps.
                                               Don't collapse to one number: clip the
                                               *requested* gap into the observed
                                               [min, max] range so NSGA-II can explore
                                               variable airgaps without extrapolating
                                               past what the AI was trained on.

    Critically: this is called ONCE, identically, both during NSGA-II's internal
    evaluation and during final design export, so the performance numbers you see
    always correspond to the geometry you actually get (previously the code zeroed
    rotor_out for prediction, then silently changed it afterward for export -- two
    different geometries, only one of which was ever actually evaluated).
    """
    if stator_idx == -1 or rotor_idx == -1 or not airgap_info:
        return x

    if airgap_info['is_fixed']:
        gap = airgap_info['observed_mean_diametric']
        x[..., rotor_idx] = x[..., stator_idx] - gap
    else:
        gmin = airgap_info['observed_min_diametric']
        gmax = airgap_info['observed_max_diametric']
        requested_gap = x[..., stator_idx] - x[..., rotor_idx]
        clipped_gap = np.clip(requested_gap, gmin, gmax)
        x[..., rotor_idx] = x[..., stator_idx] - clipped_gap

    return x


# ── 2. Vectorized Problem Definition ─────────────────────────────────────────

class HybridMotorProblemVec(Problem):
    def __init__(self, cat_model, xgb_model, winners, bounds, targets, config, var_names, airgap_info=None, manifold=None, radial_depth_budget=None):
        self.active_keys = list(targets.keys())
        
        self.stator_in_idx = -1
        self.rotor_out_idx = -1
        for i, n in enumerate(var_names):
            if "Stator::Inner" in str(n): self.stator_in_idx = i
            elif "Rotor::Outer" in str(n): self.rotor_out_idx = i

        self.airgap_info = airgap_info

        self.manifold = manifold
        n_extra_constr = 0
        if manifold is not None:
            self.mf_mean = np.array(manifold['mean'])
            self.mf_std = np.array(manifold['std'])
            self.mf_k = manifold['k']
            self.mf_threshold = float(manifold['threshold'])
            # Built ONCE here, reused for every generation's _evaluate call.
            self.mf_nn_index = NearestNeighbors(n_neighbors=self.mf_k).fit(np.array(manifold['points']))
            n_extra_constr += 1

        # ---> RADIAL DEPTH BUDGET <---
        # Resolve column-name-based info into integer indices into x, once,
        # so _evaluate can stay fully vectorized every generation.
        self.radial_depth_budget = radial_depth_budget
        if radial_depth_budget is not None:
            name_to_idx = {str(n): i for i, n in enumerate(var_names)}
            self.rdb_outer_idx = name_to_idx.get(radial_depth_budget['rotor_outer_col'], -1)
            self.rdb_inner_idx = name_to_idx.get(radial_depth_budget['rotor_inner_col'], -1)
            self.rdb_barrier_idx = [name_to_idx[c] for c in radial_depth_budget['barrier_cols'] if c in name_to_idx]
            self.rdb_min_ratio = float(radial_depth_budget['min_ratio'])
            if self.rdb_outer_idx == -1 or self.rdb_inner_idx == -1 or not self.rdb_barrier_idx:
                self.radial_depth_budget = None  # can't resolve columns, skip safely
            else:
                n_extra_constr += 1

        super().__init__(
            n_var=len(bounds['lb']),
            n_obj=len(self.active_keys),
            n_ieq_constr=len(self.active_keys) * 2 + n_extra_constr,
            xl=bounds['lb'],
            xu=bounds['ub']
        )
        self.cat_model = cat_model
        self.xgb_model = xgb_model
        self.winners = winners
        self.targets = targets
        self.config = config

    def _evaluate(self, x, out, *args, **kwargs):
        F_list = []
        G_list = []
        
        # ---> DATA-DRIVEN AIRGAP ENFORCEMENT <---
        # Instead of zeroing rotor_out (which fed the model an input it never
        # saw in training and caused unreliable, out-of-distribution predictions),
        # snap rotor_out into the airgap relationship actually observed in the
        # training data. Same function is used at export time, so what gets
        # predicted here is exactly the geometry that gets reported later.
        x = enforce_airgap(x, self.stator_in_idx, self.rotor_out_idx, self.airgap_info)

        cat_preds = self.cat_model.predict(x)
        xgb_preds = self.xgb_model.predict(x)
        
        if cat_preds.ndim == 1: cat_preds = cat_preds.reshape(1, -1)
        if xgb_preds.ndim == 1: xgb_preds = xgb_preds.reshape(1, -1)
        
        for k in self.active_keys:
            col_idx = TARGET_MAP[k]
            y_key = Y_KEYS[col_idx]
            
            y = cat_preds[:, col_idx] if self.winners[y_key] == "CatBoost" else xgb_preds[:, col_idx]
            
            f = np.abs(y - self.targets[k])
            F_list.append(f)
            
            g_lower = (self.targets[k] - self.config[k]) - y
            g_upper = y - (self.targets[k] + self.config[k])
            G_list.append(g_lower)
            G_list.append(g_upper)

        # ---> NON-CONVEX MANIFOLD FEASIBILITY CONSTRAINT <---
        # Beyond each objective's own bound, reject any candidate whose overall
        # variable COMBINATION (T1A, T2, TM1A, TM2A, VA, control angle, etc.)
        # sits too far from its nearest real, successful neighbors. This
        # follows the true (possibly curved/non-convex) shape of the feasible
        # design region, catching "every variable individually in range, but
        # this exact combination falls in a gap between real designs" cases
        # -- e.g. sketch self-intersections like LINE_P7A_PS3 / LINE_P4A_P2A.
        if self.manifold is not None:
            dist = manifold_distance(x, self.mf_mean, self.mf_std, self.mf_nn_index, self.mf_k)
            g_manifold = dist - self.mf_threshold
            G_list.append(g_manifold)

        # ---> RADIAL DEPTH BUDGET CONSTRAINT <---
        # Direct, explicit check for the mechanism behind 'Rotor Inner
        # diameter too large -> barriers don't fit -> lines intersect':
        # available radial depth = (Rotor Outer - Rotor Inner) / 2 must stay
        # large enough, RELATIVE TO the total size of the barrier dimensions
        # actually being requested, to match the tightest margin any real
        # design ever validated. Checked directly (not diffused across many
        # other dimensions the way the manifold distance is), so it catches
        # this specific failure even in higher-dimensional geometries (D03+)
        # where the manifold check alone can miss it.
        if self.radial_depth_budget is not None:
            available_depth = (x[:, self.rdb_outer_idx] - x[:, self.rdb_inner_idx]) / 2.0
            total_barrier = x[:, self.rdb_barrier_idx].sum(axis=1)
            g_radial = self.rdb_min_ratio * total_barrier - available_depth
            G_list.append(g_radial)

        out["F"] = np.column_stack(F_list)
        out["G"] = np.column_stack(G_list)


# ── 3. Helper Functions for Export & Plotting ────────────────────────────────

def flatten_designs(designs_list):
    rows = []
    for d in designs_list:
        row = {
            "Geometry ID": d["geometry_id"],
            "Mismatch Score (Lower is Better)": d["score"],
            "Torque (N.m)": d["predicted_outputs"]["torque"],
            "Efficiency (%)": d["predicted_outputs"]["efficiency"],
            "Power Factor": d["predicted_outputs"]["power_factor"],
            "Ripple (%)": d["predicted_outputs"]["ripple"],
        }
        
        for name, val in zip(d["var_names"], d["input_values"]):
            clean_name = clean_var_name(name)
            row[clean_name] = val
            
        row["Torque_Engine"] = d["winners"]["y1"]
        row["Eff_Engine"] = d["winners"]["y2"]
        row["PF_Engine"] = d["winners"]["y3"]
        row["Ripple_Engine"] = d["winners"]["y4"]
        
        rows.append(row)
    return pd.DataFrame(rows)


def save_pictorial_result(designs, targets, plot_path):
    if not designs: return
    
    labels = [f"#{i+1}\n{d['geometry_id']}" for i, d in enumerate(designs)]
    metrics = ["torque", "efficiency", "power_factor", "ripple"]
    titles = ["Torque (N.m)", "Efficiency (%)", "Power Factor", "Ripple (%)"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle("Top Optimized Motor Designs vs Targets", fontsize=16, fontweight="bold", y=0.95)
    
    for ax, metric, title, color in zip(axes.flatten(), metrics, titles, colors):
        values = [d["predicted_outputs"][metric] for d in designs]
        target_val = targets[metric]
        
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        
        ax.axhline(target_val, color="black", linestyle="--", linewidth=2, label=f"Target: {target_val}")
        
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis='y', alpha=0.3, color="#E5E5E5")
        ax.set_facecolor("#FFFFFF")
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (target_val*0.01), f"{yval:.2f}", 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    for spine in ax.spines.values(): spine.set_edgecolor("#E5E5E5")
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def save_pareto_scatter(strict_designs, soft_designs, targets, plot_path, tols=None):
    """
    Classic multi-objective trade-off plot: the best candidate from EVERY
    geometry evaluated, plotted as Torque vs Efficiency. Ripple is encoded
    as the marker FILL color, Power Factor is encoded as a separate marker
    RING color (its own colorbar) so both quantities can be read precisely
    instead of guessing relative bubble sizes. This is the standard way
    NSGA-II results get reported in engineering papers/project reports --
    it shows the actual trade-off frontier the optimizer explored across all
    geometries, not just the single final winner.

    Each point is numbered by its actual RANK -- how close it is to the
    target (distance normalized by tolerance, or by data spread if no
    tolerance is given) -- not by insertion order. Rank #1 is always the
    design closest to target. A ranking panel (closest -> farthest) replaces
    a plain geometry-name key so the numbers carry real meaning.

    If tols is provided, also shades the actual ACCEPTABLE tolerance band
    around the target (target +/- torque_tol, target +/- efficiency_tol).
    This is what 'close to target' actually means engineering-wise, and
    showing it directly is far more honest than just cropping the axes --
    cropping doesn't change the real distance between points, it only hides
    it.
    """
    pool = strict_designs if strict_designs else soft_designs
    if not pool:
        return

    best_per_geom = {}
    for d in pool:
        gid = d["geometry_id"]
        if gid not in best_per_geom or d["score"] < best_per_geom[gid]["score"]:
            best_per_geom[gid] = d
    points = list(best_per_geom.values())
    if not points:
        return

    torque = [d["predicted_outputs"]["torque"] for d in points]
    eff    = [d["predicted_outputs"]["efficiency"] for d in points]
    ripple = [d["predicted_outputs"]["ripple"] for d in points]
    pf     = [d["predicted_outputs"]["power_factor"] for d in points]
    labels = [d["geometry_id"] for d in points]

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#FFFFFF")

    has_band = (tols and "torque" in targets and "efficiency" in targets
                and "torque" in tols and "efficiency" in tols)
    has_target = "torque" in targets and "efficiency" in targets
    band_x = band_y = None
    if has_band:
        tx0, tx1 = targets["torque"] - tols["torque"], targets["torque"] + tols["torque"]
        ty0, ty1 = targets["efficiency"] - tols["efficiency"], targets["efficiency"] + tols["efficiency"]
        band_x, band_y = (tx0, tx1), (ty0, ty1)
        rect = plt.Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0,
                              facecolor="#FFD966", alpha=0.22, edgecolor="#BF9000",
                              linewidth=1.5, linestyle="--", zorder=1,
                              label=f"Acceptable range (±{tols['torque']} N.m, ±{tols['efficiency']}%)")
        ax.add_patch(rect)

    # ---> TRUE RANKING: rank by actual normalized distance to target, not
    # by whatever order the optimizer happened to emit results in. Distance
    # is normalized by the tolerance band when available (the engineering-
    # correct notion of "close"), falling back to the data spread otherwise. <---
    if has_target:
        tx, ty = targets["torque"], targets["efficiency"]
        norm_t = tols["torque"] if has_band else ((max(torque) - min(torque)) or 1.0)
        norm_e = tols["efficiency"] if has_band else ((max(eff) - min(eff)) or 1.0)
        dist = [((t - tx) / norm_t) ** 2 + ((e - ty) / norm_e) ** 2 for t, e in zip(torque, eff)]
    else:
        dist = [0.0] * len(torque)

    order = sorted(range(len(points)), key=lambda i: dist[i])
    rank_of = {idx: r for r, idx in enumerate(order, start=1)}

    # ---> DUAL ENCODING: fill color = Ripple, outer ring color = Power
    # Factor. Each is read off its own colorbar with exact values, instead
    # of inferring Power Factor from subtle differences in bubble size. <---
    base_size = 320
    pf_arr = np.array(pf, dtype=float)
    pf_lo, pf_hi = float(pf_arr.min()), float(pf_arr.max())
    pf_norm = plt.Normalize(vmin=pf_lo, vmax=(pf_hi if pf_hi > pf_lo else pf_lo + 1e-6))
    ring_cmap = plt.cm.Blues

    ax.scatter(torque, eff, s=base_size * 1.55, c=pf_arr, cmap=ring_cmap, norm=pf_norm,
               edgecolor="none", zorder=2, marker="o")
    sc = ax.scatter(torque, eff, c=ripple, cmap="RdYlGn_r", s=base_size,
                     edgecolor="black", linewidth=0.9, alpha=0.95, zorder=3)

    # Emphasize the closest-to-target design with a bold outline so it's
    # visually obvious without needing to cross-reference the panel.
    if has_target and order:
        best_idx = order[0]
        ax.scatter([torque[best_idx]], [eff[best_idx]], facecolors="none",
                   edgecolors="#1A1A1A", linewidth=2.4, s=base_size * 1.55 + 90, zorder=4)

    if has_target:
        ax.scatter([targets["torque"]], [targets["efficiency"]], marker="*",
                   s=650, color="gold", edgecolor="black", linewidth=1.3,
                   zorder=5, label="Target")

    # ---> READABILITY: rank number on each marker (white halo keeps it
    # legible against any fill color), plus a ranking panel instead of an
    # arbitrary geometry-name key -- closest to target listed first. <---
    for i, (x, y) in enumerate(zip(torque, eff)):
        txt = ax.annotate(str(rank_of[i]), (x, y), ha="center", va="center",
                           fontsize=8, fontweight="bold", color="black", zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

    panel_title = "Ranking (Closest \u2192 Farthest from Target)" if has_target else "Ranking"
    rank_lines = [f"{rank_of[idx]}.  {labels[idx]}" + ("  \u2605" if rank_of[idx] == 1 else "")
                  for idx in order]
    n = len(rank_lines)
    n_cols = 1 if n <= 9 else (2 if n <= 20 else 3)
    col_chunks = [rank_lines[i::n_cols] for i in range(n_cols)]
    rank_text = "\n".join(
        "    ".join(col[row] if row < len(col) else "" for col in col_chunks)
        for row in range(max(len(c) for c in col_chunks))
    )
    ax.text(0.015, 0.015, panel_title + "\n" + rank_text,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, family="monospace", color="#333333",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#CCCCCC", linewidth=0.9, alpha=0.92),
            zorder=7)

    ax.legend(loc="upper right", fontsize=9)

    ax.set_xlabel("Torque (N.m)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Efficiency (%)", fontweight="bold", fontsize=11)
    ax.set_title("Multi-Objective Trade-off — Best Design per Geometry\n"
                 "(fill = Ripple %, ring = Power Factor, number = Rank vs. Target)",
                 fontweight="bold", fontsize=13)

    # Fit the view to whatever is actually meaningful to show: the points,
    # the target, and the tolerance band if we have one -- with modest
    # padding, instead of letting an isolated outlier blow the frame open.
    if has_band:
        all_x = torque + [targets["torque"], band_x[0], band_x[1]]
        all_y = eff + [targets["efficiency"], band_y[0], band_y[1]]
    else:
        all_x = torque + ([targets["torque"]] if "torque" in targets else [])
        all_y = eff + ([targets["efficiency"]] if "efficiency" in targets else [])
    pad_x = (max(all_x) - min(all_x)) * 0.10 or 1.0
    pad_y = (max(all_y) - min(all_y)) * 0.10 or 1.0
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)

    ax.grid(alpha=0.3, color="#E5E5E5")
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values(): spine.set_edgecolor("#E5E5E5")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Ripple (%)", fontweight="bold")

    pf_sm = plt.cm.ScalarMappable(cmap=ring_cmap, norm=pf_norm)
    pf_sm.set_array([])
    cbar2 = fig.colorbar(pf_sm, ax=ax, pad=0.10)
    cbar2.set_label("Power Factor", fontweight="bold")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def save_decision_dashboard(strict_designs, soft_designs, targets, plot_path, tols=None, top_n=5):
    """
    A single-page "three-panel decision view" for project reports/presentations:
      A) Performance Space  -- Torque vs Efficiency scatter (color=Ripple, size=Power Factor)
      B) Ranking             -- horizontal bar chart of every geometry's Match Score
      C) Top-N Designs       -- a clean table of the final recommended designs

    All three panels rank by the EXACT SAME 'score' field (lower = better,
    summed normalized deviation across every active target) that the
    printed terminal report and Excel export use, so the #1 design here is
    always the same #1 shown everywhere else.
    """
    pool = strict_designs if strict_designs else soft_designs
    if not pool:
        return

    best_per_geom = {}
    for d in pool:
        gid = d["geometry_id"]
        if gid not in best_per_geom or d["score"] < best_per_geom[gid]["score"]:
            best_per_geom[gid] = d
    points = list(best_per_geom.values())
    if not points:
        return

    torque = [d["predicted_outputs"]["torque"] for d in points]
    eff    = [d["predicted_outputs"]["efficiency"] for d in points]
    ripple = [d["predicted_outputs"]["ripple"] for d in points]
    pf     = [d["predicted_outputs"]["power_factor"] for d in points]
    labels = [d["geometry_id"] for d in points]
    scores = [d["score"] for d in points]

    order = sorted(range(len(points)), key=lambda i: scores[i])
    rank_of = {idx: r for r, idx in enumerate(order, start=1)}
    top_n = min(top_n, len(points))

    NAVY = "#1F3A5F"
    CARD_EDGE = "#9DB8D8"
    LIGHTBLUE_BG = "#F4F8FC"
    FOOTER_BG = "#FAFAFA"

    fig = plt.figure(figsize=(20, 10.2))
    fig.patch.set_facecolor("white")

    outer = gridspec.GridSpec(3, 3, height_ratios=[5.0, 1.5, 0.7], hspace=0.45, wspace=0.34,
                               left=0.035, right=0.985, top=0.86, bottom=0.045)

    fig.text(0.5, 0.965, "Multi-Objective Optimization of SynRM Geometries \u2014 Three-Panel Decision View",
              ha="center", fontsize=19, fontweight="bold", color="#1A1A1A")
    target_bits = []
    if "torque" in targets: target_bits.append(f"{targets['torque']} N\u00b7m Torque")
    if "efficiency" in targets: target_bits.append(f"{targets['efficiency']}% Efficiency")
    if "power_factor" in targets: target_bits.append(f"{targets['power_factor']} Power Factor")
    if "ripple" in targets: target_bits.append(f"{targets['ripple']}% Ripple")
    fig.text(0.5, 0.925, "Target: " + ", ".join(target_bits), ha="center", fontsize=12.5, color="#555555")

    # ---------------- PANEL A: Performance Space ----------------
    col0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], width_ratios=[26, 1], wspace=0.10)
    axA = fig.add_subplot(col0[0, 0])
    caxA = fig.add_subplot(col0[0, 1])

    has_band = (tols and "torque" in targets and "efficiency" in targets
                and "torque" in tols and "efficiency" in tols)
    if has_band:
        tx0, tx1 = targets["torque"] - tols["torque"], targets["torque"] + tols["torque"]
        ty0, ty1 = targets["efficiency"] - tols["efficiency"], targets["efficiency"] + tols["efficiency"]
        axA.add_patch(plt.Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0, facecolor="#FFD966", alpha=0.18,
                                     edgecolor="#BF9000", linewidth=1.3, linestyle="--", zorder=1,
                                     label=f"Acceptable range\n(\u00b1{tols['torque']} N.m, \u00b1{tols['efficiency']}%)"))

    pf_arr = np.array(pf, dtype=float)
    pf_span = (pf_arr.max() - pf_arr.min()) or 1e-6
    sizes = 90 + 480 * (pf_arr - pf_arr.min()) / pf_span
    sc = axA.scatter(torque, eff, c=ripple, cmap="RdYlGn_r", s=sizes, edgecolor="black",
                      linewidth=0.8, zorder=3, alpha=0.92)
    if "torque" in targets and "efficiency" in targets:
        axA.scatter([targets["torque"]], [targets["efficiency"]], marker="*", s=420, color="gold",
                    edgecolor="black", linewidth=1.1, zorder=5, label="Target")
    axA.set_xlabel("Torque (N\u00b7m)", fontweight="bold", fontsize=10.5)
    axA.set_ylabel("Efficiency (%)", fontweight="bold", fontsize=10.5)
    axA.grid(alpha=0.3, color="#E5E5E5")
    for sp in axA.spines.values(): sp.set_edgecolor("#DDDDDD")
    axA.legend(loc="lower left", fontsize=8, framealpha=0.95)

    cbar = fig.colorbar(sc, cax=caxA)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.set_label("Ripple (%)", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    # ---------------- PANEL B: Ranking bar chart ----------------
    axB = fig.add_subplot(outer[0, 1])
    ranked_labels = [labels[idx] for idx in order]
    ranked_scores = [scores[idx] for idx in order]
    max_score = max(ranked_scores) or 1e-6
    colors = plt.cm.RdYlGn_r(np.array(ranked_scores) / max_score)
    y_pos = np.arange(len(order))[::-1]
    axB.barh(y_pos, ranked_scores, color=colors, edgecolor="black", linewidth=0.5, height=0.68)
    axB.set_yticks(y_pos)
    axB.set_yticklabels([f"#{r+1}  {ranked_labels[r]}" for r in range(len(order))], fontsize=8.5)
    axB.set_xlabel("Match Score (Lower is Better)", fontweight="bold", fontsize=10.5)
    axB.grid(alpha=0.3, axis="x", color="#E5E5E5")
    for sp in axB.spines.values(): sp.set_edgecolor("#DDDDDD")
    for i, v in enumerate(ranked_scores):
        axB.text(v + max_score * 0.018, y_pos[i], f"{v:.3f}", va="center", fontsize=8.3)
    axB.set_xlim(0, max_score * 1.18)

    # ---------------- PANEL C: Top-N table ----------------
    axC = fig.add_subplot(outer[0, 2])
    axC.axis("off")
    top_idx = order[:top_n]
    table_data = [[f"{rank_of[idx]}", labels[idx], f"{torque[idx]:.1f}", f"{eff[idx]:.1f}",
                   f"{pf[idx]:.3f}", f"{ripple[idx]:.1f}", f"{scores[idx]:.3f}"] for idx in top_idx]
    col_labels = ["Rank", "Geometry", "Torque\n(N.m)", "Eff.\n(%)", "PF", "Ripple\n(%)", "Match\nScore"]
    col_widths = [0.10, 0.30, 0.14, 0.12, 0.11, 0.12, 0.15]
    tbl = axC.table(cellText=table_data, colLabels=col_labels, loc="center",
                     cellLoc="center", colWidths=col_widths, bbox=[0, 0.06, 1, 0.88])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.3)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold", fontsize=9.3)
        else:
            cell.set_facecolor("#E8F3E3" if row == 1 else ("#F7F7F7" if row % 2 == 0 else "white"))
            if col == 1:
                cell.set_text_props(ha="left")
            if col == 6:
                cell.set_text_props(fontweight="bold")
    tbl.scale(1, 1.65)

    # ================= Card frames + headers =================
    def union_bbox(*axes):
        b = axes[0].get_position()
        for a in axes[1:]:
            b = Bbox.union([b, a.get_position()])
        return b

    def add_card_frame(fig, bbox, label, title, header_gap=0.048):
        pad = 0.004
        rect = plt.Rectangle((bbox.x0 - pad, bbox.y0 - pad), bbox.width + 2 * pad,
                              bbox.height + 2 * pad + header_gap,
                              transform=fig.transFigure, facecolor="white", edgecolor=CARD_EDGE,
                              linewidth=1.3, zorder=-1, clip_on=False)
        fig.add_artist(rect)
        badge_size = 0.020
        badge_x = bbox.x0 + 0.010
        badge_y = bbox.y0 + bbox.height + header_gap - 0.010 - badge_size
        badge = plt.Rectangle((badge_x, badge_y), badge_size, badge_size, transform=fig.transFigure,
                               facecolor=NAVY, edgecolor="none", zorder=1, clip_on=False)
        fig.add_artist(badge)
        fig.text(badge_x + badge_size / 2, badge_y + badge_size / 2, label, ha="center", va="center",
                  fontsize=11.5, fontweight="bold", color="white", zorder=2)
        fig.text(badge_x + badge_size + 0.010, badge_y + badge_size / 2, title, ha="left", va="center",
                  fontsize=12.5, fontweight="bold", color=NAVY, zorder=2)

    add_card_frame(fig, union_bbox(axA, caxA), "A", "Performance Space: Torque vs Efficiency")
    add_card_frame(fig, axB.get_position(), "B", "Ranking: Match Score (Lower is Better)")
    add_card_frame(fig, axC.get_position(), "C", f"Top {top_n} Designs (Final Selection)")

    # ================= "How to read" row =================
    def style_how_to_read(ax, bullet_color, text):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(LIGHTBLUE_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(CARD_EDGE); sp.set_linewidth(1.1)
        ax.text(0.045, 0.74, "\u25CF", color=bullet_color, fontsize=13, ha="center", va="center", transform=ax.transAxes)
        ax.text(0.085, 0.74, "How to read:", fontsize=10.5, fontweight="bold", color=NAVY,
                ha="left", va="center", transform=ax.transAxes)
        ax.text(0.045, 0.34, text, fontsize=9.2, color="#333333", ha="left", va="center",
                transform=ax.transAxes, linespacing=1.5)

    axHA = fig.add_subplot(outer[1, 0])
    style_how_to_read(axHA, "#2E7D32",
        "Points closer to the target star and greener in color\n(lower ripple) with larger size (higher power factor)\nare preferred.")
    axHB = fig.add_subplot(outer[1, 1])
    style_how_to_read(axHB, "#1565C0",
        "Lower match score means closer to the target across\nall objectives (Torque, Efficiency, Power Factor, Ripple).")
    axHC = fig.add_subplot(outer[1, 2])
    style_how_to_read(axHC, "#B8860B",
        f"These top {top_n} designs provide the best overall balance\nbetween torque, efficiency, power factor, and ripple.")

    # ================= Footer targets bar =================
    axF = fig.add_subplot(outer[2, :])
    axF.set_xticks([]); axF.set_yticks([])
    axF.set_facecolor(FOOTER_BG)
    for sp in axF.spines.values():
        sp.set_edgecolor("#DDDDDD"); sp.set_linewidth(1.0)
    axF.text(0.012, 0.5, "\u25A0 Targets:", fontsize=10.5, fontweight="bold", color=NAVY,
              ha="left", va="center", transform=axF.transAxes)
    axF.text(0.075, 0.5, "   |   ".join(target_bits),
              fontsize=10, va="center", ha="left", color="#333333", transform=axF.transAxes)
    if has_band:
        axF.text(0.99, 0.5,
                  f"\u25A0 Acceptable Range: \u00b1{tols['torque']} N\u00b7m Torque, \u00b1{tols['efficiency']}% Efficiency",
                  fontsize=10, va="center", ha="right", color="#333333", fontweight="bold", transform=axF.transAxes)

    fig.savefig(plot_path, dpi=200, facecolor="white")
    plt.close(fig)


def save_radar_chart(top_unique, targets, plot_path):
    """
    Spider/radar chart comparing the top designs across all active metrics
    at once, each normalized against the user's target (1.0 = exactly on
    target, the dashed black ring). Lets a reviewer see at a glance which
    design is a balanced all-rounder vs which one over/under-shoots a
    specific metric -- a single bar chart can't show that relationship.
    """
    if not top_unique:
        return

    metrics = [k for k in ["torque", "efficiency", "power_factor", "ripple"] if k in targets]
    if len(metrics) < 3:
        return  # a radar chart needs >= 3 axes to be meaningful

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#FFFFFF")
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_unique)))

    all_vals = []
    for d in top_unique:
        all_vals += [d["predicted_outputs"][m] / targets[m] if targets[m] else 1.0 for m in metrics]

    for i, (d, color) in enumerate(zip(top_unique, colors)):
        vals = [d["predicted_outputs"][m] / targets[m] if targets[m] else 1.0 for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=f"#{i+1} {d['geometry_id']}")
        ax.fill(angles, vals, color=color, alpha=0.07)

    ax.plot(angles, [1.0] * (n + 1), color="black", linewidth=2, linestyle="--", label="Target (=1.0)")

    # Designs near a target naturally cluster close to 1.0 -- zoom the radial
    # axis to that neighborhood instead of defaulting to 0, or the real
    # differences between designs get visually compressed to nothing.
    spread = max(0.03, max(abs(v - 1.0) for v in all_vals) * 1.4)
    ax.set_ylim(1.0 - spread, 1.0 + spread)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=11, fontweight="bold")
    ax.set_title("Top Designs vs Target (Normalized)", fontweight="bold", pad=24, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.12), fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def save_geometry_schematics(top_unique, plot_path, max_designs=6):
    """
    Simplified to-scale radial cross-section schematic for each top design:
    concentric circles for Stator Outer/Inner and Rotor Outer/Inner
    diameters. Doesn't attempt to reconstruct exact flux-barrier shapes
    (those differ per rotor topology, D0x vs C3_A0x), but gives an instant
    visual sense of the overall machine envelope and proportions for a
    report/presentation slide, without needing Motor-CAD open.
    """
    designs = top_unique[:max_designs]
    if not designs:
        return

    n = len(designs)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.patch.set_facecolor("#FFFFFF")
    axes = np.atleast_1d(axes).flatten()

    for i, (ax, d) in enumerate(zip(axes, designs)):
        names = d["var_names"]
        vals = d["input_values"]

        def find(*keys):
            for nm, v in zip(names, vals):
                if all(k.lower() in str(nm).lower() for k in keys):
                    return v
            return None

        st_out = find("stator", "outer")
        st_in = find("stator", "inner")
        ro_out = find("rotor", "outer")
        ro_in = find("rotor", "inner")

        if None in (st_out, st_in, ro_out, ro_in):
            ax.axis("off")
            continue

        for d_val, color, label in [
            (st_out, "#4C72B0", "Stator Outer"),
            (st_in, "#DD8452", "Stator Inner"),
            (ro_out, "#55A868", "Rotor Outer"),
            (ro_in, "#C44E52", "Rotor Inner"),
        ]:
            circ = plt.Circle((0, 0), d_val / 2, fill=False, edgecolor=color,
                               linewidth=2.2, label=f"{label}: {d_val:.1f}mm")
            ax.add_patch(circ)

        lim = st_out / 2 * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"#{i + 1}  {d['geometry_id']}", fontweight="bold", fontsize=10)
        ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9)
        ax.axis("off")

    for ax in axes[len(designs):]:
        ax.axis("off")

    fig.suptitle("Motor Cross-Section Envelope (to scale)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


# ── 4. Run Optimization ──────────────────────────────────────────────────────

def optimize_geometry(gid, models_dir, res_dir, targets, tols, n_designs, pop_size, n_gen, airgap_range=None):
    bounds_file = models_dir / f"{gid}_bounds.json"
    cat_path    = models_dir / f"{gid}_catboost.cbm"
    xgb_path    = models_dir / f"{gid}_xgboost.pkl"
    metric_path = res_dir / "training_metrics.json"

    if not all(p.exists() for p in [bounds_file, cat_path, xgb_path, metric_path]):
        return None

    with open(bounds_file, 'r') as f: bounds = json.load(f)
    with open(metric_path, 'r') as f: all_metrics = json.load(f)
    
    # --- BULLETPROOF BOUNDS FIX (Prevents the TypeError Crash) ---
    if isinstance(bounds.get('lb'), dict):
        bounds['lb'] = [float(bounds['lb'][k]) for k in sorted(bounds['lb'].keys(), key=lambda x: int(x))]
    if isinstance(bounds.get('ub'), dict):
        bounds['ub'] = [float(bounds['ub'][k]) for k in sorted(bounds['ub'].keys(), key=lambda x: int(x))]
    
    # ---> ADDED: ENHANCED BOUNDS UNDERSTANDING & VALIDATION <---
    # 1. Fallback to list and enforce float types in case JSON provided raw lists with strings/ints
    if isinstance(bounds.get('lb'), list):
        bounds['lb'] = [float(val) for val in bounds['lb']]
    if isinstance(bounds.get('ub'), list):
        bounds['ub'] = [float(val) for val in bounds['ub']]

    # 2. Safety check: Dimension mismatch prevents Pymoo Initialization crash
    if len(bounds.get('lb', [])) != len(bounds.get('ub', [])) or len(bounds.get('lb', [])) == 0:
        print(f"\n[!] SKIP: Bounds missing or dimension mismatch for {gid}.")
        return None

    # 3. Safety check: Prevent inverted bounds (lb > ub) and identical bounds (lb == ub)
    for b_idx in range(len(bounds['lb'])):
        if bounds['lb'][b_idx] > bounds['ub'][b_idx]:
            # Swap inverted bounds silently
            bounds['lb'][b_idx], bounds['ub'][b_idx] = bounds['ub'][b_idx], bounds['lb'][b_idx]
        if bounds['lb'][b_idx] == bounds['ub'][b_idx]:
            # Provide a microscopic search space so Pymoo doesn't crash on identical bounds
            bounds['ub'][b_idx] += 1e-6  
    # -----------------------------------------------------------

    var_names = bounds.get('var_names', [f"x{i+1}" for i in range(len(bounds['lb']))])
    
    stator_in_idx = -1
    rotor_out_idx = -1
    for i, n in enumerate(var_names):
        if "Stator::Inner" in str(n): stator_in_idx = i
        elif "Rotor::Outer" in str(n): rotor_out_idx = i
    
    geom_metrics = next((m for m in all_metrics if m["geometry_id"] == gid), None)
    if not geom_metrics: return None

    winners = {}
    for yk in ["y1", "y2", "y3", "y4"]:
        cat_r2 = geom_metrics["outputs"][yk]["cat_r2"]
        xgb_r2 = geom_metrics["outputs"][yk]["xgb_r2"]
        winners[yk] = "CatBoost" if cat_r2 >= xgb_r2 else "XGBoost"

    cat_model = CatBoostRegressor()
    cat_model.load_model(str(cat_path))
    xgb_model = joblib.load(str(xgb_path))

    # ---> DATA-DRIVEN AIRGAP <---
    # Read whatever train.py actually measured in the data for this geometry.
    # If this bounds.json predates the train.py update (no 'airgap_info' key),
    # fall back to today's known fixed 0.7mm radial gap so old runs don't break.
    airgap_info = bounds.get('airgap_info')
    if airgap_info is None and stator_in_idx != -1 and rotor_out_idx != -1:
        airgap_info = {
            'is_fixed': True,
            'observed_mean_diametric': 1.4,
            'observed_min_diametric': 1.4,
            'observed_max_diametric': 1.4,
        }

    # ---> USER-DEFINED MANUFACTURABLE AIRGAP RANGE <---
    # --airgap_min/--airgap_max let the user narrow (or shift) the gap NSGA-II
    # is allowed to choose, to whatever range their manufacturing process can
    # actually hold -- without ever letting it explore a gap the AI was never
    # trained on. Only meaningful when this geometry's data spans a real range
    # (is_fixed=False); a fixed-gap geometry only has one number to give.
    if airgap_range is not None:
        req_min, req_max = airgap_range
        if airgap_info is None:
            print(f"\n[!] WARNING ({gid}): no airgap relationship found in bounds.json — "
                  f"--airgap_min/--airgap_max ignored.")
        elif airgap_info['is_fixed']:
            print(f"\n[!] WARNING ({gid}): training data only contains a single observed airgap "
                  f"({airgap_info['observed_mean_diametric']:.3f}mm diametric); the AI has never seen "
                  f"a range, so --airgap_min/--airgap_max can't be honored without retraining on varied-gap "
                  f"data. Using the fixed {airgap_info['observed_mean_diametric']:.3f}mm gap instead.")
        else:
            obs_min = airgap_info['observed_min_diametric']
            obs_max = airgap_info['observed_max_diametric']
            eff_min = max(req_min, obs_min)
            eff_max = min(req_max, obs_max)
            if eff_min > eff_max:
                print(f"\n[!] WARNING ({gid}): requested airgap range [{req_min}, {req_max}]mm doesn't "
                      f"overlap what the AI was trained on ([{obs_min:.3f}, {obs_max:.3f}]mm). "
                      f"Falling back to the full observed range.")
            else:
                if eff_min > req_min or eff_max < req_max:
                    print(f"\n[i] ({gid}): clipping requested airgap range [{req_min}, {req_max}]mm to what "
                          f"the AI has actually seen -> [{eff_min:.3f}, {eff_max:.3f}]mm diametric.")
                airgap_info = dict(airgap_info)
                airgap_info['observed_min_diametric'] = eff_min
                airgap_info['observed_max_diametric'] = eff_max

    # ---> NON-CONVEX MANIFOLD FEASIBILITY CHECK <---
    # Present for any bounds.json regenerated with the updated train.py.
    # If missing (old bounds.json), manifold stays None and the optimizer
    # simply falls back to per-axis box bounds only -- no crash, just no
    # extra joint-feasibility protection until you retrain.
    manifold = bounds.get('manifold')

    # ---> RADIAL DEPTH BUDGET <---
    # Present for any bounds.json regenerated with the updated train.py.
    # If missing, this stays None and the optimizer simply doesn't add this
    # extra constraint -- no crash, just no protection until you retrain.
    radial_depth_budget = bounds.get('radial_depth_budget')

    problem = HybridMotorProblemVec(cat_model, xgb_model, winners, bounds, targets, tols, var_names, airgap_info, manifold, radial_depth_budget)
    algorithm = NSGA2(pop_size=pop_size)

    res = minimize(problem, algorithm, ('n_gen', n_gen), seed=42, verbose=False)

    valid_designs, soft_designs = [], []
    
    if res.X is not None:
        X_val = np.atleast_2d(res.X)
        F_val = np.atleast_2d(res.F)
        CV    = np.atleast_2d(res.CV) if res.CV is not None else np.zeros((len(X_val), 1))

        feasible = (CV <= 0).flatten()
        targets_arr = np.array([targets[k] for k in problem.active_keys])
        
        if np.any(feasible):
            X_feas, F_feas = X_val[feasible], F_val[feasible]
            I = NonDominatedSorting().do(F_feas, only_non_dominated_front=True)
            X_pareto, F_pareto = X_feas[I], F_feas[I]
            scores = np.sum(F_pareto / targets_arr, axis=1)
            best_idx = np.argsort(scores)[:n_designs]
            design_list = valid_designs
            source_X = X_pareto
        else:
            scores = np.sum(F_val / targets_arr, axis=1)
            best_idx = [np.argsort(scores)[0]]
            design_list = soft_designs
            source_X = X_val

        for idx in best_idx:
            x_dsg = source_X[idx].copy()

            # ---> DATA-DRIVEN AIRGAP, ENFORCED ONCE <---
            # Same enforcement used inside NSGA-II's _evaluate. Doing it BEFORE
            # prediction (and not changing it again afterward) guarantees the
            # torque/efficiency/etc. reported below are for the exact geometry
            # you get in the export -- previously this predicted with rotor_out
            # forced to 0.0, then silently swapped in the real value afterward,
            # so the reported numbers never matched the exported geometry.
            x_dsg = enforce_airgap(x_dsg.reshape(1, -1), stator_in_idx, rotor_out_idx, airgap_info).flatten()

            c_pred = cat_model.predict(x_dsg.reshape(1, -1))[0]
            x_pred = xgb_model.predict(x_dsg.reshape(1, -1))[0]

            design_list.append({
                "var_names": var_names,
                "input_values": x_dsg.tolist(),
                "predicted_outputs": {
                    "torque":       float(c_pred[0] if winners["y1"] == "CatBoost" else x_pred[0]),
                    "efficiency":   float(c_pred[1] if winners["y2"] == "CatBoost" else x_pred[1]),
                    "power_factor": float(c_pred[2] if winners["y3"] == "CatBoost" else x_pred[2]),
                    "ripple":       float(c_pred[3] if winners["y4"] == "CatBoost" else x_pred[3])
                },
                "score": float(scores[idx]),
                "winners": winners
            })
            
    return {"strict": valid_designs, "soft": soft_designs}

# ── 5. CLI setup ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NSGA-II Motor Geometry Optimizer (Strict 0.7mm Airgap Edition)")
    
    parser.add_argument("--torque",          type=float, default=None,  help="Target torque (N.m)")
    parser.add_argument("--efficiency",      type=float, default=None,  help="Target efficiency (%%)")
    parser.add_argument("--power_factor",    type=float, default=None,  help="Target power factor")
    parser.add_argument("--ripple",          type=float, default=None,  help="Target ripple (%%)")
    
    parser.add_argument("--torque_tol",      type=float, default=2.0,   help="Strict torque tolerance")
    parser.add_argument("--efficiency_tol",  type=float, default=0.5,   help="Strict efficiency tolerance")
    parser.add_argument("--power_factor_tol",type=float, default=0.02,  help="Strict PF tolerance")
    parser.add_argument("--ripple_tol",      type=float, default=3.0,   help="Strict ripple tolerance")
    
    parser.add_argument("--airgap_min",      type=float, default=None,  help="Minimum DIAMETRIC airgap (mm) you can manufacture. Only honored for geometries whose training data spans a real airgap range (is_fixed=False); clipped to what the AI has actually seen.")
    parser.add_argument("--airgap_max",      type=float, default=None,  help="Maximum DIAMETRIC airgap (mm) you can manufacture. Must be used together with --airgap_min.")

    parser.add_argument("--top_n",           type=int,   default=5,     help="Number of final geometries to output")
    parser.add_argument("--n_designs",       type=int,   default=10,     help="Designs extracted per scenario")
    parser.add_argument("--geometry",        type=str,   default=None,   help="Restrict to one geometry ID")
    parser.add_argument("--pop_size",        type=int,   default=200,    help="NSGA-II population size")
    parser.add_argument("--n_gen",           type=int,   default=250,    help="NSGA-II generations")
    parser.add_argument("--fast",            action="store_true",        help="Run with pop=60, gen=80 for rapid previews")

    args = parser.parse_args()

    if not any(v is not None for v in [args.torque, args.efficiency, args.power_factor, args.ripple]):
        print("\n[!] ERROR: You must specify at least one target!\n")
        return

    if args.fast:
        args.pop_size = 60
        args.n_gen = 80

    if (args.airgap_min is None) != (args.airgap_max is None):
        print("\n[!] ERROR: --airgap_min and --airgap_max must be given together.\n")
        return
    if args.airgap_min is not None and args.airgap_min > args.airgap_max:
        print("\n[!] ERROR: --airgap_min cannot be greater than --airgap_max.\n")
        return

    airgap_range = (args.airgap_min, args.airgap_max) if args.airgap_min is not None else None

    base = Path(__file__).parent
    model_dir = base / "models"
    res_dir   = base / "results"
    
    out_dir   = res_dir / "designs"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = {}
    tols = {}
    
    if args.torque is not None:       
        targets["torque"] = args.torque
        tols["torque"] = args.torque_tol
    if args.efficiency is not None:   
        targets["efficiency"] = args.efficiency
        tols["efficiency"] = args.efficiency_tol
    if args.power_factor is not None: 
        targets["power_factor"] = args.power_factor
        tols["power_factor"] = args.power_factor_tol
    if args.ripple is not None:       
        targets["ripple"] = args.ripple
        tols["ripple"] = args.ripple_tol

    if args.geometry:
        geoms = [model_dir / f"{args.geometry}_bounds.json"]
    else:
        geoms = list(model_dir.glob("*_bounds.json"))

    print("=" * 60)
    print("  MOTOR GEOMETRY — HYBRID NSGA-II OPTIMIZER")
    print("=" * 60)
    
    target_strs = [f"{k.capitalize()}={v}" for k, v in targets.items()]
    tol_strs = [f"±{v}" for v in tols.values()]
    
    print(f"\nTargets   : {', '.join(target_strs)}")
    print(f"Tolerances: {', '.join(tol_strs)}")
    print(f"Geometries to evaluate: {len(geoms)}\n")

    strict_designs = []
    soft_designs = []

    for i, bounds_file in enumerate(geoms, 1):
        gid = bounds_file.stem.replace("_bounds", "")
        print(f"\r  [Optimizing geometry {i:>2}/{len(geoms)}] : {gid} ...", end="", flush=True)
        
        results = optimize_geometry(
            gid, model_dir, res_dir, targets, tols,
            args.n_designs, args.pop_size, args.n_gen, airgap_range
        )

        if not results: continue

        for d in results["strict"]:
            strict_designs.append({"geometry_id": gid, **d})
            
        for d in results["soft"]:
            soft_designs.append({"geometry_id": gid, **d})

    print("\n\n" + "=" * 80)
    
    # ── SELECT TOP UNIQUE DESIGNS ──
    top_unique = []
    
    if strict_designs:
        print(f"  🏆 TOP {args.top_n} GEOMETRIES (STRICTLY WITHIN USER BOUNDS)")
        strict_designs.sort(key=lambda x: x["score"])
        seen = set()
        for d in strict_designs:
            if d["geometry_id"] not in seen:
                seen.add(d["geometry_id"])
                top_unique.append(d)
                if len(top_unique) == args.top_n: break
    else:
        print("  ⚠️ ZERO STRICT MATCHES FOUND. DISPLAYING CLOSEST ALTERNATIVES.")
        soft_designs.sort(key=lambda x: x["score"])
        seen = set()
        for d in soft_designs:
            if d["geometry_id"] not in seen:
                seen.add(d["geometry_id"])
                top_unique.append(d)
                if len(top_unique) == args.top_n: break

    print("=" * 80)

    # ── TERMINAL PRINTING (PERFECTLY ALIGNED FORMAT) ──
    for idx, d in enumerate(top_unique, 1):
        print(f"\n  #{idx} | {d['geometry_id']}")
        print("-" * 80)
        o = d['predicted_outputs']
        w = d['winners']
        print(f"  PERFORMANCE : Torque = {o['torque']:.2f} N.m  |  Eff = {o['efficiency']:.2f}%  |  PF = {o['power_factor']:.3f}  |  Ripple = {o['ripple']:.2f}%")
        
        print("\n  DIMENSIONS:")
        
        clean_names = [clean_var_name(name) for name in d['var_names']]
        max_vlen = max((len(name) for name in clean_names), default=3)
        
        for v_name, v_val in zip(clean_names, d['input_values']):
            print(f"         {v_name:<{max_vlen}} = {v_val:.4f}")
            
        print(f"\n  MODELS USED : Torque({w['y1'][:3]}), Eff({w['y2'][:3]}), PF({w['y3'][:3]}), Ripple({w['y4'][:3]})")

    # ── EXPORT TO EXCEL & PLOT ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    excel_path = out_dir / f"Optimization_Report_{timestamp}.xlsx"
    excel_saved = False

    if strict_designs or soft_designs:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            if strict_designs:
                df_strict = flatten_designs(strict_designs)
                df_strict.to_excel(writer, index=False, sheet_name="Strict Matches")
            if soft_designs:
                df_soft = flatten_designs(soft_designs)
                df_soft.to_excel(writer, index=False, sheet_name="Closest Alternatives")
        excel_saved = True

    plot_path = out_dir / f"Optimization_Dashboard_{timestamp}.png"
    dashboard_saved = False
    if top_unique:
        save_pictorial_result(top_unique, targets, plot_path)
        dashboard_saved = True

    # ── NEW PICTORIAL RESULTS (for project reports/presentations) ──
    pareto_path = out_dir / f"Optimization_ParetoTradeoff_{timestamp}.png"
    pareto_saved = False
    try:
        if strict_designs or soft_designs:
            save_pareto_scatter(strict_designs, soft_designs, targets, pareto_path, tols)
            pareto_saved = pareto_path.exists()
    except Exception as e:
        print(f"  [!] Could not generate Pareto trade-off plot: {e}")

    radar_path = out_dir / f"Optimization_RadarComparison_{timestamp}.png"
    radar_saved = False
    try:
        if top_unique:
            save_radar_chart(top_unique, targets, radar_path)
            radar_saved = radar_path.exists()
    except Exception as e:
        print(f"  [!] Could not generate radar comparison chart: {e}")

    decision_path = out_dir / f"Optimization_DecisionDashboard_{timestamp}.png"
    decision_saved = False
    try:
        if strict_designs or soft_designs:
            save_decision_dashboard(strict_designs, soft_designs, targets, decision_path, tols, top_n=args.top_n)
            decision_saved = decision_path.exists()
    except Exception as e:
        print(f"  [!] Could not generate three-panel decision dashboard: {e}")

    schematic_path = out_dir / f"Optimization_CrossSection_{timestamp}.png"
    schematic_saved = False
    try:
        if top_unique:
            save_geometry_schematics(top_unique, schematic_path)
            schematic_saved = schematic_path.exists()
    except Exception as e:
        print(f"  [!] Could not generate geometry cross-section schematic: {e}")

    if excel_saved or dashboard_saved or pareto_saved or radar_saved or decision_saved or schematic_saved:
        print("\n" + "=" * 80)
        if excel_saved: print(f"  📁 SAVED EXCEL REPORT: {excel_path}")
        if dashboard_saved: print(f"  📊 SAVED VISUAL DASHBOARD: {plot_path}")
        if pareto_saved: print(f"  📈 SAVED PARETO TRADE-OFF PLOT: {pareto_path}")
        if radar_saved: print(f"  🕸️  SAVED RADAR COMPARISON CHART: {radar_path}")
        if decision_saved: print(f"  🗳️  SAVED DECISION DASHBOARD: {decision_path}")
        if schematic_saved: print(f"  ⚙️  SAVED CROSS-SECTION SCHEMATIC: {schematic_path}")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()