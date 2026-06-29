"""
train.py
The Ultimate Engineering Showdown: XGBoost vs CatBoost
Includes Global Scatter Plots, Side-by-Side Heatmaps, Local 1x4 Diagnostic Scatters,
Grouped Feature Importance Bar Charts, Pearson Correlation Matrices, AND automatically generates the required _bounds.json files.
"""

import argparse
import json
import warnings
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib

import xgboost as xgb
from catboost import CatBoostRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.data_parser import load_all_data_files

warnings.filterwarnings("ignore")

# ── styling ──────────────────────────────────────────────────────────────────
BG     = "#FFFFFF"
XGB_C  = "#1f77b4"  
CAT_C  = "#ff7f0e"  
TEXT   = "#1A1A1A"
GRID   = "#E5E5E5"

plt.rcParams.update({
    "figure.facecolor"    : BG,
    "axes.facecolor"      : BG,
    "axes.edgecolor"      : GRID,
    "axes.labelcolor"     : TEXT,
    "xtick.color"         : TEXT,
    "ytick.color"         : TEXT,
    "text.color"          : TEXT,
    "grid.color"          : GRID,
    "grid.alpha"          : 0.7,
    "font.family"         : "DejaVu Sans",
    "axes.titlesize"      : 10,
    "axes.labelsize"      : 8,
})

OUTPUT_NAMES = {
    "y1": "Torque (N.m)",
    "y2": "Efficiency (%)",
    "y3": "Power Factor",
    "y4": "Torque Ripple (%)",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def _detect_airgap_info(X: pd.DataFrame, fixed_tol: float = 0.05) -> dict | None:
    """
    Measures the ACTUAL airgap present in the training data (diametric:
    Stator::Inner - Rotor::Outer) instead of assuming a hardcoded value.

    - If every sample in the data shows ~the same gap (spread < fixed_tol mm),
      the AI has only ever seen that one airgap -> mark it 'fixed' so the
      optimizer locks onto the measured value (auto-adapts if you retrain
      with a different fixed gap later, e.g. 0.6mm instead of 0.7mm).
    - If the data spans a real range of gaps, mark it 'variable' and record
      the observed min/max so the optimizer can search within that range
      instead of extrapolating beyond what the AI was actually trained on.
    """
    stator_col = next((c for c in X.columns if "Stator::Inner" in str(c) or "Stator Inner" in str(c)), None)
    rotor_col  = next((c for c in X.columns if "Rotor::Outer" in str(c) or "Rotor Outer" in str(c)), None)
    if stator_col is None or rotor_col is None:
        return None

    diametric_gap = (X[stator_col] - X[rotor_col]).astype(float)
    g_min, g_max = float(diametric_gap.min()), float(diametric_gap.max())
    g_mean = float(diametric_gap.mean())
    is_fixed = (g_max - g_min) < fixed_tol

    return {
        'stator_col': stator_col,
        'rotor_col': rotor_col,
        'is_fixed': bool(is_fixed),
        'observed_min_diametric': g_min,
        'observed_max_diametric': g_max,
        'observed_mean_diametric': g_mean,
        'observed_min_radial': g_min / 2.0,
        'observed_max_radial': g_max / 2.0,
        'observed_mean_radial': g_mean / 2.0,
    }


def _fit_manifold(X: pd.DataFrame, k: int = 5, margin: float = 1.25) -> dict:
    """
    Learns the actual SHAPE of the feasible design region, not an averaged
    approximation of it.

    Rotor/barrier geometry constraints (TM1A, TM2A, T1A, T2, VA, Control angle,
    etc.) mix lengths and angles -- the real feasible region (e.g. "do these
    sketch lines avoid crossing") is typically curved and often non-convex.
    A covariance ellipsoid is a CONVEX shape, so it can include points that
    sit inside the ellipsoid but in a gap between two real clusters of valid
    designs -- a combination that looks statistically normal but was never
    actually buildable.

    Instead, this checks each candidate against the distance to its actual
    nearest real neighbors (k-NN) in standardized variable space. That
    distance follows the true manifold's curvature rather than smoothing
    over it, so it correctly rejects "looks fine on average, never actually
    seen" combinations -- without needing to know the underlying sketch
    geometry formulas at all.
    """
    vals = X.values.astype(float)
    n = len(vals)
    k_eff = max(1, min(k, n - 1))

    scaler = StandardScaler().fit(vals)
    vals_std = scaler.transform(vals)

    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(vals_std)  # +1: a point is its own nearest neighbor
    dists, _ = nn.kneighbors(vals_std)
    kth_dists = dists[:, -1]  # distance to the k-th REAL neighbor, excluding self

    # Threshold = how far a typical real design sits from its own neighbors,
    # with margin for FEA mesh noise / minor data sparsity. Using a high
    # percentile (not max) avoids one noisy outlier row blowing the door open.
    threshold = float(np.percentile(kth_dists, 97) * margin)

    return {
        'points': vals_std.tolist(),
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist(),
        'k': k_eff,
        'threshold': threshold,
    }


def _fit_radial_depth_budget(X: pd.DataFrame, margin: float = 1.05) -> dict | None:
    """
    Learns the relationship between available rotor radial depth and total
    flux-barrier size -- the specific mechanism behind 'Rotor Inner diameter
    too large -> barriers don't fit -> sketch lines intersect'.

    Available radial depth = (Rotor Outer - Rotor Inner) / 2. Barriers
    (TM1A, TM2A, T1A, T2, and the B/C/D/E equivalents) all stack inside that
    depth. A generic statistical 'looks normal overall' check can miss a
    violation of this ONE relationship when it's diluted across 10-25 other
    dimensions that are individually fine. This instead measures, directly,
    how much depth real successful designs needed per unit of total barrier
    size, and requires future candidates to respect at least the tightest
    margin ever actually observed (with a small safety buffer).
    """
    rotor_outer_col = next((c for c in X.columns if "Rotor" in c and "Outer" in c), None)
    rotor_inner_col = next((c for c in X.columns if "Rotor" in c and "Inner" in c), None)
    barrier_cols = [c for c in X.columns if "(mm)" in c and "Stator" not in c and "Rotor" not in c]

    if rotor_outer_col is None or rotor_inner_col is None or not barrier_cols:
        return None

    available_depth = (X[rotor_outer_col] - X[rotor_inner_col]) / 2.0
    total_barrier = X[barrier_cols].sum(axis=1)

    valid = total_barrier > 0
    if not valid.any():
        return None

    ratio = available_depth[valid] / total_barrier[valid]
    # Tightest margin any real, successful design ever actually used, with a
    # small safety buffer ABOVE it (not below) -- candidates tighter than
    # this have never been validated to actually fit.
    min_ratio = float(ratio.min() * margin)

    return {
        'rotor_outer_col': rotor_outer_col,
        'rotor_inner_col': rotor_inner_col,
        'barrier_cols': barrier_cols,
        'min_ratio': min_ratio,
    }


def save_bounds(geom: dict, model_dir: Path):
    """Extracts min/max dimensions from the dataset and saves them for the optimizer."""
    gid = geom['geometry_id']
    X = geom['inputs']
    bounds = {
        'lb': X.min(axis=0).values.tolist(),
        'ub': X.max(axis=0).values.tolist(),
        'var_names': list(X.columns),  # <-- Automatically captures T1A, VA, etc.
        'geometry_id': gid,
        'airgap_info': _detect_airgap_info(X),         # data-driven, no hardcoded mm value
        'manifold': _fit_manifold(X),                    # data-driven, non-convex joint-feasibility check
        'radial_depth_budget': _fit_radial_depth_budget(X),  # data-driven rotor-depth vs barrier-size constraint
    }
    with open(model_dir / f"{gid}_bounds.json", 'w') as f:
        json.dump(bounds, f, indent=2)

    ag = bounds['airgap_info']
    if ag:
        if ag['is_fixed']:
            print(f"    ↳ {gid}: airgap detected as FIXED ≈ {ag['observed_mean_radial']:.3f}mm radial (from data)")
        else:
            print(f"    ↳ {gid}: airgap detected as VARIABLE, {ag['observed_min_radial']:.3f}-{ag['observed_max_radial']:.3f}mm radial (from data)")
    print(f"    ↳ {gid}: feasibility manifold fitted over {len(bounds['var_names'])} variables "
          f"(k={bounds['manifold']['k']}, threshold={bounds['manifold']['threshold']:.3f})")
    rdb = bounds['radial_depth_budget']
    if rdb:
        print(f"    ↳ {gid}: radial depth budget learned, min ratio={rdb['min_ratio']:.3f} "
              f"over {len(rdb['barrier_cols'])} barrier dims")


def train_and_compare(geom: dict, model_dir: Path, plot_dir: Path, global_tracker: dict, idx: int, total: int):
    gid   = geom['geometry_id']
    X     = geom['inputs']
    Y     = geom['outputs']
    n_out = Y.shape[1]
    
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.15, random_state=42)

    # --- 1. XGBOOST TRAINING ---
    xgb_start = time.time()
    xgb_base = xgb.XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05, 
        objective='reg:squarederror', random_state=42, n_jobs=-1
    )
    xgb_model = MultiOutputRegressor(xgb_base)
    xgb_model.fit(X_tr, Y_tr)
    xgb_train_time = time.time() - xgb_start
    xgb_preds = xgb_model.predict(X_te)
    joblib.dump(xgb_model, str(model_dir / f"{gid}_xgboost.pkl"))

    # --- 2. CATBOOST TRAINING ---
    cat_start = time.time()
    cat_model = CatBoostRegressor(
        iterations=1000, depth=6, learning_rate=0.05,
        loss_function='MultiRMSE', eval_metric='MultiRMSE',
        random_seed=42, verbose=False
    )
    cat_model.fit(X_tr, Y_tr)
    cat_train_time = time.time() - cat_start
    cat_preds = cat_model.predict(X_te)
    cat_model.save_model(str(model_dir / f"{gid}_catboost.cbm"))

    # --- 3. METRICS EVALUATION ---
    results = {"geometry_id": gid, "outputs": {}}
    
    print(f"\n  --- METRICS FOR {gid} ({idx}/{total}) ---")
    print(f"  {'Output':<15} | {'XGBoost R²':<12} | {'CatBoost R²':<12} | {'Winner':<10}")
    print("  " + "-" * 55)

    for oi in range(n_out):
        oname = f"y{oi+1}"
        label = OUTPUT_NAMES.get(oname, oname)
        
        y_act = Y_te.iloc[:, oi].values
        xgb_y = xgb_preds[:, oi]
        cat_y = cat_preds[:, oi]
        
        xgb_r2   = r2_score(y_act, xgb_y)
        cat_r2   = r2_score(y_act, cat_y)
        xgb_mape = mean_absolute_percentage_error(y_act, xgb_y) * 100
        cat_mape = mean_absolute_percentage_error(y_act, cat_y) * 100
        
        winner = "CatBoost" if cat_r2 > xgb_r2 else ("XGBoost" if xgb_r2 > cat_r2 else "Tie")
        print(f"  {label:<15} | {xgb_r2:12.4f} | {cat_r2:12.4f} | {winner}")
        
        results["outputs"][oname] = {
            "xgb_r2": xgb_r2, "cat_r2": cat_r2,
            "xgb_mape": xgb_mape, "cat_mape": cat_mape
        }

        global_tracker[oname]["actual"].extend(y_act)
        global_tracker[oname]["xgb_pred"].extend(xgb_y)
        global_tracker[oname]["cat_pred"].extend(cat_y)

    print(f"  [Speed] XGBoost: {xgb_train_time:.2f}s  |  CatBoost: {cat_train_time:.2f}s")
    global_tracker["time"]["xgb"].append(xgb_train_time)
    global_tracker["time"]["cat"].append(cat_train_time)

    # Generate local visuals
    _plot_local_comparison(gid, results["outputs"], plot_dir)
    _plot_correlation_scatter(gid, Y_te, xgb_preds, cat_preds, plot_dir)
    
    # Generate the grouped feature importance visual
    _plot_feature_importance(gid, xgb_model, cat_model, list(X.columns), plot_dir)

    # Generate Pearson correlation matrix for the local geometry
    _plot_pearson_correlation_matrix(gid, X, Y, plot_dir)
    
    return results

def _plot_pearson_correlation_matrix(gid, X, Y, plot_dir, is_global=False):
    """Generates a Pearson Correlation Matrix Heatmap matching the target visual style."""
    # Clean X column names
    clean_X = X.copy()
    clean_X.columns = [str(c).split("::")[-1].strip() for c in clean_X.columns]
    
    # Clean Y column names
    clean_Y = Y.copy()
    new_y_cols = []
    for i, col in enumerate(clean_Y.columns):
        if str(col) in OUTPUT_NAMES:
            new_y_cols.append(OUTPUT_NAMES[str(col)])
        elif len(clean_Y.columns) == 4:
            new_y_cols.append(OUTPUT_NAMES.get(f"y{i+1}", str(col)))
        else:
            new_y_cols.append(str(col))
    clean_Y.columns = new_y_cols
    
    # Merge for correlation (All inputs + All outputs)
    df = pd.concat([clean_X, clean_Y], axis=1)
    corr_matrix = df.corr(method='pearson')
    
    # Square figure sizing based on the number of variables
    n_vars = len(df.columns)
    fig_size = max(8, n_vars * 0.75)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    
    # Heatmap Plot using coolwarm to mimic the red/blue divergent map from the request
    cmap = plt.get_cmap("coolwarm")
    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    
    # Colorbar specific configuration matching the requested style
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, aspect=20, pad=0.04)
    cbar.set_ticks([-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])
    cbar.ax.tick_params(colors="#1A1A1A", length=4)
    cbar.outline.set_visible(False) 
    
    # Tick Setup
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    
    # Exact label rotation (45 deg) and alignment
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", rotation_mode="anchor", color="#1A1A1A", fontsize=10)
    ax.set_yticklabels(corr_matrix.index, color="#1A1A1A", fontsize=10)
    
    # Exact Title format: [geom_id] Pearson Correlation Matrix
    title_text = "[GLOBAL] Pearson Correlation Matrix" if is_global else f"[{gid}] Pearson Correlation Matrix"
    ax.set_title(title_text, pad=20, fontweight='bold', color="#1A1A1A", fontsize=14)
    
    # Annotations (Values to 2 decimal places)
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.values[i, j]
            if np.isnan(val):
                continue
            # White text for strong correlations, black for weak to ensure readability
            text_color = "white" if abs(val) > 0.65 else "#1A1A1A"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)
            
    # Matrix Grid aesthetics to create the prominent white cell borders (similar to seaborn linewidths=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Offset ticks to draw gridlines *between* the cells
    ax.set_xticks(np.arange(corr_matrix.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(corr_matrix.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle='-', linewidth=2.5)
    
    # Hide the tick marks themselves but keep the labels
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=False, pad=5)
    
    fig.tight_layout()
    filename = "_GLOBAL_CORRELATION_MATRIX.png" if is_global else f"{gid}_correlation_matrix.png"
    fig.savefig(plot_dir / filename, dpi=300, facecolor="#FFFFFF", bbox_inches="tight")
    plt.close(fig)

def _plot_local_comparison(gid, metrics, plot_dir):
    labels = [OUTPUT_NAMES.get(k, k) for k in metrics.keys()]
    xgb_scores = [v['xgb_r2'] for v in metrics.values()]
    cat_scores = [v['cat_r2'] for v in metrics.values()]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    
    rects1 = ax.bar(x - width/2, xgb_scores, width, label='XGBoost', color=XGB_C)
    rects2 = ax.bar(x + width/2, cat_scores, width, label='CatBoost', color=CAT_C)
    
    ax.set_ylabel('R² Score')
    ax.set_title(f'R² Comparison: {gid}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([max(0, min(min(xgb_scores), min(cat_scores)) - 0.1), 1.05])
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    for spine in ax.spines.values(): 
        spine.set_edgecolor(GRID)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    fig.savefig(plot_dir / f"{gid}_xgb_vs_cat_bars.png", dpi=300, facecolor=BG)
    plt.close(fig)

def _plot_correlation_scatter(gid, Y_te, xgb_preds, cat_preds, plot_dir):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Actual vs Predicted  ·  {gid} (XGBoost vs CatBoost)", color=TEXT, fontsize=12, fontweight="bold", y=1.05)
    
    for i, (ax, oname) in enumerate(zip(axes, ["y1", "y2", "y3", "y4"])):
        ax.set_facecolor(BG)
        
        y_act = Y_te.iloc[:, i].values
        xgb_y = xgb_preds[:, i]
        cat_y = cat_preds[:, i]
        
        xgb_r2 = r2_score(y_act, xgb_y)
        cat_r2 = r2_score(y_act, cat_y)
        
        ax.scatter(y_act, xgb_y, s=15, alpha=0.5, color=XGB_C, label=f"XGBoost (R²={xgb_r2:.2f})")
        ax.scatter(y_act, cat_y, s=15, alpha=0.7, color=CAT_C, label=f"CatBoost (R²={cat_r2:.2f})")
        
        mn = min(min(y_act), min(xgb_y), min(cat_y))
        mx = max(max(y_act), max(xgb_y), max(cat_y))
        ax.plot([mn, mx], [mn, mx], "--", color="black", lw=1.5, alpha=0.8)
        
        label = OUTPUT_NAMES.get(oname, oname)
        ax.set_title(f"{label}", color=TEXT, pad=6, fontweight='bold')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID)
        ax.grid(True, alpha=0.4)
        for spine in ax.spines.values(): 
            spine.set_edgecolor(GRID)
            
    fig.tight_layout()
    fig.savefig(plot_dir / f"{gid}_correlation_scatter.png", dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

def _plot_feature_importance(gid, xgb_model, cat_model, var_names, plot_dir):
    """Generates a grouped horizontal bar chart of Feature Importances for all outputs"""
    # Clean the "Saliency::" out of the variable names for a clean chart
    clean_names = np.array([str(n).split("::")[-1].strip() for n in var_names])
    n_features = len(clean_names)
    
    # XGBoost: MultiOutputRegressor holds 4 estimators (y1, y2, y3, y4)
    # Shape: (4, n_features)
    xgb_imps = np.array([est.feature_importances_ for est in xgb_model.estimators_])
    # Normalize each target to 100%
    xgb_imps = 100.0 * (xgb_imps / (xgb_imps.sum(axis=1, keepdims=True) + 1e-9))
    
    # CatBoost: Native multi-target handles importance internally (Unified 1D array)
    cat_imps_1d = cat_model.get_feature_importance()
    cat_imps_1d = 100.0 * (cat_imps_1d / (cat_imps_1d.sum() + 1e-9))
    # Duplicate to (4, n_features) to match XGBoost dimensions for the grouping loop
    cat_imps = np.tile(cat_imps_1d, (4, 1))
    
    # Sort indices based on the average importance across targets
    xgb_avg = xgb_imps.mean(axis=0)
    cat_avg = cat_imps.mean(axis=0)
    xgb_sort_idx = np.argsort(xgb_avg)
    cat_sort_idx = np.argsort(cat_avg)
    
    # Dynamic height based on number of variables so bars don't get squished
    fig_height = max(5, n_features * 0.6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, fig_height))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Feature Importance  ·  {gid}", color=TEXT, fontsize=14, fontweight="bold", y=1.02)
    
    # Plotting layout settings
    y_pos = np.arange(n_features)
    bar_height = 0.2
    # Offsets shift the 4 bars up and down slightly so they sit next to each other
    offsets = [-1.5 * bar_height, -0.5 * bar_height, 0.5 * bar_height, 1.5 * bar_height]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"] # Blue, Orange, Green, Red
    
    labels = [OUTPUT_NAMES["y1"], OUTPUT_NAMES["y2"], OUTPUT_NAMES["y3"], OUTPUT_NAMES["y4"]]
    
    # --- Plot XGBoost ---
    ax1.set_facecolor(BG)
    for i in range(4):
        ax1.barh(y_pos + offsets[i], xgb_imps[i, xgb_sort_idx], height=bar_height, color=colors[i], label=labels[i], alpha=0.85)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(clean_names[xgb_sort_idx])
    ax1.set_title("XGBoost (MultiOutput)", color=TEXT, fontweight='bold', pad=10)
    ax1.set_xlabel("Relative Importance (%)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(axis='x', alpha=0.3)
    for spine in ax1.spines.values(): spine.set_edgecolor(GRID)
    
    # --- Plot CatBoost ---
    ax2.set_facecolor(BG)
    for i in range(4):
        ax2.barh(y_pos + offsets[i], cat_imps[i, cat_sort_idx], height=bar_height, color=colors[i], label=labels[i], alpha=0.85)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(clean_names[cat_sort_idx])
    ax2.set_title("CatBoost (MultiRMSE unified)", color=TEXT, fontweight='bold', pad=10)
    ax2.set_xlabel("Relative Importance (%)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(axis='x', alpha=0.3)
    for spine in ax2.spines.values(): spine.set_edgecolor(GRID)
    
    fig.tight_layout()
    fig.savefig(plot_dir / f"{gid}_feature_importance.png", dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

def _plot_comparative_heatmap(all_metrics: list[dict], plot_dir: Path):
    geom_ids = [m['geometry_id'] for m in all_metrics]
    out_keys = ["y1", "y2", "y3", "y4"]
    out_labels = [OUTPUT_NAMES.get(k, k) for k in out_keys]

    xgb_matrix = np.array([[m['outputs'][k]['xgb_r2'] for k in out_keys] for m in all_metrics])
    cat_matrix = np.array([[m['outputs'][k]['cat_r2'] for k in out_keys] for m in all_metrics])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(out_keys) * 4), max(6, len(geom_ids) * 0.4)))
    fig.patch.set_facecolor(BG)
    fig.suptitle("COMPARATIVE R² HEATMAP: XGBoost vs CatBoost", fontsize=14, fontweight="bold", y=1.05)

    color_nodes = [
        (0.00, "#d62728"),  
        (0.70, "#ff7f0e"),  
        (0.80, "#98df8a"),  
        (0.90, "#2ca02c"),  
        (1.00, "#00441b"),  
    ]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_r2", color_nodes)

    ax1.set_facecolor(BG)
    im1 = ax1.imshow(xgb_matrix, aspect='auto', cmap=custom_cmap, vmin=0.0, vmax=1.0)
    ax1.set_title("XGBoost (MultiOutput)", color=TEXT, pad=10, fontweight='bold')
    ax1.set_xticks(range(len(out_labels)))
    ax1.set_xticklabels(out_labels, rotation=15, ha='right')
    ax1.set_yticks(range(len(geom_ids)))
    ax1.set_yticklabels(geom_ids, fontsize=8)

    ax2.set_facecolor(BG)
    im2 = ax2.imshow(cat_matrix, aspect='auto', cmap=custom_cmap, vmin=0.0, vmax=1.0)
    ax2.set_title("CatBoost (MultiRMSE)", color=TEXT, pad=10, fontweight='bold')
    ax2.set_xticks(range(len(out_labels)))
    ax2.set_xticklabels(out_labels, rotation=15, ha='right')
    ax2.set_yticks([]) 

    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.6, pad=0.05)
    cbar.ax.tick_params(colors=TEXT)
    cbar.set_label("R² Score", color=TEXT)

    for ax, matrix in zip([ax1, ax2], [xgb_matrix, cat_matrix]):
        for spine in ax.spines.values(): 
            spine.set_edgecolor(GRID)
        for i in range(len(geom_ids)):
            for j in range(len(out_keys)):
                v = matrix[i, j]
                text_color = 'white' if (v >= 0.90 or v < 0.75) else 'black'
                ax.text(j, i, f"{v:.3f}", ha='center', va='center', fontsize=7, color=text_color, fontweight='bold')

    fig.savefig(plot_dir / "_SUMMARY_COMPARATIVE_HEATMAP.png", dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

def _plot_global_comparison(global_tracker, plot_dir):
    out_keys = ["y1", "y2", "y3", "y4"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle("GLOBAL SCATTER MAPPING: XGBoost vs CatBoost", fontsize=16, fontweight="bold", y=1.02)
    
    for ax, okey in zip(axes.flatten(), out_keys):
        ax.set_facecolor(BG)
        y_act = np.array(global_tracker[okey]["actual"])
        xgb_y = np.array(global_tracker[okey]["xgb_pred"])
        cat_y = np.array(global_tracker[okey]["cat_pred"])
        
        xgb_r2 = r2_score(y_act, xgb_y)
        cat_r2 = r2_score(y_act, cat_y)
        
        ax.scatter(y_act, xgb_y, alpha=0.4, s=6, color=XGB_C, label=f"XGBoost (R²={xgb_r2:.3f})")
        ax.scatter(y_act, cat_y, alpha=0.4, s=6, color=CAT_C, label=f"CatBoost (R²={cat_r2:.3f})")
        
        mn, mx = min(y_act), max(y_act)
        ax.plot([mn, mx], [mn, mx], "--", color="black", lw=1.5, alpha=0.8)
        
        ax.set_title(OUTPUT_NAMES[okey], fontweight='bold')
        ax.set_xlabel("Actual FEA Simulation")
        ax.set_ylabel("AI Prediction")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)
        for spine in ax.spines.values(): 
            spine.set_edgecolor(GRID)
        
    fig.tight_layout()
    fig.savefig(plot_dir / "_GLOBAL_SCATTER_COMPARISON.png", dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

def _print_final_verdict(global_tracker, out_dir):
    print("\n" + "=" * 60)
    print("  🏆 GLOBAL SHOWDOWN VERDICT")
    print("=" * 60)
    
    out_keys = ["y1", "y2", "y3", "y4"]
    xgb_wins = 0
    cat_wins = 0
    
    for okey in out_keys:
        y_act = np.array(global_tracker[okey]["actual"])
        xgb_y = np.array(global_tracker[okey]["xgb_pred"])
        cat_y = np.array(global_tracker[okey]["cat_pred"])
        
        xgb_r2 = r2_score(y_act, xgb_y)
        cat_r2 = r2_score(y_act, cat_y)
        xgb_mape = mean_absolute_percentage_error(y_act, xgb_y) * 100
        cat_mape = mean_absolute_percentage_error(y_act, cat_y) * 100

        print(f"\n  {OUTPUT_NAMES[okey].upper()}:")
        print(f"    XGBoost  -> R²: {xgb_r2:.4f} | Error (MAPE): {xgb_mape:.2f}%")
        print(f"    CatBoost -> R²: {cat_r2:.4f} | Error (MAPE): {cat_mape:.2f}%")
        
        if cat_r2 > xgb_r2:
            cat_wins += 1
            print("    WINNER: CatBoost")
        else:
            xgb_wins += 1
            print("    WINNER: XGBoost")

    tot_xgb_time = sum(global_tracker["time"]["xgb"])
    tot_cat_time = sum(global_tracker["time"]["cat"])
    
    print("\n  ⏱️ TOTAL TRAINING SPEED:")
    print(f"    XGBoost : {tot_xgb_time:.2f} seconds")
    print(f"    CatBoost: {tot_cat_time:.2f} seconds")
    
    print("\n" + "=" * 60)
    if cat_wins > xgb_wins:
        print("  FINAL CONCLUSION: CATBOOST IS THE SUPERIOR ENGINE.")
    elif xgb_wins > cat_wins:
        print("  FINAL CONCLUSION: XGBOOST IS THE SUPERIOR ENGINE.")
    else:
        print("  FINAL CONCLUSION: IT IS A DEAD TIE. LOOK AT SPEED TO DECIDE.")
    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Global Showdown: XGBoost vs CatBoost")
    parser.add_argument("--data_dir",  default="data",    help="Folder containing .data files")
    parser.add_argument("--out_dir",   default="results", help="Results output folder")
    parser.add_argument("--model_dir", default="models",  help="Folder to save trained models")
    args = parser.parse_args()

    base      = Path(__file__).parent
    data_dir  = base / args.data_dir
    model_dir = base / args.model_dir
    out_dir   = base / args.out_dir
    plot_dir  = out_dir / "plots"

    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  INITIALIZING GLOBAL SHOWDOWN: XGBoost vs CatBoost")
    print("=" * 60)

    geometries = load_all_data_files(data_dir)
    total_geoms = len(geometries)
    
    global_tracker = {
        "y1": {"actual": [], "xgb_pred": [], "cat_pred": []},
        "y2": {"actual": [], "xgb_pred": [], "cat_pred": []},
        "y3": {"actual": [], "xgb_pred": [], "cat_pred": []},
        "y4": {"actual": [], "xgb_pred": [], "cat_pred": []},
        "time": {"xgb": [], "cat": []}
    }

    all_metrics = []
    
    # Added trackers for plotting global correlation matrices
    global_X_list = []
    global_Y_list = []

    for idx, geom in enumerate(geometries, 1):
        save_bounds(geom, model_dir)
        
        # Track Inputs and Outputs for the final global Pearson Correlation plot
        global_X_list.append(geom['inputs'])
        global_Y_list.append(geom['outputs'])
        
        res = train_and_compare(geom, model_dir, plot_dir, global_tracker, idx, total_geoms)
        all_metrics.append(res)

    # Save metrics cleanly for the optimizer script to read later
    with open(out_dir / "training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("\nGenerating final comparative visuals...")
    
    # Generate the unified Global Pearson Correlation Matrix
    if global_X_list and global_Y_list:
        global_X = pd.concat(global_X_list, axis=0, ignore_index=True)
        global_Y = pd.concat(global_Y_list, axis=0, ignore_index=True)
        _plot_pearson_correlation_matrix("GLOBAL", global_X, global_Y, plot_dir, is_global=True)
        
    _plot_comparative_heatmap(all_metrics, plot_dir)
    _plot_global_comparison(global_tracker, plot_dir)
    _print_final_verdict(global_tracker, out_dir)
    
    print(f"✓ All visuals and tracking json saved to: {out_dir}")

if __name__ == "__main__":
    main()