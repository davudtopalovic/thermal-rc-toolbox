import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import math


class _OneDecimalSci(ScalarFormatter):
    """ScalarFormatter that forces a 1-decimal mantissa (e.g. 0.0, 0.2, ..., 1.0)
    while keeping the scientific 10^x offset label."""
    def _set_format(self):
        self.format = "%.1f"


param_title_fontsize = 20
DPI = 300

def plot_free_param_evolution(optimizer, show=True, fig_path=None, fig_config=None):

    if fig_config is None:
        fig_config = {}
    figsize = fig_config.get("figsize", (17, 3))
    # dpi = fig_config.get("dpi", 100)

    population_params_by_gen = [entry.pop.get("X") for entry in optimizer.result.history]
    generations = np.arange(len(population_params_by_gen))

    n_params = len(optimizer.free_param_names)
    ncols = n_params
    nrows = math.ceil(n_params / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, dpi=DPI)
    axs = axs.flatten()

    for i, param in enumerate(optimizer.free_param_names):
        medians = [np.median(gen[:, i]) for gen in population_params_by_gen]
        lower_ci = [np.percentile(gen[:, i], 2.5) for gen in population_params_by_gen]
        upper_ci = [np.percentile(gen[:, i], 97.5) for gen in population_params_by_gen]

        # 🔁 swapped axes
        axs[i].fill_betweenx(generations, lower_ci, upper_ci, color='lightcoral', alpha=0.3, label="95% CI population")
        axs[i].plot(medians, generations, color='darkred', label="Median")
        axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axs[i].ticklabel_format(axis='x', style='scientific', scilimits=(0, 3))

        axs[i].set_title(optimizer.bldg_props.params_latex[param], fontsize=param_title_fontsize)
        if i == len(optimizer.free_param_names) - 1:
            axs[i].legend()

        if i == 0:
            axs[i].set_ylabel("Generation")

    for ax in axs:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)

    # y-axis is shared (sharey=True), so a single invert flips all subplots.
    # Calling it once per subplot would toggle N times and yield the wrong
    # orientation whenever N is even.
    axs[0].invert_yaxis()

    # remove unused axes
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    if fig_path is not None: plt.savefig(fig_path, dpi=DPI)
    if show: plt.show()
    else: plt.close()


def plot_criteria_distribution_and_selection(dfR_,
                                             dfRtopN,
                                             criteria="val err",
                                             threshold=None,
                                             threshold_idx=None,
                                             threshold_config=None,
                                             show=True,
                                             fig_path=None,
                                             fig_config=None):
    dfR_ = dfR_.copy()
    dfRtopN = dfRtopN.copy()
    min_criteria = dfR_["criteria"].min()
    std_criteria = dfR_["criteria"].std()

    if fig_config is None: fig_config = {}
    figsize = fig_config.get("figsize", (15, 3+1))
    # dpi = fig_config.get("dpi", 100)

    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=DPI)
    ax1, ax2, ax3 = axs

    has_val = not dfR_["validation_error"].isna().all()

    # ===== PLOT 1: Train vs Validation Error (or Train Error over Generations) =====
    df_plot = dfR_[dfR_["generation"] > 0]

    # ax1.ticklabel_format(axis='x', style='scientific', scilimits=(0, 1))
    # ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0, 1))

    if has_val:
        sc = ax1.scatter(
            df_plot["train_error"],
            df_plot["validation_error"],
            c=df_plot["generation"],
            cmap="viridis",
            edgecolor="k"
        )
        ax1.set_xlabel("Train Error")
        ax1.set_ylabel("Validation Error")
        ax1.set_title("Train vs Validation Error per Candidate")
        cb = fig.colorbar(sc, ax=ax1)
        cb.set_label("Generation")
    else:
        sc = ax1.scatter(
            df_plot["generation"],
            df_plot["train_error"],
            c=df_plot["generation"],
            cmap="viridis",
            edgecolor="k"
        )
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Train Error")
        ax1.set_title("Train Error per Candidate over Generations")
        cb = fig.colorbar(sc, ax=ax1)
        cb.set_label("Generation")
    ax1.set_axisbelow(True)

    # ===== PLOT 2: Histogram of Criteria with Threshold =====
    ax2.hist(dfR_["criteria"], bins=200, alpha=1, edgecolor='black', zorder=2)
    ax2.axvline(min_criteria, color='green', linestyle='--', linewidth=2,
                label=rf'Best error, $\epsilon_{{\mathrm{{min}}}}$ = {min_criteria:.1f}', zorder=3)

    # ax2.ticklabel_format(axis='x', style='scientific', scilimits=(0, 1))
    # ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0, 1))

    t_type = (threshold_config or {}).get("type", None)
    if t_type == "elbow":
        threshold_label = rf' $T (elbow) = {threshold:.1f}$'
    elif t_type == "idx":
        threshold_label = rf' $T (idx={threshold_config["x"]}) = {threshold:.1f}$'
    elif t_type == "percentage":
        threshold_label = rf' $T (percentage={threshold_config["x"]}) = {threshold:.1f}$'
    else:
        threshold_label = rf' $T (?) = {threshold:.1f}$'
    ax2.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label=threshold_label, zorder=3)
    ax2.set_xlim(None, min_criteria + std_criteria*3)
    ax2.set_axisbelow(True)
    ax2.set_title(f"Distribution of the Criteria ({criteria})")
    ax2.set_xlabel(criteria)
    ax2.set_ylabel("Count")
    ax2.legend()

    # # ===== Plot 3: Ranked Parameter Set Criteria =====
    dfR_sorted = dfR_.sort_values(by="criteria", ascending=True).reset_index(drop=True)
    ax3.plot(dfR_sorted["criteria"], color='black', label="Criteria")
    ax3.axvline(threshold_idx, linestyle="--", color="k", zorder=0, label="Threshold")

    # ax3.ticklabel_format(axis='x', style='scientific', scilimits=(0, 1))
    # ax3.ticklabel_format(axis='y', style='scientific', scilimits=(0, 1))

    ax3.fill_between(
        x=[0, threshold_idx],
        y1=min_criteria,
        y2=threshold,
        color='wheat',
        alpha=0.5,
        label='Accepted Solutions $\mathcal{R}^*$',
        zorder=0
    )
    ax3.fill_between(
        x=[threshold_idx, len(dfR_sorted)],
        y1=min_criteria,
        y2=threshold + 0.2*threshold,
        color='tab:red',
        alpha=0.1,
        label='Rejected Solutions',
        zorder=0
    )
    ax3.set_ylim(min_criteria - 0.01*min_criteria, threshold + 0.2*threshold)

    ax3.set_xlabel("Parameter Set Candidates")
    ax3.set_ylabel(rf"{criteria} $F(\cdot)$")
    ax3.set_title(r"Selection of the Final Set of Best Candidates $\mathcal{R}^*$")
    ax3.set_axisbelow(True)
    ax3.axhline(min_criteria, color='green', linestyle='--', linewidth=2, label=rf'Best error, $\epsilon_{{\mathrm{{min}}}}$ = {min_criteria:.1f}', zorder=3)
    ax3.axhline(threshold, color='red', linestyle='--', linewidth=2, label=threshold_label, zorder=3)
    ax3.legend()

    for ax in axs:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)

    plt.tight_layout()
    if fig_path is not None: plt.savefig(fig_path, dpi=DPI)    
    if show: plt.show()
    else: plt.close()

# def plot_free_param_distribution(optimizer, show=True, fig_path=None, fig_config=None):
    
#     dfRtopN_models = pd.DataFrame(optimizer.dfRtopN["params"].values.tolist())
#     num_params = len(dfRtopN_models.columns)

#     # --- PLOTTING ---
#     cols = num_params
#     rows = (num_params + cols - 1) // cols
    
#     if fig_config is None: fig_config = {}
#     figsize = fig_config.get("figsize", (15, 15/3))
#     # dpi = fig_config.get("dpi", 100)
#     fig, axs = plt.subplots(rows, cols, figsize=figsize, dpi=DPI)
#     axs = axs.flatten()

#     for i, column in enumerate(dfRtopN_models.columns):
        
#         sns.kdeplot(dfRtopN_models[column], ax=axs[i], fill=True, color="wheat", edgecolor="darkorange", alpha=1.0, zorder=10)
        
#         # Design
#         axs[i].set_title(f'{optimizer.bldg_props.params_latex[column]}', fontsize=param_title_fontsize)

#         if i>0: axs[i].set_ylabel("")
#         axs[i].set_xlabel("")

#         axs[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#         axs[i].ticklabel_format(axis='x', style='scientific', scilimits=(0, 2))

#         axs[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#         axs[i].ticklabel_format(axis='y', style='scientific', scilimits=(0, 2))

#     for ax in axs:
#         ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)

#     # Hide unused subplots
#     for j in range(i + 1, len(axs)):
#         fig.delaxs(axs[j])

#     plt.tight_layout()
#     if fig_path is not None: plt.savefig(fig_path, dpi=DPI)    
#     if show: plt.show()
#     else: plt.close()


# def plot_cluster_ranges(optimizer, show=True, fig_path=None, fig_config=None, mode="ci"):

#     cluster_summaries = optimizer.cluster_summaries
#     cluster_keys = list(cluster_summaries.keys())
#     cluster_to_color = optimizer.cluster_to_color

#     # --- PLOTTING ---
#     nrows = 1
#     ncols = len(optimizer.free_param_names)
    
#     if fig_config is None: fig_config = {}
#     figsize = fig_config.get("figsize", (20, 3))
#     # dpi = fig_config.get("dpi", 100)

#     fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=DPI)
#     axs = axs.flatten()

#     for i, param in enumerate(optimizer.free_param_names):
#         ax = axs[i]

#         # # # Physics bounds
#         # min_val, max_val = optimizer.bldg_props.free_param_bounds[param]
#         # ax.plot([min_val, max_val], [-1, -1], color="gray", linewidth=3, label="Initial bounds")
#         # ax.plot(min_val, -1, marker="|", color="black", markersize=12)
#         # ax.plot(max_val, -1, marker="|", color="black", markersize=12)

#         # # Cluster means and stds
#         # for cl_k in cluster_keys:
#         #     mean_val = cluster_summaries[cl_k]["best_candidate"]["params"][param]
#         #     std_val = cluster_summaries[cl_k]["std"][param]
#         #     ax.errorbar(
#         #         mean_val, cl_k, xerr=std_val, fmt="o", markersize=5, capsize=5,
#         #         color=cluster_to_color[cl_k], ecolor=cluster_to_color[cl_k],
#         #         markerfacecolor=cluster_to_color[cl_k],
#         #         label=f"Cluster {cl_k}" if i == 0 else None
#         #     )

#         # # Cluster [min, max] ranges
#         # for cl_k in cluster_keys:
#         #     lo = cluster_summaries[cl_k]["min"][param]
#         #     hi = cluster_summaries[cl_k]["max"][param]
#         #     center = (lo + hi) / 2
#         #     ax.errorbar(
#         #         center, cl_k, xerr=[[center - lo], [hi - center]], fmt="o", markersize=5, capsize=5,
#         #         color=cluster_to_color[cl_k], ecolor=cluster_to_color[cl_k],
#         #         markerfacecolor=cluster_to_color[cl_k],
#         #         label=f"Cluster {cl_k}" if i == 0 else None
#         #     )

#         # Cluster 90% CI (5th–95th percentile)
#         for cl_k in cluster_keys:
#             if mode == "ci":
#                 lo = cluster_summaries[cl_k]["p05"][param]
#                 hi = cluster_summaries[cl_k]["p95"][param]
#                 center = cluster_summaries[cl_k]["p50"][param]
#             else:  # mode == "minmax"
#                 lo = cluster_summaries[cl_k]["min"][param]
#                 hi = cluster_summaries[cl_k]["max"][param]
#                 center = (lo + hi) / 2

#             ax.errorbar(
#                 center,
#                 cl_k,
#                 xerr=[[center - lo], [hi - center]],
#                 fmt="o",
#                 markersize=5,
#                 capsize=5,
#                 color=cluster_to_color[cl_k],
#                 ecolor=cluster_to_color[cl_k],
#                 markerfacecolor=cluster_to_color[cl_k],
#                 label=f"Cluster {cl_k}" if i == 0 else None
#             )

#         # Aesthetics
#         if i == 0:
#             ax.set_ylabel("Cluster Key")
        
#         ax.set_title(optimizer.bldg_props.params_latex[param], fontsize=param_title_fontsize)
#         # ax.set_yticks([-1] + cluster_keys)
#         ax.set_yticks(cluster_keys)
#         # ax.set_yticklabels(["Initial"] + [str(cl_k) for cl_k in cluster_keys])
#         ax.set_yticklabels([str(cl_k) for cl_k in cluster_keys])
#         ax.invert_yaxis()
#         ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#         ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 2))

#     for ax in axs:
#         ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)

#     plt.tight_layout()
#     if fig_path is not None: plt.savefig(fig_path, dpi=DPI)    
#     if show: plt.show()
#     else: plt.close()
    

# def plot_cluster_ranges_dual(optimizer, show=True, fig_path=None, fig_config=None, mode="ci"):
#     """Two-row figure: top row shows clusters relative to initial bounds (zoomed out),
#     bottom row shows clusters only (zoomed in) to reveal inter-cluster differences."""

#     cluster_summaries = optimizer.cluster_summaries
#     cluster_keys = list(cluster_summaries.keys())
#     cluster_to_color = optimizer.cluster_to_color
#     n_params = len(optimizer.free_param_names)

#     if fig_config is None: fig_config = {}
#     figsize = fig_config.get("figsize", (20, 6))
#     # dpi = fig_config.get("dpi", 100)

#     fig, axs = plt.subplots(2, n_params, figsize=figsize, dpi=DPI)

#     def _plot_row(axs_row, show_initial):
#         for i, param in enumerate(optimizer.free_param_names):
#             ax = axs_row[i]

#             y_initial = -1
#             if show_initial:
#                 lo_b, hi_b = optimizer.bldg_props.free_param_bounds[param]
#                 ax.plot([lo_b, hi_b], [y_initial, y_initial], color="gray", linewidth=3, solid_capstyle="butt")
#                 ax.plot(lo_b, y_initial, marker="|", color="gray", markersize=12)
#                 ax.plot(hi_b, y_initial, marker="|", color="gray", markersize=12)

#             for cl_k in cluster_keys:
#                 if mode == "ci":
#                     lo = cluster_summaries[cl_k]["p05"][param]
#                     hi = cluster_summaries[cl_k]["p95"][param]
#                     center = cluster_summaries[cl_k]["p50"][param]
#                 else:
#                     lo = cluster_summaries[cl_k]["min"][param]
#                     hi = cluster_summaries[cl_k]["max"][param]
#                     center = (lo + hi) / 2

#                 ax.errorbar(
#                     center, cl_k,
#                     xerr=[[center - lo], [hi - center]],
#                     fmt="o", markersize=5, capsize=5,
#                     color=cluster_to_color[cl_k],
#                     ecolor=cluster_to_color[cl_k],
#                     markerfacecolor=cluster_to_color[cl_k],
#                     label=f"Cluster {cl_k}" if i == 0 else None
#                 )

#             yticks = ([y_initial] + cluster_keys) if show_initial else cluster_keys
#             yticklabels = (["Init"] + [str(k) for k in cluster_keys]) if show_initial else [str(k) for k in cluster_keys]
#             ax.set_yticks(yticks)
#             ax.set_yticklabels(yticklabels)
#             ax.invert_yaxis()
#             ax.set_title(optimizer.bldg_props.params_latex[param], fontsize=param_title_fontsize)
#             ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#             ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 2))
#             ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)

#             if i == 0:
#                 ax.set_ylabel("Cluster Key")

#     _plot_row(axs[0], show_initial=True)
#     _plot_row(axs[1], show_initial=False)

#     plt.tight_layout()
#     if fig_path is not None: plt.savefig(fig_path, dpi=DPI)
#     if show: plt.show()
#     else: plt.close()


def plot_distribution_and_cluster_ranges(optimizer, show=True, fig_path=None, fig_config=None, mode="ci"):
    """Three-row aligned figure per parameter:
      Row 0: KDE distribution of final parameter space
      Row 1: cluster ranges vs. initial bounds (zoomed out)
      Row 2: cluster ranges only (zoomed in)
    X-axes are shared per column so distributions align with cluster ranges."""

    cluster_summaries = optimizer.cluster_summaries
    cluster_keys = list(cluster_summaries.keys())
    cluster_to_color = optimizer.cluster_to_color
    n_params = len(optimizer.free_param_names)
    dfRtopN_models = pd.DataFrame(optimizer.dfRtopN["params"].values.tolist())

    if fig_config is None: fig_config = {}
    figsize = fig_config.get("figsize", (15, 5+1))
    # dpi = fig_config.get("dpi", 200)

    fig, axs = plt.subplots(
        3, n_params, figsize=figsize, dpi=DPI,
        gridspec_kw={"height_ratios": [1, 1, 1]}
    )

    # --- Row 0: KDE distributions ---
    for i, param in enumerate(optimizer.free_param_names):
        ax = axs[0, i]
        sns.kdeplot(dfRtopN_models[param], ax=ax, fill=True,
                    color="wheat", edgecolor="darkorange", alpha=1.0, zorder=10)
        ax.set_title(optimizer.bldg_props.params_latex[param], fontsize=param_title_fontsize)
        ax.set_xlabel("")
        
        # ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 1))

        # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 1))
        # ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1e}"))

        fmt_x = _OneDecimalSci(useMathText=True)
        fmt_x.set_powerlimits((0, 1))
        ax.xaxis.set_major_formatter(fmt_x)

        fmt_y = _OneDecimalSci(useMathText=True)
        fmt_y.set_powerlimits((0, 1))
        ax.yaxis.set_major_formatter(fmt_y)

        if i > 0:
            ax.set_ylabel("")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)

    # --- Rows 1 & 2: cluster ranges ---
    def _plot_cluster_row(row, show_initial):
        y_initial = -1
        for i, param in enumerate(optimizer.free_param_names):
            ax = axs[row, i]

            if show_initial:
                lo_b, hi_b = optimizer.bldg_props.free_param_bounds[param]
                ax.plot([lo_b, hi_b], [y_initial, y_initial],
                        color="gray", linewidth=3, solid_capstyle="butt")
                ax.plot(lo_b, y_initial, marker="|", color="gray", markersize=12)
                ax.plot(hi_b, y_initial, marker="|", color="gray", markersize=12)

            for cl_k in cluster_keys:
                if mode == "ci":
                    lo = cluster_summaries[cl_k]["p05"][param]
                    hi = cluster_summaries[cl_k]["p95"][param]
                    center = cluster_summaries[cl_k]["p50"][param]
                else:
                    lo = cluster_summaries[cl_k]["min"][param]
                    hi = cluster_summaries[cl_k]["max"][param]
                    center = (lo + hi) / 2

                ax.errorbar(
                    center, cl_k,
                    xerr=[[center - lo], [hi - center]],
                    fmt="o", markersize=5, capsize=5,
                    color=cluster_to_color[cl_k],
                    ecolor=cluster_to_color[cl_k],
                    markerfacecolor=cluster_to_color[cl_k],
                    label=f"Cluster {cl_k}" if i == 0 else None
                )

            yticks = ([y_initial] + cluster_keys) if show_initial else cluster_keys
            yticklabels = (["Init"] + [str(k) for k in cluster_keys]) if show_initial else [str(k) for k in cluster_keys]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 1))
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
            if i == 0:
                ax.set_ylabel("Cluster")

    _plot_cluster_row(1, show_initial=True)   # own x-axis, ticks visible
    _plot_cluster_row(2, show_initial=False)  # shared x-axis with row 0, ticks visible


    for ax in axs.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)  # adjust to taste

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    if fig_path is not None: plt.savefig(fig_path, dpi=DPI)
    if show: plt.show()
    else: plt.close()


def plot_cluster_performance(optimizer, show=True, fig_path=None, fig_config=None):

    cluster_summaries = optimizer.cluster_summaries
    cluster_keys = list(cluster_summaries.keys())

    train_errors = [cluster_summaries[k]['best_candidate']['train_error'] for k in cluster_keys]
    val_errors = [cluster_summaries[k]['best_candidate']['validation_error'] for k in cluster_keys]
    cl_eval_scores = [cluster_summaries[k]['best_candidate']['cl_eval_score'] for k in cluster_keys]

    has_val = not all(v is None or (isinstance(v, float) and np.isnan(v)) for v in val_errors)

    counts_dict = optimizer.final_models["cluster"].value_counts().to_dict()

    x = np.arange(len(cluster_keys))

    # --- PLOTTING ---
    if fig_config is None: fig_config = {}
    figsize = fig_config.get("figsize", (6, 6))
    # dpi = fig_config.get("dpi", 100)
    fig, axs = plt.subplots(2, 1, figsize=figsize, dpi=DPI, sharex=True)
    ax0, ax1 = axs

    cmap = optimizer.cmap
    colors = [cmap(i) for i in range(len(cluster_keys))]

    # ax0: Cluster evaluation criteria
    # --------------------------------
    ax0.plot(x, cl_eval_scores, linestyle='-', linewidth=2, alpha=0.8, color="gray")
    for i, (xi, score) in enumerate(zip(x, cl_eval_scores)):
        ax0.scatter(xi, score, color=colors[i], s=60, zorder=3)
    ax0.set_ylabel("Cluster Eval Score")
    ax0.set_title("Cluster Evaluation Score")
    ax0.grid(which="both", linestyle='--', linewidth=0.5, zorder=0)

    # ax1: Train (+ Validation if available)
    # ---------------------------------------
    ax1.plot(x, train_errors, linestyle='-', linewidth=1, alpha=0.8, color="gray", label="Train Error")
    for i, (xi, te) in enumerate(zip(x, train_errors)):
        ax1.scatter(xi, te, color=colors[i], s=60, zorder=3)

    if has_val:
        ax1r = ax1.twinx()
        ax1r.plot(x, val_errors, linestyle='--', linewidth=1, alpha=0.8, color="gray", label="Validation Error")
        for i, (xi, ve) in enumerate(zip(x, val_errors)):
            ax1r.scatter(xi, ve, color=colors[i], s=60, zorder=3, marker='o')
        ax1r.set_ylabel("Validation Error", color="k")
        lines2, labels2 = ax1r.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []

    ax1.set_title("Cluster Performance (Train vs Validation)" if has_val else "Cluster Performance (Train)")
    ax1.set_ylabel("Train Error", color="k")

    ax1.set_xticks(x)
    ax1.set_xticklabels(cluster_keys)
    ax1.set_xlabel("Cluster Key")
    ax1.grid(which="both", linestyle='--', linewidth=0.5, zorder=0)

    ax1.text(
        0.5, -0.2,
        f"Counts: {counts_dict}",
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment='top',
        horizontalalignment='center'
    )

    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    plt.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_free_param_heatmap(optimizer, show=False, fig_path=None, fig_config=None):

    final_models_ = optimizer.final_models.drop(columns=["cluster"])
    corr_matrix = final_models_.corr()

    # --- PLOTTING ---
    if fig_config is None: fig_config = {}
    figsize = fig_config.get("figsize", (8, 6))
    # dpi = fig_config.get("dpi", 100)
    plt.figure(figsize=figsize, dpi=DPI)
    sns.heatmap(
        corr_matrix,
        annot=True,        # show numbers
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5
    )
    plt.title("Parameter Correlation Matrix (Final Population)")
    
    plt.tight_layout()
    if fig_path is not None: plt.savefig(fig_path, dpi=DPI)    
    if show: plt.show()
    else: plt.close()


def plot_PCA_of_final_models(optimizer, show=False, fig_path=None, fig_config=None, plot_noise=False):

    final_models = optimizer.final_models
    counts_dict = final_models["cluster"].value_counts().to_dict()

    # Split data
    clustered = final_models[final_models["cluster"] != -1]
    noise = final_models[final_models["cluster"] == -1]

    
    # Prepare PCA on ALL data (important!)
    final_models_ = final_models.drop(columns=["cluster"])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(final_models_)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Split PCA results
    mask_clustered = final_models["cluster"] != -1
    mask_noise = final_models["cluster"] == -1

    X_pca_clustered = X_pca[mask_clustered]
    X_pca_noise = X_pca[mask_noise]

    clusters = final_models.loc[mask_clustered, "cluster"].values
    unique_clusters = np.unique(clusters)

    # Colormap for clusters
    # cmap = plt.get_cmap("tab10", len(unique_clusters))
    cmap = optimizer.cmap

    bounds = np.arange(len(unique_clusters) + 1) - 0.5
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm(bounds, cmap.N)

    # --- PLOTTING ---

    if fig_config is None: fig_config = {}
    figsize = fig_config.get("figsize", (4, 4))
    # dpi = fig_config.get("dpi", 100)

    fig = plt.figure(figsize=figsize, dpi=DPI)

    if plot_noise:
        if len(X_pca_noise) > 0:
            plt.scatter(
                X_pca_noise[:, 0],
                X_pca_noise[:, 1],
                color="gray",
                alpha=0.3,
                s=1,
                label="Noise"
            )

    # 🔹 Plot clustered points
    scatter = plt.scatter(
        X_pca_clustered[:, 0],
        X_pca_clustered[:, 1],
        c=clusters,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        s=5,
        label="_nolegend_"
    )

    plt.text(
    0.5, -0.2,
    f"Counts: {counts_dict}",
    transform=plt.gca().transAxes,
    fontsize=8,
    verticalalignment='top',
    horizontalalignment='center'
    )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA Projection of Parameter Space")

    # Colorbar for clusters
    cbar = plt.colorbar(scatter, ticks=unique_clusters)
    cbar.set_label("Cluster ID")

    plt.grid(which="both", linestyle='--', linewidth=0.5, zorder=0)
    # plt.legend()
    plt.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path, dpi=DPI)

    if show:
        plt.show()
    else:
        plt.close(fig)