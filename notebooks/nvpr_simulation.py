# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.8",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "plotly==6.7.0",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    from matplotlib import gridspec
    from matplotlib.colors import LogNorm
    import matplotlib.cm as cm

    return LogNorm, cm, gridspec, mcolors, mo, np, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simulation of the impact of numerical variability on the Numerical Population Variability Ratio (NPVR) and Cohen's delta
    > v1.0.0
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The goal of this notebook is to reproduce the different results of [this one](https://github.com/mina94az/Numerical-Variability-of-functional-MRI-Graph-Measures/blob/main/notebooks/simulation.ipynb), therefore it takes heavy inspiration from it
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create the low and high variability distributions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup the different constants
    """)
    return


@app.cell
def _(mo):
    output_checkbox = mo.ui.checkbox(label="Save output figs", value=False)
    sample_number = mo.ui.number(
        start=1, stop=100, step=1, label="Number of sample", value=10
    )
    low_variability_number = mo.ui.number(
        start=0, stop=1, step=0.1, label="Low variability scale", value=0.1
    )
    high_variability_number = mo.ui.number(
        start=0, stop=1, step=0.1, label="High variability scale", value=0.4
    )
    return (
        high_variability_number,
        low_variability_number,
        output_checkbox,
        sample_number,
    )


@app.cell
def _(
    high_variability_number,
    low_variability_number,
    mo,
    output_checkbox,
    sample_number,
):
    mo.vstack(
        [
            mo.hstack([output_checkbox, sample_number], widths="equal"),
            mo.hstack(
                [low_variability_number, high_variability_number], widths="equal"
            ),
        ]
    )
    return


@app.cell
def _(high_variability_number, low_variability_number, np, sample_number):
    np.random.seed(41)

    SAMPLE_NUMBER = sample_number.value

    DISTRIBUTION_MEANS = np.linspace(1, 5, SAMPLE_NUMBER)

    LOW_VARIABILITY_SCALE = low_variability_number.value
    HIGH_VARIABILITY_SCALE = high_variability_number.value

    print(
        f"Constant values:\n\t{SAMPLE_NUMBER=}\n\t{DISTRIBUTION_MEANS=}\n\t{LOW_VARIABILITY_SCALE=}\n\t{HIGH_VARIABILITY_SCALE=}"
    )
    return (
        DISTRIBUTION_MEANS,
        HIGH_VARIABILITY_SCALE,
        LOW_VARIABILITY_SCALE,
        SAMPLE_NUMBER,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary of the obtained figures
    """)
    return


@app.cell
def _(distribution_fig, mo, npvr_variation_fig, num_variation_fig):
    mo.hstack(
        [distribution_fig, num_variation_fig, npvr_variation_fig],
        justify="center",
        gap=0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup the distributions
    """)
    return


@app.cell
def _(
    DISTRIBUTION_MEANS,
    HIGH_VARIABILITY_SCALE,
    LOW_VARIABILITY_SCALE,
    SAMPLE_NUMBER,
    np,
):
    low_var_dists = np.random.normal(
        loc=DISTRIBUTION_MEANS,
        scale=LOW_VARIABILITY_SCALE,
        size=(SAMPLE_NUMBER, SAMPLE_NUMBER),
    )
    high_var_dists = np.random.normal(
        loc=DISTRIBUTION_MEANS,
        scale=HIGH_VARIABILITY_SCALE,
        size=(SAMPLE_NUMBER, SAMPLE_NUMBER),
    )
    return high_var_dists, low_var_dists


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the distributions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup the different colors
    """)
    return


@app.cell
def _(SAMPLE_NUMBER, cm, mcolors):
    colors = [
        mcolors.rgb2hex(cm.viridis(i / max(SAMPLE_NUMBER - 1, 1)))
        for i in range(SAMPLE_NUMBER)
    ]

    print(f"{colors=}")
    return (colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prepare the plot
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Set a fixed range to zoom in
    """)
    return


@app.cell
def _():
    _x_range_left = (0, 6)
    _x_range_right = (0, 6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Define a function to create the left and right parts of the plot
    """)
    return


@app.cell
def _(colors, np, sns):
    def create_distributions_side_subplot_values(
        distributions, side_axes, x_range, num_samples
    ):
        means = []
        ref1s = []
        ref2s = []
        for i, distribution in enumerate(distributions.T):
            ax = side_axes[i]

            # =======Taken as is from the original notebook=======

            # Draw horizontal baseline
            ax.axhline(y=0, color="lightgray", linewidth=1, zorder=1)

            # Plot KDE
            sns.kdeplot(distribution, fill=True, alpha=0.6, ax=ax, color=colors[i])
            sns.kdeplot(distribution, ax=ax, color="k", lw=1)

            ax.set_title(
                f"Sample {i + 1}", fontsize=24, fontweight="bold", loc="left"
            )
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_yticks([])
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_xlim(x_range)

            # Remove spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            # ---- Reference points ----
            mean_val = np.mean(distribution)
            ref1 = distribution[num_samples // 5]
            ref2 = distribution[num_samples // 2]

            # Store values for projection
            means.append(mean_val)
            ref1s.append(ref1)
            ref2s.append(ref2)

            side_axes[i] = ax

            # ======================================================

        return side_axes, means, ref1s, ref2s

    return (create_distributions_side_subplot_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Define a function to create the projection axes
    """)
    return


@app.cell
def _(colors):
    def create_axes(y_positions, means, ref1s, ref2s, x_range, ax_projection):

        # =======Taken as is from the original notebook=======
        for y_pos in y_positions:
            ax_projection.axhline(
                y=y_pos, color="lightgray", linestyle="-", alpha=0.7, linewidth=1.5
            )

        # Project means (circles) - each with its distribution color
        for i, mean_val in enumerate(means):
            ax_projection.scatter(
                mean_val,
                y_positions[0],
                color=colors[i],
                marker="o",
                s=60,
                alpha=0.8,
            )

        # Project ref1 (x markers) - each with its distribution color
        for i, ref1_val in enumerate(ref1s):
            ax_projection.scatter(
                ref1_val, y_positions[1], color=colors[i], marker="x", s=80
            )

        # Project ref2 (x markers) - each with its distribution color
        for i, ref2_val in enumerate(ref2s):
            ax_projection.scatter(
                ref2_val, y_positions[2], color=colors[i], marker="x", s=80
            )

        ax_projection.set_xlim(x_range)
        ax_projection.set_ylim(0.2, 0.8)
        ax_projection.set_yticks(y_positions)
        ax_projection.set_yticklabels(
            ["Mean", "Ref1", "Ref2"], fontsize=20, fontweight="bold"
        )
        ax_projection.set_xlabel("Value", fontsize=11, fontweight="bold")

        # Remove spines for projection panel
        ax_projection.spines["top"].set_visible(False)
        ax_projection.spines["right"].set_visible(False)
        ax_projection.spines["left"].set_visible(False)
        ax_projection.spines["bottom"].set_visible(False)

        ax_projection.set_xticks([])
        # ======================================================

        return ax_projection

    return (create_axes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create the plot
    """)
    return


@app.cell
def _(
    SAMPLE_NUMBER,
    create_axes,
    create_distributions_side_subplot_values,
    gridspec,
    high_var_dists,
    low_var_dists,
    output_checkbox,
    plt,
):
    distribution_fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(
        SAMPLE_NUMBER + 1,
        2,
        height_ratios=[1] * SAMPLE_NUMBER + [0.8],
        width_ratios=[1, 1],
        hspace=0.3,
        wspace=0.15,
    )

    left_axes = [
        distribution_fig.add_subplot(gs[i, 0]) for i in range(SAMPLE_NUMBER)
    ]
    right_axes = [
        distribution_fig.add_subplot(gs[i, 1]) for i in range(SAMPLE_NUMBER)
    ]
    left_ax_projection = distribution_fig.add_subplot(gs[SAMPLE_NUMBER, 0])
    right_ax_projection = distribution_fig.add_subplot(gs[SAMPLE_NUMBER, 1])

    x_range_left = (0, 6)
    x_range_right = (0, 6)

    left_axes, left_means, left_ref1s, left_ref2s = (
        create_distributions_side_subplot_values(
            low_var_dists, left_axes, x_range_left, SAMPLE_NUMBER
        )
    )

    right_axes, right_means, right_ref1s, right_ref2s = (
        create_distributions_side_subplot_values(
            high_var_dists, right_axes, x_range_right, SAMPLE_NUMBER
        )
    )

    y_positions = [0.7, 0.5, 0.3]

    left_ax_projection = create_axes(
        y_positions,
        left_means,
        left_ref1s,
        left_ref2s,
        x_range_left,
        left_ax_projection,
    )
    right_ax_projection = create_axes(
        y_positions,
        right_means,
        right_ref1s,
        right_ref2s,
        x_range_right,
        right_ax_projection,
    )

    # Add panel titles
    distribution_fig.text(
        0.3,
        0.9,
        "Group A: Low Numerical Variability",
        fontsize=24,
        fontweight="bold",
        ha="center",
    )
    distribution_fig.text(
        0.70,
        0.9,
        "Group B: High Numerical Variability",
        fontsize=24,
        fontweight="bold",
        ha="center",
    )

    if output_checkbox.value:
        distribution_fig.savefig(
            "res/npvr-simulation/populations_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
    distribution_fig
    return (distribution_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute the sigma values (Population and numerical variabilities)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define a function with the wanted metrics
    """)
    return


@app.cell
def _(np):
    def compute_metrics(distribution):

        within_var_each = np.var(distribution, axis=0, ddof=1)
        within_var_mean = np.mean(within_var_each)
        sigma_num = np.sqrt(within_var_mean)
        across_var_each = np.var(distribution, axis=1, ddof=1)
        across_var_mean = np.mean(across_var_each)
        sigma_pop = np.sqrt(across_var_mean)

        return (
            within_var_each,
            within_var_mean,
            sigma_num,
            across_var_each,
            across_var_mean,
            sigma_pop,
        )

    return (compute_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute for the low variability
    """)
    return


@app.cell
def _(compute_metrics, low_var_dists):
    (
        low_within_var_each,
        low_within_var_mean,
        low_sigma_num,
        low_across_var_each,
        low_across_var_mean,
        low_sigma_pop,
    ) = compute_metrics(low_var_dists)
    return low_sigma_num, low_sigma_pop


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute for the high variability distribution
    """)
    return


@app.cell
def _(compute_metrics, high_var_dists):
    (
        high_within_var_each,
        high_within_var_mean,
        high_sigma_num,
        high_across_var_each,
        high_across_var_mean,
        high_sigma_pop,
    ) = compute_metrics(high_var_dists)
    return high_sigma_num, high_sigma_pop


@app.cell
def _(high_sigma_num, high_sigma_pop, low_sigma_num, low_sigma_pop):
    # Compute the low and high variability NPVR
    low_npvr = low_sigma_num / low_sigma_pop
    high_npvr = high_sigma_num / high_sigma_pop
    return high_npvr, low_npvr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Print the results
    """)
    return


@app.cell
def _(
    high_npvr,
    high_sigma_num,
    high_sigma_pop,
    low_npvr,
    low_sigma_num,
    low_sigma_pop,
):
    print("Low σ_num =", low_sigma_num)
    print("Low σ_pop = ", low_sigma_pop)
    print("High σ_num = ", high_sigma_num)
    print("High σ_pop = ", high_sigma_pop)

    print("Low NPVR = ", low_npvr)
    print("High NPVR = ", high_npvr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the NVPR, $\sigma_{pop}$ and $\sigma_{num}$ variations

    > This section is mostly taken as is from the original notebook as it's hard to get the same plot with a different code
    >
    >And the original code was already pretty good
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup the different plot variables
    """)
    return


@app.cell
def _(np):
    # set the grid (σ_num and σ_pop both in [0.001, 1.5])
    x_grid_space = np.linspace(0.001, 1, 400)  # σ_num
    y_grid_space = np.linspace(0.001, 3, 400)  # σ_pop
    X_grid, Y_grid = np.meshgrid(x_grid_space, y_grid_space)

    # Variability ratio
    NPVR = X_grid / Y_grid
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create the plot
    """)
    return


app._unparsable_cell(
    r"""
    plt.rcParams.update(
        {
            "font.size": 28,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "legend.fontsize": 16,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
        }
    )


    num_variation_fig, ax = plt.subplots(figsize=(16, 12))

    # Add a log normalization color bar and background for visibility
    norm = LogNorm(vmin=0.01, vmax=5)
    pcm = ax.pcolormesh(
        X_grid, Y_grid, NPVR, shading="auto", cmap="Greys", norm=norm
    )
    _cbar = num_variation_fig.colorbar(pcm, ax=ax)
    _cbar.set_label(
        r"$\nu_{\mathrm{npv}} = \sigma_{\mathrm{num}}/\sigma_{\mathrm{pop}}$"
    )

    # Add NPVR contour lines
    _contour_levels = [0.1, 0.2, 0.5, 1.0]
    _contours = ax.contour(
        X_grid,
        Y_grid,
        NPVR,
        levels=_contour_levels,
        colors="black",
        linewidths=2.5,
        linestyles="solid",
        alpha=0.8,
    )
    ax.clabel(
        _contours,
        inline=True,
        fontsize=24,
        fmt=lambda v: r"$\nu_{\mathrm{npv}}$=%.2f" % v,
        inline_spacing=10,
        colors="black",
    )


    # Plot the NPVR crosses
    ax.scatter(
        low_sigma_num,
        low_sigma_pop,
        color="green",
        s=500,
        marker="x",
        linewidth=4,
        label=rf"Group A ($\nu_{{\mathrm{{npv}}}}$={low_npvr:.3f})",
        zorder=10,
    )

    ax.scatter(
        high_sigma_num, high_sigma_pop
        color="orange",
        s=500,
        marker="x",
        linewidth=4,
        label=rf"Group B ($\nu_{{\mathrm{{npv}}}}$={high_npvr:.3f})",
        zorder=10,
    )


    # Axes, title, legend
    ax.set_xlim(x_grid_space.min(), x_grid_space.max())
    ax.set_ylim(y_grid_space.min(), y_grid_space.max())
    ax.set_xlabel(r"$\sigma_{\mathrm{num}}$ (Numerical Variability)", fontsize=28)
    ax.set_ylabel(r"$\sigma_{\mathrm{pop}}$ (Sample Variability)", fontsize=28)
    ax.grid(True, alpha=0.3)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    plt.tight_layout()
    if output_checkbox.value:
        num_variation_fig.savefig(
            "res/npvr-simulation/npvr_dpop_dnum_variation.png",
            dpi=300,
            bbox_inches="tight",
        )
    num_variation_fig
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the NVPR, sample size and Cohen's delta variations

    > This section is mostly taken as is from the original notebook as it's hard to get the same plot with a different code

    > And the original code was already pretty good
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create a function to compute Cohen's d based on the NPVR
    """)
    return


@app.cell
def _(np):
    def cohens_d(n, npvr):
        return (2 / np.sqrt(n)) * npvr

    return (cohens_d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Set the different plot variables
    """)
    return


@app.cell
def _(cohens_d, np):
    # Plot parameters
    sample_sizes = np.logspace(1, 5, 300)  # 10 → 100,000
    variability_ratios = np.linspace(0.0, 1.0, 300)

    # Create the plot grid
    N, VR = np.meshgrid(sample_sizes, variability_ratios)
    D = cohens_d(N, VR)
    D_safe = np.maximum(D, 1e-5)
    return D, D_safe, N, VR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Create the plot

    > Please note that this plot gives a different result than the original one regarding the NPVR values.
    >
    > This is because the original paper had hard coded them to ```low_npvr = 0.287``` and ```high_nvr = 0.496``` instead of using the results provided by the distributions
    """)
    return


@app.cell
def _(D, D_safe, LogNorm, N, VR, high_npvr, low_npvr, output_checkbox, plt):
    # Setup styling
    plt.rcParams.update(
        {
            "font.size": 28,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "legend.fontsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
        }
    )

    # Create figure and axis explicitly
    npvr_variation_fig, _ax = plt.subplots(figsize=(18, 14))

    # Plot the background
    contourf = _ax.contourf(
        N, VR, D_safe, levels=50, cmap="pink_r", norm=LogNorm(vmin=1e-5, vmax=1)
    )

    # Add contour lines
    _contour_levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    _contours = _ax.contour(
        N,
        VR,
        D,
        levels=_contour_levels,
        colors="black",
        linewidths=2.5,
        linestyles="solid",
        alpha=0.8,
    )

    labels = _ax.clabel(
        _contours,
        inline=True,
        fontsize=24,
        fmt=lambda v: r"$\sigma_{\mathrm{d}}$ = %.2f" % v,
        inline_spacing=12,
        colors="black",
    )

    angles = [7, 48, 67, 71, 72]
    for a, txt in zip(angles, labels):
        txt.set_rotation(a)
        txt.set_rotation_mode("anchor")
        txt.set_transform_rotates_text(False)
        txt.set_ha("center")
        txt.set_va("center")

    # Sample sizes and markers
    dot_samples = [25, 150, 500, 2000, 5000]

    for _x in dot_samples:
        _ax.scatter(
            _x, low_npvr, color="green", s=500, marker="x", linewidth=3, zorder=5
        )
        _ax.vlines(
            _x,
            ymin=0,
            ymax=1,
            color="darkblue",
            linewidth=2,
            linestyles="dashed",
            alpha=0.8,
        )
        _ax.text(
            _x * 0.85,
            0.06,
            rf"$n={{{int(_x)}}}$",
            fontsize=20,
            rotation=90,
            color="darkblue",
        )

    _ax.hlines(
        y=low_npvr,
        xmin=0,
        xmax=1e5,
        color="green",
        linewidth=2,
        linestyles="dashed",
    )

    for _x in dot_samples:
        _ax.scatter(
            _x, high_npvr, color="orange", marker="x", linewidth=3, s=500, zorder=5
        )

    _ax.hlines(
        y=high_npvr,
        xmin=0,
        xmax=1e5,
        color="orange",
        linewidth=2,
        linestyles="dashed",
    )

    # Colorbar and labels
    _cbar = npvr_variation_fig.colorbar(
        contourf,
        ax=_ax,
        label=r"Cohen's d Variability ($\sigma_{\mathrm{d}}$, log scale)",
    )

    _ax.set_xscale("log")
    _ax.set_xlabel("Sample Size (log scale)")
    _ax.set_ylabel(
        r"Numerical-Population Variability Ratio ($\nu_{\mathrm{npv}}$)"
    )
    _ax.set_ylim(0, 0.8)
    _ax.set_xlim(10, 5e4)

    npvr_variation_fig.tight_layout()

    if output_checkbox.value:
        npvr_variation_fig.savefig(
            "res/npvr-simulation/npvr_sample_size_cohens_delta_variation.png",
            dpi=300,
            bbox_inches="tight",
        )

    npvr_variation_fig
    return (npvr_variation_fig,)


if __name__ == "__main__":
    app.run()
