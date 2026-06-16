import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import polars as pl
    from scipy.io import loadmat
    import h5py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pcafeat import select_pca_features, select_ttest_features


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Fuzzy PCA Dimensionality Reduction Analysis

    The goals of this notebook are:
    - Reproduce the results of [this paper](https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.1121/134875/Extraction-of-robust-functional-connectivity) using the `SRPB` dataset.
    - Perturb the FC matrix extraction.
    - Assess how the perturbations affect the original results.
    - Do the same steps for the `BMB` dataset.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reproduction of the paper results using the `SRPB` dataset
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Loading the harmonized FC matrices and their metadata
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start by loading the harmonized FC matrices metadata

    Please note that we are using the Glasser parcelation with the global signal regression.
    """)
    return


@app.cell
def _():
    harmonized_srbp_fc_matrices_path = Path(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_srpb_ts_harmonized/all_data_sub_Glasser_GSR1.csv"
    )
    return (harmonized_srbp_fc_matrices_path,)


@app.cell
def _(harmonized_srbp_fc_matrices_path):
    harmonized_srbp_fc_matrices_metadata_df = pl.read_csv(
        harmonized_srbp_fc_matrices_path
    )
    harmonized_srbp_fc_matrices_metadata_df.head()
    return (harmonized_srbp_fc_matrices_metadata_df,)


@app.cell
def _(harmonized_srbp_fc_matrices_metadata_df):
    print(harmonized_srbp_fc_matrices_metadata_df.columns)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And load the harmonized FC matrices themselves.
    """)
    return


@app.cell
def _():
    harmonized_srbp_fc_matrices = []


    with h5py.File(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_srpb_ts_harmonized/all_data_con_Glasser_GSR1_ortho1_harmonized.mat",
        "r",
    ) as f:
        harmonized_srbp_fc_matrices = f["X"][:]

    harmonized_srbp_fc_matrices
    return (harmonized_srbp_fc_matrices,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We now want to merge the metadata DF and the FC_matrices
    """)
    return


@app.cell
def _(harmonized_srbp_fc_matrices, harmonized_srbp_fc_matrices_metadata_df):
    harmonized_srbp_fc_matrices_df = (
        harmonized_srbp_fc_matrices_metadata_df.with_columns(
            harmonized_fc_matrix=harmonized_srbp_fc_matrices
        )
    )
    return (harmonized_srbp_fc_matrices_df,)


@app.cell
def _(harmonized_srbp_fc_matrices_df):
    harmonized_srbp_fc_matrices_df["harmonized_fc_matrix"].head(1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Analyze the dataframe
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per disease
    """)
    return


@app.cell
def _(harmonized_srbp_fc_matrices_df):
    harmonized_srbp_fc_matrices_df.group_by("diag").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per sex
    """)
    return


@app.cell
def _(harmonized_srbp_fc_matrices_df):
    harmonized_srbp_fc_matrices_df.group_by("sex").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per age
    """)
    return


@app.cell
def _(harmonized_srbp_fc_matrices_df):
    harmonized_srbp_fc_matrices_df.group_by("age").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per BDI score
    """)
    return


@app.cell
def _(harmonized_srbp_fc_matrices_df):
    harmonized_srbp_fc_matrices_df.group_by("bdi").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per site
    """)
    return


@app.cell
def _(harmonized_srbp_fc_matrices_df):
    harmonized_srbp_fc_matrices_df.group_by("site").len()
    return


@app.cell
def _(harmonized_srbp_fc_matrices_df):
    harmonized_srbp_fc_matrices_df.group_by("meanFD").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compute and plot the PCA feature selection
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Define a function to calculate the PCA and Ttest features
    """)
    return


@app.cell
def _(DataFrame, function):
    def calculate_features(
        target_str: str,
        target_filters: function,
        harmonized_fc_matrices_df: DataFrame,
        plot_dir: str,
        ttest_dir,
        ttest_output_prefix: str,
    ) -> tuple:
        harmonized_fc_matrices_pandas_df = harmonized_fc_matrices_df.to_pandas()

        target_col = harmonized_fc_matrices_pandas_df[target_str]

        filtered_df = harmonized_fc_matrices_pandas_df[target_filters(target_col)]

        target = filtered_df[target_str]

        harmonized_fc_matrix_pandas_df = filtered_df[
            "harmonized_fc_matrix"
        ].to_frame()

        col_name = harmonized_fc_matrix_pandas_df.columns[0]

        df_X_train = pd.DataFrame(
            harmonized_fc_matrix_pandas_df[col_name].tolist(),
            index=harmonized_fc_matrix_pandas_df.index,
        )

        (cons, cons_pc) = select_pca_features(
            df_X_train=df_X_train,
            target=target,
            method_pick_pca="fdr_bh",
            method_pick_con="fdr_bh",
            fig_dir=plot_dir,
            fig_plot=True,
        )

        if target.nunique() == 2:
            (
                selected_indices,
                t_statistics,
                p_values,
            ) = select_ttest_features(
                df_X_train=df_X_train,
                target=target,
                output_dir=ttest_dir,
                output_prefix=ttest_output_prefix,
                save_results=True,
            )

            return cons, cons_pc, selected_indices, t_statistics, p_values

        else:
            return cons, cons_pc

    return (calculate_features,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Define a function to plot the p values
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that this plot code was generated by Qwen3.7-Plus
    """)
    return


@app.function
def plot_p_values(p_values):

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(p_values, bins=50, color="skyblue", edgecolor="black")
    ax.set_xlabel("Raw p-value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of raw p-values")
    ax.axvline(0.05, color="red", linestyle="--", label="p=0.05")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- 2. QQ Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    sorted_p = np.sort(p_values)
    expected_p = np.linspace(1 / len(p_values), 1, len(p_values))

    ax.scatter(
        -np.log10(expected_p),
        -np.log10(sorted_p + 1e-300),
        alpha=0.5,
        color="purple",
        s=10,
    )
    max_val = max(-np.log10(expected_p))
    ax.plot(
        [0, max_val], [0, max_val], "r--", label="Null hypothesis (Uniform)"
    )

    ax.set_xlabel("Expected -log10(p-value)")
    ax.set_ylabel("Observed -log10(p-value)")
    ax.set_title("QQ Plot of p-values")
    ax.legend()
    plt.tight_layout()
    plt.show()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Calculate the differente PCA and T test features for the following metrics:
        - Mental Depressive Disorder (MDD) vs Healthy Control (HC)
        - Age
        - Bdi
        - Sex
        - Site
        - Mean FD
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### PCA and T test features the for MDD vs HC metric
    """)
    return


@app.cell
def _():
    plot_dir = "./res/pca-dim-reduction/plots/"
    t_test_dir = "./res/pca-dim-reduction/t-tests/"
    return plot_dir, t_test_dir


@app.cell
def _():
    diag_target_str = "diag"
    return (diag_target_str,)


@app.cell
def _():
    mdd_vs_hc_ttest_output_prefix = "ttest_mdd_vs_hc"
    return (mdd_vs_hc_ttest_output_prefix,)


@app.function
def mdd_vs_hc_target_filter(target_col):
    return (
        (target_col != -1000)
        & pd.notna(target_col)
        & ((target_col == 0) | (target_col == 2))
    )


@app.cell
def _(
    calculate_features,
    diag_target_str,
    harmonized_srbp_fc_matrices_df,
    mdd_vs_hc_ttest_output_prefix,
    plot_dir,
    t_test_dir,
):
    (
        mdd_vs_hc_cons,
        mdd_vs_hc_cons_pc,
        mdd_vs_hc_selected_indices,
        mdd_vs_hc_t_statistics,
        mdd_vs_hc_p_values,
    ) = calculate_features(
        diag_target_str,
        mdd_vs_hc_target_filter,
        harmonized_srbp_fc_matrices_df,
        plot_dir,
        t_test_dir,
        mdd_vs_hc_ttest_output_prefix,
    )
    return (mdd_vs_hc_p_values,)


@app.cell
def _(diag_target_str, plot_dir):
    mo.image(src=plot_dir + diag_target_str + ".png")
    return


@app.cell
def _(mdd_vs_hc_p_values):
    plot_p_values(mdd_vs_hc_p_values)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### PCA features for the age metric
    """)
    return


@app.cell
def _():
    age_target_str = "age"
    return (age_target_str,)


@app.cell
def _():
    age_ttest_output_prefix = "ttest_age"
    return (age_ttest_output_prefix,)


@app.function
def age_target_filter(target_col):
    return (target_col != -1000) & pd.notna(target_col)


@app.cell
def _(
    age_target_str,
    age_ttest_output_prefix,
    calculate_features,
    harmonized_srbp_fc_matrices_df,
    plot_dir,
    t_test_dir,
):
    age_cons, age_cons_pc = calculate_features(
        age_target_str,
        age_target_filter,
        harmonized_srbp_fc_matrices_df,
        plot_dir,
        t_test_dir,
        age_ttest_output_prefix,
    )
    return


@app.cell
def _(age_target_str, plot_dir):
    mo.image(src=plot_dir + age_target_str + ".png")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### PCA features for the BDI metric
    """)
    return


@app.cell
def _():
    bdi_target_str = "bdi"
    return (bdi_target_str,)


@app.cell
def _():
    bdi_ttest_output_prefix = "ttest_bdi"
    return (bdi_ttest_output_prefix,)


@app.function
def bdi_target_filter(target_col):
    return (target_col != -1000) & pd.notna(target_col)


@app.cell
def _(
    bdi_target_str,
    bdi_ttest_output_prefix,
    calculate_features,
    harmonized_srbp_fc_matrices_df,
    plot_dir,
    t_test_dir,
):
    bdi_cons, bdi_cons_pc = calculate_features(
        bdi_target_str,
        bdi_target_filter,
        harmonized_srbp_fc_matrices_df,
        plot_dir,
        t_test_dir,
        bdi_ttest_output_prefix,
    )
    return


@app.cell
def _(bdi_target_str, plot_dir):
    mo.image(src=plot_dir + bdi_target_str + ".png")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### PCA and T test features the sex metric
    """)
    return


@app.cell
def _():
    sex_target_str = "sex"
    return (sex_target_str,)


@app.cell
def _():
    sex_ttest_output_prefix = "ttest_sex"
    return (sex_ttest_output_prefix,)


@app.function
def sex_target_filter(target_col):
    return (target_col != -1000) & pd.notna(target_col)


@app.cell
def _(
    calculate_features,
    harmonized_srbp_fc_matrices_df,
    plot_dir,
    sex_target_str,
    sex_ttest_output_prefix,
    t_test_dir,
):
    sex_cons, sex_cons_pc, sex_selected_indices, sex_t_statistics, sex_p_values = (
        calculate_features(
            sex_target_str,
            sex_target_filter,
            harmonized_srbp_fc_matrices_df,
            plot_dir,
            t_test_dir,
            sex_ttest_output_prefix,
        )
    )
    return (sex_p_values,)


@app.cell
def _(plot_dir, sex_target_str):
    mo.image(src=plot_dir + sex_target_str + ".png")
    return


@app.cell
def _(sex_p_values):
    plot_p_values(sex_p_values)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### PCA features the site metric
    """)
    return


@app.cell
def _():
    site_target_str = "site"
    return (site_target_str,)


@app.cell
def _():
    site_ttest_output_prefix = "ttest_site"
    return (site_ttest_output_prefix,)


@app.function
def site_target_filter(target_col):
    return pd.notna(target_col)


@app.cell
def _(
    calculate_features,
    harmonized_srbp_fc_matrices_df,
    plot_dir,
    site_target_str,
    site_ttest_output_prefix,
    t_test_dir,
):
    (
        site_cons,
        site_cons_pc,
    ) = calculate_features(
        site_target_str,
        site_target_filter,
        harmonized_srbp_fc_matrices_df,
        plot_dir,
        t_test_dir,
        site_ttest_output_prefix,
    )
    return


@app.cell
def _(plot_dir, site_target_str):
    mo.image(src=plot_dir + site_target_str + ".png")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### PCA features the FD metric
    """)
    return


@app.cell
def _():
    fd_target_str = "meanFD"
    return (fd_target_str,)


@app.cell
def _():
    fd_ttest_output_prefix = "ttest_fd"
    return (fd_ttest_output_prefix,)


@app.function
def fd_target_filter(target_col):
    return pd.notna(target_col)


@app.cell
def _(
    calculate_features,
    fd_target_str,
    fd_ttest_output_prefix,
    harmonized_srbp_fc_matrices_df,
    plot_dir,
    t_test_dir,
):
    (
        fd_cons,
        fd_cons_pc,
    ) = calculate_features(
        fd_target_str,
        fd_target_filter,
        harmonized_srbp_fc_matrices_df,
        plot_dir,
        t_test_dir,
        fd_ttest_output_prefix,
    )
    return


@app.cell
def _(fd_target_str, plot_dir):
    mo.image(src=plot_dir + fd_target_str + ".png")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
