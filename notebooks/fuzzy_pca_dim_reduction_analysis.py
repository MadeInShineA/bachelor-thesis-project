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
    from pcafeat import select_pca_features


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

    print("Loaded successfully!")
    print("Shape:", harmonized_srbp_fc_matrices.shape)
    print("Data type:", harmonized_srbp_fc_matrices.dtype)

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
    ### Compute the PCA dimentionality reduction
    """)
    return


if __name__ == "__main__":
    app.run()
