import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import os
    from typing import Callable
    import marimo as mo
    from pathlib import Path
    import polars as pl
    from scipy.io import loadmat
    from collections import Counter
    import h5py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pcafeat import select_pca_features, select_ttest_features
    import scipy.io as sio
    from scipy.ndimage import center_of_mass
    from nilearn import plotting
    from matplotlib.lines import Line2D
    import json


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
    harmonized_srbp_fc_matrices_metadata_path = Path(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_srpb_ts_harmonized/all_data_sub_Glasser_GSR1.csv"
    )
    return (harmonized_srbp_fc_matrices_metadata_path,)


@app.cell
def _(harmonized_srbp_fc_matrices_metadata_path):
    harmonized_srbp_fc_matrices_metadata_df = pl.read_csv(
        harmonized_srbp_fc_matrices_metadata_path
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
    And only keep the HC and MDD subjects
    """)
    return


@app.cell
def _(harmonized_srbp_fc_matrices_df):
    harmonized_srbp_fc_matrices_hc_mdd_df = harmonized_srbp_fc_matrices_df.filter(
        pl.col("diag").is_in([0, 2])
    )
    return (harmonized_srbp_fc_matrices_hc_mdd_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analyze the dataframe
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per disease
    """)
    return


@app.cell
def _(harmonized_srbp_fc_matrices_hc_mdd_df):
    harmonized_srbp_fc_matrices_hc_mdd_df.group_by("diag").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per sex
    """)
    return


@app.cell
def _(harmonized_srbp_fc_matrices_hc_mdd_df):
    harmonized_srbp_fc_matrices_hc_mdd_df.group_by("sex").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per age
    """)
    return


@app.cell
def _(harmonized_srbp_fc_matrices_hc_mdd_df):
    harmonized_srbp_fc_matrices_hc_mdd_df.group_by("age").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per BDI score
    """)
    return


@app.cell
def _(harmonized_srbp_fc_matrices_hc_mdd_df):
    harmonized_srbp_fc_matrices_hc_mdd_df.group_by("bdi").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per site
    """)
    return


@app.cell
def _(harmonized_srbp_fc_matrices_hc_mdd_df):
    harmonized_srbp_fc_matrices_hc_mdd_df.group_by("site").len()
    return


@app.cell
def _(harmonized_srbp_fc_matrices_hc_mdd_df):
    harmonized_srbp_fc_matrices_hc_mdd_df.group_by("meanFD").len()
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
    Calculate the differente PCA and T test features for the following metrics:
            - Mental Depressive Disorder (MDD) vs Healthy Control (HC)
            - Age
            - Bdi
            - Sex
            - Site
            - Mean FD
    """)
    return


@app.function
def calculate_features(
    target_str: str,
    target_filters: Callable,
    harmonized_fc_matrices_df: pl.DataFrame,
    alpha_threshold: float,
    plot_dir: str,
    ttest_dir,
    ttest_output_prefix: str,
) -> dict:
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
        pca_extract_multipletests_alpha=alpha_threshold,
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

        return {
            "cons": cons,
            "cons_pc": cons_pc,
            "selected_indices": selected_indices,
            "t_statistics": t_statistics,
            "p_values": p_values,
        }

    else:
        return {
            "cons": cons,
            "cons_pc": cons_pc,
        }


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
def plot_p_values(p_values, metric_name: str):
    # Combine both plots into a single figure (1 row, 2 columns)
    fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 1. Histogram ---
    ax_hist.hist(p_values, bins=50, color="skyblue", edgecolor="black")
    ax_hist.set_xlabel("Raw p-value")
    ax_hist.set_ylabel("Frequency")

    ax_hist.set_title(f"Distribution of raw p-values\n[{metric_name}]")
    ax_hist.axvline(0.05, color="red", linestyle="--", label="p=0.05")
    ax_hist.legend()

    # --- 2. QQ Plot ---
    sorted_p = np.sort(p_values)
    expected_p = np.linspace(1 / len(p_values), 1, len(p_values))

    ax_qq.scatter(
        -np.log10(expected_p),
        -np.log10(sorted_p + 1e-300),
        alpha=0.5,
        color="purple",
        s=10,
        label=f"Observed ({metric_name})",
    )
    max_val = max(-np.log10(expected_p))
    ax_qq.plot(
        [0, max_val], [0, max_val], "r--", label="Null hypothesis (Uniform)"
    )

    ax_qq.set_xlabel("Expected -log10(p-value)")
    ax_qq.set_ylabel("Observed -log10(p-value)")

    ax_qq.set_title(f"QQ Plot of p-values\n[{metric_name}]")
    ax_qq.legend()

    plt.tight_layout()

    return fig


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Define a function to calculate the PCA features extraction for the different metris
    """)
    return


@app.function
def calculate_metrics(
    df: pd.DataFrame,
    metric_dict: dict[str, dict],
    alpha_threshold: float,
    plot_dir: str,
    ttest_dir: str,
) -> dict:

    res = {"results": {}}
    ui_elements = []

    for metric, config in metric_dict.items():
        results = calculate_features(
            target_str=metric,
            target_filters=config["filter_function"],
            harmonized_fc_matrices_df=df,
            alpha_threshold=alpha_threshold,
            plot_dir=plot_dir,
            ttest_dir=ttest_dir,
            ttest_output_prefix=config["prefix"],
        )

        image_path = os.path.join(plot_dir, f"{metric}.png")
        metric_image = mo.image(src=image_path)

        if "p_values" in results:
            p_value_plot = plot_p_values(
                results["p_values"], metric_name=metric
            )
            plot_section = mo.hstack(
                [metric_image, p_value_plot],
                justify="center",
                align="center",
                gap=2,
                widths=[1, 2],
            )
        else:
            plot_section = metric_image

        metric_ui = mo.vstack(
            [
                mo.md(f"#### Results for metric: `{metric}`"),
                plot_section,
                mo.md("---"),
            ]
        )

        ui_elements.append(metric_ui)
        res["results"][metric] = results

    res["ui"] = mo.vstack(ui_elements, gap=3)
    return res


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


@app.cell
def _():
    srbp_plot_dir = "./res/pca-dim-reduction/srbp/plots/"
    srbp_ttest_dir = "./res/pca-dim-reduction/srbp/t-tests/"
    srbp_metadata_dir = "./res/pca-dim-reduction/srbp/metadatas"
    return srbp_metadata_dir, srbp_plot_dir, srbp_ttest_dir


@app.cell
def _():
    metric_dict = {
        "diag": {
            "prefix": "ttest_diag",
            "filter_function": lambda target_col: (
                (target_col != -1000) & pd.notna(target_col)
            ),
        },
        "bdi": {
            "prefix": "ttest_bdi",
            "filter_function": lambda target_col: (
                (target_col != -1000) & pd.notna(target_col)
            ),
        },
        "age": {
            "prefix": "ttest_age",
            "filter_function": lambda target_col: (
                (target_col != -1000) & pd.notna(target_col)
            ),
        },
        "sex": {
            "prefix": "ttest_sex",
            "filter_function": lambda target_col: (
                (target_col != -1000) & pd.notna(target_col)
            ),
        },
        "site": {
            "prefix": "ttest_site",
            "filter_function": lambda target_col: pd.notna(target_col),
        },
        "meanFD": {
            "prefix": "ttest_meanFD",
            "filter_function": lambda target_col: pd.notna(target_col),
        },
    }
    return (metric_dict,)


@app.cell
def _(
    harmonized_srbp_fc_matrices_hc_mdd_df,
    metric_dict,
    srbp_plot_dir,
    srbp_ttest_dir,
):
    srbp_metrics_dict = calculate_metrics(
        df=harmonized_srbp_fc_matrices_hc_mdd_df,
        metric_dict=metric_dict,
        alpha_threshold=0.05,
        plot_dir=srbp_plot_dir,
        ttest_dir=srbp_ttest_dir,
    )

    srbp_results = srbp_metrics_dict["results"]
    srbp_ui = srbp_metrics_dict["ui"]
    return srbp_results, srbp_ui


@app.cell
def _(srbp_ui):
    srbp_ui
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analyze the results of the PCA feature selection
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We first want to check wich PC was selected only in the BDI and diag metrics
    """)
    return


@app.function
def select_mdd_pc(
    result_dict: dict, metrics_to_accept_beside_diag: list
) -> dict:

    diag_pcs = set()
    accepted_pcs = set()
    confound_pcs = set()

    pc_to_confound_metrics = {}

    for metric, value in result_dict.items():
        pcs = set(value.get("cons_pc", []))

        if metric == "diag":
            diag_pcs = pcs
        elif metric in metrics_to_accept_beside_diag:
            accepted_pcs.update(pcs)
        else:
            confound_pcs.update(pcs)
            for pc in pcs:
                if pc not in pc_to_confound_metrics:
                    pc_to_confound_metrics[pc] = []
                pc_to_confound_metrics[pc].append(metric)

    in_diag_and_confounds = diag_pcs & confound_pcs

    in_diag_and_accepted = (diag_pcs & accepted_pcs) - confound_pcs

    only_in_diag = diag_pcs - accepted_pcs - confound_pcs

    detailed_confounds = {}
    for pc in sorted(list(in_diag_and_confounds)):
        detailed_confounds[pc] = pc_to_confound_metrics[pc]

    return {
        "only_in_diag": sorted(list(only_in_diag)),
        "in_diag_and_accepted": sorted(list(in_diag_and_accepted)),
        "in_diag_and_confounds": detailed_confounds,
    }


@app.cell
def _(srbp_results):
    srbp_selected_mdd_pcs = select_mdd_pc(srbp_results, ["bdi"])
    srbp_selected_mdd_pcs
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We now want to plot the different FC of the desired PC
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start by obtaining the brain parcelation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that all the code of this cell has been written by Qwen3.7-Plus and needs to be double-checked
    """)
    return


@app.cell(hide_code=True)
def _():
    coords_mat_path = "/home/cbi/olivier.amacker/parcelation/VTXTBL_HAL-170829.mat"
    mat_data = sio.loadmat(coords_mat_path)

    # ==========================================
    # 1. INSPECT THE ROI STRUCTURE
    # ==========================================
    roi_data = mat_data["ROI"]
    print(f"ROI shape: {roi_data.shape}, dtype: {roi_data.dtype}")

    # Look at the actual structure of the first element
    sample = roi_data[0, 0]
    print(f"\nFirst element type: {type(sample)}")
    if hasattr(sample, "dtype") and sample.dtype.names:
        print(f"Field names: {sample.dtype.names}")
        for field in sample.dtype.names:
            print(f"  {field}: {type(sample[field][0, 0])}")

    # ==========================================
    # 2. EXTRACT REGION NAMES (FIXED)
    # ==========================================
    region_names = []
    network_names = []

    print("\nExtracting region names from ROI struct...")

    # The ROI is a structured array. Let's find the correct field for region names
    for i in range(roi_data.shape[1]):
        item = roi_data[0, i]

        # Try to extract region name from various possible locations
        region_name = f"Region_{i}"
        network_name = "Unknown"

        try:
            # Check if it's a structured array
            if hasattr(item, "dtype") and item.dtype.names:
                # Try 'aal' field first (common in Glasser atlases)
                if "aal" in item.dtype.names:
                    aal_data = item["aal"][0, 0]
                    if hasattr(aal_data, "dtype") and aal_data.dtype.names:
                        # Look for 'lab' or 'str' or 'name' field
                        for field in ["lab", "str", "name", "label"]:
                            if field in aal_data.dtype.names:
                                val = aal_data[field][0, 0]
                                if hasattr(val, "item"):
                                    region_name = str(val.item()).strip()
                                else:
                                    region_name = str(val).strip()
                                break

                # Try direct fields
                if region_name == f"Region_{i}":
                    for field in ["lab", "str", "name", "label", "aal"]:
                        if field in item.dtype.names:
                            val = item[field][0, 0]
                            if hasattr(val, "item"):
                                region_name = str(val.item()).strip()
                            else:
                                region_name = str(val).strip()
                            break

            # If still not found, try to access as object array
            if region_name == f"Region_{i}" and isinstance(item, np.ndarray):
                # Try to find any string in the structure
                def find_string_in_struct(obj, depth=0):
                    if depth > 5:
                        return None
                    if isinstance(obj, str) and len(obj) > 0:
                        return obj
                    if isinstance(obj, np.ndarray):
                        if obj.dtype.kind in ["U", "S", "O"]:
                            for item in obj.flat:
                                result = find_string_in_struct(item, depth + 1)
                                if result:
                                    return result
                        elif obj.dtype.names:
                            for field in obj.dtype.names:
                                result = find_string_in_struct(
                                    obj[field], depth + 1
                                )
                                if result:
                                    return result
                    elif hasattr(obj, "dtype") and obj.dtype.names:
                        for field in obj.dtype.names:
                            result = find_string_in_struct(obj[field], depth + 1)
                            if result:
                                return result
                    return None

                found_str = find_string_in_struct(item)
                if found_str:
                    region_name = found_str

        except Exception as e:
            pass

        region_names.append(region_name)

    print(f"✓ Extracted {len(region_names)} region names.")
    print(f"  First 10: {region_names[:10]}")
    print(f"  Last 10: {region_names[-10:]}")

    # ==========================================
    # 3. CALCULATE 3D COORDINATES FROM 'MAP'
    # ==========================================
    map_data = mat_data["MAP"]
    print(
        f"\nCalculating center of mass for 379 regions from 4D map (shape: {map_data.shape})..."
    )

    voxel_coords = []
    for i in range(379):
        roi_mask = map_data[:, :, :, i]
        if np.any(roi_mask):
            com = center_of_mass(roi_mask)
            voxel_coords.append(com)
        else:
            voxel_coords.append([0, 0, 0])

    voxel_coords = np.array(voxel_coords)

    # Convert to MNI space
    affine = np.array(
        [[-2, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]]
    )

    ones = np.ones((voxel_coords.shape[0], 1))
    vox_homo = np.hstack([voxel_coords, ones])
    node_coords = np.dot(vox_homo, affine.T)[:, :3]
    print(f"✓ Calculated MNI coordinates for {len(node_coords)} regions.")

    glasser_to_network = {
        # Visual1 - Primary/Early Visual
        "V1": "Visual1",
        "V2": "Visual1",
        "V3": "Visual1",
        "V3A": "Visual1",
        # Visual2 - Higher-order Visual
        "V4": "Visual2",
        "V8": "Visual2",
        "V6": "Visual2",
        "V6A": "Visual2",
        "V7": "Visual2",
        "V4t": "Visual2",
        "V3B": "Visual2",
        "V3CD": "Visual2",
        "LO1": "Visual2",
        "LO2": "Visual2",
        "LO3": "Visual2",
        "PIT": "Visual2",
        "FFC": "Visual2",
        "MST": "Visual2",
        "MT": "Visual2",
        "FST": "Visual2",
        "VMV1": "Visual2",
        "VMV2": "Visual2",
        "VMV3": "Visual2",
        "PHA1": "Visual2",
        "PHA2": "Visual2",
        "PHA3": "Visual2",
        "PH": "Visual2",
        "VVC": "Visual2",
        # Somatomotor Network
        "1": "Somatomotor",
        "2": "Somatomotor",
        "3a": "Somatomotor",
        "3b": "Somatomotor",
        "4": "Somatomotor",
        "6d": "Somatomotor",
        "6v": "Somatomotor",
        "6r": "Somatomotor",
        "6a": "Somatomotor",
        "6mp": "Somatomotor",
        "6ma": "Somatomotor",
        "SCEF": "Somatomotor",
        "5m": "Somatomotor",
        "5mv": "Somatomotor",
        "5L": "Somatomotor",
        "OP1": "Somatomotor",
        "OP2-3": "Somatomotor",
        "OP4": "Somatomotor",
        "43": "Somatomotor",
        "MI": "Somatomotor",
        "RI": "Somatomotor",
        "52": "Somatomotor",
        "PFcm": "Somatomotor",
        # Dorsal Attention Network
        "VIP": "Dorsal Attention",
        "LIPd": "Dorsal Attention",
        "LIPv": "Dorsal Attention",
        "MIP": "Dorsal Attention",
        "AIP": "Dorsal Attention",
        "IPS1": "Dorsal Attention",
        "FEF": "Dorsal Attention",
        "PEF": "Dorsal Attention",
        "55b": "Dorsal Attention",
        "i6-8": "Dorsal Attention",
        "s6-8": "Dorsal Attention",
        # Ventral Attention / Salience Network
        "TPOJ1": "Ventral Attention",
        "TPOJ2": "Ventral Attention",
        "TPOJ3": "Ventral Attention",
        "PGp": "Ventral Attention",
        "PF": "Ventral Attention",
        "PFm": "Ventral Attention",
        "PFop": "Ventral Attention",
        "PFt": "Ventral Attention",
        "IP0": "Ventral Attention",
        "IP1": "Ventral Attention",
        "IP2": "Ventral Attention",
        "DVT": "Ventral Attention",
        "PoI1": "Ventral Attention",
        "PoI2": "Ventral Attention",
        "TA2": "Ventral Attention",
        "FOP4": "Ventral Attention",
        "AVI": "Ventral Attention",
        "AAIC": "Ventral Attention",
        "FOP1": "Ventral Attention",
        "FOP2": "Ventral Attention",
        "FOP3": "Ventral Attention",
        "FOP5": "Ventral Attention",
        "PI": "Ventral Attention",
        "Ig": "Ventral Attention",
        "STSva": "Ventral Attention",
        "STSda": "Ventral Attention",
        "STSdp": "Ventral Attention",
        "STSvp": "Ventral Attention",
        "PBelt": "Ventral Attention",
        "A5": "Ventral Attention",
        "MBelt": "Ventral Attention",
        "LBelt": "Ventral Attention",
        "A4": "Ventral Attention",
        "STGa": "Ventral Attention",
        "PSL": "Ventral Attention",
        "STV": "Ventral Attention",
        # Limbic Network
        "25": "Limbic",
        "23c": "Limbic",
        "23d": "Limbic",
        "31pv": "Limbic",
        "31a": "Limbic",
        "31pd": "Limbic",
        "33pr": "Limbic",
        "a24pr": "Limbic",
        "p24pr": "Limbic",
        "a24": "Limbic",
        "p24": "Limbic",
        "d32": "Limbic",
        "p32": "Limbic",
        "p32pr": "Limbic",
        "a32pr": "Limbic",
        "8BM": "Limbic",
        "10r": "Limbic",
        "EC": "Limbic",
        "PreS": "Limbic",
        "H": "Limbic",
        "ProS": "Limbic",
        "PeEc": "Limbic",
        "Pir": "Limbic",
        "TE1m": "Limbic",
        # Frontoparietal / Executive Control Network
        "8Av": "Frontoparietal",
        "8Ad": "Frontoparietal",
        "8BL": "Frontoparietal",
        "8C": "Frontoparietal",
        "9m": "Frontoparietal",
        "9p": "Frontoparietal",
        "9a": "Frontoparietal",
        "10d": "Frontoparietal",
        "10v": "Frontoparietal",
        "10pp": "Frontoparietal",
        "a10p": "Frontoparietal",
        "p10p": "Frontoparietal",
        "46": "Frontoparietal",
        "44": "Frontoparietal",
        "45": "Frontoparietal",
        "47l": "Frontoparietal",
        "47m": "Frontoparietal",
        "47s": "Frontoparietal",
        "a47r": "Frontoparietal",
        "p47r": "Frontoparietal",
        "IFJa": "Frontoparietal",
        "IFJp": "Frontoparietal",
        "IFSp": "Frontoparietal",
        "IFSa": "Frontoparietal",
        "p9-46v": "Frontoparietal",
        "a9-46v": "Frontoparietal",
        "9-46d": "Frontoparietal",
        "11l": "Frontoparietal",
        "13l": "Frontoparietal",
        "OFC": "Frontoparietal",
        "pOFC": "Frontoparietal",
        # Default Mode Network
        "RSC": "Default",
        "POS1": "Default",
        "POS2": "Default",
        "7m": "Default",
        "7AL": "Default",
        "7PL": "Default",
        "7PC": "Default",
        "7Am": "Default",
        "7Pm": "Default",
        "d23ab": "Default",
        "v23ab": "Default",
        "PCV": "Default",
        "TGd": "Default",
        "TGv": "Default",
        "TE1a": "Default",
        "TE1p": "Default",
        "TE2a": "Default",
        "TE2p": "Default",
        "PHT": "Default",
        "TF": "Default",
        "PGi": "Default",
        "PGs": "Default",
        # Subcortical
        "Thalamus": "Subcortical",
        "Putamen": "Subcortical",
        "Stem": "Subcortical",
        "Amy": "Subcortical",
        "MidB": "Subcortical",
        "HC": "Subcortical",
        "NAcc": "Subcortical",
        "Caudate": "Subcortical",
        "Pallidum": "Subcortical",
        "Hippocampus": "Subcortical",
        "Amygdala": "Subcortical",
    }

    # Glasser Network Colors (Visual1 = Cyan, Visual2 = Darker Teal)
    network_colors = {
        "Visual1": "#00BFC4",
        "Visual2": "#0097A7",
        "Somatomotor": "#F8766D",
        "Dorsal Attention": "#7CAE00",
        "Ventral Attention": "#C77CFF",
        "Limbic": "#FF61C3",
        "Frontoparietal": "#00BAFF",
        "Default": "#FFC300",
        "Subcortical": "#969696",
    }

    # Apply mapping
    node_colors = []
    network_assignments = []

    for region_name in region_names:
        # Remove hemisphere prefix (L. or R. or B.)
        if "." in region_name:
            parts = region_name.split(".")
            if len(parts) > 1:
                region_key = parts[1]
            else:
                region_key = region_name
        else:
            region_key = region_name

        # Try exact match
        if region_key in glasser_to_network:
            network = glasser_to_network[region_key]
        else:
            # Try partial match
            network = "Subcortical"  # Default fallback
            for key, net in glasser_to_network.items():
                if key in region_key or region_key in key:
                    network = net
                    break

        network_assignments.append(network)
        node_colors.append(network_colors[network])

    print(f"✓ Mapped {len(node_colors)} regions to Glasser networks")
    print(f"\nNetwork distribution:")
    net_counts = Counter(network_assignments)
    for net, count in sorted(net_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {net}: {count} nodes")

    # ==========================================
    # 5. SUMMARY
    # ==========================================
    print("\n" + "=" * 50)
    print("SUCCESS!")
    print(f"  - node_coords: shape {node_coords.shape}")
    print(f"  - node_colors: {len(node_colors)} colors applied")
    print(f"  - region_names: {len(region_names)} names")
    print(f"  - networks: {len(set(network_assignments))} unique networks")
    print("=" * 50)
    return network_assignments, node_colors, node_coords


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And define the plotting function
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # TODO: Double check this code! why is there an unknown region
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that all the code of this cell has been written by Qwen3.7-Plus and needs to be double-checked
    """)
    return


@app.function(hide_code=True)
def plot_pcs_publication_ready(
    srbp_results: dict,
    fc_matrices_df: pl.DataFrame,
    node_coords: np.ndarray,
    pcs_to_plot: list,
    plot_dir: str,
    metadata_dir: str,
    node_colors=None,
    show_legend=True,
    network_assignments=None,
) -> dict:
    """
    Generates a complete Marimo UI for a list of Principal Components.
    Returns a dictionary with 'results' (keyed by PC index) and 'ui'.
    Saves a JSON metadata file for each PC with sorted network statistics.
    """
    res = {"results": {}}
    ui_elements = []

    # Ensure the plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # 1. Extract Data (Done once for all PCs)
    target_col = fc_matrices_df["diag"]
    mask = (
        (target_col != -1000)
        & (target_col.is_not_null())
        & ((target_col == 0) | (target_col == 2))
    )
    filtered_df = fc_matrices_df.filter(mask)
    group0_fc = filtered_df.filter(pl.col("diag") == 0)[
        "harmonized_fc_matrix"
    ].to_list()
    group2_fc = filtered_df.filter(pl.col("diag") == 2)[
        "harmonized_fc_matrix"
    ].to_list()
    mean_diff = np.mean(np.array(group2_fc), axis=0) - np.mean(
        np.array(group0_fc), axis=0
    )
    n_nodes = int((1 + np.sqrt(1 + 8 * len(mean_diff))) / 2)

    # Pad coordinates and colors once
    if n_nodes > len(node_coords):
        padding = np.zeros((n_nodes - len(node_coords), 3))
        node_coords = np.vstack([node_coords, padding])
        if network_assignments is not None:
            network_assignments = network_assignments + ["Unknown"] * (
                n_nodes - len(network_assignments)
            )

    if node_colors is None or (
        isinstance(node_colors, str) and node_colors == "royalblue"
    ):
        node_colors = [
            "#a6cee3" if coord[0] < 0 else "#fb9a99" for coord in node_coords
        ]
    elif isinstance(node_colors, list) and len(node_colors) < n_nodes:
        missing = n_nodes - len(node_colors)
        try:
            colors_arr = np.array(node_colors)
            if colors_arr.ndim == 2 and colors_arr.shape[1] in [3, 4]:
                gray_val = (
                    [0.8, 0.8, 0.8, 1.0]
                    if colors_arr.shape[1] == 4
                    else [0.8, 0.8, 0.8]
                )
                padding_colors = np.tile(gray_val, (missing, 1))
                node_colors = np.vstack([colors_arr, padding_colors])
            else:
                node_colors = node_colors + ["#000000"] * missing
        except Exception:
            node_colors = node_colors + ["#000000"] * missing

    # 2. Loop through each requested PC
    for pc_idx in pcs_to_plot:
        # Extract edges for this specific PC
        diag_data = srbp_results["diag"]
        mask_pc = np.isclose(diag_data["cons_pc"], pc_idx)
        edges_to_plot = list(diag_data["cons"][mask_pc])

        if not edges_to_plot:
            continue  # Skip if no edges found for this PC

        # Create Adjacency Matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        i_idx, j_idx = np.triu_indices(n_nodes, k=1)
        for edge_idx in edges_to_plot:
            i, j = i_idx[edge_idx], j_idx[edge_idx]
            adj_matrix[i, j] = mean_diff[edge_idx]
            adj_matrix[j, i] = mean_diff[edge_idx]

        # Initialize stats dictionaries to prevent NameError if network_assignments is None
        network_edge_counts = {}
        intra_network_edges = {}
        inter_network_edges = {}

        # Network Stats Markdown
        network_stats_md = ""
        if network_assignments is not None:
            for edge_idx in edges_to_plot:
                i, j = i_idx[edge_idx], j_idx[edge_idx]
                net_i = (
                    network_assignments[i]
                    if i < len(network_assignments)
                    else "Unknown"
                )
                net_j = (
                    network_assignments[j]
                    if j < len(network_assignments)
                    else "Unknown"
                )
                network_edge_counts[net_i] = (
                    network_edge_counts.get(net_i, 0) + 1
                )
                network_edge_counts[net_j] = (
                    network_edge_counts.get(net_j, 0) + 1
                )
                if net_i == net_j:
                    intra_network_edges[net_i] = (
                        intra_network_edges.get(net_i, 0) + 1
                    )
                else:
                    pair_name = (
                        f"{net_i} ↔ {net_j}"
                        if net_i < net_j
                        else f"{net_j} ↔ {net_i}"
                    )
                    inter_network_edges[pair_name] = (
                        inter_network_edges.get(pair_name, 0) + 1
                    )

            network_stats_md = "### FC Edges per Network\n\n"
            network_stats_md += "| Network | Total Edges | Intra-Network | % of Total |\n| :--- | :---: | :---: | :---: |\n"
            sorted_networks = sorted(
                network_edge_counts.items(), key=lambda x: x[1], reverse=True
            )
            for net_name, total_count in sorted_networks:
                intra_count = intra_network_edges.get(net_name, 0)
                percentage = (total_count / (2 * len(edges_to_plot))) * 100
                network_stats_md += f"| {net_name} | {total_count} | {intra_count} | {percentage:.1f}% |\n"

            network_stats_md += "\n### Top Inter-Network Connections\n\n| Connection Pair | Count |\n| :--- | :---: |\n"
            sorted_inter = sorted(
                inter_network_edges.items(), key=lambda x: x[1], reverse=True
            )
            for pair_name, count in sorted_inter[:10]:
                network_stats_md += f"| {pair_name} | {count} |\n"
            network_stats_md += "\n---\n\n"

        # Color Scale
        abs_vals = np.abs(adj_matrix[adj_matrix != 0])
        vmax = np.percentile(abs_vals, 95) if len(abs_vals) > 0 else 0.1
        vmin = -vmax

        # Plotting
        if show_legend:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(
                2, 3, width_ratios=[1, 1, 0.5], wspace=0.3, hspace=0.3
            )
            ax1, ax2, ax3, ax4 = (
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1]),
            )
            ax_legend = fig.add_subplot(gs[:, 2])
            axes = [ax1, ax2, ax3, ax4]
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()

        views = [
            ("l", "Left Lateral", False),
            ("z", "Axial", True),
            ("r", "Right Lateral", False),
            ("z", "Axial (different slice)", False),
        ]
        for ax, (mode, title_text, show_cbar) in zip(axes, views):
            plotting.plot_connectome(
                adj_matrix,
                node_coords,
                edge_cmap="bwr",
                edge_vmin=vmin,
                edge_vmax=vmax,
                node_size=20,
                node_color=node_colors,
                display_mode=mode,
                axes=ax,
                title=title_text,
                colorbar=show_cbar,
                alpha=0.8,
                edge_threshold=0.01,
            )

        # Legend
        if show_legend:
            glasser_network_colors = {
                "Visual1": "#00BFC4",
                "Visual2": "#0097A7",
                "Somatomotor": "#F8766D",
                "Dorsal Attention": "#7CAE00",
                "Ventral Attention": "#C77CFF",
                "Limbic": "#FF61C3",
                "Frontoparietal": "#00BAFF",
                "Default": "#FFC300",
                "Subcortical": "#969696",
                "Cerebellum": "#A0A0A0",
                "Unknown": "#000000",
            }
            legend_elements = []
            unique_nets_in_data = (
                list(dict.fromkeys(network_assignments))
                if network_assignments
                else []
            )
            for net_name in unique_nets_in_data:
                color = glasser_network_colors.get(net_name, "#000000")
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=12,
                        label=net_name,
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                    )
                )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="red",
                    linewidth=2,
                    label="Over-connectivity (MDD > HC)",
                )
            )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="blue",
                    linewidth=2,
                    label="Under-connectivity (MDD < HC)",
                )
            )
            ax_legend.axis("off")
            ax_legend.legend(
                handles=legend_elements,
                loc="center",
                fontsize=10,
                frameon=True,
                fancybox=True,
                shadow=True,
                title="Glasser Networks",
                title_fontsize=11,
            )

        plt.suptitle(
            f"PC {pc_idx + 1} (Index {pc_idx}) - {len(edges_to_plot)} edges",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        fig.patch.set_facecolor("white")

        # ==========================================
        # SAVE IMAGE & JSON METADATA
        # ==========================================

        # 1. Save the figure to disk
        image_path = os.path.join(plot_dir, f"brain_pc_{pc_idx + 1}fc.png")
        fig.savefig(
            image_path, dpi=150, bbox_inches="tight", facecolor="white"
        )

        # 2. Save the metadata to JSON
        # We cast numpy types to native Python types (int/float) to prevent JSON serialization errors
        # We also sort the dictionaries by value (highest to lowest)
        metadata = {
            "pc_index_0_based": int(pc_idx),
            "pc_number_1_based": int(pc_idx + 1),
            "num_edges": int(len(edges_to_plot)),
            "color_range": {"vmin": float(vmin), "vmax": float(vmax)},
            "edge_indices": [int(e) for e in edges_to_plot],
            "network_statistics": {
                "per_network": {
                    k: int(v)
                    for k, v in sorted(
                        network_edge_counts.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
                "intra_network": {
                    k: int(v)
                    for k, v in sorted(
                        intra_network_edges.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
                "inter_network": {
                    k: int(v)
                    for k, v in sorted(
                        inter_network_edges.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
            },
        }

        json_path = os.path.join(
            metadata_dir, f"brain_pc_{pc_idx + 1}_metadata.json"
        )
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # 3. Close the figure to free memory (CRITICAL in loops)
        plt.close(fig)

        # 4. Load the image into Marimo
        pc_image = mo.image(src=image_path)

        # 5. Create description markdown
        description_md = f"""
        ### Plot Description for PC {pc_idx + 1}
        - **Total unique edges displayed:** {len(edges_to_plot)}
        - **Range:** `[{vmin:.3f}, {vmax:.3f}]`
        - **RED:** Over-connectivity in MDD (MDD > HC)
        - **BLUE:** Under-connectivity in MDD (MDD < HC)
        """

        # 6. Assemble the UI section for this PC
        pc_ui_elements = [mo.md(description_md)]
        if network_stats_md:
            pc_ui_elements.append(mo.md(network_stats_md))

        pc_ui_elements.append(pc_image)

        pc_ui = mo.vstack(
            [
                mo.vstack(pc_ui_elements, gap=2),
                mo.md("---"),
            ],
            gap=2,
        )

        ui_elements.append(pc_ui)

        # 7. Store results keyed by PC index
        res["results"][pc_idx] = {
            "adj_matrix": adj_matrix,
            "edges": edges_to_plot,
            "num_edges": len(edges_to_plot),
        }

    # 3. Combine all PCs into one complete UI
    res["ui"] = mo.vstack(ui_elements, gap=3)
    return res


@app.cell
def _(
    harmonized_srbp_fc_matrices_hc_mdd_df,
    network_assignments,
    node_colors,
    node_coords,
    srbp_metadata_dir,
    srbp_plot_dir,
    srbp_results,
):
    srbp_target_pcs_to_plot = [1.0, 69.0]

    srbp_pc_plots_results = plot_pcs_publication_ready(
        srbp_results=srbp_results,
        fc_matrices_df=harmonized_srbp_fc_matrices_hc_mdd_df,
        node_coords=node_coords,
        pcs_to_plot=srbp_target_pcs_to_plot,
        plot_dir=srbp_plot_dir,
        metadata_dir=srbp_metadata_dir,
        node_colors=node_colors,
        show_legend=True,
        network_assignments=network_assignments,
    )
    return (srbp_pc_plots_results,)


@app.cell
def _(srbp_pc_plots_results):
    srbp_pc_plots_results["ui"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### First draft of a conclusion
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We see that if we compare the results obtained in the paper, which were obtained by splitting the srbp dataset in 2, the different PCA plots have the same first PC for almost every metric (except for the sex and FD mean metrics).

    However, when looking at which PC were significant for the MDD diagnosis, we don't obtain the same results.
    In the paper, the PC 2 was only selected for the diag metric. In our case, it was selected for the diag, BDI, age, site and mean FD metrics. The only PC which was caracterized as significant for the MDD diagnosis is PC 70.

    It should also be noted that when plotting the FC of the PC 2, we have more connections that what is said in the paper (117 vs 58 for discovery and 45 for validation) and that they are more diverse across brain regions.

    However, we can see that for the PCs 2 and 70, the MDD subjects have an under connevtivity for most of the PC's FCs
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reproduction of the paper results using the `BMB` dataset
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
    harmonized_bmb_fc_matrices_metadata_path = Path(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_bmb_ts_harmonized/all_data_sub_Glasser_GSR1.csv"
    )
    return (harmonized_bmb_fc_matrices_metadata_path,)


@app.cell
def _(harmonized_bmb_fc_matrices_metadata_path):
    harmonized_bmb_fc_matrices_metadata_df = pl.read_csv(
        harmonized_bmb_fc_matrices_metadata_path
    )
    harmonized_bmb_fc_matrices_metadata_df.head()
    return (harmonized_bmb_fc_matrices_metadata_df,)


@app.cell
def _(harmonized_bmb_fc_matrices_metadata_df):
    print(harmonized_bmb_fc_matrices_metadata_df.columns)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And load the harmonized FC matrices themselves.
    """)
    return


@app.cell
def _():
    harmonized_bmb_fc_matrices = []


    with h5py.File(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_bmb_ts_harmonized/all_data_con_Glasser_GSR1_ortho1_harmonized.mat",
        "r",
    ) as _f:
        harmonized_bmb_fc_matrices = _f["X"][:]

    harmonized_bmb_fc_matrices
    return (harmonized_bmb_fc_matrices,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We now want to merge the metadata DF and the FC_matrices
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices, harmonized_bmb_fc_matrices_metadata_df):
    harmonized_bmb_fc_matrices_df = (
        harmonized_bmb_fc_matrices_metadata_df.with_columns(
            harmonized_fc_matrix=harmonized_bmb_fc_matrices
        )
    )
    return (harmonized_bmb_fc_matrices_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And only keep the HC and MDD subjects
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_df):
    harmonized_bmb_fc_matrices_hc_mdd_df = harmonized_bmb_fc_matrices_df.filter(
        pl.col("diag").is_in([0, 2])
    )
    return (harmonized_bmb_fc_matrices_hc_mdd_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analyze the dataframe
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per disease
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_hc_mdd_df):
    harmonized_bmb_fc_matrices_hc_mdd_df.group_by("diag").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per sex
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_df):
    harmonized_bmb_fc_matrices_df.group_by("sex").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per age
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Why is the age a floating point here ?
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_hc_mdd_df):
    harmonized_bmb_fc_matrices_hc_mdd_df.group_by("age").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per BDI score
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_hc_mdd_df):
    harmonized_bmb_fc_matrices_hc_mdd_df.group_by("bdi").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per site
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_hc_mdd_df):
    harmonized_bmb_fc_matrices_hc_mdd_df.group_by("site").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per mean FD
    """)
    return


@app.cell
def _(harmonized_bmb_fc_matrices_hc_mdd_df):
    harmonized_bmb_fc_matrices_hc_mdd_df.group_by("meanFD").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compute and plot the PCA feature selection for the whole dataset
    """)
    return


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


@app.cell
def _():
    bmb_plot_dir = "./res/pca-dim-reduction/bmb/plots/"
    bmb_ttest_dir = "./res/pca-dim-reduction/bmb/t-tests/"
    bmb_metadata_dir = "./res/pca-dim-reduction/bmb/metadatas"
    return bmb_metadata_dir, bmb_plot_dir, bmb_ttest_dir


@app.cell
def _(
    bmb_plot_dir,
    bmb_ttest_dir,
    harmonized_bmb_fc_matrices_hc_mdd_df,
    metric_dict,
):
    bmb_metrics_dict = calculate_metrics(
        df=harmonized_bmb_fc_matrices_hc_mdd_df,
        metric_dict=metric_dict,
        alpha_threshold=0.05,
        plot_dir=bmb_plot_dir,
        ttest_dir=bmb_ttest_dir,
    )

    bmb_results = bmb_metrics_dict["results"]
    bmb_ui = bmb_metrics_dict["ui"]
    return bmb_results, bmb_ui


@app.cell
def _(bmb_ui):
    bmb_ui
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analyze the results of the PCA feature selection
    """)
    return


@app.cell
def _(bmb_results):
    bmb_selected_mdd_pcs = select_mdd_pc(bmb_results, ["bdi"])
    bmb_selected_mdd_pcs
    return


@app.cell
def _(
    bmb_metadata_dir,
    bmb_plot_dir,
    bmb_results,
    harmonized_bmb_fc_matrices_hc_mdd_df,
    network_assignments,
    node_colors,
    node_coords,
):
    bmb_target_pcs_to_plot = [1.0, 60.0]

    bmb_pc_plots_results = plot_pcs_publication_ready(
        srbp_results=bmb_results,
        fc_matrices_df=harmonized_bmb_fc_matrices_hc_mdd_df,
        node_coords=node_coords,
        pcs_to_plot=bmb_target_pcs_to_plot,
        plot_dir=bmb_plot_dir,
        metadata_dir=bmb_metadata_dir,
        node_colors=node_colors,
        show_legend=True,
        network_assignments=network_assignments,
    )
    return (bmb_pc_plots_results,)


@app.cell
def _(bmb_pc_plots_results):
    bmb_pc_plots_results["ui"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### First draft of a conclusion
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can see that for the BMB dataset, the different PC extracted, are quite different than the one in the SRBP dataset.

    However, we can see that for the PCs 2 and 60, the MDD subjects have an under connevtivity for most of the PC's FCs, just like in the SRBP dataset.
    """)
    return


if __name__ == "__main__":
    app.run()
