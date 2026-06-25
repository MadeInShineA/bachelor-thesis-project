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
    import nibabel as nib
    from nilearn.image import resample_to_img
    import pickle
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm


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
    harmonized_srpb_fc_matrices_metadata_path = Path(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_srpb_ts_harmonized/all_data_sub_Glasser_GSR1.csv"
    )
    return (harmonized_srpb_fc_matrices_metadata_path,)


@app.cell
def _(harmonized_srpb_fc_matrices_metadata_path):
    srpb_metadata_df = pl.read_csv(harmonized_srpb_fc_matrices_metadata_path)
    srpb_metadata_df.head()
    return (srpb_metadata_df,)


@app.cell
def _(srpb_metadata_df):
    print(srpb_metadata_df.columns)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And load the harmonized FC matrices themselves.
    """)
    return


@app.cell
def _():
    harmonized_srpb_fc_matrices = []


    with h5py.File(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_srpb_ts_harmonized/all_data_con_Glasser_GSR1_ortho1_harmonized.mat",
        "r",
    ) as _f:
        harmonized_srpb_fc_matrices = _f["X"][:]

    harmonized_srpb_fc_matrices
    return (harmonized_srpb_fc_matrices,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We now want to merge the metadata DF and the FC_matrices
    """)
    return


@app.cell
def _(harmonized_srpb_fc_matrices, srpb_metadata_df):
    harmonized_srpb_fc_matrices_df = srpb_metadata_df.with_columns(
        harmonized_fc_matrix=harmonized_srpb_fc_matrices
    )
    return (harmonized_srpb_fc_matrices_df,)


@app.cell
def _(harmonized_srpb_fc_matrices_df):
    harmonized_srpb_fc_matrices_df["harmonized_fc_matrix"].head(1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And only keep the HC and MDD subjects
    """)
    return


@app.cell
def _(harmonized_srpb_fc_matrices_df):
    harmonized_srpb_fc_matrices_hc_mdd_df = harmonized_srpb_fc_matrices_df.filter(
        pl.col("diag").is_in([0, 2])
    )
    return (harmonized_srpb_fc_matrices_hc_mdd_df,)


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
def _(harmonized_srpb_fc_matrices_hc_mdd_df):
    harmonized_srpb_fc_matrices_hc_mdd_df.group_by("diag").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per sex
    """)
    return


@app.cell
def _(harmonized_srpb_fc_matrices_hc_mdd_df):
    harmonized_srpb_fc_matrices_hc_mdd_df.group_by("sex").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per age
    """)
    return


@app.cell
def _(harmonized_srpb_fc_matrices_hc_mdd_df):
    harmonized_srpb_fc_matrices_hc_mdd_df.group_by("age").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per BDI score
    """)
    return


@app.cell
def _(harmonized_srpb_fc_matrices_hc_mdd_df):
    harmonized_srpb_fc_matrices_hc_mdd_df.group_by("bdi").len()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Display the number of subject per site
    """)
    return


@app.cell
def _(harmonized_srpb_fc_matrices_hc_mdd_df):
    harmonized_srpb_fc_matrices_hc_mdd_df.group_by("site").len()
    return


@app.cell
def _(harmonized_srpb_fc_matrices_hc_mdd_df):
    harmonized_srpb_fc_matrices_hc_mdd_df.group_by("meanFD").len()
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
    cache_dir: str,
) -> dict:

    os.makedirs(cache_dir, exist_ok=True)

    res = {"results": {}}
    ui_elements = []

    for metric, config in metric_dict.items():
        cache_path = os.path.join(cache_dir, f"{metric}_results.pkl")

        if os.path.exists(cache_path):
            print(f"Loading cached results for metric: {metric}")
            with open(cache_path, "rb") as f:
                results = pickle.load(f)
        else:
            print(f"Calculating and caching results for metric: {metric}")
            results = calculate_features(
                target_str=metric,
                target_filters=config["filter_function"],
                harmonized_fc_matrices_df=df,
                alpha_threshold=alpha_threshold,
                plot_dir=plot_dir,
                ttest_dir=ttest_dir,
                ttest_output_prefix=config["prefix"],
            )

            with open(cache_path, "wb") as f:
                pickle.dump(results, f)

        cons = results.get("cons")
        cons_pc = results.get("cons_pc")

        is_binary_target = "p_values" in results

        if is_binary_target:
            selected_indices = results["selected_indices"]
            t_statistics = results["t_statistics"]
            p_values = results["p_values"]

        image_path = os.path.join(plot_dir, f"{metric}.png")
        metric_image = mo.image(src=image_path)

        if is_binary_target:
            p_value_plot = plot_p_values(p_values, metric_name=metric)
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

        res["results"][metric] = {
            "cons": cons,
            "cons_pc": cons_pc,
            "is_binary_target": is_binary_target,
        }

        if is_binary_target:
            res["results"][metric].update(
                {
                    "selected_indices": selected_indices,
                    "t_statistics": t_statistics,
                    "p_values": p_values,
                }
            )

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
    srpb_plot_dir = "./res/pca-dim-reduction/srpb/plots/"
    srpb_ttest_dir = "./res/pca-dim-reduction/srpb/t-tests/"
    srpb_cache_dir = "./res/pca-dim-reduction/srpb/cache/"
    srpb_metadata_dir = "./res/pca-dim-reduction/srpb/metadatas/"
    return srpb_cache_dir, srpb_metadata_dir, srpb_plot_dir, srpb_ttest_dir


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
    harmonized_srpb_fc_matrices_hc_mdd_df,
    metric_dict,
    srpb_cache_dir,
    srpb_plot_dir,
    srpb_ttest_dir,
):
    srpb_metrics_dict = calculate_metrics(
        df=harmonized_srpb_fc_matrices_hc_mdd_df,
        metric_dict=metric_dict,
        alpha_threshold=0.05,
        plot_dir=srpb_plot_dir,
        ttest_dir=srpb_ttest_dir,
        cache_dir=srpb_cache_dir,
    )

    srpb_results = srpb_metrics_dict["results"]
    srpb_ui = srpb_metrics_dict["ui"]
    return srpb_results, srpb_ui


@app.cell
def _(srpb_ui):
    srpb_ui
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
def _(srpb_results):
    srpb_selected_mdd_pcs = select_mdd_pc(srpb_results, ["bdi"])
    srpb_selected_mdd_pcs
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


@app.cell
def _():
    parcel_folder = "/home/cbi-data34/ayumu/tools/Parcellations/"
    cortex_file = (
        f"{parcel_folder}/GlasserYeo/HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz"
    )
    subcortex_file = f"{parcel_folder}/Tian-Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz"
    cerebellum_file = f"{parcel_folder}/Nettekoven_2023_cerebellar/atl-NettekovenAsym32_space-MNI152NLin6AsymC_dseg.nii"

    subcortex_img = nib.load(subcortex_file)
    cortex_img = nib.load(cortex_file)
    cerebellum_img = nib.load(cerebellum_file)

    # Resample to subcortex space
    cortex_resampled = resample_to_img(
        cortex_img, subcortex_img, interpolation="nearest"
    )
    cortex_volume = np.round(cortex_resampled.get_fdata()).astype(int)
    cerebellum_resampled = resample_to_img(
        cerebellum_img, subcortex_img, interpolation="nearest"
    )
    cerebellum_volume = np.round(cerebellum_resampled.get_fdata()).astype(int)
    subcortex_volume = subcortex_img.get_fdata().astype(int)

    # Split left/right hemisphere for Cortex (adds 180 to left side)
    affine = cortex_resampled.affine
    shape = cortex_volume.shape
    i_coords = np.arange(shape[0])
    mni_x = affine[0, 0] * i_coords + affine[0, 3]
    left_hemisphere_mask = mni_x < 0
    left_mask_3d = left_hemisphere_mask[:, np.newaxis, np.newaxis]

    cortex_volume_processed = cortex_volume.copy()
    cortex_mask = cortex_volume > 0
    left_cortex_mask = cortex_mask & left_mask_3d
    cortex_volume_processed[left_cortex_mask] = (
        cortex_volume[left_cortex_mask] + 180
    )

    # Combine into single 446-label volume
    ROI = np.zeros(np.shape(subcortex_volume), dtype=np.int32)
    ROI[cortex_mask] = cortex_volume_processed[cortex_mask]
    subcortex_mask = subcortex_volume > 0
    ROI[subcortex_mask] = subcortex_volume[subcortex_mask] + np.max(ROI)
    cerebellum_mask = cerebellum_volume > 0
    ROI[cerebellum_mask] = cerebellum_volume[cerebellum_mask] + np.max(ROI)

    # ==========================================
    # 2. Calculate 3D MNI Coordinates
    # ==========================================
    # We use the affine of the subcortex_img because that's the space we resampled everything to
    target_affine = subcortex_img.affine

    labels = np.arange(1, 447)
    coords_voxel = np.array(center_of_mass(ROI, labels=ROI, index=labels))

    # Convert voxel indices to MNI coordinates (mm)
    coords_hom = np.hstack([coords_voxel, np.ones((446, 1))])
    coords_mni = (target_affine @ coords_hom.T).T[:, :3]

    # ==========================================
    # 3. Get Network Assignments for the Legend
    # ==========================================
    # Load the Glasser to Yeo mapping
    glasser_yeo_file = f"{parcel_folder}/GlasserYeo/Glasser2Yeo.csv"
    glasser_yeo = pd.read_csv(glasser_yeo_file)

    network_col = (
        "Yeo7" if "Yeo7" in glasser_yeo.columns else glasser_yeo.columns[1]
    )

    # ==========================================
    # MAP NUMBERS TO NETWORK NAMES
    # ==========================================
    # Standard Yeo 7-network mapping (adjust if your CSV uses different numbering)
    yeo_number_to_name = {
        0: "Visual",
        1: "SomatoMotor",
        2: "DorsalAttention",
        3: "DefaultMode",
        4: "Limbic",
        5: "Salience",
        6: "PrefrontalControlA",
        7: "PrefrontalControlB",
    }

    # Convert the numbers to names
    cortex_networks_raw = glasser_yeo[network_col].tolist()

    cortex_networks_clean = [
        yeo_number_to_name.get(int(val), "Unknown") for val in cortex_networks_raw
    ]

    # Build the full 446-length list
    network_assignments = []
    network_assignments.extend(
        cortex_networks_clean
    )  # 1-360: Glasser Yeo Networks
    network_assignments.extend(["Subcortical"] * 54)  # 361-414
    network_assignments.extend(["Cerebellum"] * 32)  # 415-446

    print(
        f"Network assignments created. Unique networks: {set(network_assignments)}"
    )

    # ==========================================
    # 4. Save to Disk
    # ==========================================
    out_dir = "./res/pca-dim-reduction/"
    os.makedirs(out_dir, exist_ok=True)  # Ensure the folder exists

    np.save(os.path.join(out_dir, "node_coords_446.npy"), coords_mni)

    # Save network assignments as a simple text file (one per line)
    with open(os.path.join(out_dir, "network_assignments_446.txt"), "w") as _f:
        for net in network_assignments:
            _f.write(f"{net}\n")
    return coords_mni, network_assignments


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And define the plotting function
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
    srpb_results: dict,
    fc_matrices_df,  # pl.DataFrame
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

    # ==========================================
    # DEFINE COLOR MAP ONCE AT THE TOP
    # ==========================================
    glasser_network_colors = {
        "Visual": "#00BFC4",  # Cyan
        "SomatoMotor": "#F8766D",  # Red
        "DorsalAttention": "#7CAE00",  # Green
        "DefaultMode": "#FFC300",  # Yellow/Orange
        "Limbic": "#FF61C3",  # Pink
        "Salience": "#C77CFF",  # Purple
        "PrefrontalControlA": "#00BAFF",  # Light Blue
        "PrefrontalControlB": "#0097A7",  # Darker Cyan/Teal
        "Subcortical": "#333333",  # Dark Gray
        "Cerebellum": "#D2691E",  # Brown/Orange
        "Unknown": "#000000",  # Black
    }

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

    # ==========================================
    # NEW LOGIC: Color nodes by Network
    # ==========================================
    if node_colors is None:
        if network_assignments is not None:
            # Map each node's network name to its hex color!
            node_colors = [
                glasser_network_colors.get(str(net), "#000000")
                for net in network_assignments
            ]
        else:
            # Fallback to hemisphere coloring if no network info is provided
            node_colors = [
                "#a6cee3" if coord[0] < 0 else "#fb9a99"
                for coord in node_coords
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
        diag_data = srpb_results["diag"]
        mask_pc = np.isclose(diag_data["cons_pc"], pc_idx)
        edges_to_plot = list(diag_data["cons"][mask_pc])

        if not edges_to_plot:
            continue  # Skip if no edges found for this PC

        # Create Adjacency Matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        i_idx, j_idx = np.tril_indices(n_nodes, k=1)
        for edge_idx in edges_to_plot:
            i, j = i_idx[edge_idx], j_idx[edge_idx]
            adj_matrix[i, j] = mean_diff[edge_idx]
            adj_matrix[j, i] = mean_diff[edge_idx]

        # ── hide nodes that aren't part of any drawn edge ──
        participating = np.zeros(n_nodes, dtype=bool)
        for edge_idx in edges_to_plot:
            participating[i_idx[edge_idx]] = True
            participating[j_idx[edge_idx]] = True
        node_sizes = np.where(participating, 20, 0)  # 0 = invisible
        # ──────────────────────────────────────────────────────────

        # Initialize stats dictionaries to prevent NameError if network_assignments is None
        network_node_counts = {}
        network_edge_counts = {}
        intra_network_edges = {}
        inter_network_edges = {}

        # Network Stats Markdown
        network_stats_md = ""
        if network_assignments is not None:
            # Count participating nodes per network
            for node_idx in range(n_nodes):
                if participating[node_idx]:
                    net = (
                        network_assignments[node_idx]
                        if node_idx < len(network_assignments)
                        else "Unknown"
                    )
                    network_node_counts[net] = (
                        network_node_counts.get(net, 0) + 1
                    )

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
                    # Force conversion to string to prevent TypeError when comparing int/float to str
                    str_i, str_j = str(net_i), str(net_j)
                    pair_name = (
                        f"{str_i} ↔ {str_j}"
                        if str_i < str_j
                        else f"{str_j} ↔ {str_i}"
                    )
                    inter_network_edges[pair_name] = (
                        inter_network_edges.get(pair_name, 0) + 1
                    )

            # 1. Nodes Table
            network_stats_md = "#### Participating Nodes per Network\n\n"
            network_stats_md += "| Network | Node Count |\n| :--- | :---: |\n"
            sorted_node_nets = sorted(
                network_node_counts.items(), key=lambda x: x[1], reverse=True
            )
            for net_name, node_count in sorted_node_nets:
                network_stats_md += f"| {net_name} | {node_count} |\n"

            # 2. Edges Table
            network_stats_md += "\n#### FC Edges per Network\n\n"
            network_stats_md += "| Network | Total Edges | Intra-Network | % of Total |\n| :--- | :---: | :---: | :---: |\n"
            sorted_networks = sorted(
                network_edge_counts.items(), key=lambda x: x[1], reverse=True
            )
            for net_name, total_count in sorted_networks:
                intra_count = intra_network_edges.get(net_name, 0)
                percentage = (total_count / (2 * len(edges_to_plot))) * 100
                network_stats_md += f"| {net_name} | {total_count} | {intra_count} | {percentage:.1f}% |\n"

            # 3. Inter-Network Table
            network_stats_md += "\n#### Top Inter-Network Connections\n\n| Connection Pair | Count |\n| :--- | :---: |\n"
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
                node_size=node_sizes,
                node_color=node_colors,
                display_mode=mode,
                axes=ax,
                title=title_text,
                colorbar=show_cbar,
                alpha=0.8,
                edge_threshold=0.01,
            )

        # ==========================================
        # LEGEND LOGIC
        # ==========================================
        if show_legend:
            legend_elements = []

            # Only include networks that are actually participating in the current PC's plot
            if network_assignments is not None and network_node_counts:
                # Sort by node count descending for a cleaner, consistent legend
                unique_nets_in_plot = sorted(
                    network_node_counts.keys(),
                    key=lambda x: network_node_counts[x],
                    reverse=True,
                )

                for net_name in unique_nets_in_plot:
                    color = glasser_network_colors.get(
                        str(net_name), "#000000"
                    )
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

            # Always include the connectivity direction legend elements
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
        metadata = {
            "pc_index_0_based": int(pc_idx),
            "pc_number_1_based": int(pc_idx + 1),
            "num_edges": int(len(edges_to_plot)),
            "color_range": {"vmin": float(vmin), "vmax": float(vmax)},
            "edge_indices": [int(e) for e in edges_to_plot],
            "network_statistics": {
                "nodes_per_network": {
                    k: int(v)
                    for k, v in sorted(
                        network_node_counts.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
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
    coords_mni,
    harmonized_srpb_fc_matrices_hc_mdd_df,
    network_assignments,
    srpb_metadata_dir,
    srpb_plot_dir,
    srpb_results,
):
    srpb_target_pcs_to_plot = [1.0, 69.0]

    srpb_pc_plots_results = plot_pcs_publication_ready(
        srpb_results=srpb_results,
        fc_matrices_df=harmonized_srpb_fc_matrices_hc_mdd_df,
        node_coords=coords_mni,
        pcs_to_plot=srpb_target_pcs_to_plot,
        plot_dir=srpb_plot_dir,
        metadata_dir=srpb_metadata_dir,
        show_legend=True,
        network_assignments=network_assignments,
    )
    return (srpb_pc_plots_results,)


@app.cell
def _(srpb_pc_plots_results):
    srpb_pc_plots_results["ui"]
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
    We see that if we compare the results obtained in the paper, which were obtained by splitting the srpb dataset in 2, the different PCA plots have the same first PC for almost every metric (except for the sex and FD mean metrics).

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
    bmb_cache_dir = "./res/pca-dim-reduction/bmb/cache/"
    bmb_metadata_dir = "./res/pca-dim-reduction/bmb/metadatas/"
    return bmb_cache_dir, bmb_metadata_dir, bmb_plot_dir, bmb_ttest_dir


@app.cell
def _(
    bmb_cache_dir,
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
        cache_dir=bmb_cache_dir,
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
    coords_mni,
    harmonized_bmb_fc_matrices_hc_mdd_df,
    network_assignments,
):
    bmb_target_pcs_to_plot = [1.0, 7.0, 60.0]

    bmb_pc_plots_results = plot_pcs_publication_ready(
        srpb_results=bmb_results,
        fc_matrices_df=harmonized_bmb_fc_matrices_hc_mdd_df,
        node_coords=coords_mni,
        pcs_to_plot=bmb_target_pcs_to_plot,
        plot_dir=bmb_plot_dir,
        metadata_dir=bmb_metadata_dir,
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
    We can see that for the BMB dataset, the different PC extracted, are quite different than the one in the SRPB dataset.

    However, we can see that for the PCs 2 and 60, the MDD subjects have an under connevtivity for most of the PC's FCs, just like in the SRPB dataset.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Extraction of the regular the FC matrices of the SRPB dataset
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We will now try to replicate the previous results, but with perturbated FC matrices.

    We will start by regular FC matrices extraction, to see if the extraction method is correct and compare them to the harmonized ones
    """)
    return


@app.cell
def _():
    srpb_fuzzy_fc_matrices_output_path = (
        "/home/cbi-biomark/olivier.amacker/fuzzy-fc-matrices/srpb"
    )
    return (srpb_fuzzy_fc_matrices_output_path,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Assign each time series file to its subject
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start by creating a new Dataframe with the SRPB metadata and each subject time series file path
    """)
    return


@app.cell
def _():
    srpb_time_series_directory_path = Path(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_srpb/time_series/Glasser_filtering1_GSR1_scrubbing1/"
    )
    return (srpb_time_series_directory_path,)


@app.cell
def _(srpb_time_series_directory_path):
    srpb_time_series_file_paths = [
        file
        for file in srpb_time_series_directory_path.iterdir()
        if file.is_file()
    ]
    return (srpb_time_series_file_paths,)


@app.cell
def _(srpb_time_series_file_paths):
    srpb_time_series_file_paths_dict = {
        path.name.removesuffix("_restT1_parcel.csv"): path
        for path in srpb_time_series_file_paths
        if path.name.endswith("_restT1_parcel.csv")
    }
    return (srpb_time_series_file_paths_dict,)


@app.cell
def _(srpb_time_series_file_paths_dict):
    srpb_time_series_file_paths_df = pl.DataFrame(
        {
            "sub_id": list(srpb_time_series_file_paths_dict.keys()),
            "time_series_path": [
                str(p) for p in srpb_time_series_file_paths_dict.values()
            ],
        }
    )
    return (srpb_time_series_file_paths_df,)


@app.cell
def _(srpb_metadata_df, srpb_time_series_file_paths_df):
    srpb_time_series_file_df = srpb_time_series_file_paths_df.join(
        srpb_metadata_df, on="sub_id", how="left"
    )
    return (srpb_time_series_file_df,)


@app.cell
def _(srpb_time_series_file_df):
    srpb_time_series_file_df.head()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Assign each scrub file to its subject
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We also assign each subject its scrub file path
    """)
    return


@app.cell
def _():
    srpb_scrub_directory_path = Path(
        "/home/cbi-biomark03/ayumu/HARP/data/preproc_srpb/scrub/Glasser_filtering1_GSR1_scrubbing1/"
    )
    return (srpb_scrub_directory_path,)


@app.cell
def _(srpb_scrub_directory_path):
    srpb_scrub_file_paths = [
        file for file in srpb_scrub_directory_path.iterdir() if file.is_file()
    ]
    return (srpb_scrub_file_paths,)


@app.cell
def _(srpb_scrub_file_paths):
    srpb_scrub_file_paths_dict = {
        path.name.removesuffix("_restT1_scrub.csv"): path
        for path in srpb_scrub_file_paths
        if path.name.endswith("_restT1_scrub.csv")
    }
    return (srpb_scrub_file_paths_dict,)


@app.cell
def _(srpb_scrub_file_paths_dict):
    srpb_scrub_file_paths_df = pl.DataFrame(
        {
            "sub_id": list(srpb_scrub_file_paths_dict.keys()),
            "scrub_path": [str(p) for p in srpb_scrub_file_paths_dict.values()],
        }
    )
    return (srpb_scrub_file_paths_df,)


@app.cell
def _(srpb_scrub_file_paths_df, srpb_time_series_file_df):
    srpb_time_series_scrub_file_df = srpb_scrub_file_paths_df.join(
        srpb_time_series_file_df, on="sub_id", how="left"
    )
    return (srpb_time_series_scrub_file_df,)


@app.cell
def _(srpb_time_series_scrub_file_df):
    srpb_time_series_scrub_file_df.head()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calculate each subject regular FC matrix
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We will now define the functions used to extract the FC matrices from the subject's time series and scrub paths
    """)
    return


@app.function
def extract_subject_fc_matrix(
    subject_time_series_path: str,
    subject_scrub_path: str,
    apply_scrubbing=True,
    scrubbing_threshold=0.5,
) -> list:

    # Load the time_series
    time_series_data = pl.read_csv(subject_time_series_path)
    time_series_data = time_series_data.drop(time_series_data.columns[0])

    if apply_scrubbing:
        # Load the scrub file
        scrub_data = pl.read_csv(subject_scrub_path)
        scrub_array = scrub_data["0"].to_numpy()

        scrub_first_col = scrub_data.columns[0]
        scrub_data = scrub_data.drop(scrub_first_col)
        scrub_array = scrub_data.to_numpy().flatten()

        # Compute the mask
        mask = np.ones(len(scrub_array))

        for time_i, fd_i in enumerate(scrub_array):
            if fd_i > scrubbing_threshold:
                j1 = max(0, time_i)
                j2 = min(len(scrub_array), time_i + 2)
                mask[j1:j2] = 0

        # Apply the mask using
        mask_series = pl.Series(mask == 1)
        time_series_data_filtered = time_series_data.filter(mask_series)
    else:
        time_series_data_filtered = time_series_data

    # Convert to numpy for correlation computation
    ts_array = time_series_data_filtered.to_numpy()

    # Compute correlation matrix
    corr_matrix = np.corrcoef(ts_array.T)

    # Lower triangular + Fisher Z-transformation
    lower_tri = np.tril(corr_matrix, -1)
    lower_tri_z = np.arctanh(lower_tri)

    # Extract non-zero values
    connectivity = lower_tri_z[lower_tri_z != 0]

    return connectivity.tolist()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And extract each FC matrix in parallel
    """)
    return


@app.cell
def _(srpb_fuzzy_fc_matrices_output_path):
    run_name = "regular-matrices"

    run_fc_matrices_output_dir = f"{srpb_fuzzy_fc_matrices_output_path}/{run_name}/"

    run_fc_matrices_cache_filename = "Glasser_filtering1_GSR1_fuzzy_fc_matrices.pkl"
    run_fc_matrices_cache_path = os.path.join(run_fc_matrices_output_dir, run_fc_matrices_cache_filename)
    return run_fc_matrices_cache_path, run_fc_matrices_output_dir, run_name


@app.cell
def _(
    run_fc_matrices_cache_path,
    run_fc_matrices_output_dir,
    srpb_time_series_scrub_file_df,
):
    if os.path.exists(
        run_fc_matrices_cache_path,
    ):
        print(
            f"Loading cached results for extracted FC matrices from {run_fc_matrices_cache_path}"
        )

        with open(run_fc_matrices_cache_path, "rb") as f:
            results = pickle.load(f)

        # Reconstruct the DataFrame from the cached list
        srpb_extracted_fc_matrices_df = (
            srpb_time_series_scrub_file_df.with_columns(
                pl.Series(
                    name="fc_matrix", values=results, dtype=pl.List(pl.Float64)
                )
            ).select(
                "sub_id",
                "time_series_path",
                "fc_matrix",
                pl.exclude("sub_id", "time_series_path", "fc_matrix"),
            )
        )

    else:
        print(f"Computing and caching extracted FC matrices...")
        ts_paths = srpb_time_series_scrub_file_df["time_series_path"].to_list()
        scrub_paths = srpb_time_series_scrub_file_df["scrub_path"].to_list()

        def process_subject(args):
            ts_path, sc_path = args
            return extract_subject_fc_matrix(
                subject_time_series_path=ts_path,
                subject_scrub_path=sc_path,
            )

        results = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_subject, (ts, sc))
                for ts, sc in zip(ts_paths, scrub_paths)
            ]

            for future in tqdm(futures, desc="Processing subjects"):
                results.append(future.result())

        srpb_extracted_fc_matrices_df = (
            srpb_time_series_scrub_file_df.with_columns(
                pl.Series(
                    name="fc_matrix", values=results, dtype=pl.List(pl.Float64)
                )
            ).select(
                "sub_id",
                "time_series_path",
                "fc_matrix",
                pl.exclude("sub_id", "time_series_path", "fc_matrix"),
            )
        )

        # Ensure the output directory exists before saving
        os.makedirs(run_fc_matrices_output_dir, exist_ok=True)

        # Save to cache with the correct extension
        with open(run_fc_matrices_cache_path, "wb") as _f:
            pickle.dump(results, _f)
    return (srpb_extracted_fc_matrices_df,)


@app.cell
def _(srpb_extracted_fc_matrices_df):
    srpb_extracted_fc_matrices_df.head(1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We want to save the current run FC matrices
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Comparison of the extracted FC matrices and the harmonized ones
    """)
    return


@app.cell
def _(run_name):
    srpb_fc_matrices_correlation_output_dir = f"./res/pca-dim-reduction/srpb/fc_matrices_correlation/{run_name}"
    return (srpb_fc_matrices_correlation_output_dir,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that this plotting code was generated by Qwen3.7-Plus
    """)
    return


@app.cell
def _():
    from scipy import stats

    def compare_fc_matrices(
        unharmonized_series,
        harmonized_series,
        out_dir,
    ) -> dict:
        """
        Compare unharmonized vs harmonized FC matrices.
        Returns histogram of per-subject pairwise correlations.
        Results are cached in JSON format.
        """
        os.makedirs(out_dir, exist_ok=True)
        cache_path = os.path.join(out_dir, "fc_comparison_results.json")
        plot_path = os.path.join(out_dir, "fc_comparison_histogram.png")

        # Check cache
        if os.path.exists(cache_path) and os.path.exists(plot_path):
            with open(cache_path, "r") as f:
                results = json.load(f)
        else:
            # Extract to Python lists
            unharmonized_list = unharmonized_series.to_list()
            harmonized_list = harmonized_series.to_list()

            subject_correlations = []
            subject_maes = []

            for i, (my_vec_flat, ref_vec_flat) in enumerate(
                zip(unharmonized_list, harmonized_list)
            ):
                my_vec = np.array(my_vec_flat, dtype=np.float64)
                ref_vec = np.array(ref_vec_flat, dtype=np.float64)

                # Safety check
                if len(my_vec) != len(ref_vec):
                    continue

                # Calculate metrics
                r, _ = stats.pearsonr(my_vec, ref_vec)
                subject_correlations.append(r)

                mae = np.mean(np.abs(my_vec - ref_vec))
                subject_maes.append(mae)

            results = {
                "subject_correlations": subject_correlations,
                "subject_maes": subject_maes,
                "avg_correlation": float(np.mean(subject_correlations))
                if subject_correlations
                else float("nan"),
                "avg_mae": float(np.mean(subject_maes))
                if subject_maes
                else float("nan"),
                "n_subjects": len(subject_correlations),
            }

            # Create histogram of per-subject correlations
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            if subject_correlations:
                # Create histogram with 30 bins
                n_bins = min(30, len(subject_correlations))
                counts, bins, patches = ax.hist(
                    subject_correlations,
                    bins=n_bins,
                    color="skyblue",
                    edgecolor="black",
                    alpha=0.8,
                )
            
                ax.set_title(
                    "Distribution of Per-Subject Pairwise Correlations\n(Unharmonized vs Harmonized)",
                    fontsize=12,
                )
                ax.set_xlabel("Pearson Correlation (r)", fontsize=10)
                ax.set_ylabel("Number of Subjects", fontsize=10)
            
                # Add vertical lines for mean and median
                mean_corr = np.mean(subject_correlations)
                median_corr = np.median(subject_correlations)
            
                ax.axvline(
                    mean_corr,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_corr:.4f}",
                )
                ax.axvline(
                    median_corr,
                    color="green",
                    linestyle="-.",
                    linewidth=2,
                    label=f"Median: {median_corr:.4f}",
                )
            
                ax.legend(loc="upper left", fontsize=9)
                ax.grid(axis="y", alpha=0.3)

                # Add text annotation with statistics
                stats_text = (
                    f"Mean: {mean_corr:.4f}\n"
                    f"Median: {median_corr:.4f}\n"
                    f"Std: {np.std(subject_correlations):.4f}\n"
                    f"Min: {np.min(subject_correlations):.4f}\n"
                    f"Max: {np.max(subject_correlations):.4f}"
                )
                ax.text(
                    0.95,
                    0.95,
                    stats_text,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No valid data to plot",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            # Cache results as JSON
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)

        # Create UI element
        comparison_plot = mo.image(src=plot_path)

        # Handle NaN values safely
        avg_corr = results["avg_correlation"]
        avg_mae = results["avg_mae"]
    
        avg_corr_str = f"{avg_corr:.4f}" if not np.isnan(avg_corr) else "N/A"
        avg_mae_str = f"{avg_mae:.4f}" if not np.isnan(avg_mae) else "N/A"
        median_corr_str = f"{np.median(results['subject_correlations']):.4f}" if results['subject_correlations'] else "N/A"

        metrics_md = mo.md(f"""
        #### FC Matrix Comparison Metrics
    
        | Metric | Value |
        |--------|-------|
        | **Average Per-Subject Correlation** | {avg_corr_str} |
        | **Median Per-Subject Correlation** | {median_corr_str} |
        | **Average MAE** | {avg_mae_str} |
        | **Number of Subjects** | {results["n_subjects"]} |
        """)

        ui = mo.vstack(
            [
                metrics_md,
                comparison_plot,
            ],
            gap=2,
        )

        return {
            "results": results,
            "ui": ui,
        }

    return (compare_fc_matrices,)


@app.cell
def _(
    compare_fc_matrices,
    harmonized_srpb_fc_matrices_df,
    srpb_extracted_fc_matrices_df,
    srpb_fc_matrices_correlation_output_dir,
):
    fc_matrices_comparison_results = compare_fc_matrices(
        srpb_extracted_fc_matrices_df["fc_matrix"],
        harmonized_srpb_fc_matrices_df["harmonized_fc_matrix"],
        out_dir=srpb_fc_matrices_correlation_output_dir
    )
    return (fc_matrices_comparison_results,)


@app.cell
def _(fc_matrices_comparison_results):
    fc_matrices_comparison_results["ui"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Harmonize the fuzzy FC matrices
    """)
    return


if __name__ == "__main__":
    app.run()
