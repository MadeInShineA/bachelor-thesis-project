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
    from joblib import Parallel, delayed


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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Define a function to calculate the PCA and Ttest features
    """)
    return


@app.function
def calculate_features(
    target_str: str,
    target_filters: Callable,
    harmonized_fc_matrices_df: pl.DataFrame,
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


app._unparsable_cell(
    r"""
    def calculate_pc_scores_ttest(
        pc_scores: pl.dataframe
        ttest_dir,
        ttest_output_prefix: str,
    ):

        df_X_train = pd.DataFrame(
            pc_scores[col_name].tolist(),
            index=pc_scores.index,
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
    """,
    name="_"
)


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


@app.cell
def _(cons, cons_pc, target):
    def performe_pca_scores_ttest(pca_scores, ttest_dir, ttest_output_prefix):
        (
            selected_indices,
            t_statistics,
            p_values,
        ) = select_ttest_features(
            df_X_train=pca_scores,
            target=target,
            output_dir=ttest_dir,
            output_prefix=ttest_output_prefix,
            save_results=True,
        )

        return cons, cons_pc, selected_indices, t_statistics, p_values

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


@app.function
def calculate_metrics(
    df: pd.DataFrame,
    metric_dict: dict[str, dict],
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
    return srbp_plot_dir, srbp_ttest_dir


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
    srbp_results = calculate_metrics(
        df=harmonized_srbp_fc_matrices_hc_mdd_df,
        metric_dict=metric_dict,
        plot_dir=srbp_plot_dir,
        ttest_dir=srbp_ttest_dir,
    )
    return (srbp_results,)


@app.cell
def _(srbp_results):
    srbp_results["ui"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analyze the results of the PCA feature selection
    """)
    return


@app.cell
def _(srbp_results):
    def calculate_fc_edges_counts(metrics_dict: dict) -> None:
        all_metric_edges = {}

        for metric, data in srbp_results["results"].items():
            all_metric_edges[metric] = set(data["cons"])

        all_edges_flat = []
        for edges in all_metric_edges.values():
            all_edges_flat.extend(edges)

        edge_counts = Counter(all_edges_flat)

        unique_edges = {
            edge: count for edge, count in edge_counts.items() if count == 1
        }

        unique_edges_per_metric = {
            metric: [] for metric in all_metric_edges.keys()
        }
        for edge in unique_edges.keys():
            for metric, edges_set in all_metric_edges.items():
                if edge in edges_set:
                    unique_edges_per_metric[metric].append(edge)
                    break

        for metric, edges in unique_edges_per_metric.items():
            total_edges = len(all_metric_edges[metric])
            unique_count = len(edges)
            shared_count = total_edges - unique_count

            print(f"Metric: '{metric}'")
            print(
                f"\tTotal unique FC edges (not in any other metric): {unique_count}"
            )
            print(
                f"\tShared FC edges (found in at least one other metric): {shared_count}"
            )
            print(f"\tTotal edges for this metric: {total_edges}")

    return (calculate_fc_edges_counts,)


@app.cell
def _(calculate_fc_edges_counts, srbp_results):
    calculate_fc_edges_counts(metrics_dict=srbp_results)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Please note that all the code of this cell has been written by Qwen3.7-Plus and needs to be double-checked
    """)
    return


@app.cell
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
    Please note that all the code of this cell has been written by Qwen3.7-Plus and needs to be double-checked
    """)
    return


@app.cell
def _(
    harmonized_srbp_fc_matrices_hc_mdd_df,
    network_assignments,
    node_colors,
    node_coords,
    srbp_results,
):
    def get_unique_edges_for_metric(srbp_results, target_metric):
        all_metric_edges = {}
        for metric, data in srbp_results["results"].items():
            all_metric_edges[metric] = set(data["cons"])

        all_edges_flat = []
        for edges in all_metric_edges.values():
            all_edges_flat.extend(edges)

        edge_counts = Counter(all_edges_flat)

        # Find edges that appear exactly once across ALL metrics
        unique_edges = {
            edge: count for edge, count in edge_counts.items() if count == 1
        }

        # Map them back to the specific metric
        unique_edges_per_metric = {
            metric: [] for metric in all_metric_edges.keys()
        }
        for edge in unique_edges.keys():
            for metric, edges_set in all_metric_edges.items():
                if edge in edges_set:
                    unique_edges_per_metric[metric].append(edge)
                    break

        return unique_edges_per_metric[target_metric]


    # Get the unique edges for 'diag'
    unique_diag_edges = get_unique_edges_for_metric(srbp_results, "diag")
    print(f"Found {len(unique_diag_edges)} edges unique to 'diag'.")


    def plot_publication_ready(
        srbp_results: dict,
        fc_matrices_df: pl.DataFrame,
        node_coords: np.ndarray,
        edges_to_plot: list,
        node_colors=None,
        title: str = "Unique FCs for Diagnosis",
        node_size=20,
        edge_alpha=0.8,
        show_legend=True,
        network_assignments=None,  # Pass network assignments
    ):
        # 1. Extract Data
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

        # Pad coordinates
        if n_nodes > len(node_coords):
            padding = np.zeros((n_nodes - len(node_coords), 3))
            node_coords = np.vstack([node_coords, padding])
            # Also pad network assignments
            if network_assignments is not None:
                network_assignments = network_assignments + ["Unknown"] * (
                    n_nodes - len(network_assignments)
                )

        # 2. Handle Node Colors
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
                    # CHANGED: Set missing nodes to Black (#000000)
                    node_colors = node_colors + ["#000000"] * missing
            except Exception:
                # CHANGED: Set missing nodes to Black (#000000)
                node_colors = node_colors + ["#000000"] * missing

        # 3. Create Adjacency Matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        i_idx, j_idx = np.triu_indices(n_nodes, k=1)

        for edge_idx in edges_to_plot:
            i, j = i_idx[edge_idx], j_idx[edge_idx]
            adj_matrix[i, j] = mean_diff[edge_idx]
            adj_matrix[j, i] = mean_diff[edge_idx]

        # ==========================================
        # COUNT EDGES PER NETWORK
        # ==========================================
        if network_assignments is not None:
            print("\n" + "=" * 70)
            print("FC EDGES PER NETWORK")
            print("=" * 70)

            # Count edges by network
            network_edge_counts = {}
            intra_network_edges = {}
            inter_network_edges = {}

            for edge_idx in edges_to_plot:
                i, j = i_idx[edge_idx], j_idx[edge_idx]

                # Get network for each node
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

                # Count total edges per network
                network_edge_counts[net_i] = network_edge_counts.get(net_i, 0) + 1
                network_edge_counts[net_j] = network_edge_counts.get(net_j, 0) + 1

                # Categorize as intra or inter network
                if net_i == net_j:
                    intra_network_edges[net_i] = (
                        intra_network_edges.get(net_i, 0) + 1
                    )
                else:
                    # Create a sorted pair name for inter-network edges
                    pair_name = (
                        f"{net_i} ↔ {net_j}"
                        if net_i < net_j
                        else f"{net_j} ↔ {net_i}"
                    )
                    inter_network_edges[pair_name] = (
                        inter_network_edges.get(pair_name, 0) + 1
                    )

            # Print summary
            print(f"\nTotal edges plotted: {len(edges_to_plot)}")
            print(
                f"\n{'NETWORK':<20} {'Total Edges':<15} {'Intra-Network':<15} {'% of Total':<10}"
            )
            print("-" * 70)

            # Sort networks by total edges
            sorted_networks = sorted(
                network_edge_counts.items(), key=lambda x: x[1], reverse=True
            )

            for net_name, total_count in sorted_networks:
                intra_count = intra_network_edges.get(net_name, 0)
                percentage = (
                    total_count / (2 * len(edges_to_plot))
                ) * 100  # Each edge counted twice
                print(
                    f"{net_name:<20} {total_count:<15} {intra_count:<15} {percentage:>6.1f}%"
                )

            # Print inter-network connections
            print(f"\n{'INTER-NETWORK CONNECTIONS':<40} {'Count':<10}")
            print("-" * 70)

            sorted_inter = sorted(
                inter_network_edges.items(), key=lambda x: x[1], reverse=True
            )
            for pair_name, count in sorted_inter[:15]:  # Top 15
                print(f"{pair_name:<40} {count:<10}")

            if len(sorted_inter) > 15:
                print(f"... and {len(sorted_inter) - 15} more network pairs")

            print("=" * 70 + "\n")

        # 4. Color Scale
        abs_vals = np.abs(adj_matrix[adj_matrix != 0])
        if len(abs_vals) > 0:
            vmax = np.percentile(abs_vals, 95)
        else:
            vmax = 0.1
        vmin = -vmax

        # 5. Plotting - Modified layout to fit legend
        if show_legend:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(
                2, 3, width_ratios=[1, 1, 0.5], wspace=0.3, hspace=0.3
            )

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
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
                node_size=node_size,
                node_color=node_colors,
                display_mode=mode,
                axes=ax,
                title=title_text,
                colorbar=show_cbar,
                alpha=edge_alpha,
                edge_threshold=0.01,
            )

        # 6. ADD LEGEND
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
                "Unknown": "#000000",  # <--- ADDED: Black for Unknown nodes
            }

            legend_elements = []
            # Only show networks that are actually in your data
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
            f"{title}\n({len(edges_to_plot)} unique edges)",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        fig.patch.set_facecolor("white")

        # ==========================================
        # PRINT DESCRIPTION BEFORE SHOWING PLOT
        # ==========================================
        print("\n" + "=" * 70)
        print("PLOT DESCRIPTION")
        print("=" * 70)
        print(f"\nTitle: {title}")
        print(f"Total unique edges displayed: {len(edges_to_plot)}")
        print(f"\nWhat this graph shows:")
        print(
            "  - Brain connectivity differences between MDD patients and Healthy Controls"
        )
        print(
            "  - Only FC edges UNIQUE to the diagnosis metric (not shared with other metrics)"
        )
        print(
            "  - Nodes are colored by their Glasser intrinsic network assignment"
        )
        print(f"\nColor bar interpretation:")
        print(f"  - Range: [{vmin:.3f}, {vmax:.3f}]")
        print(f"  - RED (positive values): Over-connectivity in MDD (MDD > HC)")
        print(
            f"    → Stronger functional connectivity in MDD patients than controls"
        )
        print(f"  - BLUE (negative values): Under-connectivity in MDD (MDD < HC)")
        print(
            f"    → Weaker functional connectivity in MDD patients than controls"
        )
        print(f"  - WHITE/GREY (near zero): Minimal difference between groups")
        print(
            f"\nEdge thickness: Proportional to the absolute difference strength"
        )
        print(f"Edge transparency (alpha): {edge_alpha}")

        print("\n" + "=" * 70)

        plt.show()
        return fig, adj_matrix


    _, _ = plot_publication_ready(
        srbp_results=srbp_results,
        fc_matrices_df=harmonized_srbp_fc_matrices_hc_mdd_df,
        node_coords=node_coords,
        edges_to_plot=unique_diag_edges,
        node_colors=node_colors,
        title="Unique FCs for Diagnosis (diag)",
        show_legend=True,
        network_assignments=network_assignments,
    )
    return plot_publication_ready, unique_diag_edges


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
    return bmb_plot_dir, bmb_ttest_dir


@app.cell
def _(
    bmb_plot_dir,
    bmb_ttest_dir,
    harmonized_bmb_fc_matrices_hc_mdd_df,
    metric_dict,
):
    bmb_results = calculate_metrics(
        df=harmonized_bmb_fc_matrices_hc_mdd_df,
        metric_dict=metric_dict,
        plot_dir=bmb_plot_dir,
        ttest_dir=bmb_ttest_dir,
    )
    return (bmb_results,)


@app.cell
def _(bmb_results):
    bmb_results["ui"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Analyze the results of the PCA feature selection
    """)
    return


@app.cell
def _(bmb_results, calculate_fc_edges_counts):
    calculate_fc_edges_counts(metrics_dict=bmb_results)
    return


@app.cell
def _(
    bmb_results,
    harmonized_bmb_fc_matrices_hc_mdd_df,
    network_assignments,
    node_colors,
    node_coords,
    plot_publication_ready,
    unique_diag_edges,
):
    _, _ = plot_publication_ready(
        srbp_results=bmb_results,
        fc_matrices_df=harmonized_bmb_fc_matrices_hc_mdd_df,
        node_coords=node_coords,
        edges_to_plot=unique_diag_edges,
        node_colors=node_colors,
        title="Unique FCs for Diagnosis (diag)",
        show_legend=True,
        network_assignments=network_assignments,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
