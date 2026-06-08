import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fuzzy fMRIPrep analysis

    The goal of this notebook is to analyse the different output obtained through the fuzzy-fmriprep preprocessing
    """)
    return


@app.cell
def _():
    import re
    import json
    import warnings
    from collections import defaultdict
    import marimo as mo
    import networkx as nx
    from pathlib import Path
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn import signal
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import seaborn as sns
    from itertools import batched
    from joblib import Parallel, delayed
    import nibabel as nib
    from nilearn import plotting


    import matplotlib.style

    matplotlib.style.use("default")
    return (
        ConnectivityMeasure,
        Line2D,
        NiftiLabelsMasker,
        Parallel,
        Patch,
        Path,
        ScalarFormatter,
        batched,
        defaultdict,
        delayed,
        fetch_atlas_schaefer_2018,
        json,
        mo,
        nib,
        np,
        nx,
        pl,
        plotting,
        plt,
        re,
        signal,
        sns,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generating the different FC matrices
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We start by setting up the correct data path as a `pathlib` `Path` object
    """)
    return


@app.cell
def _(Path):
    data_path = Path("/home/cbi-biomark/olivier.amacker/derivatives/fmriprep")

    data_path
    return (data_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also define the different output paths
    """)
    return


@app.cell
def _(Path):
    output_path = Path("./res/fuzzy-fmriprep-analysis/")

    version = "v2"

    fc_matrices_output_path = output_path / version / "fc-matrices"
    figures_output_path = output_path / version / "figures"
    graph_metrics_output_path = output_path / version / "graph-metrics"

    fc_matrices_output_path.mkdir(parents=True, exist_ok=True)
    figures_output_path.mkdir(parents=True, exist_ok=True)
    graph_metrics_output_path.mkdir(parents=True, exist_ok=True)

    fc_matrices_output_path, figures_output_path
    return (
        fc_matrices_output_path,
        figures_output_path,
        graph_metrics_output_path,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then we want to collect all the different run paths
    """)
    return


@app.cell
def _(data_path):
    run_paths = [run for run in data_path.iterdir() if run.is_dir()]

    run_paths
    return (run_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And also collect the individual subject paths
    """)
    return


@app.cell
def _(run_paths):
    subject_paths = [
        subject_path
        for subject_paths in run_paths
        for subject_path in subject_paths.iterdir()
        if subject_path.is_dir() and "sub" in subject_path.name
    ]

    subject_paths
    return (subject_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook connectivity analysis pipeline will be as follow:
    - Confound regression
      - The FC matrices will be generated once with and once without confound regressions.
      - The confound regression applied will be the six main sets of motion parameters
    - Parcellation atlas
      - The parcellation will be done via theSchaefer 2018 parcellation atlas with 100 cortical regions and 7 functional networks, Spatial smoothing (6 mm FWHM) and temporal standardization were applied during masking.
    - Obtention of the FC matrices
      - The FC matrices will be obtained by calculating the correlation of the different parcellations region time series
    - The rest is currently TBD
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we define functions to process a bold file and save an FC matric one at a time, to be used in parallel
    """)
    return


@app.cell
def _(
    ConnectivityMeasure,
    NiftiLabelsMasker,
    Path,
    json,
    np,
    pl,
    signal,
    warnings,
):
    def get_confounds(
        confound_file_path: Path, confound_columns: list
    ) -> pl.DataFrame:
        confounds_df = pl.read_csv(
            confound_file_path, columns=confound_columns, separator="\t"
        )
        return confounds_df


    def get_masker(repetition_time: float, brain_maps) -> NiftiLabelsMasker:
        masker = NiftiLabelsMasker(
            labels_img=brain_maps,
            standardize="zscore_sample",
            standardize_confounds=True,
            smoothing_fwhm=6,
            detrend=True,
            t_r=repetition_time,
            memory=None,
            memory_level=1,
            verbose=0,
        )

        return masker


    def save_fc_matrix(
        fc_matrix: np.ndarray,
        subject_id: str,
        subject_run: str,
        session_name: str,
        run_name: str,
        version: str,
        output_dir: Path,
    ) -> None:

        output_dir.mkdir(parents=True, exist_ok=True)

        run_dir = output_dir / subject_id / subject_run / session_name
        run_dir.mkdir(parents=True, exist_ok=True)

        out_path = run_dir / f"{run_name}-{version}.npy"
        np.save(out_path, fc_matrix)

        # print(f"Saved {subject_id} {version} to {out_path}")


    def process_single_bold(
        connectivity_measure: ConnectivityMeasure,
        preproc_bold: dict,
        subject_id: str,
        subject_run: str,
        brain_maps,
        confound_columns: list,
        framewise_displacement_threshold: float,
        output_dir: Path,
    ) -> dict:

        # Disable the future release warning
        warnings.filterwarnings(
            "ignore",
            message=".*From release 0.14.0, confounds will be standardized.*",
        )

        session_name = preproc_bold["session_name"]
        run_name = preproc_bold["run_name"]
        bold_nii_gz_path = preproc_bold["bold.nii.gz"]
        bold_json_path = preproc_bold["bold.json"]
        confounds_path = preproc_bold["confounds_timeseries.tsv"]

        confounds = get_confounds(confounds_path, confound_columns).to_pandas()

        repetition_time = None

        with open(bold_json_path, "r") as f:
            repetition_time = json.load(f)["RepetitionTime"]

        # Generate the FC matrices without confound regression

        without_confound_masker = get_masker(repetition_time, brain_maps)
        without_confound_ts = without_confound_masker.fit_transform(
            str(bold_nii_gz_path)
        )

        without_confound_correlation = connectivity_measure.fit_transform(
            [without_confound_ts]
        )[0]

        save_fc_matrix(
            without_confound_correlation,
            subject_id,
            subject_run,
            session_name,
            run_name,
            "without-confound",
            output_dir,
        )

        # Generate the FC matrices with confound regression

        """
        fd_values = np.nan_to_num(
            confounds["framewise_displacement"].values, nan=0.0
        )

        good_volumes_mask = fd_values <= framewise_displacement_threshold

        sample_mask = np.where(good_volumes_mask)[0]
        """

        with_confound_ts = signal.clean(
            without_confound_ts,
            confounds=confounds,
            standardize="zscore_sample",
            standardize_confounds=True,
        )

        with_confound_correlation = connectivity_measure.fit_transform(
            [with_confound_ts]
        )[0]

        save_fc_matrix(
            with_confound_correlation,
            subject_id,
            subject_run,
            session_name,
            run_name,
            "with-confound",
            output_dir,
        )

        return {
            "subject_id": subject_id,
            "subject_run": subject_run,
            "session_name": session_name,
            "run_name": run_name,
            "without-confound": without_confound_correlation,
            "with-confound": with_confound_correlation,
        }

    return (process_single_bold,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we will create 2 classes:

    - The `FuzzyFmriprepSub` class which will represent a fuzzy fMRIPrep subject
    - The `FuzzyFmriprepAnalysis` with the different connectivity analysis functions and settings
    """)
    return


@app.cell
def _(
    ConnectivityMeasure,
    Parallel,
    Path,
    defaultdict,
    delayed,
    fetch_atlas_schaefer_2018,
    json,
    np,
    nx,
    process_single_bold,
    re,
):
    class FuzzyFmriprepSub:
        def __init__(
            self,
            path: Path,
            figures_path: Path,
            anat_paths: list[Path],
            fmap_paths: list[Path],
            func_paths: list[Path],
        ) -> None:
            self.path = path
            self.figures_path = figures_path
            self.anat_paths = anat_paths
            self.fmap_paths = fmap_paths
            self.func_paths = func_paths

            self.set_id()
            self.set_run()
            self.set_preproc_bold_paths()

        def set_id(self):
            self.id = self.path.name

        def set_run(self):
            self.run = self.path.parent.name

        # WARNING: In the output func directory, for each patient there are both AP and PA preproc_bolf files
        #          Therefore I'm not sure which one to use, for now I will only care about the AP
        def set_preproc_bold_paths(self):
            res = []

            for func_path in self.func_paths:
                bolf_gz_paths = []
                bold_json_paths = []
                confounds_paths = []

                for file_path in func_path.iterdir():
                    if (
                        "-AP_" in file_path.name
                        and "space-MNI152NLin2009cAsym" in file_path.name
                        and "desc-preproc_bold.nii.gz" in file_path.name
                    ):
                        bolf_gz_paths.append(file_path)
                    elif (
                        "-AP_" in file_path.name
                        and "space-MNI152NLin2009cAsym" in file_path.name
                        and "desc-preproc_bold.json" in file_path.name
                    ):
                        bold_json_paths.append(file_path)
                    elif "desc-confounds_timeseries.tsv" in file_path.name:
                        confounds_paths.append(file_path)

                for gz_path in bolf_gz_paths:
                    gz_file_name = gz_path.name
                    matching_json_name = gz_file_name.replace("nii.gz", "json")
                    matching_confound_name = gz_file_name.replace(
                        "desc-preproc_bold.nii.gz", "desc-confounds_timeseries.tsv"
                    )

                    # This matching rules was written by Qwen3.6-Plus
                    matching_confound_name = re.sub(
                        r"_space-.*_desc-preproc_bold\.nii\.gz$",
                        "_desc-confounds_timeseries.tsv",
                        gz_file_name,
                    )
                    run_match = re.search(r"_run-(\d+)", gz_file_name)
                    run_name = (
                        f"run-{run_match.group(1)}" if run_match else "run-single"
                    )

                    session_match = re.search(r"_ses-(SFHARP00\d)", gz_file_name)
                    session_name = session_match.group(1)

                    matching_json_path = None
                    matching_confound_path = None

                    for json_path in bold_json_paths:
                        if json_path.name == matching_json_name:
                            matching_json_path = json_path

                    for confound_path in confounds_paths:
                        if confound_path.name == matching_confound_name:
                            matching_confound_path = confound_path

                    if (
                        matching_json_path is not None
                        and matching_confound_path is not None
                    ):
                        res.append(
                            {
                                "session_name": session_name,
                                "run_name": run_name,
                                "bold.nii.gz": gz_path,
                                "bold.json": matching_json_path,
                                "confounds_timeseries.tsv": matching_confound_path,
                            }
                        )

            self.preproc_bold_paths = res


    # What happens in this class is heavily inspired by what Qwen3.6-Plus proposed
    # As I currently don't have any experience genereting FC matrices


    class FuzzyFmriprepAnalysis:
        CONFOUND_COLUMNS = [
            "trans_x",
            "trans_y",
            "trans_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "global_signal",
            "csf",
            "white_matter",
            "a_comp_cor_00",
            "a_comp_cor_01",
            "a_comp_cor_02",
            "a_comp_cor_03",
            "a_comp_cor_04",
            "a_comp_cor_05",
        ]

        FRAMWEWISE_DISPLACEMENT_THRESHOLD = 0.2

        def __init__(self, subjects: list):
            self.subjects = subjects
            self.atlas = fetch_atlas_schaefer_2018(
                n_rois=100, yeo_networks=7, resolution_mm=2
            )
            self.brain_maps = self.atlas["maps"]
            self.roi_labels = self.atlas["labels"]
            self.connectivity_measures = ConnectivityMeasure(
                kind="correlation", standardize="zscore_sample"
            )

        def save_metadata(self, output_dir: Path) -> None:
            output_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                "confound_columns": self.CONFOUND_COLUMNS,
                # "framewise_deplacement_threshold": self.FRAMWEWISE_DISPLACEMENT_THRESHOLD,
            }

            metadata_path = output_dir / "metadata.json"

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

        def generate_fc_matrices(self, output_dir: Path, max_workers=None) -> dict:

            self.save_metadata(output_dir)

            res = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(dict))
                    )
                )
            )

            tasks = []
            for subject in self.subjects:
                for preproc_bold in subject.preproc_bold_paths:
                    tasks.append(
                        (
                            self.connectivity_measures,
                            preproc_bold,
                            subject.id,
                            subject.run,
                            self.brain_maps,
                            self.CONFOUND_COLUMNS,
                            self.FRAMWEWISE_DISPLACEMENT_THRESHOLD,
                            output_dir,
                        )
                    )

            if max_workers is None:
                max_workers = len(tasks)

            print(
                f"Starting parallel processing of {len(tasks)} tasks with {max_workers} workers."
            )

            results = Parallel(n_jobs=max_workers, backend="loky")(
                delayed(process_single_bold)(*task) for task in tasks
            )

            for result in results:
                res[result["subject_id"]][result["subject_run"]][
                    result["session_name"]
                ][result["run_name"]]["without-confound"] = result[
                    "without-confound"
                ]
                res[result["subject_id"]][result["subject_run"]][
                    result["session_name"]
                ][result["run_name"]]["with-confound"] = result["with-confound"]
                print(
                    f"Completed {result['subject_id']} | {result['session_name']} | {result['run_name']}"
                )

            return res

        def load_fc_matrices(self, input_dir: Path) -> dict:

            res = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(dict))
                    )
                )
            )

            for subject_dir in (d for d in input_dir.iterdir() if d.is_dir()):
                for run_dir in (d for d in subject_dir.iterdir() if d.is_dir()):
                    for session_dir in (
                        d for d in run_dir.iterdir() if d.is_dir()
                    ):
                        for fc_matrix_file in (
                            d for d in session_dir.iterdir() if not d.is_dir()
                        ):
                            fc_matrix = np.load(fc_matrix_file)

                            sub_run_match = re.search(
                                r"(run-\d+)-", fc_matrix_file.name
                            )
                            sub_run = sub_run_match.group(1)

                            type_match = re.search(
                                r"(with(out)?-confound)", fc_matrix_file.name
                            )
                            type = type_match.group(1)

                            res[subject_dir.name][run_dir.name][session_dir.name][
                                sub_run
                            ][type] = fc_matrix

            return res

        def compute_graphs_metrics(
            self, thresholded_data, graph_metrics_output_path
        ):
            graph_metrics = {
                threshold: [] for threshold in thresholded_data.keys()
            }

            for _threshold, _entries in thresholded_data.items():
                print(f"\nProcessing threshold: {_threshold}")

                for _entry in _entries:
                    matrix = _entry["matrix"]
                    metadata = _entry["metadata"]

                    graph = nx.from_numpy_array(matrix)

                    # Here we remove the FC matrix diagonal (self loops)
                    graph.remove_edges_from(nx.selfloop_edges(graph))

                    # Local metrics
                    degree_centrality = nx.degree_centrality(graph)
                    clustering_coefficient = nx.clustering(graph)
                    betweenness_centrality = nx.betweenness_centrality(graph)
                    eigenvector_centrality = nx.eigenvector_centrality(
                        graph, max_iter=1000
                    )

                    # Global metrics
                    largest_cc = max(nx.connected_components(graph), key=len)
                    subgraph = graph.subgraph(largest_cc)
                    avg_shortest_path = nx.average_shortest_path_length(subgraph)

                    # small_world = nx.sigma(graph, niter=1, nrand=10)

                    graph_metrics[_threshold].append(
                        {
                            "metadata": metadata,
                            "metrics": {
                                "local_metrics": {
                                    "degree_centrality": degree_centrality,
                                    "clustering_coefficient": clustering_coefficient,
                                    "betweenness_centrality": betweenness_centrality,
                                    "eigenvector_centrality": eigenvector_centrality,
                                },
                                "global_metrics": {
                                    "average_shortest_path_length": avg_shortest_path,
                                    # "small_worldness": small_world,
                                },
                            },
                        }
                    )

                graph_metrics_threshold_path = (
                    graph_metrics_output_path / f"threshold-{_threshold}.json"
                )

                with open(graph_metrics_threshold_path, "w") as f:
                    json.dump(graph_metrics[_threshold], f, indent=4)
            return graph_metrics

        def load_graph_metrics(self, thresholds, graph_metrics_input_path):
            graph_metrics = {threshold: [] for threshold in thresholds}

            for graph_metric_threshold_file in (
                d for d in graph_metrics_input_path.iterdir() if not d.is_dir()
            ):
                file_name = graph_metric_threshold_file.name

                threshold_match = re.search(r"threshold-(.+)\.json", file_name)

                threshold_str = threshold_match.group(1)

                for threshold in thresholds:
                    if threshold_str == str(threshold):
                        data = None

                        with open(graph_metric_threshold_file, "r") as file:
                            data = json.load(file)

                        graph_metrics[threshold] = data

            return graph_metrics


    FuzzyFmriprepSub, FuzzyFmriprepAnalysis
    return FuzzyFmriprepAnalysis, FuzzyFmriprepSub


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We define a utility function that will be used to transform the different subject paths into `FuzzyFmriprepSub`
    """)
    return


@app.cell
def _(Path):
    def get_output_paths_from_parent_path(
        parent_path: Path, output_dir_names: list[str]
    ):
        output_paths = [
            output_path
            for output_path in parent_path.iterdir()
            if output_path.is_dir()
        ]

        res = {output_dir_name: None for output_dir_name in output_dir_names}

        for output_path in output_paths:
            output_path_name = output_path.name
            if output_path_name in output_dir_names:
                res[output_path_name] = output_path

        return res


    get_output_paths_from_parent_path
    return (get_output_paths_from_parent_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally convert the `Path` to `FuzzyFmriprepSub`
    """)
    return


@app.cell
def _(
    FuzzyFmriprepAnalysis,
    FuzzyFmriprepSub,
    get_output_paths_from_parent_path,
    subject_paths,
):
    fuzzy_fmriprep_subjects = []

    for subject_path in subject_paths:
        subject_dir_paths = [
            subject_dir
            for subject_dir in subject_path.iterdir()
            if subject_dir.is_dir()
        ]

        figures_path = None

        anat_paths = []
        func_paths = []
        fmap_paths = []

        for subject_dir_path in subject_dir_paths:
            dir_name = subject_dir_path.name

            if dir_name == "figures":
                figures_path = subject_dir_path

            elif dir_name == "anat":
                anat_paths.append(subject_dir_path)

            elif "ses-" in dir_name and not "ses-multi" in dir_name:
                output_paths = get_output_paths_from_parent_path(
                    subject_dir_path, ["anat", "fmap", "func"]
                )

                missing_directories = []
                for key, value in output_paths.items():
                    if value is None:
                        missing_directories.append(key)

                if len(missing_directories) > 0:
                    print(
                        f"Skipping {subject_dir_path} since the following directories are missing: {missing_directories}"
                    )
                    continue

                anat_path = output_paths["anat"]
                fmap_path = output_paths["fmap"]
                func_path = output_paths["func"]

                anat_paths.append(anat_path)
                fmap_paths.append(fmap_path)
                func_paths.append(func_path)

        if not anat_paths or not func_paths or not fmap_paths:
            continue
        fuzzy_fmriprep_subject = FuzzyFmriprepSub(
            path=subject_path,
            figures_path=figures_path,
            anat_paths=anat_paths,
            fmap_paths=fmap_paths,
            func_paths=func_paths,
        )

        fuzzy_fmriprep_subjects.append(fuzzy_fmriprep_subject)

    print(f"Created {len(fuzzy_fmriprep_subjects)} subjects")
    fuzzy_fmriprep_analysis = FuzzyFmriprepAnalysis(fuzzy_fmriprep_subjects)

    fuzzy_fmriprep_subjects
    return (fuzzy_fmriprep_analysis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get the FC matrices
    """)
    return


@app.cell
def _(fc_matrices_output_path, fuzzy_fmriprep_analysis):
    fc_matrices = fuzzy_fmriprep_analysis.load_fc_matrices(fc_matrices_output_path)

    if not fc_matrices:
        fc_matrices = fuzzy_fmriprep_analysis.generate_fc_matrices(
            fc_matrices_output_path,
        )

    fc_matrices
    return (fc_matrices,)


@app.cell
def _(fc_matrices_output_path, fuzzy_fmriprep_analysis):
    fuzzy_fmriprep_analysis.save_metadata(fc_matrices_output_path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyse the FC matrices
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We start by plotting the different matrices
    """)
    return


@app.cell
def _(fc_matrices, plt, sns):
    total_matrices = 0

    subject_matrices = {
        subject: {"with-confound": [], "without-confound": []}
        for subject in fc_matrices.keys()
    }

    for _subject, _run_values in fc_matrices.items():
        for _run, _session_values in _run_values.items():
            for _session, _sub_run_values in _session_values.items():
                for _sub_run, _type_values in _sub_run_values.items():
                    for _type, _fc_matrix in _type_values.items():
                        subject_matrices[_subject][_type].append(_fc_matrix)
                        total_matrices += 1

                        plt.figure(figsize=(8, 6))

                        # This plot code was generated by Qwen3.6-Plus

                        sns.heatmap(
                            _fc_matrix,
                            cmap="RdBu_r",
                            vmin=-1,
                            vmax=1,
                            square=True,
                            cbar_kws={"shrink": 0.8},
                            xticklabels=False,
                            yticklabels=False,
                        )

                        plt.title(
                            f"Functional Connectivity Matrix of {_subject} {_run} {_session} {_sub_run} ({_type})"
                        )
                        plt.show()
    print(f"Total FC matrices: {total_matrices}")
    return (subject_matrices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And then plot the average of each patient matrices based on it's type (with or without confound)
    """)
    return


@app.cell
def _(fc_matrices, np, plt, sns, subject_matrices):
    for _subject, _type_values in subject_matrices.items():
        for _type, _fc_matrices in _type_values.items():
            if len(fc_matrices) > 0:
                mean_fc_matrix = np.mean(_fc_matrices, axis=0)

                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    mean_fc_matrix,
                    cmap="RdBu_r",
                    vmin=-1,
                    vmax=1,
                    square=True,
                    cbar_kws={"shrink": 0.8},
                    xticklabels=False,
                    yticklabels=False,
                )
                plt.title(
                    f"Mean Functional Connectivity Matrix of {_subject} ({_type})"
                )
                plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will now apply the following thresholds to the FC matrices

    - 0.05
    - 0.1
    - 0.2
    - 0.3
    - 0.4
    - 0.5
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Please note that we apply the filter to the absolute value here!
    """)
    return


@app.cell
def _(fc_matrices):
    thresholded_data = {0.05: [], 0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: []}

    for _subject, _run_values in fc_matrices.items():
        for _run, _session_values in _run_values.items():
            for _session, _sub_run_values in _session_values.items():
                for _sub_run, _type_values in _sub_run_values.items():
                    for _type, _fc_matrix in _type_values.items():
                        for _threshold in thresholded_data.keys():
                            binary_matrix = (abs(_fc_matrix) >= _threshold).astype(
                                int
                            )

                            _entry = {
                                "matrix": binary_matrix,
                                "metadata": {
                                    "subject": _subject,
                                    "run": _run,
                                    "session": _session,
                                    "sub_run": _sub_run,
                                    "type": _type,
                                },
                            }

                            thresholded_data[_threshold].append(_entry)


    thresholded_data
    return (thresholded_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And plot the different threshold with and without confound for 1 subject subrun
    """)
    return


@app.cell
def _(batched, figures_output_path, plt, sns, thresholded_data):
    for _threshold, entries in thresholded_data.items():
        for without_confound, with_confound in batched(entries[:2], n=2):
            # This plot code was generated by Qwen3.6-Plus

            _fig, _axes = plt.subplots(1, 2, figsize=(12, 6))

            sns.heatmap(
                without_confound["matrix"],
                vmin=0,
                vmax=1,
                square=True,
                xticklabels=False,
                yticklabels=False,
                cmap="Greys",
                cbar=False,
                ax=_axes[0],
            )
            _axes[0].set_title(
                f"Type: {without_confound['metadata']['type']}",
                fontsize=12,
                fontweight="bold",
            )

            sns.heatmap(
                with_confound["matrix"],
                vmin=0,
                vmax=1,
                square=True,
                xticklabels=False,
                yticklabels=False,
                cmap="Greys",
                cbar=False,
                ax=_axes[1],
            )
            _axes[1].set_title(
                f"Type: {with_confound['metadata']['type']}",
                fontsize=12,
                fontweight="bold",
            )

            _meta_shared = without_confound["metadata"]

            _fig.suptitle(
                f"Threshold: {_threshold}\n"
                f"Sub: {_meta_shared['subject']} | Run: {_meta_shared['run']} | Session: {_meta_shared['session']} | Subrun: {_meta_shared['sub_run']}\n"
                f"White = 0 | Black = 1",
                fontsize=14,
                y=1.05,
            )

            plt.tight_layout()

            _fig.savefig(
                figures_output_path
                / f"binary_fc_matrices-threshold-{_threshold}.png",
                bbox_inches="tight",
            )

            plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We want to calculate the following metrics for the different thresholds of the FC matrices:
        - local metrics
            - degree centrality
            - clustering coefficient
            - betweenness centrality
            - eigenvector centrality
        - global metrics
            - small-worldness
            - average shortest path length
    """)
    return


@app.cell
def _(fuzzy_fmriprep_analysis, graph_metrics_output_path, thresholded_data):
    graph_metrics = fuzzy_fmriprep_analysis.load_graph_metrics(
        thresholded_data.keys(), graph_metrics_output_path
    )

    if not any(graph_metrics.values()):
        graph_metrics = fuzzy_fmriprep_analysis.compute_graphs_metrics(
            thresholded_data, graph_metrics_output_path
        )

    first_threshold = list(thresholded_data.keys())[0]
    first_metric = graph_metrics[first_threshold][0]

    first_metric
    return (graph_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now want to calculate the NPVR for all of the computed graph metrics as follow:

    $$
      \sigma_{num}^2 = \frac{1}{m} \sum_{j=1}^m \left[ \frac{1}{n-1} \sum_{i=1}^n \left( x_{i,j} - \bar x_{.,j}\right)^2\right]
    $$

    $$
      \sigma_{pop}^2 = \frac{1}{n} \sum_{i=1}^n \left[ \frac{1}{m-1} \sum_{j=1}^m \left( x_{i,j} - \bar x_{i,.}\right)^2\right]
    $$

    $$
      NPVR = \frac{\sigma_{num}}{\sigma_{pop}}
    $$

    It can also be noted that it's possible to approximate the std of Cohen's $d \left( \sigma_d \right)$ as follow

    $$
    \sigma_d = \frac{2}{\sqrt{n}}NPVR
    $$

    Where $\sigma_{num}$ is the numerical variability, $\sigma_{pop}$ is the population variability, $x_{i,j}$ is the measurement for subject $j$ in MCA repetition $i$, $\bar x_{.,j}$ and $\bar x_{i,.}$ are column and row means, $n$ is the total number of MCA repetitions and $m$ is the number of subjects.

    Higher NPVR values indicate regions where computational variability potentially
    compromises the detection of true population differences
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since each subject doesn't have the same number of run, and each run doesn't have the same number of subject, we will instead use the following formula:

    $$
     \sigma_{num\_pooled}^2 = \frac{1}{\sum_{j=1}^m (n_j - 1)} \sum_{j=1}^m (n_j - 1)\left[ \frac{1}{n_j-1} \sum_{i=1}^{n_j} \left( x_{i,j} - \bar x_{.,j}\right)^2\right]
    $$

    $$
     \sigma_{pop\_pooled}^2 = \frac{1}{\sum_{i=1}^n (m_i - 1)} \sum_{i=1}^n (m_i - 1)\left[ \frac{1}{m_i-1} \sum_{j=1}^{m_i} \left( x_{i,j} - \bar x_{.,j}\right)^2\right]
    $$

    Where $n_j$ is the number of MCA repetition per subject $j$ $m_i$ is the number of subjects per MCA repetition $i$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the following code cell, we will use the following matrix to represent the different MCA run metrics
    $$
    \begin{equation}
    X_{metric}^{(r)} = \begin{bmatrix}
    x_{1,1}^{(r)} & x_{1,2}^{(r)} & x_{1,3}^{(r)} & \cdots & x_{1,m}^{(r)} \\
    x_{2,1}^{(r)} & x_{2,2}^{(r)} & \text{NaN} & \cdots & x_{2,m}^{(r)} \\
    x_{3,1}^{(r)} & \text{NaN} & x_{3,3}^{(r)} & \cdots & x_{3,m}^{(r)} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    x_{n,1}^{(r)} & x_{n,2}^{(r)} & x_{n,3}^{(r)} & \cdots & x_{n,m}^{(r)}
    \end{bmatrix}
    \end{equation}
    $$

    $$
    \text{With } metric \in \left[
    \begin{aligned}
    &\text{degree\_centrality,} \\
    &\text{clustering\_coefficient,} \\
    &\text{betweenness\_centrality,}\\
    &\text{eigenvector\_centrality,} \\
    &\text{small\_worldness,}\\
    &\text{average\_shortest\_path\_length}
    \end{aligned}
    \right]
    $$

    Where:

    - $X_{metric}$ is an $n \times m$ matrix  evaluated for a specific brain region $r$ (for global metrics $r$ represents the whole brain).
    - $n$ is the total number of MCA runs
    - $m$ is the total number of subjects.
    - $x_{i,j}^{(r)}$ is The metric value for run $i$ and subject $j$, evaluated for region $r$
    - $\text{NaN}$ represents a missing observation for example (e.g., subject $j$ is missing run $i$)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We then calculate the $\text{NPVR}$ as follows:

    $$
    \begin{equation}
    \sigma_{num\_pooled}^{2(r)} = \frac{1}{\sum_{j=1}^m (n_j - 1)} \sum_{j=1}^m (n_j - 1) \left[ \frac{1}{n_j - 1} \sum_{i \in \Omega_j} \left( x_{i,j}^{(r)} - \bar{x}_{.,j}^{(r)} \right)^2 \right]
    \end{equation}
    $$

    $$
    \begin{equation}
    \sigma_{pop\_pooled}^{2(r)} = \frac{1}{\sum_{i=1}^n (m_i - 1)} \sum_{i=1}^n (m_i - 1) \left[ \frac{1}{m_i - 1} \sum_{j \in \Omega_i} \left( x_{i,j}^{(r)} - \bar{x}_{i,.}^{(r)} \right)^2 \right]
    \end{equation}
    $$

    $$
    \begin{equation}
    \text{NPVR}_{pooled} = \frac{\sqrt{\sigma_{num\_pooled}^{2(r)}}}{\sqrt{\sigma_{pop\_pooled}^{2(r)}}} = \frac{\sigma_{num\_pooled}^{(r)}}{\sigma_{pop\_pooled}^{(r)}}
    \end{equation}
    $$

    Where:
    - $n_j$ is the number of valid MCA runs for subject $j$ (number of non-NaN elements in column $j$ of $X_{metric}^{(r)}$).
    - $m_i$ is the number of valid subjects for MCA run $i$ (number of non-NaN elements in row $i$ of $X_{metric}^{(r)}$).
    - $\Omega_j$ is the set of valid run indices for subject $j$.
    - $\Omega_i$ is the set of valid subject indices for run $i$.
    - $\bar{x}_{.,j}^{(r)}$ is the mean of column $j$ (mean across valid runs for subject $j$).
    - $\bar{x}_{i,.}^{(r)}$ is the mean of row $i$ (mean across valid subjects for run $i$).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Please note that for now, this complete cell has been written by Qwen3.7-Plus and was double checked
    """)
    return


@app.cell
def _(defaultdict, np):
    def calculate_npvr_data(fc_metrics, metric_name, session_filter=None):
        """
        Calculate σ_num, σ_pop, and NPVR for a given metric using pooled standard deviation.
        """
        thresholds = sorted(fc_metrics.keys())
        npvr_values = []
        sigma_num_all_regions = []
        sigma_pop_all_regions = []

        for threshold in thresholds:
            entries = fc_metrics[threshold]

            # Use nested dictionaries to explicitly map metric values to their specific MCA run identifiers
            region_config_subject_data = defaultdict(
                lambda: defaultdict(lambda: defaultdict(dict))
            )

            for entry in entries:
                subject_id = entry["metadata"]["subject"]
                mca_run = entry["metadata"]["run"]
                session = entry["metadata"]["session"]
                sub_run = entry["metadata"]["sub_run"]
                data_type = entry["metadata"]["type"]

                if session_filter is not None and session != session_filter:
                    continue

                if metric_name in entry["metrics"].get("local_metrics", {}):
                    values_dict = entry["metrics"]["local_metrics"][metric_name]
                elif metric_name in entry["metrics"].get("global_metrics", {}):
                    values_dict = {
                        "global": entry["metrics"]["global_metrics"][metric_name]
                    }
                else:
                    continue

                config_key = (session, sub_run, data_type)

                for region, value in values_dict.items():
                    # Assign values by run ID to ensure exact cross-subject alignment for the pooled variance formula
                    region_config_subject_data[region][config_key][subject_id][
                        mca_run
                    ] = value

            if not region_config_subject_data:
                npvr_values.append(0)
                sigma_num_all_regions.append([0])
                sigma_pop_all_regions.append([0])
                continue

            region_sigma_num = []
            region_sigma_pop = []

            # Accumulators for pooled variance calculation
            ss_num = 0.0
            df_num = 0
            ss_pop = 0.0
            df_pop = 0

            for region, config_data in region_config_subject_data.items():
                for config_key, subject_data in config_data.items():
                    all_subjects = sorted(list(subject_data.keys()))
                    m = len(all_subjects)

                    if m < 2:
                        continue

                    all_mca_runs = set()
                    for subj_data in subject_data.values():
                        # Collect all unique run IDs across subjects to account for missing or non-sequential runs
                        all_mca_runs.update(subj_data.keys())

                    # Sort run IDs to establish a deterministic row order for the alignment matrix
                    all_mca_runs = sorted(list(all_mca_runs))
                    n_mca_runs = len(all_mca_runs)

                    if n_mca_runs < 2:
                        continue

                    X = np.full((n_mca_runs, m), np.nan)

                    # Map each run ID to a matrix row index to guarantee strict cross-subject alignment
                    run_to_idx = {run: i for i, run in enumerate(all_mca_runs)}

                    for j, subj in enumerate(all_subjects):
                        subj_mca_values = subject_data[subj]
                        for run, mca_val in subj_mca_values.items():
                            X[run_to_idx[run], j] = mca_val

                    # Calculate Numerical Variability (across MCA runs for each subject)
                    for j in range(m):
                        col = X[:, j]
                        valid_vals = col[~np.isnan(col)]
                        if len(valid_vals) > 1:
                            var_j = np.var(valid_vals, ddof=1)

                            # 1. Keep for boxplot distribution
                            region_sigma_num.append(np.sqrt(var_j))

                            # 2. Accumulate for pooled standard deviation
                            df_j = len(valid_vals) - 1
                            ss_num += var_j * df_j
                            df_num += df_j

                    # Calculate Population Variability (across subjects for each MCA run)
                    for i in range(n_mca_runs):
                        row = X[i, :]
                        valid_vals = row[~np.isnan(row)]
                        if len(valid_vals) > 1:
                            var_i = np.var(valid_vals, ddof=1)

                            # 1. Keep for boxplot distribution
                            region_sigma_pop.append(np.sqrt(var_i))

                            # 2. Accumulate for pooled standard deviation
                            df_i = len(valid_vals) - 1
                            ss_pop += var_i * df_i
                            df_pop += df_i

            sigma_num_all_regions.append(
                region_sigma_num if region_sigma_num else [0]
            )
            sigma_pop_all_regions.append(
                region_sigma_pop if region_sigma_pop else [0]
            )

            # Calculate POOLED standard deviations (Root of the weighted mean of variances)
            pooled_sigma_num = np.sqrt(ss_num / df_num) if df_num > 0 else 0.0
            pooled_sigma_pop = np.sqrt(ss_pop / df_pop) if df_pop > 0 else 0.0

            # Calculate NPVR using the pooled values
            npvr = (
                pooled_sigma_num / pooled_sigma_pop
                if pooled_sigma_pop != 0
                else 0.0
            )
            npvr_values.append(npvr)

        return (
            thresholds,
            sigma_num_all_regions,
            sigma_pop_all_regions,
            npvr_values,
        )

    return (calculate_npvr_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Please note that the follwing plotting code was generated by Qwen3.6-Plus
    """)
    return


@app.cell
def _(
    Line2D,
    Patch,
    ScalarFormatter,
    calculate_npvr_data,
    defaultdict,
    figures_output_path,
    graph_metrics,
    np,
    plt,
):
    def get_session_entry_counts(fc_metrics):
        """
        Count total number of entries for each session (same across all thresholds).
        Returns: dict mapping session_name -> entry count
        """
        session_counts = defaultdict(int)

        # Just look at the first threshold since counts are identical
        first_threshold = sorted(fc_metrics.keys())[0]
        for entry in fc_metrics[first_threshold]:
            session = entry["metadata"]["session"]
            session_counts[session] += 1

        return dict(session_counts)


    def plot_single_ax(
        ax, x_positions, sigma_num_data, sigma_pop_data, npvr_values, title
    ):
        """Plot boxplots and NPVR line aligned on the same x-coordinates."""
        boxplot_data = []
        boxplot_positions = []
        boxplot_colors = []

        for i in range(len(x_positions)):
            boxplot_data.append(sigma_num_data[i])
            boxplot_positions.append(x_positions[i] - 0.2)
            boxplot_colors.append("blue")

            boxplot_data.append(sigma_pop_data[i])
            boxplot_positions.append(x_positions[i] + 0.2)
            boxplot_colors.append("orange")

        bp = ax.boxplot(
            boxplot_data,
            positions=boxplot_positions,
            widths=0.35,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.3, color="gray"),
        )

        for patch, color in zip(bp["boxes"], boxplot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        plt.setp(bp["medians"], color="black", linewidth=2)
        plt.setp(bp["whiskers"], color="black", linewidth=1, linestyle="-")
        plt.setp(bp["caps"], color="black", linewidth=1)

        for xpos in x_positions:
            ax.axvline(
                x=xpos, color="gray", linestyle="-", alpha=0.2, linewidth=0.5
            )

        ax2 = ax.twinx()
        ax2.plot(
            x_positions,
            npvr_values,
            "r--*",
            linewidth=2.5,
            markersize=10,
            markerfacecolor="red",
            markeredgecolor="black",
            markeredgewidth=1,
        )

        ax.set_ylabel("NV, PV", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Averaged NPVR", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

        ax.set_xticks(x_positions)

        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax2.grid(False)

        ax.tick_params(axis="both", which="major", labelsize=9)
        ax2.tick_params(axis="both", which="major", labelsize=9)


    def create_session_comparison_plot(fc_metrics):
        """Create a figure with columns for each session + Global for each metric."""
        metrics = [
            ("degree_centrality", "Degree Centrality"),
            ("clustering_coefficient", "Clustering Coefficient"),
            ("betweenness_centrality", "Betweenness Centrality"),
            ("eigenvector_centrality", "Eigenvector Centrality"),
            ("average_shortest_path_length", "Average Shortest Path Length"),
        ]

        # Get entry counts for each session
        session_entry_counts = get_session_entry_counts(fc_metrics)
        all_sessions = sorted(list(session_entry_counts.keys()))
        n_sessions = len(all_sessions)

        n_cols = n_sessions + 1
        n_metrics = len(metrics)

        fig_width = 6 * n_cols
        fig, axes = plt.subplots(
            n_metrics, n_cols, figsize=(fig_width, 4 * n_metrics), sharex="col"
        )

        if n_metrics == 1:
            axes = [axes]

        thresholds = sorted(fc_metrics.keys())
        threshold_labels = [
            f"T={t}" if t != int(t) else f"T={int(t)}" for t in thresholds
        ]
        x_positions = np.arange(len(thresholds))

        # Create column titles with entry counts
        column_titles = []
        total_entries = 0
        for session in all_sessions:
            entry_count = session_entry_counts[session]
            total_entries += entry_count
            column_titles.append(f"{session}\n({entry_count} entries)")

        column_titles.append(f"Global\n({total_entries} entries)")

        for idx, (metric_key, metric_title) in enumerate(metrics):
            # Plot each session
            for col_idx, session in enumerate(all_sessions):
                _, sigma_num, sigma_pop, npvr = calculate_npvr_data(
                    fc_metrics, metric_key, session_filter=session
                )
                plot_single_ax(
                    axes[idx][col_idx],
                    x_positions,
                    sigma_num,
                    sigma_pop,
                    npvr,
                    f"{metric_title}\n{column_titles[col_idx]}",
                )

            # Plot Global (last column)
            _, sigma_num, sigma_pop, npvr = calculate_npvr_data(
                fc_metrics, metric_key, session_filter=None
            )
            plot_single_ax(
                axes[idx][n_cols - 1],
                x_positions,
                sigma_num,
                sigma_pop,
                npvr,
                f"{metric_title}\n{column_titles[n_cols - 1]}",
            )

        # Format bottom axes
        for col in range(n_cols):
            axes[-1][col].set_xticks(x_positions)
            axes[-1][col].set_xticklabels(threshold_labels, fontsize=9)
            axes[-1][col].set_xlabel(
                "Threshold Values", fontsize=10, fontweight="bold"
            )

        # Create shared legend
        legend_elements = [
            Patch(
                facecolor="blue",
                alpha=0.6,
                edgecolor="blue",
                label="Numerical Variability (NV), $\\sigma$_num",
            ),
            Patch(
                facecolor="orange",
                alpha=0.6,
                edgecolor="orange",
                label="Population Variability (PV), $\\sigma$_pop",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="red",
                linestyle="--",
                markersize=8,
                label="Average NPVR",
                markerfacecolor="red",
                markeredgecolor="black",
                linewidth=2,
            ),
        ]

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=3,
            fontsize=9,
            frameon=False,
        )

        plt.tight_layout()
        return fig


    _fig = create_session_comparison_plot(graph_metrics)


    _fig.suptitle(
        "Variation of NPVR per threshold for each graph metric",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    _fig.savefig(
        figures_output_path / "graph_metrics_npvr.png",
        bbox_inches="tight",
    )
    plt.show()
    return (plot_single_ax,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now want to do the same plots but with the `without-confound` minus `with-confound`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Please note that for now, this complete cell has been written by Qwen3.6-Plus and need to be double checked
    """)
    return


@app.cell
def _(
    Line2D,
    Patch,
    calculate_npvr_data,
    defaultdict,
    figures_output_path,
    graph_metrics,
    np,
    plot_single_ax,
    plt,
):
    def calculate_npvr_difference(fc_metrics, metric_name, session_filter=None):
        """
        Calculate σ_num, σ_pop, and NPVR for the difference (without-confound - with-confound).

        This function pre-processes the data to compute the differences, then delegates
        the actual NPVR calculation to calculate_npvr_data to ensure statistical consistency
        (using pooled standard deviation) and avoid code duplication.
        """
        thresholds = sorted(fc_metrics.keys())
        fc_metrics_diff = {}

        for threshold in thresholds:
            entries = fc_metrics[threshold]

            # Group by (subject, session, sub_run, mca_run, region) to find pairs
            grouped_values = defaultdict(dict)

            for entry in entries:
                subject_id = entry["metadata"]["subject"]
                mca_run = entry["metadata"]["run"]
                session = entry["metadata"]["session"]
                sub_run = entry["metadata"]["sub_run"]
                data_type = entry["metadata"]["type"]

                if metric_name in entry["metrics"].get("local_metrics", {}):
                    values_dict = entry["metrics"]["local_metrics"][metric_name]
                elif metric_name in entry["metrics"].get("global_metrics", {}):
                    values_dict = {
                        "global": entry["metrics"]["global_metrics"][metric_name]
                    }
                else:
                    continue

                for region, value in values_dict.items():
                    key = (subject_id, session, sub_run, mca_run, region)
                    grouped_values[key][data_type] = value

            # Create new synthetic entries for the differences
            entry_cache = {}

            for key, type_values in grouped_values.items():
                subject_id, session, sub_run, mca_run, region = key

                if (
                    "without-confound" in type_values
                    and "with-confound" in type_values
                ):
                    diff_value = (
                        type_values["without-confound"]
                        - type_values["with-confound"]
                    )

                    cache_key = (subject_id, session, sub_run, mca_run)
                    if cache_key not in entry_cache:
                        entry_cache[cache_key] = {
                            "metadata": {
                                "subject": subject_id,
                                "run": mca_run,
                                "session": session,
                                "sub_run": sub_run,
                                "type": "difference",
                            },
                            "metrics": {"local_metrics": {}, "global_metrics": {}},
                        }

                    # Place the difference in the correct metric dictionary
                    if region == "global":
                        entry_cache[cache_key]["metrics"]["global_metrics"][
                            metric_name
                        ] = diff_value
                    else:
                        if (
                            metric_name
                            not in entry_cache[cache_key]["metrics"][
                                "local_metrics"
                            ]
                        ):
                            entry_cache[cache_key]["metrics"]["local_metrics"][
                                metric_name
                            ] = {}
                        entry_cache[cache_key]["metrics"]["local_metrics"][
                            metric_name
                        ][region] = diff_value

            fc_metrics_diff[threshold] = list(entry_cache.values())

        # Delegate to the main function, which will handle the pooled variance calculation
        return calculate_npvr_data(
            fc_metrics_diff, metric_name, session_filter=session_filter
        )


    def get_difference_entry_counts(fc_metrics):
        """
        Count number of valid difference pairs (where both with and without confound exist).
        Returns: total number of valid pairs
        """
        valid_pairs = 0
        first_threshold = sorted(fc_metrics.keys())[0]

        # Group by (session, sub_run, subject, mca_run)
        pairs = defaultdict(dict)

        for entry in fc_metrics[first_threshold]:
            subject_id = entry["metadata"]["subject"]
            mca_run = entry["metadata"]["run"]
            session = entry["metadata"]["session"]
            sub_run = entry["metadata"]["sub_run"]
            data_type = entry["metadata"]["type"]

            key = (session, sub_run, subject_id, mca_run)
            pairs[key][data_type] = True

        # Count pairs that have both types
        for key, types in pairs.items():
            if "without-confound" in types and "with-confound" in types:
                valid_pairs += 1

        return valid_pairs


    def create_difference_plot(fc_metrics):
        """Create a figure showing NPVR of (without-confound - with-confound) differences."""
        metrics = [
            ("degree_centrality", "Degree Centrality"),
            ("clustering_coefficient", "Clustering Coefficient"),
            ("betweenness_centrality", "Betweenness Centrality"),
            ("eigenvector_centrality", "Eigenvector Centrality"),
            ("average_shortest_path_length", "Average Shortest Path Length"),
        ]

        # Get count of valid difference pairs
        valid_pairs = get_difference_entry_counts(fc_metrics)

        n_metrics = len(metrics)
        fig, axes = plt.subplots(
            n_metrics, 1, figsize=(10, 4 * n_metrics), sharex=True
        )

        if n_metrics == 1:
            axes = [axes]

        thresholds = sorted(fc_metrics.keys())
        threshold_labels = [
            f"T={t}" if t != int(t) else f"T={int(t)}" for t in thresholds
        ]
        x_positions = np.arange(len(thresholds))

        for idx, (metric_key, metric_title) in enumerate(metrics):
            _, sigma_num, sigma_pop, npvr = calculate_npvr_difference(
                fc_metrics, metric_key
            )
            plot_single_ax(
                axes[idx],
                x_positions,
                sigma_num,
                sigma_pop,
                npvr,
                f"{metric_title}\n(Without - With Confound)\n({valid_pairs} valid pairs)",
            )

        # Format bottom axis
        axes[-1].set_xticks(x_positions)
        axes[-1].set_xticklabels(threshold_labels, fontsize=9)
        axes[-1].set_xlabel("Threshold Values", fontsize=10, fontweight="bold")

        # Create shared legend
        legend_elements = [
            Patch(
                facecolor="blue",
                alpha=0.6,
                edgecolor="blue",
                label="Numerical Variability (NV), $\\sigma$_num",
            ),
            Patch(
                facecolor="orange",
                alpha=0.6,
                edgecolor="orange",
                label="Population Variability (PV), $\\sigma$_pop",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="red",
                linestyle="--",
                markersize=8,
                label="Average NPVR",
                markerfacecolor="red",
                markeredgecolor="black",
                linewidth=2,
            ),
        ]

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=3,
            fontsize=9,
            frameon=False,
        )

        plt.tight_layout()
        return fig


    _fig = create_difference_plot(graph_metrics)

    _fig.suptitle(
        "Variation of the difference of NPVR between without and with confound graph metrics per threshold",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    _fig.savefig(
        figures_output_path / "graph_metrics_npvr_confound_difference.png",
        bbox_inches="tight",
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now want to plot the NPVR across the different brain regions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Please note that for now, this complete cell has been written by Qwen3.7-Plus and need to be double checked
    """)
    return


@app.cell
def _(
    defaultdict,
    figures_output_path,
    fuzzy_fmriprep_analysis,
    graph_metrics,
    nib,
    np,
    plotting,
    plt,
):
    brain_maps_img = nib.load(fuzzy_fmriprep_analysis.brain_maps)


    def calculate_npvr_per_region(fc_metrics, metric_name, session_filter=None):
        """Calculate pooled NPVR for each brain region individually."""
        thresholds = sorted(fc_metrics.keys())
        results = {t: {} for t in thresholds}

        for threshold in thresholds:
            entries = fc_metrics[threshold]

            # Use nested dictionaries to explicitly map metric values to their specific MCA run identifiers
            region_config_subject_data = defaultdict(
                lambda: defaultdict(lambda: defaultdict(dict))
            )

            for entry in entries:
                subject_id = entry["metadata"]["subject"]
                mca_run = entry["metadata"]["run"]
                session = entry["metadata"]["session"]
                sub_run = entry["metadata"]["sub_run"]
                data_type = entry["metadata"]["type"]

                if session_filter is not None and session != session_filter:
                    continue

                if metric_name in entry["metrics"].get("local_metrics", {}):
                    values_dict = entry["metrics"]["local_metrics"][metric_name]
                elif metric_name in entry["metrics"].get("global_metrics", {}):
                    values_dict = {
                        "global": entry["metrics"]["global_metrics"][metric_name]
                    }
                else:
                    continue

                config_key = (session, sub_run, data_type)
                for region, value in values_dict.items():
                    # Assign values by run ID to ensure exact cross-subject alignment for the pooled variance formula
                    region_config_subject_data[region][config_key][subject_id][
                        mca_run
                    ] = value

            for region, config_data in region_config_subject_data.items():
                ss_num, df_num = 0.0, 0.0
                ss_pop, df_pop = 0.0, 0.0

                for config_key, subject_data in config_data.items():
                    all_subjects = sorted(list(subject_data.keys()))
                    m = len(all_subjects)
                    if m < 2:
                        continue

                    all_mca_runs = set()
                    for subj_data in subject_data.values():
                        # Collect all unique run IDs across subjects to account for missing or non-sequential runs
                        all_mca_runs.update(subj_data.keys())

                    # Sort run IDs to establish a deterministic row order for the alignment matrix
                    all_mca_runs = sorted(list(all_mca_runs))
                    n_mca_runs = len(all_mca_runs)
                    if n_mca_runs < 2:
                        continue

                    X = np.full((n_mca_runs, m), np.nan)

                    # Map each run ID to a matrix row index to guarantee strict cross-subject alignment
                    run_to_idx = {run: i for i, run in enumerate(all_mca_runs)}

                    for j, subj in enumerate(all_subjects):
                        subj_mca_values = subject_data[subj]
                        for run, mca_val in subj_mca_values.items():
                            X[run_to_idx[run], j] = mca_val

                    for j in range(m):
                        valid_vals = X[:, j][~np.isnan(X[:, j])]
                        if len(valid_vals) > 1:
                            var_j = np.var(valid_vals, ddof=1)
                            df_j = len(valid_vals) - 1
                            ss_num += var_j * df_j
                            df_num += df_j

                    for i in range(n_mca_runs):
                        valid_vals = X[i, :][~np.isnan(X[i, :])]
                        if len(valid_vals) > 1:
                            var_i = np.var(valid_vals, ddof=1)
                            df_i = len(valid_vals) - 1
                            ss_pop += var_i * df_i
                            df_pop += df_i

                pooled_sigma_num = np.sqrt(ss_num / df_num) if df_num > 0 else 0.0
                pooled_sigma_pop = np.sqrt(ss_pop / df_pop) if df_pop > 0 else 0.0
                npvr = (
                    pooled_sigma_num / pooled_sigma_pop
                    if pooled_sigma_pop != 0
                    else np.nan
                )

                results[threshold][region] = {
                    "pooled_sigma_num": pooled_sigma_num,
                    "pooled_sigma_pop": pooled_sigma_pop,
                    "npvr": npvr,
                }

        return results


    def create_npvr_nifti(npvr_dict, atlas_img):
        """Maps regional NPVR values to the Schaefer atlas."""
        atlas_data = atlas_img.get_fdata()
        npvr_map_data = np.zeros_like(atlas_data)

        for region_str, data in npvr_dict.items():
            if region_str == "global":
                continue
            region_id = int(region_str) + 1
            if 1 <= region_id <= 100:
                npvr_map_data[atlas_data == region_id] = data["npvr"]

        return nib.Nifti1Image(npvr_map_data, atlas_img.affine)


    def plot_npvr_brain_grid(fc_metrics, atlas_img):
        # Automatically extract all thresholds
        thresholds_list = sorted(fc_metrics.keys())

        # Automatically extract metrics from the first entry
        first_entry = fc_metrics[thresholds_list[0]][0]
        metrics_list = []

        if "local_metrics" in first_entry["metrics"]:
            metrics_list.extend(first_entry["metrics"]["local_metrics"].keys())

        vmin, vmax = 0.0, 1.0
        n_metrics = len(metrics_list)
        n_thresholds = len(thresholds_list)

        # Create one figure per metric
        figures = []

        for metric_idx, metric_name in enumerate(metrics_list):
            # Calculate NPVR for this metric across all thresholds once (optimization)
            npvr_results = calculate_npvr_per_region(fc_metrics, metric_name)

            # Create figure for this metric
            fig = plt.figure(figsize=(15, 5 * n_thresholds))

            # Calculate layout
            left_margin = 0.15
            right_margin = 0.90
            top_margin = 0.92
            bottom_margin = 0.05

            # Create GridSpec for this metric (rows = thresholds)
            gs = fig.add_gridspec(
                n_thresholds, 3, width_ratios=[1, 1, 1], wspace=0.1, hspace=0.3
            )

            # Plot each threshold as a row
            for i, thresh in enumerate(thresholds_list):
                if thresh not in npvr_results:
                    continue

                npvr_img = create_npvr_nifti(npvr_results[thresh], atlas_img)

                # Add threshold label on the left (y-axis label)
                ax_label = fig.add_subplot(gs[i, 0])
                ax_label.text(
                    0.0,
                    0.5,
                    f"Threshold = {thresh}",
                    ha="right",
                    va="center",
                    fontsize=20,
                    fontweight="bold",
                    transform=ax_label.transAxes,
                )
                ax_label.axis("off")

                # Sagittal slice
                ax_sag = fig.add_subplot(gs[i, 0])
                plotting.plot_stat_map(
                    npvr_img,
                    display_mode="x",
                    cut_coords=[2],
                    axes=ax_sag,
                    colorbar=False,
                    cmap="Reds",
                    vmin=vmin,
                    vmax=vmax,
                    black_bg=False,
                )
                ax_sag.set_title("")

                # Coronal slice
                ax_cor = fig.add_subplot(gs[i, 1])
                plotting.plot_stat_map(
                    npvr_img,
                    display_mode="y",
                    cut_coords=[0],
                    axes=ax_cor,
                    colorbar=False,
                    cmap="Reds",
                    vmin=vmin,
                    vmax=vmax,
                    black_bg=False,
                )
                ax_cor.set_title("")

                # Axial slice
                ax_axi = fig.add_subplot(gs[i, 2])
                plotting.plot_stat_map(
                    npvr_img,
                    display_mode="z",
                    cut_coords=[0],
                    axes=ax_axi,
                    colorbar=False,
                    cmap="Reds",
                    vmin=vmin,
                    vmax=vmax,
                    black_bg=False,
                )
                ax_axi.set_title("")

            # Adjust layout
            fig.subplots_adjust(
                left=left_margin,
                right=right_margin,
                top=top_margin,
                bottom=bottom_margin,
                wspace=0.1,
                hspace=0.3,
            )

            # Add metric as the main title
            fig.suptitle(
                metric_name.replace("_", " ").title(),
                fontsize=24,
                fontweight="bold",
                y=0.98,
            )

            # Add colorbar
            cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
            sm = plt.cm.ScalarMappable(
                cmap="Reds", norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            sm.set_array([])
            fig.colorbar(sm, cax=cax, label="NPVR")

            figures.append(fig)

        return figures


    figures = plot_npvr_brain_grid(graph_metrics, brain_maps_img)
    for _fig in figures:
        _fig.savefig(
            figures_output_path / "npvr_across_brain_regions.png",
            bbox_inches="tight",
        )
        plt.show()
    return


if __name__ == "__main__":
    app.run()
