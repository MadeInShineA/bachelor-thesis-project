import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import polars as pl

    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.connectome import ConnectivityMeasure
    from joblib import Parallel, delayed

    from fuzzy_fmriprep_graph_metrics_analysis import (
        get_run_paths_from_data_path,
        get_subject_paths_from_run_paths,
        get_confounds,
        get_masker,
        save_fc_matrix,
        FuzzyFmriprepAnalysis,
        get_output_paths_from_parent_path,
        create_fmriprep_subjects,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Fuzzy fMRIPrep FC Matrices Analysis

    The goal of this notebook is to analyse the FC matrices obtained from the output of fuzzy-fmriprep preprocessing
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generate the different FC matrices
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We start by defining the different paths we will use throughout this notebook
    """)
    return


@app.cell
def _():
    data_path = Path("/home/cbi-biomark/olivier.amacker/derivatives/fmriprep/v2")
    return (data_path,)


@app.cell
def _():
    output_path = Path("./res/fuzzy-fmriprep-analysis/fc_matrices_analysis/")

    version = "v1"

    voxel_metrics_output_path = output_path / version / "voxel-metrics"
    fc_matrices_output_path = output_path / version / "fc-matrices"
    figures_output_path = output_path / version / "figures"
    graph_metrics_output_path = output_path / version / "graph-metrics"

    voxel_metrics_output_path.mkdir(parents=True, exist_ok=True)
    fc_matrices_output_path.mkdir(parents=True, exist_ok=True)
    figures_output_path.mkdir(parents=True, exist_ok=True)
    graph_metrics_output_path.mkdir(parents=True, exist_ok=True)

    (
        fc_matrices_output_path,
        figures_output_path,
        figures_output_path,
        graph_metrics_output_path,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Setup the different inputs

    In this section we will reuse some functions defined in [this notebook](https://olivier.amacker.dev/bachelor-thesis/site/notebooks/fuzzy-fmriprep-graph-metrics-analysis.html) (`fuzzy_fmriprep_graph_metrics_analysis`)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We define the subject class inspired from the one in the `fuzzy_fmriprep_graph_metrics_analysis` notebook
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This needs testing once the fMRIPrep runs are finished
    """)
    return


@app.cell
def _(re):
    class FuzzyFmriprepFCMatricesSub:
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

        def set_preproc_bold_paths(self):
            res = []

            for func_path in self.func_paths:
                dtseries_paths = []
                json_paths = []
                confounds_paths = []

                for file_path in func_path.iterdir():
                    file_name = file_path.name

                    # Find CIFTI dtseries files (91k density)
                    if "space-fsLR_den-91k_bold.dtseries.nii" in file_name:
                        if (
                            "dir-ap" in file_name.lower()
                            or "-ap_" in file_name.lower()
                            or "dir-" not in file_name.lower()
                        ):
                            dtseries_paths.append(file_path)

                    # Find corresponding CIFTI JSON files
                    elif "space-fsLR_den-91k_bold.json" in file_name:
                        if (
                            "dir-ap" in file_name.lower()
                            or "-ap_" in file_name.lower()
                            or "dir-" not in file_name.lower()
                        ):
                            json_paths.append(file_path)

                    # Find confounds files
                    elif "desc-confounds_timeseries.tsv" in file_name:
                        confounds_paths.append(file_path)

                for dtseries_path in dtseries_paths:
                    file_name = dtseries_path.name

                    # Derive matching JSON name
                    matching_json_name = file_name.replace("dtseries.nii", "json")

                    # Derive matching confound name
                    matching_confound_name = re.sub(
                        r"_space-fsLR_den-\d+k_bold\.dtseries\.nii$",
                        "_desc-confounds_timeseries.tsv",
                        file_name,
                    )

                    # Extract run and session for metadata
                    run_match = re.search(r"_run-(\d+)", file_name)
                    run_name = (
                        f"run-{run_match.group(1)}" if run_match else "run-single"
                    )

                    # Kept your specific session regex, but made it slightly more robust
                    session_match = re.search(r"_ses-(SFHARP00\d)", file_name)
                    session_name = (
                        session_match.group(1) if session_match else "ses-1"
                    )

                    # Find matching paths
                    matching_json_path = next(
                        (p for p in json_paths if p.name == matching_json_name),
                        None,
                    )
                    matching_confound_path = next(
                        (
                            p
                            for p in confounds_paths
                            if p.name == matching_confound_name
                        ),
                        None,
                    )

                    if (
                        matching_json_path is not None
                        and matching_confound_path is not None
                    ):
                        res.append(
                            {
                                "session_name": session_name,
                                "run_name": run_name,
                                "bold.dtseries.nii": dtseries_path,
                                "bold.json": matching_json_path,
                                "confounds_timeseries.tsv": matching_confound_path,
                            }
                        )
                    else:
                        print(
                            f"Warning: Could not find matching json or confounds for {file_name}"
                        )
                        print(f"  Looked for JSON: {matching_json_name}")
                        print(f"  Looked for Confounds: {matching_confound_name}")

            self.preproc_bold_paths = res

    return (FuzzyFmriprepFCMatricesSub,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We generate all the subjets from the `data_path`
    """)
    return


@app.cell
def _(FuzzyFmriprepFCMatricesSub, data_path):
    run_paths = get_run_paths_from_data_path(data_path)
    subject_paths = get_subject_paths_from_run_paths(run_paths)
    fmriprep_subjects = create_fmriprep_subjects(
        FuzzyFmriprepFCMatricesSub, subject_paths
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Define functions to process a single bold signal in in parallel

    This needs testing once the fMRIPrep runs are finished
    """)
    return


@app.cell
def _(json, signal, warnings):
    def process_single_bold(
        connectivity_measure: ConnectivityMeasure,
        preproc_bold: dict,
        subject_id: str,
        subject_run: str,
        brain_maps,
        confound_columns: list,
        confounds_to_filter_out: list,
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

        all_confounds = get_confounds(confounds_path, confound_columns).to_pandas()

        filtered_confounds_columns = [
            col for col in confound_columns if col not in confounds_to_filter_out
        ]

        filtered_confounds = get_confounds(
            confounds_path, filtered_confounds_columns
        ).to_pandas()

        repetition_time = None

        with open(bold_json_path, "r") as f:
            repetition_time = json.load(f)["RepetitionTime"]

        # Generate the FC matrices all the confounds

        raw_masker = get_masker(repetition_time, brain_maps)
        raw_ts = raw_masker.fit_transform(str(bold_nii_gz_path))

        with_all_confounds_ts = signal.clean(
            raw_ts,
            confounds=all_confounds,
            standardize="zscore_sample",
            standardize_confounds=True,
        )

        with_all_confounds_correlation = connectivity_measure.fit_transform(
            [with_all_confounds_ts]
        )[0]

        save_fc_matrix(
            with_all_confounds_correlation,
            subject_id,
            subject_run,
            session_name,
            run_name,
            "with-all-confounds",
            output_dir,
        )

        # Generate for only the filtered confounds

        filtered_confounds_ts = signal.clean(
            raw_ts,
            confounds=filtered_confounds,
            standardize="zscore_sample",
            standardize_confounds=True,
        )

        filtered_confounds_correlation = connectivity_measure.fit_transform(
            [filtered_confounds_ts]
        )[0]

        save_fc_matrix(
            filtered_confounds_correlation,
            subject_id,
            subject_run,
            session_name,
            run_name,
            "with-filtered-confounds",
            output_dir,
        )

        return {
            "subject_id": subject_id,
            "subject_run": subject_run,
            "session_name": session_name,
            "run_name": run_name,
            "with-all-confounds": with_all_confounds_correlation,
            "with-filtered-confounds": filtered_confounds_correlation,
        }

    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Define the `FuzzyFmriprepFCMatricesAnalysis` class

    This class is inspired from the one in the `fuzzy_fmriprep_graph_metrics_analysis` notebook, and will contain the functions used for the analysis

    This needs testing once the fMRIPrep runs are finished
    """)
    return


@app.cell
def _(FuzzyFmriprepGraphMetricsSub, defaultdict):
    class FuzzyFmriprepFCMatricesAnalysis(FuzzyFmriprepAnalysis):
        def __init__(
            self,
            subjects: list[FuzzyFmriprepGraphMetricsSub],
            confound_columns: list,
            confound_to_filter_out: list,
        ):
            super().__init__(subjects)

            self.confound_columns = confound_columns
            self.confound_to_filter_out = confound_to_filter_out

            """
            self.atlas = fetch_atlas_schaefer_2018(
                n_rois=100, yeo_networks=7, resolution_mm=2
            )
            self.brain_maps = self.atlas["maps"]
            self.roi_labels = self.atlas["labels"]
            self.connectivity_measures = ConnectivityMeasure(
                kind="correlation", standardize="zscore_sample"
            )
            """

        def generate_fc_matrices(
            self, parallel_function, output_dir: Path, max_workers=None
        ) -> dict:

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
                            self.confound_columns,
                            self.confound_to_filter_out,
                            output_dir,
                        )
                    )

            if max_workers is None:
                max_workers = len(tasks)

            print(
                f"Starting parallel processing of {len(tasks)} tasks with {max_workers} workers."
            )

            results = Parallel(n_jobs=max_workers, backend="loky")(
                delayed(parallel_function)(*task) for task in tasks
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

    return


if __name__ == "__main__":
    app.run()
