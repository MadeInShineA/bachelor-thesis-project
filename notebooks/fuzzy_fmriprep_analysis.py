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
    from pathlib import Path
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn import signal
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from joblib import Parallel, delayed

    return (
        ConnectivityMeasure,
        NiftiLabelsMasker,
        Parallel,
        Path,
        defaultdict,
        delayed,
        fetch_atlas_schaefer_2018,
        json,
        mo,
        np,
        pl,
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
    We also define the FC matrices output path
    """)
    return


@app.cell
def _(Path):
    fc_matrices_output_path = Path("./res/fc-matrices/")
    return (fc_matrices_output_path,)


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

        output_dir = Path(output_dir)
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
            "without-confounds": without_confound_correlation,
            "with-confounds": with_confound_correlation,
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
    np,
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
        ]

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

        def generate_fc_matrices(self, output_dir: Path, max_workers=None) -> dict:

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
                ][result["run_name"]]["without-confounds"] = result[
                    "without-confounds"
                ]
                res[result["subject_id"]][result["subject_run"]][
                    result["session_name"]
                ][result["run_name"]]["with-confounds"] = result["with-confounds"]
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

            for subject_dir in input_dir.iterdir():
                for run_dir in subject_dir.iterdir():
                    for session_dir in run_dir.iterdir():
                        for fc_matrix_file in session_dir.iterdir():
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyse the FC matrices
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We start by plotting some of the matrices
    """)
    return


@app.cell
def _(fc_matrices, plt, sns):
    total_matrices = 0

    subject_matrices = {
        subject: {"with-confound": [], "without-confound": []}
        for subject in fc_matrices.keys()
    }

    for _subject, run_values in fc_matrices.items():
        for run, session_values in run_values.items():
            for session, sub_run_values in session_values.items():
                for sub_run, type_values in sub_run_values.items():
                    for type, fc_matrix in type_values.items():
                        subject_matrices[_subject][type].append(fc_matrix)
                        total_matrices += 1

                        plt.figure(figsize=(8, 6))

                        # This code was generated by Qwen3.6-Plus

                        sns.heatmap(
                            fc_matrix,
                            cmap="RdBu_r",
                            vmin=-1,
                            vmax=1,
                            square=True,
                            cbar_kws={"shrink": 0.8},
                            xticklabels=False,
                            yticklabels=False,
                        )

                        plt.title(
                            f"Functional Connectivity Matrix of {_subject} {run} {session} {sub_run} ({type})"
                        )
                        plt.show()
    print(f"Total FC matrices: {total_matrices}")
    return (subject_matrices,)


@app.cell
def _(fc_matrices, np, plt, sns, subject_matrices):
    for subject, _type_values in subject_matrices.items():
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
                    f"Mean Functional Connectivity Matrix of {subject} ({_type})"
                )
                plt.show()
    return


if __name__ == "__main__":
    app.run()
