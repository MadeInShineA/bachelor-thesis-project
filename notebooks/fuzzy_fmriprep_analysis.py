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
    from collections import defaultdict
    import marimo as mo
    from pathlib import Path
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.datasets import fetch_atlas_schaefer_2018
    import polars as pl
    import numpy as np

    return (
        ConnectivityMeasure,
        NiftiLabelsMasker,
        Path,
        defaultdict,
        fetch_atlas_schaefer_2018,
        json,
        mo,
        np,
        pl,
        re,
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
    ### Setting up the different classes and functions for the creation of the FC matrices
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
    The we want to collect all the different run paths
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
    Now we will create 2 classes:

    - The `FuzzyFmriprepSub` class which will represent a fuzzy fMRIPrep subject
    - The `FuzzyFmriprepAnalysis` with the different connectivity analysis functions and settings

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


@app.cell
def _(
    ConnectivityMeasure,
    NiftiLabelsMasker,
    Path,
    defaultdict,
    fetch_atlas_schaefer_2018,
    json,
    np,
    pl,
    re,
    sub_id,
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

        def set_preproc_bold_paths(self):
            res = []

            for func_path in self.func_paths:
                bolf_gz_paths = []
                bold_json_paths = []
                confounds_paths = []

                for file_path in func_path.iterdir():
                    if "desc-preproc_bold.nii.gz" in file_path.name:
                        bolf_gz_paths.append(file_path)
                    elif "desc-preproc_bold.json" in file_path.name:
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
                                "run_name": run_name,
                                "bold.nii.gz": gz_path,
                                "bold.json": json_path,
                                "confounds_timeseries.tsv": confound_path,
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

        def __init__(self, subjects: list[FuzzyFmriprepSub]):
            self.subjects = subjects

            # Setup the atlas and it's different elements
            self.atlas = fetch_atlas_schaefer_2018(
                n_rois=100, yeo_networks=7, resolution_mm=2
            )
            self.brain_maps = self.atlas["maps"]
            self.roi_labels = self.atlas["labels"]

            # Define how the connectivity matrices will be obtained
            self.connectivity_measure = ConnectivityMeasure(kind="correlation")

        def get_confounds(self, confound_file_path: Path) -> pl.DataFrame:
            confounds_df = pl.read_csv(
                confound_file_path, columns=self.CONFOUND_COLUMNS, separator="\t"
            )
            return confounds_df

        def get_masker(self, repetition_time: float) -> NiftiLabelsMasker:
            masker = NiftiLabelsMasker(
                labels_img=self.brain_maps,
                standardize="zscore_sample",
                standardize_confounds=True,
                smoothing_fwhm=6,
                detrend=True,
                t_r=repetition_time,
                memory="nilearn_cache",
                memory_level=1,
                verbose=0,
            )

            return masker

        def get_fc_matrices(self) -> dict:
            res = defaultdict(str)

            for subject in self.subjects:
                subject_id = subject.id
                subject_run = subject.run
                print(f"Processing {subject_id} from {subject_run}")

                for preproc_bold in subject.preproc_bold_paths:
                    run_name = preproc_bold["run_name"]
                    bold_nii_gz_path = preproc_bold["bold.nii.gz"]
                    bold_json_path = preproc_bold["bold.json"]
                    confounds_path = preproc_bold["confounds_timeseries.tsv"]

                    confounds = self.get_confounds(confounds_path)

                    repetition_time = None
                    with open(bold_json_path, "r") as f:
                        repetition_time = json.load(f)["RepetitionTime"]

                    # Generate the FC matrices with confound regression

                    with_confound_masker = self.get_masker(repetition_time)
                    with_confound_masker = with_confound_masker.fit_transform(
                        str(bold_nii_gz_path), confounds=confounds
                    )
                    with_confound_correlation = (
                        self.connectivity_measure.fit_transform(
                            [with_confound_masker]
                        )[0]
                    )

                    res[sub_id][run_name]["with_confounds"] = (
                        with_confound_correlation
                    )

                    # Gemerate the FC matrices without confound regression

                    without_confound_masker = self.get_masker(repetition_time)
                    without_confound_masker = (
                        without_confound_masker.fit_transform(
                            str(bold_nii_gz_path)
                        )
                    )
                    without_confound_correlation = (
                        self.connectivity_measure.fit_transform(
                            [without_confound_masker]
                        )[0]
                    )

                    res[sub_id][run_name]["without_confounds"] = (
                        without_confound_correlation
                    )
                break

            return res

        def save_fc_matrices(self, fc_matrices:dict, output_dir: Path):
        
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
            for sub_id, runs in self.fc_matrices.items():
                sub_dir = output_dir / sub_id
                sub_dir.mkdir(exist_ok=True)
                for run_name, versions in runs.items():
                    for version, matrix in versions.items():
                        out_path = sub_dir / f"{run_name}_{version}.npy"
                        np.save(out_path, matrix)
        
            print(f"\nSaved all connectomes to {output_dir}")


    FuzzyFmriprepSub, FuzzyFmriprepAnalysis
    return FuzzyFmriprepAnalysis, FuzzyFmriprepSub


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We define a utility function that will be used to transform the different subject paths into `FuzzyFmriprepSub`
    """)
    return


@app.cell
def _(Path):
    def get_output_paths_from_parent_path(parent_path: Path, output_dir_names: list[str]):
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
def _(fuzzy_fmriprep_analysis):
    fc_matrices = fuzzy_fmriprep_analysis.get_fc_matrices()
    return (fc_matrices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Save the FC matrices
    """)
    return


@app.cell
def _(Path, fc_matrices, fuzzy_fmriprep_analysis):
    fuzzy_fmriprep_analysis.save_fc_matrices(fc_matrices, Path("./res/fc-matrices"))
    return


if __name__ == "__main__":
    app.run()
