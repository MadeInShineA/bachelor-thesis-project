import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return (Path,)


@app.cell
def _(Path):
    data_path = Path("/home/cbi-biomark/olivier.amacker/derivatives/fmriprep")
    return (data_path,)


@app.cell
def _(data_path):
    run_paths = [run for run in data_path.iterdir() if run.is_dir()]

    run_paths
    return (run_paths,)


@app.cell
def _(run_paths):
    subject_paths = [subject_path for subject_paths in run_paths for subject_path in subject_paths.iterdir() if subject_path.is_dir() and "sub" in subject_path.name]

    subject_paths
    return (subject_paths,)


@app.cell
def _():
    class FuzzyFmriprepSub:
        def __init__(self, path, figures_path, anat_paths, fmap_paths, func_paths):
            self.path = path
            self.figures_path = figures_path
            self.anat_paths = anat_paths
            self.fmap_paths = fmap_paths
            self.func_paths = func_paths


    class FuzzyFmriprepAnalysis:
        def __init__(self, subjects):
            self.subjects = subjects

    return FuzzyFmriprepAnalysis, FuzzyFmriprepSub


@app.function
def get_output_paths_from_parent_path(parent_path, output_dir_names):
    output_paths = [output_path for output_path in parent_path.iterdir() if output_path.is_dir()]

    res = {output_dir_name: None for output_dir_name in output_dir_names}
    
    for output_path in output_paths:
        output_path_name = output_path.name
        if output_path_name in output_dir_names:
            res[output_path_name] = output_path

    return res


@app.cell
def _(FuzzyFmriprepAnalysis, FuzzyFmriprepSub, subject_paths):
    fuzzy_fmriprep_subjects = []

    for subject_path in subject_paths:
        subject_dir_paths = [subject_dir for subject_dir in subject_path.iterdir() if subject_dir.is_dir()]

        figures_path = None
    
        anat_paths = []
        func_paths = []
        fmap_paths = []
    
        for subject_dir_path in subject_dir_paths:

            dir_name = subject_dir_path.name

            if dir_name == "figures":
                figures_path = subject_dir_path
            
            if dir_name == "anat":
                anat_paths.append(subject_dir_path)
            
            if "ses-" in dir_name and not "ses-multi" in dir_name:
                output_paths = get_output_paths_from_parent_path(subject_dir_path, ["anat", "fmap", "func"])

                missing_directories = []
                for key, value in output_paths.items():
                    if value is None:
                        missing_directories.append(key)

                if len(missing_directories) > 0:
                    print(f"Skipping {subject_dir_path} since the following directories are missing: {missing_directories}")
                    break
                    
            
                anat_path = output_paths["anat"]
                fmap_path = output_paths["fmap"]
                func_path = output_paths["func"]

                anat_paths.append(anat_path)
                fmap_paths.append(fmap_path)
                func_paths.append(func_path)
            
        fuzzy_fmriprep_subject = FuzzyFmriprepSub(path=subject_path, figures_path=figures_path, anat_paths=anat_paths, fmap_paths=fmap_paths, func_paths=func_paths)
    
        fuzzy_fmriprep_subjects.append(fuzzy_fmriprep_subject)

    print(f"Collected {len(fuzzy_fmriprep_subjects)} subject outputs")
    fuzzzy_fmriprep_analysis = FuzzyFmriprepAnalysis(fuzzy_fmriprep_subjects)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
