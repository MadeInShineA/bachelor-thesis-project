# Numerical Variability of functional MRI Graph Measures Summary
> Written by Mina Alizadeh, Yohan Chatelain, Gregory Kiar and Tristan Glatard (see [here](https://www.biorxiv.org/content/10.64898/2025.12.22.695524v1)).

Please note that some sentences, notably the tools, techniques or preprocessing enumerations are mostly taken as is from the research paper.

## Abstract

As pipelines become more computationally heavy, it's important to access for numerical stability to ensure results remain within an representative and acceptable range.

The results of the stability evaluation were a Numerical-Population Variability ratio expressed as:

$$NPVR = \frac{\sigma_{num}}{\sigma_{pop}}$$

Where $\sigma_{num}$ is the numerical variability and $\sigma_{pop}$ is the population variability.

ranging from 0.1 to 0.2.

It's also important to note that this ratio varies across the brain regions.

## Introduction

The following are a known cause for numerical perturbations:
 - differences in hardware architecture
 - operating systems
 - compilers
 - parallelization strategies

 They are more and more present in pipelines involving high-dimensional computations such as linear and non linear image registration or the training of DL models.

 Studies such as  Mirhakimi et al., 2025 Chatelain et al., 2025 and Kiar, Chatelain, et al., 2020 have already studied the numerical variability of structural and diffusion MRI pipelines. Yet, it's impact on functional MRI pipelines are currently unexplored.This paper aims to assess how numerical perturbations affect the reliability of functional connectivity matrices and graph metrics across multiple datasets and pre-processing configurations.

## Materials and Methods

The `fMRIPrep` pipeline was used as the preprocessing framework on the PPMI dataset with a focus on cross-sectional analyses. The numerical variability was introduced with Monte Carlo Arithmetic (MCA) via `fuzzy lib math`, a technique that brings small perturbations to the the floating point operations, which allows to quantify how much variability comes from numerical errors.

### Dataset

Individuals with available resting state (rs) FMRI data were selected from the PPMI dataset, including patients with and without Parkinson's disease (PD).

Only the right left (RL) runs of the 38 healthy controls and 147 Parkinson's patients was analyzed. The RL rs-FMRI acquisition takes 240 volumes with 10 minutes of total scan time.
Here are the different settings of the acquisition phase:
 - The repetition time (TR) is 2500ms. 
 - The echo time is 30ms.
 - The slice thickness is 3.5mm.
 - There were 40 axial slices.
 - The field of view (FOV) is 224x224mm.
 - The matrix size is 64x64.
 - Participants were instructed to close their eyes and remain still.

The data was converted from `DICOM` to `NIfTI` with `HeuDiConv 1.2`

### Data Processing

The data were processed using `fMRIPrep 23.2.1`. The following operations were applied to
  - T1-weighted images:
    - Intensity non-uniformity correction
    - Skull stripping
    - Tissue segmentation
    - Non linear normalization to the MNI152NLin2009cAsym template
  - BOLD time series
    - Generation of a BOLD reference volume
    - Head motion correction
    - Co-registration of the BOLD references to the T1-weighted image with boundary-based registration
    - Transformation into MNI space using the composed anatomical and functional transformations

[comment]: <> TODO: Add the confound regressor fMRIPrep produced

After the preprocessing, functional connectomes were generated with `Nilearn`.
Regional time series were extracted using `NiftiLabelsMasker` applied to the preprocessed BOLD images with the Schaefer 2018 parcellation with 100 cortical regions and 7 functional networks. A Spatial smoothing of 6mm FWHM and temporal standardization were applied during masking.

Then 2 versions of the functional connectome were created for each subject and each MCA run:
  - One with confound regression of the 6 main sets of motion parameters (translations and rotations)
  - One without confound regression isolating numerical variability arising solely from preprocessing.

  For each version, a Pearson correlation matrix was computed using Nilearn's `ConnectivityMeasure` resulting in 1 functional connectivity matrix per subjet per MCA run.

  ### Graph Metrics

  Since correlation matrices are complete graphs, 6 correlations values were used as threshold (0.05, 0.1, 0.2, 0.3, 0.4, 0.5) to then generate binarized adjacency matrices.

  Then based on those networks, 4 local graph metrics were computed:
    - Degree centrality
    - Clustering coefficient
    - Betweenness centrality
    - Eigenvector centrality

  As well as 2 global metrics:
    - Small-worldness
    - Average shortest path length

All measures were computed using `Pzthon 3.12` and `Network 3.5`.

### Evaluation of Numerical Variability

The fMRIPrep pipeline was repeated 10 times per subject, allowing to gather a distribution of outputs that have the expected numerical variability across OS or library configurations.

### Numerical variability measures

Here is how the NPVR is calculated in details:

$$
  \sigma_{num}^2 = \frac{1}{m} \sum_{j=1}^m \left[ \frac{1}{n-1} \sum_{i=1}^n \left( x_{i,j} - \bar x_{.,j}\right)^2\right]
$$

$$
  \sigma_{pop}^2 = \frac{1}{n} \sum_{i=1}^n \left[ \frac{1}{m-1} \sum_{j=1}^m \left( x_{i,j} - \bar x_{i,.}\right)^2\right]
$$

$$
  NPVR = \frac{\sigma_{num}}{\sigma_{pop}}
$$


Where $\sigma_{num}$ is the numerical variability, $\sigma_{pop}$ is the population variability, $x_{i,j}$ is the measurement for subject $j$ in MCA repetition $i$, $\bar x_{.,j}$ and $\bar x_{i,.}$ are column and row means, $n$ is the total number of MCA repetitions and $m$ is the number of subjects.

Higher NPVR values indicate regions where computational variability potentially
compromises the detection of true population differences

## Results

### The influence of numerical variability on statistical inference
