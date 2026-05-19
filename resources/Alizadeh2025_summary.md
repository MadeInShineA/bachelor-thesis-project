# Numerical Variability of functional MRI Graph Measures Summary
> Written by Mina Alizadeh, Yohan Chatelain, Gregory Kiar and Tristan Glatard (see [here](https://www.biorxiv.org/content/10.64898/2025.12.22.695524v1)).

## Abstract

As pipelines become more computationally heavy, it's important to access for numerical stability to ensure results remain within an representative and acceptable range.

The results of the stability evaluation were a Numerical-Population Variability ratio expressed as:

$$\left(NPVR = \frac{\sigma_{num}}{\sigma_{pop}}\right)$$

Where $\sigma_{num}$ is the numerical variability and $\sigma_{pop}$ is the population variability. [^1]

ranging from 0.1 to 0.2.

It's also important to note that this ratio varies across the brain regions.


[^1]: See [here](https://www.biorxiv.org/content/10.64898/2026.01.09.698203v1) in section 2.2

## Introduction

The following are a known cause for numerical perturbations:
 - differences in hardware architecture
 - operating systems
 - compilers
 - parallelization strategies

 They are more and more present in pipelines involving high-dimensional computations such as linear and non linear image registration or the training of DL models.

 Studies such as  Mirhakimi et al., 2025 Chatelain et al., 2025 and Kiar, Chatelain, et al., 2020 have already studied the numerical variability of structural and diffusion MRI pipelines. Yet, it's impact on functional MRI pipelines are currently unexplored.This paper aims to assess how numerical perturbations affect the reliability of functional connectivity matrices and graph metrics across multiple datasets and pre-processing configurations.

## Materials and Methods

The fMRIPrep pipeline was used as the preprocessing framework on the PPMI dataset with a focus on cross-sectional analyses. The numerical variability was introduced with Monte Carlo Arithmetic (MCA), a technique that brings small perturbations to the the floating point operations, which allows to quantify how much variability comes from numerical errors.

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

 The data was converted from DICOM to NIfTI with HeuDiConv 1.2

 ### Data Processing

 The data were processed using fMRIPrep 23.2.1
