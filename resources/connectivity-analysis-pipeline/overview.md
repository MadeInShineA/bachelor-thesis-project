# Connectivity analysis pipeline basic explanations

Please note that this document tries to explain what that the connectivity pipeline is with my current understanding on 01.06.2026

## What is an connectivity analysis pipeline?

A connectivity analysis pipeline refers to the the different steps used to estimate the functional connectivity of a brain
from already preprocessed neuro imaging data. This is done by computing functional connectivity matrices and graph metrics based on them.

There are a lot of different possible setting to take into account when designing such a pipeline.

## What are the different steps ?

The different steps are as follow:
  1. Confound regression
   - Allows to clean the preprocessed data a step further by cleaning it relatively to the non brain activity noise (confound)
  2. Parcellation atlas
   - How to establish the different brain regions we want to work on
  3. ROI time series extraction
   - Compute the different BOLD signal variation across the different time points per brain region
  4. FC matrix computation
   - Compute the correlation per brain region based on the BOLD signal variation
  5. Convert the FC matrix to a binary or weighted graph
  6. Compute graph metrics
   - Either local metrics or global ones

## What could the pipeline look like for the fuzzy fMRIPrep version ?

### Confound regression

I think it could be interesting to create the different FC matrices once with and without the confound regression.
This could allow some interesting information about how confound regression affects the NPVR.

### Parcellation atlas

I currently haven't looked too much in details what the different options are. Maybe I could go with the same configuration as
[this paper](https://www.biorxiv.org/content/10.64898/2025.12.22.695524v3.full.pdf) which means:

Schaefer 2018 parcellation with 100 cortical regions and 7 functional networks (fetch atlas schaefer 2018, n ROIs = 100, n networks = 7)

### ROI time series extraction

I don't thing there are different settings for this type of the pipeline. At least that I'm currently aware of.

### FC matrix computation

Same as above

### Convert the FC matrix to a binary or weighted graph (TBD)

I think I will have to look at how the different MDD bio-markers are obtained to determine if I want to compute a binary or weighted graph. This is currently what I'm trying to understand.

### Compute graph metrics (TBD)

Those would depend of how the bio-markers are obtained. It is still TBD.
