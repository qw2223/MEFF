# Multi-Layer Exponential Family Factor Models for Integrative Analysis and Learning Disease Progression
Current diagnosis of neurological disorders often relies on late-stage clinical symptoms, which poses barriers for developing effective interventions at the premanifest stage. Recent research suggests that biomarkers and subtle changes in clinical markers may occur in a time-ordered fashion and can be used as indicators of early disease. 

In this project, we tackle challenges to leverage multi-domain markers to learn early disease  progression of neurological disorders. We propose to **integrate** heterogeneous types of measures from multiple domains  (e.g., discrete clinical symptoms, ordinal cognitive markers, continuous neuroimaging and blood biomarkers) using  a hierarchical Multi-layer Exponential Family Factor (MEFF) model with **lower-dimensional** latent factors. The latent factors are decomposed into **shared** factors across multiple domains  and **domain-specific** factors. The MEFF model also captures nonlinear trajectory of disease progression and order critical events of neurodegeneration measured by each marker.   

To overcome computational challenges, we  fit our model by approximate inference techniques for large-scale data. We apply the developed method to Parkinson's Progression Markers Initiative (PPMI) data to integrate biological, clinical and cognitive  markers  arising from heterogeneous distributions. The model learns lower-dimensional representations of Parkinson's disease (PD) and the temporal ordering of the neurodegeneration of PD.

* Author: **Qinxia Wang<sup>a</sup>,** **Yuanjia Wang<sup>a,b</sup>**
* Affilication: 
  + a. **Department of Biostatistics, Mailman School of Public Health, Columbia University, New York, NY, USA**
  + b. **Department of Psychiatry, Columbia University, New York, NY, USA**
  
## Modeling Framework
An example of the generative model of MEFF with two data modalities for a patient. The observations space is represented by the lower-dimensional latent factor shared across modalities and the unique latent factor  specific for each modality.
![](https://github.com/qw2223/MEFF/blob/main/figure/DAG.png)


## Application to PPMI 
The model estimation and inference can be conducted using the Python library [Edward](http://edwardlib.org/).

An example code for fitting MEFF model on [PPMI](https://www.ppmi-info.org/) data can be found in `MEFF_PPMI_example.ipynb`. The helper functions are in `MEFF_PPMI_function.py`.
