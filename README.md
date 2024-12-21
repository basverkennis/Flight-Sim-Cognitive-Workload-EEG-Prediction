# Predicting Workload in Virtual Flight Simulations using EEG Spectral and Connectivity Features

**Note: Our paper, "Predicting Workload in Virtual Flight Simulations using EEG Spectral and Connectivity Features," has been accepted for presentation at the 7th IEEE International Conference on Artificial Intelligence & eXtended and Virtual Reality, as part of the special session on AI-driven Brain-Computer Interfaces (BCI) with VR/AR: Current Innovations and Future Directions. The camera-ready version has been submitted and will be presented at the conference, followed by publication in the proceedings.

Details of the publication will be provided once available.**

### Research Overview
In this research, we aim to investigate the predictive potential of frontal-parietal connectivity in determining cognitive workload during flight simulation. We utilize EEG connectivity features to analyze the brain's electrical activity and its relationship to cognitive workload levels. The study emphasizes the potential of EEG-based Brain-Computer Interfaces (BCIs) for real-time workload assessment in dynamic and immersive environments.

### Code and Features
The predictive modeling approach uses ensemble-based stacked classifiers, trained on EEG feature sets, to distinguish between high and low workload conditions. Recursive Feature Elimination (RFE) was employed to identify influential features. Two main feature sets were extracted:

- Spectral Features: Derived from the alpha, beta, and theta band ranges.
- Connectivity Features: Phase Locking Value (PLV) measures (derived from the alpha, beta, and theta band ranges).

### Repository Structure
- raw/: Contains the original datasets, including preprocessed EEG data and processed spectral feature data for each participant, categorized by VR and Desktop conditions. (Not available on GitHub.)
- data/: Stores processed feature sets per participant, categorized by VR and Desktop conditions, as .npz files. (Not available on GitHub.)
- code/: Includes scripts for data preprocessing, feature extraction, feature selection, ensemble modeling, and evaluation.
- results/: Contains plots and performance metrics derived from the experiments.

### Citation
If you find this work or the code provided in this repository useful for your research, please consider citing our paper once it becomes available. Citation details will be provided in the publication.

### Contact
For any questions or inquiries regarding this repository, please reach out to us via email at bas.verkennis.code@gmail.com.

<img src="https://github.com/basverkennis/Flight-Sim-Cognitive-Workload-EEG-Prediction/blob/main/logo.jpeg" alt="Tilburg University Logo" width="20%" padding="5%">
