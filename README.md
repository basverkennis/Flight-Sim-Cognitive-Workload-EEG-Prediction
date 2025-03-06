# Predicting Workload in Virtual Flight Simulations using EEG Spectral and Connectivity Features

Our paper has been published in the proceedings of the 7th IEEE International Conference on Artificial Intelligence & eXtended and Virtual Reality, as part of the special session on AI-driven Brain-Computer Interfaces (BCI) with VR/AR: Current Innovations and Future Directions. **You can access the paper here: [Predicting Workload in Virtual Flight Simulations using EEG Spectral and Connectivity Features](https://ieeexplore.ieee.org/abstract/document/10896018).**

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

### Contact
For any questions or inquiries regarding this repository, please reach out to us via email at bas.verkennis.code@gmail.com.

<img src="https://github.com/basverkennis/Flight-Sim-Cognitive-Workload-EEG-Prediction/blob/main/logo.jpeg" alt="Tilburg University Logo" width="20%" padding="5%">

### Citation
If you find this work or the code provided in this repository useful for your research, please consider citing our paper once it becomes available. The citation in BibTeX format is:

```bibtex
@INPROCEEDINGS{Verkennis2025WorkloadPredictionBCIxVR,
  author={Verkennis, Bas and van Weelden, Evy and Marogna, Francesca L. and Alimardani, Maryam and Wiltshire, Travis J. and Louwerse, Max M.},
  booktitle={2025 IEEE International Conference on Artificial Intelligence and eXtended and Virtual Reality (AIxVR)}, 
  title={Predicting Workload in Virtual Flight Simulations Using EEG Spectral and Connectivity Features}, 
  year={2025},
  volume={},
  number={},
  pages={82-89},
  keywords={Training;Solid modeling;Adaptation models;Accuracy;Aerospace simulation;Virtual reality;Brain modeling;Electroencephalography;Real-time systems;Brain-computer interfaces;Brain-Computer Interface (BCI);Cognitive Workload;Virtual Reality (VR);Flight Simulation;Electroen-cephalogram (EEG);Functional Connectivity;Phase-Locking Value (PLV);NASA Task Load Index (NASA-TLX)},
  doi={10.1109/AIxVR63409.2025.00019}
}
