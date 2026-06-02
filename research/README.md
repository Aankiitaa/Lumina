# Research Package: Evaluating Spatial Normalization Techniques

This folder contains the complete research package for evaluating spatial normalization techniques on 3D skeletal landmark classification, specifically for Sign Language Recognition.

## Contents

- `experiment.py`: Main python script to run the benchmarks.
- `accuracy_comparison.png`: Generated visualization comparing metrics.
- `confusion_matrix_*.png`: Confusion matrices for each strategy.
- `experiment_results.json`: Raw metric outputs from the run.
- `research_paper.md`: The final formal computer science research paper.

## Requirements

To run the experiments, ensure you have the following packages installed:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## How to Run

Navigate to the `research` directory and run the python script:

```bash
cd research
python experiment.py
```

The script will automatically look for `your_landmarks.npy` and `your_labels.npy` in the parent directory, preprocess the data using three distinct normalization strategies, train a Random Forest classifier, and output all figures and metrics into this directory.
