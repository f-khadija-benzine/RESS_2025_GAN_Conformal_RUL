# GAN-Augmented Conformal Multi-task Framework for Bearing RUL Prediction

Target journal: **Reliability Engineering & System Safety (RESS)**

## Setup (Google Colab)

### One-time: Store dataset on Google Drive
1. Download XJTU-SY from [Kaggle](https://www.kaggle.com/datasets/zwming/xjtu-sy)
2. Upload the zip to your Google Drive
3. Extract so you have: `MyDrive/datasets/XJTU-SY/35Hz12kN/` etc.

### Every session: Run setup
```python
!git clone https://github.com/YOUR_USERNAME/RESS_2025_GAN_Conformal_RUL.git
%cd RESS_2025_GAN_Conformal_RUL
%run notebooks/colab_setup.py
```
This mounts Google Drive, finds the dataset, installs dependencies.

## Notebooks (run in order)

| Notebook | Step | Description |
|---|---|---|
| colab_setup.py | 0 | Mount Drive, find data, install deps |
| 01_explore_data.py | 1 | Load and visualize XJTU-SY |
| 02_health_indicator.py | 2 | Health indicator + stage labels |
| 03_train_gan.py | 4-5 | Conditional GAN training |
| 04_train_model.py | 6 | Multi-task BiLSTM |
| 05_conformal.py | 7 | Conformal prediction |
| 06_ablation.py | 8-9 | Evaluation + ablation |

## Contributions

1. **Conditional GAN** — Stage-conditioned multivariate sensor augmentation
2. **Multi-task BiLSTM** — Joint RUL + degradation stage prediction
3. **Conformal Prediction** — Distribution-free coverage-guaranteed intervals
