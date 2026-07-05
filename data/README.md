# Data Directory

Data is NOT stored in this repo. It lives on Google Drive and is mounted at runtime.

## One-time setup
1. Download XJTU-SY from https://www.kaggle.com/datasets/zwming/xjtu-sy
2. Upload zip to Google Drive
3. Extract to: MyDrive/datasets/XJTU-SY/

## Expected structure on Google Drive
```
MyDrive/datasets/XJTU-SY/
├── 35Hz12kN/
│   ├── Bearing1_1/
│   │   ├── 1.csv, 2.csv, ...
│   ├── Bearing1_2/
│   └── ...
├── 37.5Hz11kN/
│   └── ...
└── 40Hz10kN/
    └── ...
```

The colab_setup.py script auto-detects the dataset location on Drive.
