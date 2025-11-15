# Data Setup Guide

## ⚠️ Important: Data Not Included in Repository

The `Data/` folder is **NOT included** in this repository due to its large size (~6GB). You need to set it up separately.

## 📁 Expected Data Structure

The project expects data in the **parent directory**:

```
D:\Ptyxiakhh\
├── Data/                              ← Place your data here
│   ├── User1/
│   │   ├── Day1/
│   │   │   ├── Hand_Motion.txt       (Raw sensor data)
│   │   │   ├── Label.txt             (Activity labels)
│   │   │   ├── Data.csv              (Preprocessed features) ✓
│   │   │   └── Labels.csv            (Preprocessed labels) ✓
│   │   ├── Day2/
│   │   └── Day3/
│   ├── User2/
│   └── User3/
└── Smartphone_activity_recognition/   ← This project
    ├── src/
    ├── scripts/
    └── ...
```

## 🔄 Setup Options

### Option 1: Use Preprocessed Data (Recommended)

If you already have preprocessed `Data.csv` and `Labels.csv` files:

1. Place the `Data/` folder in `D:\Ptyxiakhh\`
2. You're ready to train models!

```bash
python scripts/train_baseline.py --model random_forest
```

### Option 2: Preprocess Raw Data

If you only have raw sensor files (`Hand_Motion.txt`, `Label.txt`):

1. Place raw data in the structure above
2. Run preprocessing:

```bash
cd D:\Ptyxiakhh
python Preprocessing_fixed.py
```

This will:
- Read raw sensor files
- Normalize features to [-1, 1]
- Extract activity labels (column 1 of Label.txt)
- Create `Data.csv` and `Labels.csv` files
- Takes ~25 minutes for full dataset

### Option 3: Use Reduced Dataset (Faster)

For slower laptops, the training scripts automatically use **30% of the data**:

- Training is 3x faster
- Still achieves good accuracy
- No additional setup needed!

## 📊 Dataset Information

- **Source**: Sussex-Huawei Locomotion (SHL) Dataset
- **Users**: 3 users
- **Days**: 3 days per user
- **Samples**: ~30 million total
- **Features**: 9 sensor features (Accelerometer, Gyroscope, Magnetometer - 3 axes each)
- **Labels**: 8 activity classes (Still, Walking, Running, Biking, Car, Bus, Train, Subway)

## 🔗 Obtaining the Dataset

The SHL dataset is publicly available. You can:

1. **Contact the project maintainer** for preprocessed data
2. **Download from SHL website**: [http://www.shl-dataset.org/](http://www.shl-dataset.org/)
3. **Use your own sensor data** (must match the format)

## ✅ Verify Data Setup

Check if data is correctly set up:

```python
import pandas as pd
from pathlib import Path

# Check if data exists
data_path = Path("../Data/User1/Day1/Data.csv")
labels_path = Path("../Data/User1/Day1/Labels.csv")

print(f"Data exists: {data_path.exists()}")
print(f"Labels exist: {labels_path.exists()}")

if data_path.exists():
    data = pd.read_csv(data_path, sep=";", header=None, nrows=5)
    print(f"\nData shape: {data.shape}")
    print(f"Expected: (5, 9) - 9 sensor features")
```

## 🛠️ Troubleshooting

### "Data file not found"

**Problem**: Training script can't find data files

**Solution**: Make sure data is in `../Data/` relative to the project folder

```bash
# Check structure
cd D:\Ptyxiakhh
dir Data\User1\Day1
```

### "Found 28 million activity classes"

**Problem**: Labels contain timestamps instead of activity IDs

**Solution**: Re-run preprocessing with correct column:

```bash
python scripts/quick_fix_labels.py
```

### "Memory error during loading"

**Problem**: Dataset too large for your RAM

**Solution**: The training scripts automatically use 30% of data. If still too large, reduce further in `scripts/train_baseline.py`:

```python
sample_size = int(len(X_all) * 0.2)  # Use 20% instead of 30%
```

## 📞 Need Help?

Open an issue on GitHub if you need help with data setup!
