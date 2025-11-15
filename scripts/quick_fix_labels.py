"""Quick script to fix Labels.csv files - reads only activity labels from Label.txt"""

import pandas as pd
from pathlib import Path

data_dir = Path("../Data")

for user in [1, 2, 3]:
    for day in [1, 2, 3]:
        path = data_dir / f"User{user}" / f"Day{day}"
        label_file = path / "Label.txt"

        print(f"Fixing User {user}, Day {day}...")

        # Read ONLY column 1 (activity labels 1-8, column 0 is timestamp)
        labels = pd.read_csv(label_file, delim_whitespace=True, header=None, usecols=[1])

        # Match length with Data.csv
        data_file = path / "Data.csv"
        if data_file.exists():
            data = pd.read_csv(data_file, sep=";", header=None, nrows=1)
            # Just save the labels
            output_file = path / "Labels.csv"
            labels.to_csv(output_file, index=False, sep=";", header=False)
            print(f"  ✅ Saved {len(labels)} labels to {output_file}")

print("\n✅ All Labels.csv files fixed!")
print("Now run: python scripts/train_baseline.py --model random_forest")
