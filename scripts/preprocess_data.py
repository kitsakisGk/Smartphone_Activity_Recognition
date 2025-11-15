"""Preprocess raw sensor data from SHL dataset."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger


def preprocess_user_day(user_id: int, day: int, phone_position: str = "Hand"):
    """
    Preprocess data for a specific user and day.

    Args:
        user_id: User ID (1, 2, or 3)
        day: Day number (1, 2, or 3)
        phone_position: Phone position ('Hand', 'Torso', 'Bag', 'Hips')
    """
    logger = get_logger(__name__)

    # Paths
    data_dir = Path("../Data")
    file_path = data_dir / f"User{user_id}" / f"Day{day}"

    logger.info(f"Processing User {user_id}, Day {day}, Position: {phone_position}")

    # Check if raw file exists
    raw_file = file_path / f"{phone_position}_Motion.txt"
    label_file = file_path / "Label.txt"

    if not raw_file.exists():
        logger.warning(f"Raw file not found: {raw_file}")
        return False

    if not label_file.exists():
        logger.warning(f"Label file not found: {label_file}")
        return False

    try:
        # Load raw data (whitespace delimited)
        logger.info(f"Loading {raw_file.name}...")
        data_df = pd.read_csv(raw_file, delim_whitespace=True, header=None, low_memory=False)

        logger.info(f"Loaded {len(data_df)} samples, {len(data_df.columns)} columns")

        # Load labels (only first column = activity label)
        logger.info("Loading labels...")
        labels_df = pd.read_csv(label_file, delim_whitespace=True, header=None, usecols=[0], low_memory=False)

        # Keep only sensor columns (first 9 columns: Accel, Gyro, Mag)
        # Drop timestamp and other metadata columns
        if len(data_df.columns) > 9:
            data_df = data_df.iloc[:, :9]

        logger.info(f"Using {len(data_df.columns)} sensor features")

        # Remove rows with NaN values
        before_len = len(data_df)
        data_df = data_df.dropna()
        labels_df = labels_df.iloc[:len(data_df)]
        after_len = len(data_df)

        if before_len != after_len:
            logger.info(f"Removed {before_len - after_len} rows with NaN values")

        # Normalize each column to [-1, 1]
        logger.info("Normalizing features to [-1, 1]...")
        scaler = MinMaxScaler(feature_range=(-1, 1))

        for col in data_df.columns:
            data_df[col] = scaler.fit_transform(data_df[[col]])

        # Ensure labels match data length
        if len(labels_df) > len(data_df):
            labels_df = labels_df.iloc[:len(data_df)]

        # Combine data and labels
        preprocessed_df = data_df.copy()
        preprocessed_df[9] = labels_df.values.flatten()  # Add labels as column 9

        # Save preprocessed data
        preprocessed_file = file_path / "Preprocessed_Data.csv"
        preprocessed_df.to_csv(preprocessed_file, index=False, sep=";", header=False)
        logger.info(f"Saved preprocessed data: {preprocessed_file}")

        # Save features separately (Data.csv)
        data_file = file_path / "Data.csv"
        data_df.to_csv(data_file, index=False, sep=";", header=False)
        logger.info(f"Saved features: {data_file}")

        # Save labels separately (Labels.csv)
        labels_file = file_path / "Labels.csv"
        labels_df.iloc[:len(data_df)].to_csv(labels_file, index=False, sep=";", header=False)
        logger.info(f"Saved labels: {labels_file}")

        logger.info(f"✅ Successfully preprocessed User {user_id}, Day {day}")
        return True

    except Exception as e:
        logger.error(f"❌ Error processing User {user_id}, Day {day}: {e}")
        return False


def main():
    """Preprocess all user data."""
    setup_logging()
    logger = get_logger(__name__)

    logger.info("=" * 70)
    logger.info("PREPROCESSING RAW SENSOR DATA")
    logger.info("=" * 70)

    phone_position = "Hand"  # You can change this to Torso, Bag, or Hips

    success_count = 0
    total_count = 0

    # Process all users and days
    for user_id in [1, 2, 3]:
        for day in [1, 2, 3]:
            total_count += 1
            if preprocess_user_day(user_id, day, phone_position):
                success_count += 1
            logger.info("")  # Empty line for readability

    logger.info("=" * 70)
    logger.info(f"PREPROCESSING COMPLETE: {success_count}/{total_count} successful")
    logger.info("=" * 70)

    if success_count < total_count:
        logger.warning(f"⚠️  {total_count - success_count} files failed to process")
    else:
        logger.info("✅ All files processed successfully!")
       

if __name__ == "__main__":
    main()
