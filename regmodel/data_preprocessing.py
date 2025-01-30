# Author: Shaurya K, Rutgers NB

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

INPUT_FILE = "tox21.csv"
OUTPUT_FILE = "csv_bin/tox21_cleaned.csv" # may seem redundant, but did it to visulize how many null values existed

def main():
    df = pd.read_csv(INPUT_FILE)

    label_cols = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
        'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
        'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

    df[label_cols] = df[label_cols].replace({np.nan: -1})

    df.to_csv(OUTPUT_FILE, index=False) #just to see visually how many -1 are in the csv

    df2 = pd.read_csv(OUTPUT_FILE)
    train_df, temp_df = train_test_split(df2, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv("csv_bin/train.csv", index=False)
    val_df.to_csv("csv_bin/val.csv", index=False)
    test_df.to_csv("csv_bin/test.csv", index=False)


if __name__ == "__main__":
    main()
