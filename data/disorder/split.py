## split all.csv to train:val:test=8:1:1

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(input_file, train_file, val_file, test_file):
    df = pd.read_csv(input_file)
    train_df, test_val_df = train_test_split(df, train_size=0.8, random_state=42)
    val_df, test_df = train_test_split(test_val_df, train_size=0.5, random_state=42)
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(
        f"Data split completed: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples."
    )


if __name__ == "__main__":
    for dataset in ["sd", "pd"]:
        base_dir = f"data/disorder/cod_20/{dataset}"
        input_file = f"{base_dir}/all.csv"
        train_file = f"{base_dir}/train.csv"
        val_file = f"{base_dir}/val.csv"
        test_file = f"{base_dir}/test.csv"
        split_data(input_file, train_file, val_file, test_file)
