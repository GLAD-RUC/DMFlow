# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#### motivated by this repo: https://github.com/microsoft/mattergen/blob/main/mattergen/scripts/csv_to_dataset.py

import argparse
import os
import torch

import sys

sys.path.append(".")

from dmflow.data.disorder_utils import preprocess_disorder


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV files to PyTorch cache format."
    )

    parser.add_argument(
        "--csv_folder",
        type=str,
        required=True,
        help="Path to the folder containing the csv files. All csv files in the folder will be processed (e.g., 'train.csv', 'val.csv', 'test.csv') and the resulting datasets will be placed under {cache_path/dataset_name/filename_without_extension}, e.g, /path/to/project/dataset/mp_20/train.",
    )

    parser.add_argument(
        "--cache_path",
        type=str,
        default="",
        help="Path to the cache folder where the processed datasets will be saved.",
    )

    parser.add_argument(
        "--niggli",
        default=True,
        type=bool,
        help="Whether to use Niggli reduction for the crystal structure.",
    )

    parser.add_argument(
        "--primitive",
        default=True,
        type=bool,
        help="Whether to use primitive cell for the crystal structure.",
    )

    parser.add_argument(
        "--graph-method",
        type=str,
        default="crystalnn",
        help="Method to convert crystal structure to graph.",
    )

    parser.add_argument(
        "--prop",
        type=str,
        default="ehull",
        help="Property to be extracted from the csv files. Default is 'energy_per_atom'.",
    )

    args = parser.parse_args()

    for file in os.listdir(f"{args.csv_folder}"):
        if file.endswith(".csv"):
            file_name = file.split(".")[0]
            if file_name not in ["train", "val", "test"]:
                print(
                    f"Skipping file {file} as it is not one of the expected names: train, val, test."
                )
                continue
            file_path = os.path.join(args.csv_folder, file)
            save_path = args.cache_path or os.path.join(
                args.csv_folder, f"{file_name}_ori.pt"
            )
            if os.path.exists(save_path):
                print(f"File {save_path} already exists. Skipping processing.")
                continue

            print(f"Processing {file_path}")
            processed_data = preprocess_disorder(
                input_file=file_path,
                niggli=args.niggli,
                primitive=args.primitive,
                graph_method=args.graph_method,
                prop_list=[args.prop],
                num_workers=args.num_workers if hasattr(args, "num_workers") else 30,
            )
            torch.save(
                processed_data,
                save_path,
            )


if __name__ == "__main__":
    main()
