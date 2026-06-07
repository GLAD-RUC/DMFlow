import pandas as pd
import numpy as np
import os

column_mapping = {
    "material_id": "material_id",
    "e_above_hull": "ehull",  ## for mp_20
    "cif": "cif",
}

order_dir = "data/mp_20"
disorder_type = "sd"
disorder_dir = f"data/disorder/cod_20/{disorder_type}"

target_dir = f"data/disorder/cod_20/mp_aug_{disorder_type}"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for subset in ["train", "val"]:
    order_df = pd.read_csv(f"{order_dir}/{subset}.csv")
    disorder_df = pd.read_csv(f"{disorder_dir}/{subset}.csv")

    order_df = order_df.rename(columns=column_mapping)
    order_df = order_df[list(column_mapping.values())]

    combined_df = pd.concat([order_df, disorder_df], axis=0)
    ## shuffle combined_df
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    combined_df.to_csv(f"{target_dir}/{subset}.csv", index=False)


os.system(f"cp {disorder_dir}/test.csv {target_dir}/test.csv")
