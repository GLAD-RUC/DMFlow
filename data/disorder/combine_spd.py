import pandas as pd

for split_name in ["train", "val", "test", "all"]:
    print("Combining data for split:", split_name)

    sd_data = pd.read_csv(f"data/disorder/cod_20/sd/{split_name}.csv")
    pd_data = pd.read_csv(f"data/disorder/cod_20/pd/{split_name}.csv")

    spd_data = pd.concat([sd_data, pd_data], ignore_index=True)
    spd_data.to_csv(f"data/disorder/cod_20/spd/{split_name}.csv", index=False)
    print(
        f"Combined data saved to data/disorder/cod_20/spd/{split_name}.csv with {len(spd_data)} samples."
    )
