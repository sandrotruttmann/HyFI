### Pre-Processing of SECOS catalog

import pandas as pd

# Load catalog data
# file = "./data_examples/SECOS_20250305_HyFI_VS.csv"
file = "./data_examples/StLeonard/hypoDD_StLeonard.csv"

df = pd.read_csv(file, sep=",")

# Select rows where "PreMet" is in method list
methods =['LSQ', 'SVD']
df = df[df["PreMet"].isin(methods)]

# Set EX, EY and EZ to fixed value
df["EX"] = 1
df["EY"] = 1
df["EZ"] = 1

# Export df
# df.to_csv("./data_examples/SECOS_20250305_HyFI_VS_filtered.csv", sep=",", index=False)
df.to_csv("./data_examples/StLeonard/hypoDD_StLeonard_filtered.csv", sep=",", index=False)
