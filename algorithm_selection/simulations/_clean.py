# System imports
import os

# Third-party imports
import numpy as np
import pandas as pd

# Constants
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Read in data to dataframe
file_path = f"{CUR_DIR}/runtime_data_original.csv"
df_original = pd.read_csv(file_path)
df = df_original.copy()

# Drop duplicate rows
duplicate_rows = df.duplicated(subset=["Oligo", "Sites"], keep="first")
df = df[~duplicate_rows]

# Fix the sites column
df["Sites"] = [s[5:-2] for s in df["Sites"]]
df["Sites"] = [s.replace("\"", "").replace(" ", "") for s in df["Sites"]]
df["Sites"] = [s.split(",") for s in df["Sites"]]

# Add time discrepancy column
df["Time_Discrepancy"] = df["CutFree_Time"] - df["CutFreeRL_Time"]

# Add correct algorithm column
conditions = [
    (df["CutFree_Time"] <= df["CutFreeRL_Time"]),
    (df["CutFree_Time"] > df["CutFreeRL_Time"])
]
values = [0, 1] # 0 = CutFree, 1 = CutFreeRL
df["Correct_Algorithm_Choice"] = np.select(conditions, values)

# Adjust correct algorithm based on degeneracy if it outside of the confidence 
# interval (i.e., ignore cutfreerl if the degeneracy loss is too significant, 
# typically caused by incomplete cutfreerl output)
df.loc[df["CutFree_Degeneracy"] == 0, "Correct_Algorithm_Choice"] = 1
df.loc[
    df["CutFreeRL_Degeneracy"] <= df["CutFree_Degeneracy"]
    - (df["CutFree_Degeneracy"] * 0.10),
    "Correct_Algorithm_Choice"
] = 0

# Count classifcations
class_counts = df["Correct_Algorithm_Choice"].value_counts()
print(class_counts)

df["Input"] = df["Sites"].apply(lambda x: ' '.join(x))
df["Oligo_Input"] = df["Oligo_Length"]
df["Target"] = df["Correct_Algorithm_Choice"]
df = df[["Input", "Oligo_Input", "Target"]]

print(df.head())

df.to_csv(f"{CUR_DIR}/runtime_data.csv", index=False)