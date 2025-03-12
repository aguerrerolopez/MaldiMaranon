import pandas as pd

# Read the CSV file.
df = pd.read_csv("t_test_variability_results.csv")

# Create a combined condition column (e.g. "Medio Ch_Semana 1").
df['Condition'] = df['Media'] + "_" + df['Time']

# Pivot the table so that each unique combination of RTX, Feature bin, and Peak Da is a row.
# For CODE_I:
pivot_code_I = df.pivot_table(
    index=['RTX', 'Feature bin', 'Peak Da'],
    columns='Condition',
    values='CODE I',
    aggfunc='first'
)

# For CODE_S:
pivot_code_S = df.pivot_table(
    index=['RTX', 'Feature bin', 'Peak Da'],
    columns='Condition',
    values='CODE S',
    aggfunc='first'
)

# Rename pivoted columns to reflect the type.
pivot_code_I.columns = [f"{col}_CODE_I" for col in pivot_code_I.columns]
pivot_code_S.columns = [f"{col}_CODE_S" for col in pivot_code_S.columns]

# Merge the two pivoted DataFrames.
result = pd.concat([pivot_code_I, pivot_code_S], axis=1).reset_index()

# (Optional) Sort rows by RTX and Feature bin.
result = result.sort_values(by=['RTX', 'Feature bin'])

# Write the final wide-format CSV.
result.to_csv("variability_codes_matrix.csv", index=False)

print("Saved the pivoted variability codes matrix to variability_codes_matrix.csv")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Read the CSV.
df = pd.read_csv("variability_codes_matrix.csv")

# Create a unique index for each biomarker by combining RTX, Feature bin, and Peak Da.
df['Biomarker'] = df['RTX'] + "_" + df['Feature bin'].astype(str) + "_" + df['Peak Da'].astype(str)
df = df.set_index('Biomarker')

# Separate out the CODE_I and CODE_S columns.
code_I_cols = [col for col in df.columns if col.endswith("CODE_I")]
code_S_cols = [col for col in df.columns if col.endswith("CODE_S")]

data_code_I = df[code_I_cols]
data_code_S = df[code_S_cols]

# Create legend patches for CODE I.
legend_patches_code_I = [
    mpatches.Patch(color='red', label='1: more intensity than baseline'),
    mpatches.Patch(color='green', label='0: robust'),
    mpatches.Patch(color='orange', label='-1: less intensity than baseline')
]

# Create legend patches for CODE S.
legend_patches_code_S = [
    mpatches.Patch(color='red', label='1: shifted to the left'),
    mpatches.Patch(color='green', label='0: robust'),
    mpatches.Patch(color='orange', label='-1: shifted to the right')
]

# Define a discrete colormap for the code values (-1, 0, 1).
# Here we use a custom palette mapping -1 (blue), 0 (white), and 1 (red).
from matplotlib.colors import ListedColormap, BoundaryNorm

cmap = ListedColormap(["orange", "green", "red"])
norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

# Plot heatmap for CODE I.
plt.figure(figsize=(12, 30))
ax1 = sns.heatmap(data_code_I, annot=True, cmap=cmap, norm=norm, cbar_kws={'ticks': [-1, 0, 1]})
plt.title("Variability Pattern: CODE I")
plt.ylabel("Biomarker (RTX_Feature bin_Peak Da)")
plt.xlabel("Condition")
plt.xticks(rotation=90)
plt.legend(handles=legend_patches_code_I, bbox_to_anchor=(1.05, 1), loc='best')
plt.tight_layout()
plt.savefig("heatmap_code_I.png", dpi=300)
plt.show()

# Plot heatmap for CODE .
plt.figure(figsize=(12, 30))
ax2 = sns.heatmap(data_code_S, annot=True,  cmap=cmap, norm=norm, cbar_kws={'ticks': [-1, 0, 1]})
plt.title("Variability Pattern: CODE S")
plt.ylabel("Biomarker (RTX_Feature bin_Peak Da)")
plt.xlabel("Condition")
plt.xticks(rotation=90)
plt.legend(handles=legend_patches_code_S, bbox_to_anchor=(1.05, 1), loc='best')
plt.tight_layout()
plt.savefig("heatmap_code_S.png", dpi=300)
plt.show()

