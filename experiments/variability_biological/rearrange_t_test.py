### This script processes CSV files containing variability codes and generates heatmaps for CODE I and CODE S.
# It reads the CSV files, pivots the data to create matrices for CODE I and CODE S, and then generates heatmaps for each condition.
# The heatmaps are saved as PNG files, and the pivot matrices are saved as CSV files.


import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

# 1) pick up all csv files except our own outputs
csv_files = [
    fn for fn in glob.glob("results/*.csv")
    if not fn.endswith("_variability_codes_matrix.csv")
]

for csv_file in csv_files:
    name = os.path.splitext(csv_file)[0]
    print(f"\n→ Processing {csv_file} …")

    # --- READ & PIVOT --------------------------------------------
    df = pd.read_csv(csv_file)
    
    print("  • Read data from CSV")
    print(df.head())

    # build a “Condition” column
    df['Condition'] = df['Media'] + "_" + df['Time']

    # pivot CODE I
    pivot_I = df.pivot_table(
        index=['RTX','Feature bin','Peak Da'],
        columns='Condition',
        values='CODE I',
        aggfunc='first'
    )
    pivot_I.columns = [f"{c}_CODE_I" for c in pivot_I.columns]

    # pivot CODE S
    pivot_S = df.pivot_table(
        index=['RTX','Feature bin','Peak Da'],
        columns='Condition',
        values='CODE S',
        aggfunc='first'
    )
    pivot_S.columns = [f"{c}_CODE_S" for c in pivot_S.columns]

    # merge into one matrix
    result = pd.concat([pivot_I, pivot_S], axis=1).reset_index()
    result = result.sort_values(by=['RTX','Feature bin'])

    # save the matrix
    out_matrix = f"{name}_variability_codes_matrix.csv"
    result.to_csv(out_matrix, index=False)
    print(f"  • Saved pivot matrix to {out_matrix}")

    # --- PREP FOR HEATMAP ---------------------------------------
    # unique biomarker index
    result['Biomarker'] = (
        result['RTX'] + "_"
        + result['Feature bin'].astype(str) + "_"
        + result['Peak Da'].astype(str)
    )
    result = result.set_index('Biomarker')

    # split CODE_I vs CODE_S columns
    code_I_cols = [c for c in result.columns if c.endswith("CODE_I")]
    code_S_cols = [c for c in result.columns if c.endswith("CODE_S")]

    data_I = result[code_I_cols]
    data_S = result[code_S_cols]

    # legends
    legend_I = [
        mpatches.Patch(color='red',   label='1: more intensity than baseline'),
        mpatches.Patch(color='green', label='0: robust'),
        mpatches.Patch(color='orange',label='-1: less intensity than baseline')
    ]
    legend_S = [
        mpatches.Patch(color='red',   label='1: shifted to the left'),
        mpatches.Patch(color='green', label='0: robust'),
        mpatches.Patch(color='orange',label='-1: shifted to the right')
    ]

    # discrete colormap
    cmap = ListedColormap(["orange","green","red"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    # --- PLOT CODE I ---------------------------------------------
    plt.figure(figsize=(12, 30))
    sns.heatmap(
        data_I,
        annot=True,
        cmap=cmap,
        norm=norm,
        cbar_kws={'ticks':[-1,0,1]}
    )
    plt.title(f"Variability Pattern: CODE I ({name})")
    plt.ylabel("Biomarker (RTX_Feature bin_Peak Da)")
    plt.xlabel("Condition")
    plt.xticks(rotation=90)
    plt.legend(handles=legend_I, bbox_to_anchor=(1.05,1), loc='best')
    plt.tight_layout()

    out_I = f"{name}_heatmap_code_I.png"
    plt.savefig(out_I, dpi=300)
    plt.close()
    print(f"  • Saved CODE I heatmap to {out_I}")

    # --- PLOT CODE S ---------------------------------------------
    plt.figure(figsize=(12, 30))
    sns.heatmap(
        data_S,
        annot=True,
        cmap=cmap,
        norm=norm,
        cbar_kws={'ticks':[-1,0,1]}
    )
    plt.title(f"Variability Pattern: CODE S ({name})")
    plt.ylabel("Biomarker (RTX_Feature bin_Peak Da)")
    plt.xlabel("Condition")
    plt.xticks(rotation=90)
    plt.legend(handles=legend_S, bbox_to_anchor=(1.05,1), loc='best')
    plt.tight_layout()

    out_S = f"{name}_heatmap_code_S.png"
    plt.savefig(out_S, dpi=300)
    plt.close()
    print(f"  • Saved CODE S heatmap to {out_S}")
