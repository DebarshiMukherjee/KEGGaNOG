import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


# Function to generate the heatmap
def generate_heatmap_multi(kegg_decoder_file, output_folder, dpi, color, figsize=None):
    print("Generating heatmap...")

    # Process data for heatmap with progress bar
    with tqdm(total=3, desc="Preparing heatmap data") as pbar:
        df = pd.DataFrame(kegg_decoder_file)
        pbar.update(1)

        # Split into three parts for separate heatmaps
        # df1, df2, df3 = np.array_split(df, 3) - DEPRECATED

        # Get the number of rows
        num_rows = len(df)

        # Calculate the split indices for 3 parts
        split_size = num_rows // 3

        # Split the dataframe manually
        df1 = df.iloc[:split_size]
        df2 = df.iloc[split_size : 2 * split_size]
        df3 = df.iloc[2 * split_size :]
        pbar.update(2)

    # Dimensions and margin in inches, width scales
    n_cols = df.shape[1] - 2
    cell_size = 0.25
    panel_w = max(n_cols * cell_size, 1.0)
    fig_h = 20.0
    margin_left = 0.3
    cbar_w = 0.1
    cbar_gap = 0.5
    margin_right = 0.3
    panel_fig_w = margin_left + panel_w + cbar_gap + cbar_w + margin_right
    panel_fig_w = max(panel_fig_w, 8)
    panel_figsize = (panel_fig_w, fig_h)

    # In interest of limiting image size and RAM usage, an upper limit of 200 megapixels is set. If user supplied dpi crosses that, then image is rescaled to a lower limit of 600 dpi
    #if resolution is below 200 MP, user supplied dpi is used
    max_dpi = int((200 * 1_000_000 / (panel_fig_w * fig_h)) ** 0.5)
    if dpi > max_dpi:
        effective_dpi = max(600, max_dpi)
        print(f"Warning: requested DPI {dpi} would exceed 200MP. "
              f"Reducing to {effective_dpi} DPI.")
        dpi = effective_dpi

    def make_panel_fig():
        f = plt.figure(figsize=panel_figsize, constrained_layout=False)
        ax_left = margin_left / panel_fig_w
        ax_right = (margin_left + panel_w) / panel_fig_w
        ax = f.add_axes([ax_left, 0.15, ax_right - ax_left, 0.78])
        cb_left = (margin_left + panel_w + cbar_gap) / panel_fig_w
        cb_ax = f.add_axes([cb_left, 0.40, cbar_w / panel_fig_w, 0.20])
        return f, ax, cb_ax
    # panels are output to separate files
    panels = [
        (df1, "Part 1", "heatmap_figure_part1.png"),
        (df2, "Part 2", "heatmap_figure_part2.png"),
        (df3, "Part 3", "heatmap_figure_part3.png"),
    ]

    figs = []
    with tqdm(total=3, desc="Creating and saving heatmap parts") as pbar:
        for df_part, title, fname in panels:
            fig, ax, cb_ax = make_panel_fig()
            sns.heatmap(
                df_part.set_index("Function"),
                cmap=f"{color}",
                annot=False,
                linewidths=0.5,
                ax=ax,
                cbar_ax=cb_ax,
                cbar_kws={"label": "Pathway completeness"},
            )
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=45)
            ax.set_xticklabels(ax.get_xticklabels(), ha="right")
            ax.set_ylabel("")
            out = os.path.join(output_folder, fname)
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            figs.append(out)
            pbar.update(1)

    return figs