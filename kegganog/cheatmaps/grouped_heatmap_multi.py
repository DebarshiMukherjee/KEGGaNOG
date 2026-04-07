import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .grouped_heatmap import function_groups


def generate_grouped_heatmap_multi(
    kegg_decoder_file, output_folder, dpi, color, figsize=None
):
    with tqdm(total=6, desc="Preparing heatmap data") as pbar:

        function_groups_lower = {
            group: {func.lower() for func in funcs}
            for group, funcs in function_groups.items()
        }

        kegg_decoder_file["Group"] = kegg_decoder_file["Function"].apply(
            lambda x: next(
                (
                    group
                    for group, funcs in function_groups_lower.items()
                    if x.lower() in funcs
                ),
                "Miscellaneous",
            )
        )
        pbar.update(1)

        kegg_decoder_file = kegg_decoder_file.sort_values(
            by=["Group", "Function"]
        ).reset_index(drop=True)
        pbar.update(1)

        kegg_decoder_file["Function"] = pd.Categorical(
            kegg_decoder_file["Function"],
            categories=kegg_decoder_file["Function"],
            ordered=True,
        )
        pbar.update(1)

        # Define the group ranges for each part
        part1_groups = [
            "Amino acid metabolism",
            "Arsenic reduction",
            "Bacterial secretion systems",
            "Biofilm formation",
            "Carbohydrate metabolism",
            "Photosynthesis",
        ]
        part2_groups = [
            "Carbon degradation",
            "Carbon fixation",
            "Cell mobility",
            "Genetic competence",
            "Hydrogen redox",
            "Metal transporters",
            "Methanogenesis",
            "Miscellaneous",
        ]
        part3_groups = [
            "Nitrogen metabolism",
            "Oxidative phosphorylation",
            "Sulfur metabolism",
            "Transporters",
            "Vitamin biosynthesis",
        ]

        # Split the dataframe into 3 parts based on the groupings
        part1 = kegg_decoder_file[
            kegg_decoder_file["Group"].isin(part1_groups)
        ].reset_index(drop=True)
        pbar.update(1)
        part2 = kegg_decoder_file[
            kegg_decoder_file["Group"].isin(part2_groups)
        ].reset_index(drop=True)
        pbar.update(1)
        part3 = kegg_decoder_file[
            kegg_decoder_file["Group"].isin(part3_groups)
        ].reset_index(drop=True)
        pbar.update(1)

    # Function to add empty rows between groups
    with tqdm(total=6, desc="Adding split between groups") as pbar:

        def add_empty_rows(df, groups):
            new_rows = []
            for group in groups:
                group_rows = df[df["Group"] == group]
                new_rows.append(group_rows)
                # Add an empty row if this is not the last group
                if group != groups[-1]:
                    # Create an empty row with 'split' in the 'Function' column
                    empty_row = pd.DataFrame(
                        [["split_" + f"{group}"] + [np.nan] * (df.shape[1] - 1)],
                        columns=df.columns,
                    )  # First column is 'Function'
                    # empty_row['Group'] = 'split'  # Set the group to 'split'
                    new_rows.append(empty_row)  # Append the empty row
            return pd.concat(new_rows, ignore_index=True)

        part1 = add_empty_rows(
            kegg_decoder_file[kegg_decoder_file["Group"].isin(part1_groups)],
            part1_groups,
        ).reset_index(drop=True)
        pbar.update(1)
        part2 = add_empty_rows(
            kegg_decoder_file[kegg_decoder_file["Group"].isin(part2_groups)],
            part2_groups,
        ).reset_index(drop=True)
        pbar.update(1)
        part3 = add_empty_rows(
            kegg_decoder_file[kegg_decoder_file["Group"].isin(part3_groups)],
            part3_groups,
        ).reset_index(drop=True)
        pbar.update(1)

        part1["Function"] = pd.Categorical(
            part1["Function"], categories=part1["Function"], ordered=True
        )
        pbar.update(1)
        part2["Function"] = pd.Categorical(
            part2["Function"], categories=part2["Function"], ordered=True
        )
        pbar.update(1)
        part3["Function"] = pd.Categorical(
            part3["Function"], categories=part3["Function"], ordered=True
        )
        pbar.update(1)

    # Dimensions and margins in inches, only width scales, height fixed
    n_cols = part1.shape[1] - 2
    cell_size = 0.25
    panel_w = max(n_cols * cell_size, 1.0)
    fig_h = 20.0
    margin_left = 0.3
    cbar_w = 0.1
    cbar_gap = 4.2
    margin_right = 0.3

    panel_fig_w = margin_left + panel_w + cbar_gap + cbar_w + margin_right
    panel_fig_w = max(panel_fig_w, 8)
    panel_figsize = (panel_fig_w, fig_h)

    # In interest of limiting image size and RAM usage, an upper limit of 200 megapixels is set. If user supplied dpi crosses that, then image is rescaled to a lower limit of 600 dpi
    #if resolution is below 200 MP, user supplied dpi is used
    max_megapixels = 200
    max_dpi = int((max_megapixels * 1_000_000 / (panel_fig_w * fig_h)) ** 0.5)
    if dpi > max_dpi:
        effective_dpi = max(600, max_dpi)
        print(f"Warning: requested DPI {dpi} would produce a "
              f"{panel_fig_w * dpi:.0f}x{fig_h * dpi:.0f} px image (>{max_megapixels}MP). "
              f"Reducing to {effective_dpi} DPI.")
        dpi = effective_dpi

    from matplotlib.gridspec import GridSpec

    def make_panel_fig():
        f = plt.figure(figsize=panel_figsize, constrained_layout=False)
        ax_left = margin_left / panel_fig_w
        ax_right = (margin_left + panel_w) / panel_fig_w
        ax_bottom = 0.15
        ax_top = 0.93
        ax = f.add_axes([ax_left, ax_bottom, ax_right - ax_left, ax_top - ax_bottom])
        cb_left = (margin_left + panel_w + cbar_gap) / panel_fig_w
        cb_w = cbar_w / panel_fig_w
        cb_bot = 0.40
        cb_top = 0.60
        cb_ax = f.add_axes([cb_left, cb_bot, cb_w, cb_top - cb_bot])
        return f, ax, cb_ax

    def add_group_labels(axes, part, group_labels):
        for i, group in enumerate(group_labels):
            group_indices = np.where(part["Group"] == group)[0]
            if len(group_indices) > 0:
                y_position = np.mean(group_indices) + 0.5
                x_position = -0.5
                axes.text(
                    x_position,
                    y_position,
                    group,
                    fontsize=12,
                    ha="right",
                    va="center",
                    weight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3", edgecolor="none", facecolor="white"
                    ),
                )

    def plot_heatmap(part, group_labels, ax, cbar, cbar_ax=None, cbar_kws=None):
        # Create the pivot table
        value_columns = part.columns[
            1:-1
        ]  # Adjust this based on your DataFrame structure

        # Fill NaN values in the selected columns
        part[value_columns] = part[value_columns].fillna(0)

        pivot_table = part.set_index("Function")[value_columns]  # Preserve column order

        # Create a mask for rows starting with 'split_'
        mask = pivot_table.index.str.startswith("split_")

        # Create the heatmap
        sns.heatmap(
            pivot_table,
            cmap=f"{color}",
            annot=False,
            linewidths=0.5,
            ax=ax,
            cbar=cbar,
            cbar_ax=cbar_ax,
            cbar_kws=cbar_kws,
            square=True,
            mask=np.tile(mask[:, None], (1, pivot_table.shape[1])),
        )
        ax.tick_params(axis="y", labelrotation=0)
        add_group_labels(ax, part, group_labels)

        # Hide ticks and labels for rows starting with 'split_'
        for tick, label in zip(ax.yaxis.get_major_ticks(), ax.get_yticklabels()):
            if label.get_text().startswith("split_"):
                label.set_visible(False)  # Hide the label
                tick.set_visible(False)  # Hide the tick completely
                tick.tick1line.set_visible(False)  # Hide major tick mark
                tick.tick2line.set_visible(False)  # Hide minor tick mark
    # panels are output to separate files
    panels = [
        (part1, part1_groups, "Part 1", "heatmap_figure_part1.png"),
        (part2, part2_groups, "Part 2", "heatmap_figure_part2.png"),
        (part3, part3_groups, "Part 3", "heatmap_figure_part3.png"),
    ]

    figs = []
    with tqdm(total=3, desc="Creating and saving heatmap parts") as pbar:
        for part, groups, title, fname in panels:
            fig, ax, cb_ax = make_panel_fig()

            plot_heatmap(
                part, groups, ax,
                cbar=True,
                cbar_ax=cb_ax,
                cbar_kws={"label": "Pathway completeness"},
            )
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=45)
            ax.set_xticklabels(ax.get_xticklabels(), ha="right")
            ax.set_ylabel("")
            ax.yaxis.tick_right()
            ax.set_yticklabels(
                ax.get_yticklabels(), rotation=0, va="center", ha="left"
            )

            out = os.path.join(output_folder, fname)
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            figs.append(out)
            pbar.update(1)

    return figs