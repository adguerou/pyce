import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyce.tools.lc_mapping import LandCoverMap

from pyce_scripts import lc_distrib


def get_stats_table(
    df_stats,
    col_name_country="Country",
    col_name_glacier="glacier",
    col_name_snow_and_ice="snow_and_ice",
    col_name_veget="veget",
    col_name_veget_and_water="vegetation_and_water",
    col_name_landcover="landcover",
    col_name_donuts_pieces="landcover",
    val_rocks=0,
    val_snow=4,
    val_water=5,
    slope_correction=True,
):
    # ==================
    # GET DONUTS NUMBERS
    # ==================
    df = df_stats.copy()

    # Define categories for donuts
    # ----------------------------
    # Snow and ice: glacier + snow on deiced area
    df[col_name_snow_and_ice] = df[col_name_glacier]
    df.loc[df[col_name_landcover] == val_snow, col_name_snow_and_ice] = True

    # Water and veget
    df[col_name_veget_and_water] = df[col_name_veget]
    df.loc[df[col_name_landcover] == val_water, col_name_veget_and_water] = True

    # Get surfaces
    # ------------
    surf_country = lc_distrib.get_lc_surface(
        df,
        groupby=[
            col_name_country,
            col_name_snow_and_ice,
            col_name_veget_and_water,
            col_name_landcover,
        ],
        add_total=True,
        slope_correction=slope_correction,
    )

    # Add Alps numbers
    # ----------------
    df_surf = pd.concat(
        [
            surf_country,
            pd.concat(
                [surf_country.groupby(level=[1, 2]).sum()],
                keys=["Alps"],
                names=[col_name_country],
            ),
        ]
    )

    # Get percentage
    # --------------
    df_surf["percent"] = (
        df_surf["Total_LC"].div(df_surf.groupby([col_name_country])["Total_LC"].sum())
        * 100
    )

    # Reformat table
    # --------------
    df_surf.reset_index(inplace=True)

    # Add column corresponding to final categories: snow&ice, rocks, water&veget
    df_surf.loc[
        df_surf[col_name_snow_and_ice] == True,
        col_name_donuts_pieces,
    ] = col_name_snow_and_ice
    df_surf.loc[
        (df_surf[col_name_snow_and_ice] == False)
        & (df_surf[col_name_veget_and_water] == False),
        col_name_donuts_pieces,
    ] = "rocks"
    df_surf.loc[
        (df_surf[col_name_snow_and_ice] == False)
        & (df_surf[col_name_veget_and_water] == True),
        col_name_donuts_pieces,
    ] = col_name_veget_and_water

    # Reset multi-index to Country and categories + selection columns
    donuts_table = df_surf.set_index([col_name_country, col_name_donuts_pieces])[
        ["Total_LC", "percent"]
    ].rename(columns={"Total_LC": "Surface"})

    # Add total surface by country
    donuts_table = pd.concat(
        [
            donuts_table,
            pd.concat(
                [donuts_table.groupby(level=0).sum()],
                keys=["Total"],
                names=[col_name_donuts_pieces],
            ).reorder_levels([col_name_country, col_name_donuts_pieces]),
        ]
    ).sort_index()

    # ==================
    # GET BAR NUMBERS
    # ==================
    # Get veget and water only with LC_XX values
    bar_table = (
        df_surf.loc[df_surf[col_name_donuts_pieces] == col_name_veget_and_water]
        .reset_index(drop=True)
        .drop(
            columns=[
                f"LC_{val_rocks}",
                f"LC_{val_snow}",
                col_name_snow_and_ice,
                col_name_veget_and_water,
                col_name_donuts_pieces,
                "Total_LC",
                "percent",
            ]
        )
    )

    # Reindex on Country + stack to transform column to lines
    bar_table = pd.DataFrame(
        bar_table.set_index([col_name_country]).stack(future_stack=True)
    ).rename(columns={0: "Surface"})

    # Add percent values
    bar_table["percent"] = (
        bar_table.div(bar_table.groupby([col_name_country]).sum()) * 100
    )

    return donuts_table, bar_table


def plot_donuts(
    df_donuts: pd.DataFrame,
    df_bars: pd.DataFrame,
    lcmap: LandCoverMap,
    col_veget="vegetation_and_water",
    col_rocks="rocks",
    col_snow_and_ice="snow_and_ice",
    col_g1850="Total",
    save_dir=None,
):
    # Reshape df so each country is a dataframe line
    # ==============================================
    df_donuts = df_donuts.unstack(level=1).fillna(0)["Surface"]
    df_bars = df_bars.unstack(level=1).fillna(0)["Surface"]

    # general parameters
    # ==================
    global_size = 0.7
    wedge_size = 0.2
    inner_ratio_factor = 1.2

    # ======
    # Plots
    # =====
    for area in range(df_donuts.shape[0]):
        # ==============
        # DONUTS
        # ==============
        fig, ax = plt.subplots(figsize=(5, 5))

        # ========
        # Get Data
        # ========
        df_donuts_area = df_donuts.iloc[area]

        # =====
        # INNER
        # =====
        inner_vals = [df_donuts_area[col_g1850]] * 2
        inner_colors = ["#74a9cf"] * 2

        # Define radius of inner pie
        # --------------------------
        inner_ratio = (
            df_donuts_area[col_g1850]
            / df_donuts.loc[df_donuts.index == "Alps", col_g1850].iloc[0]
        )
        if inner_ratio != 1:
            inner_ratio *= inner_ratio_factor

        inner_radius = (global_size - wedge_size * 1.2) * inner_ratio
        if inner_radius < 0.05:
            inner_radius = 0.02

        # add external black line
        # -----------------------
        circle = plt.Circle((0, 0), inner_radius, color="k", linewidth=5)
        ax.add_patch(circle)

        # Plot the pie
        # ------------
        wedges, labels = ax.pie(
            inner_vals,
            radius=inner_radius,
            colors=inner_colors,
            pctdistance=0.0,
            wedgeprops=dict(width=inner_radius),
            counterclock=False,
            startangle=0,
        )

        # Put labels to inner pie
        # -----------------------
        if df_donuts_area.name != "Alps":
            inner_label_pos = inner_radius + 0.1
        else:
            inner_label_pos = 0.1

        labels[0].update(
            dict(
                text=f"{df_donuts_area.name}",
                color="k",
                weight="bold",
                fontsize=20,
                fontstyle="normal",
                horizontalalignment="center",
                x=0,
                y=inner_label_pos,
            )
        )
        labels[1].update(
            dict(
                text=f"{inner_vals[0]:.0f} km²",
                color="k",
                fontsize=18,
                fontstyle="italic",
                horizontalalignment="center",
                x=0,
                y=-inner_label_pos,
            )
        )

        # =====
        # OUTER
        # =====
        outer_vals = [
            df_donuts_area[col_snow_and_ice],
            df_donuts_area[col_rocks],
            df_donuts_area[col_veget],
        ]

        outer_colors = [
            lcmap.get_color_of_code(code=4),
            lcmap.get_color_of_code(code=0),
            lcmap.get_color_of_code(code=3),
        ]

        # Redefine outer_vals for small values of snow and ice
        # ----------------------------------------------------
        shift_snow_and_ice = 0.0
        if outer_vals[0] / df_donuts_area[col_g1850] * 100 < 1:  # Below 1% of total
            shift_snow_and_ice = (
                df_donuts_area[col_g1850] / 100
            )  # 1% of total in surface
            outer_vals[0] += shift_snow_and_ice  # add 1% to snow and ice
            outer_vals[1] -= shift_snow_and_ice  # remove 1% to rocks

        # Shift start angle for no vegetation
        # -----------------------------------
        if outer_vals[2] == 0.0:
            startangle = -45
        else:
            startangle = 0

        # Plot pie
        # --------
        wedges, labels, autotexts = ax.pie(
            outer_vals,
            radius=global_size,
            colors=outer_colors,
            autopct="%1.1f%%",
            pctdistance=1.2,
            wedgeprops=dict(width=wedge_size, edgecolor="k", linewidth=0.5),
            counterclock=False,
            startangle=startangle,
        )

        # Update percentage
        # -----------------
        # Remove 0% labels
        for at in autotexts:
            if float(at.get_text()[:-1]) != 0.0:
                at.update(
                    {
                        "fontsize": 16,
                        "fontstyle": "italic",
                        "horizontalalignment": "center",
                        "verticalalignment": "center_baseline",
                    }
                )
            else:
                at.update({"text": ""})

        # Move percentage of vegetation
        autotexts[2].update(
            dict(
                x=autotexts[2].get_position()[0] + 0.1,
                y=autotexts[2].get_position()[1] - 0.05,
            )
        )

        # Put correct values for snow and ice
        # -----------------------------------
        if shift_snow_and_ice != 0:
            for at, sign in zip([autotexts[0], autotexts[1]], [-1, 1]):
                orig_surf = (
                    float(at.get_text()[:-1]) * df_donuts_area[col_g1850] / 100
                    + sign * shift_snow_and_ice
                )
                orig_percent = orig_surf / df_donuts_area[col_g1850] * 100
                at.set_text(f"{orig_percent:.1f}%")

        # =============
        # BAR PLOT
        # =============
        if df_donuts_area.name in df_bars.index:
            # Get data
            # --------
            df_bars_area = df_bars.loc[df_bars.index == df_donuts_area.name].iloc[0]
            df_bars_area = df_bars_area[
                [
                    "LC_6",
                    "LC_1",
                    "LC_2",
                    "LC_3",
                    "LC_5",
                ]
            ]

            # Define subplots
            # ---------------
            ax_bar = ax.inset_axes([0.0, 0.0, 1, 1], polar=True, zorder=10)
            ax_bar.set_ylim(-global_size - 0.5, 1)
            ax_bar.set_frame_on(False)
            ax_bar.xaxis.grid(False)
            ax_bar.yaxis.grid(False)
            ax_bar.set_xticks([])
            ax_bar.set_yticks([])

            # Set the coordinates limits
            x_lower_limit = 0 + wedge_size
            angle_offset = (
                (df_donuts_area[col_veget] / np.sum(outer_vals) * 100) / 50 * np.pi
            )

            df_bars_max = (df_bars.loc[df_bars.index == "CH"].iloc[0]).max()
            indexes = list(range(0, len(df_bars_area.index)))

            width = np.pi / 2.8 / len(df_bars_area.index)
            angles = [
                angle_offset + element * (width + np.pi / 180) for element in indexes
            ]
            heights = df_bars_area.values / df_bars_max
            colors = [
                lcmap.get_color_of_code(int(code[-1])) for code in df_bars_area.index
            ]

            # Draw bars
            bars = ax_bar.bar(
                x=angles,
                height=heights,
                width=width,
                bottom=x_lower_limit,
                align="edge",
                color=colors,
            )

        # Layout
        # ======
        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            plt.savefig(
                os.path.join(save_dir, f"donuts_{df_donuts_area.name}_v4.png"),
                dpi=200,
                transparent=False,
            )
