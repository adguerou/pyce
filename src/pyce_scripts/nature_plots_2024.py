import os

import numpy as np
import pandas as pd
import seaborn as sbn
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MultipleLocator
from pyce.tools import lc_mapping
from pyce.tools.lc_mapping import LandCoverMap

from pyce_scripts import lc_distrib


def get_stats_table(
    df_stats,
    round_dec: int = None,
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
        round_dec=round_dec,
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


def get_fig1_tables(
    df_stats,
    round_dec: int = None,
    col_name_country="Country",
    col_name_snow_and_ice="snow_and_ice",
    col_name_deglaciated="deglaciated",
    col_name_aquatic="aquatic",
    col_name_veget="veget",
    col_name_rocks="rocks",
    col_name_landcover="landcover",
    val_rocks=0,
    val_snow=4,
    val_water=5,
    slope_correction=False,
):
    def _get_surf_and_perc(
        df, groupby=[col_name_country, "category"], columns="category"
    ):
        surf = lc_distrib.get_lc_surface(
            df,
            groupby=groupby,
            index=col_name_country,
            columns=columns,
            round_dec=round_dec,
            add_total=True,
            slope_correction=slope_correction,
        )

        # Add alpine total
        surf.loc["ALPS"] = surf.sum()

        # Add percentage
        perc = surf.div(surf["Total_LC"], axis=0) * 100

        # Add metrics
        perc["metric"] = "percent"
        surf["metric"] = "surface [km2]"

        # Concat surface and percentage
        surf_and_perc = (
            pd.concat([surf, perc])
            .reset_index()
            .set_index([col_name_country, "metric"])
            .sort_index()
        )
        return surf_and_perc

    # Copy dataframe and create category column for which we derive stats
    df_lia = df_stats.copy()
    df_lia["category"] = None

    # 1. Statistics over LIA area
    # ===========================

    # Create categories
    # -----------------
    # Snow & Ice
    df_lia.loc[
        df_lia[col_name_landcover] == val_snow, "category"
    ] = col_name_snow_and_ice

    # Deglaciated
    df_lia.loc[
        (df_lia.glacier == False) & (df_lia.landcover != 4), "category"
    ] = col_name_deglaciated

    # Get surfaces and percentages
    # ----------------------------
    surfaces_lia = _get_surf_and_perc(df_lia)

    # 2. Statistics over DEGLACIATED area
    # ===================================
    df_deglaciated = df_lia.loc[(df_lia.glacier == False) & (df_lia.landcover != 4)]
    df_deglaciated.loc[:, "category"] = None

    # Create categories
    # -----------------
    # Rocks
    df_deglaciated.loc[
        df_deglaciated[col_name_landcover] == val_rocks, "category"
    ] = col_name_rocks

    # Veget
    df_deglaciated.loc[df_deglaciated.veget == True, "category"] = col_name_veget

    # Aquatic
    df_deglaciated.loc[
        df_deglaciated[col_name_landcover] == val_water, "category"
    ] = col_name_aquatic

    # Get surfaces and percentages
    # ----------------------------
    surfaces_deglaciated = _get_surf_and_perc(df_deglaciated)

    # 3. Statistics over VEGETATION area
    # ==================================
    df_veget = df_lia.loc[(df_lia.veget == True)]

    surfaces_veget = _get_surf_and_perc(
        df_veget,
        groupby=[col_name_country, col_name_landcover],
        columns=col_name_landcover,
    )

    return surfaces_lia, surfaces_deglaciated, surfaces_veget


def plot_donuts(
    df_donuts: pd.DataFrame,
    df_bars: pd.DataFrame,
    lcmap: LandCoverMap,
    col_veget="vegetation_and_water",
    col_rocks="rocks",
    col_snow_and_ice="snow_and_ice",
    col_g1850="Total",
    country_total="Alps",
    save_dir=None,
    save_name_ext=None,
):
    # Reshape df so each country is a dataframe line
    # ==============================================
    df_donuts = df_donuts.unstack(level=1).fillna(0)["Surface"]
    df_bars_perc = df_bars.unstack(level=1).fillna(0)["percent"]
    df_bars = df_bars.unstack(level=1).fillna(0)["Surface"]

    # general parameters
    # ==================
    ax_size = 1
    pie_size = 0.65
    wedge_size = 0.2
    margin_inner_pie = 0.04
    margin_pie_bar = 0.01
    pad_bar_label = 0.05
    ratio_bar_angle = 2.5

    # ======
    # Plots
    # =====
    for area in range(df_donuts.shape[0]):
        # ==============
        # DONUTS
        # ==============
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim([-ax_size, ax_size])
        ax.set_ylim([-ax_size, ax_size])

        # ========
        # Get Data
        # ========
        df_donuts_area = df_donuts.iloc[area]

        # =====
        # INNER
        # =====
        inner_ratio_factor = 1.1
        inner_radius_min_thresh = 0.01
        inner_radius_min = 0.005

        inner_vals = [df_donuts_area[col_g1850]] * 2
        inner_colors = [lcmap.get_color_of_code(code=4)] * 2

        # Define radius of inner pie
        # --------------------------
        inner_ratio = (
            df_donuts_area[col_g1850]
            / df_donuts.loc[df_donuts.index == country_total, col_g1850].iloc[0]
        )
        if inner_ratio != 1:
            inner_ratio *= inner_ratio_factor

        inner_radius = (pie_size - wedge_size) * inner_ratio - margin_inner_pie
        if inner_radius < inner_radius_min_thresh:
            inner_radius = inner_radius_min

        # add external black line
        # -----------------------
        circle = plt.Circle((0, 0), inner_radius, color="k", linewidth=3)
        ax.add_patch(circle)

        # Plot the pie
        # ------------
        wedges, labels = ax.pie(
            inner_vals,
            radius=inner_radius,
            colors=inner_colors,
            pctdistance=0.0,
            wedgeprops=dict(width=inner_radius, color=inner_colors[0], linewidth=1),
            counterclock=False,
            startangle=0,
        )

        # Put labels to inner pie
        # -----------------------
        if df_donuts_area.name != country_total:
            inner_label_pos = inner_radius + 0.1
        else:
            inner_label_pos = 0.1

        labels[0].update(
            dict(
                text=f"{df_donuts_area.name}",
                color="k",
                weight="bold",
                fontsize=18,
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
                fontsize=14,
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
            lcmap.get_color_of_code(code=9),
        ]

        # Shift start angle for no vegetation
        # -----------------------------------
        if outer_vals[2] == 0.0:
            startangle = -60
        else:
            startangle = 0

        if outer_vals[0] == 0 and outer_vals[-1] == 0:
            outer_vals = [outer_vals[1]]
            outer_colors = [outer_colors[1]]

        # Redefine outer_vals for small values of snow and ice
        # ----------------------------------------------------
        # shift_snow_and_ice = 0.0
        # if outer_vals[0] / df_donuts_area[col_g1850] * 100 < 1:  # Below 1% of total
        #     shift_snow_and_ice = (
        #         df_donuts_area[col_g1850] / 100
        #     )  # 1% of total in surface
        #     outer_vals[0] += shift_snow_and_ice  # add 1% to snow and ice
        #     outer_vals[1] -= shift_snow_and_ice  # remove 1% to rocks

        # Plot pie
        # --------
        wedges, labels, autotexts = ax.pie(
            outer_vals,
            radius=pie_size,
            colors=outer_colors,
            autopct=lambda per: "{:.1f}%".format(per),
            pctdistance=0.82,
            labels=[f"{surf:.0f}" for surf in outer_vals],
            labeldistance=1.2,
            wedgeprops=dict(width=wedge_size, edgecolor="k", linewidth=1),
            counterclock=False,
            startangle=startangle,
        )

        # Update surface / percentage
        # --------------
        # Remove 0 km
        for at, lbl in zip(autotexts, labels):
            if float(at.get_text()[:-1]) != 0.0:
                at.update(
                    {
                        "fontsize": 11,
                        "fontstyle": "italic",
                        "horizontalalignment": "center",
                        "verticalalignment": "center",
                    }
                )
                lbl.update(
                    {
                        "fontsize": 14,
                        "color": "k",
                        "horizontalalignment": "center",
                        "verticalalignment": "center",
                    }
                )
            else:
                at.update({"text": ""})
                lbl.update({"text": ""})

        # Move vegetation
        if len(labels) > 1:
            labels[2].update(
                dict(
                    x=labels[2].get_position()[0] + 0.15,
                    y=labels[2].get_position()[1] - 0.05,
                )
            )
            autotexts[2].update(
                dict(
                    text=f"({autotexts[2].get_text()})",
                    x=labels[2].get_position()[0] - 0.01,
                    y=labels[2].get_position()[1] - 0.12,
                    color="k",
                )
            )
        # Move percent of rocks for swiss
        if df_donuts_area.name == "CH":
            labels[1].update(
                dict(
                    x=labels[1].get_position()[0] - 0.2,
                    y=labels[1].get_position()[1] - 0.08,
                )
            )

        # Move surface of snow for Austria
        if df_donuts_area.name == "AT":
            labels[0].update(
                dict(
                    x=labels[0].get_position()[0] + 0.3,
                    y=labels[0].get_position()[1] + 0.18,
                )
            )

        # Move surface of rocks Himalaya
        print(df_donuts_area.name)
        if df_donuts_area.name == "Himalaya":
            print(df_donuts_area.name)
            labels[1].update(
                dict(
                    x=labels[1].get_position()[0] - 0.65,
                    y=labels[1].get_position()[1] + 0.25,
                )
            )

        # Put correct values for snow and ice
        # -----------------------------------
        # if shift_snow_and_ice != 0:
        #     for at, sign in zip([autotexts[0], autotexts[1]], [-1, 1]):
        #         orig_surf = (
        #             float(at.get_text()[:-1]) * df_donuts_area[col_g1850] / 100
        #             + sign * shift_snow_and_ice
        #         )
        #         orig_percent = orig_surf / df_donuts_area[col_g1850] * 100
        #         at.set_text(f"{orig_percent:.1f}%")

        # =============
        # BAR PLOT
        # =============

        # Define subplots
        # ---------------
        ax_bar = ax.inset_axes([0, 0, 1, 1], polar=True, zorder=10)
        ax_bar.set_ylim([0, ax_size])
        ax_bar.set_frame_on(False)
        ax_bar.xaxis.grid(False)
        ax_bar.yaxis.grid(False)
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])

        # Functions
        # ---------
        def get_perc_line(
            val,
            ymin=None,
            ymax=None,
            per_max=None,
        ):
            return ymin + val * ymax / per_max

        def plot_perc_lines(
            vals,
            angles=None,
            ymin=None,
            ymax=None,
            per_max=None,
            color="k",
            lw=0.6,
            ls="--",
            zorder=0,
            ax=ax_bar,
        ):
            for val in vals:
                ax.plot(
                    angles,
                    [
                        get_perc_line(
                            val,
                            ymin=ymin,
                            ymax=ymax,
                            per_max=per_max,
                        )
                    ]
                    * angles.shape[0],
                    color=color,
                    lw=lw,
                    ls=ls,
                    zorder=zorder,
                )

        def plot_perc_label(
            vals, lbls, angle=None, ymin=None, ymax=None, per_max=None, ax=ax_bar
        ):
            for val, lbl in zip(vals, lbls):
                ax.text(
                    x=angle,
                    y=get_perc_line(
                        val,
                        ymin=ymin,
                        ymax=ymax,
                        per_max=per_max,
                    ),
                    s=lbl,
                    ha="right",
                    va="center",
                    fontsize=8,
                    color="k",
                    rotation=np.rad2deg(angle) - 90,
                    rotation_mode="anchor",
                )

        # Start plot
        # ==========
        if df_donuts_area.name in df_bars.index:
            # Get data
            # --------
            df_bars_area = df_bars.loc[df_bars.index == df_donuts_area.name].iloc[0]
            df_bars_area_percent = df_bars_perc.loc[
                df_bars_perc.index == df_donuts_area.name
            ].iloc[0]

            df_bars_area = df_bars_area[
                [
                    "LC_6",
                    "LC_1",
                    "LC_2",
                    "LC_3",
                    "LC_5",
                ]
            ]

            df_bars_area_percent = df_bars_area_percent[
                [
                    "LC_6",
                    "LC_1",
                    "LC_2",
                    "LC_3",
                    "LC_5",
                ]
            ]

            colors = [
                lcmap.get_color_of_code(int(code[-1])) for code in df_bars_area.index
            ]

            # Define the angles
            # -----------------
            indexes = list(range(0, len(df_bars_area.index)))
            width = np.pi / len(df_bars_area.index) / ratio_bar_angle

            angle_offset = (
                (df_donuts_area[col_veget] / np.sum(outer_vals) * 100) / 50 * np.pi
            )
            angles = [angle_offset + element * (width) for element in indexes]

            # Scale and offset size of each bar
            # ---------------------------------
            y_lower_limit = pie_size - margin_pie_bar  # bottom of bars
            if df_bars_area.name != country_total:
                df_bars_max = (df_bars.loc[df_bars.index == "CH"].iloc[0]).max()
            else:
                df_bars_max = (
                    df_bars.loc[df_bars.index == country_total].iloc[0]
                ).max()

            heights = df_bars_area.values * (1 - y_lower_limit) / df_bars_max

            # Add lines percentage
            # --------------------
            per_lines = np.linspace(angles[0], angles[-1] + width, num=50)
            df_bars_area_percent_max = df_bars_area_percent.max()

            # Plot percentage lines
            # ---------------------
            if df_bars_area.name in ["FR", "IT"]:
                plot_perc_lines(
                    vals=[20, 60],
                    angles=per_lines,
                    ymin=y_lower_limit,
                    ymax=heights.max(),
                    per_max=df_bars_area_percent_max,
                )
                plot_perc_label(
                    vals=[20, 60],
                    lbls=["20  ", "60%  "],
                    angle=angles[-1] + width,
                    ymin=y_lower_limit,
                    ymax=heights.max(),
                    per_max=df_bars_area_percent_max,
                )
            else:
                plot_perc_lines(
                    vals=[5, 20, 60],
                    angles=per_lines,
                    ymin=y_lower_limit,
                    ymax=heights.max(),
                    per_max=df_bars_area_percent_max,
                )
                plot_perc_label(
                    vals=[5, 20, 60],
                    lbls=["5  ", "20  ", "60%  "],
                    angle=angles[-1] + width,
                    ymin=y_lower_limit,
                    ymax=heights.max(),
                    per_max=df_bars_area_percent_max,
                )

            # Draw bars
            # ---------
            bars = ax_bar.bar(
                x=angles,
                height=heights,
                width=width,
                bottom=y_lower_limit,
                align="edge",
                color=colors,
            )

            # Add labels
            # ----------
            for angle, height, label, index in zip(
                angles, heights, df_bars_area.values, indexes
            ):
                if index == 0:
                    the_lbl = f"{label:.1f}\nkm²"
                else:
                    the_lbl = f"{label:.1f}"
                ax_bar.text(
                    x=angle + width / 2.0,
                    y=heights.max() + y_lower_limit + pad_bar_label,
                    s=the_lbl,
                    ha="left",
                    va="center",
                    fontsize=12,
                    rotation=np.rad2deg(angle + width / 2.0),
                    rotation_mode="anchor",
                )

            # Add connection line
            # -------------------
            bottom_ls = "-"
            bottom_lw = 1.5
            bottom_color = "k"
            bottom_line = np.linspace(
                angles[0] - angle_offset / 2, angles[-1] + width, num=50
            )

            # Bottom arc
            plot_perc_lines(
                vals=[0],
                angles=bottom_line,
                ymin=y_lower_limit - margin_pie_bar,  # pie_size * 0.92,
                ymax=heights.max(),
                per_max=df_bars_area_percent_max,
                color=bottom_color,
                lw=bottom_lw,
                ls=bottom_ls,
            )

            # Straigth line
            y_bottom_line = get_perc_line(
                val=0,
                ymin=y_lower_limit - margin_pie_bar,  # pie_size * 0.92,
                ymax=heights.max(),
                per_max=df_bars_area_percent_max,
            )
            ax_bar.annotate(
                "",
                xy=(
                    angles[0] - angle_offset / 2,
                    pie_size - wedge_size / 1.5,
                ),
                xytext=(
                    angles[0] - angle_offset / 2,
                    y_bottom_line,
                ),
                xycoords="data",
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-|>",
                    ls=bottom_ls,
                    lw=bottom_lw,
                    color=bottom_color,
                    patchB=None,
                    shrinkA=0,
                    shrinkB=0,
                    connectionstyle="arc",
                ),
            )

        # Connection line vegetation for no vegetation
        # --------------------------------------------
        else:
            # Add connection line
            # -------------------
            bottom_ls = "-"
            bottom_lw = 1.5
            bottom_color = "k"
            bottom_line = np.linspace(0, np.pi / ratio_bar_angle, num=50)

            # Bottom arc
            plot_perc_lines(
                vals=[0],
                angles=bottom_line,
                ymin=y_lower_limit - margin_pie_bar,
                ymax=1,
                per_max=60,
                color=bottom_color,
                lw=bottom_lw,
                ls=bottom_ls,
            )

            # Straigth line
            y_bottom_line = get_perc_line(
                0,
                ymin=y_lower_limit - margin_pie_bar,
                ymax=heights.max(),
                per_max=df_bars_area_percent_max,
            )
            ax_bar.annotate(
                "",
                xy=(
                    bottom_line[0],
                    pie_size - wedge_size / 1.5,
                ),
                xytext=(
                    bottom_line[0],
                    y_bottom_line,
                ),
                xycoords="data",
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-|>",
                    ls=bottom_ls,
                    lw=bottom_lw,
                    color=bottom_color,
                    patchB=None,
                    shrinkA=0,
                    shrinkB=0,
                    connectionstyle="arc",
                ),
            )

            # Text
            # ----
            ax_bar.text(
                np.pi / ratio_bar_angle / 2.5,
                pie_size + 0.35,
                "No vegetation\nNo water",
                style="italic",
                fontsize=11,
                ha="right",
            )

        # Layout
        # ======
        # plt.tight_layout()
        plt.show()

        if save_dir is not None:
            plt.savefig(
                os.path.join(
                    save_dir, f"donuts_{df_donuts_area.name}_{save_name_ext}.png"
                ),
                dpi=200,
                transparent=True,
            )


def plot_donuts_v2(
    df_lia: pd.DataFrame,
    df_deglaciated: pd.DataFrame,
    df_veget: pd.DataFrame,
    lcmap: LandCoverMap,
    index_surface: str = "surface [km2]",
    index_percent: str = "percent",
    col_snow_and_ice="snow_and_ice",
    col_deglaciated="deglaciated",
    col_aquatic="aquatic",
    col_veget="veget",
    col_rocks="rocks",
    col_country_total="Total_LC",
    row_total_alps="ALPS",
    save_dir=None,
    save_name_ext=None,
):
    # Reshape df so each country is a dataframe line
    # ==============================================

    # LIA with snow_and_ice / deglaciated
    # -----------------------------------
    df_lia = df_lia.loc[df_lia.index.get_level_values(1) == index_surface].droplevel(1)

    # Deglaciated with aquatic/rocks/veget
    # -------------------------------------
    df_deglaciated_surf = df_deglaciated.loc[
        df_deglaciated.index.get_level_values(1) == index_surface
    ].droplevel(1)
    df_deglaciated_perc = df_deglaciated.loc[
        df_deglaciated.index.get_level_values(1) == index_percent
    ].droplevel(1)

    # Veget with each landcover type
    # ------------------------------
    df_veget_surf = df_veget.loc[
        df_veget.index.get_level_values(1) == index_surface
    ].droplevel(1)
    df_veget_percent = df_veget.loc[
        df_veget.index.get_level_values(1) == index_percent
    ].droplevel(1)

    # general parameters
    # ==================
    # First donuts
    ax_size = 1
    pie_size = 0.7 - 0.05
    wedge_size = 0.2
    margin_inner_pie = 0.05

    # Second donuts
    pie_size_ext = 0.94 - 0.05
    wedge_size_ext = 0.2

    # ================================
    #               Plots
    # =================================
    for area in range(df_lia.shape[0]):
        # ======
        # Figure
        # ======
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.set_aspect("equal")
        ax.set_xlim([-ax_size, ax_size])
        ax.set_ylim([-ax_size, ax_size])

        # ==========================
        # First pie and first donut
        # ========================
        df_lia_area = df_lia.iloc[area]

        # =====
        # INNER
        # =====
        inner_vals = [df_lia_area[col_country_total]] * 2
        inner_colors = ["#85bfde"] * 2

        # Define radius of inner pie
        # --------------------------
        max_radius = (pie_size - wedge_size) - margin_inner_pie
        max_surf = df_lia.loc[df_lia.index == row_total_alps, col_country_total].iloc[0]
        surf_area = df_lia_area[col_country_total]
        delta_surf_max = max_surf - 1  # 1 km2 is SL
        delta_radius_max = max_radius - 0.05  # must be <=max_radius

        inner_radius = (
            max_radius - (max_surf - surf_area) / delta_surf_max * delta_radius_max
        )

        # add external black line
        # -----------------------
        circle = plt.Circle((0, 0), inner_radius, color="k", linewidth=4)
        ax.add_patch(circle)

        # =============
        # Plot the pie
        # =============
        wedges, labels = ax.pie(
            inner_vals,
            radius=inner_radius,
            colors=inner_colors,
            pctdistance=0.0,
            wedgeprops=dict(width=inner_radius, color=inner_colors[0], linewidth=1),
            counterclock=False,
            startangle=0,
        )

        # Put labels to inner pie
        # -----------------------
        if df_lia_area.name != row_total_alps:
            inner_label_pos = inner_radius + 0.1
        else:
            inner_label_pos = 0.1

        labels[0].update(
            dict(
                text=f"{df_lia_area.name}",
                color="k",
                weight="bold",
                fontsize=18,
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
                fontsize=14,
                horizontalalignment="center",
                x=0,
                y=-inner_label_pos,
            )
        )

        # ===============
        #   FIRST DONUTS
        # ================
        outer_vals = [
            df_lia_area[col_snow_and_ice],  # snow
            df_lia_area[col_deglaciated],  # deglaciated
        ]

        outer_colors = [
            lcmap.get_color_of_code(code=4),  # snow
            lcmap.get_color_of_code(code=9),  # deglaciated
        ]

        # Shift start angle for no water/vegetation
        # -----------------------------------------
        if outer_vals[0] == 0.0:
            startangle = -60
        else:
            startangle = 0

        # Plot pie
        # --------
        wedges, labels, autotexts = ax.pie(
            outer_vals,
            radius=pie_size,
            colors=outer_colors,
            autopct=lambda per: "({:.1f}%)".format(per),
            pctdistance=0.8,
            labels=[f"{surf:.0f}" for surf in outer_vals],
            labeldistance=0.82,
            wedgeprops=dict(width=wedge_size, edgecolor="k", linewidth=1),
            counterclock=True,
            startangle=startangle,
        )

        # Update surface / percentage
        # ---------------------------
        # Remove 0 km
        for at, lbl in zip(autotexts, labels):
            if float(at.get_text()[1:-2]) != 0.0:
                at.update(
                    {
                        "fontsize": 10,
                        "fontstyle": "italic",
                        "horizontalalignment": "center",
                        "verticalalignment": "center",
                    }
                )
                lbl.update(
                    {
                        "fontsize": 13,
                        "color": "k",
                        "horizontalalignment": "center",
                        "verticalalignment": "center",
                    }
                )
            else:
                at.update({"text": ""})
                lbl.update({"text": ""})

        # FUNCTION
        # ---------------------------------------------------------------------------
        def update_lbl_pct(wedge, lbl_pct, delta_ang, delta_dist, rot=0):
            lbl_pct_ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
            lbl_pct_dist = lbl_pct.get_position()[1] / np.sin(np.deg2rad(lbl_pct_ang))

            update_ang = lbl_pct_ang + delta_ang
            x = np.cos(np.deg2rad(update_ang)) * (lbl_pct_dist + delta_dist)
            y = np.sin(np.deg2rad(update_ang)) * (lbl_pct_dist + delta_dist)

            lbl_pct.update(dict(x=x, y=y, rotation=rot))

        # ---------------------------------------------------------------------------

        # Move snow&ice
        update_lbl_pct(wedges[0], labels[0], delta_ang=10, delta_dist=-0.02)
        update_lbl_pct(wedges[0], autotexts[0], delta_ang=-40, delta_dist=0, rot=-45)

        # Move deglaciated
        if df_lia_area.name in df_veget_surf.index:
            update_lbl_pct(wedges[1], labels[1], delta_ang=0, delta_dist=0.0)
            update_lbl_pct(wedges[1], autotexts[1], delta_ang=55, delta_dist=0, rot=40)
        else:
            update_lbl_pct(
                wedges[1], autotexts[1], delta_ang=0, delta_dist=0.25, rot=30
            )

        # ============================================================================
        #                          Deglaciated plot
        # =============================================================================
        df_deglaciated_area = df_deglaciated_surf.iloc[area]

        outer_vals = [
            df_lia_area[col_snow_and_ice],  # snow
            df_deglaciated_area[col_aquatic],  # water
            df_deglaciated_area[col_rocks],  # rocks
            df_deglaciated_area[col_veget],  # veget
        ]

        outer_colors = [
            "#ffffff00",  # snow / transparent
            lcmap.get_color_of_code(code=5),  # rocks
            lcmap.get_color_of_code(code=0),  # rocks
            lcmap.get_color_of_code(code=8),  # vegetation
        ]

        # Shift start angle for no water/vegetation
        # -----------------------------------------
        index_water = 1
        index_rocks = 2

        if outer_vals[0] == 0.0:
            startangle = -60
        else:
            startangle = 0

        # Redefine outer_vals for small values of water
        # ----------------------------------------------------
        shift_water = 0
        if (outer_vals[index_water] / df_lia_area[col_country_total] * 100 < 1) & (
            outer_vals[index_water] != 0
        ):  # Below 1% of total
            shift_water = df_lia_area[col_country_total] / 100  # 1% of total in surface
            outer_vals[index_water] += shift_water  # add 1% to water
            outer_vals[index_rocks] -= shift_water  # remove 1% to rocks

        if outer_vals[index_rocks] / df_lia_area[col_country_total] == 1:
            outer_vals = [outer_vals[index_rocks]]
            outer_colors = [outer_colors[index_rocks]]

        # Plot pie
        # --------
        wedges, labels, autotexts = ax.pie(
            outer_vals,
            radius=pie_size_ext,
            colors=outer_colors,
            autopct="(%.1f%%)",
            pctdistance=0.86,
            labels=[f"{surf:.0f}" for surf in outer_vals],
            labeldistance=0.88,
            wedgeprops=dict(width=wedge_size_ext, edgecolor="k", linewidth=1),
            counterclock=True,
            startangle=startangle,
        )

        # Remove edgeline of snow
        # =======================
        if df_lia_area.name in df_veget_surf.index:
            wedges[0].update({"edgecolor": "#ffffff00"})

        # Put correct values for snow and ice
        # -----------------------------------
        if shift_water != 0:
            for at, sign in zip(
                [autotexts[index_water], autotexts[index_rocks]], [-1, 1]
            ):
                orig_surf = (
                    float(at.get_text()[1:-2]) * df_lia_area[col_country_total] / 100
                    + sign * shift_water
                )
                orig_percent = orig_surf / df_lia_area[col_country_total] * 100
                at.set_text(f"({orig_percent:.1f}%)")

        # Remove values for snow & ice
        autotexts[0].update({"text": ""})
        labels[0].update({"text": ""})

        # Update surface / percentage
        # ---------------------------
        for at, lbl in zip(autotexts, labels):
            at.update(
                {
                    "fontsize": 10,
                    "fontstyle": "italic",
                    "horizontalalignment": "center",
                    "verticalalignment": "center",
                }
            )
            lbl.update(
                {
                    "fontsize": 13,
                    "color": "k",
                    "horizontalalignment": "center",
                    "verticalalignment": "center",
                }
            )

        if df_lia_area.name in df_veget_surf.index:
            # Move water
            update_lbl_pct(wedges[1], labels[1], delta_ang=0, delta_dist=0.3)
            update_lbl_pct(wedges[1], autotexts[1], delta_ang=8, delta_dist=0.3, rot=0)

            # Move rocks
            update_lbl_pct(wedges[2], labels[2], delta_ang=0, delta_dist=0.0)
            update_lbl_pct(wedges[2], autotexts[2], delta_ang=45, delta_dist=0, rot=35)

            # Move veget
            update_lbl_pct(wedges[3], labels[3], delta_ang=0, delta_dist=0.25)
            update_lbl_pct(wedges[3], autotexts[3], delta_ang=-8, delta_dist=0.3, rot=0)

        # ========================================================
        #                       BAR PLOT
        # ========================================================

        # Define subplots
        # ---------------
        ax_bar = ax.inset_axes([0, 0, 1, 1], polar=True, zorder=10)
        ax_bar.set_ylim([0, ax_size])
        ax_bar.set_frame_on(False)
        ax_bar.xaxis.grid(False)
        ax_bar.yaxis.grid(False)
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])

        # Functions
        # ---------
        def get_perc_line(
            val,
            ymin=None,
            ymax=None,
            per_max=None,
        ):
            return ymin + val * ymax / per_max

        def plot_perc_lines(
            vals,
            angles=None,
            ymin=None,
            ymax=None,
            per_max=None,
            color="k",
            lw=0.6,
            ls="--",
            zorder=0,
            ax=ax_bar,
        ):
            for val in vals:
                ax.plot(
                    angles,
                    [
                        get_perc_line(
                            val,
                            ymin=ymin,
                            ymax=ymax,
                            per_max=per_max,
                        )
                    ]
                    * angles.shape[0],
                    color=color,
                    lw=lw,
                    ls=ls,
                    zorder=zorder,
                )

        def plot_perc_label(
            vals, lbls, angle=None, ymin=None, ymax=None, per_max=None, ax=ax_bar
        ):
            for val, lbl in zip(vals, lbls):
                ax.text(
                    x=angle,
                    y=get_perc_line(
                        val,
                        ymin=ymin,
                        ymax=ymax,
                        per_max=per_max,
                    ),
                    s=lbl,
                    ha="right",
                    va="center",
                    fontsize=8,
                    color="k",
                    rotation=np.rad2deg(angle) - 90,
                    rotation_mode="anchor",
                )

        # ==========
        # Parameters
        # ==========
        delta_ax_size_donuts = -0.05
        ax_size_bar = 1.15
        pad_bar_label = 0.05
        bar_angle_start = np.pi / 10
        bar_angle_span = np.pi / 2 - bar_angle_start

        # Start plot
        # ==========
        if df_lia_area.name in df_veget_surf.index:
            # Get data
            # --------
            df_bars_area = df_veget_surf.loc[
                df_veget_surf.index == df_lia_area.name
            ].iloc[0]
            df_bars_area_percent = df_veget_percent.loc[
                df_veget_percent.index == df_lia_area.name
            ].iloc[0]

            df_bars_area = df_bars_area[
                [
                    "LC_6",
                    "LC_1",
                    "LC_2",
                    "LC_3",
                ]
            ]

            df_bars_area_percent = df_bars_area_percent[
                [
                    "LC_6",
                    "LC_1",
                    "LC_2",
                    "LC_3",
                ]
            ]

            colors = [
                lcmap.get_color_of_code(int(code[-1])) for code in df_bars_area.index
            ]

            # Define the angles
            # -----------------
            indexes = list(range(0, len(df_bars_area.index)))
            width = bar_angle_span / len(df_bars_area.index)
            angles = [bar_angle_start + element * (width) for element in indexes]

            # Scale and offset size of each bar
            # ---------------------------------
            y_lower_limit = pie_size - delta_ax_size_donuts  # bottom of bars
            if df_bars_area.name != row_total_alps:
                df_bars_max = (
                    df_veget_surf.loc[df_veget_surf.index == "CH"].iloc[0]
                ).max()
            else:
                df_bars_max = (
                    df_veget_surf.loc[df_veget_surf.index == row_total_alps].iloc[0]
                ).max()

            heights = df_bars_area.values * (ax_size_bar - y_lower_limit) / df_bars_max

            # Add lines percentage
            # --------------------
            per_lines = np.linspace(angles[0], angles[-1] + width + 0.01, num=50)
            df_bars_area_percent_max = df_bars_area_percent.max()

            # Plot percentage lines
            # ---------------------
            if df_bars_area.name in ["FR", "IT"]:
                plot_perc_lines(
                    vals=[20, 60],
                    angles=per_lines,
                    ymin=y_lower_limit,
                    ymax=heights.max(),
                    per_max=df_bars_area_percent_max,
                )
                plot_perc_label(
                    vals=[20, 60],
                    lbls=["20  ", "60%  "],
                    angle=angles[-1] + width,
                    ymin=y_lower_limit,
                    ymax=heights.max(),
                    per_max=df_bars_area_percent_max,
                )
            else:
                plot_perc_lines(
                    vals=[5, 20, 60],
                    angles=per_lines,
                    ymin=y_lower_limit,
                    ymax=heights.max(),
                    per_max=df_bars_area_percent_max,
                )
                plot_perc_label(
                    vals=[5, 20, 60],
                    lbls=["5  ", "20  ", "60%  "],
                    angle=angles[-1] + width,
                    ymin=y_lower_limit,
                    ymax=heights.max(),
                    per_max=df_bars_area_percent_max,
                )

            # Draw bars
            # ---------
            bars = ax_bar.bar(
                x=angles,
                height=heights,
                width=width,
                bottom=y_lower_limit,
                align="edge",
                color=colors,
            )

            # Add labels
            # ----------
            for angle, height, label, index in zip(
                angles, heights, df_bars_area.values, indexes
            ):
                if index == 0:
                    the_lbl = f"{label:.1f}\n[km²]"
                else:
                    the_lbl = f"{label:.1f}\n"
                ax_bar.text(
                    x=angle + width / 2.0,
                    y=heights.max() + y_lower_limit + pad_bar_label,
                    s=the_lbl,
                    ha="left",
                    va="center",
                    fontsize=12,
                    rotation=np.rad2deg(angle + width / 2.0),
                    rotation_mode="anchor",
                )

            # Add connection line
            # -------------------
            bottom_ls = "-"
            bottom_lw = 1.8
            bottom_color = "k"

            p = wedges[-1]  # Outter pie / veget part
            line_start_angle = (np.deg2rad(p.theta2 // 360) + bar_angle_start) / 2
            bottom_line = np.linspace(line_start_angle, angles[-1] + width, num=50)

            # Bottom arc
            plot_perc_lines(
                vals=[0],
                angles=bottom_line,
                ymin=y_lower_limit - 0.005,
                ymax=heights.max(),
                per_max=df_bars_area_percent_max,
                color=bottom_color,
                lw=bottom_lw,
                ls=bottom_ls,
            )

            # Straigth line
            y_bottom_line = get_perc_line(
                val=0,
                ymin=y_lower_limit - 0.005,
                ymax=heights.max(),
                per_max=df_bars_area_percent_max,
            )
            ax_bar.annotate(
                "",
                # The pie veget / arrow
                xy=(
                    0,
                    y_bottom_line - wedge_size_ext / 2.5,
                ),
                # The arc bottom line / no arrow
                xytext=(
                    line_start_angle,
                    y_bottom_line,
                ),
                xycoords="data",
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-|>",
                    ls=bottom_ls,
                    lw=bottom_lw,
                    color=bottom_color,
                    patchB=None,
                    shrinkA=0,
                    shrinkB=0,
                    connectionstyle="arc",
                ),
            )

        # Connection line vegetation for no vegetation
        # --------------------------------------------
        else:
            # Bottom arc
            plot_perc_lines(
                vals=[0],
                angles=bottom_line,
                ymin=y_lower_limit + wedge_size / 1.2,
                ymax=heights.max(),
                per_max=df_bars_area_percent_max,
                color=bottom_color,
                lw=bottom_lw,
                ls=bottom_ls,
            )

            # Straigth line
            y_bottom_line = get_perc_line(
                val=0,
                ymin=y_lower_limit,
                ymax=heights.max(),
                per_max=df_bars_area_percent_max,
            )
            ax_bar.annotate(
                "",
                # The pie veget / arrow
                xy=(
                    0,
                    y_bottom_line,
                ),
                # The arc bottom line / no arrow
                xytext=(
                    line_start_angle,
                    y_lower_limit + wedge_size / 1.2,
                ),
                xycoords="data",
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-|>",
                    ls=bottom_ls,
                    lw=bottom_lw,
                    color=bottom_color,
                    patchB=None,
                    shrinkA=0,
                    shrinkB=0,
                    connectionstyle="arc",
                ),
            )

            # Text
            # ----
            ax_bar.text(
                bar_angle_span / 2.0,
                pie_size_ext + 0.4,
                "No vegetation\nNo water",
                style="italic",
                fontsize=11,
                ha="right",
            )

        if save_dir is not None:
            plt.savefig(
                os.path.join(
                    save_dir, f"donuts_{df_lia_area.name}_{save_name_ext}.png"
                ),
                dpi=300,
                transparent=True,
            )


def plot_donuts_classic(
    df_donuts: pd.DataFrame,
    df_bars: pd.DataFrame,
    lcmap: LandCoverMap,
    col_veget="vegetation_and_water",
    col_rocks="rocks",
    col_snow_and_ice="snow_and_ice",
    col_g1850="Total",
    figsize=(6, 5),
    ratio_heights_bar=0.6,
    save_dir=None,
    save_name_ext=None,
):
    # Reshape df so each country is a dataframe line
    # ==============================================
    df_donuts = df_donuts.unstack(level=1).fillna(0)["Surface"]
    df_bars_perc = df_bars.unstack(level=1).fillna(0)["percent"]
    df_bars = df_bars.unstack(level=1).fillna(0)["Surface"]

    # general parameters
    # ==================

    # ======
    # Plots
    # =====
    for area in range(df_donuts.shape[0]):
        # ========
        # Get Data
        # ========
        df_donuts_area = df_donuts.iloc[area]

        # ======
        # Figure
        # =======
        fig, ax = plt.subplots(
            1,
            1,
            sharex=False,
            layout="constrained",
            figsize=figsize,
        )

        # general parameters of the figure
        # --------------------------------
        pie_size = 1
        arrow_size = 0.3
        arrow_space_size = 0.3
        h_total_size = pie_size * 2 * 2 + arrow_size + arrow_space_size * 2

        v_space = 0.2
        v_bar_size = pie_size * 2 * ratio_heights_bar
        v_total_size = pie_size * 2 + v_space + v_bar_size

        bar_width = 0.5
        bar_pad = 0.05
        bar_x_shift = 1.2
        bar_margins = 0.15

        ax.set_xlim([pie_size - h_total_size, pie_size])
        ax.set_ylim([-pie_size, -pie_size + v_total_size])

        ax.set_aspect("equal")
        ax.axis("off")

        # ======================
        # B plot - Total surface
        # ======================
        inner_ratio_factor = 1.1
        inner_radius_min_thresh = 0.01
        inner_radius_min = 0.025

        inner_vals = df_donuts_area[col_g1850]
        inner_colors = lcmap.get_color_of_code(code=4)

        # Define radius of circle
        # -----------------------
        inner_ratio = (
            inner_vals / df_donuts.loc[df_donuts.index == "Alps", col_g1850].iloc[0]
        )
        if inner_ratio != 1:
            inner_ratio *= inner_ratio_factor

        if inner_ratio < inner_radius_min_thresh:
            inner_ratio = inner_radius_min

        # Plot circle
        # -----------
        x_pos_circle = -pie_size - arrow_size - 2 * arrow_space_size - inner_ratio
        circle = plt.Circle(
            (x_pos_circle, 0),
            inner_ratio,
            facecolor=inner_colors,
            linewidth=2,
            edgecolor="k",
        )
        ax.add_patch(circle)

        # Put labels to circle
        # ---------------------
        if df_donuts_area.name != "Alps":
            inner_label_pos = inner_ratio + 0.1
        else:
            inner_label_pos = 0.1

        ax.text(
            x=x_pos_circle,
            y=inner_label_pos,
            s=f"{df_donuts_area.name}",
            color="k",
            weight="bold",
            fontsize=22,
            fontstyle="normal",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        ax.text(
            x=x_pos_circle,
            y=-inner_label_pos,
            s=f"{inner_vals:.0f} km²",
            color="k",
            fontsize=16,
            fontstyle="italic",
            horizontalalignment="center",
            verticalalignment="top",
        )

        # ====================
        # B plot - Pie surface
        # ====================
        outer_vals = [
            df_donuts_area[col_snow_and_ice],
            df_donuts_area[col_rocks],
            df_donuts_area[col_veget],
        ]

        outer_colors = [
            lcmap.get_color_of_code(code=4),
            lcmap.get_color_of_code(code=0),
            lcmap.get_color_of_code(code=9),
        ]

        # Shift start angle for no vegetation
        # -----------------------------------
        if outer_vals[2] == 0.0:
            startangle = 90 + 90
            pctdistance = 0
            labeldistance = 0
        else:
            startangle = 90
            pctdistance = 0.5
            labeldistance = 0.5

        # To avoid extra lines for only rocks
        if outer_vals[0] == 0 and outer_vals[-1] == 0:
            outer_vals = [outer_vals[1]]
            outer_colors = [outer_colors[1]]

        # Plot pie
        # --------
        wedges, labels, autotexts = ax.pie(
            outer_vals,
            radius=pie_size,
            colors=outer_colors,
            autopct="%.1f%%",
            pctdistance=pctdistance,
            labels=[f"{surf:.0f} km²" for surf in outer_vals],
            labeldistance=labeldistance,
            wedgeprops=dict(edgecolor="k", linewidth=2),
            counterclock=True,
            startangle=startangle,
            frame=True,
        )

        # Update surface / percentage
        # ---------------------------
        # Remove 0 km
        for at, lbl in zip(autotexts, labels):
            if float(at.get_text()[:-1]) != 0.0:
                at.update(
                    dict(
                        fontsize=16,
                        color="k",
                        ha="center",
                        va="center",
                    )
                )
                lbl.update(
                    dict(
                        fontsize=13,
                        fontstyle="italic",
                        color="white",
                        ha="center",
                        va="center",
                        x=lbl.get_position()[0],
                        y=lbl.get_position()[1] + 0.25,
                    )
                )
            else:
                at.update({"text": ""})
                lbl.update({"text": ""})

        # Move vegetation
        if len(labels) > 1:
            labels[2].update(
                dict(
                    x=labels[2].get_position()[0] + 0.3,
                    y=labels[2].get_position()[1] + 0.35,
                )
            )
            autotexts[2].update(
                dict(
                    x=autotexts[2].get_position()[0] + 0.2,
                    y=autotexts[2].get_position()[1] + 0.25,
                )
            )

        # ============================
        # A plot - Vegetation bar plot
        # ============================
        if df_donuts_area.name in df_bars.index:
            # Get data
            # --------
            df_bars_area = df_bars.loc[df_bars.index == df_donuts_area.name].iloc[0]
            df_bars_area_percent = df_bars_perc.loc[
                df_bars_perc.index == df_donuts_area.name
            ].iloc[0]

            df_bars_area = df_bars_area[
                [
                    "LC_6",
                    "LC_1",
                    "LC_2",
                    "LC_3",
                    "LC_5",
                ]
            ]
            df_bars_area_percent = df_bars_area_percent[
                [
                    "LC_6",
                    "LC_1",
                    "LC_2",
                    "LC_3",
                    "LC_5",
                ]
            ]

            colors = [
                lcmap.get_color_of_code(int(code[-1])) for code in df_bars_area.index
            ]
            labels = [
                lcmap.get_type_of_code(int(code[-1])) for code in df_bars_area.index
            ]

            # Scale and offset size of each bar
            # ---------------------------------
            if df_bars_area.name != "Alps":
                df_bars_max = (df_bars.loc[df_bars.index == "CH"].iloc[0]).max()
            else:
                df_bars_max = (df_bars.loc[df_bars.index == "Alps"].iloc[0]).max()

            heights = df_bars_area.values / df_bars_max * 2 * ratio_heights_bar

            # Parameters
            # ----------
            bar_y_bottom = pie_size + v_space
            bar_start = pie_size - bar_x_shift - bar_margins
            x_pos = [bar_start - (bar_width + bar_pad) * i for i in range(len(labels))]

            # FUNCTIONS FOR PLOT PERCENTAGE LINES
            # ===================================
            def get_yval_of_perc(
                perc,
                bottom=0,
                ymax=None,
                perc_max=None,
            ):
                return perc * ymax / perc_max + bottom

            def add_perc_label(x, y, perc_label, nopercent=False):
                if nopercent:
                    lbl = " "
                else:
                    lbl = "%"

                ax.text(
                    x,
                    y,
                    f"{int(perc_label)}{lbl} ",
                    ha="right",
                    va="center",
                    fontsize=10,
                    color="k",
                )

            def plot_perc_lines(
                perc,
                bottom=0,
                xmin=x_pos[-1] - bar_width / 2 - bar_margins,
                xmax=x_pos[0] + bar_width / 2 + bar_margins,
                ymax=heights.max(),
                perc_max=df_bars_area_percent.max(),
                nopercent=False,
            ):
                yval = get_yval_of_perc(
                    perc, bottom=bottom, ymax=ymax, perc_max=perc_max
                )

                ax.hlines(
                    yval,
                    xmin,
                    xmax,
                    colors="k",
                    ls="--",
                    lw=1,
                    zorder=0,
                )
                add_perc_label(xmin, yval, perc, nopercent=nopercent)

            # ===================================

            # PLots percentage lines
            # ----------------------
            ax.hlines(  # bottom lines
                bar_y_bottom - 0.01,
                x_pos[0] + bar_width / 2 + bar_margins,
                x_pos[-1] - bar_width / 2 - bar_margins,
                colors="k",
                lw=1.5,
            )
            if (df_bars_area.name != "FR") and (df_bars_area.name != "IT"):
                plot_perc_lines(perc=5, nopercent=True, bottom=bar_y_bottom)
            plot_perc_lines(perc=20, nopercent=True, bottom=bar_y_bottom)
            plot_perc_lines(perc=60, bottom=bar_y_bottom)

            # Plots bars
            # ===========
            bars = ax.bar(
                x=x_pos,
                height=heights,
                width=bar_width,
                bottom=bar_y_bottom,
                align="center",
                color=colors,
            )
            bars_label = ax.bar(
                x=x_pos,
                height=[heights.max()] * len(heights),
                width=bar_width,
                bottom=bar_y_bottom,
                align="center",
                color=["#ffffff00"] * 6,
            )
            ax.bar_label(
                bars_label,
                labels=df_bars_area.values,
                fmt="%.1f",
                padding=5,
                fontsize=14,
            )
            ax.text(
                x_pos[0] + bar_width / 2,
                bar_y_bottom + heights.max() + 0.12,
                "  km²",
                fontsize=11,
            )

            # Connection line
            # ----------------
            ax.annotate(
                "",
                xy=(x_pos[0] + bar_width / 2 + bar_margins, bar_y_bottom - 0.01),
                xytext=(x_pos[0] + bar_width / 2 + bar_margins + 0.05, pie_size + 0.01),
                xycoords="data",
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-",
                    ls="-",
                    lw=1.5,
                    color="k",
                    patchB=None,
                    shrinkB=0,
                    connectionstyle="arc",
                ),
            )
        # For DE and SL
        # -------------
        else:
            ax.hlines(bar_y_bottom, x_pos_circle, 0, colors="k", lw=2.0)
            ax.text(
                x_pos_circle,
                bar_y_bottom + bar_margins,
                "No vegetation / No water",
                fontsize=12,
                style="italic",
            )

        # Arrow
        # -----
        ax.annotate(
            "",
            xy=(-pie_size - arrow_space_size, 0),
            xytext=(-pie_size - arrow_space_size - arrow_size, 0),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(
                arrowstyle="->",
                ls="-",
                lw=1.8,
                color="k",
            ),
        )

        # Show and save
        # =============
        plt.show()
        if save_dir is not None:
            plt.savefig(
                os.path.join(
                    save_dir, f"donuts_{df_donuts_area.name}_{save_name_ext}.png"
                ),
                dpi=200,
                transparent=True,
            )


def plot_violin(df, lcmap, save_dir, save_name):
    """

    :param df:
    :param lcmap:
    :param save_dir:
    :param save_name:
    :return:
    """

    # Reorder lcmap for plottings
    # ===========================
    lcmap_reindex = lcmap.reindex(reverse=True, in_place=False)
    lcmap_reindex.remove_item(
        col_name="Code", col_val=[9], in_place=True
    )  # Do it after reindex otherwise bug

    # Figure
    # ======
    f1, axes = plt.subplots(1, 1, figsize=(5, 5))

    # Plot
    ax = sbn.violinplot(
        data=df,
        x="altitude",
        y="landcover",
        orient="horizontal",
        hue="lia",
        order=lcmap_reindex.get_code(),
        hue_order=[True, False],
        density_norm="width",
        cut=0,
        width=1,
        split=True,
        gap=0.05,
        inner="quart",
        legend=True,
        ax=axes,
    )

    # Colors and legend
    # =================
    colors_double = []
    for color in list(lcmap_reindex.get_colors()):
        colors_double.append(color)
        colors_double.append(color)

    handles = []
    for (ind, violin), color in zip(
        enumerate(ax.findobj(PolyCollection)), colors_double
    ):
        alpha = 1
        if ind % 2 != 0:
            alpha = 0.4
        violin.set_facecolor(color)
        violin.set_alpha(alpha=alpha)
        handles.append(
            plt.Rectangle(
                (0, 0), 0, 0, facecolor=color, alpha=alpha - 0.1, edgecolor="black"
            )
        )

    # Legend
    labels = [lbl.replace(" ", "\n") for lbl in lcmap_reindex.get_type()]
    ax.legend(
        handles=[tuple(handles[::2]), tuple(handles[1::2])],
        labels=["LIA", "Outside LIA"],
        title=None,
        handlelength=4,
        loc=4,
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
    )

    ax.set_xlabel("Altitude [m]", fontsize=14, labelpad=10)
    ax.set_yticklabels(labels)
    ax.set(ylabel=None)

    # Add percentage for each violin
    # ==============================
    # Get percentage for LIA deiced
    surf_deiced = lc_distrib.get_lc_surface(
        df.loc[df.lia == True],
        groupby=["Country", "landcover"],
        round_dec=1,
        add_total=True,
        slope_correction=True,
    )
    surf_deiced.loc["Alps"] = surf_deiced.sum(numeric_only=True)
    perc_deiced = (surf_deiced.div(surf_deiced.Total_LC, axis=0) * 100).drop(
        "Total_LC", axis=1
    )
    perc_deiced_alps = (
        perc_deiced[
            [f"LC_{code}" for code in lcmap_reindex.get_code()]
        ]  # Reorder landcover
        .loc[perc_deiced.index == "Alps"]
        .values[0]  # Get as array
    )

    # Get percentage for outside LIA
    surf_out = lc_distrib.get_lc_surface(
        df.loc[df.lia == False],
        groupby=["Country", "landcover"],
        round_dec=2,
        add_total=True,
        slope_correction=True,
    )
    surf_out.loc["Alps"] = surf_out.sum(numeric_only=True)
    perc_out = (surf_out.div(surf_out.Total_LC, axis=0) * 100).drop("Total_LC", axis=1)
    perc_out_alps = (
        perc_out[
            [f"LC_{code}" for code in lcmap_reindex.get_code()]
        ]  # Reorder landcover
        .loc[perc_out.index == "Alps"]
        .values[0]  # Get as array
    )

    # Plot
    # ====
    x_pos = (
        df.groupby(["landcover"])["altitude"]
        .max()
        .reindex(lcmap_reindex.get_code())
        .tolist()
    )
    x_pos[2] = 3800  # modify rocks position
    y_pos = range(len(x_pos))

    for i in range(len(x_pos)):
        ax.text(
            x_pos[i],
            y_pos[i] - 0.15,  # -0...as y axis origin is up
            f"{perc_deiced_alps[i]:.1f}%",
            va="center",
        )
        ax.text(
            x_pos[i],
            y_pos[i] + 0.2,  # +0... as y axis origin is up
            f"{perc_out_alps[i]:.1f}%",
            alpha=0.6,
            va="center",
        )

    # Grid and general parameters
    # ===========================
    sbn.despine(ax=ax, left=True)
    ax.tick_params(left=False, rotation=0, pad=-5, axis="y")
    ml = MultipleLocator(250)
    ax.xaxis.set_minor_locator(ml)

    ax.set_axisbelow(True)
    ax.grid(ls="--", which="minor", color="lightgrey", zorder=0)
    ax.grid(ls="--", which="major", color="lightgrey", zorder=0)

    ax.yaxis.grid(False)

    plt.tight_layout()
    plt.show()

    if save_name is not None:
        plt.savefig(os.path.join(save_dir, save_name), dpi=300)
