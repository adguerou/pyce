import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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


def plot_donuts(
    df_donuts: pd.DataFrame,
    df_bars: pd.DataFrame,
    lcmap: LandCoverMap,
    col_veget="vegetation_and_water",
    col_rocks="rocks",
    col_snow_and_ice="snow_and_ice",
    col_g1850="Total",
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
    pie_size = 0.6
    wedge_size = 0.2
    margin_inner_pie = 0.04
    margin_pie_bar = -0.05
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
        inner_colors = ["#74a9cf"] * 2

        # Define radius of inner pie
        # --------------------------
        inner_ratio = (
            df_donuts_area[col_g1850]
            / df_donuts.loc[df_donuts.index == "Alps", col_g1850].iloc[0]
        )
        if inner_ratio != 1:
            inner_ratio *= inner_ratio_factor

        inner_radius = (pie_size - wedge_size) * inner_ratio - margin_inner_pie
        if inner_radius < inner_radius_min_thresh:
            inner_radius = inner_radius_min

        # add external black line
        # -----------------------
        circle = plt.Circle((0, 0), inner_radius, color="k", linewidth=2)
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
        if df_donuts_area.name != "Alps":
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
            autopct=lambda per: "{:.0f}".format(per * np.sum(outer_vals) / 100),
            pctdistance=0.8,
            labels=[f"{per:.1f}%" for per in outer_vals / np.sum(outer_vals) * 100],
            labeldistance=1.25,
            wedgeprops=dict(width=wedge_size, edgecolor="k", linewidth=0.5),
            counterclock=False,
            startangle=startangle,
        )

        # Update surface / percentage
        # --------------
        # Remove 0 km
        for at, lbl in zip(autotexts, labels):
            if float(at.get_text()) != 0.0:
                at.update(
                    {
                        "fontsize": 12,
                        "fontstyle": "italic",
                        "color": "white",
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
                    x=labels[2].get_position()[0] + 0.1,
                    y=labels[2].get_position()[1] - 0.05,
                )
            )

        # Move percent of roks for swiss
        if df_donuts_area.name == "CH":
            labels[1].update(
                dict(
                    x=labels[1].get_position()[0] - 0.2,
                    y=labels[1].get_position()[1] - 0.08,
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
            y_lower_limit = pie_size + margin_pie_bar  # bottom of bars
            if df_bars_area.name != "Alps":
                df_bars_max = (df_bars.loc[df_bars.index == "CH"].iloc[0]).max()
            else:
                df_bars_max = (df_bars.loc[df_bars.index == "Alps"].iloc[0]).max()

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
            bottom_lw = 0.6
            bottom_color = "black"
            bottom_line = np.linspace(
                angles[0] - angle_offset / 2, angles[-1] + width, num=50
            )

            # Bottom arc
            plot_perc_lines(
                vals=[-5],
                angles=bottom_line,
                ymin=y_lower_limit,
                ymax=heights.max(),
                per_max=df_bars_area_percent_max,
                color=bottom_color,
                lw=bottom_lw,
                ls=bottom_ls,
            )

            # Straigth line
            y_bottom_line = get_perc_line(
                -5,
                ymin=y_lower_limit,
                ymax=heights.max(),
                per_max=df_bars_area_percent_max,
            )
            ax_bar.annotate(
                "",
                xy=(
                    angles[0] - angle_offset / 2,
                    y_bottom_line,
                ),
                xytext=(angles[0] - angle_offset / 2, y_bottom_line + margin_pie_bar),
                xycoords="data",
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-",
                    ls=bottom_ls,
                    lw=bottom_lw,
                    color=bottom_color,
                    patchB=None,
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
            bottom_lw = 1.1
            bottom_color = "grey"
            bottom_line = np.linspace(0, np.pi / ratio_bar_angle, num=50)

            # Bottom arc
            plot_perc_lines(
                vals=[-5],
                angles=bottom_line,
                ymin=pie_size,
                ymax=1,
                per_max=60,
                color=bottom_color,
                lw=bottom_lw,
                ls=bottom_ls,
            )

            # Straigth line
            y_bottom_line = get_perc_line(
                -5,
                ymin=y_lower_limit,
                ymax=heights.max(),
                per_max=df_bars_area_percent_max,
            )
            ax_bar.annotate(
                "",
                xy=(
                    0,
                    y_bottom_line,
                ),
                xycoords="data",
                xytext=(0, y_bottom_line + margin_pie_bar),
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-",
                    ls=bottom_ls,
                    lw=bottom_lw,
                    color=bottom_color,
                    patchB=None,
                    shrinkB=0,
                    connectionstyle="arc",
                ),
            )

            # Text
            # ----
            ax_bar.text(
                np.pi / ratio_bar_angle / 2.5,
                pie_size + 0.3,
                "No vegetation\nNo water",
                style="italic",
                fontsize=10,
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
        fig, axes = plt.subplot_mosaic(
            """
            AA
            BC
            """,
            layout="constrained",
            figsize=figsize,
            height_ratios=[ratio_heights_bar, 1 - ratio_heights_bar],
        )
        for ax in [axes["B"], axes["C"]]:
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_aspect("equal")

        for ax in [axes["A"], axes["B"], axes["C"]]:
            ax.axis("off")

        # ======================
        # B plot - Total surface
        # ======================
        ax = axes["B"]
        inner_ratio_factor = 1.1
        inner_radius_min_thresh = 0.01
        inner_radius_min = 0.025

        inner_vals = df_donuts_area[col_g1850]
        inner_colors = "#74a9cf"

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
        circle = plt.Circle(
            (0, 0),
            inner_ratio,
            facecolor=inner_colors,
            linewidth=0.5,
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
            x=0,
            y=inner_label_pos,
            s=f"{df_donuts_area.name}",
            color="k",
            weight="bold",
            fontsize=18,
            fontstyle="normal",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        ax.text(
            x=0,
            y=-inner_label_pos,
            s=f"{inner_vals:.0f} km²",
            color="k",
            fontsize=16,
            fontstyle="italic",
            horizontalalignment="center",
            verticalalignment="top",
        )

        # ====================
        # C plot - Pie surface
        # ====================
        ax = axes["C"]
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

        # Shift start angle for no vegetation
        # -----------------------------------
        if outer_vals[2] == 0.0:
            startangle = 90 + 90
            pctdistance = 0
        else:
            startangle = 90
            pctdistance = 0.4

        # To avoid extra lines for only rocks
        if outer_vals[0] == 0 and outer_vals[-1] == 0:
            outer_vals = [outer_vals[1]]
            outer_colors = [outer_colors[1]]

        # Plot pie
        # --------
        wedges, labels, autotexts = ax.pie(
            outer_vals,
            radius=1,
            colors=outer_colors,
            autopct=lambda per: "{:.0f}".format(per * np.sum(outer_vals) / 100),
            pctdistance=pctdistance,
            labels=[f"{per:.1f}%" for per in outer_vals / np.sum(outer_vals) * 100],
            labeldistance=1.4,
            wedgeprops=dict(edgecolor="k", linewidth=0.5),
            counterclock=True,
            startangle=startangle,
            frame=True,
        )
        ax.axis("off")  # Need to do so to have pie same size as left circle
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])

        # Update surface / percentage
        # ---------------------------
        # Remove 0 km
        for at, lbl in zip(autotexts, labels):
            if float(at.get_text()) != 0.0:
                at.update(
                    {
                        "fontsize": 14,
                        "fontstyle": "italic",
                        "color": "white",
                        "horizontalalignment": "center",
                        "verticalalignment": "center",
                    }
                )
                lbl.update(
                    {
                        "fontsize": 16,
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
                    x=labels[2].get_position()[0] + 0.5,
                    y=labels[2].get_position()[1] - 0.3,
                )
            )
            autotexts[2].update(
                dict(
                    x=autotexts[2].get_position()[0] + 0.05,
                    y=autotexts[2].get_position()[1] + 0.4,
                )
            )

        # ============================
        # A plot - Vegetation bar plot
        # ============================
        ax = axes["A"]
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

            heights = df_bars_area.values / df_bars_max

            # Parameters
            # ----------
            width = 0.2
            bar_pad = 0.2
            x_pos = [1 + (width + bar_pad) * i for i in range(len(labels))]
            x_shift = 0.5

            ax.set_xlim([x_pos[0] - x_shift * 0.9, x_pos[-1] + x_shift * 1.1])
            ax.set_ylim([-0.05, 1.1])

            # Add percentage lines
            # --------------------
            def get_perc_line(
                val,
                ymax=None,
                per_max=None,
            ):
                return val * ymax / per_max

            def plot_perc_lines(
                val,
                xmin=x_pos[0] - width * 1.2,
                xmax=x_pos[-1] + width,
                ymax=heights.max(),
                per_max=df_bars_area_percent.max(),
                ax=ax,
                nopercent=False,
            ):
                yval = get_perc_line(val, ymax=ymax, per_max=per_max)

                ax.hlines(
                    yval,
                    xmin,
                    xmax,
                    colors="k",
                    ls="--",
                    lw=1,
                    zorder=0,
                )
                add_perc_label(val, yval, nopercent=nopercent)

            def add_perc_label(y, yval, nopercent=False):
                if nopercent:
                    lbl = " "
                else:
                    lbl = "%"

                ax.text(
                    x_pos[0] - width * 1.2,
                    yval,
                    f"{int(y)}{lbl} ",
                    ha="right",
                    va="center",
                    fontsize=8,
                    color="k",
                )

            # PLots lines
            ax.hlines(0, x_pos[0] - width, x_pos[-1] + width, colors="k", lw=2.0)
            plot_perc_lines(val=5, nopercent=True)
            plot_perc_lines(val=20, nopercent=True)
            plot_perc_lines(val=60)

            # Plots bars
            # ===========
            bars = ax.bar(
                x=x_pos,
                height=heights,
                width=width,
                bottom=0,
                align="center",
                color=colors,
            )
            bars_label = ax.bar(
                x=x_pos,
                height=[heights.max()] * len(heights),
                width=width,
                bottom=0,
                align="center",
                color=["#ffffff00"] * 6,
            )
            ax.bar_label(
                bars_label,
                labels=df_bars_area.values,
                fmt="%.1f",
                padding=5,
                fontsize=12,
            )
            ax.text(x_pos[-1] + width / 2, heights.max() + 0.05, " [km²]", fontsize=11)

        # For DE and SL
        # -------------
        else:
            ax.set_xlim([-1, 1])
            ax.set_ylim([-0.05, 1.1])
            ax.hlines(0, -1 * 0.6, 1 * 0.4, colors="k", lw=2.0)
            ax.text(
                -0.5,
                0.2,
                "No vegetation / No water",
                fontsize=12,
                style="italic",
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
