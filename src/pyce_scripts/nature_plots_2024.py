import os

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
    df: pd.DataFrame,
    lcmap: LandCoverMap,
    col_veget="vegetation_and_water",
    col_rocks="rocks",
    col_snow_and_ice="snow_and_ice",
    col_g1850="Total",
    save_dir=None,
):
    # Reshape df so each country is a dataframe line
    # ==============================================
    df_stats = df.unstack(level=1).fillna(0)["Surface"]

    # ==============
    # DONUTS
    # ==============

    # general parameters
    wedge_size = 0.5
    inner_ratio_factor = 1.2

    # Plots
    for area in range(df_stats.shape[0]):
        fig, ax = plt.subplots(figsize=(5, 5))

        # ========
        # Get Data
        # ========
        df_area = df_stats.iloc[area]

        # =====
        # INNER
        # =====
        inner_vals = [df_area[col_g1850]] * 2
        inner_colors = ["#74a9cf"] * 2

        # Define radius of inner pie
        # --------------------------
        inner_ratio = (
            df_area[col_g1850]
            / df_stats.loc[df_stats.index == "Alps", col_g1850].iloc[0]
        )
        if inner_ratio != 1:
            inner_ratio *= inner_ratio_factor

        inner_radius = (1 - wedge_size * 1.1) * inner_ratio
        if inner_radius < 0.05:
            inner_radius = 0.05

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
        if df_area.name != "Alps":
            inner_label_pos = inner_radius + 0.1
        else:
            inner_label_pos = 0.1

        labels[0].update(
            dict(
                text=f"{df_area.name}",
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
            df_area[col_snow_and_ice],
            df_area[col_rocks],
            df_area[col_veget],
        ]

        outer_colors = [
            lcmap.get_color_of_code(code=4),
            lcmap.get_color_of_code(code=0),
            lcmap.get_color_of_code(code=3),
        ]

        # Redefine outer_vals for small values
        # ------------------------------------
        shift_snow_and_ice = 0.0
        if outer_vals[0] / df_area[col_g1850] * 100 < 1:  # Below 1% of total
            shift_snow_and_ice = df_area[col_g1850] / 100  # 1% of total in surface
            outer_vals[0] += shift_snow_and_ice  # add 1% to snow and ice
            outer_vals[1] -= shift_snow_and_ice  # remove 1% to rocks

        # Plot pie
        # --------
        wedges, labels, autotexts = ax.pie(
            outer_vals,
            radius=1,
            colors=outer_colors,
            autopct="%1.1f%%",
            pctdistance=0.75,
            wedgeprops=dict(width=wedge_size, edgecolor="k", linewidth=0.5),
            counterclock=False,
            startangle=0,
        )

        # Update percentage
        # -----------------
        # Remove 0% labels
        for at in autotexts:
            if float(at.get_text()[:-1]) != 0.0:
                at.update(
                    {
                        "fontsize": 18,
                        "fontstyle": "italic",
                        "horizontalalignment": "center",
                        "verticalalignment": "center_baseline",
                    }
                )
            else:
                at.update({"text": ""})

        # Put correct values for snow and ice
        if shift_snow_and_ice != 0:
            for at, sign in zip([autotexts[0], autotexts[1]], [-1, 1]):
                orig_surf = (
                    float(at.get_text()[:-1]) * df_area[col_g1850] / 100
                    + sign * shift_snow_and_ice
                )
                orig_percent = orig_surf / df_area[col_g1850] * 100
                at.set_text(f"{orig_percent:.1f}%")

        # Layout
        # ======
        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            plt.savefig(
                os.path.join(save_dir, f"donuts_{df_area.name}_v4.png"),
                dpi=200,
                transparent=True,
            )
