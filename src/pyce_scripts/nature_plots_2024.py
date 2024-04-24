import pandas as pd
import seaborn as sbn
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
):
    # Reshape df so each country is a dataframe line
    # ==============================================
    df_stats = df.unstack(level=1).fillna(0)["Surface"]

    # ==============
    # DONUTS
    # ==============

    # general parameters
    size = 0.4

    # Plots
    for area in range(df_stats.shape[0]):
        fig, ax = plt.subplots(figsize=(5, 5))

        # ========
        # Get Data
        # ========
        df_area = df_stats.iloc[area]

        inner_vals = [df_area[col_g1850]]
        inner_colors = [lcmap.get_color_of_code(code=4)]

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

        # =====
        # INNER
        # =====
        wedges, labels, autotexts = ax.pie(
            inner_vals,
            radius=1 - size - ,
            colors=inner_colors,
            autopct="%1.1f%%",
            pctdistance=0.0,
            wedgeprops=dict(width=0.1, edgecolor="w"),
            counterclock=False,
            startangle=0,
        )

        autotexts[0].update(
            dict(
                text=f"{df_area.name}",
                color="w",
                weight="bold",
                fontsize=20,
                fontstyle="normal",
                x=autotexts[0].get_position()[0] + 0.1,
                y=autotexts[0].get_position()[1] - 0.0,
            )
        )

        # =====
        # OUTER
        # =====
        wedges, labels, autotexts = ax.pie(
            outer_vals,
            radius=1,
            colors=outer_colors,
            autopct="%1.1f%%",
            labeldistance=0.45,
            pctdistance=0.8,
            wedgeprops=dict(width=size, edgecolor="#ffffff00"),
            counterclock=False,
            startangle=0,
        )

        for at in autotexts:
            at.update({"fontsize": 15, "fontstyle": "italic"})

        plt.tight_layout()
        plt.show()
