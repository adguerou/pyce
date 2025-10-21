import os
from operator import index
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sbn
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import PathPatch
from matplotlib.ticker import MultipleLocator
from pyce.tools.lc_mapping import LandCoverMap

from pyce_scripts import lc_distrib


def get_fig1_numbers(
    df_stats,
    round_dec: int = None,
    col_name_country: str = "Country",
    col_name_snow_and_ice: str = "snow_and_ice",
    col_name_deglaciated: str = "deglaciated",
    col_name_aquatic: str = "aquatic",
    col_name_veget: str = "veget",
    col_name_rocks: str = "rocks",
    col_name_landcover="landcover",
    val_rocks=0,
    val_snow=4,
    val_water=5,
    slope_correction=False,
):
    def _get_surf_and_perc(df, groupby=None, columns=None, round=round_dec):
        if columns is None:
            columns = ["category"]
        if groupby is None:
            groupby = [col_name_country, "category"]

        surf = lc_distrib.get_lc_surface(
            df,
            groupby=groupby,
            reshape_index=[col_name_country],
            reshape_columns=columns,
            round_dec=round,
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
        ).round(round_dec)

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
        (df_lia.glacier == False) & (df_lia[col_name_landcover] != 4), "category"
    ] = col_name_deglaciated

    # Get surfaces and percentages
    # ----------------------------
    surfaces_lia = _get_surf_and_perc(df_lia)

    # 2. Statistics over DEGLACIATED area
    # ===================================
    df_deglaciated = df_lia.loc[
        (df_lia.glacier == False) & (df_lia[col_name_landcover] != 4)
    ]
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


def get_table_SI_1(
    df,
    lcmap: LandCoverMap,
    lcmap_deglaciated_name: Union[None, str] = "deglaciated",
    lcmap_glacier_name="snow & ice",
    lcmap_veget_name="vegetation",
    lcmap_rocks_name="rocks & sediments",
    lcmap_water_name: Union[None, str] = "water",
    veget_codes=[1, 2, 3, 6],
    surface_to_divide_by: Union[None, pd.DataFrame] = None,
    round_dec=1,
):
    # Test on names
    if not lcmap_deglaciated_name in lcmap.df["Type"].values:
        raise IOError(f"{lcmap_deglaciated_name} not in LandCoverMap")
    if lcmap_glacier_name is not None:
        if not (lcmap_glacier_name in lcmap.df["Type"].values):
            raise IOError(f"{lcmap_glacier_name} not in LandCoverMap")
    if not (lcmap_veget_name in lcmap.df["Type"].values):
        raise IOError(f"{lcmap_veget_name} not in LandCoverMap")

    # ==========
    #  SURFACES
    # ==========
    # Surface by country and landcovers
    df_surf = lc_distrib.get_lc_surface(df, groupby=["Country", "LC"])

    # Deglaciated
    if lcmap_glacier_name is None:
        not_deg_cols = ["Total_LC"]
    else:
        not_deg_cols = [f"LC_{lcmap.get_code_of_type(lcmap_glacier_name)}", "Total_LC"]

    df_surf[f"LC_{lcmap.get_code_of_type(lcmap_deglaciated_name)}"] = df_surf[
        df_surf.columns[~df_surf.columns.isin(not_deg_cols)]
    ].sum(axis=1)

    # Vegetation
    lc_veget = [f"LC_{code}" for code in veget_codes]
    df_surf[f"LC_{lcmap.get_code_of_type(lcmap_veget_name)}"] = df_surf[
        df_surf.columns[df_surf.columns.isin(lc_veget)]
    ].sum(axis=1)

    # Add line for ALPS / Z_ALPS to keep alps at last row
    df_surf.loc["Z_ALPS"] = df_surf.sum()

    # Reorder columns
    surface_SI_1 = df_surf[["Total_LC"] + [f"LC_{code}" for code in lcmap.get_code()]]

    # ==========
    # PERCENTAGE
    # ==========
    if surface_to_divide_by is None:
        surface_to_divide_by = surface_SI_1

    percent_lia = surface_SI_1.div(surface_to_divide_by["Total_LC"], axis=0) * 100
    percent_lia = percent_lia.map(f"{{:.{round_dec}f}}%".format)
    percent_lia["Total_LC"] = np.nan
    percent_lia["zone"] = "1_lia"

    # Deglaciated
    # -----------
    percent_deiced = (
        surface_SI_1.div(
            surface_to_divide_by[
                [f"LC_{lcmap.get_code_of_type(lcmap_deglaciated_name)}"]
            ].sum(axis=1),
            axis=0,
        )
        * 100
    )
    percent_deiced = percent_deiced.map(f"{{:.{round_dec}f}}%".format)
    if lcmap_glacier_name is not None:
        percent_deiced[f"LC_{lcmap.get_code_of_type(lcmap_glacier_name)}"] = np.nan
    percent_deiced["Total_LC"] = np.nan
    percent_deiced["zone"] = "2_deiced"

    # Vegetation
    # -----------
    percent_veget = (
        surface_SI_1.div(
            surface_to_divide_by[
                [f"LC_{lcmap.get_code_of_type(lcmap_veget_name)}"]
            ].sum(axis=1),
            axis=0,
        )
        * 100
    )
    percent_veget = percent_veget.map(f"{{:.{round_dec}f}}%".format)

    if lcmap_glacier_name is None:
        if lcmap_water_name is not None:
            not_veget_cols = [
                f"LC_{lcmap.get_code_of_type(lcmap_deglaciated_name)}",
                f"LC_{lcmap.get_code_of_type(lcmap_rocks_name)}",
                f"LC_{lcmap.get_code_of_type(lcmap_water_name)}",
            ]
        else:
            not_veget_cols = [
                f"LC_{lcmap.get_code_of_type(lcmap_deglaciated_name)}",
                f"LC_{lcmap.get_code_of_type(lcmap_rocks_name)}",
            ]
    else:
        not_veget_cols = [
            f"LC_{lcmap.get_code_of_type(lcmap_glacier_name)}",
            f"LC_{lcmap.get_code_of_type(lcmap_deglaciated_name)}",
            f"LC_{lcmap.get_code_of_type(lcmap_rocks_name)}",
            f"LC_{lcmap.get_code_of_type(lcmap_water_name)}",
        ]
    percent_veget[not_veget_cols] = np.nan
    percent_veget.loc[percent_veget.index == "SI"] = np.nan
    percent_veget["Total_LC"] = np.nan
    percent_veget["zone"] = "3_veget"

    # Concat surface and percentage + rename Alps
    # ===========================================
    surface_SI_1 = surface_SI_1.map(f"{{:.{round_dec}f}}".format)
    surface_SI_1 = surface_SI_1.assign(zone="0_lia")

    table_SI_1 = (
        pd.concat([surface_SI_1, percent_lia, percent_deiced, percent_veget])
        .reset_index()
        .set_index(["Country", "zone"])
        .sort_index()
        .rename(
            index={
                "Z_ALPS": "ALPS",
                "0_lia": "[km²]",
                "1_lia": "LIA",
                "2_deiced": "Deglaciated",
                "3_veget": "Vegetation",
            },
        )
    ).fillna("-")

    def rename_lc_cols(cols):
        mydict = {}
        for col in cols:
            mydict[col] = lcmap.get_type_of_code(int(col[-1]))
        return mydict

    return table_SI_1.rename(
        columns={**{"Total_LC": "LIA"}, **rename_lc_cols(table_SI_1.columns[1:])}
    )


def get_table_SI_2(df_lia, df_buffer, lcmap_lia, lcmap_buffer, round_dec=1):
    # Get percentage of buffer through TABLE SI 1
    # ===========================================
    table_buffer = get_table_SI_1(
        df_buffer, lcmap_buffer, lcmap_glacier_name=None, round_dec=round_dec
    )

    table_buffer_SI_2 = table_buffer.reset_index()
    table_buffer_SI_2 = (
        table_buffer_SI_2.drop(
            table_buffer_SI_2.loc[table_buffer_SI_2.zone == "LIA"].index
        )
        .drop(table_buffer_SI_2.loc[table_buffer_SI_2.zone == "[km²]"].index)
        .drop(columns=["deglaciated", "LIA"])
        .assign(LIA="Out")
        .set_index(["Country"])
        .rename(
            index={
                "ALPS": "Z_ALPS",
            }
        )
    )

    # Get percentage of LIA through TABLE SI 1
    # ===========================================
    table_lia = get_table_SI_1(df_lia, lcmap_lia)

    table_lia_SI_2 = table_lia.reset_index()
    table_lia_SI_2 = (
        table_lia_SI_2.drop(table_lia_SI_2.loc[table_lia_SI_2.zone == "LIA"].index)
        .drop(table_lia_SI_2.loc[table_lia_SI_2.zone == "[km²]"].index)
        .drop(columns=["deglaciated", "LIA", "snow & ice"])
        .assign(LIA="In")
        .set_index(["Country"])
        .rename(
            index={
                "ALPS": "Z_ALPS",
            }
        )
    )

    # Concat both
    # ===========
    table_SI_2 = (
        pd.concat([table_lia_SI_2, table_buffer_SI_2])
        .reset_index()
        .set_index(
            [
                "Country",
                "zone",
                "LIA",
            ]
        )
        .sort_index()
        .rename(
            index={
                "Z_ALPS": "ALPS",
            }
        )
    )
    return table_SI_2


def get_table_SI_2_uncert(
    df_lia, df_buffer, lcmap_lia, lcmap_buffer, surface_to_divide_by, round_dec=1
):
    # Get percentage of buffer through TABLE SI 1
    # ===========================================
    table_buffer = get_table_SI_1(
        df_buffer,
        lcmap_buffer,
        surface_to_divide_by=surface_to_divide_by,
        lcmap_glacier_name=None,
        lcmap_water_name=None,
        round_dec=round_dec,
    )

    table_buffer_SI_2 = table_buffer.reset_index()
    table_buffer_SI_2 = (
        table_buffer_SI_2.drop(
            table_buffer_SI_2.loc[table_buffer_SI_2.zone == "LIA"].index
        )
        .drop(table_buffer_SI_2.loc[table_buffer_SI_2.zone == "[km²]"].index)
        .drop(columns=["deglaciated", "LIA"])
        .assign(LIA="Out")
        .set_index(["Country"])
        .rename(
            index={
                "ALPS": "Z_ALPS",
            }
        )
    )

    # Get percentage of LIA through TABLE SI 1
    # ===========================================
    table_lia = get_table_SI_1(
        df_lia,
        lcmap_lia,
        surface_to_divide_by=surface_to_divide_by,
        lcmap_glacier_name=None,
        lcmap_water_name=None,
        round_dec=round_dec,
    )

    table_lia_SI_2 = table_lia.reset_index()
    table_lia_SI_2 = (
        table_lia_SI_2.drop(table_lia_SI_2.loc[table_lia_SI_2.zone == "LIA"].index)
        .drop(table_lia_SI_2.loc[table_lia_SI_2.zone == "[km²]"].index)
        .drop(columns=["deglaciated", "LIA"])
        .assign(LIA="In")
        .set_index(["Country"])
        .rename(
            index={
                "ALPS": "Z_ALPS",
            }
        )
    )

    # Concat both
    # ===========
    table_SI_2 = (
        pd.concat([table_lia_SI_2, table_buffer_SI_2])
        .reset_index()
        .set_index(
            [
                "Country",
                "zone",
                "LIA",
            ]
        )
        .sort_index()
        .rename(
            index={
                "Z_ALPS": "ALPS",
            }
        )
    )
    return table_SI_2


def get_table_SI_4(table_SI_1, table_SI_2, lcmap_veget, stocks, factor):
    # Functions to get CO tons from stocks and surface
    def get_CO(df, cols, stocks, factor):
        df_CO = pd.DataFrame(index=df.index)

        for col, stock in zip(cols, stocks):
            df_CO[col] = df[col] * stock * factor

        df_CO["Total"] = df_CO.sum(axis=1)

        return df_CO

    # Today estimation
    # ================
    # CO tons
    table_surf_today = (
        table_SI_1.loc[
            table_SI_1.zone == "[km²]", ["Country"] + list(lcmap_veget.df.Type)
        ]
        .set_index("Country")
        .astype(np.float64)
    )

    table_co_today = get_CO(
        table_surf_today,
        cols=list(lcmap_veget.get_type()),
        stocks=stocks,
        factor=factor,
    )

    # Replace real zero with nan
    table_co_today = table_co_today.replace({0: np.nan})

    # CO percentage
    table_co_today_perc = table_co_today.div(table_co_today["Total"], axis=0) * 100

    table_co_today_perc.loc[table_co_today_perc.index != "ALPS", "Total"] = (
        table_co_today.loc[table_co_today.index != "ALPS", "Total"]
        / table_co_today.loc[table_co_today.index == "ALPS", "Total"].values
        * 100
    )  # Modify percentage of total as the percentage of each country compared to ALPS

    # Define future index of the final table
    table_co_today["time"] = "2015"
    table_co_today_perc["time"] = "2015"

    table_co_today["unit"] = "xxtC"
    table_co_today_perc["unit"] = "%"

    # Future estimation
    # =================
    # Get future surface for each vegetation type
    surf_deglaciated = (
        table_SI_1.loc[table_SI_1.zone == "[km²]", ["Country", "deglaciated"]]
        .set_index("Country")
        .astype(np.float64)
    )

    percent_futur_str = table_SI_2.loc[
        (table_SI_2.LIA == "Out") & (table_SI_2.zone == "Deglaciated"),
        ["Country"] + list(lcmap_veget.df.Type),
    ].set_index("Country")

    def remove_percent(x):
        return np.float64(x[:-1])

    percent_futur = percent_futur_str.map(remove_percent)
    surf_future = percent_futur.div(100).mul(surf_deglaciated.values, axis=1)

    # CO tons
    table_co_future = get_CO(
        surf_future,
        cols=list(lcmap_veget.get_type()),
        stocks=stocks,
        factor=factor,
    )

    # CO percentage
    table_co_future_perc = table_co_future.div(table_co_future["Total"], axis=0) * 100

    # Modify percentage of total as the percentage of each country compared to ALPS
    table_co_future_perc.loc[table_co_future_perc.index != "ALPS", "Total"] = (
        table_co_future.loc[table_co_future.index != "ALPS", "Total"]
        / table_co_future.loc[table_co_future.index == "ALPS", "Total"].values
        * 100
    )

    # Define future index of the final table
    table_co_future["time"] = "future"
    table_co_future_perc["time"] = "future"

    table_co_future["unit"] = "xxtC"
    table_co_future_perc["unit"] = "%"

    # Concatenation and formatting
    # ============================
    table_SI_4_co = (
        pd.concat([table_co_today, table_co_future])
        .rename(index={"ALPS": "Z_ALPS"})
        .reset_index()
        .sort_values(["Country", "unit"])
        .set_index(["Country", "unit", "time"])
        .reindex(["xxtC", "%"], level=1)
        .rename(index={"Z_ALPS": "ALPS"})
        .round(1)
    )
    table_SI_co_str = (
        table_SI_4_co.map(f"{{:.0f}}".format)
        .replace({"nan": ""})
        .astype(str)
        .droplevel(1)
    )

    table_SI_4_perc = (
        pd.concat([table_co_today_perc, table_co_future_perc])
        .rename(index={"ALPS": "Z_ALPS"})
        .reset_index()
        .sort_values(["Country", "unit"])
        .set_index(["Country", "unit", "time"])
        .reindex(["xxtC", "%"], level=1)
        .rename(index={"Z_ALPS": "ALPS"})
        .round(1)
        .replace({100: np.nan})
    )
    table_SI_perc_str = (
        table_SI_4_perc.map(f"{{:.0f}}".format)
        .replace({"nan": ""})
        .astype(str)
        .replace({"0": "<1"})
        .astype(str)
        .droplevel(1)
    )

    # Append columns of alpine contribution of each country to the co stocks table
    table_SI_co_str["Alpine contribution"] = (
        table_SI_perc_str[["Total"]] + "%"
    ).replace({"%": ""})

    # Final formatting
    table_SI_4 = (
        table_SI_co_str.reset_index(level=1)
        .rename(columns={"time": "[ktC]"})
        .replace({"2015": "2020", "future": "potential"})
        .set_index(["[ktC]"], append=True)
        .sort_index()
    )

    return table_SI_4, table_SI_4_co, table_SI_4_perc


def get_table_SI_fig3_pp(
    pp_perc,
):
    def myformat(val):
        try:
            float(val)
            if np.isnan(val):
                return val
            else:
                return int(val)
        except ValueError:
            return val

    # Reindex + format
    pp_perc_format = (
        pp_perc.reindex(["LIA", "GLACIER", "DEGLACIATED", "VEGET", "WATER"], level=1)
        .replace({np.nan: -1})  # To force columns with only floats to convert to int
        .replace({0.1: "<1"})  # Annotations for 0.1 (less than 15)
        .map(lambda x: str(myformat(x)) + "%")  # Convert to int and str
        .replace({"-1%": "-"})
    )

    return pp_perc_format


def get_table_SI_fig3_ski(pp_ski, infra=None):
    def myformat(val):
        try:
            float(val)
            if np.isnan(val):
                return val
            else:
                return round(val, 0)
        except ValueError:
            return val

    # Reindex + format
    df_ski = pp_ski.replace({"ALL": "ALPS"})

    # Length
    df_length = pd.concat(
        [
            df_ski,
            pd.DataFrame(
                data={
                    "infra": infra,
                    "Country": "SI",
                    "LIA": np.nan,
                    "GLACIER": np.nan,
                    "DEICED": np.nan,
                    "ALPS": np.nan,
                },
                index=[0],
            ),
        ]
    )
    df_length = df_length.set_index(["infra", "Country"])
    df_length = df_length.loc[:, df_length.columns != "ALPS"]

    # percentage
    df_perc = df_length.div(df_length["LIA"], axis=0) * 100
    df_perc["unit"] = "%"
    df_perc["LIA"] = ""

    df_length["unit"] = "[km]"  # here otherwise cannot divide str

    # Concat
    df_ski = (
        pd.concat([df_length, df_perc])
        .set_index(["unit"], append=True)
        .sort_index(level=[1, 2], ascending=[True, False])
        .rename(columns={"DEICED": "DEGLACIATED"})
        .replace({np.nan: -1})  # To force columns with only floats to convert to int
        .map(lambda x: myformat(x))  # Convert to int and str
        .replace({0: "<1"})  # Annotations for 0.1 (less than 15)
        .replace({-1: "-"})
    )

    return df_ski


def get_table_SI_fig3_dams(
    df,
):
    # Add missing countries
    df_all = pd.concat(
        [
            df,
            pd.DataFrame(
                [[np.nan] * len(df.columns)], index=["FR"], columns=df.columns
            ),
            pd.DataFrame(
                [[np.nan] * len(df.columns)], index=["DE"], columns=df.columns
            ),
            pd.DataFrame(
                [[np.nan] * len(df.columns)], index=["SI"], columns=df.columns
            ),
        ]
    )

    # Format
    df_format = df_all.replace({"count": np.nan}, 0).astype({"count": int})
    df_format["%_total_water_area"] = df_format["%_total_water_area"].apply(
        "{:.0f}%".format
    )
    df_format["artificial_water_area"] = df_format["artificial_water_area"].apply(
        "{:.1f}".format
    )

    df_format = (
        df_format.rename(
            columns={
                "count": "Number",
                "%_total_water_area": "Total water surface contribution",
                "artificial_water_area": "Surface [km²]",
            }
        )
        .replace({"nan": "-", "nan%": "-"})
        .sort_index()
    )

    return df_format


def get_table_SI_fig3_appendix(df, lcmap):
    def rename_lc_cols(cols):
        mydict = {}
        for col in cols:
            mydict[col] = lcmap.get_type_of_code(int(col[-1]))
        return mydict

    def myformat(val):
        try:
            float(val)
            if np.isnan(val):
                return val
            elif val < 1:
                return "<1"
            else:
                return f"{val:.0f}"
        except ValueError:
            return val

    df_format = (
        df.rename(
            columns={
                **{
                    "Total": "Total length [km²]",
                    "perc": "Part of total infrastructure [%]",
                },
                **rename_lc_cols(df.columns[:-2]),
            },
            index={
                "type": "infrastructure",
                "IUCN_strong": "IUCN+",
                "IUCN_weak": "IUCN",
                "WH": "UNESCO",
            },
        )
        .map(lambda x: myformat(x))
        .replace({np.nan: ""})
    )

    df_format["Part of total infrastructure [%]"] += "%"

    return df_format


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


def plot_fig_1_donuts(
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
            autopct=lambda per: "({:.0f}%)".format(per),
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
            if (
                float(at.get_text()[1:-2]) != 0.0
            ):  # [1:-2] to remove parenthesis from returned str
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
        # Move snow&ice except very low snow & ice (DE)
        if outer_vals[0] >= 1:
            update_lbl_pct(
                wedges[0], labels[0], delta_ang=10, delta_dist=-0.02
            )  # Surface
            update_lbl_pct(
                wedges[0], autotexts[0], delta_ang=-40, delta_dist=0, rot=-45
            )  # percentage
        else:
            update_lbl_pct(
                wedges[0], autotexts[0], delta_ang=0, delta_dist=0.35, rot=0
            )  # percentage
            if outer_vals[0] != 0.0:
                labels[0].update({"text": "<1"})  # surface

        # Move deglaciated
        if (df_lia_area.name != "SI") & (df_lia_area.name != "DE"):
            update_lbl_pct(wedges[1], labels[1], delta_ang=0, delta_dist=0.0)
            update_lbl_pct(wedges[1], autotexts[1], delta_ang=55, delta_dist=0, rot=40)
        elif df_lia_area.name == "SI":
            update_lbl_pct(
                wedges[1], autotexts[1], delta_ang=0, delta_dist=0.25, rot=30
            )
        elif df_lia_area.name == "DE":
            update_lbl_pct(wedges[1], autotexts[1], delta_ang=80, delta_dist=0.0, rot=0)

        # ============================================================================
        #                          Deglaciated plot
        # =============================================================================
        df_deglaciated_area = df_deglaciated_surf.iloc[area]
        df_deglaciated_perc_area = df_deglaciated_perc.iloc[area]

        outer_vals = [
            df_lia_area[col_snow_and_ice],  # snow
            df_deglaciated_area[col_aquatic],  # water
            df_deglaciated_area[col_rocks],  # rocks
            df_deglaciated_area[col_veget],  # veget
        ]

        outer_perc = [
            0,  # snow
            df_deglaciated_perc_area[col_aquatic],  # water
            df_deglaciated_perc_area[col_rocks],  # rocks
            df_deglaciated_perc_area[col_veget],  # veget
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

        # When no vegetation nor water
        # ----------------------------
        if outer_vals[index_rocks] / df_lia_area[col_country_total] == 1:
            outer_vals = [outer_vals[index_rocks]]
            outer_colors = [outer_colors[index_rocks]]

        # Plot pie
        # --------
        wedges, labels, autotexts = ax.pie(
            outer_vals,
            radius=pie_size_ext,
            colors=outer_colors,
            autopct="(%.0f%%)",
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

        # Put correct values for rocks
        # ----------------------------
        if shift_water != 0:
            outer_vals[index_water] -= shift_water  # remove 1% to water
            outer_vals[index_rocks] += shift_water  # add 1% to rocks

        # Update percentage without snow cover / change surface values for water_shift < 1%
        # -----------------------------------------------------------------------------
        for at, lbl, perc, val in zip(autotexts, labels, outer_perc, outer_vals):
            at.update(
                {
                    "text": f"({perc:.0f}%)",
                    "fontsize": 10,
                    "fontstyle": "italic",
                    "horizontalalignment": "center",
                    "verticalalignment": "center",
                }
            )
            lbl.update(
                {
                    "text": f"{val:.0f}",
                    "fontsize": 13,
                    "color": "k",
                    "horizontalalignment": "center",
                    "verticalalignment": "center",
                }
            )

        # Remove values for snow & ice
        autotexts[0].update({"text": ""})
        labels[0].update({"text": ""})

        # Update water percentage below 1% @ round 0
        if (outer_perc[1] <= 0.5) and (df_lia_area.name != "SI"):
            autotexts[1].update(
                {
                    "text": f"(<1%)",
                    "fontsize": 10,
                    "fontstyle": "italic",
                    "horizontalalignment": "center",
                    "verticalalignment": "center",
                }
            )
        if df_lia_area.name in df_veget_surf.index:
            if df_lia_area.name != "DE":
                # Move water
                update_lbl_pct(wedges[1], labels[1], delta_ang=0, delta_dist=0.3)
                update_lbl_pct(
                    wedges[1], autotexts[1], delta_ang=8, delta_dist=0.3, rot=0
                )

                # Move rocks
                update_lbl_pct(wedges[2], labels[2], delta_ang=0, delta_dist=0.0)
                update_lbl_pct(
                    wedges[2], autotexts[2], delta_ang=45, delta_dist=0, rot=35
                )

                # Move veget
                update_lbl_pct(wedges[3], labels[3], delta_ang=0, delta_dist=0.25)
                update_lbl_pct(
                    wedges[3], autotexts[3], delta_ang=-8, delta_dist=0.3, rot=0
                )
            else:  # DE
                # Water
                labels[1].update({"text": ""})
                autotexts[1].update({"text": ""})

                # Veget
                labels[3].update({"text": ""})
                autotexts[3].update({"text": ""})

                # Rocks
                update_lbl_pct(
                    wedges[2], autotexts[2], delta_ang=80, delta_dist=0, rot=0
                )  # percentage

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
        if (df_lia_area.name in df_veget_surf.index) & (df_lia_area.name != "DE"):
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
            if df_bars_area.name in ["FR"]:
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
            elif df_bars_area.name == "ALPS":
                plot_perc_lines(
                    vals=[
                        5,
                        20,
                        65,
                    ],  # put 65 so the line is not on the border and not visible
                    angles=per_lines,
                    ymin=y_lower_limit,
                    ymax=heights.max(),
                    per_max=df_bars_area_percent_max,
                )
                plot_perc_label(
                    vals=[
                        5,
                        20,
                        65,
                    ],  # put 65 so the line is not on the border and not visible
                    lbls=["5  ", "20  ", "70%  "],
                    angle=angles[-1] + width,
                    ymin=y_lower_limit,
                    ymax=heights.max(),
                    per_max=df_bars_area_percent_max,
                )
            else:
                plot_perc_lines(
                    vals=[20, 65],
                    angles=per_lines,
                    ymin=y_lower_limit,
                    ymax=heights.max(),
                    per_max=df_bars_area_percent_max,
                )
                plot_perc_label(
                    vals=[20, 65],
                    lbls=["20  ", "70%  "],
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
                    the_lbl = f"{label:.0f}\n"  # [km²]"
                else:
                    the_lbl = f"{label:.0f}\n"
                ax_bar.text(
                    x=angle + width / 2.0,
                    y=heights.max() + y_lower_limit + pad_bar_label,
                    s=the_lbl,
                    ha="center",
                    va="center",
                    fontsize=12,
                    rotation=np.rad2deg(angle + width / 2.0) - 90,
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
            if df_lia_area.name == "DE":
                p_noveget = wedges[2]  # Outter pie / rocks

            line_start_angle = np.deg2rad(p_noveget.theta1 + 5)
            bottom_line = np.linspace(line_start_angle, np.deg2rad(70), num=50)

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
                # The deglaciated pie / arrow
                xy=(
                    line_start_angle,
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


def plot_fig_2a(
    df,
    lcmap,
    table_percent_in=None,
    table_percent_out=None,
    x_violin="altitude",
    save_dir=None,
    save_name=None,
):
    """

    :param df:
    :param lcmap:
    :param table_percent_in:
    :param table_percent_out:
    :param x_violin:
    :param save_dir:
    :param save_name:
    :return:

    """

    # Reorder lcmap for plottings
    # ===========================
    lcmap_reindex = lcmap.reindex(reverse=True, in_place=False)

    # Figure
    # ======
    f1, axes = plt.subplots(1, 1, figsize=(5, 5))

    #       Plot
    # ==================
    ax = sbn.violinplot(
        data=df,
        x=x_violin,
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
            alpha = 0.3
        violin.set_facecolor(color)
        violin.set_alpha(alpha=alpha)
        handles.append(
            plt.Rectangle(
                (0, 0), 0, 0, facecolor=color, alpha=alpha - 0.1, edgecolor="black"
            )
        )

    # Legend
    ax.legend(
        handles=[tuple(handles[::2]), tuple(handles[1::2])],
        labels=["LIA", "Outside LIA"],
        title=None,
        handlelength=4,
        loc=4,
        frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
    )

    ax.set_xlabel("Altitude [m a.s.l]", fontsize=13)
    labels = [
        lbl.replace(" v", "\nv").replace("& s", "&\ns")
        for lbl in lcmap_reindex.get_type()
    ]
    ax.set_yticklabels(labels)
    ax.set(ylabel=None)

    # Add percentage for each violin
    # ==============================
    # Set positions
    x_pos = (
        df.groupby(["landcover"])["altitude"]
        .max()
        .reindex(lcmap_reindex.get_code())
        .tolist()
    )

    x_pos[1] = 4000  # modify rocks position
    x_pos[2] = 3500  # modify sparse position
    y_pos = range(len(x_pos))

    # Plots
    # -----
    for i in range(len(x_pos)):
        # LIA
        ax.text(
            x_pos[i],
            y_pos[i] - 0.15,  # -0... as y axis origin is up
            f"{table_percent_in[i]}",
            va="center",
        )
        # OUT
        ax.text(
            x_pos[i],
            y_pos[i] + 0.2,  # +0... as y axis origin is up
            f"{table_percent_out[i]}",
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


def plot_fig_2b(
    df: pd.DataFrame,
    lcmap: LandCoverMap,
    lcmap_veget: LandCoverMap,
    dict_params: dict = None,
    save_name: str = None,
    save_dir: str = None,
):
    # Make sure landcover maps are well ordered
    # -----------------------------------------
    lcmap = lcmap.reindex(reverse=True, in_place=False)
    lcmap_veget = lcmap_veget.reindex(reverse=True, in_place=False)

    # Colormaps
    # ---------
    palette = [lcmap.get_color_of_code(code=0), lcmap.get_color_of_code(code=8)]
    palette_veget = list(lcmap_veget.get_colors())

    if dict_params is None:
        dict_params = dict()
        dict_params["lia"] = "lia"
        dict_params["veget"] = "veget"
        dict_params["landcover"] = "landcover"
        dict_params["var_x"] = "maat"
        dict_params["var_y"] = "slope"
        dict_params["binwidth_x"] = 0.2
        dict_params["binwidth_y"] = 2
        dict_params["thresh"] = 0.01
        dict_params["vmin"] = 0.05
        dict_params["vmax"] = 0.3
        dict_params["xlim"] = (-11.0, 7.8)
        dict_params["ylim"] = (-0.0, 80)
        dict_params["xlabel"] = "MAAT [°C]"
        dict_params["ylabel"] = "Slope [degree]"
        dict_params["label_margins"] = "LIA"
        dict_params["dpi"] = 300

    #          Grid
    # ======================
    # main plot for LIA only (veget vs rocks)
    g = sbn.JointGrid(
        data=df.loc[df[dict_params["lia"]] == True],
        x=dict_params["var_x"],
        y=dict_params["var_y"],
        hue=dict_params["veget"],
        palette=palette,
        marginal_ticks=True,
        xlim=dict_params["xlim"],
        ylim=dict_params["ylim"],
        ratio=3,
        height=5,
        space=0.05,
    )

    # 2D scatter plots + density
    # ==========================
    # Scatter plot of veget and rocks
    g.plot_joint(
        sbn.histplot,
        binwidth=(dict_params["binwidth_x"], dict_params["binwidth_y"]),
        stat="percent",
        common_norm=False,
        thresh=dict_params["thresh"],
        vmin=dict_params["vmin"],
        vmax=dict_params["vmax"],
        cbar=False,
        alpha=1,
        legend=True,
    )
    # Contours of rocks and veget
    g.plot_joint(
        sbn.kdeplot,
        common_norm=False,
        fill=False,
        linewidths=1,
        levels=[0.2, 0.4, 0.6, 0.8, 0.95],
        legend=False,
    )

    #          Box plots
    # =============================
    # X axis
    h1 = sbn.boxplot(
        data=df,
        x=dict_params["var_x"],
        y=dict_params["lia"],
        hue=dict_params["veget"],
        order=[False, True],
        hue_order=[
            False,
            True,
            "wfefwe",
        ],  # fake extra hue to shift box plot to insert veget bix plots
        orient="horizontal",
        width=0.8,
        ax=g.ax_marg_x,
        legend=False,
        palette=palette,
        gap=0.1,
        fliersize=0.05,
    )
    # Y axis
    h2 = sbn.boxplot(
        data=df,
        x=dict_params["lia"],
        y=dict_params["var_y"],
        hue=dict_params["veget"],
        order=[True, False],
        hue_order=[
            False,
            True,
            "wfefwe",
        ],  # fake extra hue to shift box plot to insert veget bix plots
        width=0.8,
        ax=g.ax_marg_y,
        legend=False,
        palette=palette,
        gap=0.1,
        fliersize=0.05,
    )

    # Vegetation box plot
    # ===================
    # X axis
    h1 = sbn.boxplot(
        data=df.loc[(df.veget == True)],
        x=dict_params["var_x"],
        y=dict_params["lia"],
        hue=dict_params["landcover"],
        orient="horizontal",
        hue_order=[0, 0, 0, 0, 0, 0, 0, 0]
        + list(
            lcmap_veget.get_code()
        ),  # fake extra hue to shift box plot / number is empirical
        width=0.9,
        ax=g.ax_marg_x,
        legend=True,  # To get handles to create my own later on
        palette=palette_veget,
        fliersize=0.05,
        gap=0.05,
    )
    # Y axis
    h2 = sbn.boxplot(
        data=df.loc[(df.veget == True)],
        x=dict_params["lia"],
        y=dict_params["var_y"],
        hue=dict_params["landcover"],
        order=[True, False],
        hue_order=[0, 0, 0, 0, 0, 0, 0, 0]
        + list(
            lcmap_veget.get_code()
        ),  # fake extra hue to shift box plot / number is empirical
        width=0.9,
        ax=g.ax_marg_y,
        legend=False,
        palette=palette_veget,
        fliersize=0.05,
        gap=0.05,
    )

    #             LEGEND
    # ==========================
    h1.legend_.remove()

    # Colors - alpha for non LIA
    # --------------------------
    # Color of rock/veget/landcover type x 2 for LIA in and out
    palette_marginal = (
        np.concatenate((np.array([palette]).T, np.array([palette]).T), axis=1)
        .ravel()
        .tolist()
    )
    palette_marginal_veget = (
        np.concatenate(
            (np.array([palette_veget]).T, np.array([palette_veget]).T), axis=1
        )
        .ravel()
        .tolist()
    )

    lines_per_boxplot_h1 = len(h1.lines) // len(h1.findobj(PathPatch))
    lines_per_boxplot_h2 = len(h2.lines) // len(h2.findobj(PathPatch))

    # Loop over each box plot
    # -----------------------
    # X axis
    for (ind, box), color in zip(
        enumerate(h1.findobj(PathPatch)), palette_marginal + palette_marginal_veget
    ):
        alpha = 1
        if ind % 2 == 0:
            alpha = 0.4
        box.set_facecolor(color)
        box.set_alpha(alpha=alpha)

        for line in h1.lines[
            ind * lines_per_boxplot_h1 : (ind + 1) * lines_per_boxplot_h1
        ]:
            # line.set_color("green")
            line.set_alpha(alpha=alpha)
            line.set_mec(
                (0, 0, 0, alpha / 2),
            )  # edgecolor of fliers

    # Y axis
    for (ind, box), color in zip(
        enumerate(h2.findobj(PathPatch)), palette_marginal + palette_marginal_veget
    ):
        alpha = 1
        if ind % 2 != 0:
            alpha = 0.4
        box.set_facecolor(color)
        box.set_alpha(alpha=alpha)

        for line in h2.lines[
            ind * lines_per_boxplot_h2 : (ind + 1) * lines_per_boxplot_h2
        ]:
            line.set_alpha(alpha=alpha)
            line.set_mec(
                (0, 0, 0, alpha / 2),
            )  # edgecolor of fliers

    # General parameters
    # =================
    h1.xaxis.set_label_position("top")
    h1.set_xlabel(dict_params["xlabel"], fontsize=13, labelpad=8)
    h1.set_ylabel(dict_params["label_margins"], fontsize=10)
    h1.set_yticklabels(labels=["out", "in"], fontsize=10)
    sbn.despine(ax=h1, left=True, bottom=True, top=False)
    h1.tick_params(left=False, bottom=False, labelbottom=False, top=True, labeltop=True)

    h2.yaxis.set_label_position("right")
    h2.set_xlabel(dict_params["label_margins"], fontsize=10)
    h2.set_xticklabels(labels=["in", "out"], fontsize=10)
    h2.set_ylabel(dict_params["ylabel"], fontsize=13, labelpad=8)
    sbn.despine(ax=h2, left=True, bottom=True, right=False)
    h2.tick_params(
        bottom=False, left=False, labelleft=False, right=True, labelright=True
    )

    # Figure parameters
    # =================
    plt.subplots_adjust(right=0.94, top=0.92)
    sbn.move_legend(
        g.ax_joint,
        "upper left",
        ncol=1,
        title=None,
        frameon=False,
        fontsize=9,
        labels=["Rocks & sediments", "Vegetation"],
    )
    g.set_axis_labels(dict_params["xlabel"], dict_params["ylabel"], fontsize=14)
    g.refline(x=0, y=30, marginal=False)

    # Sve figure
    if save_name is not None:
        plt.savefig(
            os.path.join(save_dir, save_name), dpi=dict_params["dpi"], transparent=False
        )


def plot_fig_3_pp(
    df_perc,
    lcmap: LandCoverMap,
    colors: dict = None,
    xlabels=["IUCN+", "IUCN", "WH", "RAMSAR"],
    ylabels=["", "Glacier", "Deglaciated", "Veget", "Water"],
    vmin=0.1,  # vmin à 0.1 / comme cela les valeurs à zeros (pas la protection) sont de la couleur set_under
    # 0.1 et pas 1 pour quand le max de pp_perc =1, il y ai qd meme un range de couleur entre vmin/vmax, sinon heatmap fonctionne pas bien
    # et 0.1 pour que ce soit == aux valeurs processés pour qd il y a moins de 1% dans la catégorie (0 dans le excel / mis à 0.1 pr faire la diff avec les zeros de pas de protection)
    save_dir: str = None,
    save_name: str = None,
):
    # Function for colormaps
    # ======================
    if colors is None:
        colors = {
            "strong": "#1b9e77",
            "weak": "#d95f02",
            "wh": "#7570b3",
            "ramsar": "#e7298a",
            "glacier": lcmap.get_color_of_code(code=4),
            "deglaciated": lcmap.get_color_of_code(code=9),
            "veget": lcmap.get_color_of_code(code=8),
            "water": lcmap.get_color_of_code(code=5),
        }

    def get_cmap(color, color_under=None):
        if color_under is None:
            cmap = LinearSegmentedColormap.from_list(
                name="mycmap", colors=[color, color], N=256
            )
            cmap.set_under(color)  # couleur sous vmin / pas la protection
        else:
            cmap = LinearSegmentedColormap.from_list("mycmap", ["white", color], N=256)
            cmap.set_under(color_under)  # couleur sous vmin / pas la protection
            cmap.set_bad(
                "grey", alpha=0.2
            )  # Couleur pour les NaN / categorie de landcover non exitantes
        return cmap

    # Format df to annotate heatmaps
    def myformat(val):
        try:
            float(val)
            if np.isnan(val):
                return val
            else:
                return int(val)
        except ValueError:
            return val

    df_annot = (
        df_perc.replace(
            {np.nan: -1}
        )  # To force columns with only floats to convert to int
        .replace({0.1: "<1"})  # Annotations for 0.1 (less than 15)
        .map(lambda x: myformat(x))  # Convert to int and str
        .replace(
            {-1: ""}
        )  # Convert back -1 to empty string (rather than NaN otherwise broadcast to float)
    )

    # ==========================================
    #            PLOT per country
    # ==========================================
    for country in list(set(df_perc.index.get_level_values(0))):
        # Get data
        df = df_perc.loc[df_perc.index.get_level_values(0) == country]

        # PLOT
        fig, axes = plt.subplots(
            5,
            1,
            figsize=(1.5, 1.5 * 5 / 4),
            gridspec_kw={
                "hspace": 0,
                "wspace": 0,
                "left": 0,
                "right": 1,
                "bottom": 0,
                "top": 1,
            },
        )

        # Total - first line
        # ------------------
        # Strong
        df_plot = df[df.index.get_level_values(1) == "LIA"]
        annot = df_annot.loc[df_plot.index]

        sbn.heatmap(
            df_plot,
            annot=annot,
            fmt="",
            annot_kws={"weight": "bold", "fontsize": 11},
            cmap=get_cmap(colors["strong"]),
            cbar=False,
            vmin=vmin,
            vmax=5,
            linewidths=1,
            linecolor="k",
            xticklabels=False,
            yticklabels=False,
            square=True,
            ax=axes[0],
        )

        # Weak
        df_plot.loc[:, "IUCN_strong"] = np.nan
        annot.loc[:, "IUCN_strong"] = np.nan

        sbn.heatmap(
            df_plot,
            annot=annot,
            fmt="",
            annot_kws={"weight": "bold", "fontsize": 11},
            cmap=get_cmap(colors["weak"]),
            cbar=False,
            vmin=vmin,
            vmax=5,
            linewidths=1,
            linecolor="k",
            xticklabels=False,
            yticklabels=False,
            square=True,
            ax=axes[0],
        )

        # WH
        df_plot.loc[:, "IUCN_weak"] = np.nan
        annot.loc[:, "IUCN_weak"] = np.nan

        sbn.heatmap(
            df_plot,
            annot=annot,
            fmt="",
            annot_kws={"weight": "bold", "fontsize": 11},
            cmap=get_cmap(colors["wh"]),
            cbar=False,
            vmin=vmin,
            vmax=5,
            linewidths=1,
            linecolor="k",
            xticklabels=False,
            yticklabels=False,
            square=True,
            ax=axes[0],
        )

        # RAMSAR
        df_plot.loc[:, "WH"] = np.nan
        annot.loc[:, "WH"] = np.nan

        # In case one want to add label inside plots
        if country == "ALPS":
            xlabel = False
            ylabel = False
        else:
            xlabel = False
            ylabel = False

        sbn.heatmap(
            df_plot,
            annot=annot,
            fmt="",
            annot_kws={"weight": "bold", "fontsize": 11},
            cmap=get_cmap(colors["ramsar"]),
            cbar=False,
            vmin=vmin,
            vmax=5,
            linewidths=1,
            linecolor="k",
            xticklabels=xlabel,
            yticklabels=False,
            square=True,
            ax=axes[0],
        )

        # Glacier - second line
        df_plot = df.loc[df.index.get_level_values(1) == "GLACIER"]
        annot = df_annot.loc[df_plot.index]
        sbn.heatmap(
            df_plot,
            annot=annot,
            fmt="",
            annot_kws={"weight": "normal", "color": "k"},
            cmap=get_cmap(colors["glacier"], color_under="k"),
            cbar=False,
            vmin=vmin,
            vmax=np.max([df_plot.max().max(), vmin + 1]),
            linewidths=1,
            linecolor="k",
            xticklabels=False,
            yticklabels=ylabel,
            square=True,
            ax=axes[1],
        )

        # Deglaciated - third line
        df_plot = df.loc[df.index.get_level_values(1) == "DEGLACIATED"]
        annot = df_annot.loc[df_plot.index]
        sbn.heatmap(
            df_plot,
            annot=annot,
            fmt="",
            annot_kws={"weight": "normal", "color": "k"},
            cmap=get_cmap(colors["deglaciated"], color_under="k"),
            cbar=False,
            vmin=vmin,
            vmax=np.max([df_plot.max().max(), vmin + 1]),
            linewidths=1,
            linecolor="k",
            xticklabels=False,
            yticklabels=ylabel,
            square=True,
            ax=axes[2],
        )

        # Veget - fourth line
        df_plot = df.loc[df.index.get_level_values(1) == "VEGET"]
        annot = df_annot.loc[df_plot.index]
        sbn.heatmap(
            df_plot,
            annot=annot,
            fmt="",
            annot_kws={"weight": "normal", "color": "k"},
            cmap=get_cmap(colors["veget"], color_under="k"),
            cbar=False,
            vmin=vmin,
            vmax=np.max([df_plot.max().max(), vmin + 1]),
            linewidths=1,
            linecolor="k",
            xticklabels=False,
            yticklabels=ylabel,
            square=True,
            ax=axes[3],
        )

        # Water - fifth line
        df_plot = df.loc[df.index.get_level_values(1) == "WATER"]
        annot = df_annot.loc[df_plot.index]
        sbn.heatmap(
            df_plot,
            annot=annot,
            fmt="",
            annot_kws={"weight": "normal", "color": "k"},
            cmap=get_cmap(colors["water"], color_under="k"),
            cbar=False,
            vmin=vmin,
            vmax=np.max([df_plot.max().max(), vmin + 1]),
            linewidths=1,
            linecolor="k",
            xticklabels=False,
            yticklabels=ylabel,
            square=True,
            ax=axes[4],
        )

        # Settings
        for ax in axes:
            ax.set(xlabel="", ylabel="")
            ax.tick_params(left=False, top=False)

        if save_name is not None:
            plt.savefig(os.path.join(save_dir, save_name + f"_{country}.png"), dpi=300)


def plot_fig_SI_2(
    df_lia: pd.DataFrame,
    df_training: pd.DataFrame,
    xplot: str = "NCRI",
    yplot: str = "NARI",
    hue: str = "landcover",
    lcmap_lia: LandCoverMap = None,
    lcmap_training: LandCoverMap = None,
    lcmap_legend: LandCoverMap = None,
    xlim: tuple[float, float] = (-0.2, 0.7),
    ylim: tuple[float, float] = (-0.4, 0.5),
    title=None,
    save_dir=None,
    save_name=None,
):
    # TRAINING / Scatter plot + kde for marginal
    # ==========================================
    g = sbn.jointplot(
        data=df_training,
        x=xplot,
        y=yplot,
        kind="scatter",
        hue=hue,
        hue_order=lcmap_training.get_code().tolist(),
        palette=lcmap_training.get_colors().tolist(),
        joint_kws=dict(s=2, legend=True, edgecolor="None"),
        marginal_kws={"common_grid": True, "common_norm": True},
        marginal_ticks=False,
        xlim=xlim,
        ylim=ylim,
        alpha=0.4,
        height=5,
        ratio=5,
        space=0.8,
    )

    # TRAINING - KDE contours on main plot
    # ------------------------------------
    sbn.kdeplot(
        data=df_training,
        x=xplot,
        y=yplot,
        hue=hue,
        hue_order=lcmap_training.get_code().tolist(),
        palette=lcmap_training.get_colors().tolist(),
        legend=False,
        linewidths=1,
        alpha=1,
        fill=False,
        ax=g.ax_joint,
    )

    # Scatter only of LIA (all + sparse)
    # =================================
    sbn.scatterplot(
        data=df_lia,
        x=xplot,
        y=yplot,
        hue=hue,
        hue_order=lcmap_lia.get_code().tolist(),
        palette=lcmap_lia.get_colors().tolist(),
        legend=True,
        s=2,
        alpha=0.2,
        zorder=0,
        ax=g.ax_joint,
    )

    # LAYOUT
    # ======
    # Legend
    sbn.move_legend(g.ax_joint, "lower right", title=None, markerscale=5)

    # Get handles and plot legend
    labels = lcmap_legend.get_type().tolist()
    handles, _labels = g.ax_joint.get_legend_handles_labels()
    handles = [handles[2], handles[0], handles[3], handles[1], handles[4], handles[5]]

    leg = g.ax_joint.legend(
        handles=handles, labels=labels, loc="lower right", title=None, markerscale=4
    )
    for hdl in leg.legend_handles:
        hdl.set_alpha(1)

    # Title
    if title is not None:
        g.ax_marg_x.text(
            0.35, 0.65, title, weight="bold", transform=g.ax_marg_x.transAxes
        )

    # X/Y lables
    g.ax_joint.set_xlabel(xlabel=xplot[:-2], fontweight="bold")
    g.ax_joint.set_ylabel(ylabel=yplot[:-2], fontweight="bold")

    # Grid
    plt.grid(ls="--")

    # Saving
    if save_dir is not None and save_name is not None:
        plt.savefig(os.path.join(save_dir, save_name) + ".png", dpi=200)


def plot_fig_SI_5(
    ds_cook: pd.DataFrame,
    ds_shugar: pd.DataFrame,
    ds_lia: pd.DataFrame,
    save_dir=None,
    save_name=None,
):
    # FIGURE
    # ======
    fig, ax = plt.subplots(figsize=(6, 4))

    # Shugard
    ax.scatter(
        ds_shugar["area_m2"] / 1e6,
        ds_shugar["volume_1e6m3"],
        s=20,
        alpha=1,
        color="#a1dab4",
        edgecolors="k",
        lw=0.5,
        label="Shugar et al. (2020)",
        zorder=0,
    )

    # Cook
    ax.scatter(
        ds_cook["area_m2"] / 1e6,
        ds_cook["volume_1e6m3"],
        s=30,
        alpha=1,
        color="#41b6c4",
        marker="d",
        edgecolors="k",
        lw=0.5,
        label="Cook et al. (2015)",
        zorder=0,
    )

    # LIA lakes
    ax.scatter(
        ds_lia["area_m2"] / 1e6,
        ds_lia["volume_1e6m3_combined"],
        s=20,
        alpha=1,
        edgecolors="k",
        color="#225ea8",
        label="LIA lakes",
        zorder=20,
    )

    condi_size = ds_lia["area_m2"] < 1e3
    ax.errorbar(
        ds_lia.loc[condi_size, "area_m2"] / 1e6,
        ds_lia.loc[condi_size, "volume_1e6m3_combined"],
        yerr=ds_lia.loc[condi_size, "errors_vol_1e6m3_std"],
        color="None",
        ecolor="k",
        lw=0.5,
        ls="None",
        errorevery=(0, 1),
    )

    condi_size = (ds_lia["area_m2"] >= 1e3) & (ds_lia["area_m2"] < 1e4)
    ax.errorbar(
        ds_lia.loc[condi_size, "area_m2"] / 1e6,
        ds_lia.loc[condi_size, "volume_1e6m3_combined"],
        yerr=ds_lia.loc[condi_size, "errors_vol_1e6m3_std"],
        color="None",
        ecolor="k",
        lw=0.5,
        ls="None",
        errorevery=(0, 5),
    )

    condi_size = (ds_lia["area_m2"] >= 1e4) & (ds_lia["area_m2"] < 2 * 1e4)
    ax.errorbar(
        ds_lia.loc[condi_size, "area_m2"] / 1e6,
        ds_lia.loc[condi_size, "volume_1e6m3_combined"],
        yerr=ds_lia.loc[condi_size, "errors_vol_1e6m3_std"],
        color="None",
        ecolor="k",
        lw=0.5,
        ls="None",
        errorevery=(0, 10),
    )

    condi_size = (ds_lia["area_m2"] >= 2 * 1e4) & (ds_lia["area_m2"] < 1e5)
    ax.errorbar(
        ds_lia.loc[condi_size, "area_m2"] / 1e6,
        ds_lia.loc[condi_size, "volume_1e6m3_combined"],
        yerr=ds_lia.loc[condi_size, "errors_vol_1e6m3_std"],
        color="None",
        ecolor="k",
        lw=0.5,
        ls="None",
        errorevery=(0, 2),
    )

    condi_size = (ds_lia["area_m2"] >= 1e5) & (ds_lia["area_m2"] < 1e7)
    ax.errorbar(
        ds_lia.loc[condi_size, "area_m2"] / 1e6,
        ds_lia.loc[condi_size, "volume_1e6m3_combined"],
        yerr=ds_lia.loc[condi_size, "errors_vol_1e6m3_std"],
        color="None",
        ecolor="k",
        lw=0.5,
        ls="None",
        errorevery=(0, 1),
    )

    # Settings
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Lake surface [km²]", fontsize=12)
    ax.set_ylabel("Lake volume [1e6 m³]", fontsize=12)

    ax.legend(fontsize=10)
    plt.tight_layout()

    # Saving
    if save_dir is not None and save_name is not None:
        plt.savefig(os.path.join(save_dir, save_name) + ".png", dpi=200)
