from typing import Union

import contextily as cx
import geopandas as gpd
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import rioxarray as rioxr
import seaborn as sbn
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from pyce import raster
from pyce.tools.lc_mapping import LandCoverMap


def get_lc_percent(
    df_input: Union[str, gpd.GeoDataFrame, pd.DataFrame],
    groupby: list[str],
    lc_col_name: str = "LC",
):
    # Open shp file + remove corrupted item with no geometry
    if isinstance(df_input, str):
        df = gpd.read_file(df_input)
        df = df.iloc[np.where(df["geometry"].values != None)]
    else:
        df = df_input

    # Get a multi_index dataframe (groupby) with each colum being
    # the LC sum values per group for each LC class
    df_percent_by_group = df.groupby(groupby)[
        df.columns[df.columns.str.startswith(lc_col_name)]
    ].sum()

    # Add columns with the LC total counts (all LC types) per group (rows)
    df_percent_by_group[f"Total_{lc_col_name}"] = df_percent_by_group.sum(axis=1)

    # Transform each LC columns count to percentage, for each group
    df_percent_by_group[
        df_percent_by_group.columns[
            df_percent_by_group.columns.str.startswith(lc_col_name)
        ]
    ] = (
        df_percent_by_group[
            df_percent_by_group.columns[
                df_percent_by_group.columns.str.startswith(lc_col_name)
            ]
        ].div(df_percent_by_group[f"Total_{lc_col_name}"], axis=0)
        * 100.0
    )

    return df_percent_by_group


def get_lc_surface(
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    groupby=None,
    reshape_index: list[str] = None,
    reshape_columns: list[str] = None,
    round_dec: int = None,
    add_total: bool = True,
    area_col_name: str = "area",
    convert_factor: int = 1e6,
    slope_correction: bool = False,
    slope_col_name: str = "slope",
    area_corrected_col_name: str = "_slope_corrected",
):
    """
    Get surfaces by groups from a dataframes with an "area" column.
    Reshape outputs as (reshape_index, reshape_columns)
    - by defaut set as (groupby[:-1], groupby[-1] = "LC")

    A column of the total surface for each group combination can be added.

    :param df: Input dataframe
    :param groupby: List of columns name to groupby the dataframe
    :param reshape_index: List of name to set as index in the output
    :param reshape_columns: List of name to set as columns in the output
    :param round_dec: Number of decimal to  use in outputs
    :param add_total: Default True, add a column Total per combination of group
    :param area_col_name: Name of the area column in df
    :param convert_factor: Converting factor used to scale outputs (division). Default 1e6 (m->km)
    :param slope_correction: Default False - If True, use a column slope to correct surface
    :param slope_col_name: Name of the slope column to use (in df) if slope correction is asked
    :param area_corrected_col_name: Name of the slope corrected surface column to create

    :return: Dataframe with the surface per group
    """

    # Function
    # ========
    def change_lc_name_in_df(lc_name):
        return f"LC_{int(lc_name)}"

    # Slope correction - if needed
    # ============================
    if slope_correction is False:
        area_corrected_col_name = ""
    else:
        if f"{area_col_name}{area_corrected_col_name}" in list(df.columns):
            pass
        else:
            df.insert(
                loc=len(df.columns),
                column=f"{area_col_name}{area_corrected_col_name}",
                value=raster.get_area_slope_corrected(
                    df[area_col_name], df[slope_col_name], sum=False
                ),
            )

    # Get dataframe of surface per group
    # ==================================
    if groupby is None:
        groupby = ["Country", "glacier", "veget", "LC"]

    df_surface = (
        df.groupby(groupby)[f"{area_col_name}{area_corrected_col_name}"].sum()
        / convert_factor
    ).reset_index()  # transform multi index to columns

    # Reshape groups as row and landcover surface as columns
    # =====================================================
    if reshape_index is None:
        reshape_index = groupby[:-1]
    if reshape_columns is None:
        reshape_columns = groupby[-1]

    ds_surface_all_index = df_surface.pivot_table(
        index=reshape_index,
        columns=reshape_columns,
        values=f"{area_col_name}{area_corrected_col_name}",
    ).fillna(
        0
    )  # Replace NaN by zero

    # Change name of lc_col_name to convention
    # ========================================
    ds_surface_all_index.rename(
        columns={
            col: change_lc_name_in_df(col)
            for col in ds_surface_all_index.columns
            if isinstance(col, float) or isinstance(col, int)
        },
        inplace=True,
    )

    # Round numbers
    # =============
    if round_dec is not None:
        ds_surface_all_index = ds_surface_all_index.round(round_dec)

    # Add total surface column
    # ========================
    if add_total is True:
        ds_surface_all_index[f"Total_LC"] = ds_surface_all_index.sum(axis=1)

    return ds_surface_all_index


def plot_lc_map_and_hist(
    raster_file: str,
    gdf: gpd.GeoDataFrame,
    lc_percent: pd.DataFrame,
    lc_surface: pd.DataFrame,
    lc_map: LandCoverMap,
    title: str = None,
    bbox_anchor_legend: list[float, float] = [0.5, 0.1],
    basemap=True,
    gdf_outline: pd.DataFrame = None,
    hatch="//",
    save_file: str = None,
    return_ax: bool = False,
):
    """
    Scripts used for OSO Theia and Carto H1A plots

    :param raster_file:
    :param gdf:
    :param lc_percent:
    :param lc_surface:
    :param lc_map:
    :param title:
    :param bbox_anchor_legend:
    :param basemap:
    :param gdf_outline:
    :param hatch:
    :param save_file:
    :param return_ax:
    :return:
    """
    # Open raster + masking to land cover map
    # =======================================
    ds = rioxr.open_rasterio(raster_file, mask_and_scale=True)
    gdf.to_crs(ds.rio.crs, inplace=True)
    ds_glacier = ds.rio.clip(gdf.geometry.values, from_disk=True)
    if lc_map.codes_masked is not None:
        ds_glacier = raster.mask_raster(
            ds_glacier,
            val_to_mask=lc_map.codes_masked,
            mask_val=lc_map.mask_val,
        )

    # Projection and distance
    ds_glacier = ds_glacier.rio.reproject("EPSG:4326")
    distance_meters = raster.distance_meter_from_deg(ds_glacier)
    if gdf_outline is not None:
        gdf_outline.to_crs(ds_glacier.rio.crs, inplace=True)

    # ========
    #   PLOT
    # ========
    f, ax = plt.subplots(1, 2, figsize=(10, 8), gridspec_kw=dict(width_ratios=[1, 10]))
    f.subplots_adjust(wspace=0)

    # Maps
    # ====
    im = ds_glacier.isel(band=0).plot.imshow(
        ax=ax[1],
        cmap=lc_map.get_cmap(cmap_name="glacier"),
        norm=lc_map.get_norm(),
        alpha=0.7,
        add_colorbar=False,
    )

    # Settings
    if title is not None:
        ax[1].set_title(title, fontsize=14, weight="bold")

    lc_map_legend = lc_map.reindex(in_place=False)
    patches = [
        mpatches.Patch(
            color=lc_map_legend.get_colors()[i], label=lc_map_legend.get_type()[i]
        )
        for i in range(len(lc_map_legend.get_colors()))
    ]

    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set(xlabel="Longitude", ylabel="Latitude")
    ax[1].axis("equal")
    ax[1].legend(
        handles=patches,
        loc="center",
        borderaxespad=1.0,
        ncol=3,
        fontsize="small",
        bbox_to_anchor=bbox_anchor_legend,
    )
    ax[1].add_artist(ScaleBar(distance_meters, location="lower right"))

    # Background
    if basemap is True:
        cx.add_basemap(
            ax[1],
            crs=ds_glacier.rio.crs.to_string(),
            reset_extent=False,
            source=cx.providers.GeoportailFrance.orthos,
            zorder=-1,
        )

    # Contours of glacier
    if gdf_outline is not None:
        gdf_outline.plot(
            ax=ax[1],
            facecolor="None",
            edgecolor="k",
            lw=0.8,
            hatch=hatch,
        )

    # Histogram
    # ==========
    codes_list = [f"LC_{code}" for code in lc_map_legend.get_code()]
    lc_percent = lc_percent.drop("Total_LC", axis=1)[codes_list]
    lc_surface = lc_surface[codes_list]

    a0 = sbn.barplot(
        lc_percent, orient="h", palette=lc_map_legend.get_colors(), ax=ax[0]
    )

    labels = [lbl.replace(" ", "\n") for lbl in lc_map_legend.get_type()]
    percent_str = np.array([f"{a:.2f}% " for a in lc_percent.values[0]])
    percent_str[percent_str == "0.000% "] = ""
    percent_str = [
        f"{surf:.2f}" + " km2" + "\n\n (" + percent + ")"
        for (percent, surf) in zip(percent_str, lc_surface.values[0])
    ]
    for i in range(len(a0.containers)):
        a0.bar_label(
            a0.containers[i],
            labels=percent_str,
            label_type="edge",
            padding=3,
            fontsize=8,
            weight="bold",
        )

    a0.set_xlim([0, 80])
    a0.set_yticklabels(labels, rotation=45, fontsize=8, va="center")
    a0.get_xaxis().set_visible(False)
    a0.spines["bottom"].set_visible(False)
    a0.spines["top"].set_visible(False)
    a0.spines["right"].set_visible(False)

    plt.tight_layout()

    # Put it here after the layout otherwise will be destroyed
    limits = list(ds_glacier.rio.bounds())
    ax[1].set_xlim([limits[0], limits[2]])

    if save_file is not None:
        plt.savefig(save_file, dpi=300)

    if return_ax:
        return ax
