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
    shp_file: Union[str, gpd.GeoDataFrame],
    groupby: list[str],
    lc_col_name: str = "LC",
):

    # Open shp file + remove corrupted item with no geometry
    if isinstance(shp_file, str):
        gdf = gpd.read_file(shp_file)
        gdf = gdf.iloc[np.where(gdf["geometry"].values != None)]
    elif isinstance(shp_file, gpd.GeoDataFrame):
        gdf = shp_file
    else:
        raise TypeError("'shp_file' must be either a str or a GeoDataFrame")

    # Get a multi_index dataframe (groupby) with each colum being
    # the LC sum values per group for each LC class
    gdf_percent_by_group = gdf.groupby(groupby)[
        gdf.columns[gdf.columns.str.startswith(lc_col_name)]
    ].sum()

    # Add columns with the LC total counts (all LC types) per group (rows)
    gdf_percent_by_group[f"Total_{lc_col_name}"] = gdf_percent_by_group.sum(axis=1)

    # Transform each LC columns count to percentage, for each group
    gdf_percent_by_group[
        gdf_percent_by_group.columns[
            gdf_percent_by_group.columns.str.startswith(lc_col_name)
        ]
    ] = (
        gdf_percent_by_group[
            gdf_percent_by_group.columns[
                gdf_percent_by_group.columns.str.startswith(lc_col_name)
            ]
        ].div(gdf_percent_by_group[f"Total_{lc_col_name}"], axis=0)
        * 100.0
    )

    return gdf_percent_by_group


def plot_lc_map_and_hist(
    raster_file: str,
    gdf: gpd.GeoDataFrame,
    lc_percent: pd.DataFrame,
    lc_map: LandCoverMap,
    title: str = None,
    bbox_anchor_legend: list[float, float] = [0.5, 0.1],
    basemap=True,
    gdf_outline: pd.DataFrame = None,
    hatch="//",
    save_file: str = None,
    return_ax: bool = False,
):

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

    a0 = sbn.barplot(
        lc_percent, orient="h", palette=lc_map_legend.get_colors(), ax=ax[0]
    )

    labels = [lbl.replace(" ", "\n") for lbl in lc_map_legend.get_type()]
    percent_str = np.array([f"{a:.3f}% " for a in lc_percent.values[0]])
    percent_str[percent_str == "0.000% "] = ""
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
