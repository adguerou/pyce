import os

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rioxr
import seaborn as sbn
import xarray as xr
from matplotlib import pyplot as plt
from pyce.shape import concat, str_to_datetime
from pyce.tools.lc_mapping import LandCoverMap, oso_mapping_fusion_in_df


def fusion_oso_shape(
    list_shapes: list[str],
    lc_map_to: LandCoverMap,
    lc_map_from: LandCoverMap,
    save: str = None,
):
    gdf = concat(list_shapes, ignore_index=False)

    gdf = str_to_datetime(gdf, column="datetime")
    gdf.set_index("datetime", inplace=True)

    gdf = oso_mapping_fusion_in_df(gdf, lc_map_to=lc_map_to, lc_map_from=lc_map_from)

    if save:
        gdf.to_file(save)

    return gdf


@dask.delayed
def _dask_confidence_df(
    glacier: gpd.GeoDataFrame, oso_rst: str, oso_conf: str, lcmap: LandCoverMap
):
    ds = rioxr.open_rasterio(oso_rst, mask_and_scale=True)
    ds_glacier = ds.rio.clip([glacier.geometry], from_disk=True)

    ds_c = rioxr.open_rasterio(oso_conf, mask_and_scale=True)
    ds_c_glacier = ds_c.rio.clip([glacier.geometry], from_disk=True)

    df = pd.DataFrame()

    for code in lcmap.get_code():
        # Get a DataFrame with the confidence values for each class
        mask = ds_c_glacier.where(ds_glacier == code)
        data = mask.data[0][~np.isnan(mask.data[0])]
        df_tmp = pd.DataFrame(
            {
                "LC": np.ones_like(data) * code,
                "conf": data,
                "Massif": glacier.Massif,
                "Nom": glacier.Nom,
            }
        )
        df = pd.concat([df, df_tmp])

    return df


def get_confidence_df(
    gdf: gpd.GeoDataFrame, oso_rst: str, oso_conf: str, lcmap: LandCoverMap
):
    result = []
    for glacier in gdf.itertuples():
        result.append(_dask_confidence_df(glacier, oso_rst, oso_conf, lcmap))

    df_list = dask.compute(result)

    df_conf = pd.DataFrame()
    for df in df_list[0]:
        df_conf = pd.concat([df_conf, df])

    if df_conf.empty:
        raise ValueError(
            "Result dataframe is empty - please check coherence between rasters and LandCoverMap codes"
        )
    return df_conf


def plot_confidence_ridge(df_conf, lcmap, title=None):
    sbn.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    g = sbn.FacetGrid(
        df_conf,
        row="LC",
        hue="LC",
        aspect=10,
        height=0.8,
        palette=list(lcmap.get_colors()),
        row_order=list(lcmap.get_code()),
        hue_order=list(lcmap.get_code()),
    )
    g.map(
        sbn.kdeplot,
        "conf",
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
        common_norm=False,
    )
    g.map(sbn.kdeplot, "conf", clip_on=False, color="w", lw=2, common_norm=False)
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    for ax, index in zip(g.axes.flatten(), range(len(g.axes))):
        ax.text(
            0,
            0.2,
            lcmap.get_type()[index],
            fontweight="bold",
            color=lcmap.get_colors()[index],
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.figure.subplots_adjust(hspace=-0.25)
    g.set_titles("")
    g.fig.suptitle(title)
    g.set(
        xlabel="OSO Confidence [%]",
        ylabel="",
        yticks=[],
        xticks=[0, 20, 40, 60, 80, 100],
        xlim=[0, 110],
    )
    g.despine(bottom=True, left=True)

    return g


def run_confidence(
    gdf,
    years,
    mapping_files,
    mapping_names,
    oso_rasters,
    oso_confs,
    title="French Alps",
    fig_dir="",
    save_name=None,
):
    for yr, index in zip(years, range(len(years))):
        lcmap = LandCoverMap(mapping_files[index], name=mapping_names[index])
        lcmap.reindex(in_place=True)
        lcmap.remove_item(col_name="Code", col_val=[lcmap.mask_val], in_place=True)

        df_conf = get_confidence_df(gdf, oso_rasters[index], oso_confs[index], lcmap)
        g = plot_confidence_ridge(df_conf, lcmap, title=title + " - " + str(yr))

        if save_name is not None:
            plt.savefig(
                os.path.join(fig_dir, f"OSO_LC_confidence_{save_name}_{yr}.png"),
                dpi=200,
            )


def run_confidence_massif(
    gdf,
    years,
    mapping_files,
    mapping_names,
    oso_rasters,
    oso_confs,
    fig_dir="",
    save_name=None,
):
    list_massif = [str(massif) for massif in gdf["Massif"].dropna().unique()]
    list_massif_save = [str(massif).replace(" ", "_") for massif in list_massif]

    for massif, massif_save in zip(list_massif, list_massif_save):
        gdf_massif = gdf[gdf["Massif"] == massif]

        if save_name is not None:
            name = save_name + "_" + massif_save
        else:
            name = save_name

        run_confidence(
            gdf_massif,
            years,
            mapping_files,
            mapping_names,
            oso_rasters,
            oso_confs,
            title=massif,
            fig_dir=fig_dir,
            save_name=name,
        )


@dask.delayed(nout=2)
def _dask_confidence_df_and_mask(glacier, oso_rst, oso_conf, lcmap):
    ds = rioxr.open_rasterio(oso_rst, mask_and_scale=True)
    ds_glacier = ds.rio.clip([glacier.geometry], from_disk=True)

    ds_c = rioxr.open_rasterio(oso_conf, mask_and_scale=True)
    ds_c_glacier = ds_c.rio.clip([glacier.geometry], from_disk=True)

    ds_mask = xr.Dataset()
    df = pd.DataFrame()

    for code in lcmap.get_code():
        # Get a dataset with each variable the confidence per class
        mask = ds_c_glacier.where(ds_glacier == code)
        ds_mask_tmp = mask.to_dataset(dim="band").rename_vars({1: code})
        ds_mask = xr.merge([ds_mask, ds_mask_tmp])

        # Get a DataFrame with the confidence values for each class
        data = mask.data[0][~np.isnan(mask.data[0])]
        df_tmp = pd.DataFrame(
            {
                "LC": np.ones_like(data) * code,
                "conf": data,
                "Massif": glacier.Massif,
                "Nom": glacier.Nom,
            }
        )
        df = pd.concat([df, df_tmp])

    return df, ds_mask


def get_confidence_mask(
    gdf,
    rst_lc,
    rst_conf,
    lcmap,
):
    result = []

    for glacier in gdf.itertuples():
        result.append(_dask_confidence_df_and_mask(glacier, rst_lc, rst_conf, lcmap))

    result = dask.compute(result)

    return result[0]


def plot_conf_mask(ds, gdf_outline, code, title="", vmin=None, vmax=None):
    ds_glacier = ds.rio.reproject("EPSG:4326", nodata=np.nan)
    gdf = gdf_outline.to_crs(ds_glacier.rio.crs)

    f, ax = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw=dict(height_ratios=[1, 10]))
    f.subplots_adjust(wspace=0)

    ds_glacier[code].plot.hist(
        ax=ax[0], bins=np.linspace(20, 100, 17), orientation="vertical"
    )
    ax[0].spines[["right", "top"]].set_visible(False)
    ax[0].set_xlabel("")

    ax[0].set_title(title)

    ds_glacier[code].plot(
        ax=ax[1],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar_kwargs={
            "orientation": "horizontal",
            "location": "top",
            "pad": 0.01,
            "aspect": 50,
            "label": "confidence [%]",
        },
    )
    gdf.plot(ax=ax[1], facecolor="None", edgecolor="k")

    ax[1].set(xlabel="Longitude", ylabel="Latitude")
    ax[1].axis("equal")
    ax[1].set_title("")

    plt.tight_layout()


def run_conf_mask(
    gdf,
    oso_rst,
    oso_conf,
    year,
    lcmap,
    code,
    vmin=None,
    vmax=None,
):
    list_and_mask = get_confidence_mask(gdf, oso_rst, oso_conf, lcmap)

    title = (
        gdf.Nom.values[0]
        + " - "
        + str(year)
        + " - "
        + lcmap.get_type()[lcmap.get_code() == code][0].upper()
    )

    plot_conf_mask(
        list_and_mask[0][1], gdf, code=code, title=title, vmin=vmin, vmax=vmax
    )

    return list_and_mask
