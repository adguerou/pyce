import glob
import os
from typing import Union

import contextily as cx
import geopandas as gpd
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import rioxarray as rioxr
import seaborn as sbn
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib_scalebar.scalebar import ScaleBar
from pyce.tools.lc_mapping import LandCoverMap
from pyproj import Proj, transform

from pyce_scripts import lc_distrib


def get_stats_surface(
    lakes_list: list,
    data_dir: Union[str, list[str]],
    save_dir: str,
    lcmap: LandCoverMap,
    metadata: dict = None,
    save: bool = False,
):
    # Create dataframe for csv saving
    stat_shp = pd.DataFrame()

    # State parameters if not set
    if metadata is None:
        metadata = {
            "shed_ext": "sheds*",
            "lake_ext": "lakes*",
            "landcover_ext": "landcoverS2_2017_2023",
            "dem_ext": "",
            "coords_ext": "coords*",
        }

    # Ensure list / even of one str
    if isinstance(data_dir, str):
        data_dir = [data_dir]

    # Statistics over each lake
    # =========================
    for lake in lakes_list:
        print(lake)

        # Get sheds data
        shed_file = glob.glob(
            os.path.join(
                save_dir,
                f"lake_by_lake/{lake}/lacsSentinelles_{metadata['shed_ext']}_{lake}.shp",
            )
        )
        shed_shp = gpd.read_file(shed_file[0])

        # Get lake data
        lake_file = glob.glob(
            os.path.join(
                save_dir,
                f"lake_by_lake/{lake}/lacsSentinelles_{metadata['lake_ext']}_{lake}.shp",
            )
        )
        lake_shp = gpd.read_file(lake_file[0])

        # Get variables
        shed_area = shed_shp.area / 1e4
        lake_area = lake_shp.loc[lake_shp["lake_name"] == lake].area / 1e4
        ratio_shed = shed_area / lake_area

        # Get altitude lake/ max/ median
        mnt_flag = 0
        for d_dir in data_dir:
            try:
                mnt_lake = rioxr.open_rasterio(
                    os.path.join(d_dir, f"dem/dem_1m/{lake}.tif")
                )
                mnt_flag += 1
            except IOError:
                pass

        if mnt_flag == 0:
            print(f"DEM not found in 'data_dir' for {lake}")
            continue

        # lake / get coordinates / ensure to read all files in delivery
        coords_files = glob.glob(
            os.path.join(
                save_dir,
                f"ancillary/lacsSentinelles_{metadata['coords_ext']}.csv",
            )
        )
        coords = pd.DataFrame()
        for f in coords_files:
            cf = pd.read_csv(f)
            coords = pd.concat([coords, cf])

        coords_lake = coords.loc[coords["lake_name"] == lake]

        alt_lake = mnt_lake.sel(
            x=coords_lake["x_center"].values[0],
            y=coords_lake["y_center"].values[0],
            method="nearest",
        ).values[0]

        # max/min
        mnt_lake_clip = mnt_lake.rio.clip(
            shed_shp.to_crs(mnt_lake.rio.crs).geometry, drop=True
        )
        alt_max = mnt_lake_clip.max()
        alt_median = mnt_lake_clip.median()

        # Get landcover data
        # ==================
        lc_shp = gpd.read_file(
            os.path.join(
                save_dir,
                f"lake_by_lake/{lake}/lacsSentinelles_{metadata['landcover_ext']}_{lake}.shp",
            )
        )
        lc_shp = lc_shp.loc[lc_shp.LC >= 0]  # Filter no data

        # Get landcover stats
        # ===================
        lc_shp["area"] = lc_shp.area
        stat_lake = (
            lc_shp.groupby(["LC"])["area"]
            .sum()
            .reset_index()
            .pivot_table(columns="LC", values="area")
            / 1e4  # For Ha
        )  # area of each class / reshape table
        stat_lake["lake_name"] = lake

        # Fill with lake and sheds values
        stat_lake["lake_area [ha]"] = lake_area.values[0]
        stat_lake["shed_area [ha]"] = shed_area.values[0]
        stat_lake["ratio_shed_to_lake"] = ratio_shed.values[0]
        stat_lake["alt_lake [m]"] = alt_lake
        stat_lake["alt_max [m]"] = alt_max.data
        stat_lake["alt_median [m]"] = alt_median.data

        # Concat all lakes
        stat_shp = pd.concat([stat_shp, stat_lake])

    # Edit dataframe
    stat_shp = (
        stat_shp.set_index("lake_name").sort_index().fillna(0)
    )  # fill nan with 0 / reset index to lake names

    stat_shp = stat_shp[
        [
            "lake_area [ha]",
            "shed_area [ha]",
            "ratio_shed_to_lake",
            "alt_lake [m]",
            "alt_max [m]",
            "alt_median [m]",
        ]
        + list(lcmap.get_code())
    ]  # reorder columns to lcmap order

    stat_shp = stat_shp.rename(
        columns={
            0: "LC_0",
            1: "LC_1",
            6: "LC_6",
            2: "LC_2",
            3: "LC_3",
            4: "LC_4",
            5: "LC_5",
        }
    )

    # Save csv file
    if save:
        stat_shp.to_csv(
            path_or_buf=os.path.join(
                save_dir,
                "statistics/lacsSentinelles_statistics.csv",
            ),
            header=[
                "lake_area [ha]",
                "shed_area [ha]",
                "ratio_shed_to_lake",
                "alt_lake [m]",
                "alt_max [m]",
                "alt_median [m]",
            ]
            + [f"{type_ha} [ha]" for type_ha in lcmap.get_type()],
            float_format="%.1f",
        )
    return stat_shp


def get_stats_percent(stat_shp, lcmap, save_dir):
    stat_shp_percent = stat_shp.reset_index()
    stat_shp_percent.index.name = "index"

    stat_shp_percent = lc_distrib.get_lc_percent(
        stat_shp_percent, groupby=["index"], lc_col_name="LC"
    )

    stat_shp_percent["lake_name"] = stat_shp.reset_index()["lake_name"]
    stat_shp_percent["lake_area [ha]"] = stat_shp.reset_index()["lake_area [ha]"]
    stat_shp_percent["shed_area [ha]"] = stat_shp.reset_index()["shed_area [ha]"]
    stat_shp_percent["ratio_shed_to_lake"] = stat_shp.reset_index()[
        "ratio_shed_to_lake"
    ]
    stat_shp_percent["alt_lake [m]"] = stat_shp.reset_index()["alt_lake [m]"]
    stat_shp_percent["alt_max [m]"] = stat_shp.reset_index()["alt_max [m]"]
    stat_shp_percent["alt_median [m]"] = stat_shp.reset_index()["alt_median [m]"]
    stat_shp_percent.set_index(["lake_name"], inplace=True)

    stat_shp_percent = stat_shp_percent[
        [
            "lake_area [ha]",
            "shed_area [ha]",
            "ratio_shed_to_lake",
            "alt_lake [m]",
            "alt_max [m]",
            "alt_median [m]",
        ]
        + [f"LC_{code}" for code in list(lcmap.get_code())]
    ]  # reorder columns to lcmap order

    if save_dir is not None:
        stat_shp_percent.to_csv(
            path_or_buf=os.path.join(
                save_dir,
                "statistics/lacsSentinelles_statistics_percent.csv",
            ),
            header=[
                "lake_area [ha]",
                "shed_area [ha]",
                "ratio_shed_to_lake",
                "alt_lake [m]",
                "alt_max [m]",
                "alt_median [m]",
            ]
            + [f"{type_lcmap} [%]" for type_lcmap in lcmap.get_type()],
            float_format="%.1f",
        )
    return stat_shp_percent


def plot_landcover(
    stat_file: str,
    lcmap: LandCoverMap,
    data_dir: str,
    save_fig: str = None,
    metadata: dict = None,
):
    stat = pd.read_csv(stat_file)
    lakes_list = list(set(stat["lake_name"]))

    # State parameters if not set
    if metadata is None:
        metadata = {
            "shed_ext": "sheds*",
            "landcover_ext": "landcoverS2_2017_2023",
        }

    for lake in lakes_list:
        # SHED
        shed_file = glob.glob(
            os.path.join(
                data_dir,
                f"lake_by_lake/{lake}/lacsSentinelles_{metadata['shed_ext']}_{lake}.shp",
            )
        )
        shed = gpd.read_file(shed_file[0])

        # LANDCOVER
        lc_shp = gpd.read_file(
            os.path.join(
                data_dir,
                f"lake_by_lake/{lake}/lacsSentinelles_{metadata['landcover_ext']}_{lake}.shp",
            )
        )

        # Derive surface from shp instead of csv to plot with seaborn
        lc_shp["area"] = lc_shp.area / 1e4
        lc_shp_area = lc_shp.groupby(["LC"])["area"].sum().reset_index()

        # ===============
        #      Plot
        # ===============
        f, ax = plt.subplots(
            1, 2, figsize=(8, 7), gridspec_kw=dict(width_ratios=[1, 15])
        )

        # Histograms
        # ==========
        g = sbn.barplot(
            data=lc_shp_area,
            x="area",
            y="LC",
            orient="h",
            order=list(lcmap.get_code()),
            palette=list(lcmap.get_colors()),
            legend=False,
            errorbar=None,
            ax=ax[0],
        )

        # Settings
        f.suptitle(lake.upper(), fontsize=14)
        g.set_yticklabels(list(lcmap.get_type()), fontsize=10, rotation=45)
        g.axes.set_ylabel("")

        for i in g.containers:
            labels = [
                f"{(v.get_width()):.1f}\n({(v.get_width() / (lc_shp_area['area'].sum()) * 100):.1f}%)"
                for v in i
            ]
            g.bar_label(i, labels=labels, style="italic", padding=3)

        g.axes.set_xlabel("Surface [Ha]", fontsize=12, labelpad=10)
        g.spines["top"].set_visible(False)
        g.spines["right"].set_visible(False)

        #   Map
        # =========
        lcmap_map = lcmap.reindex_from_list(range(len(lcmap.df.index)))
        lcmap_map.df = lcmap_map.df.sort_values(
            "Code"
        )  # need increasing values of codes

        im_shp = lc_shp.plot(
            column="LC",
            cmap=lcmap_map.get_cmap(cmap_name="glacier"),
            vmin=lcmap.get_code().min(),  # needed to get correct colors
            vmax=lcmap.get_code().max(),  # needed to get correct colors
            ax=ax[1],
            alpha=0.9,
        )

        # Countours of BV
        # ---------------
        shed.plot(ax=ax[1], facecolor="None", edgecolor="k", ls="--", lw=1)

        # Text
        stat_lake = stat.loc[stat["lake_name"] == lake]
        ax[1].text(
            0.1,
            0.92,
            f"Lake area: {stat_lake['lake_area [ha]'].values[0]:.2f} ha\n"
            f"Shed area: {stat_lake['shed_area [ha]'].values[0]:.2f} ha\n"
            f"Ratio: {stat_lake['ratio_shed_to_lake'].values[0]:.0f}",
            fontsize="medium",
            transform=ax[1].transAxes,
            va="center",
            bbox=dict(boxstyle="round", ec="black", fc="lightgrey"),
        )

        ax[1].text(
            0.55,
            0.92,
            f"Lake altitude: {stat_lake['alt_lake [m]'].values[0]:.0f} m\n"
            f"Max altitude: {stat_lake['alt_max [m]'].values[0]:.0f} m\n"
            f"Median altitude: {stat_lake['alt_median [m]'].values[0]:.0f} m",
            fontsize="medium",
            transform=ax[1].transAxes,
            va="center",
            bbox=dict(boxstyle="round", ec="black", fc="lightgrey"),
        )

        # Settings
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()
        ax[1].set(xlabel="Longitude", ylabel="Latitude")
        ax[1].axis("equal")

        # Legend
        patches = [
            mpatches.Patch(color=lcmap.get_colors()[i], label=lcmap.get_type()[i])
            for i in range(len(lcmap.get_colors()))
        ]
        ax[1].legend(
            handles=patches,
            loc="center",
            borderaxespad=1.0,
            ncol=3,
            fontsize="small",
            bbox_to_anchor=[0.35, 0.1],
            fancybox=True,
            framealpha=1,
        )

        # Scale bar
        ax[1].add_artist(
            ScaleBar(
                dx=1,
                location="lower right",
                pad=0.3,
                border_pad=3,
                frameon=True,
            )
        )

        # Extent
        extent_lc = lc_shp.total_bounds
        ax[1].set_xlim([extent_lc[0] - 500, extent_lc[2] + 500])
        ax[1].set_ylim([extent_lc[1] - 500, extent_lc[3] + 500])

        # Girds and labels
        def convert_to_4326(x, y, in_proj, out_proj=Proj("epsg:4326")):
            return transform(in_proj, out_proj, x, y)

        ax[1].xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax[1].yaxis.set_major_locator(ticker.MaxNLocator(4))

        xticks_loc = ax[1].get_xticks().tolist()
        yticks_loc = ax[1].get_yticks().tolist()

        # to ensure the same number of ticks on each axes
        min_ticks_number = np.min([len(xticks_loc), len(yticks_loc)])
        xticks_loc = xticks_loc[:min_ticks_number]
        yticks_loc = yticks_loc[:min_ticks_number]

        ax[1].xaxis.set_major_locator(ticker.FixedLocator(xticks_loc))
        ax[1].yaxis.set_major_locator(ticker.FixedLocator(yticks_loc))

        x_4326, y_4326 = convert_to_4326(
            xticks_loc, yticks_loc, in_proj=Proj(lc_shp.crs.to_string())
        )

        ax[1].set_xticklabels(["{:.2f}".format(x) for x in x_4326])
        ax[1].set_yticklabels(["{:.2f}".format(y) for y in y_4326])

        # Background
        cx.add_basemap(
            ax[1],
            crs=lc_shp.crs.to_string(),
            reset_extent=True,
            source=cx.providers.Esri.WorldImagery,  # IGN does not work anymore  - 14th November 2024
            zorder=-1,
            attribution_size=6,
        )

        plt.tight_layout()

        # =========
        # Save figure
        # =========
        if save_fig:
            plt.savefig(
                os.path.join(
                    save_fig,
                    f"lake_by_lake/{lake}/lacsSentinelles_{lake}.png",
                ),
                dpi=200,
            )
            plt.savefig(
                os.path.join(save_fig, f"figures/lacsSentinelles_{lake}.png"),
                dpi=200,
            )
