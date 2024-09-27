# A.Guerou - 27.09.2024
# Codes to select and process MNT tiles from IGN
# Either RGE Alti or LidarHD

import glob
import os
from typing import List, Union

import dask
import geopandas as gpd
import pandas as pd
import requests
from IPython.display import display


# ====================================
#     GENERAL FUNCTIONS
# ===================================
def df_to_gdf(df: pd.DataFrame, x_col: str, y_col: str, crs: str):
    """
    Transform a dataframe containing x and y coordinates to a GeoDataFrame with points geometry
    :param df: dataframe
    :param x_col: x column name
    :param y_col: y column name
    :param crs: crs projection
    :return: geodataframe
    """
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[x_col], df[y_col])
    ).set_crs(crs)


# ====================================
#           RGE Alti functions
# ====================================
def identify_rge_tiles_from_coords(
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    rge_folder: str,
    buffer: int = 0,
    df_metadata: dict = None,
    rge_type: str = "RGEALTI_MNT_5M_ASC_LAMB93_IGN",
    rge_col_name_tiles: str = "NOM_DALLE",
    plot: bool = False,
):
    """
    Identify RGE Alti tiles to built DEM around locations within a certain buffer distance.
    Usually takes a dataframe with locations and add two columns with tiles names and directory.
    !!! The RGE alti data need to be in a local directory !!!

    :param df: Dataframe or GeoDataFrame containing the locations.
               Only points are accepted when DataFrame is given
    :param rge_folder:  Local folder containing RGE Alti data
    :param buffer: Buffer zone to consider in meters
    :param df_metadata: Only used if DataFrame is given
                        Metadata of x, y columns names in df, and crs value
                        Dict keys must be : "x"=..., "y"=..., "crs"=...
    :param rge_type: Folder name suffix of RGE Alti data containing the footprint shapes
    :param rge_col_name_tiles: column name of the RGE alti data footprint shape
                               containing the name of the tile
    :param plot: If True, plot the selection
    :return: GeoDataFrame of the original dataframe with tiles names and folder of RGE Alti data

    """
    # Creates GeoDataFrame with geometries as points
    # ----------------------------------------------
    if isinstance(df, pd.DataFrame):
        if df_metadata is None:
            df_metadata = dict(
                x="X Lambert 2 Centre", y="Y Lambert 2 Centre", crs="EPSG:27572"
            )
        gdf_coords = df_to_gdf(
            df,
            x_col=df[df_metadata["x"]],
            y_col=df[df_metadata["y"]],
            crs=df_metadata["crs"],
        )
    elif isinstance(df, gpd.GeoDataFrame):
        gdf_coords = df.copy()
    else:
        raise TypeError(
            f"df must be part of Union[pd.DataFrame, gpd.GeoDataFrame], got :{type(df)}"
        )

    # Get list of shape files from RGE Alti containing the footprint of the tiles
    # One per French department, as distributed on their website
    # https://geoservices.ign.fr/rgealti
    # -------------------------------------------------------
    rge_footprint_list = glob.glob(
        os.path.join(
            rge_folder,
            f"*/RGEALTI/3_SUPPLEMENTS_LIVRAISON*/{rge_type}*/dalles.shp",
        )
    )

    # Create GeoDataFrame with all tiles + geometry
    # ---------------------------------------------
    gdf_tiles = gpd.GeoDataFrame()
    for rge_footprint in rge_footprint_list:
        footprint = gpd.read_file(rge_footprint).to_crs(gdf_coords.crs, inplace=False)
        footprint["DIR_DALLE"] = os.path.dirname(rge_footprint).replace(
            "3_SUPPLEMENTS_LIVRAISON", "1_DONNEES_LIVRAISON"
        )
        gdf_tiles = pd.concat([gdf_tiles, footprint])

    gdf_tiles.drop_duplicates(subset=[rge_col_name_tiles], inplace=True)
    gdf_tiles["geometry_mnt"] = gdf_tiles["geometry"]

    #  Process new GeoDataFrame: add MNT tiles columns
    # -------------------------------------------------
    if any(gdf_coords.columns.isin([rge_col_name_tiles])):
        gdf_coords.drop(columns=[rge_col_name_tiles], inplace=True)

    # Create buffer zones
    if buffer != 0:
        gdf_coords["geometry"] = gdf_coords.buffer(buffer)

    # Get intersections between coords+buffer and all tiles
    gdf_sel = gdf_coords.sjoin(gdf_tiles, how="inner", predicate="intersects")
    gdf_sel = gdf_sel[
        gdf_coords.columns.union(
            [rge_col_name_tiles, "geometry_mnt", "DIR_DALLE"], sort=False
        )
    ]

    # Sort by first columns (usually the sites/lakes) and reset index
    gdf_sel.drop_duplicates(
        subset=[gdf_sel.columns[0], rge_col_name_tiles]
    ).reset_index(
        drop=True, inplace=True
    )  # Some tiles are contains in multiple regions
    gdf_sel.sort_values(gdf_sel.columns[0], inplace=True)
    gdf_sel.reset_index(drop=True, inplace=True)

    if plot:
        m = gdf_sel.explore()
        m = gdf_sel.set_geometry("geometry_mnt").explore(m=m, color="red")
        display(m)

    return gdf_sel.drop(columns=["geometry", "geometry_mnt"])


def identify_lidarhd_tiles_from_coords(
    df, gdf_lidar, buffer: int = 0, df_metadata: dict = None, plot: bool = False
):
    # Creates GeoDataFrame with geometries as points
    # ----------------------------------------------
    if isinstance(df, pd.DataFrame):
        if df_metadata is None:
            df_metadata = dict(
                x="X Lambert 2 Centre", y="Y Lambert 2 Centre", crs="EPSG:27572"
            )
        gdf_coords = df_to_gdf(
            df,
            x_col=df_metadata["x"],
            y_col=df_metadata["y"],
            crs=df_metadata["crs"],
        )
    elif isinstance(df, gpd.GeoDataFrame):
        gdf_coords = df.copy()
    else:
        raise TypeError(
            f"df must be part of Union[pd.DataFrame, gpd.GeoDataFrame], got :{type(df)}"
        )

    # Intersections gdf
    # -----------------
    if buffer != 0:
        gdf_coords["geometry"] = gdf_coords.buffer(buffer)

    # Copy lidar + geometry col with other name to keep it after sjoin
    if not isinstance(gdf_lidar, gpd.GeoDataFrame):
        raise TypeError(f"'gdf_lidar' must be a GeoDataFrame, got: {type(gdf_lidar)}")

    gdf_hd = gdf_lidar.copy()
    gdf_hd["geometry_mnt"] = gdf_hd["geometry"]

    # Intersection
    gdf_sel = gdf_coords.to_crs(gdf_hd.crs).sjoin(
        gdf_hd, how="inner", predicate="intersects"
    )
    gdf_sel.drop(columns=["index_right"], inplace=True)

    # Sort by first columns (usually the sites/lakes) and reset index
    gdf_sel.sort_values(gdf_sel.columns[0], inplace=True)
    gdf_sel.reset_index(drop=True, inplace=True)

    #             Checks
    # ----------------------------------------
    if plot:
        gdf_sel["geometry"] = gdf_sel.buffer(-buffer + 5)
        m = gdf_sel.explore()
        m = gdf_sel.set_geometry("geometry_mnt").explore(m=m, color="red")
        display(m)

    return gdf_sel[df.columns.union(gdf_lidar.columns[gdf_lidar.columns != "geometry"])]


def download_lidarhd_tiles(
    df: pd.DataFrame, dwnld_dir: str = None, num_workers=6, force=False, compute=True
):
    @dask.delayed
    def _request_url(url, path_out):
        r = requests.get(url, allow_redirects=True)
        open(path_out, "wb").write(r.content)
        print(f"Download done: {url}")

    # Checks
    if dwnld_dir is None and compute is True:
        raise ValueError(f"dwnld_dir must be specified")

    # Drop duplicates
    df_to_dwnld = df.drop_duplicates(subset=["url_telech"])

    # Download with dask
    request_list: list[None] = []

    for n_tile in range(df_to_dwnld["url_telech"].shape[0]):
        name_tile = df_to_dwnld["nom_pkk"].iloc[n_tile]
        url_tile = df_to_dwnld["url_telech"].iloc[n_tile]
        tile_path = os.path.join(dwnld_dir, name_tile)

        if not os.path.exists(tile_path) or force is True:
            request_list.append(_request_url(url=url_tile, path_out=tile_path))

    if compute:
        dask.compute(request_list, num_workers=num_workers)
