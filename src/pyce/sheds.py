import os
from typing import Union

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio.features
import rioxarray as rioxr
from geemap import shp_to_gdf
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder
from rioxarray import merge
from shapely.geometry import MultiPolygon, shape


# =====================================
#       General functions
# =====================================
def merge_rioxr(mnt_list: Union[list, str] = None, epsg: str = None, save: str = None):
    """
    Merge rasters with rioxarray

    :param mnt_list: List of files to merge.
    :param epsg: EPSG code to project the tiles to before merging in case the file
                 format do not contain this information. It must be the same for
                 all files.
    :return: A rioxarray rasters of the merged tiles
    """
    rasters = []

    if isinstance(mnt_list, str):
        mnt_list = [mnt_list]

    for mnt in mnt_list:
        rst = rioxr.open_rasterio(mnt, mask_and_scale=True)
        if epsg is not None:
            rst = rst.rio.write_crs(epsg)
        rasters.append(rst)

    if len(rasters) == 1:
        return rasters[0]

    merged_raster = merge.merge_arrays(rasters)

    if save is not None:
        merged_raster.rio.to_raster(save)

    return merged_raster


def shape_to_gdf(site_and_shapes, crs=None):
    """
    Transform dask outputs of run_pyshed to a GeoDataframe containing
    the list of sites/lakes with their shed geometry

    :param site_and_shapes: Output of run_pyshed, containing the site
                            and shed geometry
    :param crs: crs code to set on GeoDataFrame

    :return: GeoDataframe containing name and geometry shade
    """

    gdf_bv = gpd.GeoDataFrame(
        {"Lake": site_and_shapes[:, 0]}, geometry=site_and_shapes[:, 1]
    )

    if crs is not None:
        gdf_bv.set_crs(crs, inplace=True)

    return gdf_bv


# =====================================
#       pyshed function shortcut
# =====================================
def raster_shed_from_array(array: np.array, raster_like: Raster) -> Raster:
    """
    Transform a numpy array to a Raster object with the same p
    arameters as Raster_like input
    :param array:
    :param raster_like:
    :return:
    """

    return Raster(
        array,
        viewfinder=ViewFinder(
            shape=array.shape,
            affine=raster_like.affine,
            crs=raster_like.crs,
            nodata=raster_like.nodata,
        ),
    )


def raster_shed_from_rioxr(raster_rioxr: rioxr.raster_array, band=0) -> Raster:
    """
    Transform a rioxarray object to a Raster object of Pyshed module

    :param raster_rioxr: rioxarray
    :param band: band of the rioxarray to be transformed
    :return: Raster of Pyshed
    """

    array = raster_rioxr.isel(band=band).to_numpy()

    return Raster(
        array,
        viewfinder=ViewFinder(
            shape=array.shape,
            affine=raster_rioxr.rio.transform(),
            crs=pyproj.Proj(raster_rioxr.rio.crs.to_epsg()),
            nodata=raster_rioxr.rio.nodata,
        ),
    )


# =====================================
#       pyshed running functions
# =====================================
def raster_shed_processing(raster_shed: Raster):
    """
    Process a raster shed by filling pits, depressions and flats,
    return the flow direction

    :param raster_shed: Raster
    :return: dem, grid, fdir
        dem, Digital Elevation Model with Raster format
        grid, Grid object obtained from dem by pyshed
        fdir, Flow direction derived by pyshed
    """

    # TODO: ALLOW INPUT AS TIFF and direct example from pyshed documentation

    # Create pyshed objects
    dem = raster_shed.copy()
    grid = Grid.from_raster(raster_shed)

    # pits
    pit_filled_dem = grid.fill_pits(dem)

    # depressions
    flooded_dem = grid.fill_depressions(pit_filled_dem)

    # flats
    inflated_dem = grid.resolve_flats(flooded_dem)
    flats = grid.detect_flats(flooded_dem)

    # Flow and accumulation
    fdir = grid.flowdir(inflated_dem)

    return dem, grid, fdir, flats


def get_catchment(
    x: float,
    y: float,
    grid: Grid,
    fdir: Grid,
):
    """
    Return catchment using Pyshed

    :param x:
    :param y:
    :param grid:
    :param fdir:
    :return: catch

    """
    # Get accumulation map
    acc = grid.accumulation(fdir)

    # Snap pour point to high accumulation cell
    x_snap, y_snap = grid.snap_to_mask(acc > 100000, (x, y))

    # Catchment
    catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype="coordinate")

    return [catch, x_snap, y_snap]


def catch_to_shape(
    grid: Grid, catch: Raster, as_polygon=True
) -> Union[rasterio.features.shapes, shape]:
    """
    Transform catchment from Pyshed to a Multipolygon shape
    :param grid: Grid object from pyshed
    :param catch: Catchment object from pyshed
    :param as_polygon: Default True, return a Mulipolygon from shape,
           If False, return a list of tuple (geojson dict, value)
    :return: A shape Mulipolygon, except if 'as_polygon' is False
             then return a list of tuple (geojson dict, value)
    """
    # Clip to catchment
    grid.clip_to(catch)

    # Create view
    catch_view = grid.view(catch, dtype=np.uint8)

    # Create a vector representation of the catchment mask
    # It is a list of tuple (geojson dict, value)
    my_shape = [*grid.polygonize(catch_view)]

    # Transform geojson dict to shapely Polygon/Multipolygon
    if as_polygon:
        polys = list(map(lambda x: x[0], my_shape))
        if len(polys) > 1:
            my_shape = MultiPolygon([shape(p) for p in polys])
        else:
            my_shape = shape(polys[0])

    return my_shape


@dask.delayed
def run_pyshed(
    name: str,
    mnt: Union[str, list, pd.DataFrame],
    xy_outlet: list = None,
    df_metadata: dict = None,
    epsg_mnt: str = None,
    epsg_outlet: str = None,
    save_mnt_dir: str = None,
    dem_lake_flattening: bool = True,
    flattening_threshold: int = 1,
):
    # Create name of merged MNT if needed
    # ===================================
    if save_mnt_dir is not None:
        save_mnt_name = os.path.join(save_mnt_dir, f"{name}_merged.tif")
    else:
        save_mnt_name = None

    # Select /create mnt raster
    # =========================
    if isinstance(mnt, str) or isinstance(mnt, list):
        try:
            dem_rioxr = merge_rioxr(mnt, epsg=epsg_mnt, save=save_mnt_name)
        except Exception as e:
            print(e)

    elif isinstance(mnt, pd.DataFrame):
        metadata_col = [
            "col_name_dir",
            "col_name_tile",
            "col_name_x_outlet",
            "col_name_y_outlet",
            "tile_extension",
        ]
        if df_metadata is None:
            df_metadata = dict(
                col_name_dir="DIR_DALLE",
                col_name_tile="NOM_DALLE",
                col_name_x_outlet="X",
                col_name_y_outlet="Y",
                tile_extension=".asc",
            )

        if not all(pd.Series(df_metadata.keys()).isin(metadata_col)):
            raise IOError(
                f"df_metadata must contains the following keys: {metadata_col}"
            )

        for key, item in df_metadata.items():
            if key.startswith("col"):
                if not any(mnt.columns.isin([item])):
                    raise IOError(f"mnt dataframe column '{item}' not found")

        # Get list of tiles within dataframe
        list_mnt = [
            os.path.join(dir_tile, name_tile)
            for dir_tile, name_tile in zip(
                mnt[df_metadata["col_name_dir"]],
                mnt[df_metadata["col_name_tile"]] + df_metadata["tile_extension"],
            )
        ]

        # Merge the list of files
        dem_rioxr = merge_rioxr(list_mnt, epsg=epsg_mnt, save=save_mnt_name)

    # Get coordinates of the outlet
    # =============================
    if epsg_outlet is None:  # make sure epsg are the same between mnt and outlet
        print(
            Warning(f"EPSG_SHED is not set, it will be assumed to be equal to EPSG_MNT")
        )
        epsg_outlet = epsg_mnt

    if xy_outlet is None:
        xy_outlet = (
            gpd.GeoSeries.from_xy(
                mnt[df_metadata["col_name_x_outlet"]],
                mnt[df_metadata["col_name_y_outlet"]],
                crs=epsg_outlet,
            )
            .to_crs(epsg_mnt)
            .get_coordinates()
        )
        xy_outlet = [xy_outlet["x"].values[0], xy_outlet["y"].values[0]]

    # Ensure that altitude of lake around outlet points are the same
    # Ensure flats to be real flats for shed delimitation
    # Use a threshold to flatten data points
    # ===============================================================
    if dem_lake_flattening:
        alt_lake = dem_rioxr.sel(
            x=xy_outlet[0],
            y=xy_outlet[1],
            method="nearest",
        ).values[0]

        dem_rioxr = dem_rioxr.where(
            (dem_rioxr.data < alt_lake - flattening_threshold)
            | (dem_rioxr.data > alt_lake + flattening_threshold),
            other=alt_lake,
        )

    # Create the pyshed Raster Object from rioxarray dem
    # ==================================================
    raster_shed = raster_shed_from_rioxr(raster_rioxr=dem_rioxr)

    # Process the Raster object
    # =========================
    dem, grid, fdir, flats = raster_shed_processing(raster_shed=raster_shed)

    # Get the catchment
    # =================
    catch, x_snap, y_snap = get_catchment(
        x=xy_outlet[0],
        y=xy_outlet[1],
        grid=grid,
        fdir=fdir,
    )

    # Transform catchment to shape
    shp_polygon = catch_to_shape(grid=grid, catch=catch, as_polygon=True)

    return gpd.GeoDataFrame(
        {
            "Name": [name],
            "X outlet": [xy_outlet[0]],
            "Y outlet": [xy_outlet[1]],
            "X snap": x_snap,
            "Y snap": y_snap,
        },
        geometry=[shp_polygon],
        crs=epsg_mnt,
    )  # Data and geom as list not to provide an index
