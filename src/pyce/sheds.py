import os
from typing import Union

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio.features
import rioxarray as rioxr
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder
from rioxarray import merge
from shapely.geometry import MultiPolygon, shape


# =====================================
#       DEM raster processings
# =====================================
def merge_rioxr_dem_from_df(
    df: pd.DataFrame = None,
    mnt_dir: str = None,
    col_name: str = "NOM_DALLE",
    epsg: str = "EPSG:2154",
):
    """
    Merge rasters list from a dataframe.
    Usefull with IGN dataframes built to list tiles of different sites

    :param df: pandas dataframe containing list of tiles to merge
    :param mnt_dir: directory where the tiles are stored
    :param col_name: Name of the dataframe columns containing the tiles names
    :param epsg: EPSG code to project the tiles to before merging.
                 The original files must be in this system. No projection are done
    :return: A rioxarray rasters of the merged tiles
    """
    rasters = []
    if mnt_dir is None:
        mnt_root_dir = "./"
    else:
        mnt_root_dir = mnt_dir

    for index, row in df.iterrows():
        rst = rioxr.open_rasterio(
            os.path.join(mnt_root_dir, row[col_name] + ".asc"),
            mask_and_scale=True,
        )
        rst.rio.write_crs(epsg, inplace=True)
        rasters.append(rst)

    return merge.merge_arrays(rasters)


# =====================================
#       pyshed function shortcut
# =====================================
def array_to_raster_shed(array: np.array, raster_like: Raster) -> Raster:
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


def rioxr_to_raster_shed(raster_rioxr: rioxr.raster_array, band=0) -> Raster:
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

    # Flow and accumulation
    fdir = grid.flowdir(inflated_dem)

    return dem, grid, fdir


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
    x_snap, y_snap = grid.snap_to_mask(acc > 1000, (x, y))

    # Catchment
    catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype="coordinate")

    return catch


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


def shape_to_gdf(shapes_and_sites, crs=None):
    """
    Transform dask outputs of run_pyshed to GeoDataframe

    :param shapes_and_sites: output of run_pyshed
    :param crs: crs code to set on GeoDataFrame
    :return: GeoDataframe containing name and geometry shade

    """
    shps, sites = (
        np.array(shapes_and_sites[0])[:, 0],
        np.array(shapes_and_sites[0])[:, 1],
    )

    gdf_bv = gpd.GeoDataFrame({"Lac": sites}, geometry=shps)

    if crs is not None:
        gdf_bv.set_crs(crs, inplace=True)

    return gdf_bv


@dask.delayed
def run_pyshed(
    df,
    my_site,
    mnt_dir: str = None,
    col_name_site: str = "Lac",
    col_name_mnt: str = "NOM_DALLE",
    col_name_x: str = "X",
    col_name_y: str = "Y",
    epsg_mnt="EPSG:2154",
    epsg_sites="EPSG:27572",
):
    # Select site rows in the dataframe
    df_site = df.loc[df[col_name_site] == my_site]
    dem_rioxr = merge_rioxr_dem_from_df(
        df_site, mnt_dir=mnt_dir, col_name=col_name_mnt, epsg=epsg_mnt
    )

    # Create the pyshed Raster Object from rioxarray dem
    raster_shed = rioxr_to_raster_shed(raster_rioxr=dem_rioxr)

    # Process the Raster object
    dem, grid, fdir = raster_shed_processing(raster_shed=raster_shed)

    # Make sure coordinates are aligned between MNT and sites df
    sites_coords = (
        gpd.GeoSeries.from_xy(
            df_site[col_name_x],
            df_site[col_name_y],
            crs=epsg_sites,
        )
        .to_crs(epsg_mnt)
        .get_coordinates()
    )

    # Get the catchment
    catch = get_catchment(
        x=sites_coords["x"].values[0],
        y=sites_coords["y"].values[0],
        grid=grid,
        fdir=fdir,
    )

    # Transform catchment to shape
    shp_polygon = catch_to_shape(grid=grid, catch=catch, as_polygon=True)

    return [shp_polygon, my_site]
