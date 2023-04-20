from typing import Union

import numpy as np
import pyproj
import rasterio.features
import rioxarray as rioxr
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder
from shapely.geometry import MultiPolygon, Polygon, shape


def array_to_raster_shed(array: np.array, raster_like: Raster) -> Raster:

    """
    Transform a numpy array to a Raster object with the same parameters as Raster_like input
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
    Transform a rioxarray object to a Raster object of Pyshed moduel

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


def raster_shed_formatting(raster_shed: Raster):

    """
    Format a raster shed by filling pits, depressions and flats, return the flow direction

    :param raster_shed: Raster
    :return: dem, grid, fdir
        dem, Digital Elevation Model with Raster format
        grid, Grid object obtained from dem by pyshed
        fdir, Flow direction derived by pyshed
    """

    # TODO: ALLOW INPUT AS TIFF and direct exemple from pyshed documentation

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
    # Catchment
    catch = grid.catchment(x=x, y=y, fdir=fdir, xytype="coordinate")

    return catch


def catch_to_shape(
    grid: Grid, catch: Raster, as_polygon=True
) -> Union[rasterio.features.shapes, shape]:

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
