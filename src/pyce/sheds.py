from typing import Union

import numpy as np
import pyproj
import rasterio.features
import rioxarray as rioxr
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder
from shapely.geometry import MultiPolygon, Polygon, shape


def rioxr_to_raster_shed(raster: rioxr.raster_array, band=0) -> Raster:

    array = raster.isel(band=band).to_numpy()

    return Raster(
        array,
        viewfinder=ViewFinder(
            shape=array.shape,
            affine=raster.rio.transform(),
            crs=pyproj.Proj(raster.rio.crs.to_epsg()),
            nodata=raster.rio.nodata,
        ),
    )


def get_shed(
    x: float,
    y: float,
    raster_shed: Raster,
    debug: bool = False,
):

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
    acc = grid.accumulation(fdir)

    # Catchment
    catch = grid.catchment(x=x, y=y, fdir=fdir, xytype="coordinate")

    # Debug
    debug_dict = {
        "pits": None,
        "depressions": None,
        "flats": None,
    }
    if debug:
        debug_dict["pits"] = grid.detect_pits(dem)
        debug_dict["depressions"] = grid.detect_depressions(pit_filled_dem)
        debug_dict["flats"] = grid.detect_flats(flooded_dem)

    return dem, grid, acc, catch, debug_dict


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
