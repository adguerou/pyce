import os
from typing import Union

import dask
import geopandas as gpd
import numpy as np
import pyproj
import rioxarray as rioxr
import xarray as xr
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder
from shapely.geometry import MultiPolygon, Point, shape
from shapely.geometry.base import BaseGeometry

from pyce.raster import polygonize_raster
from pyce.shape import fill_geom, select_poly_from_multipoly


# =====================================
#       General functions
# =====================================
# TODO: change name and update usage
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


def raster_shed_to_geom(
    grid: Grid, raster: Raster, as_geom=True, dtype: type = np.uint8
) -> Union[tuple[dict, float], BaseGeometry]:
    """
    Transform catchment from Pyshed to a Multipolygon shape
    :param grid: Grid object from pyshed
    :param raster: Raster object from pyshed
    :param as_geom: Default True, return a BaseGeometry,
           If False, return a list of tuple (geojson dict, value)
    :return: BaseGeometry, except if 'as_polygon' is False
             then return a list of tuple (geojson dict, value)
    """
    # Clip to catchment
    grid.clip_to(raster)

    # Create view
    catch_view = grid.view(raster, dtype=dtype)

    # Create a vector representation of the catchment mask
    # It is a list of tuple (geojson dict, value)
    geom = [*grid.polygonize(catch_view)]

    # Transform geojson dict to shapely Polygon/Multipolygon
    if as_geom:
        polys = list(map(lambda x: x[0], geom))
        if len(polys) > 1:
            geom = MultiPolygon([shape(p) for p in polys])
        else:
            geom = shape(polys[0])

    return geom


def raster_shed_to_rioxr(raster: Raster, crs=None, dtype="float32") -> xr.DataArray:
    return (
        xr.DataArray(
            raster,
            dims=["y", "x"],
            coords={
                "y": raster.coords[:, 0].reshape(raster.shape)[:, 0],
                "x": raster.coords[:, 1].reshape(raster.shape)[0],
            },
        )
        .astype(dtype)
        .rio.write_nodata(raster.nodata)
        .rio.write_transform(raster.affine)
        .rio.write_crs(crs)
    )


# =====================================
#       pyshed running functions
# =====================================
def get_lake_shape(
    dem: Union[str, xr.DataArray],
    xy_lake,
    flattening_thresh=0.3,
    buffer_size: float = 8,
    bufffer_delta: float = 1.5,
    crs_dem: str = None,
    crs_lake: str = None,
    as_gdf=False,
    name_lake: str = "lake",
    save_name: str = None,
):
    # Get dem
    # =======
    if isinstance(dem, str):
        dem = rioxr.open_rasterio(dem, mask_and_scale=True)
    if not isinstance(dem, xr.DataArray):
        raise TypeError(f"'dem' type must either str or xr.DataArray. Got {type(dem)}")

    # Check projection
    # ================
    if crs_dem is None:
        crs_dem = dem.rio.crs

    elif crs_dem != dem.rio.crs:
        raise ValueError(
            f"'dem' has a different projection ({dem.rio.crs}) than 'crs_dem'({crs_dem})"
        )

    if crs_lake is None:
        crs_lake = crs_dem
        print(
            Warning(f"EPSG_LAKE is not set, it will be assumed to be equal to EPSG_DEM")
        )

    # Get lake coordinates into dem coordinates
    # ========================================
    xy_lake = (
        gpd.GeoSeries.from_xy([xy_lake[0]], [xy_lake[1]], crs=crs_lake)
        .to_crs(crs_dem)
        .get_coordinates()
    )
    xy_lake = [xy_lake["x"].values[0], xy_lake["y"].values[0]]

    # Lake selection
    # ==============
    altitude_lake = dem.sel(
        x=xy_lake[0],
        y=xy_lake[1],
        method="nearest",
    ).values[0]

    if np.isnan(altitude_lake):
        raise ValueError(
            f"'xy_lake' fell onto NaN value of dem. Please verify projections and/or dem data"
        )

    lake_sel = np.abs(dem.data - altitude_lake) < flattening_thresh
    dem_lake = dem.where(lake_sel)

    # Polygonize
    # ==========
    lake_shp = polygonize_raster(
        data=dem_lake.isnull(),
        mask=~dem_lake.isnull(),
        connectivity=4,
        transform=dem_lake.rio.transform(),
    )

    # Fill geometries
    if buffer_size is not None:
        lake_shp = fill_geom(
            lake_shp, buffer_size=buffer_size, buffer_delta=bufffer_delta
        )

    # Select geometry overlapping lake coordinates
    lake_shp = select_poly_from_multipoly(lake_shp, selection=Point(xy_lake))

    # Transform to geodataframe + save if asked
    # ================================
    lake_shp_gdf = gpd.GeoDataFrame(
        {"lake": [name_lake]}, geometry=[lake_shp], crs=crs_lake
    )
    if save_name is not None:
        lake_shp_gdf.to_file(save_name)

    if as_gdf:
        lake_shp = lake_shp_gdf

    return lake_shp


def burn_lake_to_dem(
    dem: xr.DataArray,
    lake_shp: Union[BaseGeometry, gpd.GeoDataFrame],
    crs_lake: str = None,
    method: Union[str, float] = "min",
    save_name: str = None,
):
    # Check type and get correct geometry
    if not isinstance(lake_shp, Union[gpd.GeoDataFrame, BaseGeometry]):
        raise TypeError(
            f"'lake_shp' must be part of  Union[BaseGeometry, gpd.GeoDataFrame],"
            f" got {type(lake_shp)}"
        )

    # Check projections
    if isinstance(lake_shp, BaseGeometry):
        if crs_lake is None:
            crs_lake = dem.rio.crs
            print(
                Warning(
                    f"CRS_LAKE is not set, it will be assumed to be equal to CRS_DEM"
                )
            )
        lake_shp = gpd.GeoDataFrame(
            {"lake": ["lake"]}, geometry=[lake_shp], crs=crs_lake
        )

    # Reproject to same coordinates + selection of lake
    lake_shp = lake_shp.to_crs(dem.rio.crs)
    lake_sel = ~dem.rio.clip(lake_shp.geometry, all_touched=True, drop=False).isnull()

    # Get value to replace lake value with / Min by default
    allowed_meth = ["min", "max", "mean", "median"]
    lake_alt = dem.where(lake_sel).min().data

    if isinstance(method, Union[int, float]):
        lake_alt = method
    elif not any([method == meth for meth in allowed_meth]):
        raise ValueError(f"'method' must be part of {allowed_meth}, got {method}")
    elif method == "max":
        lake_alt = dem.where(lake_sel).max().data
    elif method == "mean":
        lake_alt = dem.where(lake_sel).mean().data
    elif method == "median":
        lake_alt = dem.where(lake_sel).median().data

    # Burn dem
    dem_burnt = dem.where(~lake_sel, other=lake_alt)

    if save_name is not None:
        dem_burnt.rio.to_raster(save_name)
    return dem_burnt


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

    return [catch, x_snap, y_snap, acc]


def get_shed(
    dem: xr.DataArray,
    xy_outlet: list = None,
    crs_outlet: str = None,
    name_shed: str = "shed",
    buffer_size=5,
    buffer_delta=0,
    save_name: str = None,
):
    # Check projection
    # ================
    crs_dem = dem.rio.crs
    if crs_outlet is None:
        crs_outlet = crs_dem
        print(
            Warning(f"CRS_OUTLET is not set, it will be assumed to be equal to CRS_DEM")
        )

    # Get outlet coordinates into dem coordinates
    # ===========================================
    xy_outlet = (
        gpd.GeoSeries.from_xy([xy_outlet[0]], [xy_outlet[1]], crs=crs_outlet)
        .to_crs(crs_dem)
        .get_coordinates()
    )
    xy_outlet = [xy_outlet["x"].values[0], xy_outlet["y"].values[0]]

    # Create the pyshed Raster Object from rioxarray dem
    # ==================================================
    raster_shed = raster_shed_from_rioxr(raster_rioxr=dem)

    # Process the Raster object
    # =========================
    dem, grid, fdir, flats = raster_shed_processing(raster_shed=raster_shed)

    # Get the catchment
    # =================
    catch, x_snap, y_snap, acc = get_catchment(
        x=xy_outlet[0],
        y=xy_outlet[1],
        grid=grid,
        fdir=fdir,
    )

    # Transform catchment to shape + clean interiors
    catch_geom = raster_shed_to_geom(grid=grid, raster=catch)
    catch_geom_filled = fill_geom(
        catch_geom, buffer_size=buffer_size, buffer_delta=buffer_delta
    )
    catch_geom_sel = select_poly_from_multipoly(catch_geom_filled, selection="area")

    # Create gdf
    # ==========
    gdf_shed = gpd.GeoDataFrame(
        {
            "lake": [name_shed],
            "X outlet": [xy_outlet[0]],
            "Y outlet": [xy_outlet[1]],
            "X snap": x_snap,
            "Y snap": y_snap,
        },
        geometry=[catch_geom_sel],
        crs=crs_dem,
    )  # Data and geom as list not to provide an index

    # Save
    if save_name is not None:
        gdf_shed.to_file(save_name)

    return gdf_shed


@dask.delayed
def run_get_shed(
    dem: xr.DataArray,
    xy_outlet: list = None,
    crs_outlet: str = None,
    name_shed: str = "shed",
    buffer_size=5,
    buffer_delta=0,
    save_name: str = None,
):
    return get_shed(
        dem=dem,
        xy_outlet=xy_outlet,
        crs_outlet=crs_outlet,
        name_shed=name_shed,
        buffer_size=buffer_size,
        buffer_delta=buffer_delta,
        save_name=save_name,
    )


@dask.delayed
def run_get_lake_shape(
    dem: Union[str, xr.DataArray],
    xy_lake,
    flattening_thresh=0.3,
    buffer_size: float = 8,
    bufffer_delta: float = 1.5,
    crs_dem: str = None,
    crs_lake: str = None,
    as_gdf=False,
    name_lake: str = "lake",
    save_name: str = None,
):
    return get_lake_shape(
        dem=dem,
        xy_lake=xy_lake,
        flattening_thresh=flattening_thresh,
        buffer_size=buffer_size,
        bufffer_delta=bufffer_delta,
        crs_dem=crs_dem,
        crs_lake=crs_lake,
        as_gdf=as_gdf,
        name_lake=name_lake,
        save_name=save_name,
    )
