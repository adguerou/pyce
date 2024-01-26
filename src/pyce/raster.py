from threading import Lock
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import richdem as rd
import rioxarray as rioxr
import xarray as xr


def crop_raster_from_shp(
    rst_file: str,
    shp_file: Union[str, gpd],
    chunks: dict = None,
    num_workers: int = 14,
    save_file: str = None,
    no_data: float = None,
):
    # Open raster + no lock to allow parallel processing
    ds = rioxr.open_rasterio(rst_file, chunks=chunks, lock=False)

    # Open shape file and convert coordinates
    if isinstance(shp_file, str):
        gdf = gpd.read_file(shp_file)
    elif isinstance(shp_file, gpd.GeoDataFrame):
        gdf = shp_file
    else:
        raise ValueError("shp_file must either a str or a GeoDataFrame")
    gdf.to_crs(ds.rio.crs, inplace=True)

    # Crop raster to the limit of the shape file
    ds = ds.isel({"x": ds["x"] >= gdf.total_bounds[0]})
    ds = ds.isel({"y": ds["y"] >= gdf.total_bounds[1]})
    ds = ds.isel({"x": ds["x"] <= gdf.total_bounds[2]})
    ds = ds.isel({"y": ds["y"] <= gdf.total_bounds[3]})

    # Set no data
    if ds.rio.nodata is not None:
        ds = ds.where(ds != ds.rio.nodata)
        ds.rio.write_nodata(ds.rio.nodata, inplace=True, encoded=True)
    else:
        if no_data is not None:
            ds.rio.write_nodata(no_data, inplace=True, encoded=True)

    # Save raster using the dask xarray
    if save_file is not None and chunks is not None:
        ds_rst = ds.rio.to_raster(save_file, tiled=True, lock=Lock(), compute=False)
        ds_rst.compute(scheduler="threads", num_workers=num_workers)
    else:
        ds_rst = ds.rio.to_raster(save_file)

    return ds


def clip_raster_from_shp(
    rst_file: str,
    shp_file: str,
    chunks: dict = None,
    num_workers: int = 14,
    save_file: str = None,
    no_data: float = None,
):
    # Open raster + no lock to allow parallel processing
    ds = rioxr.open_rasterio(rst_file, chunks=chunks, lock=False)

    # Open shape file and convert coordinates
    gdf = gpd.read_file(shp_file)
    gdf.to_crs(ds.rio.crs, inplace=True)

    # Clip raster to the limit of the shape file, keep shape's border touching pixels
    ds = ds.rio.clip(gdf.geometry, all_touched=True)

    # Set no data
    if ds.rio.nodata is not None:
        ds = ds.where(ds != ds.rio.nodata)
        ds.rio.write_nodata(ds.rio.nodata, inplace=True, encoded=True)
    else:
        if no_data is not None:
            ds.rio.write_nodata(no_data, inplace=True, encoded=True)

    # Save raster using the dask xarray
    if save_file is not None:
        ds_rst = ds.rio.to_raster(save_file, tiled=True, lock=Lock(), compute=False)
        ds_rst.compute(scheduler="threads", num_workers=num_workers)

    return ds


def mask_raster(ds: rioxr, val_to_mask: list[int], mask_val=np.nan):
    """
    Mask raster where data is equal to a list of value.
    Take care of keeping rioxarray metadata not transfered by xarray operation

    :param ds: rioxarray to mask
    :param val_to_mask: list of value to mask
    :param mask_val: value to use for the values to mask (value of replacement).
    By default, these values are replaced by NaN
    :return: rioxarray with masked value
    """
    nodata = ds.rio.encoded_nodata
    encoding = ds.encoding

    ds = ds.where(~ds.isin(val_to_mask), other=mask_val)

    ds.rio.update_encoding(encoding, inplace=True)
    ds.rio.write_nodata(nodata, inplace=True, encoded=True)

    return ds


def raster_to_dataset(
    path,
    var_name: str = None,
    var_out_name: str = None,
    band_name: str = "band",
    band_out_names: dict = None,
    chunks=None,
):
    """
    Transform a raster file to an xarray dataset
    :param path: Path of the raster file
    :param var_name: variable to use in the raster
    :param var_out_name: name of the variable for the dataset
    :param band_name: str used to define bands in the raster file
    :param band_out_names: name of the band to give as for the variables definition in the datset
    :param chunks: as for rioxr.open_rasterio
    :return: dataset
    """
    if chunks is None:
        chunks = {"x": 5000, "y": 5000}
    ds = rioxr.open_rasterio(path, mask_and_scale=True, chunks=chunks)

    # TODO: Can be replaced by options 'band_as_variable' of open_rasterio
    if ds.band.size > 1:
        if band_out_names is None:
            try:
                band_out_names = ds.long_name
            except ValueError:
                print("Attributes long_name not found, provide 'band_out_names'")
        dataset = ds.to_dataset(dim="band").rename_vars(
            {k: v for (k, v) in zip(range(1, ds.band.size + 1), band_out_names)}
        )
    else:
        dataset = ds.to_dataset(name=var_name).rename_vars({var_name: var_out_name})

    return dataset


def distance_meter_from_deg(rio_ds: rioxr, crs: int = 3035) -> float:
    """
    Get unit distance (one degree) in meters at the mean latitude of a raster.
    The conversion is done in a new CRS that, by default, correspond to European zone

    :param rio_ds: rio xarray containing the raster
    :param crs: CRS to use to compute the distance in meters (3035 by default for European zone)
    :return: distance in meters of one degree

    """
    from shapely.geometry.point import Point

    lat = (rio_ds.rio.bounds()[1] + rio_ds.rio.bounds()[3]) / 2.0
    lon = rio_ds.rio.bounds()[0]

    # Geographic WGS 84 - degrees
    points = gpd.GeoSeries([Point(lon, lat), Point(lon + 1, lat)], crs=4326)
    # Projected WGS 84 - meters
    points = points.to_crs(crs)

    return points[0].distance(points[1])


def get_area_slope_corrected(
    area: Union[float, np.ndarray, pd.Series] = 10,
    slope: Union[float, np.ndarray, pd.Series] = 0,
    sum: bool = False,
):
    """
    Get the true surface covered by a pixel or a serie of pixel taking
    into account the slope of the terrain.

    :param area: area of pixel(s) in [unit]
    :param slope: Mean slope of the pixel in degrees
    :param sum: If True, perform the sum of all surface
                (if area and slope are vectors of same size)
    :return: Surface in [unit]
    """

    if not (isinstance(area, Union[int, float, np.ndarray, pd.Series])):
        raise TypeError(
            "'area' type must be within Union[float, np.array, pd.DataFrame]"
        )

    if not (isinstance(slope, Union[int, float, np.ndarray, pd.Series])):
        raise TypeError(
            "'slope' type must be within Union[float, np.array, pd.DataFrame]"
        )

    if np.size(area) != np.size(slope):
        raise ValueError("Parameters 'area' and 'slope' must be of same length")

    surface = area / np.cos(slope * np.pi / 180.0)

    if sum is True:
        surface = np.nansum(surface)

    return surface


def rd_terrain_slope_and_aspect(
    mnt_file: str, projection: str, save_file_slope=None, save_file_aspect=None
):
    # Open file and add projection
    rda = rd.LoadGDAL(mnt_file)
    rda.projection = projection

    # Derive slope
    slope = rd.TerrainAttribute(rda, attrib="slope_degrees")

    # Set points with no slope to no data / to get no orientation
    rda[slope == 0.0] = rda.no_data

    # Derive aspect
    aspect = rd.TerrainAttribute(rda, attrib="aspect")

    # Savings
    if save_file_slope is not None:
        rd.SaveGDAL(save_file_slope, slope)
    if save_file_aspect is not None:
        rd.SaveGDAL(save_file_aspect, aspect)


def mnt_interp_like(
    mnt_file: str,
    projection: str,
    ds_like: xr.DataArray,
    save_file=None,
):
    # Open mnt raster
    ds_mnt = rioxr.open_rasterio(
        mnt_file, chunks={"x": 100, "y": 100}, mask_and_scale=True
    )

    ds_mnt = ds_mnt.rio.write_crs(projection, inplace=True)
    bounds = ds_mnt.rio.bounds()

    # Clip the dataset to be interpolated on to the raster limits
    ds_like_clip = ds_like.rio.clip_box(bounds[0], bounds[1], bounds[2], bounds[3])

    # Do the interpolation + rewrite nodata
    ds_mnt_interp = ds_mnt["band" == 1].interp_like(ds_like_clip["band" == 1])
    ds_mnt_interp.rio.write_nodata(
        ds_mnt.rio.encoded_nodata, inplace=True, encoded=True
    )

    # Save files
    if save_file is not None:
        ds_mnt_interp.rio.to_raster(save_file, compute=False)
