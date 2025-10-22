from threading import Lock
from typing import Union

import affine
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rioxr
import xarray as xr
from rioxarray import merge
from shapely.geometry import shape
from shapely.geometry.multipolygon import MultiPolygon


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


def merge_raster(
    raster_list: Union[list, str] = None,
    epsg: str = None,
    save_name: str = None,
    mask_and_scale: bool = True,
    compress="lzw",
):
    """
    Merge rasters using merge function from rioxarray

    :param raster_list: List of files to merge.
    :param epsg: EPSG code to project the tiles to before merging in case the file
                 format do not contain this information. It must be the same for
                 all files.
    :param save_name: Full path to save merged raster. Default None (no savings)

    :return: A rioxarray rasters of the merged tiles
    """
    if isinstance(raster_list, str):
        raster_list = [raster_list]

    rasters = []

    for raster in raster_list:
        rst = rioxr.open_rasterio(raster, mask_and_scale=mask_and_scale)
        if epsg is not None:
            rst = rst.rio.write_crs(epsg)
        rasters.append(rst)

    if len(rasters) == 1:
        return rasters[0]

    merged_raster = merge.merge_arrays(rasters)

    if save_name is not None:
        merged_raster.rio.to_raster(save_name, compress=compress)

    return merged_raster


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


def raster_interp_like(
    raster_file: str,
    projection: str,
    ds_like: xr.DataArray,
    save_file=None,
):
    # Open mnt raster
    ds_raster = rioxr.open_rasterio(
        raster_file, chunks={"x": 100, "y": 100}, mask_and_scale=True
    )

    ds_raster = ds_raster.rio.write_crs(projection, inplace=True)
    bounds = ds_raster.rio.bounds()

    # Clip the dataset to be interpolated on to the raster limits
    ds_like_clip = ds_like.rio.clip_box(bounds[0], bounds[1], bounds[2], bounds[3])

    # Do the interpolation + rewrite nodata
    ds_raster_interp = ds_raster["band" == 1].interp_like(ds_like_clip["band" == 1])
    ds_raster_interp.rio.write_nodata(
        ds_raster.rio.encoded_nodata, inplace=True, encoded=True
    )

    # Save files
    if save_file is not None:
        ds_raster_interp.rio.to_raster(save_file, compute=False)


def polygonize_raster(
    data: xr.DataArray,
    mask: xr.DataArray = None,
    connectivity: int = 4,
    transform: affine.Affine = None,
):
    # Check inputs
    if mask is None:
        mask = ~data.isnull()
    if transform is None:
        transform = data.rio.transform()

    # Use polygonize from rasterio
    list_geoms = [
        *rasterio.features.shapes(
            data.astype(np.uint8),
            connectivity=connectivity,
            mask=mask,
            transform=transform,
        )
    ]

    # Transform list of list to list
    polys = list(map(lambda x: x[0], list_geoms))

    return MultiPolygon([shape(p) for p in polys])
