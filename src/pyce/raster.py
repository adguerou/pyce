from threading import Lock

import geopandas as gpd
import numpy as np
import rioxarray as rioxr


def crop_raster_from_shp(
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
    if save_file is not None:
        ds_rst = ds.rio.to_raster(save_file, tiled=True, lock=Lock(), compute=False)
        ds_rst.compute(scheduler="threads", num_workers=num_workers)

    return ds


def mask_raster(ds: rioxr, val_to_mask: list[int], mask_val=np.nan):
    """
    Mask raster where data is equal to a list of value.
    Take care of keeping rioxarray metadata not transfer by xarray operation

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
