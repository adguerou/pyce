import dask
import ee
import geemap
import geopandas as gpd
import pandas as pd


# =====================================================================================
# Functions to compute spectral indices of satellites data based on Google Earth Engine
# =====================================================================================
def add_s2_gbr(image):
    NIR = image.select("B8")
    SWIR1 = image.select("B11")
    return image.addBands((NIR / SWIR1).rename("GBR"))


def add_s2_nari(image):
    green = image.select("B3")
    red_edge = image.select("B5")

    return image.addBands(
        (((1 / green) - (1 / red_edge)) / ((1 / green) + (1 / red_edge))).rename("NARI")
    )


def add_s2_ncri(image):
    red_edge = image.select("B5")
    red_edge_bis = image.select("B7")

    return image.addBands(
        (
            ((1 / red_edge) - (1 / red_edge_bis))
            / ((1 / red_edge) + (1 / red_edge_bis))
        ).rename("NCRI")
    )


# =====================================================================================
# General tools
# =====================================================================================
@dask.delayed
def _ee_to_gdf_dask(
    fc: ee.FeatureCollection, offset: int, chunk: int = 5000, crs: str = None
):
    """
    Transform a FeatureCollection object from GEE to a pandas GeoDataframe.
    This is a way to overcome the GEE limit of 5000 items to be displayed.

    :param fc: featureCollection to convert to gdf
    :param offset: index of first feature to convert
    :param chunk: number of feature to convert
    :return: geodataframe containing the featureCollection
    """
    subset = ee.FeatureCollection(fc.toList(chunk, offset=offset))
    gdf_subset = geemap.ee_to_gdf(subset)

    if crs is not None:
        gdf_subset.set_crs(crs, inplace=True)

    return gdf_subset


def ee_to_gdf_by_slice(
    fc: ee.FeatureCollection,
    chunk: int = 5000,
    chunk_shift: int = 0,
    max_size: int = None,
    crs: str = "EPSG:4326",
    num_workers: int = 14,
):
    """
    Transform a FeatureCollection object from GEE to a pandas GeoDataframe.
    Perform the operation over chunks of givin size (5000 max), to overcome
    the GEE limit of 5000 items to be displayed. Uses dask to compute over large
    datasets.

    :param fc: featureCollection to convert to gdf
    :param chunk: Number of feature to convert at once, 5000 is the GEE limit
    :param chunk_shift: Starting index of FeatureCollection to transform
    :param max_size: Maximum number of feature to transform in total
    :param crs: CRS to apply to the geodataframe
    :param num_workers: number of cores to use with dask

    :return: geopandas.GeoDataFrame containing the featureCollection. Each property is a column
    """

    gdf_subsets = []

    if max_size is None:
        max_size = fc.size().getInfo()

    # Define number of chunk depending
    if max_size / chunk % 1 == 0:
        number_of_chunk = int(max_size / chunk)
    else:
        number_of_chunk = int(max_size / chunk) + 1

    # Loop over chunks / resize the last chunk to number of feature left
    for n_chunk in range(number_of_chunk):
        # Derive before since chunk can be modified later on
        offset = chunk_shift + n_chunk * chunk
        # To get the last chunk size
        if chunk * (n_chunk + 1) > max_size:
            chunk = max_size - n_chunk * chunk
        gdf_subsets.append(_ee_to_gdf_dask(fc, offset=offset, chunk=chunk, crs=None))

    gdf_subsets_computed = dask.compute(gdf_subsets, num_workers=num_workers)

    gdf = pd.concat([d for d in gdf_subsets_computed[0]])
    gdf.reset_index(inplace=True)

    if crs is not None:
        gdf.set_crs(crs, inplace=True)

    return gdf
