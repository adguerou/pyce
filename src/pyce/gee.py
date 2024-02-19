import dask
import ee
import geemap
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


def set_month(image: ee.Image, first_day_of_month: int):
    """
    Return the month of an GEE image based on the first day given. Useful to filter
    images between, for instance, the 15th of each month.

    :param image: GEE image
    :param first_day_of_month: First of the month to consider.
           If 15 is given, images from the 14th will be tagged month-1
    :return: Month of the image as int
    """
    if not 1 <= first_day_of_month <= 31:
        raise ValueError(
            f"first_day_of_month must be within the range [1,31], got {first_day_of_month}"
        )
    date = image.date()

    month = ee.Algorithms.If(
        date.get("day").gte(first_day_of_month),
        date.get("month"),
        date.get("month") - 1,
    )

    return month


def ic_monthly_median(
    ic: ee.ImageCollection,
    month_list: list = None,
    first_day_of_month=1,
    month_prop_name: str = "MONTH",
    rename_band: bool = True,
    band_names: list[str] = None,
):
    # -------------------------------------------------------------------------------
    def _get_monthly_median(
        ic: ee.ImageCollection,
        month: int = None,
        month_prop_name: str = "MONTH",
    ):
        """
        Select images from a given month within an ImageCollection, return its median
        :param ic:
        :param month:
        :param month_prop_name:
        :return:
        """

        return (
            ic.filter(ee.Filter.eq(month_prop_name, month))
            .median()
            .set(month_prop_name, month)
        )

    def _create_band_names(bands_list: list[str] = None, month_list: list[int] = None):
        """
        Return a new band list to rename ImageCollection with the month indicated per band.
        This is  usefull to transform imageCollection to single Imge with multiple bands.

        :param bands_list:
        :param month_list:
        :return:
        """
        new_band_names = []

        # For loop in this order to be consistent with the flattening of bands when transforming
        # an ImageCollection to single Image with bands. First is the month as one image in the
        # ImageCollection corresponds to each month

        for month in month_list:
            for band in bands_list:
                new_band_names.append(f"{band}_{month}")
        return new_band_names

    # -------------------------------------------------------------------------------

    # =============
    # Function start
    # =============

    # Add a tagg to each image according to its month belonging
    ic_month_tagged = ic.map(
        lambda img: img.set(
            month_prop_name,
            set_month(img, first_day_of_month=first_day_of_month),
        )
    )

    # Get the monthly median
    monthly_median_coll = ee.ImageCollection.fromImages(
        ee.List(month_list).map(
            lambda month: _get_monthly_median(
                ic_month_tagged,
                month,
                month_prop_name=month_prop_name,
            )
        )
    )

    # Transform the ImageCollection to an image with renamed bands
    if rename_band:
        new_band_names = _create_band_names(
            bands_list=band_names, month_list=month_list
        )

        return monthly_median_coll.toBands().rename(new_band_names)
    else:
        return monthly_median_coll
