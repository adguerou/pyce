import os.path
import types

import dask
import ee
import geemap
import numpy as np
import pandas as pd
import seaborn as sbn
from matplotlib import pyplot as plt

# =====================================================================================
# Functions to compute spectral indices of satellites data based on Google Earth Engine
# =====================================================================================


# SENTINEL-2
# ----------
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


def add_s2_cvi(image):
    green = image.select("B3")
    red = image.select("B4")
    nir = image.select("B8")

    return image.addBands(nir * red / green**2)


# LANDSAT
# -------
def rename_bands(
    imageCollection: ee.ImageCollection,
    input_bands: list,
    output_bands: list,
):
    return imageCollection.select(input_bands, output_bands)


def rename_bands_LT457(imageCollection: ee.ImageCollection, invert=False):
    input_bands = [
        "SR_B1",
        "SR_B2",
        "SR_B3",
        "SR_B4",
        "SR_B5",
        "SR_B7",
        "ST_B6",
        "QA_PIXEL",
    ]
    output_bands = [
        "BLUE",
        "GREEN",
        "RED",
        "NIR",
        "SWIR1",
        "SWIR2",
        "THERMAL1",
        "QA_PIXEL",
    ]
    if invert is True:
        return rename_bands(imageCollection, output_bands, input_bands)
    else:
        return rename_bands(imageCollection, input_bands, output_bands)


def rename_bands_LT457_invert(imageCollection: ee.ImageCollection):
    return rename_bands_LT457(imageCollection, invert=True)


def rename_bands_LC8(imageCollection: ee.ImageCollection, invert=False):
    input_bands = [
        "SR_B1",
        "SR_B2",
        "SR_B3",
        "SR_B4",
        "SR_B5",
        "SR_B6",
        "SR_B7",
        "ST_B10",
        "QA_PIXEL",
    ]
    output_bands = [
        "AEROSOLS",
        "BLUE",
        "GREEN",
        "RED",
        "NIR",
        "SWIR1",
        "SWIR2",
        "THERMAL1",
        "QA_PIXEL",
    ]
    if invert:
        return rename_bands(imageCollection, output_bands, input_bands)
    else:
        return rename_bands(imageCollection, input_bands, output_bands)


def rename_bands_LC8_invert(imageCollection: ee.ImageCollection):
    return rename_bands_LC8(imageCollection, invert=True)


def add_landsat457_gbr(image: ee.Image):
    NIR = image.select("SR_B4")
    SWIR1 = image.select("SR_B5")
    return image.addBands((NIR / SWIR1).rename("GBR"))


def add_landsat8_gbr(image: ee.Image):
    NIR = image.select("SR_B5")
    SWIR1 = image.select("SR)B6")
    return image.addBands((NIR / SWIR1).rename("GBR"))


def get_landsat_cal(factors, bandName, factorNumber):
    return ee.List(factors.get(bandName)).getNumber(factorNumber)


def apply_landsat_cal(image: ee.Image, factors, bandName):
    return (
        ee.Image(get_landsat_cal(factors, bandName, 0))
        .add(image.select(bandName).multiply(get_landsat_cal(factors, bandName, 1)))
        .add(
            image.select(bandName)
            .pow(2)
            .multiply(get_landsat_cal(factors, bandName, 2))
        )
        .add(
            image.select(bandName)
            .pow(3)
            .multiply(get_landsat_cal(factors, bandName, 3))
        )
        .rename(bandName)
    )


def apply_crosscal_LT5(image: ee.Image, region: str = "temp"):
    # Source: https://www.nature.com/articles/s41558-024-02177-x#Sec14

    # Temperate mountains
    factors_temp = ee.Dictionary(
        {
            "RED": [-0.0042, 1.0049, 0.0000, 0.0000],
            "NIR": [0.0070, 0.8904, 0.4760, -0.5883],
            "GREEN": [0.0008, 0.8998, 0.2289, 0.0000],
            "SWIR1": [-0.0027, 1.0503, -0.4189, 0.5941],
        }
    )
    # Arctic
    factors_arct = ee.Dictionary(
        {
            "RED": [-0.0068, 1.0075, 0.0000, 0.0000],
            "GREEN": [-0.0011, 0.8733, 0.3266, 0.0000],
            "NIR": [0.0057, 0.9686, 0.0000, 0.0000],
            "SWIR1": [0.0010, 0.9791, 0.0000, 0.0000],
        }
    )

    # Region selection
    factors_dict = {"temp": factors_temp, "arctic": factors_arct}
    if region not in ["temp", "arctic"]:
        raise IOError("Region must be within ['temp', 'arctic']")
    else:
        factors = factors_dict[region]

    cal = (
        apply_landsat_cal(image, factors, "RED")
        .addBands(apply_landsat_cal(image, factors, "NIR"))
        .addBands(apply_landsat_cal(image, factors, "GREEN"))
        .addBands(apply_landsat_cal(image, factors, "SWIR1"))
    )

    return image.addBands(cal, None, True)


def apply_crosscal_LC8(image: ee.Image, region: str = "temp"):
    # Source: https://www.nature.com/articles/s41558-024-02177-x#Sec14

    # Temperate mountains
    factors_temp = ee.Dictionary(
        {
            "RED": [0.0027, 1.0517, -0.1658, 0.0000],
            "NIR": [0.0140, 0.8557, 0.0000, 0.0000],
            "GREEN": [0.0047, 0.9191, 0.9488, -2.9558],
            "SWIR1": [0.0171, 0.7561, 1.1901, -1.6099],
        }
    )
    # Arctic
    factors_arct = ee.Dictionary(
        {
            "RED": [-0.0012, 1.1592, -0.9166, 1.8795],
            "GREEN": [-0.0005, 1.0412, 0.0000, 0.0000],
            "NIR": [0.0221, 0.8442, 0.1811, 0.0000],
            "SWIR1": [0.0071, 0.9698, 0.0685, 0.0000],
        }
    )

    # Region selection
    factors_dict = {"temp": factors_temp, "arctic": factors_arct}
    if region not in ["temp", "arctic"]:
        raise IOError("Region must be within ['temp', 'arctic']")
    else:
        factors = factors_dict[region]

    cal = (
        apply_landsat_cal(image, factors, "RED")
        .addBands(apply_landsat_cal(image, factors, "NIR"))
        .addBands(apply_landsat_cal(image, factors, "GREEN"))
        .addBands(apply_landsat_cal(image, factors, "SWIR1"))
    )

    return image.addBands(cal, None, True)


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
        gdf_subset.to_crs(crs, inplace=True)

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

    # Set max_size if not set in parameters
    if max_size is None:
        max_size = fc.size().getInfo()

    # Ensure not to look for empty feature
    if max_size > fc.size().getInfo():
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
        gdf.to_crs(crs, inplace=True)

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


def split_FeatureCollection(
    fc: ee.FeatureCollection, chunk: int = None, n_parts: int = None
) -> list[ee.FeatureCollection]:
    """
    Split a FeatureCollection object into a list of FeatureCollection. The number of Feature
    to be included in each resulting FeatureCollection is controlled by the chunk size or the
    number of FeatureCollection in the resulting list.
    This is useful to lower the memory usage when using a sampleRegions onto the FeatureCollection.


    :param fc: FeatureCollection
    :param chunk: Number of Features to be included in each resulting FeatureCollection
    :param n_parts: Number of FeatureCollection to create.
    :return: list[ee.FeatureCollection]
    """
    if chunk is None and n_parts is None:
        raise IOError("chunk OR n_parts must be set")
    if chunk is not None and n_parts is not None:
        raise IOError("chunk AND n_parts CANNOT be set simultaneously")

    # Get the number of Feature
    fc_size = fc.size().getInfo()

    if n_parts is not None:
        chunk = int(fc_size / n_parts) + 1

    if chunk is not None:
        n_parts = int(fc_size / chunk) + 1

    return [fc.toList(count=chunk, offset=chunk * npart) for npart in range(n_parts)]


# =====================================================================================
# RANDOM FOREST CLASSIFICATION
# =====================================================================================
def rf_circular(
    rf_gee: ee.Classifier.smileRandomForest,
    ds_training: ee.FeatureCollection,
    ds_indices: ee.Image,
    indices_names: list[str],
    n_areas: int = 4,
    areas_name: str = "rf_area_id",
    label_classif_col: str = "landcover",
    labels_lc: list[str] = None,
):
    # Final output list
    rf_proba = []  # Classif proba
    error_matrix = []  # Accuracy of validation sample for each RF
    calibration_scores = []  # Proba value of the true class for the calibration set

    # Indices list of the circular areas
    list_rf_id = list(range(1, n_areas + 1))

    # Classification for each circular areas
    # --------------------------------------
    for rf_id in list_rf_id:
        # Get training and validation sample
        rf_training_id = [x for x in list_rf_id if x != rf_id]
        rf_validation_id = [rf_id]

        sample_training = ds_training.filter(
            ee.Filter.inList(areas_name, rf_training_id)
        )
        sample_validation = ds_training.filter(
            ee.Filter.inList(areas_name, rf_validation_id)
        )

        # Train the model
        classifier = rf_gee.train(sample_training, label_classif_col, indices_names)

        # Get the classification
        classified_image_proba = ds_indices.classify(
            classifier.setOutputMode("MULTIPROBABILITY")
        ).arrayFlatten([labels_lc])
        rf_proba.append(classified_image_proba)

        # Get the accuracy on the validation sample
        classified_validation = sample_validation.classify(classifier)
        error_matrix.append(
            classified_validation.errorMatrix(label_classif_col, "classification")
        )

        # Used for uncertainty estimation
        # -------------------------------
        # Get the proba value from the calibration set
        calibration_proba = sample_validation.classify(
            classifier.setOutputMode("MULTIPROBABILITY")
        )
        calibration_scores.append(calibration_proba)

    # UNCERTAINTIES
    # --------------
    # Function to select scores on df
    def sel_scores(row, landcover=label_classif_col):
        return row["classification"][row[landcover]]

    # Get the scores (the proba of the true class)
    df_scores = []
    for rf_scores in calibration_scores:
        df_rf_scores = geemap.ee_to_df(rf_scores)
        df_rf_scores["scores"] = df_rf_scores.apply(lambda row: sel_scores(row), axis=1)
        df_scores.append(df_rf_scores["scores"])
    df_scores = pd.concat(df_scores)
    # -----------------

    # Get the mean proba for each landcover class from the circular RF
    rf_mean = ee.ImageCollection.fromImages(rf_proba).mean()
    rf_std = ee.ImageCollection.fromImages(rf_proba).reduce(ee.Reducer.stdDev())

    return rf_mean, rf_std, error_matrix, df_scores


def classify(
    rf_mean: ee.Image,
    lc_labels: list[str],
    lc_values: list[str],
    name: str = "classification",
    extra_class: list[tuple] = None,
    extra_class_value: int = None,
):
    def add_condition(cond1, cond2):
        return cond1 & cond2

    # Basic checks for inputs
    # =======================
    if len(lc_labels) != len(lc_values):
        raise IOError(
            f"Lenght of 'lc_labels' and 'lc_values' must be equal,"
            f" got {len(lc_labels)} and {len(lc_values)}"
        )

    # Extra class
    if extra_class is not None:
        if not isinstance(extra_class, list):
            raise TypeError(
                f"'extra_class' argument must be a list, got {type(extra_class)}"
            )
        for extra_class_item in extra_class:
            if not isinstance(extra_class_item, tuple):
                raise TypeError(
                    f"'extra_class' items must be tuples, got {type(extra_class_item)}"
                )
            if len(extra_class_item) != 3:
                raise IOError(
                    f"'extra_class' items must be of length 3, "
                    f"got {len(extra_class_item)}"
                )
            if not isinstance(extra_class_item[0], str):
                raise TypeError(
                    f"'extra_class' tuples 1st argument be a str, "
                    f"got {type(extra_class_item[0])}"
                )
            if not isinstance(extra_class_item[1], types.BuiltinFunctionType):
                raise TypeError(
                    f"'extra_class' tuples 2nd argument be a built-in operator, "
                    f"got {type(extra_class_item[1])}"
                )
            if not isinstance(extra_class_item[2], float):
                raise TypeError(
                    f"'extra_class' tuples 3rd argument be a float,"
                    f" got {type(extra_class_item[2])}"
                )

        if extra_class_value is None:
            raise IOError("extra_class_value must be filled, got None")

    # Classification by landcover type / create ee.Image
    # ---------------------------------------------------
    classif = None

    for label, value in zip(lc_labels, lc_values):
        other_labels = [lbl for lbl in lc_labels if lbl != label]
        cond = rf_mean.select(label) > rf_mean.select(other_labels[0])
        for other_label in other_labels[1:]:
            cond = add_condition(
                cond, rf_mean.select(label) > rf_mean.select(other_label)
            )
        try:
            if isinstance(classif, ee.Image):
                classif = classif.where(cond, int(value))
            else:
                classif = ee.Image.constant(-1).rename(name).where(cond, int(value))
        except:
            raise AssertionError(
                f"Image classification couldn't be initialize and/or modified"
            )

    # Add a class on top of other classification
    # ------------------------------------------
    if extra_class is not None:
        cond = extra_class[0][1](rf_mean.select(extra_class[0][0]), extra_class[0][2])

        for condition in extra_class[1:]:
            cond = add_condition(
                cond, condition[1](rf_mean.select(condition[0]), condition[2])
            )

        classif = classif.where(cond, int(extra_class_value))

    return classif


def plot_errorMatrix(errorMatrix, lcmap, label=None, save_dir=None, save_name=None):
    mat = errorMatrix.getInfo()
    mat_percent = np.divide(mat, np.sum(mat, axis=1)) * 100

    accuracy = errorMatrix.accuracy().getInfo()
    kappa = errorMatrix.kappa().getInfo()
    fscore = errorMatrix.fscore()

    f, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, gridspec_kw={"width_ratios": [0.08, 1]}
    )

    g1 = sbn.heatmap(
        mat_percent,
        annot=mat,
        vmin=0,
        vmax=100,
        cmap="crest_r",
        linewidth=0.5,
        fmt=".0f",
        ax=ax2,
    )
    f.subplots_adjust(wspace=0.5)

    ax1.axis("off")
    ax1.set_xlim([0, 1])
    fscore_str = [f"{fs:.3f}" for fs in fscore.getInfo()]

    for i in range(len(fscore_str)):
        ax1.text(0.1, i + 0.6, f"F-score:\n {fscore_str[i]}")

    labels = [lbl.replace(" ", "\n") for lbl in lcmap.get_type()]

    ax2.set(xticklabels=labels, yticklabels=labels)
    ax2.set_xticklabels(g1.get_xmajorticklabels(), fontsize=9)
    ax2.set_xlabel(g1.get_xlabel(), fontsize=12, fontweight="bold")
    ax2.set_ylabel(g1.get_ylabel(), fontsize=12, fontweight="bold")
    ax2.yaxis.set_tick_params(labelleft=True)

    f.suptitle(f"{label} - accuracy: {accuracy:.2f} - kappa: {kappa:.2f}")

    cbar = g1.collections[0].colorbar
    cbar.set_ticks([0, 50, 100])
    cbar.set_label(f"[%] of prediction")

    ax2.tick_params(axis="x", rotation=30)
    ax2.tick_params(axis="y", rotation=30)

    f.tight_layout()

    if save_dir is not None and save_name is not None:
        plt.savefig(os.path.join(save_dir, save_name), dpi=200)


# =====================================================================================
# Digital Elevation Model
# =====================================================================================
def get_slope(image: ee.Image):
    return ee.Terrain.slope(image)


def get_aspect(image: ee.Image):
    return ee.Terrain.aspect(image)
