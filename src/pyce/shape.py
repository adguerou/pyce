import warnings
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd

from pyce.tools.lc_mapping import LandCoverMap, oso_mapping_fusion_in_df


def concat(list_shapes: list[str], ignore_index: bool = True):
    # Create empty dataframe
    gdf_years = pd.DataFrame()

    # Loop over shape files
    for shape in list_shapes:

        # Open shape file
        gdf = gpd.read_file(shape)

        # Concatenate dataframe
        if not gdf_years.empty:
            gdf_years = pd.concat([gdf_years, gdf], axis=0, ignore_index=ignore_index)
        else:
            gdf_years = gdf

    return gdf_years


def str_to_datetime(df: Union[pd.DataFrame, gpd.GeoDataFrame], column="datetime"):
    df[column] = pd.to_datetime(df[column])

    return df


def fusion_oso_shape(
    list_shapes: list[str],
    lc_map_to: LandCoverMap,
    lc_map_from: LandCoverMap,
    save: str = None,
):
    gdf = concat(list_shapes, ignore_index=False)

    gdf = str_to_datetime(gdf, column="datetime")
    gdf.set_index("datetime", inplace=True)

    gdf = oso_mapping_fusion_in_df(gdf, lc_map_to=lc_map_to, lc_map_from=lc_map_from)

    if save:
        gdf.to_file(save)

    return gdf


def remove_bad_geometry(gdf: gpd.GeoDataFrame):
    return gdf.iloc[np.where(gdf["geometry"].values != None)[0]]


def select_overlapping_shapes(shape1: str, shape2: str, save_name=None, **kwargs):

    gdf1 = gpd.read_file(
        shape1,
    )
    gdf1 = remove_bad_geometry(gdf1)

    gdf2 = gpd.read_file(shape2, **kwargs)
    gdf2 = remove_bad_geometry(gdf2)
    gdf2 = gdf2.to_crs(gdf1.crs, inplace=False)

    overlapping = gdf1.sjoin(gdf2, how="inner", predicate="intersects")
    gdf1_overlap = gdf1.iloc[np.unique(overlapping.index)]
    gdf1_overlap = gdf1_overlap[
        gdf1_overlap.columns[gdf1_overlap.columns.isin(gdf1.columns)]
    ]

    if save_name is not None:
        if gdf1_overlap.empty:
            warnings.warn(
                f"Overlapping dataframe is empty {shape1}. Skip to avoid writing empty file"
            )
        else:
            gdf1_overlap.to_file(save_name)

    return gdf1_overlap
