from typing import Union

import geopandas as gpd
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
