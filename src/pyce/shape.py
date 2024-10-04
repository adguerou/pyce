import warnings
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon

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


def df_to_gdf(df: pd.DataFrame, x_col: str, y_col: str, crs: str):
    """
    Transform a dataframe containing x and y coordinates to a GeoDataFrame with points geometry
    :param df: dataframe
    :param x_col: x column name
    :param y_col: y column name
    :param crs: crs projection
    :return: geodataframe
    """
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[x_col], df[y_col])
    ).set_crs(crs)


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


def select_overlapping_shapes(
    shape1: Union[str, gpd.GeoDataFrame],
    shape2: Union[str, gpd.GeoDataFrame],
    save_name=None,
    **kwargs,
):
    if isinstance(shape1, str):
        gdf1 = gpd.read_file(shape1)
    elif isinstance(shape1, gpd.GeoDataFrame):
        gdf1 = shape1
    else:
        raise TypeError("shape2 : Union[str,gpd.GeoDataFrame]")

    if isinstance(shape2, str):
        gdf2 = gpd.read_file(shape2, **kwargs)
    elif isinstance(shape2, gpd.GeoDataFrame):
        gdf2 = shape2
    else:
        raise TypeError("shape2 : Union[str,gpd.GeoDataFrame]")

    gdf1 = remove_bad_geometry(gdf1)
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


def remove_interiors_geom(
    geom: Union[BaseGeometry, gpd.GeoDataFrame],
    selection: Union[str, BaseGeometry] = "area",
):
    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)

    if not isinstance(geom, MultiPolygon):
        raise ValueError(f"'geom' must be Multi or single Polygon, got {type(geom)}")
    else:
        list_poly = list(geom.geoms)

        if selection == "area":
            list_areas = [p.area for p in list_poly]
            poly_sel = np.array(list_poly)[list_areas == np.max(list_areas)][0]
            return Polygon(poly_sel.exterior)

        elif isinstance(selection, BaseGeometry):
            sel = [shapely.contains(p, selection) for p in list_poly]
            poly_sel = np.array(list_poly)[sel][0]
            return Polygon(poly_sel.exterior)


def rename_lcmap_df_col(
    df: pd.DataFrame,
    lcmap: LandCoverMap,
    col: str = "landcover",
    prefix: bool = True,
    inplace: bool = False,
):
    """
    Rename the columns of a dataframe that corresponds to LandCoverMap codes.
    Changes from codes to litteral names.

    :param df: dataframe to renamed the columns from
    :param lcmap: LandCoverMap that contains the correspondance code<->littereal type
    :param col: column of the dataframe containing the landcover codes
    :param inplace: If true, changes directly the dataframe (default FALSE)
    :return: pd.Dataframe
    """

    lc_cols = df.columns[df.columns.str.startswith(col)]
    if len(lc_cols) == 0:
        raise IOError("No LandCoverMap column found. Check column name")

    if not inplace:
        df = df.copy()

    if prefix:
        df[col] = df[col].apply(lambda x: lcmap.get_type_of_code(int(x[-1])))
    else:
        df[col] = df[col].apply(lambda x: lcmap.get_type_of_code(x))

    if inplace is False:
        return df
