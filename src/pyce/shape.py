import warnings
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon


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
    Transform a dataframe containing x and y coordinates to
    a GeoDataFrame with points geometry

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


def remove_bad_geometry(gdf: gpd.GeoDataFrame):
    return gdf.iloc[np.where(gdf["geometry"].values != None)[0]]


def remove_linestring(gdf: gpd.GeoDataFrame):
    """
    Remove LineString geometry within GeometryCollection from GeodataFrame object

    :param gdf:
    :return:
    """
    gdf_explode = gdf.explode(index_parts=False)
    gdf_clean = gdf_explode[gdf_explode.geometry.type != "LineString"]
    return gdf_clean.reset_index(drop=False).dissolve(by="index")


def fix_geometry(gdf: gpd.GeoDataFrame):
    """
    Fix geometries as in QGIS (self-ring etc)
    :param gdf:
    :return:
    """
    gdf["geometry"] = gdf["geometry"].make_valid()
    return gdf


def clean_multipolygon(geom, area_min=20 * 20 * 2):
    """
    Remove
    :param geom:
    :param area_min:
    :return:
    """
    if geom.geom_type == "MultiPolygon":
        return shapely.MultiPolygon([p for p in geom.geoms if p.area > area_min])
    else:
        return geom


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


def fill_geometry(
    geom: Union[BaseGeometry, gpd.GeoDataFrame],
    buffer_size: float = 10,
    buffer_delta: float = 0,
    cap_style: str = "round",
    join_style: str = "round",
):
    # Test types
    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)
    if not isinstance(geom, MultiPolygon):
        raise ValueError(f"'geom' must be Multi or single Polygon, got {type(geom)}")

    # Create a buffered geometry + merged
    geom_buff_merged = shapely.unary_union(
        geom.buffer(buffer_size, cap_style=cap_style, join_style=join_style)
    )

    # Depending on the resulting merge, need to split Multigeometry
    if isinstance(geom_buff_merged, MultiPolygon):
        list_geom_buff = list(geom_buff_merged.geoms)
    else:
        list_geom_buff = [geom_buff_merged]

    # Take the exterior of each single buffered geometry
    list_geom_buff_ext = [Polygon(g.exterior) for g in list_geom_buff]

    # Buffered back the filled geometries
    # Some complications needed to get a simple list of Polygons / buffer negative
    # values can lead to Multipolygon type whereas got Polygon as input
    list_geom_ext = []
    for g_buff_ext in list_geom_buff_ext:
        g_ext = g_buff_ext.buffer(
            -buffer_size - buffer_delta, cap_style=cap_style, join_style=join_style
        )

        if isinstance(g_ext, MultiPolygon):
            g_multi = list(g_ext.geoms)
            for g in g_multi:
                list_geom_ext.append(g)
        else:
            list_geom_ext.append(g_ext)

    return MultiPolygon(list_geom_ext)


def select_poly_from_multipoly(poly, selection: Union[str, BaseGeometry] = "area"):
    if isinstance(poly, Polygon):
        return poly

    if not isinstance(poly, MultiPolygon):
        raise ValueError(f"'geom' must be MultiPolygon, got {type(poly)}")

    list_poly = list(poly.geoms)

    if selection == "area":
        list_areas = [p.area for p in list_poly]
        poly_sel = np.array(list_poly)[list_areas == np.max(list_areas)][0]
        return poly_sel

    elif isinstance(selection, BaseGeometry):
        sel = [shapely.contains(p, selection) for p in list_poly]
        poly_sel = np.array(list_poly)[sel][0]
        return poly_sel
