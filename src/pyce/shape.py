import warnings
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon


# =====================================================================================================================
#                                               SIMPLE GEOMETRY
# =====================================================================================================================
def fix_geometry(gdf: gpd.GeoDataFrame):
    """
    Fix geometries as in QGIS (self-ring etc)
    :param gdf:
    :return:
    """
    gdf["geometry"] = gdf["geometry"].make_valid()
    return gdf


def remove_bad_geometry(gdf: gpd.GeoDataFrame):
    """
    Remove line where geometry is not existing - this is different from invalid geometries
    :param gdf: Geodataframe to process
    :return: Geodataframe with removed line where geometry do not exist
    """
    return gdf.iloc[np.where(gdf["geometry"].values != None)[0]]


def remove_linestring(gdf: gpd.GeoDataFrame, include_points: bool = False):
    """
    Remove LineString geometry within GeometryCollection from GeodataFrame object

    :param gdf:
    :param include_points: if TRUE, remove also POINT geometries
    :return:
    """
    gdf_explode = gdf.explode(index_parts=False)
    gdf_clean = gdf_explode[gdf_explode.geometry.type != "LineString"]
    if include_points:
        gdf_clean = gdf_clean[gdf_clean.geometry.type != "Point"]
    return gdf_clean.reset_index(drop=False).dissolve(by="index")


def clean_multipolygon_by_area(geom, area_min=20 * 20 * 2):
    """
    Remove sub-polygon from a MultiPolygon Geometry that are smaller than a given area

    :param geom: Multipolygon Geometry
    :param area_min: Minimum area to keep sub-geometry (in same unit as geom crs)
    :return: Multipolygon geometry
    """
    if geom.geom_type == "MultiPolygon":
        return shapely.MultiPolygon([p for p in geom.geoms if p.area > area_min])
    else:
        Warning("Input geometry is not a MultiPolygon. No modifications has been done.")
        return geom


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


def select_poly_from_multipoly(
    multi_poly, selection: Union[str, BaseGeometry] = "area"
) -> Polygon:
    """
    Select one single polygon among a MultiPolygon based on either:
        - its surfaces (return the largest)
        - a point coordinates
    :param multi_poly:
    :param selection: Method to be used for poly selection, can be either "Area" to select the largest polygon
    or a point coordinates (as a BaseGeometry) that must be contained by the polygon.

    :return: Polygon
    """
    if isinstance(multi_poly, Polygon):
        return multi_poly

    if not isinstance(multi_poly, MultiPolygon):
        raise ValueError(f"'geom' must be MultiPolygon, got {type(multi_poly)}")

    list_poly = list(multi_poly.geoms)

    if selection == "area":
        list_areas = [p.area for p in list_poly]
        poly_sel = np.array(list_poly)[list_areas == np.max(list_areas)][0]
        return poly_sel

    elif isinstance(selection, BaseGeometry):
        sel = [shapely.contains(p, selection) for p in list_poly]
        poly_sel = np.array(list_poly)[sel][0]
        return poly_sel

    else:
        raise IOError(
            f"'selection' must be either 'area' or a BaseGeometry, got {type(selection)}"
        )


# =====================================================================================================================
#                                               GEODATAFRAMES
# =====================================================================================================================
def df_to_gdf(df: pd.DataFrame, x_col: str, y_col: str, crs: str, drop_xy=False):
    """
    Transform a dataframe containing x and y coordinates to
    a GeoDataFrame with points geometry

    :param df: dataframe
    :param x_col: x column name
    :param y_col: y column name
    :param crs: crs projection
    :param drop_xy: If True, drop the orginal columns of x_col and y_col
    :return: geodataframe
    """

    if drop_xy:
        df_out = df.drop(columns=[x_col, y_col])
    else:
        df_out = df

    return gpd.GeoDataFrame(
        df_out, geometry=gpd.points_from_xy(df[x_col], df[y_col])
    ).set_crs(crs)


def clean_multipolygon_by_area_in_gdf(gdf: GeoDataFrame, area_min: float = 1):
    """
    Within a geodataframe, remove polygon from MultiPolygon that are below a given area.
    Conserve the geometry index and format of the original geodataframe

    :param gdf:
    :param area_min:
    :return:
    """
    if not isinstance(gdf, GeoDataFrame):
        raise IOError(f"gdf input must be a GeoDataframe, got {type(gdf)}.")

    # Explode all geometries to single polygon + get index as column
    gdf_explode = gdf.explode(ignore_index=False).reset_index(drop=False)

    # Select polygon by area and merge geometries back based on their original index
    return gdf_explode.loc[gdf_explode.area > area_min].dissolve(by="index")


def add_country_to_gdf(
    gdf: GeoDataFrame,
    country_shp: GeoDataFrame = None,
    country_col: str = "FID",
    renamed_country_col: str = None,
):
    """
    Add a country column to a GeodataFrame based on the largest intersection area with a given country shape file

    :param gdf: GeoDataframe where to add the country column
    :param country_shp: GeoDataFrame of the countries geometries
    :param country_col: Name of the country column to use in the country_shp
    :param renamed_country_col: (optional) New name of the country column in the output geodataframe
    :return:
    """
    if not isinstance(gdf, GeoDataFrame):
        raise IOError(f"'gdf' input must be a GeoDataframe, got {type(gdf)}.")

    if not isinstance(country_shp, GeoDataFrame):
        raise IOError(f"'country_shp' input must be a GeoDataframe, got {type(gdf)}.")

    # Make the intersection of the two dataframes, adding first the geometry of the countries as a simple column
    # Reset index to also get the index as a column
    gdf_join_countries = gdf.sjoin(
        country_shp.assign(geometry_country=lambda x: x.geometry),
        how="inner",
        predicate="intersects",
    ).reset_index(drop=False)

    # Add a country column based on the largest intersecting areas between the gdf and the country shape
    # To do so : add an intersection area column. Sort the lines by index (=single gdf geometry) and intersection area
    # in the descending order (largest first). Groupby index and get the first
    # Keep only original columns + the country one
    gdf_with_countries = (
        gdf_join_countries.assign(
            area_inter=lambda x: x["geometry"].intersection(x["geometry_country"]).area
        )
        .sort_values(by=["index", "area_inter"], ascending=[False, False])
        .groupby("index")
        .first()[[country_col] + list(gdf.columns)]
    ).set_geometry(col="geometry", crs=gdf.crs, inplace=False)

    # Rename country col if set
    if renamed_country_col is not None:
        gdf_with_countries.rename(
            columns={country_col: renamed_country_col}, inplace=True
        )

    return gdf_with_countries


def dissolve_without_merging(gdf):
    """

    :param gdf:
    :return:
    """
    # Create spatial join of the gdf overlapping with itself + pass the index as column
    # This gdf contains duplicates as geom X overlaps with geom Y and its reverse is also true
    gdf_overlapping = gdf.sjoin(gdf, how="inner", predicate="overlaps").reset_index(
        drop=False
    )

    # Add the geometry of the right geom as a column to be able to do operations more easily
    gdf_overlapping["geometry_right"] = gdf_overlapping.apply(
        lambda x: gdf_overlapping[
            gdf_overlapping["index_left"] == x["index_right"]
        ].geometry.iloc[0],
        axis=1,
    ).set_crs(gdf_overlapping.crs)

    # Add a column containing a sorted list of the geometries index that are overlapping at the given row
    # This way we can identify the paired duplicates within the dataframe and remove them
    gdf_overlapping["paired"] = gdf_overlapping.apply(
        lambda x: list(np.sort([x["index_left"], x["index_right"]])), axis=1
    )

    # Remove duplicates - since we want to clip out the overlapping regions only once
    # At this stage we have index_left (that are not present anymore in index_right), and for which we have all the
    # intersecting geometries in index_right/geometry_right
    gdf_no_duplicates = gdf_overlapping.loc[
        gdf_overlapping.duplicated(subset="paired", keep="last")
    ]

    # Trick to merge all existing overlapping regions of the same index_left geometry
    gdf_no_duplicates = (
        gdf_no_duplicates.assign(  # create a column of the overlapping geometry for each pair of geometry
            geom_intersection=gdf_no_duplicates.intersection(
                gdf_no_duplicates["geometry_right"]
            )
        )
        .set_geometry("geom_intersection")  # define this column as the new geometry
        .dissolve(
            by="index_left"
        )  # merge all the intersecting regions that concern the same input geom
    )
    gdf_no_duplicates.index.name = "index"  # rename index_left to index

    # Clean possible linestrings and points
    gdf_no_duplicates = remove_linestring(gdf_no_duplicates, include_points=True)

    # Replace original geometries with the one not overlapping
    gdf_no_overlap = gdf.copy()

    gdf_no_overlap.loc[
        gdf_no_overlap.index.isin(gdf_no_duplicates.index), "geometry"
    ] = (
        gdf_no_duplicates["geometry"]
        .set_crs(gdf_no_duplicates.crs)
        .difference(gdf_no_duplicates["geom_intersection"])
    )

    return gdf_no_overlap
