# A.Guerou - 27.09.2024
# Codes to select and process MNT tiles from IGN
# Either RGE Alti or LidarHD

import glob
import os
from typing import Union

import dask
import geopandas as gpd
import pandas as pd
import requests
import richdem as rd
from IPython.display import display

from pyce import shape as pyshape
from pyce.raster import merge_raster


# ===========================================
#  Identifications RGE Alti / LidarHD dataset
# ============================================
def identify_rge_tiles(
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    buffer: int = 0,
    df_metadata: dict = None,
    rge_folder: str = "",
    rge_type: str = "RGEALTI_MNT_5M_ASC_LAMB93_IGN",
    rge_col_name_tiles: str = "NOM_DALLE",
    plot: bool = False,
):
    """
    Identify RGE Alti tiles to built DEM around locations within a certain buffer distance.
    Usually takes a dataframe with locations and add two columns with tiles names and directory.
    !!! The RGE alti data need to be in a local directory !!!

    :param df: Dataframe or GeoDataFrame containing the locations.
               Only points are accepted when DataFrame is given
    :param rge_folder:  Local folder containing RGE Alti data
    :param buffer: Buffer zone to consider in meters
    :param df_metadata: Only used if DataFrame is given
                        Metadata of x, y columns names in df, and crs value
                        Dict keys must be : "x"=..., "y"=..., "crs"=...
    :param rge_type: Folder name suffix of RGE Alti data containing the footprint shapes
    :param rge_col_name_tiles: column name of the RGE alti data footprint shape
                               containing the name of the tile
    :param plot: If True, plot the selection
    :return: GeoDataFrame of the original dataframe with tiles names and folder of RGE Alti data

    """
    # Creates GeoDataFrame with geometries as points
    # ----------------------------------------------
    if not isinstance(df, gpd.GeoDataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"df must be part of Union[pd.DataFrame, gpd.GeoDataFrame], got :{type(df)}"
            )
        else:
            if df_metadata is None:
                df_metadata = dict(
                    x="X Lambert 2 Centre", y="Y Lambert 2 Centre", crs="EPSG:27572"
                )
            gdf_coords = pyshape.df_to_gdf(
                df,
                x_col=df_metadata["x"],
                y_col=df_metadata["y"],
                crs=df_metadata["crs"],
            )
    else:
        gdf_coords = df.copy()

    # Get list of shape files from RGE Alti containing the footprint of the tiles
    # One per French department, as distributed on their website
    # https://geoservices.ign.fr/rgealti
    # -------------------------------------------------------
    rge_footprint_list = glob.glob(
        os.path.join(
            rge_folder,
            f"*/RGEALTI/3_SUPPLEMENTS_LIVRAISON*/{rge_type}*/dalles.shp",
        )
    )

    # Create GeoDataFrame with all tiles + geometry
    # ---------------------------------------------
    gdf_tiles = gpd.GeoDataFrame()
    for rge_footprint in rge_footprint_list:
        footprint = gpd.read_file(rge_footprint).to_crs(gdf_coords.crs, inplace=False)
        footprint["DIR_DALLE"] = os.path.dirname(rge_footprint).replace(
            "3_SUPPLEMENTS_LIVRAISON", "1_DONNEES_LIVRAISON"
        )
        gdf_tiles = pd.concat([gdf_tiles, footprint])

    gdf_tiles.drop_duplicates(subset=[rge_col_name_tiles], inplace=True)
    gdf_tiles["geometry_mnt"] = gdf_tiles["geometry"]

    #  Process new GeoDataFrame: add MNT tiles columns
    # -------------------------------------------------
    if any(gdf_coords.columns.isin([rge_col_name_tiles])):
        gdf_coords.drop(columns=[rge_col_name_tiles], inplace=True)

    # Create buffer zones
    if buffer != 0:
        gdf_coords["geometry"] = gdf_coords.buffer(buffer)

    # Get intersections between coords+buffer and all tiles
    gdf_sel = gdf_coords.sjoin(gdf_tiles, how="inner", predicate="intersects")
    gdf_sel = gdf_sel[
        gdf_coords.columns.union(
            [rge_col_name_tiles, "geometry_mnt", "DIR_DALLE"], sort=False
        )
    ]

    # Sort by first columns (usually the sites/lakes) and reset index
    gdf_sel.drop_duplicates(
        subset=[gdf_sel.columns[0], rge_col_name_tiles]
    ).reset_index(
        drop=True, inplace=True
    )  # Some tiles are contains in multiple regions
    gdf_sel.sort_values(gdf_sel.columns[0], inplace=True)
    gdf_sel.reset_index(drop=True, inplace=True)

    if plot:
        m = gdf_sel.explore()
        m = gdf_sel.set_geometry("geometry_mnt").explore(m=m, color="red")
        display(m)

    return gdf_sel.drop(columns=["geometry", "geometry_mnt"])


def identify_lidarhd_tiles(
    df, gdf_lidar, buffer: int = 0, df_metadata: dict = None, plot: bool = False
):
    """
    Identify LidarHD tiles that intersects a list of geometries.
    Geometries can be either a list of coordinates, or proper shapes

    The lidarHD tiles reference commonly used has been downloaded in 2023
    while the IGN website lidar page was under construction:
    "/home/aguerou/data/mnt/ign/lidar_hd/grille/TA_diff_pkk_lidarhd_classe.shp"
    It is NOT available anymore in 2024. An interactive process online
    is now (09/2024) used: https://diffusion-lidarhd.ign.fr/

    :param df: Dataframe or GeoDataFrame containing the list of coordinates
               Dataframe must contain a column of X, and Y coordinates
    :param gdf_lidar: Shapefile of the lidarHD tiles geometries
    :param buffer: Add a buffer zone to the input geometries before
                   deriving the intersections with the lidarHD tiles
    :param df_metadata: Only used if DataFrame is given
                        Metadata of x, y columns names in df, and crs value
                        Dict keys must be : "x"=..., "y"=..., "crs"=...
    :param plot: If True, plot the selection
    :return: GeoDataFrame of the original dataframe with tiles names
             and URL link of lidarHD tiles
    """
    # Creates GeoDataFrame with geometries as points
    # ----------------------------------------------
    if not isinstance(df, gpd.GeoDataFrame):
        if isinstance(df, pd.DataFrame):
            if df_metadata is None:
                df_metadata = dict(
                    x="X Lambert 2 Centre", y="Y Lambert 2 Centre", crs="EPSG:27572"
                )
            gdf_coords = pyshape.df_to_gdf(
                df,
                x_col=df_metadata["x"],
                y_col=df_metadata["y"],
                crs=df_metadata["crs"],
            )
        else:
            raise TypeError(
                f"df must be part of Union[pd.DataFrame, gpd.GeoDataFrame], got :{type(df)}"
            )
    else:
        gdf_coords = df.copy()

    # Intersections gdf
    # -----------------
    if buffer != 0:
        gdf_coords["geometry"] = gdf_coords.buffer(buffer)

    # Copy lidar + geometry col with other name to keep it after sjoin
    if not isinstance(gdf_lidar, gpd.GeoDataFrame):
        raise TypeError(f"'gdf_lidar' must be a GeoDataFrame, got: {type(gdf_lidar)}")

    gdf_hd = gdf_lidar.copy()
    gdf_hd["geometry_mnt"] = gdf_hd["geometry"]

    # Intersection
    gdf_sel = gdf_coords.to_crs(gdf_hd.crs).sjoin(
        gdf_hd, how="inner", predicate="intersects"
    )
    gdf_sel.drop(columns=["index_right"], inplace=True)

    # Sort by first columns (usually the sites/lakes) and reset index
    gdf_sel.sort_values(gdf_sel.columns[0], inplace=True)
    gdf_sel.reset_index(drop=True, inplace=True)

    #             Checks
    # ----------------------------------------
    if plot:
        gdf_sel["geometry"] = gdf_sel.buffer(-buffer + 5)
        m = gdf_sel.explore()
        m = gdf_sel.set_geometry("geometry_mnt").explore(m=m, color="red")
        display(m)

    return gdf_sel[df.columns.union(gdf_lidar.columns[gdf_lidar.columns != "geometry"])]


def download_lidarhd_tiles(
    df: pd.DataFrame,
    dwnld_dir: str = None,
    num_workers=6,
    force=False,
    compute=True,
    col_name_tiles: str = "nom_pkk",
    col_name_link: str = "url_telech",
):
    """
    Download lidarHD tiles from IGN website based on a list of URL links

    :param df: Dataframe containing the list of URL links and tiles names
    :param dwnld_dir: Directory where to download tiles
    :param num_workers: Number of cores to used
    :param force: If True, force downloading and overwriting potential
                  existing file with the same name
    :param compute: If False, do not download files but indicate how many
                    files will be downloaded
    :param col_name_tiles: Name of the column containing the tiles name
    :param col_name_link: Name of the column containing the URL links
    """

    @dask.delayed
    def _request_url(url, path_out):
        r = requests.get(url, allow_redirects=True)
        open(path_out, "wb").write(r.content)
        print(f"Download done: {url}")

    # Checks
    if dwnld_dir is None and compute is True:
        raise ValueError(f"dwnld_dir must be specified")

    # Drop duplicates
    df_to_dwnld = df.drop_duplicates(subset=[col_name_link])

    # Download with dask
    request_list: list[None] = []

    for n_tile in range(df_to_dwnld[col_name_link].shape[0]):
        name_tile = df_to_dwnld[col_name_tiles].iloc[n_tile]
        url_tile = df_to_dwnld[col_name_link].iloc[n_tile]
        tile_path = os.path.join(dwnld_dir, name_tile)

        if not os.path.exists(tile_path) or force is True:
            request_list.append(_request_url(url=url_tile, path_out=tile_path))

    print(f"Number of tiles to download: {len(request_list)}")

    if compute:
        print("Download will start ...")
        dask.compute(request_list, num_workers=num_workers)


# ============================================
#             LIDAR processing
# ============================================
def _save_name(dir: str = None, file_name: str = None, extension: str = None):
    """
    Return complete path dir + file_name + extension

    :param dir: directory name
    :param file_name: file name
    :param extension:  extension (with the .)
    :return: path dir + file_name + extension
    """
    return os.path.join(dir, file_name + extension)


def _get_lidar_basename(file: str):
    """
    Return the basename of lidarHD files without the extensions (.copc.laz)

    :param file: file name
    :return: str containing the file basename
    """
    return os.path.splitext(os.path.splitext(os.path.basename(file))[0])[0]


def process_lidarhd_tiles(
    files_list, save_dir, resolution=1, save_extension=".tif", num_workers=3, redo=False
):
    """
    Compute DEM tile from lidarHD files list, using parallel processing

    It runs the QGIS function 'pdal_wrench to_raster_tin' that is the function
    used in QGIS 3.34 version for 'Export to raster (using triangulation)' function
    in the toolbox

    :param files_list: list of lidarHD tiles absolute path
    :param save_dir: Directory where to save the DEM file
    :param resolution: Resolution of the output DEM file, in meter
    :param save_extension: Extension of the DEM file
    :param num_workers: NUmber of cores to used for parallel processing
    :param redo: if True, force computation even if processed tile already exist

    """
    # Check if MNT has been already processed
    # Return only non-processed ones / or at different resolutions
    if redo is False:
        files_list = [
            file
            for file in files_list
            if not os.path.exists(
                _save_name(
                    dir=save_dir,
                    file_name=_get_lidar_basename(file),
                    extension=save_extension,
                )
            )
        ]
        if len(files_list) == 0:
            print("All files were already processed")

    # Create a dask list of lidar processing
    results = []
    for file in files_list:
        results.append(
            _pdal_raster_tin(
                file,
                resolution=resolution,
                save_dir=save_dir,
                save_extension=save_extension,
            )
        )

    # Compute the MNT tiles
    mnt = dask.compute(results, num_workers=num_workers)


def merge_lidarhd_tiles(
    file_list: list = None,
    save_dir: str = None,
    filter: str = "'Classification == 2'",
    compute=True,
    qgis_dir="/home/aguerou/miniconda3/envs/py311/lib/qgis",
):
    """
    Merge two or more classified lidar tiles to a single one in .laz format.

    It runs the QGIS function 'pdal_wrench merge' that is the function used
    in QGIS 3.34 version for 'merge' function of the 'Point Cloud data management'
    section of the toolbox

    :param file_list: List of lidar tiles to merge. They must be lidar format (.copc.laz)
    :param save_dir: Directory where to seave the merged file
    :param filter: Expression used to filter input lidar data before the merge. Used to select ground, vegetation, etc.
                   Default: "Classification == 2" to select only the ground points.
    :param compute: If False, return only the name of the merged file without conputing the merge. Default = True
    :param qgis_dir: Directory where the function 'pdal_wrench merge' is installed

    :return: Full path of the merged file.
             The name of the merged file is constructed automatically as follow: it gets the second tiles number of
             each file and concatenate them into one single file name using the first file name as a base.
             Exemple: files_list = ["LHD_FXX_0985_6480_PTS_O_LAMB93_IGN69",
                                    "LHD_FXX_0985_6481_PTS_O_LAMB93_IGN69"]
                    -> out_name = "LHD_FXX_0985_6480_6481_PTS_O_LAMB93_IGN69"

    """
    # Create automatically the name of the merged file
    # It gets the second tiles number of each file and concatenate them in one single file name
    # Exemple: files_list = ["LHD_FXX_0985_6480_PTS_O_LAMB93_IGN69",
    #                        "LHD_FXX_0985_6481_PTS_O_LAMB93_IGN69"]
    #                        -> out_name = "LHD_FXX_0985_6480_6481_PTS_O_LAMB93_IGN69"
    # =======================================================================================
    # Get first file name in a list, with first item being the 4 first group linked with _
    out_name_base_split = _get_lidar_basename(file_list[0]).rsplit("_", 4)

    # Get extensions number of all other files
    extra_names = [_get_lidar_basename(file).split("_")[3] for file in file_list[1:]]

    # Add extension to first item of the list, joining them with "_"
    out_name_base_split[0] += "_" + "_".join(extra_names)

    # Concatenate back the first file name list joining each group with "_"
    out_name = "_".join(out_name_base_split)

    # Create save_name
    save_name = _save_name(dir=save_dir, file_name=out_name, extension=".laz")

    # Create tmp text containing the file list
    # ========================================
    input_file_list = os.path.join(save_dir, "merge_list_tmp.txt")
    tmp_file = open(input_file_list, "w")

    for i in range(len(file_list)):
        tmp_file.write(file_list[i])
        tmp_file.write("\n")

    tmp_file.close()

    # Compute DEM from lidar "ground" points (Classif=2) / for IGN
    # ============================================================
    pdal_cmd = f"{qgis_dir}/pdal_wrench merge\
    --input-file-list={input_file_list}\
    --output={save_name}\
    --filter={filter}\
    --threads=20"

    # Execute command in OS + remove tmp file
    if compute:
        os.system(pdal_cmd)
    os.system(f"rm {input_file_list} -f")

    return save_name


def merge_dem_tiles(
    dem: Union[str, list, pd.DataFrame],
    df_metadata: dict = None,
    epsg_mnt: str = None,
    save_name: str = None,
):
    # Check type of input
    # ===================
    if isinstance(dem, str) or isinstance(dem, list):
        try:
            dem_rioxr = merge_raster(dem, epsg=epsg_mnt)
            return dem_rioxr

        except Exception as e:
            print(e)
    elif not isinstance(dem, pd.DataFrame):
        raise ValueError(
            f"'dem' type must be within Union[str, list, pd.DataFrame], got {type(dem)}"
        )

    # Deal with metadata in case of a dataframe
    # =========================================
    metadata_col = ["col_name_dir", "col_name_tile", "tile_extension"]

    # Default
    if df_metadata is None:
        df_metadata = dict(
            col_name_dir="DIR_DALLE",
            col_name_tile="NOM_DALLE",
            tile_extension=".asc",
        )

    # Check validity of metadata name given vs metadata needed
    if not all(pd.Series(df_metadata.keys()).isin(metadata_col)):
        raise IOError(f"df_metadata must contains the following keys: {metadata_col}")

    # Check validaity of dataframe columns name
    for key, item in df_metadata.items():
        if key.startswith("col"):
            if not any(dem.columns.isin([item])):
                raise IOError(f"'dem' dataframe column '{item}' not found")

    # Select mnt raster
    # =================
    list_dem = [
        os.path.join(dir_tile, name_tile)
        for dir_tile, name_tile in zip(
            dem[df_metadata["col_name_dir"]],
            dem[df_metadata["col_name_tile"]] + df_metadata["tile_extension"],
        )
    ]

    # Merge the list of files
    # =======================
    merged_dem = merge_raster(list_dem, epsg=epsg_mnt)

    # Save file
    if save_name is not None:
        merged_dem.rio.to_raster(save_name)

    return merged_dem


@dask.delayed
def _pdal_raster_tin(
    file,
    resolution=0.5,
    save_dir=None,
    save_extension=None,
    filter: str = "'Classification == 2'",
    qgis_dir="/home/aguerou/miniconda3/envs/py311/lib/qgis",
):
    """
    Dask function to compute DEM tile from lidarHD data.
    It runs the QGIS function 'pdal_wrench to_raster_tin' that is the function
    used in QGIS 3.34 version for 'Export to raster (using triangulation)' function
    in the toolbox

    :param file: Absolute path of the lidarHD tiles to process
    :param resolution: Resolution of the output DEM file, in meter
    :param save_dir: Directory where to save the DEM file
    :param save_extension: Extension of the DEM file
    :param qgis_dir: Directory where the function 'pdal_wrench to_raster_tin'
                     is installed
    """
    file_name = _get_lidar_basename(file)
    save_name = _save_name(dir=save_dir, file_name=file_name, extension=save_extension)

    # Compute DEM from lidar "ground" points (Classif=2) / for IGN
    pdal_cmd = f"{qgis_dir}/pdal_wrench to_raster_tin\
    --input={file}\
    --output={save_name}\
    --resolution={resolution}\
    --tile-size=1000\
    --filter={filter}\
    --threads=20"

    # Execute command in OS
    os.system(pdal_cmd)


def rd_terrain_slope_and_aspect(
    mnt_file: str, projection: str, save_file_slope=None, save_file_aspect=None
):
    # Open file and add projection
    rda = rd.LoadGDAL(mnt_file)
    rda.projection = projection

    # Derive slope
    slope = rd.TerrainAttribute(rda, attrib="slope_degrees")

    # Set points with no slope to no data / to get no orientation
    rda[slope == 0.0] = rda.no_data

    # Derive aspect
    aspect = rd.TerrainAttribute(rda, attrib="aspect")

    # Savings
    if save_file_slope is not None:
        rd.SaveGDAL(save_file_slope, slope)
    if save_file_aspect is not None:
        rd.SaveGDAL(save_file_aspect, aspect)
