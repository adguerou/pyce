import copy
from importlib import reload
from typing import Union

import numpy as np
import pandas as pd
import xarray
from matplotlib.colors import BoundaryNorm, ListedColormap
from pyce.tools import lc_colormaps

reload(lc_colormaps)

# GLOBAL PARAMATERS
# =================
_type_name = "Type"
_code_name = "Code"
_color_name = "Color"

_lcm_colors = "colors"
_lcm_codes_to_mask = "codes_to_mask"
_lcm_mask_val = "mask_val"
_lcm_reindex = "reindex"

_mapping_kwargs = "mapping_kwargs"

# Dictionaries of Land Cover Map settings
# ========================================
oso17 = {
    _mapping_kwargs: {"col_type": 0, "col_code": 1, "header": None, "sep": ":"},
    _lcm_colors: lc_colormaps.colors_oso17,
    _lcm_codes_to_mask: [11, 12, 41, 42, 43, 44, 46, 221, 222],
    _lcm_mask_val: 223,
    _lcm_reindex: [6, 4, 1, 0, 3, 7, 2, 5, 8],
}

oso18 = {
    _mapping_kwargs: {"col_type": 0, "col_code": 1, "header": None, "sep": ":"},
    _lcm_colors: lc_colormaps.colors_oso18,
    _lcm_codes_to_mask: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 21, 255],
    _lcm_mask_val: 223,
    _lcm_reindex: [6, 5, 2, 1, 4, 0, 3, 7, 8],
}

h1a = {
    _mapping_kwargs: {"col_type": 0, "col_code": 1, "header": None, "sep": ":"},
    _lcm_colors: lc_colormaps.colors_h1a,
    _lcm_codes_to_mask: None,
    _lcm_mask_val: None,
    _lcm_reindex: [6, 3, 2, 0, 1, 4, 5],
}

h1a_rf = {
    _mapping_kwargs: {"col_type": 0, "col_code": 1, "header": None, "sep": ":"},
    _lcm_colors: lc_colormaps.colors_h1a_rf,
    _lcm_codes_to_mask: None,
    _lcm_mask_val: None,
    _lcm_reindex: [4, 0, 3, 2, 1, 6, 5],
}

h1b = {
    _mapping_kwargs: {"col_type": 0, "col_code": 1, "header": None, "sep": ":"},
    _lcm_colors: lc_colormaps.colors_h1b,
    _lcm_codes_to_mask: [10],
    _lcm_mask_val: None,
    _lcm_reindex: [3, 2, 1, 6, 0, 4, 5],
}

h1b_paper = {
    _mapping_kwargs: {"col_type": 0, "col_code": 1, "header": None, "sep": ":"},
    _lcm_colors: lc_colormaps.colors_h1b_paper,
    _lcm_codes_to_mask: None,
    _lcm_mask_val: None,
    _lcm_reindex: [3, 2, 1, 6, 0, 4, 5, 8, 9],
}

prioritice = {
    _mapping_kwargs: {"col_type": 0, "col_code": 1, "header": None, "sep": ":"},
    _lcm_colors: lc_colormaps.prioritice,
    _lcm_codes_to_mask: None,
    _lcm_mask_val: None,
    _lcm_reindex: [0, 1, 2, 3],
}

s2glc = {
    _mapping_kwargs: {
        "col_type": 5,
        "col_code": 0,
        "header": None,
        "comment": "#",
        "skiprows": 7,
    },
    _lcm_colors: lc_colormaps.colors_s2glc,
    _lcm_codes_to_mask: [0, 62, 73, 75, 105, 106, 255],
    _lcm_mask_val: 223,
    _lcm_reindex: [6, 5, 1, 0, 3, 4, 2, 7, 8],
}

tp_lc10 = {
    _mapping_kwargs: {"col_type": 0, "col_code": 1, "header": None, "sep": ":"},
    _lcm_colors: lc_colormaps.colors_tp_lc10,
    _lcm_codes_to_mask: None,
    _lcm_mask_val: None,
    _lcm_reindex: None,
}

# Dictionnary that compile all mappings
# --------------------------------------
dict_lc_maps = {
    "oso17": oso17,
    "oso18": oso18,
    "h1a": h1a,
    "h1a_rf": h1a_rf,
    "s2glc": s2glc,
    "h1b": h1b,
    "h1b_paper": h1b_paper,
    "tp_lc10": tp_lc10,
    "prioritice": prioritice,
}


# CLASSES
# =======
class LandCoverMap:
    """
    Class to handle a Land Cover Mapping based on panda.DataFrame
    """

    def __init__(
        self, df: Union[pd.DataFrame, str] = None, colors: list = None, name: str = None
    ):
        # Give a name
        self.name = name
        self.codes_masked = None
        self.mask_val = None
        self.index = None

        # Create dataframe from direct object or from file
        if df is None:
            raise ValueError("df must be provided")
        if isinstance(df, pd.DataFrame):
            self.df = df
        elif isinstance(df, str):
            if name in dict_lc_maps:
                kwargs = dict_lc_maps[name][_mapping_kwargs]
                self.df = get_lc_mapping(df, **kwargs)
            elif name is not None:
                raise ValueError(f"Name: {name} not recognized in lc_mapping.py")
            elif name is None:
                raise ValueError(f"A colormap name (name=) must be provided")

        # Set colors column of dataframe
        # + group type column
        # + modify codes_masked/mask_val
        self._set_colors(colors)
        self._set_param_from_name()

    def _set_colors(self, colors: list = None):
        if colors is not None:
            if self.df.shape[0] != len(colors):
                raise ValueError(
                    "The list 'colors' must have the same length"
                    "that the number of rows of 'df'. "
                    f"{self.df.shape[0]} rows for 'df' and {len(colors)} colors given"
                )
            self.df[f"{_color_name}"] = colors
        else:
            self.df[f"{_color_name}"] = np.nan

    def _set_param_from_name(self):
        if self.name in dict_lc_maps:
            param = dict_lc_maps[self.name]

            self._set_colors(param[_lcm_colors])
            self.index = param[_lcm_reindex]

            if param[_lcm_codes_to_mask] is not None:
                self.group_to(
                    codes=param[_lcm_codes_to_mask],
                    group_type="Other class",
                    group_code=param[_lcm_mask_val],
                    group_color="#000000",
                )
                self.codes_masked = param[_lcm_codes_to_mask]
                self.mask_val = param[_lcm_mask_val]

        elif self.name is not None:
            Warning(f"Name: {self.name} not recognized. LandCoverMap not parametrized.")

    def get_type(self) -> np.array:
        """
        Return the column 'Type' du pd.DataFrame
        :return: np.array
        """
        return self.df[f"{_type_name}"].to_numpy()

    def get_code(self) -> np.array:
        """
        Return the column 'Code' du pd.DataFrame
        :return: np.array
        """
        return self.df[f"{_code_name}"].to_numpy()

    def get_colors(self) -> np.array:
        """
        Return the column 'Color' du pd.DataFrame
        :return: np.array
        """
        return self.df[f"{_color_name}"].to_numpy()

    def get_color_of_code(self, code=None) -> str:
        """
        Return the 'Color' value corresponding to 'Code'
        :return: str
        """
        return self.df.loc[self.df[f"{_code_name}"] == code][f"{_color_name}"].iloc[0]

    def get_color_of_type(self, typ=None) -> str:
        """
        Return the 'Color' value corresponding to 'Code'
        :return: np.array
        """
        return self.df.loc[self.df[f"{_type_name}"] == typ][f"{_color_name}"].iloc[0]

    def get_cmap(self, cmap_name: str) -> ListedColormap:
        """
        Create a colormap based on the 'Color' column
        :param cmap_name:
        :return: ListedColormap
        """
        return ListedColormap(self.get_colors(), name=cmap_name)

    def get_bins(self) -> np.array:
        """
        Create a numpy array corresponding to the Type bins values.
        :return: np.array
        """
        bins = self.get_code()
        bins = np.append(bins, bins[-1] + 1)

        return bins

    def get_norm(self) -> BoundaryNorm:
        """
        Return a BoundaryNorm object that is useful to set
        the color limits of the colormap for each type
        :return:
        """
        bins = self.get_bins()
        return BoundaryNorm(bins, np.size(bins) - 1)

    def get_type_of_code(self, code: int = None) -> str:
        """
        Return the column 'Type' having the code given in parameter
        :return: str
        """
        if code is None:
            raise IOError("parameter 'code' missing")

        return self.df.loc[self.df[f"{_code_name}"] == code][f"{_type_name}"].iloc[0]

    def get_code_of_type(self, type: str = None) -> int:
        """
        Return the column 'Type' having the code given in parameter
        :return: int
        """
        if type is None:
            raise IOError("parameter 'type' missing")

        return self.df.loc[self.df[f"{_type_name}"] == type][f"{_code_name}"].iloc[0]

    def group_to(
        self,
        codes: list[int],
        group_type: str,
        group_code: int,
        group_color: str,
        in_place: bool = True,
        _code_name: str = _code_name,
    ):
        """

        :param codes:
        :param group_type:
        :param group_code:
        :param group_color:
        :param in_place:
        :param _code_name:
        :return:
        """
        if in_place:
            self.remove_item(col_name=_code_name, col_val=codes, in_place=in_place)
            self.add_item(
                type=group_type, code=group_code, color=group_color, in_place=in_place
            )
        else:
            lc_map = copy.deepcopy(self)
            lc_map.remove_item(col_name=_code_name, col_val=codes, in_place=in_place)
            lc_map.add_item(
                type=group_type, code=group_code, color=group_color, in_place=in_place
            )
            return lc_map

    def remove_item(
        self,
        col_name: str,
        col_val: list[int],
        in_place: bool = False,
    ):
        """
        Remove a line from a LandCoverMap dataframe based on values of a given column
        :param col_name: name of the column of the dataframe
        :param col_val: The col_name value of the item to be removed.
                        Can be a single value or a list
        :param in_place: If True, modify the LandCoverMap
        :return:
        """

        # TODO: Remove the in_place for remove_item as it modify the dict_lc_maps itself and create bug situation when creating another lcamp with the same name, index have disapearred
        #
        if in_place is True:
            for val in col_val:
                del self.index[self.index.index(val)]
            self.df = self.df.loc[~self.df[col_name].isin(col_val)]
        else:
            lc_map = copy.deepcopy(self)
            for val in col_val:
                del lc_map.index[lc_map.index.index(val)]
            lc_map.df = lc_map.df.loc[~lc_map.df[col_name].isin(col_val)]

            return lc_map

    def add_item(
        self,
        type: Union[str, list[str]],
        code: Union[int, list[int]],
        color: Union[str, list[str]],
        in_place: bool = True,
    ):
        """
        Add
        :param type:
        :param code:
        :param color:
        :param in_place:
        :return:
        """
        new_item = pd.DataFrame(
            {_type_name: [type], _code_name: [code], _color_name: [color]}
        )
        if in_place is True:
            self.df = pd.concat([self.df, new_item], ignore_index=True)
            self.df.reset_index()
        else:
            lc_map = copy.deepcopy(self)
            lc_map.df = pd.concat([lc_map.df, new_item], ignore_index=True)
            lc_map.df.reset_index()
            return lc_map

    def reindex_from_list(self, index_list: list[str], in_place=False):
        """
        Reindex dataframe based on a list of index values
        :param index_list:
        :param in_place:
        :return:
        """
        if len(index_list) != self.df.shape[0]:
            raise ValueError(
                "Lenght of 'index_list' must be equal to the LandCoverMap rows"
            )
        if in_place:
            self.df = self.df.reindex(index=index_list)
            self.df.reset_index(drop=True, inplace=True)
        else:
            lc_map = copy.deepcopy(self)
            lc_map.df = lc_map.df.reindex(index=index_list)
            lc_map.df.reset_index(drop=True, inplace=True)
            return lc_map

    def reindex_from_col_val(
        self, values: list, col_name: str = _code_name, in_place=False
    ):
        """
        Reindex dataframe based on a given column values order

        :param values: Values in the order to be used for reindexing
        :param col_name: Name of the dataframe columns. "Code" by default
        :param in_place: Return a new LandCoverMap is False, otherwise change directly the object
        :return: self or new landCoverMap
        """
        index_list = [
            self.df.index[self.df[col_name] == val].values[0] for val in values
        ]
        if in_place:
            self.reindex_from_list(index_list, in_place=in_place)
        else:
            return self.reindex_from_list(index_list, in_place=in_place)

    def reindex(self, reverse=False, in_place=False):
        """
        Reindex dataframe based on the default index values of the LandCoverMap
        :param reverse:
        :param in_place:
        :return:
        """
        if not reverse:
            if in_place:
                self.reindex_from_col_val(
                    values=self.index, col_name=_code_name, in_place=in_place
                )
            else:
                return self.reindex_from_col_val(
                    values=self.index, col_name=_code_name, in_place=in_place
                )
        else:
            if in_place:
                self.reindex_from_col_val(
                    values=self.index[::-1], col_name=_code_name, in_place=in_place
                )
            else:
                return self.reindex_from_col_val(
                    values=self.index[::-1], col_name=_code_name, in_place=in_place
                )

    def swap_rows_from_index(self, id1, id2, in_place=False):
        """
        Swap two rows of the lcmap dataframe based on their index values

        :param id1: first index to swap
        :param id2: second index to swap with
        :param in_place: If True, modify the dataframe, otherwise return a new object
          !!! not working properly so far !!!!
        :return: LandCoverMap object if in_place is False
        """
        # TODO: in_place is not working properly
        row_1 = self.df.iloc[id1].copy()
        row_2 = self.df.iloc[id2].copy()

        if in_place:
            self.df.iloc[id1] = row_2
            self.df.iloc[id2] = row_1
        else:
            lc_map = copy.copy(self)
            lc_map.df.iloc[id1] = row_2
            lc_map.df.iloc[id2] = row_1

            return lc_map

    def swap_rows_from_col_val(self, col_name, val1, val2, in_place=False):
        """
        Swap two rows of the lcmap dataframe based on their values of a given column

        :param col_name: Name of the dataframe column to use
        :param val1: first column value of the row to swap
        :param val2: second column value of the row to swap
        :param in_place: If True, modify the dataframe, otherwise return a new object
            !!! not working properly so far !!!!
        :return: LandCoverMap object if in_place is False
        """
        # TODO: in_place is not working properly
        id_1 = self.df.loc[self.df[col_name] == val1].index
        id_2 = self.df.loc[self.df[col_name] == val2].index

        if in_place:
            self.swap_rows_from_index(id_1, id_2, in_place=in_place)
        else:
            return self.swap_rows_from_index(id_1, id_2, in_place=in_place)

    def clip_from_ds(self, ds: xarray, in_place=False):
        """
        Remove all rows (e.g. based on the code value) not present in dataset

        :param ds:
        :param in_place:
        :return:
        """
        sel = np.unique(ds.data[~np.isnan(ds.data)])

        if in_place is True:
            self.df = self.df.loc[self.df[_code_name].isin(sel)]
        else:
            lc_map = copy.copy(self)
            lc_map.df = lc_map.df.loc[lc_map.df[_code_name].isin(sel)]
            return lc_map

    def clip_from_df(self, df: pd.Series, in_place=False):
        """
        Remove all rows (e.g. based on the code value) not present in pandas Series

        :param df:
        :param in_place:
        :return:
        """
        sel = np.unique(df.loc[~np.isnan(df)])

        if in_place is True:
            self.df = self.df.loc[self.df[_code_name].isin(sel)]
        else:
            lc_map = copy.deepcopy(self)
            lc_map.df = lc_map.df.loc[lc_map.df[_code_name].isin(sel)]
            return lc_map


# FUNCTIONS
# =========
def get_lc_mapping(
    file, col_type: int = None, col_code: int = None, **kwargs
) -> pd.DataFrame:
    """
    Read a csv file containing a land cover nomenclature and return a
     panda.DataFrame with two columns, one with the type of land cover and
     one with the associated code number

    :param file:
    :param col_type: the index number of the column containing the land cover type
    :param col_code: the index number of the column containing the land cover code
    :**kwargs: kwargs of the pd.read_csv functions

    :return: pd.DataFrame
    """
    df = pd.read_csv(file, **kwargs)

    if col_type is None:
        raise ValueError(f"col_type is missing")
    if col_code is None:
        raise ValueError(f"col_code is missing")

    dict_mapping = {
        f"{_type_name}": df.iloc[:, col_type],
        f"{_code_name}": df.iloc[:, col_code],
    }
    df_mapping = pd.DataFrame(data=dict_mapping)

    return df_mapping


def oso_mapping_fusion_in_df(
    df: pd.DataFrame, lc_map_to: LandCoverMap, lc_map_from: LandCoverMap
):
    lc_map_to.reindex(in_place=True)
    lc_map_from.reindex(in_place=True)

    for i in lc_map_to.df.index:
        code_from = lc_map_from.df.iloc[i][_code_name]
        code_to = lc_map_to.df.iloc[i][_code_name]
        df.loc[
            df.index.get_level_values("datetime") > pd.to_datetime("2017-01-01"),
            f"LC_{code_to}",
        ] = df.loc[
            df.index.get_level_values("datetime") > pd.to_datetime("2017-01-01"),
            f"LC_{code_from}",
        ]
    return df.drop(
        labels=[f"LC_{code}" for code in lc_map_from.get_code()[:-1]], axis=1
    )


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
        df = copy.deepcopy(df)

    if prefix:
        df[col] = df[col].apply(lambda x: lcmap.get_type_of_code(int(x[-1])))
    else:
        df[col] = df[col].apply(lambda x: lcmap.get_type_of_code(x))

    if inplace is False:
        return df
