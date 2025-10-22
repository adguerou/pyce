import geopandas as gpd
import pandas as pd


def add_length(df, factor=1e3):
    df["length"] = df["geometry"].length / factor
    return df


def get_runs_length(
    df,
    zone=None,
    infra="runs",
    country_col="FID",
    country_name="Country",
    save_name=None,
):
    df = df.copy()

    df.loc[df["uses"].isin([["skitour"]]), "uses_class"] = "skitour"
    df.loc[
        (~df["uses"].isin([["skitour"]])) & (df.grooming == "backcountry"), "uses_class"
    ] = "backcountry"
    df.loc[
        (~df["uses"].isin([["skitour"]])) & (df.grooming != "backcountry"), "uses_class"
    ] = "groomed"

    df.drop(columns="uses", inplace=True)
    df.rename(columns={"uses_class": "uses", country_col: country_name}, inplace=True)

    df = add_length(df)

    if save_name is not None:
        df.to_parquet(save_name)

    runs_country = (
        df[[country_name, "uses", "geometry", "length"]]
        .dissolve(by=[country_name, "uses"], aggfunc="sum")
        .drop(columns=["geometry"])
        .reset_index()
    )
    runs_country["zone"] = zone
    runs_country["infra"] = infra

    runs_country_sum = (
        df[[country_name, "geometry", "length"]]
        .dissolve(by=[country_name], aggfunc="sum")
        .drop(columns=["geometry"])
        .reset_index()
    )
    runs_country_sum["uses"] = "Total"
    runs_country_sum["zone"] = zone
    runs_country_sum["infra"] = infra

    runs_europe = (
        df[["uses", "geometry", "length"]]
        .dissolve(by=["uses"], aggfunc="sum")
        .drop(columns=["geometry"])
        .reset_index()
    )
    runs_europe[country_name] = "ALL"
    runs_europe["zone"] = zone
    runs_europe["infra"] = infra

    runs_europe_sum = (
        df[["geometry", "length"]]
        .dissolve(aggfunc="sum")
        .drop(columns=["geometry"])
        .reset_index()
    )
    runs_europe_sum[country_name] = "ALL"
    runs_europe_sum["uses"] = "Total"
    runs_europe_sum["zone"] = zone
    runs_europe_sum["infra"] = infra

    return pd.concat(
        [runs_country, runs_country_sum, runs_europe, runs_europe_sum]
    ).drop(columns=["index"])


def get_lifts_length(
    df,
    zone=None,
    infra="lifts",
    country_col="FID",
    country_name="Country",
    save_name=None,
):
    df = df.copy()

    df.rename(columns={country_col: country_name}, inplace=True)

    df = add_length(df)

    if save_name is not None:
        df.to_parquet(save_name)

    lifts_country = (
        df[[country_name, "geometry", "length"]]
        .dissolve(by=[country_name], aggfunc="sum")
        .drop(columns=["geometry"])
        .reset_index()
    )
    lifts_country["uses"] = infra
    lifts_country["zone"] = zone
    lifts_country["infra"] = infra

    lifts_europe = (
        df[["geometry", "length"]]
        .dissolve(aggfunc="sum")
        .drop(columns=["geometry"])
        .reset_index()
    )
    lifts_europe[country_name] = "ALL"
    lifts_europe["uses"] = infra
    lifts_europe["zone"] = zone
    lifts_europe["infra"] = infra

    return pd.concat([lifts_country, lifts_europe]).drop(columns=["index"])


def get_pp_surf_and_perc(
    pp_shp: gpd.GeoDataFrame,
    reg_shp: gpd.GeoDataFrame,
    categories: list[str] = None,
    category_sum: str = None,
):
    # Intersections between pp_shp and reg_shp
    pp_reg = pp_shp.overlay(
        reg_shp[["geometry"]], how="intersection", keep_geom_type=False
    )

    # Get surface of PP reg
    pp_reg_diss = pp_reg.dissolve(by=categories)
    pp_reg_surf = (
        pp_reg_diss.assign(area=pp_reg_diss.area)
        .groupby(by=categories)
        .sum(["area"])["area"]
        / 1e6
    )

    pp_reg_diss_sum = pp_reg.dissolve(
        by=category_sum
    )  # Dissolve global shp to merge overlapping surfaces within categories
    pp_reg_surf_sum = (
        pp_reg_diss_sum.assign(area=pp_reg_diss_sum.area)
        .groupby(by=category_sum)
        .sum(["area"])["area"]
        / 1e6
    )

    pp_reg_all = (
        pd.concat(
            [
                pp_reg_surf,
                pd.concat([pp_reg_surf_sum], keys=["ALPS"], names=["Country"]),
            ]
        )
        .to_frame()
        .reset_index(level=1)
    )

    # Surface of reg shp
    reg_shp_surf = reg_shp.dissolve(by=["Country"]).area / 1e6
    reg_shp_surf["ALPS"] = reg_shp_surf.sum()

    # Add percentage for all
    pp_reg_all["perc"] = (
        pp_reg_all["area"] / reg_shp_surf.reindex_like(pp_reg_all) * 100
    )

    return pp_reg_all


def get_pp_pressures(
    pp_shp: gpd.GeoDataFrame,
    reg_shp: gpd.GeoDataFrame,
    categories: list[str] = None,
    category_sum: str = None,
):
    # IntersectioZns between pp_shp and reg_shp
    pp_reg = pp_shp.overlay(reg_shp, how="intersection", keep_geom_type=False).rename(
        columns={"Country_1": "Country"}
    )

    # Get surface of PP reg
    pp_reg_diss = pp_reg.dissolve(by=categories)
    pp_reg_surf = (
        pp_reg_diss.assign(length=pp_reg_diss.length)
        .groupby(by=categories)
        .sum(["length"])["length"]
        / 1e3
    )

    pp_reg_diss_sum = pp_reg.dissolve(
        by=category_sum
    )  # Dissolve global shp to merge overlapping surfaces within categories
    pp_reg_surf_sum = (
        pp_reg_diss_sum.assign(length=pp_reg_diss_sum.length)
        .groupby(by=category_sum)
        .sum(["length"])["length"]
        / 1e3
    )

    pp_reg_all = pd.concat(
        [pp_reg_surf, pd.concat([pp_reg_surf_sum], keys=["ALPS"], names=["Country"])]
    ).to_frame()

    # Surface of reg shp
    reg_shp_surf = reg_shp.dissolve(["Country", "type"]).length / 1e3
    reg_shp_surf = pd.concat(
        [
            reg_shp_surf,
            pd.concat(
                [reg_shp_surf.groupby(["type"]).sum()], keys=["ALPS"], names=["Country"]
            ),
        ]
    )

    # Add percentage for all
    pp_reg_all["perc"] = (
        pp_reg_all["length"] / reg_shp_surf.reindex_like(pp_reg_all) * 100
    )

    return pp_reg, pp_reg_all
