# Functions to compute spectral indices of satellites data based on Google Earth Engine


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
