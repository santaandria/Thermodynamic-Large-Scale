from pathlib import Path


def init():
    global TMEAN_OE_PATH
    global DPT_OE_PATH
    global DATA_DOC_PATH
    global KG_RASTER_PATH
    global OUTPUT_PATH

    TMEAN_OE_PATH = Path("./data/OE_tmean_era5/")
    DPT_OE_PATH = Path("./data/OE_tmean_era5/")
    DATA_DOC_PATH = Path("./data/dataset_docs/")
    KG_RASTER_PATH = Path("./data/CONUS_Beck_KG_V1_Present.tif")
    OUTPUT_PATH = Path("./output/")
