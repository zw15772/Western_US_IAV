#!/usr/bin/env python3
"""
GEE download — cloud-free SEASONAL composites for Kabul, saved to a LOCAL folder.

For each year and each meteorological season, builds a cloud-masked median
composite clipped to the Kabul boundary, for two sensor stacks, and downloads
the GeoTIFF straight to disk (via geemap/geedim, which auto-tiles large images):
  - Landsat 5/7/8/9 Collection-2 Level-2 surface reflectance (2000- )
  - Sentinel-2 surface reflectance L2A                        (2017- )

Seasons (meteorological — aligned to Kabul's phenology):
  DJF = Dec(year-1)+Jan+Feb   MAM = Mar-May   JJA = Jun-Aug   SON = Sep-Nov
=> 4 images per year per sensor.

Each file: harmonized 6-band optical stack + spectral indices (16 bands total):
  optical : ['blue','green','red','nir','swir1','swir2']
  veg     : NDVI, SAVI, EVI, GNDVI         (greenness / condition)
  function: NIRv (productivity), NDMI (canopy water / irrigation)
  built   : NDBI (construction), BSI (bare ground / rubble)
  water   : MNDWI (Kabul River / riparian)
  burn    : NBR  (fire / damage)
Output CRS = UTM 42N (EPSG:32642), pixels in metres (good for landscape metrics).

--------------------------------------------------------------------------------
SETUP (run once, in the `Fujiang` conda env; ee/geemap/geopandas/geedim present)
    earthengine authenticate          # opens a browser; pastes a token
    # If ee.Initialize() complains about a project, set EE_PROJECT below.
RUN
    python gee_download_kabul.py
Files appear under OUT_DIR/<sensor>/.
--------------------------------------------------------------------------------
"""

import os

# This conda env sets PROJ_LIB to an old (2019) proj.db that shadows rasterio's
# newer bundled one and breaks all EPSG lookups during download. Drop it so
# rasterio/geedim use their own proj.db. Must happen before importing geemap.
os.environ.pop("PROJ_LIB", None)

import ee
import geemap

# ============================== CONFIG ========================================
SHP_PATH    = "/Users/jifujiang/Desktop/war_biodiversity/0_shapefiles/1_kabul_Afghanistan/kabul.shp"
OUT_DIR     = "/Volumes/drive/war_biodiversity_data/1_kabul_Afghanistan"
EE_PROJECT  = None          # e.g. "ee-yourusername"; leave None for legacy init

# Expand the AOI outward from the shapefile (metres). Captures peri-urban
# agriculture / rural surroundings for the irrigated-vs-rain-fed comparison.
# Set to 0 to use the exact shapefile boundary.
BUFFER_M    = 2000          # ~2 km halo around the city polygon

# Time window
LS_START_YEAR = 2000        # Landsat: after 2000 as requested
LS_END_YEAR   = 2025
S2_START_YEAR = 2017        # Sentinel-2 SR (L2A) only reliable from 2017
S2_END_YEAR   = 2025

# Seasons (meteorological). DJF of year Y uses Dec of Y-1.
SEASONS = ["DJF", "MAM", "JJA", "SON"]

# Cloud filtering
MAX_SCENE_CLOUD = 50        # drop scenes with >X% scene cloud before masking

# Landsat 7 ETM+ has had the Scan Line Corrector failure since 2003-05-31 ("SLC-off"):
# ~22% wedge-shaped data gaps per scene. We KEEP it and let the seasonal median +
# other sensors fill the stripes. Set True to exclude SLC-off L7 (cleaner, but 2012
# — which relies almost entirely on L7 — would have little/no data).
DROP_L7_SLC_OFF = False
L7_SLC_OFF_DATE = "2003-05-31"

# Export / download
EXPORT_CRS   = "EPSG:32642" # UTM zone 42N (metres)
LS_SCALE     = 30           # Landsat native resolution (m)
S2_SCALE     = 10           # Sentinel-2 native resolution (m)
DOWNLOAD_SENTINEL2 = True
DOWNLOAD_LANDSAT   = True
OVERWRITE    = False        # skip files that already exist on disk
# =============================================================================

COMMON_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]


# ----------------------------- init ------------------------------------------
def init_ee():
    try:
        ee.Initialize(project=EE_PROJECT) if EE_PROJECT else ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT) if EE_PROJECT else ee.Initialize()


def load_aoi():
    """Load the Kabul shapefile as an ee.Geometry (dissolved), optionally buffered."""
    fc = geemap.shp_to_ee(SHP_PATH)
    geom = fc.geometry().dissolve(maxError=1)
    if BUFFER_M and BUFFER_M > 0:
        geom = geom.buffer(BUFFER_M, 1)   # metres; maxError = 1 m
    return geom


def season_dates(year, season):
    """Return (start, end-exclusive) ee.Dates for a meteorological season."""
    if season == "DJF":
        return ee.Date.fromYMD(year - 1, 12, 1), ee.Date.fromYMD(year, 3, 1)
    if season == "MAM":
        return ee.Date.fromYMD(year, 3, 1), ee.Date.fromYMD(year, 6, 1)
    if season == "JJA":
        return ee.Date.fromYMD(year, 6, 1), ee.Date.fromYMD(year, 9, 1)
    if season == "SON":
        return ee.Date.fromYMD(year, 9, 1), ee.Date.fromYMD(year, 12, 1)
    raise ValueError(season)


# ----------------------- index helper ----------------------------------------
def add_indices(img):
    """Add spectral indices, all derived from the harmonized 6-band stack so they
    are comparable between Landsat and Sentinel-2 (no red-edge — Landsat lacks it).
    """
    b = {k: img.select(k) for k in COMMON_BANDS}

    # --- Vegetation amount / greenness ---
    ndvi = img.normalizedDifference(["nir", "red"]).rename("NDVI")
    # SAVI with L=0.5 (soil-adjusted; preferred in arid low-cover Kabul)
    savi = img.expression("1.5 * (nir - red) / (nir + red + 0.5)", b).rename("SAVI")
    # EVI — less saturation, better in denser/irrigated canopy
    evi = img.expression(
        "2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)", b).rename("EVI")
    # GNDVI — chlorophyll / nitrogen sensitive
    gndvi = img.normalizedDifference(["nir", "green"]).rename("GNDVI")

    # --- Vegetation FUNCTION (productivity / water) ---
    # NIRv = NDVI * NIR — proxy for canopy productivity (GPP)
    nirv = ndvi.multiply(b["nir"]).rename("NIRv")
    # NDMI — canopy/vegetation water content; tracks irrigation & drought stress
    ndmi = img.normalizedDifference(["nir", "swir1"]).rename("NDMI")

    # --- Built-up / CONSTRUCTION & bare ground / rubble ---
    # NDBI — built-up / impervious surfaces (construction)
    ndbi = img.normalizedDifference(["swir1", "nir"]).rename("NDBI")
    # BSI — Bare Soil Index; bare ground & rubble (war destruction signal)
    bsi = img.expression(
        "((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))",
        b).rename("BSI")

    # --- Water (Kabul River / riparian corridor) ---
    # MNDWI — open water; more robust than NDWI in built areas
    mndwi = img.normalizedDifference(["green", "swir1"]).rename("MNDWI")

    # --- Burn / fire damage ---
    # NBR — burned areas; relevant to conflict fires & damage
    nbr = img.normalizedDifference(["nir", "swir2"]).rename("NBR")

    return img.addBands(
        [ndvi, savi, evi, gndvi, nirv, ndmi, ndbi, bsi, mndwi, nbr])


# ----------------------- Landsat C2 L2 ---------------------------------------
def _ls_qa_mask(img):
    """Cloud/shadow/cirrus/dilated-cloud mask from QA_PIXEL (Collection-2)."""
    qa = img.select("QA_PIXEL")
    dilated, cirrus, cloud, shadow = 1 << 1, 1 << 2, 1 << 3, 1 << 4
    return (qa.bitwiseAnd(dilated).eq(0)
            .And(qa.bitwiseAnd(cirrus).eq(0))
            .And(qa.bitwiseAnd(cloud).eq(0))
            .And(qa.bitwiseAnd(shadow).eq(0)))


def _prep_tm_etm(img):       # Landsat 5 (TM) & 7 (ETM+)
    sr = (img.select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"])
          .multiply(0.0000275).add(-0.2).rename(COMMON_BANDS))
    return sr.updateMask(_ls_qa_mask(img)).copyProperties(img, ["system:time_start"])


def _prep_oli(img):          # Landsat 8 & 9 (OLI)
    sr = (img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
          .multiply(0.0000275).add(-0.2).rename(COMMON_BANDS))
    return sr.updateMask(_ls_qa_mask(img)).copyProperties(img, ["system:time_start"])


def landsat_collection(aoi, start, end):
    cf = ee.Filter.lt("CLOUD_COVER", MAX_SCENE_CLOUD)
    l5 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
          .filterBounds(aoi).filterDate(start, end).filter(cf).map(_prep_tm_etm))
    l7_col = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
              .filterBounds(aoi).filterDate(start, end).filter(cf))
    if DROP_L7_SLC_OFF:
        l7_col = l7_col.filterDate("1999-01-01", L7_SLC_OFF_DATE)  # keep only SLC-on
    l7 = l7_col.map(_prep_tm_etm)
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(aoi).filterDate(start, end).filter(cf).map(_prep_oli))
    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
          .filterBounds(aoi).filterDate(start, end).filter(cf).map(_prep_oli))
    return l5.merge(l7).merge(l8).merge(l9)


# ----------------------- Sentinel-2 SR (L2A) ---------------------------------
def _prep_s2(img):
    scl = img.select("SCL")
    mask = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
            .And(scl.neq(10)).And(scl.neq(11)))   # shadow/cloud/cirrus/snow
    sr = (img.select(["B2", "B3", "B4", "B8", "B11", "B12"])
          .divide(10000).rename(COMMON_BANDS))
    return sr.updateMask(mask).copyProperties(img, ["system:time_start"])


def s2_collection(aoi, start, end):
    cf = ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_SCENE_CLOUD)
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi).filterDate(start, end).filter(cf).map(_prep_s2))


# ----------------------- composite + local download --------------------------
def download_composite(collection_fn, aoi, year, season, sensor_tag, scale):
    sub_dir = os.path.join(OUT_DIR, sensor_tag)
    os.makedirs(sub_dir, exist_ok=True)
    name = f"kabul_{sensor_tag}_{year}_{season}"
    out_path = os.path.join(sub_dir, name + ".tif")
    if os.path.exists(out_path) and not OVERWRITE:
        print(f"  [exists] {name}")
        return

    start, end = season_dates(year, season)
    coll = collection_fn(aoi, start, end)
    n = coll.size().getInfo()
    if n == 0:
        print(f"  [skip]   {sensor_tag} {year} {season}: 0 usable scenes")
        return

    composite = add_indices(coll.median()).clip(aoi).toFloat()
    print(f"  [get]    {name}: {n} scenes")
    geemap.download_ee_image(
        image=composite,
        filename=out_path,
        region=aoi,
        crs=EXPORT_CRS,
        scale=scale,
    )


def main():
    init_ee()
    aoi = load_aoi()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"AOI loaded. Output: {OUT_DIR}  CRS: {EXPORT_CRS}\n")

    if DOWNLOAD_LANDSAT:
        print("Landsat 5/7/8/9 (C2 L2):")
        for y in range(LS_START_YEAR, LS_END_YEAR + 1):
            for s in SEASONS:
                download_composite(landsat_collection, aoi, y, s, "LS", LS_SCALE)

    if DOWNLOAD_SENTINEL2:
        print("\nSentinel-2 SR (L2A):")
        for y in range(S2_START_YEAR, S2_END_YEAR + 1):
            for s in SEASONS:
                download_composite(s2_collection, aoi, y, s, "S2", S2_SCALE)

    print("\nDone.")


if __name__ == "__main__":
    main()
