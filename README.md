# licrice

This repository contains the code and data required to reproduce the historical tropical cyclone surface wind speed dataset used in Hsiang et al. XXXX, “Limited Information Cyclone Reconstruction and Integration for Climate and Economics (LICRICE).”


LICRICE is a tropical cyclone wind field modeling pipeline that generates surface wind fields from historical IBTrACS storm tracks. Given an IBTrACS NetCDF (or Zarr) dataset and a defined geographic region, the system preprocesses the tracks, estimates missing storm radii using a random forest model, computes wind fields using a parametric vortex model. The resulting storm-level wind fields are written as gridded Zarr datasets for each domain. Optional post-processing tools can then aggregate these wind fields to administrative regions (e.g., ADM0–ADM2) using spatial, population-weighted, or asset-weighted averaging, producing datasets suitable for downstream exposure and impact analyses.


## Setup

All scripts are written in Python. Throughout this README, when indicating paths to code and data, it is assumed that you’ll execute scripts from the folder structure provided in this repository using Python 3.9 or greater.

Please cite as:
XXXXX

**Folder Structure**
```
licrice-standalone/
├── README.md
├── run_licrice.py          # Command-Line Interface entry point
├── params/
│   └── licrice/v1.1.json   # Model parameters (vortex func, grid resolution, etc.)
├── licrice/
│   ├── spatial.py          # Great circle distances, grid index conversions
│   ├──  utilities.py        # smooth_fill, unit conversions, Zarr workflow checks
│   ├── io/
│   │   └── ibtracs.py      # IBTrACS ingestion and preprocessing
│   ├── tracks/
│   │   ├── radius.py       # RF-based RMW and ROCI estimation
│   │   ├── velocity.py     # Translational and circular velocity calculations
│   │   └── utils.py        # Track filtering and cleaning utilities
│   ├── licrice/
│   │   ├── run.py          # LICRICE execution loop
│   │   ├── preprocess.py   # Track → pixel step conversion, Zarr setup
│   │   ├── vortex_funcs.py # Parametric vortex models (modified Rankine, Holland 1980)
│   │   ├── dist_funcs.py   # Distance/angle grid construction
│   │   └── utils.py        # Wind field accumulation helpers
│   ├── aggregation/
│   │   └── aggregate_storm_admin.py  # After wind fields are constructed aggregates to admin0-2 level
├── data/
│   ├── raw/
│   │   ├── IBTrACS.ALL.v04r01.nc                            # IBTrACS files go here after they are extracted
│   │   ├── admin/                                           # GADM administrative boundaries
│   │   │   └── gadm_410.gpkg
│   │   ├── LitPop_v1_2/                                     # gridded asset data
│   │   │   └── LitPop_pc_30arcsec_<country>.csv/
│   │   └── landscan-global-2010-assets/                     # 2010 gridded population data
│   │       └── population.csv/
│   ├── output/
│   │   ├── hazard_wind_licrice_hist_<domain>.zarr           # output LICRICE wind files
│   │   └── aggregated/
│   │       ├── spatial/
│   │       │   └── storm_<admin>_<domain>_<haz>.parquet     # output spaitally weighted wind fields
│   │       ├── population/
│   │       │   └── storm_<admin>_<domain>_<haz>.parquet     # output population weighted wind fields
│   │       └── asset/
│   │           └── storm_<admin>_<domain>_<haz>.parquet     # output asset weighted wind fields
└── pyproject.toml
```


## Replication Process

### Install packages

```bash
git clone https://github.com/rachelmyoung/licrice-standalone.git 
cd licrice-standalone
pip install -e .
```

Dependencies (`numpy`, `xarray`, `pandas`, `dask`, `zarr`, `scipy`, `scikit-learn`, `joblib`, `beautifulsoup4`, `tqdm`, `netCDF4`, `bottleneck`, `pyarrow`, `fastparquet`, `fiona`, `rasterio`) are declared in `pyproject.toml` and installed automatically.

Create directories for the IBTrACS input data and LICRICE outputs:

```bash
mkdir -p data/raw
mkdir -p data/output
```

### Data retrieval from the IBTrACs portal

Download International Best Track Archive for Climate Stewardship (IBTrACS) [v04r01] dataset from the NOAA National Centers for Environmental Information (NCEI) with:

```bash
python - <<'EOF'
from licrice.io.ibtracs import download
download(
    url="https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/",
    outdir="data/raw/",
)
EOF
```

Or manually from [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive).

### Run LICRICE to construct storm wind swaths

The entire pipeline for all of the domains (areas of the global, see below for additional details) and including aggregating to three administrative boundary areas with three version of weigthing, can be run from the master workflow with the following bash:

```bash
# Standard run LICRICE (recommended)
python run_licrice.py \
    --input data/raw/IBTrACS.ALL.v04r01.nc \
    --domain all \
    --outdir data/output/

# Standard run LICRICE + aggregation:
python run_licrice.py \
    --input data/raw/IBTrACS.ALL.v04r01.nc \
    --domain all \
    --outdir data/output \
    --aggregate \
    --agg-schemes all \
    --admin-file data/raw/admin/gadm_410.gpkg \
    --litpop-dir data/raw/LitPop_v1_2 \
    --landscan-path data/raw/landscan-global-2015-assets/landscan-global-2015.tif

# Standard run aggregation:
python licrice/aggregation/aggregate_storm_admin.py \
    --zarr-dir data/output \
    --scheme all \
    --admin-file data/raw/admin/gadm_410.gpkg \
    --litpop-dir data/raw/LitPop_v1_2 \
    --landscan-path data/raw/landscan-global-2015-assets/landscan-global-2015.tif
```

If instead you wish to run the pipeline for specific domains or if you wish to run parts of the pipeline, you can also run the following bash commands.

```bash
# Run selected domains without aggregations
python run_licrice.py \
    --input data/raw/IBTrACS.ALL.v04r01.nc \
    --domain north_atlantic_southwest western_pacific_south \
    --outdir data/output/

# Run selected domains + selectd aggregations
python run_licrice.py \
    --input data/raw/IBTrACS.ALL.v04r01.nc \
    --domain north_atlantic_southwest western_pacific_south \
    --outdir data/output/
    --aggregate \
    --agg-schemes spatial

# Advanced options: If you already have preprocessed tracks
python run_licrice.py \
    --input data/raw/IBTrACS.ALL.v04r01.nc \
    --domain north_atlantic_southwest \
    --outdir data/output/ \
    --no-overwrite-preproc

# Advanced options: Use an already-preprocessed Zarr directly
python run_licrice.py \
    --input data/raw/ibtracs_preprocessed.zarr \
    --domain north_atlantic_southwest \
    --outdir data/output/

# Advance options: Use already run LICRICE and just run spatial aggregation
python -m licrice.aggregation.aggregate_storm_admin \
  --zarr-dir data/output \
  --scheme spatial

# Advance options: Population-weighted only
python -m licrice.aggregation.aggregate_storm_admin \
  --zarr-dir data/output \
  --scheme population \
  --landscan-path data/raw/landscan-global-2015-assets/landscan-global-2015.tif

# Advance options: Asset-weighted only
python -m licrice.aggregation.aggregate_storm_admin \
  --zarr-dir data/output \
  --scheme asset \
  --litpop-dir data/raw/LitPop_v1_2/ 
```

### Command Line Interface (CLI) flags

| Flag | Description |
|---|---|
| `--input` | Path to raw IBTrACS `.nc` or preprocessed `.zarr` |
| `--domain` | One or more domain names, or `all` |
| `--outdir` | Output directory |
| `--aggregate` | Run administrative aggregation after LICRICE finishes  |
| `--agg-schemes` | One or more aggregation scheme names (spatial, population, asset), or all; only used with --aggregate |
| `--litpop-dir` | Path to LitPop asset files; required for asset aggregation and for all |
| `--population-dir` | Path to population grid files; required for population aggregation and for all |
| `--agg-script` | Path to the aggregation script (default: licrice/aggregation/aggregate_storm_admin.py) |
| `--preproc-zarr` | Custom path for the preprocessed zarr (default: `<outdir>/ibtracs_preprocessed.zarr`) |
| `--no-overwrite-preproc` | Skip preprocessing if the zarr already exists |
| `--no-overwrite-output` | Skip LICRICE run if the output zarr already exists |
| `--params` | Path to a custom LICRICE params JSON (default: `params/licrice/v1.1.json`) |
| `--storm-chunksize` | Storms per processing chunk (default 25) |
| `--list-domains` | Print all built-in domain names and bounds, then exit |

Each domain produces: `data/output/hazard_wind_licrice_hist_<domain>.zarr`

The first run processes the NetCDF (cleans tracks, trains RF radius models, saves to `ibtracs_preprocessed.zarr`). Only needs to be run once. Later runs with `--no-overwrite-preproc` skip straight to LICRICE.

### Available domains

Domains outlined below are geographically defined coastal regions used to organize the analysis. They do not correspond exactly to tropical cyclone basins and may span portions of multiple basins. Instead, domains were defined to capture all regions where tropical cyclones meaningfully affect land.

| Domain | Description |
|---|---|
| `south_atlantic` | Southern coast of Brazil and Uruguay |
| `east_pacific_north` | Western Canada and Southern Alaska |
| `east_pacific_southwest` | Hawaii |
| `north_america_northeast` | Eastern Canada and Northeastern US |
| `north_atlantic_southwest` | Gulf of Mexico, Caribbean Sea |
| `north_america_south` | Mexico and Southwestern US |
| `north_america_west` | Western United States |
| `north_atlantic_east` | West Africa, Western Europe, Iceland |
| `western_pacific_south` | South China Sea, Philippine Sea |
| `western_pacific` | East China Sea, Sea of Japan |
| `western_pacific_north` | Sea of Okhotsk, Bering Sea |
| `south_pacific_central` | Western Polynesia |
| `south_pacific_northwest` | Northern Australia, Papua New Guinea |
| `south_pacific_southwest` | Southern Australia, New Zealand |
| `south_pacific_east` | Eastern Polynesia |
| `south_indian` | Southern Africa, South Indian Ocean islands |
| `north_indian` | Bay of Bengal, Arabian Sea |
| `conus` | Atlantic and Gulf coasts of the US |

## Output
Each domain produces one Zarr file at `<outdir>/hazard_wind_licrice_hist_<domain>.zarr` with dimensions `(storm, y_ix, x_ix)`.

Coordinates
- `lat (y_ix)` — latitude of grid cell centers
- `lon (x_ix)` — longitude of grid cell centers
- `storm (storm)` — IBTrACS storm ID
- `x_ix (x_ix)` — x-direction grid index
- `y_ix (y_ix)` — y-direction grid index

Data variables
- `maxs (storm, y_ix, x_ix)` — maximum sustained wind speed (m/s, float32) experienced at each grid cell during a storm
- `pddi (storm, y_ix, x_ix)` — power dissipation density index per grid cell per storm (m³/s², float32), the integral of wind speed cubed over time
- `start_date (storm)` — storm start datetime

Grid cell size is 0.1° (configurable via the params JSON).

## Aggregation to adminstative boundary
In addition to the gridded storm wind fields, this repository provides storm-level hazard exposures aggregated to administrative boundaries using GADM polygons. Aggregations are produced for administrative levels ADM0 (country), ADM1 (first-level), and ADM2 (second-level).

Three weighting schemes are provided:
- `spatial` — area-weighted mean hazard within the administrative unit
- `asset` — asset-weighted mean using LitPop asset values
- `population` — population-weighted mean using LandScan population counts

All measures are computed as unconditional means over the full administrative unit, so grid cells with zero hazard still enter the denominator.

### Storm-level outputs
The primary outputs are Parquet files containing storm–administrative unit observations. Each row corresponds to one storm intersecting one administrative unit.

Typical columns include:
| Column |	Description |
|---|---|
| `storm_idx` |	Storm identifier within basin |
| `storm_date` |	Storm start date |
| `year, month` |	Date components |
| `basin` |	Storm basin |
| `admin_level` |	GADM level (0, 1, or 2) |
| `admin_id` |	Administrative unit identifier |
| `scheme` |	Weighting scheme (spatial, asset, population) |
| `haz_var` |	Hazard variable (e.g., maxs) |
| `value` |	Aggregated hazard value |


Example filename:
```bash
storm_admin0_assetw_west_pacific_maxs.parquet
```

## Notes

This package is extracted from coastal-core-main and strips out the cloud I/O, GeoClaw surge model, damage estimation, and `Settings` configuration object. It is intended to run the IBTRACS to LICRICE pipeline locally without complicated dependencies.
