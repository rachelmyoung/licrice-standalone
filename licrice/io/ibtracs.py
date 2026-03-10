"""IBTrACS I/O for LICRICE.

Extracted from pyTC/io/tracks/ibtracs.py. Settings/GCS dependencies removed.
download() signature changed to explicit url/outdir params. preprocess_ibtracs() is new.
"""

import numpy as np
import requests
import xarray as xr
from bs4 import BeautifulSoup

from licrice import tracks, utilities  # NEW
from licrice.tracks import radius as _radius  # NEW

# constructed using:
# https://www.ncdc.noaa.gov/ibtracs/pdf/IBTrACS_version4_Technical_Details.pdf
# unless specified in Table 3 use 10 min averaging period (para 2 of Section 2.2)
AGENCY_AVERAGING_PERIOD = {
    "usa": 1,
    "cma": 2,
    "bom": 10,
    "wellington": 10,
    "nadi": 10,
    "tokyo": 10,
    "reunion": 10,
    "newdelhi": 3,
    "hko": 10,
    "ds824": 1,
    "td9636": 1,
    "td9635": 1,
    "neumann": 10,
    "mlc": 1,
}


def download(url, outdir):
    """Download ibtracs and save files to a specified location.

    Parameters
    ----------
    url : str
        Base URL to the IBTrACS file listing.
    outdir : str or Path
        Directory to save downloaded files.

    """
    import pathlib  # NEW

    outdir = pathlib.Path(outdir)  # NEW
    download_url = requests.get(url)  # NEW
    soup = BeautifulSoup(download_url.text, features="html.parser")
    links = soup.findAll("a")

    for link in links:
        if ".nc" in link["href"] or ".txt" in link["href"]:
            loc = outdir / str(link["href"])  # NEW
            print(f"Downloading {link['href']} to {loc}")
            with loc.open("wb") as f:
                f.write(requests.get(url + link["href"]).content)  # NEW


def format_standard(
    _ds,
    agency_pref=[
        "wmo",
        "usa",
        "tokyo",
        "newdelhi",
        "reunion",
        "bom",
        "nadi",
        "wellington",
        "cma",
        "hko",
        "ds824",
        "td9636",
        "td9635",
        "neumann",
        "mlc",
    ],
    ibt_storm_meta=[
        "numobs",
        "sid",
        "season",
        "basin",
        "subbasin",
        "name",
        "number",
        "nature",
        "track_type",
        "main_track_sid",
        "dist2land",
        "iflag",
        "storm_speed",
    ],
):
    """Reformat ibtracs track dataset with standard track dataset conventions.

    Parameters
    ----------
    _ds : :py:class:`xarray.Dataset`
        ibtracs tracks dataset
    agency_pref : list of strings, optional
        agency preference order for filling missing data. Note: the code is
        written such that wmo is automatically the preferred data and usa data
        is second best.
    ibt_storm_meta : list of strings, optional
        non essential data variables to perserve in newly formatted trackset

    Returns
    -------
    :py:class:`xarray.Dataset`
        ibtracs tracks dataset reformatted like emanuel tracks dataset

    """
    ###########################################################################
    # Setup

    ds = _ds.copy()

    # assign coords
    ds.coords["storm"] = ds.storm
    ds.coords["date_time"] = ds.date_time

    # find wmo agencies that correspond to usa_* vars
    # ATCF is explicitly added b/c of a bug that appears at least in the IN
    # set as of Mar 2024, where atcf is listed as wmo agency for one storm but
    # hurdat_atl listed for usa_agency
    usa_agencies = list(set([b"atcf"] + list(np.unique(ds.usa_agency))))
    usa_agencies.remove(b"")

    # construct new dataset to populate with ibtracs info with standard trackset
    # stucture
    newds = xr.Dataset(
        data_vars=ds[ibt_storm_meta],
        coords={
            "date_time": np.arange(len(ds.coords["date_time"])),
            "storm": np.arange(len(ds.coords["storm"])),
        },
    )

    ###########################################################################
    # fix slightly off timestamps
    newds["time"] = newds.time.dt.round("s")

    ###########################################################################
    # decode string variables of interest if they are preserved
    for dv in [
        dv
        for dv in [
            "name",
            "main_track_sid",
            "sid",
            "usa_agency",
            "iflag",
            "nature",
            "usa_atcf_id",
            "subbasin",
            "basin",
        ]
        if dv in ibt_storm_meta
    ]:
        newds[dv] = newds[dv].str.decode("utf-8")

    ###########################################################################
    # Convert all wind variables to the same averaging period

    # access conversion factor based on agency and global dictionaries
    def get_averaging_period_conversion_divisor(x):
        return (
            float(
                tracks.velocity.AVERAGING_PERIOD_CONVERSION_DIVISOR[
                    AGENCY_AVERAGING_PERIOD[x.decode()]
                ],
            )
            if x
            else np.nan
        )

    wind_units = ds.wmo_wind.units
    ds["wmo_wind"] = ds["wmo_wind"] / xr.apply_ufunc(
        get_averaging_period_conversion_divisor,
        ds.wmo_agency.where(~np.isin(ds.wmo_agency, usa_agencies), b"usa"),
        vectorize=True,
    )
    ds.wmo_wind.attrs.update(units=wind_units)

    for agency in agency_pref:
        if agency != "wmo":
            wind_units = ds[agency + "_wind"].units
            ds[agency + "_wind"] = ds[
                agency + "_wind"
            ] / get_averaging_period_conversion_divisor(agency.encode())
            ds[agency + "_wind"].attrs.update(units=wind_units)

    ###########################################################################
    # Create DataArrays of preferred values for each storm x time observation.
    # Wherever available first try to use the wmo value, then try to use the
    # wmo agency value, then use the usa agency value, then refer to the agency
    # preference list as supplied by the function caller.

    # pool data from all agencies
    pref_vals = {}
    for v in ["wind", "pres", "rmw", "roci"]:
        all_vals = ds[
            [f"{i}_{v}" for i in agency_pref if f"{i}_{v}" in list(ds.data_vars)]
        ]

        # ensure all units match up and set up starting val_pref (nan if no wmo_* value)
        for dv in all_vals.data_vars:
            assert ds["usa_" + v].units.lower() == ds[dv].units.lower()
        if v in ["wind", "pres"]:
            assert ds["usa_" + v].units.lower() == ds["wmo_" + v].units.lower()
            val_pref = ds[f"wmo_{v}"].copy().rename(v)
        else:
            val_pref = ds[f"usa_{v}"].copy().rename(v) * np.nan

        # remove the suffix in agency and convert to array
        all_vals = all_vals.rename(
            {x: x.split("_")[0] for x in all_vals.variables if x != "date_time"},
        )
        all_vals = all_vals.to_array(dim="agency")

        names = (
            ds.wmo_agency.where(~ds.wmo_agency.isin(usa_agencies), "usa")
            .astype(str)
            .copy()
        )

        # first update any null values with their reporting agency preference
        for a in all_vals.agency:
            val_pref = val_pref.where(
                (names != a) | val_pref.notnull(),
                all_vals.sel(agency=a),
            )

        # sometimes there will be intermediate time step data reported from the
        # preferred agency that is not in the wmo reported values (and thus for these
        # datetimes, there is no wmo_agency value). Here we choose the "most popular"
        # agencies reporting for each storm and use their values to fill in the
        # intermediate times in wmo_wind. For each round of "next most popular agency",
        # we perform interpolation (using fractional change in the filler value combined
        # with levels of the pre-fill value b/c we don't want to be bouncing back and
        # forth between different agencies since this could cause oscillations in wind
        # size that are not physical
        counts = (names == all_vals.agency).sum("date_time")

        n_agencies = (counts > 0).sum("agency")
        valid = xr.ones_like(n_agencies)
        main_agency = "dummy"

        # first fill with the reporting agency for that track
        for i in range(n_agencies.max().item()):
            valid = valid & (n_agencies > i) & (counts.agency != main_agency)
            main_agency = counts.where(valid).idxmax(dim="agency")
            filler = all_vals.sel(
                agency=main_agency.where(
                    main_agency.notnull(),
                    all_vals.agency.values[0],
                ),
            ).where(main_agency.notnull())
            val_pref = utilities.smooth_fill(
                val_pref,
                filler,
                time_dim="date_time",
                interpolate=True,
            )
            names = names.where((names != "") | val_pref.isnull(), main_agency)

        # next fill with preferred agencies
        for i in all_vals.agency.values:
            val_pref = utilities.smooth_fill(
                val_pref,
                all_vals.sel(agency=i),
                time_dim="date_time",
                interpolate=True,
            )

        # check no infinites in prefered vals
        assert (np.abs(val_pref) != np.inf).any(dim="date_time").all()

        # add to dict (drop added variable if present)
        try:
            pref_vals[v] = val_pref.drop_vars(["agency"])
        except ValueError:
            pref_vals[v] = val_pref

    ###########################################################################
    # assign total windspeed to v_total

    newds["v_total"] = utilities.convert_units(
        pref_vals["wind"],
        (ds.wmo_wind.units.lower(), "m/s"),
    )

    newds["v_total"] = newds.v_total.where(newds.v_total >= 0)

    newds["v_total"].attrs.update(
        {
            "long_name": "maximum total windspeed",
            "units": "m/s",
            "description": (
                "maximum 1-minute sustained windspeed, including "
                "translational and rotational components"
            ),
            "method": "IBTrACS *_wind (NaNs are filled based on agency preference)",
        },
    )

    ###########################################################################
    # assign pressure to pstore

    newds["pstore"] = utilities.convert_units(
        pref_vals["pres"],
        (ds.wmo_pres.units.lower(), "hPa"),
    )

    newds["pstore"].attrs.update(
        {
            "long_name": "Minimum central pressure",
            "units": "hPa",
            "description": "Minimum central pressure",
            "method": "IBTrACS *_pres (NaNs are filled based on agency preference)",
        },
    )

    ###########################################################################
    # assign radius of max winds to rmstore

    newds["rmstore"] = utilities.convert_units(
        pref_vals["rmw"],
        (ds.usa_rmw.units.lower(), "km"),
    )

    newds["rmstore"] = newds.rmstore.where(newds.rmstore >= 0)

    newds["rmstore"].attrs.update(
        {
            "long_name": "Radius of maximum winds",
            "units": "km",
            "description": "Radius of maximum winds",
            "method": ("IBTrACS *_rmw (NaNs are filled based on agency preference)"),
        },
    )

    ###########################################################################
    # assign radius of last closed isobar to storm_radius

    newds["storm_radius"] = utilities.convert_units(
        pref_vals["roci"],
        (ds.usa_roci.units.lower(), "km"),
    )

    newds["storm_radius"] = newds.storm_radius.where(newds.storm_radius >= 0)

    newds["storm_radius"].attrs.update(
        {
            "long_name": "Radius of outermost closed isobar",
            "units": "km",
            "description": (
                "Radius of outermost closed isobar, a measure of the "
                "total radial extent of the storm"
            ),
            "method": ("IBTrACS *_roci (NaNs are filled based on agency preference)"),
        },
    )

    ###########################################################################
    # rename/set coordinates for standard conventions

    newds["lon"] = ((ds.lon + 180) % 360) - 180
    newds["lat"] = ds.lat

    newds = newds.rename(
        {
            "time": "datetime",
            "lon": "longstore",
            "lat": "latstore",
        },
    ).rename(
        {
            "date_time": "time",
        },
    )

    newds = newds.reset_coords(["datetime", "longstore", "latstore"])

    newds.attrs.update(ds.attrs)

    return newds


def format_clean(ds):
    """Reformat ibtracs track dataset for generalized use throughout system.

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
        ibtracs tracks dataset

    Returns
    -------
    ds : :py:class:`xarray.Dataset`
        ibtracs tracks dataset formatted and cleaned

    """
    ds = format_standard(ds)

    # drop stationary storms
    ds, _ = tracks.utils.drop_stationary_storms(ds)

    # mask all variables when leading and trailing v_total is nan
    ds = tracks.utils.drop_leading_and_trailing_nans(ds)

    # mask all variables when datetime is null
    ds = tracks.utils.mask_invalid_values(ds)

    # drop storms with missing max wind speed obs or with only one observation
    ds, _, _ = tracks.utils.assess_var_missingness(ds)

    # fill nans
    ds = ds.set_coords("datetime")
    ds = xr.concat(
        [
            tracks.utils.interpolate_nans(
                ds.isel(storm=i).dropna(dim="time", subset=["datetime"]),
                use_coordinate="datetime",
            ).reset_coords("datetime", drop=False)
            for i in range(len(ds.sid))
        ],
        dim="storm",
    )

    # calculate translational and circular velocity
    ds = tracks.velocity.calculate_v_trans_x_y(ds, "latstore", "longstore")
    ds = tracks.velocity.calculate_v_circular(ds)

    # drop any storms that never show a circular velocity
    ds = ds.isel(storm=ds.v_circular.max(dim="time") > 0)

    # cast season var
    ds["season"] = ds.season.astype(np.uint16)

    # delete any manually found duplicates (first one in each list is kept)
    # If using a non "ALL" ibtracs set where only one of the two duplicate SIDs is
    # present, keep that storm
    duplicate_storms = [
        ["1961299N11281", "1961301N12279"],  # Hattie
        ["1962292N11119", "1962298N08107"],  # Harriet
        ["1963267N11309", "1963267N11308"],  # Edith
        ["1963270N08327", "1963272N09314"],  # Flora
        ["1977365S09188", "1978032S09187"],  # Bob
        ["1995241N11333", "1995240N11337"],  # Luis
        ["2017081S13152", "2017082S14152"],  # Debbie
        ["2020092S09155", "2020094S10160"],  # Harold
        ["2020019S25191", "2020017S14178"],  # Tino
    ]
    for storms in duplicate_storms:
        to_drop = ds.sid.isel(storm=ds.sid.isin(storms))
        if len(to_drop) == len(storms):
            ds = ds.isel(storm=~(ds.sid.isin(storms[1:])))

    # fix "BRENDAN" that should be "BRENDA"
    # PAKA- should be PAKA
    # unnamed 2018 WP storm should be Tropical Depression JOSIE
    ds["name"] = (
        ds.name.where(ds.sid != "1985268N03161", "BRENDA")
        .where(ds.sid != "1997333N06194", "PAKA")
        .where(ds.sid != "2018202N18116", "JOSIE")
        .where(ds.sid != "2022342N09084", "MANDOUS")
    )

    # storms where one picks up where the other leaves off
    combine_sid_groups = [
        ["1991196N06153", "1991207N20105"],  # 1991 BRENDAN
        ["2017122S13170", "2017131S27168"],  # 2017 DONNA
    ]

    ds = _combine_tracks(ds, combine_sid_groups)

    # final formatting
    for v in ["basin", "subbasin", "nature", "iflag", "track_type"]:
        ds[v] = ds[v].where(ds[v].notnull(), "").astype("unicode")
    ds["storm"] = np.arange(ds.storm.size)

    return ds


def _combine_tracks(ds, sids_groups):
    time_vars = [k for k, v in ds.data_vars.items() if "time" in v.dims]
    other_vars = [k for k, v in ds.data_vars.items() if k not in time_vars]

    all_sids = [i for sublist in sids_groups for i in sublist]
    ds_out = ds.isel(storm=~(ds.sid.isin(all_sids)))

    all_ds = [ds_out]
    for sids in sids_groups:
        if not np.isin(sids, ds.sid.values).all():
            continue

        this_ds = ds.isel(storm=ds.sid.isin(sids)).copy()

        stackable = this_ds[time_vars]
        unstackable = this_ds[other_vars]
        unstackable = unstackable.isel(
            storm=unstackable.numobs.argmax(unstackable.numobs.dims[0]),
        )

        stacked = stackable.stack(tmp=["time", "storm"])
        stacked = stacked.sel(tmp=stacked.datetime.notnull()).sortby("datetime")
        stacked = stacked.isel(
            tmp=~(
                stacked.swap_dims({"tmp": "datetime"})
                .get_index("datetime")
                .duplicated()
            ),
        )
        stacked = stacked.swap_dims({"tmp": "time"}).drop_vars(["tmp", "storm"])
        stacked["time"] = np.arange(stacked.time.size)

        out = unstackable.merge(stacked)
        out["numobs"] = out.time.size
        all_ds.append(out)
    return xr.concat(all_ds, "storm").sortby("sid").reset_coords("sid")


def preprocess_ibtracs(nc_path, zarr_outpath, overwrite=False):  # NEW
    """IBTrACS netCDF file to zarr.

    Runs format_clean, trains/loads RF radius models, estimates radii, saves result.

    Parameters
    ----------
    nc_path : str or Path
        Path to the raw IBTrACS netCDF file.
    zarr_outpath : str or Path
        Destination zarr directory path.
    overwrite : bool, optional
        If True, overwrite an existing zarr and retrain models. Default False.

    """
    import pathlib

    zarr_outpath = pathlib.Path(zarr_outpath)
    if zarr_outpath.exists() and not overwrite:
        print(f"Preprocessed zarr already exists at {zarr_outpath}. Skipping.")
        return

    print(f"Loading {nc_path} ...")
    raw_ds = xr.open_dataset(str(nc_path))

    print("Formatting IBTrACS tracks ...")
    ds = format_clean(raw_ds)

    model_dir = pathlib.Path(__file__).parent.parent.parent / "params" / "radius"
    _model_files = ["rmw_to_rad.pkl", "rad_to_rmw.pkl", "rmw.pkl", "cols.pkl"]
    models_exist = all((model_dir / f).exists() for f in _model_files)
    if models_exist and not overwrite:
        print(f"Loading radius models from {model_dir} ...")
        rmw_to_rad, rad_to_rmw, rmw_model, cols = _radius.load_radius_models(model_dir)
    else:
        print("Training radius RF models (this may take a few minutes) ...")
        rmw_to_rad, rad_to_rmw, _, rmw_model, cols = _radius.get_radius_ratio_models(
            ds, model_dir
        )

    print("Estimating missing radii ...")
    ds = _radius.estimate_radii(ds, rmw_to_rad, rad_to_rmw, rmw_model, reg_cols=cols)

    n_storms = ds.sizes["storm"]
    chunk_size = min(50, n_storms)
    ds = ds.chunk({"storm": chunk_size, "time": ds.sizes["time"]})

    print(f"Saving preprocessed tracks to {zarr_outpath} ({n_storms} storms) ...")
    ds.to_zarr(str(zarr_outpath), mode="w", consolidated=True)
    print("Done.")
