"""LICRICE preprocessing functions.

Extracted from pyTC/licrice/preprocess.py. clawpack.geoclaw.units dependency replaced
by licrice.utilities.geoclaw_convert.
"""

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from licrice import spatial, testing, tracks  # NEW
from licrice.utilities import geoclaw_convert  # NEW
from .utils import get_output_grid

# variables necessary to hold onto for licrice

_KEEP_VARS = ["rmw", "radius", "v_circular", "pres", "datetime"]

########################################################################################
# Functions for prepping whole trackset for use in LICRICE
########################################################################################


def _clean_emanuel(ds):
    ds = ds[
        [
            "latstore",
            "longstore",
            "v_circular",
            "pstore",
            "rmstore",
            "storm_radius",
            "ensemble",
            "sid",
            "datetime",
        ]
    ].load()

    if "ensemble" in ds.dims:
        ds = (
            ds.stack(tmp=["ensemble", "storm"])
            .reset_index("tmp", drop=False)
            .rename(storm="storm_ix")
            .swap_dims(tmp="sid")
            .rename(sid="storm")
        )

    # "season" indexes freqyear in the raw emanuel data, but LICRICE expects it to
    # index "year of storm".
    ds["season"] = ds.datetime.dt.year.isel(time=0).astype(np.uint16)

    return ds


def _clean_ibtracs(ds, p):
    ds = ds[
        [
            "latstore",
            "longstore",
            "v_circular",
            "datetime",
            "pstore",
            "sid",
            "rmstore_estimated",
            "storm_radius_estimated",
        ]
    ].load()

    # reset storm coordinates
    ds["storm"] = ds.sid
    ds = ds.drop_vars("sid")

    ##################################################################
    # check units
    ##################################################################

    assert ds.storm_radius_estimated.units.lower() == "km"
    assert ds.rmstore_estimated.units.lower() == "km"

    ##################################################################
    # use estimated storm radii values when needed
    ##################################################################

    ds = ds.rename(rmstore_estimated="rmstore", storm_radius_estimated="storm_radius")

    assert not ds.storm_radius.isel(time=0).isnull().any()

    return ds


def clean_tracks(ds, params, trackset_type):
    if trackset_type == "ibtracs":
        return _clean_ibtracs(ds, params)
    elif trackset_type == "emanuel":
        return _clean_emanuel(ds)
    raise ValueError(trackset_type)


def load_tracks(fpath, params, chunks=None, trackset_type="ibtracs", **sel_kwargs):
    # load tracks
    ds = clean_tracks(
        xr.open_zarr(fpath, consolidated=True, chunks=chunks),
        params,
        trackset_type,
    ).sel(sel_kwargs)

    return _convert_units(
        ds.rename(
            {
                "longstore": "storm_lon",
                "latstore": "storm_lat",
                "rmstore": "rmw",
                "pstore": "pres",
                "storm_radius": "radius",
            },
        ),
    )


def _convert_units(ds):
    # distances in meters
    dist_vars = ["rmw", "radius"]
    for dv in dist_vars:
        attr = ds[dv].attrs.copy()
        ds[dv] = geoclaw_convert(
            ds[dv],
            old_units=ds[dv].units.lower(),
            new_units="m",
        )
        attr.update({"units": "m"})
        ds[dv].attrs = attr

    # pressure in Pa
    attr = ds["pres"].attrs.copy()
    ds["pres"] = geoclaw_convert(
        ds.pres,
        old_units=ds.pres.units,
        new_units="Pa",
    )
    attr.update({"units": "Pa"})
    ds["pres"].attrs = attr

    return ds


def pixels_per_segment(
    storm_ds,
    p,
    lat_var="storm_lat",
    lon_var="storm_lon",
):
    """Calculate the number of pixel steps between time steps.

    Parameters
    ----------
    storm_ds : :class:`xarray.Dataset`
        1-D Dataset indexed by 'time'. Containing `storm_lat` and `storm_lon` --
        variables representing the storm eye coordinates at each time step
    p : dictionary
        licrice parameters loaded as a dictionary
    lat_var : str
        name of latitude variable
    lon_var : str
        name of longitude variable

    Returns
    -------
    :class:`numpy.ndarray`
        The number of pixel steps taken between each observation in `storm_ds`.

    """
    # get grid cell indices of each lon/lat point
    # note that we specifically do NOT use the `lon_ix` kwarg in this transformation b/c
    # we want to allow smooth transitions across the dateline rather than jumps from
    # 179 to -179
    storm_x = spatial.grid_val_to_ix(
        storm_ds[lon_var].values,
        cell_size=p["grid"]["res_spatial_deg"],
    )
    storm_y = spatial.grid_val_to_ix(
        storm_ds[lat_var].values,
        cell_size=p["grid"]["res_spatial_deg"],
    )

    # get the number of pixel steps in the x and y direction per time step
    n_x = np.diff(storm_x)
    n_y = np.diff(storm_y)

    # find the number of pixel steps that will need to be taken per time step
    return np.maximum(np.abs(n_x), np.abs(n_y))


def calculate_time_per_pixel_step(dts, n_steps):
    """Calculate the amount of time a storm eye spends at each pixel the eye crosses.

    Parameters
    ----------
    dts : :class:`numpy.ndarray`
        1-D array with the length of time between observations
    n_steps : :class:`numpy.ndarray`
        1-D array with the number of pixel steps taken between each observation.

    Returns
    -------
    :class:`numpy.ndarray`
        1-D array with the amount of time a storm eye spends at each pixel it crosses.
        Array length is the number of pixel steps a storm eye takes during its
        duration.

    """
    ####################################################################################
    # Iteratively calculate the amount of time the storm eye spends at each pixel
    # it crosses
    ####################################################################################

    # create an array to populate with the amount of time the storm eye spends at each
    # pixel -- each index represents a different pixel the storm eye crosses
    dt_new = np.zeros(n_steps.sum(), dtype=np.float64)

    ctr = 0
    prev_dt = 0

    # loop over each segment
    for ix in range(len(n_steps)):
        # n of pixelsteps in this timestep
        this_n_steps = n_steps[ix]

        # if no pixel steps, storm doesn't move and we just add this time into the next
        # pixel step
        if this_n_steps == 0:
            prev_dt += dts[ix]
            continue

        # add the time interval for each pixelstep within this timestep to any previous
        # timesteps in which no pixel movement occured and then calculate
        this_dt = dts[ix] / this_n_steps

        # assign that time interval for each pixelstep within this timestep
        dt_new[ctr : ctr + this_n_steps] = this_dt

        # add the previous non-moving timesteps to the FIRST pixelstep in this timestep
        dt_new[ctr] += prev_dt

        # increment array element pointer
        ctr += this_n_steps

        # reset the no-movement time counter
        prev_dt = 0

    # if there are any trailing no-pixel-movement time steps, add that time into the
    # last pixel step
    dt_new[-1] += prev_dt

    return dt_new


def timesteps_to_pixelsteps(storm_ds, p, addl_vars=_KEEP_VARS):
    """Transform single storm dataset from time step level observations to pixel step
    level observations. One observation in the dataset now translates to one pixel step.
    This function assumes that the longitude values for this storm have already been
    transformed to be continuous, i.e. no jumps from ~180 to ~(-180) or from ~360 to ~0.

    Parameters
    ----------
    storm_ds : :class:`xarray.Dataset`
        1-D Dataset indexed by 'time'. Containing all variables necessary for running
        LICRICE
    p : dictionary
        licrice parameters loaded as a dictionary

    Returns
    -------
    :class:`xarray.Dataset`
        Indexed by time. The number of pixel steps taken between each observation.

    """
    # if more than one storm, just run on each storm
    if "storm" in storm_ds.dims:
        pixel_step_list = []
        for ii in range(len(storm_ds.storm)):
            this_ds = timesteps_to_pixelsteps(
                storm_ds.isel(storm=ii),
                p,
                addl_vars=addl_vars,
            )
            # if no time points, just ignore - these occur when storm only has one time
            # obs
            if len(this_ds.time) > 0:
                pixel_step_list.append(this_ds)

        return xr.concat(pixel_step_list, dim="storm", join="outer")

    # clean up storm before converting from time steps to pixel steps
    storm_ds = storm_ds[
        [
            dv
            for dv in storm_ds.data_vars
            if dv in addl_vars + ["storm_lon", "storm_lat"]
        ]
    ]
    storm_ds = storm_ds.isel(time=storm_ds.storm_lon.notnull())

    # calculate the number of pixels a storm eye crosses between observations
    n_steps = pixels_per_segment(storm_ds, p)

    # return empty dataset if no pixel steps
    if n_steps.sum() == 0:
        return storm_ds.isel(time=[])

    # make the time coord an actual datetime variable
    storm_ds = storm_ds.swap_dims({"time": "datetime"}).drop_vars("time")

    # Ensure datetime coordinate has nanosecond precision for interpolation
    storm_ds["datetime"] = storm_ds.datetime.astype("datetime64[ns]")

    # calculate the amount of time the storm eye spends at each pixel it crosses
    dt_new = calculate_time_per_pixel_step(
        storm_ds.datetime.diff(dim="datetime", label="lower") / np.timedelta64(1, "s"),
        n_steps,
    )

    # convert from time per pixel step to "time since t=0"
    out = dt_new.cumsum()

    # add the initial point (t=0)
    out = np.insert(out, 0, 0)

    # add time at t=0 to these "time since t=0" values
    out = pd.to_timedelta(out, "s").values
    out = xr.DataArray(data=out, dims=("datetime",))

    # this funky syntax is needed for xarray 0.14 and earlier. Once we move to newer
    # images, we can just do t_new = storm_ds.datetime[0] + out
    start = np.datetime64(storm_ds.datetime[0].item(), "ns")

    t_new = (start + out).dt.round("s")

    # confirm we are starting and ending at the right times
    # Use pd.Timestamp for robust comparison across datetime types
    old_st = pd.Timestamp(storm_ds.datetime[0].dt.round("s").values)
    old_end = pd.Timestamp(storm_ds.datetime[-1].dt.round("s").values)
    new_st = pd.Timestamp(t_new[0].values)
    new_end = pd.Timestamp(t_new[-1].values)
    assert old_end == new_end, (
        f"Last time step ({old_end}) does not match last pixel step ({new_end})"
    )
    assert old_st == new_st, (
        f"First time step ({old_st}) does not match first pixel step ({new_st})"
    )

    # interpolate all values in ds_storm to these new time points
    storm_new = storm_ds.interp({"datetime": t_new}, assume_sorted=True)

    # convert datetime dim/coord back to data var
    storm_new["time"] = ("datetime", np.arange(len(storm_new.datetime)))
    storm_new = storm_new.swap_dims({"datetime": "time"})
    storm_new = storm_new.reset_coords(["datetime"])

    return storm_new


def prep_tracks(
    ds_path,
    xlim,
    ylim,
    params,
    valid_storms=None,
    trackset_type="ibtracs",
):
    """Format trackset for processing by licrice.

    Parameters
    ----------
    ds : :class:`xarray.Dataset`
        The dataset containing track information (either historical or synthetic)
    xlim : list of length 2
        Upper and lower x limit for bounding box.
    ylim : list of length 2
        Upper and lower y limit for bounding box.
    params : dict
        LICRICE model params, i.e. from `pyTC.settings.Settings().PARAMS["licrice"]`
    client : :py:class:`dask.distributed.Client`
        If not None, the client to use to parallelize the track prepping
    dask_priority : int, optional
        Only used if ``client`` is not None. The priority assigned to tasks submitted to
        ``client``.

    Returns
    -------
    :py:class:`xarray.Dataset`
        tracks dataset ready for use in LICRICE

    """
    load_kwargs = {"chunks": None, "trackset_type": trackset_type}
    if valid_storms is not None:
        load_kwargs["storm"] = valid_storms
    ds = load_tracks(ds_path, params, **load_kwargs)

    if valid_storms is None:
        # filter to only tracks that at least once intersect buffered domain of interest
        valid_times = tracks.utils.find_valid_times(
            ds,
            lat_var="storm_lat",
            lon_var="storm_lon",
            xlim=xlim,
            ylim=ylim,
            include_middle=True,
        )

        ds = ds.isel(storm=valid_times.any(dim="time"))

    ##############
    # convert observation level from time step to pixel step.
    # The conversion from time step to pixel step level data must be done before
    # filtering out time steps that dont fall within a domain, because
    # this step includes interpolation between time steps.
    ##############
    # first convert to continuous longitude scale before longitude interpolation
    ds = ds.assign(storm_lon=tracks.utils.longitude_to_continuous_scale(ds.storm_lon))

    vars_to_keep = [
        dv for dv in ds.data_vars if dv in _KEEP_VARS + ["season", "u850store"]
    ]

    ds_pixelstep = timesteps_to_pixelsteps(ds, params, addl_vars=vars_to_keep)

    # now convert back to [-180,180] longitudes
    ds_pixelstep = ds_pixelstep.assign(
        storm_lon=tracks.utils.longitude_to_discontinuous_scale(ds_pixelstep.storm_lon),
    )

    # calculate x and y translational velocities using new pixelsteps
    ds_pixelstep = tracks.velocity.calculate_v_trans_x_y(
        ds_pixelstep,
        lat_var="storm_lat",
        lon_var="storm_lon",
    )
    ds_pixelstep = tracks.velocity.calculate_v_total(
        ds_pixelstep,
        "storm_lat",
        "storm_lon",
        baroclinic_effect="u850store" in vars_to_keep,
    )

    # check to make sure no missing values where there shouldnt be
    for test_var in _KEEP_VARS:
        # dont need pres when using the modified rakine
        if (test_var == "pres") and (
            "modified_rankine" in params["wind"]["vortex_func"]
        ):
            continue
        testing.trackset_integrity_check(
            ds_pixelstep,
            test_var,
            ["datetime", "storm_lat", "storm_lon"],
        )

    # now we can filter out time steps that fall outside of the buffered domain by
    # setting ``include_middle`` to false
    ds_out = tracks.utils.filter_track_times(
        ds_pixelstep,
        xlim,
        ylim,
        False,
        lat_var="storm_lat",
        lon_var="storm_lon",
        addl_vars=vars_to_keep + ["v_trans_x", "v_trans_y", "v_total"],
    )

    return ds_out


def find_valid_tracks(ds_path, params, bboxes, trackset_type="ibtracs"):
    out_tracks = {}

    ds = load_tracks(ds_path, params, trackset_type=trackset_type)

    for bbox_ix, (bbox, metadata) in enumerate(bboxes.items()):
        xlim = metadata["xlim"]
        ylim = metadata["ylim"]

        # filter to only tracks that at least once intersect buffered domain of interest
        valid_times = tracks.utils.find_valid_times(
            ds,
            lat_var="storm_lat",
            lon_var="storm_lon",
            xlim=xlim,
            ylim=ylim,
            include_middle=True,
        )

        # which storms have any valid times
        valid_storms = valid_times.any(dim="time").values
        valid_storms_names = ds.storm[valid_storms]
        valid_storms_names = valid_storms_names.reset_coords(
            [
                c
                for c in valid_storms_names.coords
                if c not in ["storm", "ensemble", "storm_ix"]
            ],
            drop=True,
        )

        if valid_storms.sum():
            out_tracks[bbox] = {}
            out_tracks[bbox]["valid_track_ix"] = np.arange(len(valid_storms))[
                valid_storms
            ]
            out_tracks[bbox]["valid_tracks"] = valid_storms_names
            out_tracks[bbox]["start_dates"] = ds.datetime.isel(
                time=0,
                storm=valid_storms,
                drop=True,
            ).values

    return out_tracks


def augment_jobs_with_storm_info(res, bboxes, extra_info=None):
    if extra_info is None:
        extra_info = [{}] * len(res)
    run_jobs = []
    for rx, r in enumerate(res):
        for bbox in r.keys():
            this_j = extra_info[rx].copy()
            this_j["bbox"] = bbox
            this_j["metadata"] = bboxes[bbox]
            this_j["valid_tracks"] = r[bbox]["valid_tracks"]
            this_j["start_dates"] = r[bbox]["start_dates"]
            run_jobs.append(this_j)
    return run_jobs


def init_output_zarr(
    storms,
    start_dates,
    xlim,
    ylim,
    grid_width,
    attr_dict,
    storm_chunksize,
    outpath,
    n_tracks_complete=None,
    overwrite=False,
):
    if n_tracks_complete is not None and n_tracks_complete >= 0:
        return n_tracks_complete

    # prep output zarr
    out_grid = get_output_grid(xlim, ylim, grid_width)
    data_chunk = da.empty(
        (len(storms), out_grid.y_ix.size, out_grid.x_ix.size),
        chunks=(storm_chunksize, out_grid.y_ix.size, out_grid.x_ix.size),
        dtype=np.float32,
    )
    out = xr.Dataset(
        {
            "maxs": (("storm", "y_ix", "x_ix"), data_chunk),
            "pddi": (("storm", "y_ix", "x_ix"), data_chunk.copy()),
            "start_date": (("storm",), start_dates),
        },
        coords={
            "storm": storms,
            "y_ix": out_grid.y_ix,
            "x_ix": out_grid.x_ix,
        },
    )

    # document variables
    out = _update_attrs(out, attr_dict, grid_width)

    # only don't overwrite if the exact same formatted dataset is there
    if (not overwrite) and outpath.is_dir():
        ok = True
        exists = xr.open_zarr(str(outpath), chunks=None)

        if set(exists.coords) != set(out.coords):
            ok = False

        for d in exists.coords:
            if d not in out.coords or (exists[d] != out[d]).any():
                ok = False
                break
        if ok:
            return None

    # Use NaN as fill_value so unwritten chunks are correctly identified as
    # "not yet computed" by check_finished_zarr_workflow (0.0 fill would
    # make the check think computation was already done).
    out.to_zarr(
        str(outpath),
        mode="w",
        compute=False,
        encoding={
            "maxs": {"dtype": "float32", "fill_value": float("nan")},
            "pddi": {"dtype": "float32", "fill_value": float("nan")},
        },
    )


def _update_attrs(ds, attr_dict, grid_width):
    ds.maxs.attrs.update(
        {
            "units": "m/s",
            "long_name": "maximum total wind speed",
            "description": (
                "Maximum total wind speed experienced by each pixel during the "
                "duration of the storm."
            ),
        },
    )
    ds.pddi.attrs.update(
        {
            "units": "m^3/s^2",
            "long_name": "power dissapation density index",
            "description": (
                "The sum over each time step of total wind speed cubed times the "
                "length of a time step for each pixel."
            ),
        },
    )
    ds.y_ix.attrs.update(
        {
            "long_name": "latitude grid index",
            "description": (
                "latitude transformed to an index where the origin is the grid cell "
                "that has a South edge at 0 lat and one index is equivalent to one cell"
            ),
            "cell_size": grid_width,
        },
    )
    ds.x_ix.attrs.update(
        {
            "long_name": "longitude grid index",
            "description": (
                "longitude transformed to an index where the origin is the grid cell "
                "that has a West edge at 0 lon and one index is equivalent to one cell"
            ),
            "cell_size": grid_width,
        },
    )
    ds.attrs.update(attr_dict)
    return ds
