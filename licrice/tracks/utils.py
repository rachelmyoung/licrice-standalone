"""Track utility functions for LICRICE.

Extracted from pyTC.tracks.utils with only the functions needed by the LICRICE wind model.
"""

import numpy as np
import pandas as pd
import xarray as xr

from licrice import spatial

COMMON_VAR_ATTRS = {
    "v_total": {
        "long_name": "maximum total windspeed",
        "units": "m/s",
        "description": (
            "maximum 1-minute sustained windspeed, including translational and "
            "rotational components"
        ),
    },
    "v_circular": {
        "long_name": "maximum circular wind speed",
        "units": "m/s",
        "description": (
            "The maximum circular wind speed at each point along each "
            "track. Note that this is not the maximum wind speed, only "
            "the circular component. Maximum total speed is given by "
            "``v_total``."
        ),
    },
    "v_translational": {
        "long_name": "translational velocity",
        "units": "m/s",
        "description": (
            "Translational velocity at each point along each track. Calculated using "
            "the central difference of the great circle distance between each point. "
            "Additionally, `pyTC.tracks.velocity.smooth_velocity` was used to smooth "
            "this velocity. Note that this is not the maximum wind speed, only the "
            "translational component."
        ),
    },
    "pstore": {
        "long_name": "Minimum central pressure",
        "units": "hPa",
        "description": "Minimum central pressure",
    },
    "rmstore": {
        "long_name": "Radius of maximum winds",
        "units": "km",
        "description": "Radius of maximum winds",
    },
    "storm_radius": {
        "long_name": "Radius of outermost closed isobar",
        "units": "km",
        "description": (
            "Radius of outermost closed isobar, a measure of the total radial extent "
            "of the storm"
        ),
    },
}


###############################################################################
# Helper functions for cleaning and wrangling track
###############################################################################


def mask_invalid_values(ds, time_var="datetime"):
    """Mask values of data variables for which time dimension is NaT.

    Warning: this is necessary to keep the original shape/indexing of
    the dataset instead of using .where, which broadcasts `time`
    coordinate to other dimensions that are not indexed by time.
    """
    new_ds = xr.Dataset()
    new_ds = new_ds.assign_coords(dict(ds.coords))
    for var in ds.data_vars.keys():
        if "time" in ds[var].dims:
            new_ds[var] = ds[var].where(ds[time_var].notnull(), drop=True)
        else:
            new_ds[var] = ds[var]
    return new_ds


def get_delta_time(ds):
    """Calculate delta time between each observations.

    Parameters
    ----------
    ds : :class:`xarray.Dataset`
        Tracks with time coordinates.
        Time is observation indices. 'datetime' variable
        defines observation time in datetime64[ns], and
        can be np.nan. Number of observations varies by
        storm.

    Returns
    -------
    :class:`xarray.DataArray`
        Delta time between each observation in seconds.
        Last observation for each storm does not have
        timedelta value.

    """
    timedelta = (ds.datetime.shift(time=-1) - ds.datetime) / np.timedelta64(1, "s")
    timedelta.name = "timedelta"
    return timedelta


def find_valid_times(
    ds_storm,
    xlim=[-129, -65],
    ylim=[23, 52],
    lat_var="lat",
    lon_var="lon",
    rad_var="radius",
    include_middle=True,
):
    """Construct boolean DataArray of observations (storm x time) that come within a
    ROCI-width of a domain. If ``include_middle`` is True, include **all** steps between
    and including the first time the storm enters a domain and the last time it leaves,
    even if some of those interior points lie outside of the domain. In this case, give
    a one time step buffer on either side of the valid points in order to model the edge
    behavior just as the storm is entering the domain.

    Parameters
    ----------
    ds_storm : :class:`xarray.Dataset`
        2-D Dataset indexed by `time` and `storm` which contains the tracks for
        one or more storms.
    xlim : list
        list of length 2. Upper and lower x limit for bounding box. If
        ``xlim[1] < xlim[0]``, then domain will wrap around dateline.
    ylim : list
        list of length 2. Upper and lower y limit for bounding box.
    lat_var : str
        variable name for latitude data_var
    lon_var : str
        variable name for longitude data_var

    Returns
    -------
    :class:`xarray.Dataset`
        ``ds_storm`` with the following data variables: [`lat_var`,`lon_var`] +
        `addl_vars`, subsetted to only include times in the bounding box

    """
    # make sure radius is in meters
    rad = ds_storm[rad_var]
    if hasattr(rad, "attrs") and "units" in rad.attrs:
        units = rad.attrs["units"].lower()
        if units == "km":
            rad = rad * 1000.0
        elif units != "m":
            # Try pint_xarray if available, otherwise assume meters
            try:
                import pint_xarray  # noqa: F401
                rad = rad.pint.quantify().pint.to("m").pint.dequantify()
            except ImportError:
                pass
    ds_storm[rad_var] = rad

    rad_y = ds_storm[rad_var] / spatial.LAT_TO_M
    rad_x = rad_y / np.cos(np.deg2rad(ds_storm[lat_var]))
    bound_y = xr.concat((ylim[0] - rad_y, ylim[1] + rad_y), dim="bound")
    bound_x = xr.concat((xlim[0] - rad_x, xlim[1] + rad_x), dim="bound")

    if xlim != [-180, 180]:  # dont need to buffer if whole domain captured
        bound_x = ((bound_x + 180) % 360) - 180

    # observations inside bounding box
    valid_y = (ds_storm[lat_var] >= bound_y.isel(bound=0)) & (
        ds_storm[lat_var] <= bound_y.isel(bound=1)
    )

    # if xlim[0] > xlim[1], need to wrap around the anti-meridian so we use "or"
    valid_x_inv = (ds_storm[lon_var] >= bound_x.isel(bound=0)) | (
        ds_storm[lon_var] <= bound_x.isel(bound=1)
    )
    # otherwise, need to constrain to be within the two bounds, so we use "and"
    valid_x_normal = (ds_storm[lon_var] >= bound_x.isel(bound=0)) & (
        ds_storm[lon_var] <= bound_x.isel(bound=1)
    )

    valid_x = valid_x_normal.where(
        bound_x.isel(bound=1) >= bound_x.isel(bound=0),
        valid_x_inv,
    )

    valid = valid_y & valid_x

    # if we don't need to include interior points, stop here
    if not include_middle:
        return valid

    #################
    # Now we need to include the inner time points in case a storm enters, leaves, then
    # enters a domain again
    #################

    # first crop any storms with no intersection
    any_intersection = valid.sum(dim="time") > 0
    valid_subset = valid.isel(storm=any_intersection)

    # include both 1 step before and one step after the intersection
    valid_subset = (
        valid_subset.shift(time=1).fillna(False)
        | valid_subset.shift(time=-1).fillna(False)
        | valid_subset
    ) & ds_storm[lat_var].notnull()

    # fill all the times in between first entrance and last exit
    valid_subset = valid_subset.where(valid_subset).interpolate_na(
        dim="time",
        method="nearest",
    )

    # add back in the storms with no intersection
    return valid_subset.reindex(storm=ds_storm.storm).fillna(False).astype(bool)


def find_valid_indices(vt, time_dim="time"):
    """Find time indices in `vt` with True values for each storm.

    Parameters
    ----------
    vt : :class:`xarray.DataArray`
        2-D array indexed by `time` and `storm` with the tracks for
        one or more storms. Values are booleans
    time_dim : str, optional
        name of ``vt`` time dimension

    Returns
    -------
    :class:`xarray.DataArray`
        2-D array indexed by `valid_time` and `storm`. Values are indices of
        `times` in `valid_times` with a True value. Array shape equivalent to
        `valid_times`

    """
    valid_times = vt.copy()

    # want to extract valid indices not valid coordinate values so ensure time
    # coordinate and index are equal
    valid_times.coords[time_dim] = np.arange(len(valid_times[time_dim]))

    valid_indices = (
        valid_times[time_dim]
        .where(valid_times)
        .to_series()
        .dropna()
        .astype(int)
        .groupby(level="storm")
        .apply(lambda x: x.reset_index(drop=True).rename_axis("valid_time"))
    )
    if valid_indices.size == 0:
        valid_indices = valid_indices.to_frame()
        valid_indices["valid_time"] = []
        valid_indices = valid_indices.set_index("valid_time", append=True).iloc[:, 0]
    valid_indices = valid_indices.to_xarray().reindex_like(
        valid_times,
    )  # want to maintain storm dimension length in case dropped storms
    return valid_indices


def filter_valid_indices(ds_storm, valid_indices, pvars=[], time_dim="time"):
    """Reindex track dataset (`ds_storm`) to index only desired time indices.

    Parameters
    ----------
    ds_storm : :class:`xarray.Dataset`
        2-D Dataset indexed by `time` and `storm` which contains the tracks for
        one or more storms.
    valid_indices : :class:`xarray.DataArray`
        2-D array indexed by `valid_time` and `storm`. Values are indices of
        `time` observations in `ds_storm` that should be perserved.
        Array shape equivalent to `ds_storm`
    pvars : list, optional
        List of data vars in `ds_storm` to preserve in output dataset.
    time_dim : str, optional
        name of `ds_storm` time dimension

    Returns
    -------
    :class:`xarray.Dataset`
        `ds` with the following data variables: `pvars`,
        subsetted to only include time indices in `valid_indices`

    """
    # valid indices must be time indices of `ds` not time coordinates
    assert valid_indices.fillna(0).max() <= len(ds_storm[time_dim])
    assert valid_indices.fillna(0).min() >= 0

    out_ds = xr.Dataset()
    for v in pvars:
        if time_dim not in ds_storm[v].dims:
            out_ds[v] = ds_storm[v]
        else:
            out_ds[v] = (
                ds_storm[v]
                .isel({time_dim: valid_indices.fillna(-1).astype(int)})
                .where(valid_indices.notnull())
                .drop_vars([c for c in ds_storm[v].coords if c not in ds_storm[v].dims])
            )

    # replace time coordinate with valid_time coordinate --
    # the old time coordinate is no longer a valid measure of time for dataset (i think)

    if "valid_time" in list(out_ds.data_vars) + list(out_ds.coords) + list(out_ds.dims):
        out_ds = out_ds.drop_vars(time_dim).rename({"valid_time": time_dim})

    return out_ds


def filter_track_times(
    ds_storm,
    xlim,
    ylim,
    include_middle,
    lat_var="lat",
    lon_var="lon",
    rad_var="radius",
    time_dim="time",
    addl_vars=[],
):
    """Reindex storm to index only times and storms between (and including) the first
    and last times the storm intersects the domain. We add 1 time point on either side
    so that we can model the in-between time step points in which the storm is just
    barely entering the domain.

    Parameters
    ----------
    ds_storm : :class:`xarray.Dataset`
        2-D Dataset indexed by `time` and `storm` which contains the tracks for
        one or more storms.
    xlim : Sequence of length 2
        Upper and lower x limit for bounding box.
    ylim : Sequence of length 2
        Upper and lower y limit for bounding box.
    include_middle : bool
        If True, include **all** points in between first and last point the storm
        falls within the buffered domain. This is useful for a first pass in cropping,
        prior to timestep interpolation and followed by a second pass where we crop out
        these intermediate points after they have been properly interpolated.
    lat_var : str, optional
        name of latitude data variable
    lon_var : str, optional
        name of longitude data variable
    time_dim : str, optional
        name of `ds_storm` time dimension
    addl_vars : list, optional
        List of coordinates beyond `lat` and `lon` in `ds_storm` in outputted dataset.

    Returns
    -------
    :class:`xarray.Dataset`
        ``ds_storm`` with the following data variables: [`lat_var`,`lon_var`] +
        `addl_vars`, subsetted to only include times after the storm first intersected
        the domain and before the storm last intersected the domain.

    """
    # create boolean data array -- true where lat/lon coordinate falls in bbox
    valid_times = find_valid_times(
        ds_storm,
        lat_var=lat_var,
        lon_var=lon_var,
        xlim=xlim,
        ylim=ylim,
        rad_var=rad_var,
        include_middle=include_middle,
    )

    # which storms have any valid times
    valid_storms = valid_times.any(dim="time")
    if valid_storms.sum() == 0:
        return ds_storm[[]]

    valid_times = valid_times.isel(storm=valid_storms)

    # create a data array of time indices for obs where lat/lon coordinate falls in bbox
    valid_indices = find_valid_indices(valid_times, time_dim=time_dim)

    if valid_indices.isnull().all():
        return ds_storm[[]]

    # filter on valid time indices for each storm and reset time coordinates to reflect
    # new position
    out_ds = filter_valid_indices(
        ds_storm.isel(storm=valid_storms),
        valid_indices,
        pvars=addl_vars + ["datetime", lat_var, lon_var],
        time_dim=time_dim,
    )

    return mask_invalid_values(out_ds)


def find_last_valid_time_point(ds, var="datetime"):
    """Calculates last valid time point for each storm in dataset. Works only if ``var``
    values are masked (np.nan instead of 0 where timestep is not a valid observation)

    Parameters
    ----------
    ds : :class:`xarray.Dataset`
        2-D Dataset indexed by `time` and `storm` which contains the tracks for
        one or more storms.
    var : str, optional
        name of variable over which to calculate last valid time point

    Returns
    -------
    Dataarray indexed by storm, containing last
    valid time point (int) for each storm.

    """
    return ds[var].notnull().sum("time") - 1


def longitude_to_continuous_scale(longitude_array, time_dim="time"):
    """Convert longitude for a trackset to a continuous scale for each storm.

    Parameters
    ----------
    longitude_array : :class:`xarray.DataArray`
        Array of longitudes.
    time_dim : str, optional
        temporal dimension name

    Returns
    -------
    :class:`xarray.DataArray`
        A measure of longitude that is continuous within each storm

    """
    attrs = longitude_array.attrs.copy()

    # get degree distance to next longitude point
    lon_diff = (
        longitude_array.diff(dim=time_dim, label="upper")
        .reindex_like(longitude_array)
        .fillna(0)
    )

    # if this jumps > 180 degrees, assume you go the other direction
    lon_diff = lon_diff.where(lon_diff < 180, lon_diff - 360).where(
        lon_diff > -180,
        lon_diff + 360,
    )

    # now retrace path
    out = lon_diff.cumsum(dim=time_dim, skipna=False) + longitude_array.isel(
        {time_dim: 0},
    )
    out = out.where(longitude_array.notnull())

    out.attrs.update(attrs)

    return out


def longitude_to_discontinuous_scale(longitude_array):
    """Convert longitude values to the normal scale ie longitude values fall in the
    following interval [-180,180].

    Parameters
    ----------
    longitude_array : array-like
        longitude values that wrap the dateline but do not wrap zero

    Returns
    -------
    array-like
        longitude array placed on a discontinuous scale

    """
    return ((longitude_array + 180) % 360) - 180


def interpolate_nans(
    ds: xr.Dataset,
    var_list: list = ["storm_radius", "rmstore", "pstore", "v_total"],
    use_coordinate: bool = True,
) -> xr.Dataset:
    """Wrapper of da.interpolate_na applied to specified variables, preserving attrs."""
    attribs = {i: ds[i].attrs.copy() for i in var_list}
    ds = ds.assign(
        ds[var_list].interpolate_na(
            dim="time",
            method="linear",
            use_coordinate=use_coordinate,
        ),
    )
    [ds[var].attrs.update(attrib) for var, attrib in attribs.items()]
    return ds


def drop_leading_and_trailing_nans(ds_storm, var="v_total", time_dim="time"):
    """Drop leading and trailing nans for each storm in a track dataset. A leading or
    trailing nan here refers to timesteps either at the beginning or end of a storm time
    series that have nan values for ``var`` and non nan values for other variables in
    the trackset such as latitude and longitude. This function will mask all variables
    at the beginning and end of the storm time series that are not nans where ``var`` is
    a nan.

    Parameters
    ----------
    ds_storm : :class:`xarray.Dataset`
        track dataset with only a storm and time dimension
    var : str, optional
        variable used to assess whether tracks have leading nans
    time_dim : str, optional
        temporal dimension over which dropping leading nans

    Returns
    -------
    :class:`xarray.Dataset`
        track dataset with leading nans removed from all tracks

    """
    # find time index of first and last non nan value
    valid_times = ds_storm[var].notnull()
    valid_indices = find_valid_indices(valid_times, time_dim=time_dim)

    vi = np.array([np.arange(valid_times.time.shape[0])] * valid_times.storm.shape[0])
    if vi.shape[valid_times.dims.index("storm")] != valid_times.storm.shape[0]:
        vi = np.swapaxes(vi, 0, 1)

    vi = xr.DataArray(vi, coords=valid_times.coords, dims=valid_times.dims)
    vi = vi.where(vi <= valid_indices.max(dim="valid_time"))
    vi = vi.where(vi >= valid_indices.min(dim="valid_time"))

    valid_indices_updated = find_valid_indices(vi.notnull(), time_dim=time_dim)

    return filter_valid_indices(
        ds_storm,
        valid_indices_updated,
        pvars=list(ds_storm.data_vars),
        time_dim=time_dim,
    )


def drop_stationary_storms(
    ds,
    lat_var="latstore",
    lon_var="longstore",
    time_dim="time",
):
    """Drop stationary storms (storms with more than one observations that don't move)

    Parameters
    ----------
    ds : :class:`xarray.Dataset`
        track dataset with a time and storm dimension
    lat_var : str
        latitude variable name
    lon_var : str
        longitude variable name
    time_dim : str
        time dimension name

    Returns
    -------
    ds : :class:`xarray.Dataset`
        `ds` without storms with more than one observation that don't move
    stationary_storms : :class:`xarray.Dataset`
        stationary storms in `ds`

    """
    num_obs = ds[lat_var].notnull().sum(dim=time_dim)

    stationary_storms = (
        (ds[lat_var] == ds[lat_var].isel({time_dim: 0}))
        & (ds[lon_var] == ds[lon_var].isel({time_dim: 0}))
    ).sum(dim=time_dim) == num_obs

    keepers = ~(stationary_storms)

    return ds[{"storm": keepers}], ds[{"storm": ~keepers}]


def assess_var_missingness(
    ds,
    var="v_total",
    lat_var="latstore",
    missingness_tolerance=0.5,
):
    """Divide `ds` into 3 categories:

    - g2g: tracks with <= `missingness_tolerance` missing `var` observations and more
      than one `var` observation
    - one_ol_ob: tracks with <= 1 `var` observation
    - missing: tracks with > 1 `var` observation as well as > 50% missing `var`
      observations

    Parameters
    ----------
    ds_in : :class:`xarray.Dataset`
        track dataset
    var : str, optional
        variable in `ds` used to break `ds` tracks into 3 categories
    lat_var : str, optional
        name of latitude variable in `ds`
    missingness_tolerance : float, optional
        ratio of missing observations deemed acceptable

    Returns
    -------
    g2g : :class:`xarray.Dataset`
        tracks with no missing `var` observations
    one_ol_ob : :class:`xarray.Dataset`
        tracks with <= 1 `var` observation
    missing : :class:`xarray.Dataset`
        tracks with > 1 `var` observation and some number of missing `var`
        observations

    """
    # calculate percent of `var` observations missing
    p_missing = (
        ds[lat_var].notnull().sum("time") - ds[var].notnull().sum("time")
    ) / ds[lat_var].notnull().sum("time")

    if "ensemble" in p_missing.dims:
        p_missing = p_missing.any(dim="ensemble")
        ds = ds.isel(ensemble=0)

    # should never have more observations of `var` than lat so pdiff should
    # never be negative
    assert p_missing.min() >= 0

    # calculate number of `var` observations for each storm
    num_obs = ds[var].notnull().sum("time")

    # storms with no missing `var` observations
    # and a sufficient number of observations
    g2g = ds[{"storm": ((p_missing <= missingness_tolerance) & (num_obs > 1))}]

    # storms with one or less observation
    one_ol_ob = ds[{"storm": (num_obs < 2)}]

    # storms with missing observations
    missing = ds[{"storm": ((num_obs > 1) & (p_missing > missingness_tolerance))}]

    return g2g, one_ol_ob, missing
