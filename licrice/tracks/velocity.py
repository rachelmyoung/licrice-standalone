"""Track velocity functions for LICRICE.

Extracted from pyTC/tracks/velocity.py.
"""

import numpy as np
import xarray as xr

from licrice.spatial import great_circle_dist  # NEW
from licrice.utilities import convert_units  # NEW
from . import utils as tutils

# Get conversion factor to convert max wind speed averaging period to 1 minute.
# constructed using:
# https://www.wmo.int/pages/prog/www/tcp/documents/WMO_TD_1555_en.pdf
AVERAGING_PERIOD_CONVERSION_DIVISOR = {
    10: 0.88,
    3: 0.92,
    2: 0.96,
    1: 1,
}

###############################################################################
# Caculating Translational Velocity Workflow
###############################################################################


def add_lon_lat_diffs(
    ds,
    lat_var="latstore",
    lon_var="longstore",
):
    """Add longitude and latitude difference variables to given dataset. Used for
    calculating great circle distance between observation points for each storm.

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
        Track dataset indexed by storm and time.
    lat_var : str
        name of latitude variable
    lon_var : str
        name of longitude variable

    Returns
    -------
    ds : :py:class:`xarray.Dataset`
        returns ``ds`` with two additional variables: ``v_trans_x`` and ``v_trans_y``

    """
    ds["lat_diff"] = (
        ds[lat_var].diff(dim="time", label="lower").reindex_like(ds[lat_var])
    )
    ds["lon_diff"] = (
        ds[lon_var].diff(dim="time", label="lower").reindex_like(ds[lon_var])
    )

    # correct for wrapping dateline or meridian
    ds["lon_diff"] = ds.lon_diff.where(ds.lon_diff < 180, ds.lon_diff - 360).where(
        ds.lon_diff > -180,
        ds.lon_diff + 360,
    )

    # update metadata
    ds["lon_diff"].attrs.update(
        {
            "long_name": "zonal distance to next point",
            "units": "degrees",
            "description": (
                "distance, in degrees longitude, to the next point. ",
                "EQN: lon_diff_{t} = lon_{t+1} - lon_{t}",
            ),
        },
    )

    ds["lat_diff"].attrs.update(
        {
            "long_name": "meridonal distance to next point",
            "units": "degrees",
            "description": (
                "distance, in degrees latitude, to the next point. ",
                "EQN: lat_diff_{t} = lat_{t+1} - lat_{t}",
            ),
        },
    )

    return ds


###############################################################################
# Calculating max wind speed and circular velocity related work flow
###############################################################################


def estimate_scaling_factor(latitude):
    r"""Estimate scaling factor used to calculate v_total:

    $$v_total = v\_circular + scaling\_factor * \sqrt{v\_trans\_x^2 + v\_trans\_y^2}$$

    From WindRiskTech algorithm. Refer to `utransfull`, `utrans`, and `transfunction`
    functions in matlab scripts.

    Parameters
    ----------
    latitude: xr.DataArray
        Latitude values (latstore data variable in WindRiskTech tracks), in storm x time
        dimension dataarray

    Returns
    -------
    DataArray:
        DataArray of scaling factors indexed by storm and time

    """
    # default constants from Kerry
    transfac = 0.8
    transcap = 1  # Changed to 1 on May 30th 2014
    amplitude = 0.35
    centerlat = 35
    latscale = 10

    transfactor = np.minimum(
        transfac + amplitude * (1 + np.tanh((np.abs(latitude) - centerlat) / latscale)),
        transcap,
    )

    # change name
    if isinstance(transfactor, xr.DataArray):
        transfactor.name = "scaling_factor"
    return transfactor


def smooth_velocity(da):
    """Smooth a specified velocity in the input dataset.

    Uses generalized matlab implementation by Kerry, smfac
    (smoothing factor) and equations from
    utrans function in Kerry's matlab code accompanying
    05-2019 tracks.

    Parameters
    ----------
    da : :class:`xarray.DataArray`
        Velocity DataArray indexed by storm and time.

    Returns
    -------
    :class:`xarray.DataArray`
        smoothed `velocity_var` with same dimension and coordinates as
        ``da``.

    """
    smfac = 0.4

    return da + smfac * (
        da.shift(time=1).fillna(da) + da.shift(time=-1).fillna(da) - 2 * da
    )


########################################################################################
# General functions for adding non native velocities to a trackset
########################################################################################


def calculate_v_trans_x_y(ds, lat_var, lon_var, method="centered", smooth=True):
    """Calculate x- and y- components of translational velocity using centered
    difference. Used in LICRICE.

    Parameters
    ----------
    ds : :class:`xarray.Dataset`
        Track dataset indexed by storm and time. ``ds`` must include ``datetime``
        and ``dist_var`` variables.
    lat_var : str, optional
        name of latitude variable in `_ds`
    lon_var : str, optional
        name of longitude variable in `_ds`
    method : "backward", "forward", or "centered" (default)
        Approach to use for Euler velocity calculation
    smooth : bool, default True
        Whether to smooth velocities (as WindRiskTech does in post-processed output but
        not in filter-crossing speed)

    Returns
    -------
    da_out : :py:class:`xarray.DataArray`
        Velocity data array indexed identically to `_ds`

    """
    # use equation in pyTC.utilities, convert km to meters
    out = ds.copy()
    out = add_lon_lat_diffs(out, lat_var, lon_var)
    fwd_dist, fwd_angle = great_circle_dist(
        ax=out[lon_var],
        ay=out[lat_var],
        bx=out[lon_var] + out.lon_diff,
        by=out[lat_var] + out.lat_diff,
        return_angles=True,
    )

    # adjust dist to m
    fwd_dist *= 1e3

    # get x and y components
    fwd_dist_x = fwd_dist * np.cos(fwd_angle)
    fwd_dist_y = fwd_dist * np.sin(fwd_angle)

    # get backward distance, filling to make first timestep use forward dist
    bkwd_dist_x, bkwd_dist_y = map(
        lambda x: x.shift(time=1).bfill("time", 1),
        [fwd_dist_x, fwd_dist_y],
    )

    # fill last timestep for forward dist
    fwd_dist_x, fwd_dist_y = map(lambda x: x.ffill("time", 1), [fwd_dist_x, fwd_dist_y])

    # get timedeltas for both forward and backward step, filling to make first timestep
    # use forward dist
    fwd_timedelta = tutils.get_delta_time(ds)
    bkwd_timedelta = fwd_timedelta.shift(time=1).bfill("time", 1)
    fwd_timedelta = fwd_timedelta.ffill("time", 1)

    if method == "centered":
        dist_x, dist_y, timesteps = map(
            sum,
            [
                (fwd_dist_x, bkwd_dist_x),
                (fwd_dist_y, bkwd_dist_y),
                (fwd_timedelta, bkwd_timedelta),
            ],
        )
    elif method == "backward":
        dist_x = bkwd_dist_x
        dist_y = bkwd_dist_y
        timesteps = bkwd_timedelta
    elif method == "forward":
        dist_x = fwd_dist_x
        dist_y = fwd_dist_y
        timesteps = fwd_timedelta
    else:
        raise ValueError(method)

    # velocity = dist / time
    out["v_trans_x"] = dist_x / timesteps
    out["v_trans_y"] = dist_y / timesteps

    for comp in ["x", "y"]:
        var = f"v_trans_{comp}"

        assert (
            out[var].notnull().sum(dim="time") == out[lon_var].notnull().sum(dim="time")
        ).all(), f"Missing {var} values in middle of storm"

        if smooth:
            out[var] = smooth_velocity(out[var])

    # drop diff vars
    out = out.drop_vars(["lon_diff", "lat_diff"])

    return out


def calculate_v_circular(ds, lat_var="latstore", lon_var="longstore"):
    """Calculate circular velocity from ``v_total``, ``v_trans_x``, and ``v_trans_y``;
    then add to the tracks dataset. Implementation is based on matlab code included with
    Kerry Emanuel's 05-2019 tracks, where a latitudinally dependent fraction of
    translational velocity is subtracted from total velocity.

    Parameters
    ----------
    ds : Dataset
        ibtracs trackset formatted like Emanuel track sets, indexed by storm and time.
    lat_var : str, optional
        name of latitude variable in ``ds``
    lon_var : str, optional
        name of longitude variable in ``ds``

    Returns
    -------
    ds : Dataset
        Same as original tracks, with added ``v_circular``

    """
    if "v_circular" not in ds.data_vars:
        for var in ["v_total", "v_trans_x", "v_trans_y"]:
            assert var in ds.data_vars, (
                f"for desired output, `{var}` must be present in dataset."
            )

        lat_scaling_factor = estimate_scaling_factor(ds[lat_var])

        # calculate maximum circular wind speed
        ds["v_circular"] = ds["v_total"] - (
            lat_scaling_factor * np.sqrt(ds.v_trans_x**2 + ds.v_trans_y**2)
        )

        # circular winds should never be less than zero
        ds["v_circular"] = ds.v_circular.where(ds.v_circular >= 0, 0).where(
            ds.v_total.notnull(),
        )
    else:
        assert (
            ds.v_circular.attrs["units"]
            == tutils.COMMON_VAR_ATTRS["v_circular"]["units"]
        )

    ds["v_circular"].attrs.update(tutils.COMMON_VAR_ATTRS["v_total"])
    return ds


def calculate_v_total(
    ds,
    lat_var="latstore",
    lon_var="longstore",
    baroclinic_effect=True,
):
    """Calculate total velocity from ``v_circular``, ``v_trans_x``, and ``v_trans_y``;
    then add to the tracks dataset. Implementation is based on matlab code included with
    Kerry Emanuel's 05-2019 tracks, where a latitudinally dependent fraction of
    translational velocity is subtracted from total velocity.

    Parameters
    ----------
    ds : Dataset
        ibtracs trackset formatted like Emanuel track sets, indexed by storm and time
    lat_var : str, optional
        name of latitude variable in ``ds``
    lon_var : str, optional
        name of longitude variable in ``ds``
    baroclinic_effect : bool, optional
        If True (default), account for baroclinic effect when calculating v_total from
        gradient and translational components. See ``utransfull.m`` in Kerry Emanuel's
        matlab code. If True, ``ds`` must contain ``u850store`` and ``v850store``

    Returns
    -------
    Dataset
        Same as original tracks, with added ``v_total``.

    """
    out = ds.copy()
    lat_scaling_factor = estimate_scaling_factor(out[lat_var])

    for var in ["v_circular", "v_trans_x", "v_trans_y"]:
        assert var in out.data_vars, (
            f"for desired output, `{var}` must be present in dataset."
        )

    uinc = lat_scaling_factor * out.v_trans_x
    vinc = lat_scaling_factor * out.v_trans_y

    # The below code is taken verbatim from utransfull.m, except translated from kts to
    # m/s
    adjuster = convert_units(1, ("kts", "m/s"))
    if baroclinic_effect:
        udrift = -0.9
        vdrift = 1.5 * np.sign(out[lat_var])
        uinc += (
            0.5
            * (out.v_trans_x - udrift - out.u850store)
            * out.v_circular
            / (15 * adjuster)
        )
        vinc += (
            0.5
            * (out.v_trans_y - vdrift - out.v850store)
            * out.v_circular
            / (15 * adjuster)
        )

        # avoid making each translational increment component greater than circular wind
        uinc, vinc = map(
            lambda x: x * np.minimum(1, out.v_circular / (0.1 * adjuster + np.abs(x))),
            [uinc, vinc],
        )

    out["Uinc"] = uinc
    out["Vinc"] = vinc
    out["v_total"] = out.v_circular + np.sqrt(uinc**2 + vinc**2)
    out["v_total"].attrs.update(tutils.COMMON_VAR_ATTRS["v_total"])

    return out
