"""Utility functions for LICRICE.

Extracted from pyTC/utilities.py. pint/clawpack dependencies replaced with a
lookup-table approach. convert_units, geoclaw_convert, and _UNIT_CONVERSIONS are new.
_smooth_interp_w_other_data_inner, smooth_fill, bin_data, and
check_finished_zarr_workflow are verbatim from coastal-core (GCS checks simplified).
"""

import numpy as np
import xarray as xr


# unit conversion (replaces pint/clawpack dependency)  # NEW
_UNIT_CONVERSIONS = {  # NEW
    ("kts", "m/s"): 0.514444,
    ("m/s", "kts"): 1 / 0.514444,
    ("km", "m"): 1000.0,
    ("m", "km"): 0.001,
    ("hpa", "pa"): 100.0,
    ("pa", "hpa"): 0.01,
    ("mb", "pa"): 100.0,
    ("mbar", "pa"): 100.0,
    ("nmile", "km"): 1.852,
    ("mb", "hpa"): 1.0,
}


def convert_units(data, from_to):  # NEW
    """data conversion
    
    Parameters
    data : scalar or array, original value to convert from
    from_to : tuple, 2-tuple of strings, as (Units to convert from, Units to convert to)
    Returns
    converted : scalar or array
    """
    key = (from_to[0].lower(), from_to[1].lower())
    if key not in _UNIT_CONVERSIONS:
        raise ValueError(f"Unsupported unit conversion: {from_to[0]} -> {from_to[1]}")
    return data * _UNIT_CONVERSIONS[key]


def geoclaw_convert(data, old_units, new_units):  # NEW
    """unit conversion replacing clawpack.geoclaw.units.convert.

    Supports km<->m and hPa/mb<->Pa conversions needed by LICRICE
    """
    key = (old_units.lower(), new_units.lower())
    if key not in _UNIT_CONVERSIONS:
        raise ValueError(f"Unsupported unit conversion: {old_units} -> {new_units}")
    return data * _UNIT_CONVERSIONS[key]


def _smooth_interp_w_other_data_inner(to_fill, filler, time_dim="date_time"):
    """Interpolate with secondary dataset but do not extrapolate."""
    # create data array that looks like the one you're trying to fill
    filler_w_holes = filler.interpolate_na(dim=time_dim, use_coordinate=True).where(
        to_fill.notnull(),
    )

    # linearly interpolatle it
    filler_interp = filler_w_holes.interpolate_na(dim=time_dim, use_coordinate=True)

    # linearly interpolate original data
    to_fill_interp = to_fill.interpolate_na(dim=time_dim, use_coordinate=True)

    # calculate multiplicative anomaly from linear interpolation
    rat_filler = (filler / filler_interp).where(
        filler_w_holes.isnull() & filler.notnull(),
    ) * to_fill_interp

    # calculate additive anomaly
    diff_filler = (filler - filler_interp).where(
        filler_w_holes.isnull() & filler.notnull(),
    ) + to_fill_interp

    # take minimum of multiplicative and additive, to be on the conservative end of
    # adjusting from the linear interpolation
    filler_out = rat_filler.where(
        np.abs(rat_filler - to_fill_interp) < np.abs(diff_filler - to_fill_interp),
        diff_filler,
    )

    # apply interpolated value
    return to_fill.fillna(filler_out)


def smooth_fill(
    da1_in,
    da2_in,
    fill_all_null=True,
    time_dim="time",
    other_dim="storm",
    interpolate=False,
):
    """Fill values from 2D dataarray ``da1`` with values from 2D dataarray ``da2``. If
    filling the beginning or end of a storm, pin the ``da2`` value to the ``da1`` value
    at the first/last point of overlap and then use the ``da2`` values only to estimate
    the `change` in values over time, using a ratio of predicted value in the desired
    time to the reference time. This can also be used when, for example, ``da1`` refers
    to RMW and ``da2`` refers to ROCI. In this case, you want to define
    ``fill_all_null=False`` to avoid filling RMW with ROCI when no RMW values are
    available but some ROCI values are available.

    Parameters
    ----------
    da1,da2 : :class:`xarray.DataArray`
        DataArrays indexed by storm and time
    fill_all_null : bool, optional
        If True, fill storms where da1 is entirely null using da2 values directly.

    Returns
    -------
    :class:`xarray.DataArray`
        Same as ``da1`` but with NaN's filled by the described algorithm.

    Raises
    ------
    AssertionError :
        If there are "interior" NaN's in either dataset, i.e. if any storm has a NaN
        after the first non-NaN but before the last non-NaN. These should have
        previously been interpolated.

    """
    da1 = da1_in.copy()
    da2 = da2_in.copy()
    either_non_null = da1.notnull() | da2.notnull()

    if interpolate:
        da1 = _smooth_interp_w_other_data_inner(da1, da2, time_dim=time_dim)
    da1 = da1.interpolate_na(dim=time_dim, use_coordinate=True)
    da2 = da2.interpolate_na(dim=time_dim, use_coordinate=True)
    for da in [da1, da2]:
        assert da.interpolate_na(dim=time_dim).notnull().sum() == da.notnull().sum()

    adjust = da1.reindex({other_dim: da2[other_dim]})
    first_valid_index = (adjust.notnull() & da2.notnull()).argmax(dim=time_dim)
    last_valid_index = (
        adjust.bfill(time_dim).isnull() | da2.bfill(time_dim).isnull()
    ).argmax(dim=time_dim) - 1

    all_null = adjust.isnull().all(dim=time_dim)
    if not fill_all_null:
        all_null *= False

    est_to_obs_rat_first = adjust.isel({time_dim: first_valid_index}) / da2.isel(
        {time_dim: first_valid_index},
    )

    est_val = da2.where(
        all_null | adjust.ffill(time_dim).notnull(),
        da2 * est_to_obs_rat_first,
    )

    est_to_obs_rat_last = adjust.isel({time_dim: last_valid_index}) / da2.isel(
        {time_dim: last_valid_index},
    )

    est_val = est_val.where(
        all_null | adjust.bfill(time_dim).notnull(),
        da2 * est_to_obs_rat_last,
    )

    # fill storms with da1 vals using the full da2 time series. For storms with some da1
    # vals, fill the tails using da2 scaled so that it matches at the first and last
    # points seen in both da1 and da2
    out = da1.fillna(est_val)

    # make sure we didn't add vals
    return out.where(either_non_null)


def bin_data(da, res_spatial):
    """Bin data.

    Parameters
    ----------
    da : :py:class:`xarray.DataArray`
        Data to get binned
    res_spatial : float
        bin size

    Returns
    -------
    `da` binned in bins of size `res_spatial`

    """
    return (np.floor(da / res_spatial) * res_spatial) + res_spatial / 2


def check_finished_zarr_workflow(
    finalstore=None,
    tmpstore=None,
    varname=None,
    final_selector={},
    mask=None,
    check_final=True,
    check_temp=True,
    how="all",
):
    def _check_notnull(da, how):
        out = da.notnull()
        if how == "all":
            return out.all().item()
        elif how == "any":
            return out.any().item()
        raise ValueError(how)

    finished = False
    temp = False
    if check_final:
        finished = xr.open_zarr(finalstore, chunks=None)[varname].sel(
            final_selector,
            drop=True,
        )
        if mask is not None:
            finished = finished.where(mask, 1)
        finished = _check_notnull(finished, how)

    if finished:
        return True
    if check_temp:
        if tmpstore.fs.isdir(tmpstore.root):
            try:
                temp = xr.open_zarr(tmpstore, chunks=None)
                if mask is not None:
                    temp = temp.where(mask, 1)
                if (
                    varname in temp.data_vars
                    and "year" in temp.dims
                    and len(temp.year) > 0
                ):
                    finished = _check_notnull(temp[varname], how)
            except Exception:
                ...
    return finished
