"""Spatial utility functions for LICRICE.
Extracted from pyTC.spatial
The climada dependency replaced with an inline implementation of latlon_to_geosph_vector.
"""

from functools import wraps
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

EARTH_RADIUS = 6371.009
LAT_TO_M = 111131.745


def _latlon_to_geosph_vector(lat, lon, rad=False, basis=False):  # NEW
    """Convert lat/lon to 3D unit vector on geosphere.

    Replaces climada.util.coordinates.latlon_to_geosph_vector.
    """
    if not rad:
        lat, lon = np.radians(lat), np.radians(lon)

    cos_lat, sin_lat = np.cos(lat), np.sin(lat)
    cos_lon, sin_lon = np.cos(lon), np.sin(lon)

    vec = np.stack([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], axis=-1)

    if not basis:
        return vec

    e_lat = np.stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], axis=-1)
    e_lon = np.stack([-sin_lon, cos_lon, np.zeros_like(lon)], axis=-1)

    return vec, np.stack([e_lat, e_lon], axis=-2)


def great_circle_dist(
    ax: Any,
    ay: Any,
    bx: Any,
    by: Any,
    radius: float = EARTH_RADIUS,
    return_angles: bool = False,
) -> Any:
    """Calculate pair-wise Great Circle Distance (in km) between two sets of points.

    Note: ``ax``, ``ay``, ``bx``, ``by`` must be either:
        a. 1-D, with the same length, in which case the distances are element-wise and
           a 1-D array is returned, or
        b. Broadcastable to a common shape, in which case a distance matrix is returned.

    Parameters
    ----------
    ax, bx : array-like
        Longitudes of the two point sets
    ay, by : array-like
        Latitudes of the two point sets
    radius : float, optional
        Assumed radius of earth. Default is 6371.009 km
    return_angles : bool, optional
        If True, also return the angles from ``a`` points to ``b`` points

    Returns
    -------
    array-like
        The distance vector (if inputs are 1-D) or distance matrix (if inputs are
        multidimensional and broadcastable to the same shape).

    """
    lat1, lon1, lat2, lon2 = map(np.radians, (ay, ax, by, bx))

    # broadcast so it's easier to work with einstein summation below
    if all(map(lambda x: isinstance(x, xr.DataArray), (lat1, lon1, lat2, lon2))):
        lat1, lon1, lat2, lon2 = xr.broadcast(lat1, lon1, lat2, lon2)
    else:
        lat1, lon1, lat2, lon2 = np.broadcast_arrays(lat1, lon1, lat2, lon2)

    dlat = 0.5 * (lat2 - lat1)
    dlon = 0.5 * (lon2 - lon1)

    # haversine formula:
    hav = np.sin(dlat) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon) ** 2
    dist_km = 2 * np.arcsin(np.sqrt(hav)) * radius

    if not return_angles:
        return dist_km

    # calculate angles using Riemannian logarithm (accounting for fact that degree of
    # lon is not equivalent distance to degree of lat)
    # borrowed from climada.util.coordinates.dist_approx

    # Convert to numpy for angle computation to avoid DataArray/numpy mixing issues
    is_dataarray = isinstance(dist_km, xr.DataArray)
    _lat1 = np.asarray(lat1)
    _lon1 = np.asarray(lon1)
    _lat2 = np.asarray(lat2)
    _lon2 = np.asarray(lon2)
    _hav = np.asarray(hav)
    _dist_km = np.asarray(dist_km)

    vec1, vbasis = _latlon_to_geosph_vector(_lat1, _lon1, rad=True, basis=True)
    vec2 = _latlon_to_geosph_vector(_lat2, _lon2, rad=True)
    scal = 1 - 2 * _hav
    fact = _dist_km / np.fmax(np.spacing(1), np.sqrt(1 - scal**2))
    vtan = np.expand_dims(fact, -1) * (vec2 - np.expand_dims(scal, -1) * vec1)

    # count number of extra dims needed in summation
    array_indices = "abcdefgh"
    extra_dims = vec1.ndim - 1
    ix = array_indices[:extra_dims]

    # compute summation and calculate thetas
    vtan = np.einsum(f"{ix}k,{ix}lk->{ix}l", vtan, vbasis)
    thetas = np.arctan2(vtan[..., 0], vtan[..., 1])

    # cast back to dataarray if needed
    if is_dataarray:
        thetas = xr.DataArray(thetas, coords=dist_km.coords, dims=dist_km.dims)

    return dist_km, thetas


def get_dlon_dlat_from_heading_dist(lat_in, head, dist_m):
    lat1 = np.deg2rad(lat_in)
    head_r = np.deg2rad(head)

    # first convert dist_m to radians
    d = dist_m / (EARTH_RADIUS * 1e3)

    # next get output lat
    lat = np.arcsin(
        np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(head_r),
    )

    # finally calculate delta lon and delta lat
    dlon = np.arctan2(
        np.sin(head_r) * np.sin(d) * np.cos(lat1),
        np.cos(d) - np.sin(lat1) * np.sin(lat),
    )
    dlat = lat - lat1

    return np.rad2deg(dlon), np.rad2deg(dlat)


def coriolis(lat, omega):
    return 2 * omega * np.sin(np.deg2rad(lat))


def constrain_lons(arr, lon_mask):
    if lon_mask is False:
        return arr
    out = arr.copy()
    out = np.where((out > 180) & lon_mask, -360 + out, out)
    out = np.where((out <= -180) & lon_mask, 360 + out, out)
    return out


def grid_conversion_wrapper(func):
    @wraps(func)
    def inner(data, *args, cols=None, name=None, **kwargs):
        if cols is None and hasattr(data, "columns"):
            cols = data.columns
        if name is None and hasattr(data, "name"):
            name = data.name
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(
                func(data.values, *args, **kwargs),
                index=data.index,
                columns=cols,
            )
        elif isinstance(data, pd.Series):
            return pd.Series(
                func(data.values, *args, **kwargs),
                index=data.index,
                name=name,
            )
        else:
            return func(data, *args, **kwargs)

    return inner


@grid_conversion_wrapper
def grid_val_to_ix(
    vals: Any,
    cell_size: Union[int, Sequence],
    map_nans: Optional[Union[int, Sequence]] = None,
    lon_mask: Union[bool, Sequence, np.ndarray] = False,
) -> Any:
    """Converts grid cell lon/lat/elevation values to i/j/k values. The function is
    indifferent to order, of dimensions, but the order returned matches the order of the
    inputs, which in turn must match the order of ``cell_size``. The origin of the grid
    is the grid cell that has West, South, and bottom edges at (0,0,0) in (lon, lat,
    elev) space, and we map everything to (-180,180] longitude.

    Parameters
    ----------
    vals : array-like
        The values in lon, lat, or elevation-space. The dimensions of this array should
        be n_vals X n_dims (where dims is either 1, 2, or 3 depending on which of
        lat/lon/elev are in the array).
    cell_size : int or Sequence
        The size of a cells along the dimensions included in ``vals``. If int, applies
        to all columns of ``vals``. If Sequence, must be same length as the number of
        columns of ``vals``.
    map_nans : int or Sequence, optional
        If not None, map this value in the input array to ``np.nan`` in the output
        array. If int, applies to all columns of ``vals``. If Sequence, must be the same
        length as ``vals``, with each element applied to the corresponding column of
        ``vals``.
    lon_mask : bool or array-like, optional
        Specify an mask for values to constrain to (-180, 180] space. If value is a
        bool, apply mask to all (True) or none of (False) the input ``vals``. If value
        is array-like, must be broadcastable to the shape of ``vals`` and castable to
        bool.

    Returns
    -------
    out : array-like
        An integer dtype object of the same type as vals defining the index of each grid
        cell in ``vals``.

    Raises
    ------
    ValueError
        If `vals` contains null values but `map_nans` is None.

    """
    # handle nans
    nan_mask = np.isnan(vals)
    is_nans = nan_mask.sum()

    out = vals.copy()

    if is_nans:
        if map_nans is None:
            raise ValueError(
                "NaNs not allowed in `vals`, unless `map_nans` is specified.",
            )
        else:
            # convert to 0s to avoid warning in later type conversion
            out = np.where(nan_mask, 0, out)

    out = constrain_lons(out, lon_mask)

    # convert to index
    out = np.floor(out / cell_size).astype(np.int32)

    # fix nans to our chosen no data int value
    if is_nans:
        out = np.where(nan_mask, map_nans, out)  # type: ignore

    return out


@grid_conversion_wrapper
def grid_ix_to_val(
    vals: Any,
    cell_size: Union[int, Sequence],
    map_nans: Optional[Union[int, Sequence]] = None,
    lon_mask: Union[bool, Sequence, np.ndarray] = False,
) -> Any:
    """Converts grid cell i/j/k values to lon/lat/elevation values. The function is
    indifferent to order, of dimensions, but the order returned matches the order of the
    inputs, which in turn must match the order of ``cell_size``. The origin of the grid
    is the grid cell that has West, South, and bottom edges at (0,0,0) in (lon, lat,
    elev) space, and we map everything to (-180,180] longitude.

    Parameters
    ----------
    vals : array-like
        The values in i, j, or k-space. The dimensions of this array should be
        n_vals X n_dims (where dims is either 1, 2, or 3 depending on which of i/j/k are
        in the array).
    cell_size : Sequence
        The size of a cells along the dimensions included in ``vals``. If int, applies
        to all columns of ``vals``. If Sequence, must be same length as the number of
        columns of ``vals``.
    map_nans : int or Sequence, optional
        If not None, map this value in the input array to ``np.nan`` in the output
        array. If int, applies to all columns of ``vals``. If Sequence, must be the same
        length as ``vals``, with each element applied to the corresponding column of
        ``vals``.
    lon_mask : bool or array-like, optional
        Specify an mask for values to constrain to (-180, 180] space. If value is a
        bool, apply mask to all (True) or none of (False) the input ``vals``. If value
        is array-like, must be broadcastable to the shape of ``vals`` and castable to
        bool.

    Returns
    -------
    out : array-like
        A float dtype object of the same type as vals defining the lat/lon/elev of each
        grid cell in ``vals``.

    Raises
    ------
    AssertionError
        If `vals` is not an integer object

    """
    assert np.issubdtype(vals.dtype, np.integer)

    out = cell_size * (vals + 0.5)
    out = constrain_lons(out, lon_mask)

    # apply nans
    if map_nans is not None:
        valid = vals != map_nans
        out = np.where(valid, out, np.nan)

    return out


@grid_conversion_wrapper
def bin_grid_vals(
    vals: Any,
    cell_size: Union[int, Sequence],
    map_nans: Optional[Union[int, Sequence]] = None,
    lon_mask: Union[bool, Sequence] = False,
) -> Any:
    """Bins/snaps grid cell lon/lat/elev values to a specified grid size, simply calling
    ``grid_val_to_ix`` followed by ``grid_ix_to_val``.

    Parameters
    ----------
    vals : array-like
        The input values in i, j, or k-space. The dimensions of this array should be
        n_vals X n_dims (where dims is either 1, 2, or 3 depending on which of i/j/k are
        in the array).
    cell_size : Sequence
        The size of cells along the dimensions included in ``vals``. If int,
        applies to all columns of ``vals``. If Sequence, must be same length as the
        number of columns of ``vals``.
    map_nans : int or Sequence, optional
        If not None, map this value in the input array to the same value in the output
        array, rather than treating it as an index value. If Sequence, must be the same
        length as ``vals``, with each element applied to the corresponding column of
        ``vals``.
    lon_mask : bool or array-like, optional
        Specify an mask for values to constrain to (-180, 180] space. If value is a
        bool, apply mask to all (True) or none of (False) the input ``vals``. If value
        is array-like, must be broadcastable to the shape of ``vals`` and castable to
        bool.

    Returns
    -------
    out : array-like
        A float dtype object of the same type as vals defining the lat/lon/elev of each
        grid cell in ``vals``.

    Raises
    ------
    AssertionError
        If `vals` is not an integer object

    """
    out = grid_val_to_ix(
        vals,
        cell_size=cell_size,
        map_nans=map_nans,
        lon_mask=lon_mask,
    )
    out = grid_ix_to_val(out, cell_size=cell_size, map_nans=map_nans, lon_mask=lon_mask)

    return out
