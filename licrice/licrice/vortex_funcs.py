"""Vortex profile functions for LICRICE.

Extracted from pyTC/licrice/vortex_funcs.py.
"""

import numpy as np
import xarray as xr

from licrice.spatial import coriolis  # NEW


def inv_modified_rankine(vmax, rmw, alpha, speed):
    """Return radius at which circular speed ``speed`` would be reached."""
    return rmw * ((speed / vmax) ** (-1 / alpha))


def _get_modified_rankine_vortex(vmax, rmw, alpha, dist):
    return (
        (vmax * dist / rmw).where((dist / rmw) < 1).fillna(vmax * (rmw / dist) ** alpha)
    )


def _get_rankine_alpha(ds):
    """Construct alpha (TC Notes):

    v_circular < 30 m/s -> alpha = 0.31 v_circular >= 30 m/s and <= 50 m/s -> alpha =
    0.35 v_circular > 50 m/s -> alpha = 0.48
    """
    alpha = xr.DataArray(
        np.full(ds.v_circular.shape, 0.35),
        dims=ds.v_circular.dims,
        coords=ds.v_circular.coords,
    )
    alpha = alpha.where(ds.v_circular >= 30, 0.31)
    alpha = alpha.where(ds.v_circular <= 50, 0.48)
    alpha = alpha.where(ds.v_circular.notnull())
    return alpha


def get_modified_rankine_vortex(
    ds,
    ds_dist_angle,
):
    """Estimate circular wind speed on field using a modified rankine vortex model.

    Parameters
    ----------
    ds: :py:class:`xarray.Dataset`
        1-D Dataset indexed by `time` which contains variables for the RMW and circular
        velocity.
    ds_dist_angle : :py:class:`xarray.Dataset`
        4-D Dataset indexed by 'storm', lat', 'lon', 'time'. Containing the
        distance (in m) of each cell to the center cell (time dimension necessary b/c
        distance  corresonding to 1 degree of latitude is different depending
        on what latitude a storm is at)

    Returns
    -------
    wind_circ : :py:class:`xarray.DataArray`
        3-D DataArray indexed by  `lat`, `lon`, `time`. Containing the
        circular wind speed of each pixel.

    """
    ds["alpha"] = _get_rankine_alpha(ds)

    # construct v_circular for each pixel
    return _get_modified_rankine_vortex(
        ds.v_circular,
        ds.rmw,
        ds.alpha,
        ds_dist_angle.dist,
    )


def get_holland_1980_vortex(
    ds,
    ds_dist_angle,
    Pa=101.3e3,
    atmos_boundary_layer=0.9,
    rho_air=1.15,
    omega=2 * np.pi / 86164.2,
):
    """Estimate circular wind speed on field using the Holland 1980 model.

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
        1-D Dataset indexed by ``time`` which contains variables for the RMW,
        the max velocity (either gradient or total velocity, or both), and,
        optionally, the radius of vanishing winds.
    ds_dist_angle : :py:class:`xarray.Dataset`
        3-D Dataset indexed by `lat`, `lon`, `time`. Containing the
        distance of each cell to the center cell (time dimension necessary b/c
        distance  corresonding to 1 degree of latitude is different depending
        on what latitude a storm is at)
    Pa : float, optional
        Ambient air pressure, in Pa. Default 101.3e3.
    atmos_boundary_layer : float, optional
        Atmospheric boundary layer, input variable in ADCIRC but always is set
        to 0.9 (default). Used to scale 10m wind speeds to top of boundary
        layer.
    rho_air : float, optional
        Density of air, in kg m-3. Default 1.15.
    sampling_time_adj : float, optional
        Wind speed conversion performed to account for sampling time. IBTrACS
        and Emanuel wind speeds are reported as 1 min averages. To convert to
        10 min averages (what is done in GEOCLAW), set this to 0.88. Default 1.
    omega : float, optional
        Rotational velocity of earth. Default :math:`2 * pi / 86164.2`

    Returns
    -------
    :py:class:`xarray.DataArray`
        3-D DataArray indexed by `lat`, `lon`, `time`. Containing the
        circular wind speed of each pixel.

    """
    # Convert wind speed from 10 meter altitude (which is what the
    # NHC forecast and IBTRACS contains) to wind speed at the top of the
    # atmospheric boundary layer (which is what the Holland curve fit
    # requires).
    mod_mws = ds.v_circular / atmos_boundary_layer

    # Calculate central pressure difference
    dp = Pa - ds.pres

    # calculate coriolis force
    f = coriolis(ds.storm_lat, omega)

    # Limit central pressure deficit due to bad ambient pressure,
    # really should have better ambient pressure...
    dp = dp.where(dp >= 100, 100)

    # Calculate Holland parameters and limit the result
    B = rho_air * np.exp(1) * (mod_mws**2) / dp
    B = B.where(B >= 1, 1)
    B = B.where(B <= 2.5, 2.5)

    # calculate gradient winds
    wind_gradient = (
        np.sqrt(
            (ds.rmw / ds_dist_angle.dist) ** B
            * np.exp(1 - (ds.rmw / ds_dist_angle.dist) ** B)
            * mod_mws**2
            + (ds_dist_angle.dist * f) ** 2 / 4,
        )
        - ds_dist_angle.dist * f / 2
    )

    # Convert wind velocity from top of atmospheric boundary layer (which is what
    # the Holland curve fit produces) to wind velocity at 10 m above the earth's
    # surface.
    return wind_gradient * atmos_boundary_layer
