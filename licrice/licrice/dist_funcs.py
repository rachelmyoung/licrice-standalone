"""Distance calculation functions for LICRICE.

Extracted from pyTC/licrice/dist_funcs.py.
"""

from math import ceil

import numpy as np
import xarray as xr

from licrice import spatial  # NEW

from .utils import apply_tanh_ramp
from .vortex_funcs import _get_modified_rankine_vortex, _get_rankine_alpha


def get_vortex_angles_dists_great_circle(
    ds_storm,
    res_spatial_deg=0.1,
    ramp_width=100e3,
    v_min_ms=None,
    vortex_func=None,
):
    """Parameters
    ----------
    buffer_lon_rat : float
        Amount to buffer in the x direction to make sure that a circular storm on the
        spherical globe does not extend futher in the longitudinal direction when the
        angle is not 0.

    """
    rad_model_m = ds_storm.radius.copy()

    if v_min_ms is not None:
        # if storm isn't powerful enough then just skip it
        if ds_storm.v_circular.max().item() < v_min_ms:
            return None

        rad_max = ds_storm.radius.max().item()
        rmw_max = ds_storm.rmw.max().item()
        rads_test = xr.DataArray(
            np.arange(rmw_max, rad_max, 10e3),
            dims=["radius"],
            coords={"radius": np.arange(rmw_max, rad_max, 10e3)},
        )
        if vortex_func == "get_modified_rankine_vortex":
            alpha = _get_rankine_alpha(ds_storm)
            v = _get_modified_rankine_vortex(
                ds_storm.v_circular,
                ds_storm.rmw,
                alpha,
                rads_test,
            )
            v = apply_tanh_ramp(
                v,
                dist=v.radius,
                storm_radius=ds_storm.radius,
                ramp_width=ramp_width,
            )

            # find last radius value that has sufficient speed to qualify
            invalid = v < v_min_ms
            ix_max = invalid.argmax(dim="radius") - 1

            # if all radii count, make it equal to the last radius
            ix_max = ix_max.where(invalid.any(dim="radius"), len(v.radius) - 1)

            # if no radii count, make it equal to 0
            rad_m_max = v.radius.isel(ix_max).where(ix_max >= 0, 0)

            # take minimum of this value and storm radius
            rad_model_m = np.minimum(rad_model_m, rad_m_max)

    # for longitude, it's possible that the max dlon does not occur when traveling
    # directly E-W, so we text max lon along all 360 degrees
    lons, lats = spatial.get_dlon_dlat_from_heading_dist(
        ds_storm.storm_lat,
        xr.DataArray(
            np.arange(0, 360),
            dims=("heading",),
            coords={"heading": np.arange(0, 360)},
        ),
        rad_model_m,
    )
    rad_storm_deg_lon, rad_storm_deg_lat = list(
        map(
            lambda x: (
                ceil(np.abs(x).max().item() / res_spatial_deg) * res_spatial_deg
            ),
            (lons, lats),
        ),
    )

    # we got problems if we're jumping over the poles
    assert (np.abs(rad_storm_deg_lat + ds_storm.storm_lat) < 90).all().item()

    # calculate distance from storm_lat and storm_lon to the centroid of the grid cell
    # they fall into. NOT using the `lon_ix` flag b/c we want to allow continuous
    # longitudes within a storm
    storm_lat_adder = (
        spatial.bin_grid_vals(ds_storm.storm_lat, cell_size=res_spatial_deg)
        - ds_storm.storm_lat
    )
    storm_lon_adder = (
        spatial.bin_grid_vals(ds_storm.storm_lon, cell_size=res_spatial_deg)
        - ds_storm.storm_lon
    )

    # calculate a grid centered at the grid cell centroid of storm_lon and storm_lat
    # (later will adjust by the adders calculated above)
    storm_grid_xx = np.arange(
        -rad_storm_deg_lon,
        rad_storm_deg_lon + res_spatial_deg / 2,
        res_spatial_deg,
    )
    storm_grid_yy = np.arange(
        -rad_storm_deg_lat,
        rad_storm_deg_lat + res_spatial_deg / 2,
        res_spatial_deg,
    )
    xx, yy = np.meshgrid(storm_grid_xx, storm_grid_yy)

    dist_km, thetas = spatial.great_circle_dist(
        ds_storm.storm_lon.astype(np.float32).values,
        ds_storm.storm_lat.astype(np.float32).values,
        (xx[..., None] + storm_lon_adder.values + ds_storm.storm_lon.values).astype(
            np.float32,
        ),
        (yy[..., None] + storm_lat_adder.values + ds_storm.storm_lat.values).astype(
            np.float32,
        ),
        return_angles=True,
    )

    out = xr.Dataset(
        {
            "dist": (["lat", "lon", "time"], dist_km * 1e3),
            "theta": (["lat", "lon", "time"], thetas),
        },
        coords={"lat": storm_grid_yy, "lon": storm_grid_xx, "time": ds_storm.time},
    )

    return out
