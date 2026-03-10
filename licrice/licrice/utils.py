import numpy as np
import xarray as xr

from licrice import spatial, utilities
from . import vortex_funcs


def get_output_grid(lon_bounds, lat_bounds, res_spatial):
    def get_array(lb, ub):
        return np.arange(lb + res_spatial / 2, ub, res_spatial)

    if lon_bounds[0] > 0 and lon_bounds[1] < 0:
        x = np.concatenate(
            [get_array(lon_bounds[0], 180), get_array(-180, lon_bounds[1])],
            axis=None,
        )
    else:
        x = get_array(*lon_bounds)

    y = get_array(*lat_bounds)

    return xr.Dataset(
        coords={
            "y_ix": spatial.grid_val_to_ix(y, res_spatial),
            "x_ix": spatial.grid_val_to_ix(x, res_spatial, lon_mask=True),
        },
    ).rename()


def get_wind_field(
    ds,
    ds_dist_angle,
    vortex_func=None,
    vortex_func_kwargs={},
    ramp_width=100e3,
    sampling_time_adj=1,
    scale_translational_velocity=True,
    cap_translational_velocity=15.4,
    wind_rat_0=None,
):
    """Combine circular wind speed at each pixel with storm's translational speed to
    calculate total wind speed at the pixel level. Scaling vectors/scalars occurs to
    both limit storm scope and taper winds towards the edge and center of storm.

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
        1-D Dataset indexed by `time` which contains variables for the
        the circular velocity, the x and y translational velocities, and the
        ROCI.
    ds_dist_angle : :class:`xarray.Dataset`
        3-D Dataset indexed by `lat`, `lon`, `time`. Containing the
        distance (in m) of each cell to the center cell (time dimension necessary b/c
        distance  corresonding to 1 degree of latitude is different depending
        on what latitude a storm is at)
    vortex_func : function
        Function used to estimate circular wind field
    vortex_func_kwargs : dicitonary
        Dictionary of keyword arguments to pass to `vortex_func`
    ramp_width: float, optional
        Storm field ramping width, in meters - Represents crudely the ramping
        radial area. Default 100e3. If None, no total velocity scaling will occur.
    sampling_time_adj : float, optional
        Wind speed conversion performed to account for sampling time. Emanuel wind
        speeds are reported as 1 min averages. IBTrACS cleaning code converts wind
        speeds to 1 min averages To convert to 10 min averages (what is done in
        GEOCLAW), set this to 0.88. Default 1.
    scale_translational_velocity : boolean, optional
        Taper x and y translational velocity to zero as the circular wind
        tapers to zero toward the eye of the storm and at long distances
        from the storm. Default True.
    cap_translational_velocity : float, optional
        If not None, cap the added translational speed at the specified threshold
        (in m/s). Larger translational velocities are assumed to be non-physical for
        TCs. Default is 30kts = 15.4 m/s.
    wind_rat_0 : :class:`xarray.DataArray` or None
        An array of the same size as the Lagrangian model domain, indexed by ``lat`` and
        ``lon``, which defines the initial scaling ratio with which translational wind
        speeds are adjusted downward near the eye and the edges of the storm. This is
        only used in the case where all timesteps in this chunk have a 0 circular wind
        speed, because otherwise we would have no wind_rat to forward-fill wind_rat. If
        None, do not use this to forward fill.

    Returns
    -------
    :class:`xarray.DataArray`
        2-D DataArray indexed by `lat`, `lon`, `time`. Containing the
        average wind speed for each pixel.

    """
    if ds_dist_angle is None:
        return None, None

    # v_circular should never be negative
    assert ((ds.v_circular >= 0) | ds.v_circular.isnull()).all()

    # calculate circular velocity field using specified vortex function
    vortex_func = getattr(vortex_funcs, vortex_func)
    wind_circ = vortex_func(ds, ds_dist_angle, **vortex_func_kwargs)

    # Determine translation speed that should be added to final storm wind speed.
    # This is tapered to zero as the storm wind tapers to zero toward the eye of
    # the storm and at long distances from the storm. If there is no circular wind,
    # translational speed is capped at max observed wind speed.

    maxes = wind_circ.max(dim=["lat", "lon"])
    if scale_translational_velocity:
        # make wind_rat nan whenever maxes is 0
        wind_rat = xr.full_like(wind_circ, np.nan).where(maxes == 0, wind_circ / maxes)

        # replace initial wind_rat if provided
        if wind_rat_0 is not None and wind_rat.isel(time=0).isnull().all().item():
            # need to reindex for when grid size has changed in between time chunks
            wind_rat[{"time": 0}] = wind_rat_0.reindex_like(
                wind_rat.isel(time=0),
            ).fillna(0)

        # fill nans forward, then backward (needed in case the beginning of the storm
        # has 0 circular wind speed)
        wind_rat = wind_rat.ffill("time").bfill("time")

        # if storm never appears to have circular winds, then just assume no wind from
        # this storm
        if wind_rat.isnull().all():
            wind_rat[:] = 0
    else:
        wind_rat = xr.ones_like(wind_circ)

    # calculate translational velocity scaling factor.

    # first, ensure sqrt(u_trans_adder + v_trans_adder) + v_circular <= v_total
    trans_spd = np.sqrt(ds.v_trans_x**2 + ds.v_trans_y**2)
    v_total_cap_factor = (ds.v_total - ds.v_circular) / trans_spd

    # third, ensure that v_trans stays below specified cap
    if cap_translational_velocity is not None:
        v_trans_cap_factor = cap_translational_velocity / trans_spd
    else:
        v_trans_cap_factor = xr.ones_like(trans_spd)

    trans_scaling_factor = xr.concat(
        [v_total_cap_factor, v_trans_cap_factor],
        dim="factor",
    ).min(dim="factor")

    # don't want to INCREASE translational velocity
    trans_scaling_factor = np.minimum(1, trans_scaling_factor)

    # get vector field of translational speed adders
    u_trans_adder = wind_rat * ds.v_trans_x * trans_scaling_factor
    v_trans_adder = wind_rat * ds.v_trans_y * trans_scaling_factor

    # convert wind averaging period
    wind_circ = wind_circ * sampling_time_adj

    # account for different rotations in different hemispheres
    # assumes a storm can't cross the equator -- seems legit
    hemisphere_adjuster = xr.full_like(ds.storm_lat, 1).where(ds.storm_lat > 0, -1)

    # Velocity components of storm (assumes perfect vortex shape), adding in
    # translational velocity
    # dx/dtheta + dx/dt
    u = hemisphere_adjuster * (-wind_circ) * np.sin(ds_dist_angle.theta) + u_trans_adder
    # dy/dtheta + dy/dt
    v = hemisphere_adjuster * wind_circ * np.cos(ds_dist_angle.theta) + v_trans_adder

    # calculate total speed
    total_speed = np.sqrt(u**2 + v**2)

    # Apply distance ramp down(up) to fields to limit scope
    total_speed = apply_tanh_ramp(
        total_speed,
        dist=ds_dist_angle.dist,
        storm_radius=ds.radius,
        ramp_width=ramp_width,
    )

    # make sure we're never predicting a wind speed greater than Vmax
    max_spd = total_speed.max(dim=["lat", "lon"])
    assert (
        ((max_spd / (ds.v_total * sampling_time_adj)) <= 1)
        | (ds.v_total.isnull() & max_spd.isnull())
    ).all(), (
        "Wind speed estimated to be greater than observed Vmax for storm "
        f"{ds.storm.item()}, which should not occur."
    )

    return total_speed, wind_rat.isel(time=-1).squeeze()


def apply_tanh_ramp(*args, dist=None, storm_radius=None, ramp_width=None):
    if ramp_width:
        ramp = 0.5 * (1 - np.tanh((dist - storm_radius) / ramp_width))
    else:
        ramp = 1

    out = [ramp * a for a in args]
    if len(out) == 1:
        out = out[0]
    return out


def lagrange_to_euler(speeds, ds, output_grid, p):
    """Construct PDDI and MAXS fields for each storm from total wind speed grids for
    each storm at each time step.

    Parameters
    ----------
    speeds : :py:class:`xarray.Dataset`
        3-D Dataset indexed by `lat`, `lon`, `time`. Containing the
        10 min average wind speed for each pixel, for each time, for each storm.
    ds : :py:class:`xarray.Dataset`
        1-D Dataset indexed by `time` which contains the eye center
        lat long coordinates
    output_grid : :py:class:`xarray.Dataset`
        2-D Dataset indexed by `y_ix` and `x_ix`. This dataset's coordinates are
        the lat and lon indices
    p : dictionary
        licrice parameters loaded as a dictionary

    Returns
    -------
    pddi_arr :class:`numpy.ndarray`
        2-D Array lat x lon. pddi field for a given storm length.
    maxs_arr :class:`numpy.ndarray`
        2-D Array lat x lon. maxs field for a given storm length.

    """
    ####################################################################################
    # Create variables which place grids on earth
    # * These variables will be in x/y index space rather than lat/lon space. in other
    #   words each pixel on the earth gird is referenced by an integer index.
    # * Place storm eyes on grid through a binning routine
    ####################################################################################

    y_ix_in = spatial.grid_val_to_ix(
        (
            speeds.lat + utilities.bin_data(ds.storm_lat, p["grid"]["res_spatial_deg"])
        ).values,
        p["grid"]["res_spatial_deg"],
    )

    x_ix_in = spatial.grid_val_to_ix(
        (
            speeds.lon + utilities.bin_data(ds.storm_lon, p["grid"]["res_spatial_deg"])
        ).values,
        p["grid"]["res_spatial_deg"],
        lon_mask=True,
    )

    x_ix_out = output_grid.x_ix.values
    y_ix_out = output_grid.y_ix.values

    ####################################################################################
    # Find x and y values that overlap between the speeds grids and the output_grid
    # create dictionary mapping x and y values that overlap to array indices on the
    # output grid
    ####################################################################################

    x_indices_out = np.nonzero(np.isin(x_ix_out, x_ix_in))[0]
    x_values_overlap = x_ix_out[x_indices_out]
    y_indices_out = np.nonzero(np.isin(y_ix_out, y_ix_in))[0]
    y_values_overlap = y_ix_out[y_indices_out]

    # mapping between x/y overlapping values and output_grid array indices
    x_ix_to_ix_out = dict(zip(x_values_overlap, x_indices_out))
    y_ix_to_ix_out = dict(zip(y_values_overlap, y_indices_out))

    ####################################################################################
    # Find array indices from speeds array for pixels that fall on output grid
    ####################################################################################

    x_indices_in = np.stack(np.nonzero(np.isin(x_ix_in, x_values_overlap)))
    y_indices_in = np.stack(np.nonzero(np.isin(y_ix_in, y_values_overlap)))

    ####################################################################################
    # place pixels from uniform grid for each storm x time on
    # output grid and then calculate wind and pddi fields for each storm
    ####################################################################################

    speeds_arr = speeds.fillna(0).values

    # calculate amount of time storm spends in each pixel for pddi calc
    timedelta_lower = (
        (ds.datetime.diff(dim="time", label="lower") / np.timedelta64(1, "s"))
        .reindex_like(ds.time)
        .fillna(0)
    )
    timedelta_upper = (
        (ds.datetime.diff(dim="time", label="upper") / np.timedelta64(1, "s"))
        .reindex_like(ds.time)
        .fillna(0)
    )

    timedelta = (0.5 * timedelta_lower + 0.5 * timedelta_upper).values

    # create output grid arrays for populating with results

    # array to keep track of the maxs calculation for each storm
    maxs_arr = np.zeros(
        (output_grid.y_ix.shape[0], output_grid.x_ix.shape[0]),
        dtype=np.float32,
    )

    # array to keep track of the pddi calculation for each storm
    pddi_arr = maxs_arr.copy()

    # if no overlap between storm and grid skip storm
    if not (x_indices_in.shape[1] == 0 or x_indices_in.shape[1] == 0):
        # iterate through timestep indices with both x and y values in output grid
        for tt in np.intersect1d(x_indices_in[1, :], y_indices_in[1, :]):
            x_indices_in_tt = x_indices_in[:, (x_indices_in[1, :] == tt)][0, :]
            y_indices_in_tt = y_indices_in[:, (y_indices_in[1, :] == tt)][0, :]

            # retrieve array indices for array to be placed
            y_st_in = y_indices_in_tt.min()
            y_end_in = y_indices_in_tt.max()
            x_st_in = x_indices_in_tt.min()
            x_end_in = x_indices_in_tt.max()

            # retrieve array indices for location of array to be placed on output array
            x_st_out, x_end_out = (
                x_ix_to_ix_out[val] for val in x_ix_in[[x_st_in, x_end_in], tt]
            )
            y_st_out, y_end_out = (
                y_ix_to_ix_out[val] for val in y_ix_in[[y_st_in, y_end_in], tt]
            )

            # iteratively calculate summary statistics to minimize memory footprint

            # (re)calculate maxs given this new time step
            maxs_arr[y_st_out : (y_end_out + 1), x_st_out : (x_end_out + 1)] = (
                np.maximum(
                    speeds_arr[y_st_in : (y_end_in + 1), x_st_in : (x_end_in + 1), tt],
                    maxs_arr[y_st_out : (y_end_out + 1), x_st_out : (x_end_out + 1)],
                )
            )

            # (re)calculate pddi given this new time step

            pddi_arr[y_st_out : (y_end_out + 1), x_st_out : (x_end_out + 1)] = pddi_arr[
                y_st_out : (y_end_out + 1),
                x_st_out : (x_end_out + 1),
            ] + (
                speeds_arr[y_st_in : (y_end_in + 1), x_st_in : (x_end_in + 1), tt] ** 3
                * timedelta[tt]
            )

    return pddi_arr, maxs_arr
