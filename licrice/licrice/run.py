"""LICRICE run functions.

Extracted from pyTC/licrice/run.py. rhg_compute_tools.gcs dependency removed.
"""

import shutil  # NEW

import dask.config
import numpy as np
import xarray as xr

from licrice import spatial  # NEW
from licrice.utilities import check_finished_zarr_workflow  # NEW
from . import dist_funcs
from . import utils as lutils
from .preprocess import init_output_zarr, prep_tracks


def _construct_grid(ds, dist_func=None, **kwargs):
    """Construct grid dataset indexed by storm, time, lat and long. The dataset will
    include two data arrays: dist (pixel distance from grid center) and
    theta (pixel angle).

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
        1-D Dataset indexed by `time` which contains two variables:
        `storm_lat` and `storm_lon`, the storm eye lat and long coordinates.
    dist_func : function
        Name of function used to calculate the distance of each pixel from the grid
        center. Must be a function within ``licrice.licrice.dist_funcs``

    Returns
    -------
    :py:class:`xarray.Dataset`
        3-D Dataset indexed by  `lat`, `lon`, `time`. Containing `dist`
        and `theta`. Where `dist` holds the pixel distance from the grid center
        for each storm and time in the dataset. `theta` holds the angle between
        each pixel the grid center and the x-axis.

    """
    assert dist_func is not None

    dist_func = getattr(dist_funcs, dist_func)

    # construct grid dataset
    ds_dist_angle = dist_func(ds, **kwargs)

    if ds_dist_angle is not None:
        # replace all distances with nans if outside desired storm radius
        out = ds_dist_angle.where(ds_dist_angle.dist <= ds.radius)
        return out.dropna("lon", how="all").dropna("lat", how="all")


# composition function to ensure ds_dist_angle is deleted from memory
# as soon as possible
def get_speeds(ds_storm, p, wind_rat_0):
    ds_dist_angle = _construct_grid(
        ds_storm,
        vortex_func=p["wind"]["vortex_func"],
        ramp_width=p["wind"]["ramp_width"],
        **p["grid"],
    )
    return lutils.get_wind_field(
        ds_storm,
        ds_dist_angle,
        wind_rat_0=wind_rat_0,
        **p["wind"],
    )


def run_licrice_on_track(
    ds,
    xlim,
    ylim,
    params,
    return_dataset=True,
    out_grid=None,
):
    """Calculate maxs and pddi for a given storm or set of storms using licrice.

    Parameters
    ----------
    ds : :class:`xarray.Dataset`
        preprocessed licrice input for a single storm. Dataset indexed by ``time`` and
        by a length-one dimension of ``storm``.
    xlim : list
        list of length 2. Upper and lower x limit for bounding box.
    ylim : list
        list of length 2. Upper and lower y limit for bounding box.
    params : dict
        LICRICE model params, i.e. from `pyTC.settings.Settings().PARAMS["licrice"]`

    """
    ds_storm = ds.isel(time=ds.datetime.notnull())

    if out_grid is None:
        out_grid = lutils.get_output_grid(xlim, ylim, params["grid"]["res_spatial_deg"])

    # array to keep track of the maxs calculation for each storm
    maxs_arr = np.zeros(
        (out_grid.y_ix.shape[0], out_grid.x_ix.shape[0]),
        dtype=np.float32,
    )

    # array to keep track of the pddi calculation for each storm
    pddi_arr = maxs_arr.copy()

    # control memory when calculating vortex for each time step by keeping lat*lon*time
    # product equal to a constant (1e6). This number trades off more vectorization with
    # having to execute more computations
    ix_breaks = [0]
    n_times = ds_storm.storm_lat.notnull().sum()
    while ix_breaks[-1] < n_times:
        this_break = ix_breaks[-1]
        this_ds = ds_storm.isel(time=slice(this_break, None))
        n_lon = (
            2
            * this_ds.radius
            / spatial.LAT_TO_M
            / np.cos(np.deg2rad(this_ds.storm_lat))
            // params["grid"]["res_spatial_deg"]
        )
        n_lat = (
            2 * this_ds.radius / spatial.LAT_TO_M // params["grid"]["res_spatial_deg"]
        )
        grid_size = (n_lat * n_lon).to_series().cummax()
        cum_grid_size = grid_size * np.arange(1, len(grid_size) + 1)
        n_grp = max(((cum_grid_size // 1e6) == 0).sum(), 1)
        ix_breaks.append(this_break + n_grp)

    wind_rat_0 = None

    for ii in range(len(ix_breaks) - 1):
        ds_storm_chunk = ds_storm.isel(time=slice(ix_breaks[ii], ix_breaks[ii + 1]))
        speeds, wind_rat_0 = get_speeds(ds_storm_chunk, params, wind_rat_0)
        if speeds is None:
            continue
        pddi_arr_chunk, maxs_arr_chunk = lutils.lagrange_to_euler(
            speeds,
            ds_storm_chunk,
            out_grid,
            params,
        )
        maxs_arr = np.maximum(maxs_arr, maxs_arr_chunk)
        pddi_arr += pddi_arr_chunk

    if not return_dataset:
        return pddi_arr, maxs_arr

    out = xr.Dataset(
        data_vars={
            "pddi": (("y_ix", "x_ix"), pddi_arr),
            "maxs": (("y_ix", "x_ix"), maxs_arr),
            "start_date": ds.datetime.isel(time=0, drop=True),
        },
        coords={
            "y_ix": out_grid.y_ix,
            "x_ix": out_grid.x_ix,
        },
    )

    return out


def run_licrice_on_chunk(
    ds_in,
    params,
    region_start=None,
    xlim=None,
    ylim=None,
    outpath=None,
):
    out_grid = lutils.get_output_grid(xlim, ylim, params["grid"]["res_spatial_deg"])
    out = [
        run_licrice_on_track(
            ds_in.isel(storm=i),
            xlim,
            ylim,
            params,
            out_grid=out_grid,
            return_dataset=False,
        )
        for i in range(ds_in.storm.size)
    ]
    pddi_arr = np.stack([o[0] for o in out])
    maxs_arr = np.stack([o[1] for o in out])
    out = xr.Dataset(
        {
            "pddi": (("storm", "y_ix", "x_ix"), pddi_arr),
            "maxs": (("storm", "y_ix", "x_ix"), maxs_arr),
        },
    )

    # if we're not modifying existing zarr, can add these extra vars/coords
    if region_start is None:
        out.coords["storm"] = ds_in.storm
        out.coords["y_ix"] = out_grid.y_ix
        out.coords["x_ix"] = out_grid.x_ix
        out["start_date"] = (("storm",), ds_in.datetime.isel(time=0, drop=True))
        region = None
    else:
        region = {"storm": slice(region_start, region_start + len(out.storm))}

    # if saving, we want to return None so that workers don't keep holding onto result
    if outpath is not None:
        out.to_zarr(
            outpath,
            region=region,
        )
        return None
    # otherwise, we want to actually return the result
    else:
        return out


def run_licrice_on_trackset(
    ds_path,
    valid_storms,
    start_dates,
    params,
    xlim=None,
    ylim=None,
    outpath=None,
    tmppath=None,
    checkfile_path=None,
    attr_dict={},
    storm_chunksize=25,
    trackset_type="ibtracs",
    client=None,
    overwrite=True,
    **client_kwargs,
):
    if client is not None:
        _submitter = client.submit
        _mapper = client.map
    else:

        def _submitter(x, *args, **kwargs):
            return x(*args, **kwargs)

        def _mapper(fun, arglist, **kwargs):
            return [fun(a, **kwargs) for a in arglist]

    def _check_final(overwrite, checkfile_path, outpath, varname_to_check):
        if (not overwrite) and checkfile_path.is_file():
            with checkfile_path.open("r") as f:
                n_tracks_check = int(f.read())
            if n_tracks_check == 0:
                return 0

            if outpath.is_dir():
                n_tracks_file = xr.open_zarr(str(outpath), chunks=None).storm.size

                if n_tracks_file == n_tracks_check:
                    with dask.config.set(scheduler="single-threaded"):
                        if xr.open_zarr(str(outpath))[varname_to_check].notnull().all():
                            return n_tracks_check
        return -1

    n_tracks_complete = _submitter(
        _check_final,
        overwrite,
        checkfile_path,
        outpath,
        [i for i in ["maxs", "pddi"] if params[i]][0],
    )

    n_tracks_complete = _submitter(
        init_output_zarr,
        valid_storms,
        start_dates,
        xlim,
        ylim,
        params["grid"]["res_spatial_deg"],
        attr_dict,
        storm_chunksize,
        tmppath,
        n_tracks_complete=n_tracks_complete,
        overwrite=overwrite,
        **client_kwargs,
    )

    # map the licrice prep/run/save steps
    def _prep_and_run_licrice_chunk(region, n_tracks_complete=None):
        if n_tracks_complete is not None and n_tracks_complete >= 0:
            return n_tracks_complete
        valid_storm_chunk = valid_storms.values[region]
        ds = prep_tracks(
            str(ds_path),
            xlim,
            ylim,
            params,
            valid_storms=valid_storm_chunk,
            trackset_type=trackset_type,
        )

        # account for storms that didn't have any valid times
        ds = ds.reindex(storm=valid_storm_chunk)

        assert len(ds.storm) == (region.stop - region.start), ds.storm

        if check_finished_zarr_workflow(
            finalstore=str(tmppath),
            varname=[i for i in ["maxs", "pddi"] if params[i]][0],
            final_selector={"storm": ds.storm.values},
            check_final=True,
            check_temp=False,
            how="all",
        ):
            return None

        return run_licrice_on_chunk(
            ds,
            params,
            region_start=region.start,
            xlim=xlim,
            ylim=ylim,
            outpath=str(tmppath),
        )

    regions = [
        slice(c, min(len(valid_storms), c + storm_chunksize))
        for c in np.arange(0, len(valid_storms), storm_chunksize)
    ]

    tmpfile_futs = _mapper(
        _prep_and_run_licrice_chunk,
        regions,
        n_tracks_complete=n_tracks_complete,
        **client_kwargs,
    )

    return _submitter(
        _cleanup_wrapper,
        tmppath,
        outpath,
        checkfile_path,
        *tmpfile_futs,
        **client_kwargs,
    )


def _cleanup_wrapper(inpath, outpath, checkfile_path, *futs):
    if futs[0] is not None and futs[0] >= 0:
        return futs[0]

    # if not already run, perform the clean and copy now
    return cleanup_zarr(inpath, outpath, checkfile_path, "single-threaded")


def cleanup_zarr(tmppath, outpath, checkfile_path, scheduler):
    with dask.config.set(scheduler=scheduler):
        out = xr.open_zarr(str(tmppath))
        test_var = "maxs" if "maxs" in out.data_vars else "pddi"
        attrs = out.attrs.copy()

        grid_width = out.x_ix.attrs["cell_size"]

        assert out[test_var].notnull().all()

        valid = (out[test_var] > 0).any(dim=["y_ix", "x_ix"]).load()

        if not valid.any():
            with checkfile_path.open("w") as f:
                f.write("0")
            out.close()
            shutil.rmtree(str(tmppath))
            return 0

        for v in out.variables:
            out[v].encoding = {}

        out.coords["lon"] = (
            ("x_ix",),
            spatial.grid_ix_to_val(out.x_ix.values, grid_width),
        )
        out.coords["lat"] = (
            ("y_ix",),
            spatial.grid_ix_to_val(out.y_ix.values, grid_width, lon_mask=True),
        )

        out = (
            out.isel(storm=valid)
            .chunk({"storm": min(25, int(valid.sum()))})
            .reset_coords("start_date")
        )
        out.attrs = attrs

        out.to_zarr(str(outpath), mode="w")
        out.close()
        n_tracks = valid.sum().item()
        with checkfile_path.open("w") as f:
            f.write(str(n_tracks))

        shutil.rmtree(str(tmppath))

        return n_tracks
