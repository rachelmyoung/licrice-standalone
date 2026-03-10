"""Radius estimation functions for LICRICE.

Extracted from pyTC/tracks/radius.py. estimate_rmw, estimate_rmw_climada,
estimate_rmw_licrice, create_radius_reg_dataset, and estimate_radii are verbatim.
get_radius_ratio_models and load_radius_models are new (adapted from coastal-core's
get_radius_ratio_models which takes a Settings object; standalone takes ds directly).
"""

import pickle  # NEW
from pathlib import Path  # NEW

import numpy as np
import pandas as pd
import xarray as xr
from joblib import dump, load  # NEW (coastal-core only imports dump)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from licrice.utilities import smooth_fill  # NEW


def estimate_rmw(pres, v_circular, lat, time_dim="time"):
    """Estimate RMW using a central pressure-based regression first and, if P is
    unavailable, then using the lat+Vmax regression from LICRICE.

    Parameters
    ----------
    pres : float or :class:`xarray.DataArray`
        Maximum azimuthal wind, indexed by ``storm`` and ``time``
    v_circular : float or :class:`xarray.DataArray`
        Central Pressure estimates, indexed by ``storm`` and ``time``
    lat : float or :class:`xarray.DataArray`
        Latitude, indexed by ``storm`` and ``time``

    Returns
    -------
    float or :class:`numpy.ndarray` or :class:`xarray.DataArray`
        Estimated Radius of Maximum Wind, in meters, indexed by ``storm`` and ``time``

    """
    pres_ermw = estimate_rmw_climada(pres).interpolate_na(dim="time")
    lat_ermw = estimate_rmw_licrice(v_circular, lat).interpolate_na(dim="time")
    return smooth_fill(pres_ermw, lat_ermw)


def estimate_rmw_climada(pres):
    """Uses central pressure to estimate RMW (in km). Borrowed from
    https://github.com/CLIMADA-project/climada_python/blob/main/climada/hazard/tc_tracks.py#L1067.

    Parameters
    ----------
    pres : float or :class:`numpy.ndarray` or :class:`xarray.DataArray`
        Central Pressure estimates, in hPa.

    Returns
    -------
    float or :class:`numpy.ndarray` or :class:`xarray.DataArray`
        Estimated Radius of Maximum Wind, in km

    """
    pres_l = [872, 940, 980, 1021]
    rmw_l = [14.907318, 15.726927, 25.742142, 56.856522]

    ermw = pres * 0  # maintain nans
    for i, pres_l_i in enumerate(pres_l):
        slope_0 = 1.0 / (pres_l_i - pres_l[i - 1]) if i > 0 else 0
        slope_1 = 1.0 / (pres_l[i + 1] - pres_l_i) if i + 1 < len(pres_l) else 0
        ermw += rmw_l[i] * np.fmax(
            0,
            (
                1
                - slope_0 * np.fmax(0, pres_l_i - pres)
                - slope_1 * np.fmax(0, pres - pres_l_i)
            ),
        )
    return ermw


def estimate_rmw_licrice(v_circular, lat):
    """Combines latitude, translational velocity, and maximum wind speed to estimate the
    radius of maximum wind.

    Eqn from p.3 of LICRICE docs

    Parameters
    ----------
    v_circular : float or :class:`numpy.ndarray` or :class:`xarray.DataArray`
        Maximum azimuthal wind, in m/s
    lat : float or :class:`numpy.ndarray` or :class:`xarray.DataArray`
        Latitude, in degrees

    Returns
    -------
    float or :class:`numpy.ndarray` or :class:`xarray.DataArray`
        Estimated Radius of Maximum Wind, in km

    """
    ermw = 63.273 - 0.8683 * v_circular + 1.07 * np.abs(lat)

    return ermw


def create_radius_reg_dataset(
    ds,
    rmw_var="rmstore",
    radius_var="storm_radius",
    reg_cols=None,
):
    rat_estimator = (
        ds[
            [
                rmw_var,
                radius_var,
                "latstore",
                "basin",
                "subbasin",
                "v_circular",
                "v_total",
                "dist2land",
                "nature",
                "v_trans_x",
                "v_trans_y",
                "longstore",
            ]
        ]
        .to_dataframe()
        .rename(columns={rmw_var: "rmstore", radius_var: "storm_radius"})
    )

    rat_estimator = rat_estimator.dropna(
        how="any",
        subset=[
            v for v in rat_estimator.columns if v not in ["rmstore", "storm_radius"]
        ],
    )
    rat_estimator["hemisphere"] = (rat_estimator.latstore > 0).astype(np.uint8)
    rat_estimator["abslat"] = np.abs(rat_estimator.latstore)
    rat_estimator["subbasin"] = rat_estimator.basin + rat_estimator.subbasin
    rat_estimator = rat_estimator.drop(columns=["latstore", "basin"])

    categoricals = ["subbasin", "nature"]
    enc = OneHotEncoder(sparse_output=False, drop="first")
    enc.fit(rat_estimator[categoricals])
    basins = pd.DataFrame(
        enc.transform(rat_estimator[categoricals]),
        columns=enc.get_feature_names_out(categoricals),
        index=rat_estimator.index,
    )
    out = pd.concat((rat_estimator.drop(columns=categoricals), basins), axis=1)

    if reg_cols is not None:
        addl_cols = pd.DataFrame(
            0,
            index=out.index,
            columns=[c for c in reg_cols if c not in out.columns],
        )
        out = pd.concat((out, addl_cols), axis=1).reindex(columns=reg_cols)
    return out


def estimate_radii(ds, rmw_to_rad, rad_to_rmw, rmw, reg_cols=None):
    # fix contexts in which outer radius is smaller than RMW in ibtracs data
    ds["storm_radius_estimated"] = (
        xr.concat((ds.storm_radius, ds.rmstore), dim="var")
        .max(dim="var")
        .where(ds.storm_radius.notnull())
    )

    # use ratios of storm radius and RMW observed in storm to extrapolate when only one
    # of the two variables is missing at start or end of storm
    ds["rmstore_estimated"] = smooth_fill(
        ds.rmstore,
        ds.storm_radius,
        fill_all_null=False,
    )
    ds["storm_radius_estimated"] = smooth_fill(
        ds.storm_radius_estimated,
        ds.rmstore,
        fill_all_null=False,
    )

    # Fill in RMW using relationship to other vars INCLUDING ROCI
    X = create_radius_reg_dataset(
        ds,
        rmw_var="rmstore_estimated",
        radius_var="storm_radius_estimated",
        reg_cols=reg_cols,
    )
    X = X[X.storm_radius.notnull()]
    if len(X):
        rmw_estimated = (
            pd.Series(rad_to_rmw.predict(X.drop(columns="rmstore")), index=X.index)
            .to_xarray()
            .reindex(time=ds.time)
        )
        rmw_estimated = xr.concat(
            (rmw_estimated, ds.storm_radius_estimated),
            dim="var",
        ).min(dim="var")
        ds["rmstore_estimated"] = smooth_fill(ds.rmstore_estimated, rmw_estimated)

    # Fill in RMW using relationship to other vars WHEN ROCI NOT AVAILABLE
    X = create_radius_reg_dataset(
        ds,
        rmw_var="rmstore_estimated",
        radius_var="storm_radius_estimated",
        reg_cols=reg_cols,
    )
    rmw_estimated = (
        pd.Series(
            rmw.predict(X.drop(columns=["rmstore", "storm_radius"])),
            index=X.index,
        )
        .to_xarray()
        .reindex(time=ds.time)
    )
    rmw_estimated = xr.concat(
        (rmw_estimated, ds.storm_radius_estimated),
        dim="var",
    ).min(dim="var")
    ds["rmstore_estimated"] = smooth_fill(ds.rmstore_estimated, rmw_estimated)

    # Fill in ROCI using relationship to other vars INCLUDING RMW
    X = create_radius_reg_dataset(
        ds,
        rmw_var="rmstore_estimated",
        radius_var="storm_radius_estimated",
        reg_cols=reg_cols,
    )
    assert X.rmstore.notnull().all()
    storm_radius_estimated = (
        pd.Series(rmw_to_rad.predict(X.drop(columns="storm_radius")), index=X.index)
        .to_xarray()
        .reindex(time=ds.time)
    )
    storm_radius_estimated = xr.concat(
        (storm_radius_estimated, ds.rmstore_estimated),
        dim="var",
    ).max(dim="var")
    ds["storm_radius_estimated"] = smooth_fill(
        ds.storm_radius_estimated,
        storm_radius_estimated,
    )

    ds["rmstore_estimated"].attrs.update(
        {
            "long_name": "radius of maximum wind estimated",
            "units": "km",
            "method": (
                "When obs are available, use obs. When unavailable, check central "
                "pressure. If available, use pcen-->rmw relationship to estimate rmw. "
                "If not, use lat+v_circ-->rmw relationship (LICRICE) to estimate rmw. "
                "After estimation, bias correct estimates to first and last observed "
                "values. Cap at observed ROCI."
            ),
        }
    )

    ds["storm_radius_estimated"].attrs.update(
        {
            "long_name": "radius of last closed isobar estimated",
            "units": "km",
            "method": (
                "When obs are available, use obs. When unavailable for some time "
                "points, extrapolate RMW->ROCI relationship from last observed ROCI. "
                "When unavailable for any time points, leave as NaN. Clip so that it "
                "is at least as big as rmstore_estimated."
            ),
        }
    )

    # ensure no negative estimated values and no missing values when lat/long is
    # non-missing
    test = ds[["storm_radius_estimated", "rmstore_estimated"]]
    assert ((test > 0) | test.isnull()).to_array().all().item()
    assert (
        (test.to_array(dim="var").notnull().all(dim="var") | ds.latstore.isnull())
        .all()
        .item()
    )

    return ds


def get_radius_ratio_models(ds, model_dir=None):  # NEW
    """Train RF models for RMW and ROCI estimation and optionally save to disk.

    Adapted from coastal-core's get_radius_ratio_models(ps, save). Takes ``ds``
    directly instead of loading via ibtracs.load_processed_ibtracs.

    Parameters
    ----------
    ds : xarray.Dataset
        Cleaned track dataset (output of format_clean).
    model_dir : Path or None
        Directory to save .pkl model files. If None, models are not saved.

    Returns
    -------
    tuple
        (rmw_to_rad, rad_to_rmw, rad, rmw, cols)

    """
    df = create_radius_reg_dataset(ds)
    cols = list(df.columns)

    this_df = df.dropna(subset=["storm_radius", "rmstore"], how="any")
    rmw_to_rad = RandomForestRegressor(random_state=0, oob_score=True)
    rmw_to_rad.fit(this_df.drop(columns="storm_radius"), this_df.storm_radius)
    rad_to_rmw = RandomForestRegressor(random_state=0, oob_score=True)
    rad_to_rmw.fit(this_df.drop(columns="rmstore"), this_df.rmstore)

    this_df = df.dropna(subset=["storm_radius"])
    rad = RandomForestRegressor(random_state=0, oob_score=True)
    rad.fit(this_df.drop(columns=["storm_radius", "rmstore"]), this_df.storm_radius)

    this_df = df.dropna(subset=["rmstore"])
    rmw = RandomForestRegressor(random_state=0, oob_score=True)
    rmw.fit(this_df.drop(columns=["storm_radius", "rmstore"]), this_df.rmstore)

    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        for obj, name in (
            (rmw_to_rad, "rmw_to_rad"),
            (rad_to_rmw, "rad_to_rmw"),
            (rad, "rad"),
            (rmw, "rmw"),
        ):
            with (model_dir / f"{name}.pkl").open("wb") as f:
                dump(obj, f)
        with (model_dir / "cols.pkl").open("wb") as f:
            pickle.dump(cols, f)

    return rmw_to_rad, rad_to_rmw, rad, rmw, cols


def load_radius_models(model_dir):  # NEW
    """Load RF radius models from disk.

    Parameters
    ----------
    model_dir : Path or str
        Directory containing rmw_to_rad.pkl, rad_to_rmw.pkl, rmw.pkl, cols.pkl.

    Returns
    -------
    tuple
        (rmw_to_rad, rad_to_rmw, rmw, cols)

    """
    model_dir = Path(model_dir)
    rmw_to_rad = load(model_dir / "rmw_to_rad.pkl")
    rad_to_rmw = load(model_dir / "rad_to_rmw.pkl")
    rmw = load(model_dir / "rmw.pkl")
    with (model_dir / "cols.pkl").open("rb") as f:
        cols = pickle.load(f)
    return rmw_to_rad, rad_to_rmw, rmw, cols
