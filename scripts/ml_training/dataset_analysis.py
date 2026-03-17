from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from scripts.config import unittests_dir


def load_xy_from_netcdf(
    input_path: Path,
    output_path: Path,
    input_dims=("time", "column"),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load NetCDF inputs/outputs and produce (X, Y) arrays with the same
    preprocessing as your training Dataset, but without converting to torch.

    Returns:
        X: (n_samples, in_features)
        Y: (n_samples, out_features)
        meta: dict with basic info (var names, shapes, etc.)
    """
    ds_in = xr.open_dataset(input_path)
    ds_out = xr.open_dataset(output_path)

    # Basic consistency check
    for d in input_dims:
        assert ds_in.sizes[d] == ds_out.sizes[d], f"dim mismatch on {d}"

    in_vars = [
        v for v in ds_in.data_vars
        if np.issubdtype(ds_in[v].dtype, np.floating)
    ]
    out_vars = [
        v for v in ds_out.data_vars
        if np.issubdtype(ds_out[v].dtype, np.floating)
    ]

    ds_in_sel = ds_in[in_vars]
    ds_out_sel = ds_out[out_vars]

    X_da = (
        ds_in_sel
        .to_array("feature")
        .transpose("time", "column", "feature", ...)
        .stack(sample=("time", "column"))
    )

    Y_da = (
        ds_out_sel
        .to_array("target")
        .transpose("time", "column", "target", ...)
        .stack(sample=("time", "column"))
    )

    n_samples = ds_in.sizes["time"] * ds_in.sizes["column"]
    assert X_da.sizes["sample"] == n_samples
    assert Y_da.sizes["sample"] == n_samples

    X = X_da.data.reshape(n_samples, -1)
    Y = Y_da.data.reshape(n_samples, -1)

    x_mask = np.isfinite(X).all(axis=1)
    y_mask = np.isfinite(Y).all(axis=1)
    mask = x_mask & y_mask
    X = X[mask]
    Y = Y[mask]

    meta: Dict[str, Any] = {
        "in_vars": in_vars,
        "out_vars": out_vars,
        "input_shape": X.shape,
        "output_shape": Y.shape,
        "n_samples_raw": int(n_samples),
        "n_samples_after_mask": int(X.shape[0]),
    }
    return X, Y, meta


def basic_stats(X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
    """
    Compute basic distribution stats for features and targets.
    """
    x_std = X.std(axis=0)
    y_std = Y.std(axis=0)

    stats: Dict[str, Any] = {
        "X": {
            "mean": float(X.mean()),
            "std": float(X.std()),
            "per_feature_std_min": float(x_std.min()),
            "per_feature_std_med": float(np.median(x_std)),
            "per_feature_std_max": float(x_std.max()),
        },
        "Y": {
            "mean": float(Y.mean()),
            "std": float(Y.std()),
            "per_feature_std_min": float(y_std.min()),
            "per_feature_std_med": float(np.median(y_std)),
            "per_feature_std_max": float(y_std.max()),
        },
    }
    return stats


def filter_low_variance_targets(
    Y: np.ndarray,
    std_threshold: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop target dimensions whose std < std_threshold.

    Returns:
        Y_filtered: (n_samples, n_kept)
        keep_mask: (n_original_targets,) boolean mask of kept dims
    """
    std = Y.std(axis=0)
    keep = std > std_threshold
    return Y[:, keep], keep


def pca_reduce(
    X: np.ndarray,
    variance_ratio: float = 0.99,
    max_components: Optional[int] = None,
) -> Tuple[np.ndarray, PCA]:
    """
    Fit PCA on X and project to a lower dimension retaining given variance_ratio.

    Args:
        X: (n_samples, n_features)
        variance_ratio: desired cumulative explained variance (0–1)
        max_components: optional absolute cap on number of components.

    Returns:
        X_pca: (n_samples, n_components)
        pca: fitted sklearn PCA object
    """
    n_features = X.shape[1]
    if max_components is None:
        max_components = n_features

    pca = PCA(n_components=min(max_components, n_features))
    X_pca_full = pca.fit_transform(X)

    # Determine how many components we need for desired variance_ratio
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumsum, variance_ratio) + 1)
    k = min(k, X_pca_full.shape[1])

    X_pca = X_pca_full[:, :k]
    return X_pca, pca


def summarize_and_print(
    X: np.ndarray,
    Y: np.ndarray,
    meta: Dict[str, Any],
    std_threshold: float = 1e-3,
    pca_variance: float = 0.99,
    pca_max_components: int = 512,
) -> None:
    """
    Convenience function: print a concise summary + simple reduction plan.
    """
    print("=== Dataset summary ===")
    print(f"Raw samples (time*column): {meta['n_samples_raw']}")
    print(f"Samples after NaN mask    : {meta['n_samples_after_mask']}")
    print(f"Input shape  (N, D_in)    : {X.shape}")
    print(f"Output shape (N, D_out)   : {Y.shape}")

    stats = basic_stats(X, Y)
    print("\nX global mean/std:", stats["X"]["mean"], stats["X"]["std"])
    print(
        "X per-feature std min/med/max:",
        stats["X"]["per_feature_std_min"],
        stats["X"]["per_feature_std_med"],
        stats["X"]["per_feature_std_max"],
    )
    print("Y global mean/std:", stats["Y"]["mean"], stats["Y"]["std"])
    print(
        "Y per-feature std min/med/max:",
        stats["Y"]["per_feature_std_min"],
        stats["Y"]["per_feature_std_med"],
        stats["Y"]["per_feature_std_max"],
    )

    # Low-variance target filtering
    Y_filt, keep_mask = filter_low_variance_targets(Y, std_threshold=std_threshold)
    n_kept = int(keep_mask.sum())
    n_dropped = int(keep_mask.size - n_kept)
    print(f"\nLow-variance target filter (thr={std_threshold:g}):")
    print(f"  kept {n_kept} targets, dropped {n_dropped}")

    # PCA suggestion on inputs
    X_pca, pca = pca_reduce(
        X,
        variance_ratio=pca_variance,
        max_components=pca_max_components,
    )
    print(
        f"\nPCA suggestion on X: retain {X_pca.shape[1]} components "
        f"to explain ~{pca_variance*100:.1f}% variance "
        f"(capped at {pca_max_components})."
    )
    print("  Explained variance ratios (first 10):",
          pca.explained_variance_ratio_[:10])


def summarize_data():
    data_dir = Path(unittests_dir)/ "input-data"
    input_nc = data_dir / "spel-inputs-training_samples.nc"
    output_nc = data_dir / "spel-outputs-training_samples.nc"

    assert input_nc.exists() and output_nc.exists()

    X, Y, meta = load_xy_from_netcdf(input_nc, output_nc)
    summarize_and_print(X, Y, meta)

