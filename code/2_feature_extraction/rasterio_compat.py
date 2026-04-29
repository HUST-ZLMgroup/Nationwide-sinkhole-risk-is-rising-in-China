import ctypes
import os
import sys
from pathlib import Path


def _iter_libnetcdf19_candidates():
    env_prefix = Path(sys.prefix).resolve()
    seen = set()

    direct_candidate = env_prefix / "lib" / "libnetcdf.so.19"
    if direct_candidate.exists():
        seen.add(str(direct_candidate))
        yield direct_candidate

    conda_roots = []
    if len(env_prefix.parents) >= 2:
        conda_roots.append(env_prefix.parents[1])
    home_anaconda = Path.home() / "anaconda3"
    if home_anaconda not in conda_roots:
        conda_roots.append(home_anaconda)

    for conda_root in conda_roots:
        pkgs_dir = conda_root / "pkgs"
        if not pkgs_dir.exists():
            continue
        for candidate in sorted(pkgs_dir.glob("libnetcdf-*/lib/libnetcdf.so.19")):
            if str(candidate) in seen:
                continue
            seen.add(str(candidate))
            yield candidate


def _preload_libnetcdf19():
    mode = getattr(os, "RTLD_GLOBAL", 0)
    errors = []

    for candidate in _iter_libnetcdf19_candidates():
        try:
            ctypes.CDLL(str(candidate), mode=mode)
            print(f"[rasterio_compat] Use the compatibility library:{candidate}")
            return True
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")

    joined = "\n".join(errors)
    raise ImportError(
        "libnetcdf.so.19.\\n"
        ":\\n"
        f"{joined or 'No candidate files found'}"
    )


def _import_rasterio():
    try:
        import rasterio as _rasterio

        return _rasterio
    except ImportError as exc:
        if "libnetcdf.so.19" not in str(exc):
            raise

        _preload_libnetcdf19()
        sys.modules.pop("rasterio", None)
        import rasterio as _rasterio

        return _rasterio


rasterio = _import_rasterio()
