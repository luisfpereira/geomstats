"""Engine resolution for kernel computations."""

import importlib


def resolve_engine(engine="auto"):
    """Resolve a kernel computation engine.

    Parameters
    ----------
    engine : str
        Engine specification. ``"auto"`` uses ``"keops_genred"`` when
        PyKeOps is available and ``"geomstats"`` otherwise. ``"keops"`` is
        an alias for ``"keops_genred"``. Engine names starting with
        ``"geomstats"`` resolve to ``"geomstats"``.

    Returns
    -------
    engine : {"geomstats", "keops_genred", "keops_lazy"}
        Resolved engine.
    """
    if engine == "auto":
        has_keops = importlib.util.find_spec("pykeops") is not None
        return "keops_genred" if has_keops else "geomstats"

    if engine.startswith("geomstats"):
        return "geomstats"

    if engine == "keops":
        return "keops_genred"

    if engine not in ("keops_genred", "keops_lazy"):
        raise ValueError(f"Unknown engine: {engine!r}")

    return engine
