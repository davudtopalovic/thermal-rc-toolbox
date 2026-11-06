"""Building information used by the physics-derived parameter spaces.

The building config (``--building-config``) is optional: only models whose
bounds or setup depend on the building read it. The keys they use:

    "geometry":         {"A_f": <floor area of one floor, m2>, "n_floors": <number of floors>}
    "exact_A_f_bounds": [A_f_min, A_f_max]   # conditioned floor area range, instead of "geometry"
    "max_HP_power":     <W>                  # heat-demand tasks: upper limit of the heat power

Bounds chosen for a calibration belong to the identification config
(``model_config["free_param_bounds"]``), not to the building config.
"""


def require(building_config: dict, key: str, model_name: str, purpose: str):
    """``building_config[key]``, or a ValueError naming the model and what it needs the key for."""
    value = (building_config or {}).get(key)
    if value is None:
        raise ValueError(
            f"{model_name} needs building_config[{key!r}] ({purpose}); "
            f"pass --building-config with that key"
        )
    return value


def floor_area_bounds(building_config: dict, model_name: str) -> tuple[float, float]:
    """Range (A_f_min, A_f_max) of the conditioned floor area [m2].

    From ``exact_A_f_bounds`` if given, otherwise from ``geometry``: at least
    50 % of the total floor area (all floors) is conditioned, at most 130 %
    (the extra 30 % for e.g. balconies or terraces counted in the zone).
    """
    building_config = building_config or {}
    if "JPD" in building_config:
        raise ValueError(
            "building_config['JPD'] was renamed to 'geometry': "
            '{"geometry": {"A_f": ..., "n_floors": ...}}'
        )
    exact = building_config.get("exact_A_f_bounds")
    if exact:
        A_f_min, A_f_max = exact
        return float(A_f_min), float(A_f_max)
    geometry = building_config.get("geometry")
    if geometry and geometry.get("A_f") and geometry.get("n_floors"):
        A_f, n_floors = geometry["A_f"], geometry["n_floors"]
        return 0.5 * n_floors * A_f, (n_floors + 0.3) * A_f
    raise ValueError(
        f"{model_name} derives its bounds from the floor area: pass --building-config with "
        '"geometry": {"A_f": ..., "n_floors": ...} or "exact_A_f_bounds": [min, max]'
    )
