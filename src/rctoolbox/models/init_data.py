"""Helpers shared by the models' ``build_data_dict`` / ``predict_functions``.

The data dict a model is trained on holds measurements only: the forecast
inputs of each window and, under ``data_for_init_state``, the measured
values at the forecast start (``<signal>_0`` for the model's
``SEED_SIGNALS``) and the lag window before it (``t_`` and ``<signal>_``).
The model builds its starting state itself (``RCBaseModel.prepare_init_data``
and the model's ``seed_state``).

``columns`` maps a model signal to the data column (dataloader feature) that
feeds it; the defaults are each model's ``DEFAULT_COLUMNS``.
"""
import warnings

import numpy as np


def resolve_columns(defaults: dict, columns=None, model_name: str = "model", **shorthands) -> dict:
    """Signal -> data column: ``defaults``, then the older keyword shorthands
    (``Phi_HC_col=...`` etc., given as ``signal=column``; None = not given),
    then ``columns``."""
    out = {**defaults, **{k: v for k, v in shorthands.items() if v is not None}, **(columns or {})}
    unknown = set(out) - set(defaults)
    if unknown:
        raise KeyError(f"Unknown {model_name} signals {sorted(unknown)}; expected a subset of {list(defaults)}")
    return out


def init_data(di, columns: dict, seed_signals, lag_signals) -> dict:
    """``data_for_init_state`` of a batch (``di`` = its ``data_inputs``): the
    seed signals at the forecast start and the lag-window measurements."""
    return {
        **{f"{s}_0": np.asarray(di[columns[s]])[:, 0] for s in seed_signals},
        "t_": di["t_array_1Dlag"],
        **{f"{s}_": di[f"{columns[s]}_1Dlag"] for s in lag_signals},
    }


# Sentinel for the deprecated ``T_m0`` argument of the data helpers: the start
# value of a hidden mass is a model setting now (``init_config["T_m0"]``).
UNSET = object()


def legacy_seeds(model, init: dict, **options) -> dict:
    """``init`` plus the explicit seeds ``y_0`` / ``y_0_`` built by ``model``
    with the given options (e.g. T_m0). Used when a caller still passes the
    deprecated ``T_m0=`` to a data helper; the model keeps given seeds."""
    warnings.warn(
        "passing T_m0 to build_data_dict / predict_for_batch is deprecated; set it in "
        "init_config['T_m0'] (the model builds its starting state itself)",
        DeprecationWarning, stacklevel=3,
    )
    for k, v in options.items():
        setattr(model, k, v)
    sig = model.SEED_SIGNALS
    return {**init,
            "y_0":  model.seed_state(**{s: init[f"{s}_0"] for s in sig}),
            "y_0_": model.seed_state(**{s: np.asarray(init[f"{s}_"])[:, 0] for s in sig})}
