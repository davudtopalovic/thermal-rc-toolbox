import pandas as pd
import json
from pathlib import Path
from .model import FiveR1C_ENISO13790

def load_best_5r1c(results_path, best_from_cluster=False):

    with open(results_path, "r") as f:
        results = json.load(f)

    max_hp_power = results["setup"]["building_config"]["max_HP_power"]
    init_config  = results["setup"]["identification_config"]["model_config"]["init_config"]
    model = FiveR1C_ENISO13790()
    model.set_heat_power_range([0, max_hp_power])
    model.set_initialization(init_config=init_config)

    if best_from_cluster:
        c = results["results"]["clustering-results"]
        best_key = min(c, key=lambda k: c[k]["best_candidate"]["cl_eval_score"])
        params   = c[best_key]["best_candidate"]["params"]
    else: 
        params = results["results"]["best_candidate"]["params"]

    model.set_params(params)

    return model


def get_best_candidates(results_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
        df:     DataFrame, rows = parameters (and optionally scores),
                columns = ['best', '0', '1', ...]
        params: dict mapping column name -> clean params dict
                (always parameters only, no scores)
    """

    with open(results_path, "r") as f:
        results = json.load(f)
        
    clusters = results["results"]["clustering-results"]
    overall_best = results["results"]["best_candidate"]

    params: dict[str, dict] = {"best": dict(overall_best["params"])}
    for cid, c in clusters.items():
        params[cid] = dict(c["best_candidate"]["params"])

    cols = {k: dict(p) for k, p in params.items()}
      
    candidates = {"best": overall_best, **{cid: c["best_candidate"] for cid, c in clusters.items()}}
    for k, cand in candidates.items():
        cols[k].update({
            "train_error": cand.get("train_error"),
            "validation_error": cand.get("validation_error"),
        })

    return pd.DataFrame(cols), params


def get_best_candidates_2(directory_path) -> tuple[pd.DataFrame, dict]:
    """Enriched summary of the overall-best and per-cluster best candidates.

    Reads ``results.json`` and ``dfRtopN.csv`` from ``directory_path`` and returns a
    summary table that — in addition to the parameters of each best candidate —
    reports per-cluster spread (median ± std) of train/val MAE, the generation in
    which the best candidate of each group appeared, and the cluster member counts.

    Parameters
    ----------
    directory_path : str or pathlib.Path
        Directory containing both ``results.json`` and ``dfRtopN.csv``.

    Returns
    -------
    cols : pandas.DataFrame
        Rows = ``{'best', '0', '1', ...}``. Columns = the 7 free parameters plus
        ``'train MAE (W)'``, ``'val MAE (W)'``, ``'N generation'``, ``'N samples'``.
        For cluster rows the MAE columns are formatted ``"median ± std"`` over the
        cluster members; the ``'best'`` row reports the single overall-best
        candidate's value.
    d : dict
        ``{entity_label: params_dict}`` — parameters only, no scores. Provided for
        downstream code that needs to instantiate a model per entity.
    """
    directory_path = Path(directory_path)
    results_path = directory_path / "results.json"
    dfRtopN_path = directory_path / "dfRtopN.csv"

    with open(results_path, "r") as f:
        results = json.load(f)
    dfRtopN = pd.read_csv(dfRtopN_path)

    clusters = results["results"]["clustering-results"]
    overall_best = results["results"]["best_candidate"]

    # Params dict — parallel to get_best_candidates' second return value
    d: dict[str, dict] = {"best": dict(overall_best["params"])}
    for cid, c in clusters.items():
        d[cid] = dict(c["best_candidate"]["params"])

    # Base wide table: rows = parameters, columns = entities. object dtype so that
    # we can later write "median ± std" strings into the MAE rows.
    cols = pd.DataFrame({k: dict(v) for k, v in d.items()}).astype(object)

    # Fill train/val MAE for each entity from results.json
    candidates = {"best": overall_best, **{cid: c["best_candidate"] for cid, c in clusters.items()}}
    for k, cand in candidates.items():
        cols.at["train_error", k] = cand.get("train_error")
        cols.at["validation_error", k] = cand.get("validation_error")

    def _fmt_med_std(series: pd.Series) -> str:
        return f"{series.median():.1f} ± {series.std():.1f}"

    # Replace cluster columns' train/val with median ± std over cluster members
    for cid in [c for c in cols.columns if c != "best"]:
        members = dfRtopN[dfRtopN["best_cluster"] == int(cid)]
        cols.at["train_error", cid] = _fmt_med_std(members["train_error"])
        cols.at["validation_error", cid] = _fmt_med_std(members["validation_error"])

    # Format the 'best' column train/val as single floats with 1 decimal
    cols.at["train_error", "best"] = f"{float(cols.at['train_error', 'best']):.1f}"
    cols.at["validation_error", "best"] = f"{float(cols.at['validation_error', 'best']):.1f}"

    # Generation of the best candidate within each group + group sizes
    best_row = dfRtopN.loc[dfRtopN["criteria"].idxmin()]
    extra: dict[str, dict] = {
        "best": {"N generation": int(best_row["generation"]), "N samples": 1}
    }
    for cid in [c for c in cols.columns if c != "best"]:
        members = dfRtopN[dfRtopN["best_cluster"] == int(cid)]
        cl_best = members.loc[members["criteria"].idxmin()]
        extra[cid] = {
            "N generation": int(cl_best["generation"]),
            "N samples": int(len(members)),
        }

    # Transpose so entities become rows; rename MAE columns for readability
    cols = pd.concat([cols, pd.DataFrame(extra)]).T
    cols = cols.rename(columns={
        "train_error": "train MAE (W)",
        "validation_error": "val MAE (W)",
    })

    return cols, d
