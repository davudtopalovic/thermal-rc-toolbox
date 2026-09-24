import numpy as np
import pandas as pd

def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray, axis: int = None) -> dict:
    """
    Compute standard forecast error metrics.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    axis   : int, optional
        - None → global metrics (flattened)
        - 1    → per-batch metrics for shape (B, T)

    Returns
    -------
    dict
        - axis=None → dict of scalars
        - axis=1    → dict of lists (per-batch)
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert y_true.shape == y_pred.shape, "Shapes must match"

    # 🔹 GLOBAL METRICS
    if axis is None:
        diff = y_true.flatten() - y_pred.flatten()

        mae = np.nanmean(np.abs(diff))
        mse = np.nanmean(diff**2)
        rmse = np.sqrt(mse)

        mean_true = np.nanmean(y_true)
        sum_true = np.nansum(y_true)
        sum_pred = np.nansum(y_pred)

        cv_rmse = rmse * 100 / mean_true if mean_true != 0 else np.nan
        RE = np.abs(sum_true - sum_pred) * 100 / sum_true if sum_true != 0 else np.nan
        nmbe = np.nanmean(diff) * 100 / mean_true if mean_true != 0 else np.nan

        return {
                "MAE": np.round(mae, 2),
                "MSE": np.round(mse, 2),
                "RMSE": np.round(rmse, 2),
                "CV(RMSE)": np.round(cv_rmse, 2) if not np.isnan(cv_rmse) else np.nan,
                "RE": np.round(RE, 2) if not np.isnan(RE) else np.nan,
                "NMBE": np.round(nmbe, 2) if not np.isnan(nmbe) else np.nan
                }

    # 🔹 PER-BATCH METRICS (vectorized, axis=1 assumed)
    if axis == 1:
        diff = y_true - y_pred

        mae = np.nanmean(np.abs(diff), axis=1)
        mse = np.nanmean(diff**2, axis=1)
        rmse = np.sqrt(mse)

        mean_true = np.nanmean(y_true, axis=1)
        sum_true = np.nansum(y_true, axis=1)
        sum_pred = np.nansum(y_pred, axis=1)

        cv_rmse = np.where(mean_true != 0, rmse * 100 / mean_true, np.nan)
        RE = np.where(sum_true != 0, np.abs(sum_true - sum_pred) * 100 / sum_true, np.nan)
        nmbe = np.where(mean_true != 0, np.nanmean(diff, axis=1) * 100 / mean_true, np.nan)

        return {
            "MAE": np.round(mae, 2),
            "MSE": np.round(mse, 2),
            "RMSE": np.round(rmse, 2),
            "CV(RMSE)": np.round(cv_rmse, 2),
            "RE": np.round(RE, 2),
            "NMBE": np.round(nmbe, 2),
        }

    # 🔹 Optional: extend later for other axes
    raise NotImplementedError("Currently only supports axis=None or axis=1")

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, error_thresholds: dict = None) -> dict:

    err_dict = compute_forecast_metrics(y_true, y_pred, axis=1)
    err_df = pd.DataFrame(err_dict)

    summary = {}
    for metric in err_df.columns:

        values = err_df[metric].values
        
        q25 = np.nanpercentile(values, 25)
        q75 = np.nanpercentile(values, 75)

        summary[metric] = {
            "mean": np.round(np.nanmean(values),2),
            "median": np.round(np.nanmedian(values),2),
            "std": np.round(np.nanstd(values),2),
            "iqr": [np.round(q25,2), np.round(q75,2)]
        }

    # Threshold compliance
    threshold_results = {}

    if error_thresholds is not None:

        for metric, bounds in error_thresholds.items():

            values = err_df[metric].values
            valid = ~np.isnan(values)

            if np.sum(valid) == 0:
                percentage_within = np.nan
            else:
                if len(bounds) == 1:
                    threshold = bounds[0]
                    within = (values <= threshold) & valid

                elif len(bounds) == 2:
                    lower, upper = bounds
                    within = (values >= lower) & (values <= upper) & valid

                percentage_within = 100 * np.sum(within) / np.sum(valid)

            threshold_results[metric] = np.round(percentage_within,2)


    return {
            "summary_statistics": summary,
            "threshold_compliance_%": threshold_results,
            "all_errors": err_df
        }


def organise_results_in_dataframe(summary, thresholds=None):
    
    rows = []
    for metric, stats in summary.items():
        rows.append({
            "metric": metric,
            "mean": stats["mean"],
            "median": stats["median"],
            "std": stats["std"],
            "iqr": stats["iqr"]
        })
        if thresholds is not None and metric in thresholds:
            rows[-1]["threshold_compliance_%"] = thresholds[metric]
    df_results = pd.DataFrame(rows).set_index("metric")
    df_results = df_results.reindex(["CV(RMSE)", "NMBE", "MAE", "RE", "RMSE"])
    
    return df_results