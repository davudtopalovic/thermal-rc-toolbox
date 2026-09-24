import numpy as np
from scipy.signal import savgol_filter
from rctoolbox.utils.get_scale import get_scale


def select_best_solutions(
    dfR,
    threshold_config={"type": "percentage", "x": 0.05},
):
    """
    Select best solutions based on criteria and thresholding strategy.

    Parameters
    ----------
    dfR : pd.DataFrame
    threshold_config : dict
        {"type": "percentage", "x": float}
        or
        {"type": "elbow"}
        or
        {"type": "idx", "x": int}

    Returns
    -------
    dfRtopN : pd.DataFrame
    threshold : float
    threshold_idx : int
    """

    dfR_ = dfR.copy()

    min_criteria = dfR_["criteria"].min()

    dfR_sorted = (
        dfR_.sort_values(by="criteria", ascending=True).reset_index(drop=True)
    )

    t_type = threshold_config.get("type", None)

    if t_type == "percentage":

        if "x" not in threshold_config:
            raise ValueError("For 'percentage', provide 'x' in threshold_config")

        ratio = threshold_config["x"]
        threshold = min_criteria * (1 + ratio)

        mask = dfR_sorted["criteria"] <= threshold

        dfRtopN = dfR_sorted[mask].reset_index(drop=True)
        threshold_idx = len(dfRtopN)

    elif t_type == "elbow":

        x = dfR_sorted.index.to_numpy()
        y = dfR_sorted["criteria"].values

        # ---- smoothing ----
        polyorder = 2
        window = int(get_scale(len(dfR_sorted)) * 0.1)
        # ---- safety guards ----
        # 1. ensure strictly larger than polyorder
        window = max(polyorder + 1, window)
        # 2. ensure odd
        window = window if window % 2 == 1 else window + 1
        # 3. ensure it does not exceed data length
        window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
        # 4. final guard (in case len(y) is very small)
        if window <= polyorder:
            window = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2

        y_smooth = savgol_filter(y, window_length=window, polyorder=2)

        # ---- curvature ----
        dy = np.gradient(y_smooth, x)
        d2y = np.gradient(dy, x)

        knee_idx = int(np.argmax(np.abs(d2y)))

        # optional safeguard (avoid plateau artifacts)
        knee_idx = max(knee_idx, int(0.02 * len(y)))

        # ---- selection (LEFT side: best → elbow) ----
        start_idx = knee_idx
        dfRtopN = dfR_sorted.iloc[:start_idx+1].reset_index(drop=True)
        threshold_idx = len(dfRtopN)

        threshold = y[knee_idx]

    elif t_type == "idx":

        if "x" not in threshold_config:
            raise ValueError("For 'idx', provide 'x' in threshold_config")

        idx = threshold_config["x"]

        if idx < 1 or idx > len(dfR_sorted):
            raise ValueError(f"Index out of bounds. Must be between 1 and {len(dfR_sorted)}")

        dfRtopN = dfR_sorted.iloc[:idx].reset_index(drop=True)
        threshold = dfR_sorted.iloc[idx-1]["criteria"]
        threshold_idx = idx

    else:
        raise ValueError("threshold_config['type'] must be 'percentage', 'elbow', or 'idx'")

    return dfRtopN, threshold, threshold_idx