import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List


def get_tws_from_dates(dates, wdhs):
    tw_batches = []
    for wdh in wdhs:
        twss = []
        for date in dates:
            t1 = pd.to_datetime(date)
            t2 = t1 + pd.Timedelta(hours=wdh)
            spb, ssh = int(1e10), 1
            tws = generate_time_windows(start_time=t1, end_time=t2, samples_per_batch=spb, window_duration_hours=wdh, slide_step_hours=ssh)
            twss.append(tws)

        if len(twss) > 0:
            twss = flatten_batches_to_single_batch(concat_time_window_dicts(twss))
            tw_batches.append(twss)

    tw_batches = concat_time_window_dicts(tw_batches)
    return tw_batches

def get_tws_from_data_segments(data_segments, wdhs, ssh):
    tw_batches = []
    for wdh in wdhs:
        twss = []
        for segment in data_segments:
            t1, t2 = segment
            t1, t2 = pd.to_datetime(t1), pd.to_datetime(t2)
            wdh_max = (t2 - t1).total_seconds() / 3600 # window duration in hours
            if wdh <= wdh_max:
                spb, ssh = int(1e10), ssh
                tws = generate_time_windows(start_time=t1, end_time=t2, samples_per_batch=spb, window_duration_hours=wdh, slide_step_hours=ssh)
                twss.append(tws)

        if len(twss) > 0:
            twss = flatten_batches_to_single_batch(concat_time_window_dicts(twss))
            tw_batches.append(twss)

    tw_batches = concat_time_window_dicts(tw_batches)
    return tw_batches


# def generate_train_val_time_windows(DR_time_windows, train_keys, val_keys, args={"wdh":24, "spb":64, "ssh":2}):
#     wdh, spb, ssh = args["wdh"], args["spb"], args["ssh"]
#     train_time_windows, val_time_windows = [], []
#     train_time_windows_1b = []
#     for key in train_keys:
#         t1, t2 = DR_time_windows[key]["time-window"]
#         tw = generate_time_windows(start_time=t1, end_time=t2,
#             samples_per_batch=spb, window_duration_hours=wdh, slide_step_hours=ssh)
#         if len(tw["batch_0"]) > 1:
#             train_time_windows.append(tw)
#         else: 
#             train_time_windows_1b.append(tw) 
        
#     for key in val_keys:
#         t1, t2 = DR_time_windows[key]["time-window"]
#         tw = generate_time_windows(start_time=t1, end_time=t2,
#             samples_per_batch=spb, window_duration_hours=wdh, slide_step_hours=ssh)
#         val_time_windows.append(tw)
    
#     if len(train_time_windows_1b) != 0:
#         tws1b = flatten_batches_to_single_batch(concat_time_window_dicts(train_time_windows_1b))
#         train_time_windows = train_time_windows + [tws1b]        
#     tws = {"train": concat_time_window_dicts(train_time_windows), "val": flatten_batches_to_single_batch(concat_time_window_dicts(val_time_windows))}
    
#     return tws


def generate_time_windows(
    start_time: datetime,
    end_time: datetime,
    window_duration_hours: int = 24,
    samples_per_batch: int = int(1e10),
    slide_step_hours: int = 24
) -> Dict:
    """
    Creates time windows grouped into batches, ensuring the final window ends at end_time.
    If the last batch is smaller than samples_per_batch, its samples are merged into the previous batch.
    
    Args:
        start_time: datetime — start time
        end_time: datetime — final sample window must end here
        samples_per_batch: int — nominal samples per batch
        window_duration_hours: int — duration of each sample
        slide_step_hours: int — step between samples

    Returns:
        Dict[str, Dict[str, Dict[str, List[datetime]]]]
    """
    time_windows = {}
    window_delta = timedelta(hours=window_duration_hours)
    slide_delta = timedelta(hours=slide_step_hours)

    # Step 1: Generate all valid sample windows
    all_windows = []
    current_time = start_time
    while current_time + window_delta <= end_time:
        all_windows.append((current_time, current_time + window_delta))
        current_time += slide_delta

    # Step 2: Group into batches
    num_samples = len(all_windows)
    batch_idx = 0
    sample_idx_global = 0
    i = 0

    while i < num_samples:
        # Check if this is the last batch and it's smaller than expected
        if (num_samples - i) < samples_per_batch and batch_idx > 0:
            # Merge remaining samples into the previous batch
            last_batch_key = f"batch_{batch_idx - 1}"
            current_batch = all_windows[i:]
            for local_idx, (t_start, t_end) in enumerate(current_batch):
                sample_key = f"sample_{sample_idx_global}"
                time_windows[last_batch_key][sample_key] = {
                    "fitting_hour_range": [t_start, t_end]
                }
                sample_idx_global += 1
            break  # all samples assigned

        # Normal batch creation
        batch_key = f"batch_{batch_idx}"
        time_windows[batch_key] = {}
        current_batch = all_windows[i : i + samples_per_batch]

        for local_idx, (t_start, t_end) in enumerate(current_batch):
            sample_key = f"sample_{sample_idx_global}"
            time_windows[batch_key][sample_key] = {
                "fitting_hour_range": [t_start, t_end]
            }
            sample_idx_global += 1

        i += samples_per_batch
        batch_idx += 1

    return time_windows

def concat_time_window_dicts(dicts: List) -> Dict:
    """Concatenate multiple time-window dicts end-to-end, re-indexing the keys.

    Each input dict has the shape ``{batch_key: {sample_key: sample}}``. The
    returned dict carries all batches from the inputs in order, with batches
    renamed ``batch_0``, ``batch_1``, ... and samples within each batch
    renamed ``sample_0``, ``sample_1``, ... — guaranteeing the keys are
    contiguous and unique across the whole result.

    Batch boundaries from the inputs are **preserved**: only the keys change.
    To collapse those boundaries into a single batch, use
    :func:`flatten_batches_to_single_batch`.

    Parameters
    ----------
    dicts : list of dict
        Time-window dicts to concatenate.

    Returns
    -------
    dict
        A single dict with re-indexed batch and sample keys.
    """
    merged = {}
    batch_counter = 0  # Global batch index
    sample_counter = 0  # Global sample index

    for d in dicts:
        for old_batch_key, samples in d.items():
            new_batch_key = f"batch_{batch_counter}"
            merged[new_batch_key] = {}
            for _, sample_dict in samples.items():
                sample_key = f"sample_{sample_counter}"
                merged[new_batch_key][sample_key] = sample_dict
                sample_counter += 1
            batch_counter += 1
            sample_counter = 0  # Reset sample counter for the next batch
    return merged


def flatten_batches_to_single_batch(tws: dict) -> dict:
    """Flatten every batch of a time-window dict into a single ``batch_0``.

    Takes a dict shaped ``{batch_key: {sample_key: sample}}`` (any number of
    batches) and returns ``{"batch_0": {sample_0: ..., sample_1: ..., ...}}``
    where all samples from all input batches are pooled in order and
    renumbered contiguously. Batch boundaries are **dissolved**.

    Use this when downstream code expects exactly one batch (e.g. the
    prediction/evaluation utilities that read ``batch["batch_0"]``).

    Parameters
    ----------
    tws : dict
        A time-window dict with one or more batches.

    Returns
    -------
    dict
        A dict with a single ``batch_0`` key containing every input sample.
    """
    merged = {"batch_0": {}}
    sample_counter = 0

    for batch in tws.values():
        for sample in batch.values():
            merged["batch_0"][f"sample_{sample_counter}"] = sample
            sample_counter += 1
    return merged

def rename_batches_by_start_time(tws: dict, fmt: str = "%Y-%m-%dT%H:%M") -> dict:
    renamed = {}
    for batch in tws.values():
        sample = batch["sample_0"]
        start_time = sample["fitting_hour_range"][0]
        new_key = start_time.strftime(fmt)
        renamed[new_key] = batch
    return renamed

def split_tws_train_val(tws, val_ratio=0.2):
      keys = list(tws["batch_0"].keys())
      n_val = max(1, int(len(keys) * val_ratio))
      train_keys = keys[:-n_val]
      val_keys   = keys[-n_val:]
      tws_train = {"batch_0": {k: tws["batch_0"][k] for k in train_keys}}
      tws_val   = {"batch_0": {k: tws["batch_0"][k] for k in val_keys}}
      return tws_train, tws_val

def print_data_structure(tws, split="train"):
    if split in tws.keys():
        n_train_batches = len(tws[split].keys())
        n_samples_per_batch = [len(tws[split][batch]) for batch in tws[split].keys()]
        print(f"number of batches: {n_train_batches},\nsamples per batch: {n_samples_per_batch}")
    else: raise KeyError(f">{split}< is not in tws")
