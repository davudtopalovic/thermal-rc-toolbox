import numpy as np
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, hvac_df):

        self.hvac_df = hvac_df

        inferred_freq = hvac_df.index.freq
        if inferred_freq == "h": inferred_freq = "1h"
        if inferred_freq is None: raise ValueError("Could not infer frequency from datetime index.")

        self.datetime_resolution = self.hvac_df.index.to_series().diff().dropna().mode()[0]
        self.dt_sec = self.datetime_resolution.total_seconds()

    def load_all_data_batches(self, time_windows, features, pre_window_lag):
        self.batches = {}
        self.raw_dfs = {}
        for split, batches in time_windows.items():
            split_batches, split_raw_dfs = self._prepare_data_split_batches(split, batches, features, pre_window_lag)
            self.batches[split] = split_batches
            self.raw_dfs[split] = split_raw_dfs

    # ------------ build batches for each split ------------
    def _prepare_data_split_batches(self, split, batch_time_windows, data_features, pre_window_lag):
        data_features = np.asarray(data_features)
        setattr(self, f"{split}_time_windows", batch_time_windows)
        if batch_time_windows is None:
            return

        setattr(self, f"num_{split}_batches", len(batch_time_windows))
        batches_dict = {}
        raw_dfs_dict = {}

        for batch_idx, sample_time_windows in batch_time_windows.items():

            first_sample_key = next(iter(sample_time_windows))

            # time vector
            t1, t2 = sample_time_windows[first_sample_key]["fitting_hour_range"]
            duration_sec = (t2 - t1).total_seconds()
            N = int(duration_sec / self.dt_sec)
            t_array = np.arange(0, N * self.dt_sec+1, step=self.dt_sec)

            # === Processed tensor inputs ===
            data_inputs = build_batch_ndarray_dict(
                sample_time_windows,
                self.hvac_df,
                self.dt_sec,
                data_features,
                pre_window_lag
            )

            # === Add time indices for each sample in the batch ===
            data_inputs["t_datetime"] = [self.hvac_df.loc[sample_time_windows[s]["fitting_hour_range"][0]:sample_time_windows[s]["fitting_hour_range"][1]].index 
                                       for s in sample_time_windows]
            
            batches_dict[batch_idx] = {"t_array": t_array,
                                       "data_inputs": data_inputs}

            # === Raw input DataFrames for debugging/inspection ===
            raw_dfs_dict[batch_idx] = {}
            for sample_key, sample in sample_time_windows.items():
                t1, t2 = sample["fitting_hour_range"]
                # t2 = t2 - self.datetime_resolution  # Align with ndarray slicing
                df_slice = self.hvac_df.loc[t1:t2,data_features].copy()

                # # for each feature append previous 24 hour observations
                # t1_ = t1-timedelta(days=1); t2_ = t1
                # for feature in data_features: df_slice[str(feature) + "_1Dlag"] = self.hvac_df.loc[t1_:t2_,feature].values
            
                raw_dfs_dict[batch_idx][sample_key] = df_slice

        return batches_dict, raw_dfs_dict


# -------------------  dict builder with normalized copies -------------------
def build_batch_ndarray_dict(sample_time_windows, df, dt_sec, data_features, pre_window_lag):
    
    data = {key: [] for key in data_features} | {f"{key}_1Dlag": [] for key in data_features}

    for _, value in sample_time_windows.items():
        t1 = value["fitting_hour_range"][0]
        t2 = value["fitting_hour_range"][1]
        
        for feature in data_features: 
            data[str(feature)].append(df.loc[t1:t2, feature].values)

        # 1D LAG FEATURES
        t1_ = t1-pre_window_lag; t2_ = t1
        
        # time vector
        duration_sec = (t2_ - t1_).total_seconds()
        N = int(duration_sec / dt_sec)
        t_array = np.arange(0, N * dt_sec+1, step=dt_sec)
        data["t_array_1Dlag"] = t_array

        # For each feature append previous 24 hour observations
        for feature in data_features:
            data[str(feature) + "_1Dlag"].append(df.loc[t1_:t2_, feature].values)
    
    ndarray = lambda x: np.stack(x, 0)
    out = {name2: ndarray(data_) for name2, data_ in data.items()}
    return out

# def create_interpolation_funcs(input_data_dict, t_array):
    # return {str(key): FastTimeSeriesInterpolator(t_array, data) for key, data in input_data_dict.items()}