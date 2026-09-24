"""Build the per-batch data dicts for calibrating a 2R2C model.

``build_data_dict``             -> TemperatureProblem (temperature task)
``build_data_dict_heat_demand`` -> HeatDemandProblem (setpoint-to-power task)

State vector for 2R2C is [T_i, T_e]; ``y_0`` and ``y_0_`` (lag) are
pre-computed here with ``T_e = T_i - 2`` (an ad-hoc cooler-mass offset;
the static-inversion / regime-aware initializers refine T_e at run time).
"""
import numpy as np


def build_data_dict(dataloader, split_name, Phi_HC_col):
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]

    T_i  = di["y1"]
    T_i0 = T_i[:, 0]

    # T_a0 = di["T_a"][:, 0]
    # T_e0 = 0.5 * (T_i0 + T_a0)
    T_e0 = T_i0 - 2
    y_0  = np.stack([T_i0, T_e0], axis=1)

    T_i_ = di["y1_1Dlag"]
    T_i0_ = T_i_[:, 0]
    # T_a0_ = di["T_a_1Dlag"][:, 0]
    # T_e0_ = 0.5 * (T_i0_ + T_a0_)
    T_e0_ = T_i0_ - 2
    y_0_ = np.stack([T_i0_, T_e0_], axis=1)

    return {
        "t":       b["t_array"],
        "T_i":     T_i,
        "T_a":     di["T_a"],
        "Phi_sol": di["Phi_sol"],
        "Phi_HC":  di[Phi_HC_col],
        "data_for_init_state": {
            "y_0":      y_0,      "y_0_":     y_0_,
            "t_":       di["t_array_1Dlag"],
            "T_i_":     T_i_,
            "T_a_":     di["T_a_1Dlag"],
            "Phi_sol_": di["Phi_sol_1Dlag"],
            "Phi_HC_":  di[f"{Phi_HC_col}_1Dlag"],
        },
    }


def build_data_dict_heat_demand(dataloader, split_name, Phi_HC_col="Phi_HC", y1_set_col="y1_set"):
    """Data dict for the inverse (setpoint-to-power) 2R2C identification.

    Mirrors five_r1c_eniso13790.build_data_dict — same keys, 2-state y_0.
    """
    b  = dataloader.batches[split_name]["batch_0"]
    di = b["data_inputs"]

    T_i  = di["y1"]
    T_i0 = T_i[:, 0]
    # T_a0 = di["T_a"][:, 0]
    # T_e0 = 0.5 * (T_i0 + T_a0)
    T_e0 = T_i0 - 2
    y_0  = np.stack([T_i0, T_e0], axis=1)

    T_i_ = di["y1_1Dlag"]
    T_i0_ = T_i_[:, 0]
    # T_a0_ = di["T_a_1Dlag"][:, 0]
    # T_e0_ = 0.5 * (T_i0_ + T_a0_)
    T_e0_ = T_i0_ - 2
    y_0_ = np.stack([T_i0_, T_e0_], axis=1)

    return {
        "t":        b["t_array"],
        "T_i":      T_i,
        "T_a":      di["T_a"],
        "Phi_sol":  di["Phi_sol"],
        "Phi_HC":   di[Phi_HC_col],
        "Phi_HC_0": di[Phi_HC_col][:, 0],
        "T_i_set":  di[y1_set_col],
        "DR":       di.get("DR"),
        "controlled_flag": di.get("controlled_flag"),
        "data_for_init_state": {
            "y_0":      y_0,       "y_0_":     y_0_,
            "t_":       di["t_array_1Dlag"],
            "T_i_":     T_i_,
            "T_a_":     di["T_a_1Dlag"],
            "Phi_sol_": di["Phi_sol_1Dlag"],
            "Phi_HC_":  di[f"{Phi_HC_col}_1Dlag"],
            "T_i_set_": di[f"{y1_set_col}_1Dlag"],
        },
    }
