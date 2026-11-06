"""Identification problems for the 3R2C model.

Two views of the same model, both built on ``BaseProblem``:

- ``TemperatureProblem`` — Phi_HC is a known forcing; predicted state(s) are
  compared against measured state(s). ``output_states`` lists the
  ``(state_index, truth_key)`` pairs to supervise; the default is the
  indoor-air node against ``"T_i"``.
- ``HeatDemandProblem`` — the inverse (setpoint-to-power) view; the predicted
  heat-power demand is compared against the measured ``Phi_HC``.

The direct path R_ia and the gain parameters are only box-bounded.
"""
import numpy as np

from rctoolbox.identification.base_problem import BaseProblem


class TemperatureProblem(BaseProblem):
    REQUIRED_KEYS = (
        "t", "T_a", "Phi_sol", "Phi_HC", "data_for_init_state",
    )
    OPTIONAL_KEYS = ()

    N_CONSTR = 0

    def __init__(
        self,
        model,
        parameter_space,
        data,
        objective_function,
        n_constr,
        output_states=None,
    ):
        if output_states is None:
            output_states = [(model.T_i_idx, "T_i")]
        self.output_states = [tuple(x) for x in output_states]

        missing = [k for _, k in self.output_states if k not in data]
        if missing:
            raise KeyError(
                f"{type(self).__name__} output_states reference missing data "
                f"keys: {missing}. Available keys: {sorted(data.keys())}"
            )
        super().__init__(
            model=model,
            parameter_space=parameter_space,
            data=data,
            objective_function=objective_function,
            n_constr=n_constr,
        )

    # ------------------------------------------------------------------ hooks
    def process_data(self):
        self.t = self.data["t"]
        self.T_a = self.data["T_a"]
        self.Phi_sol = self.data["Phi_sol"]
        self.Phi_HC = self.data["Phi_HC"]
        self.data_for_init_state = self.data["data_for_init_state"]
        # Cache each truth array under its own attribute name for convenience
        # (plotting, post-hoc inspection).
        for _, key in self.output_states:
            setattr(self, key, self.data[key])

    def predict(self, params: dict, data: dict) -> np.ndarray:
        self.model.set_params(params)
        y_0 = self.model.get_initial_state(data=data["data_for_init_state"])
        y_pred = self.model.forecast_temperature(
            y_0, data["t"], data["T_a"], data["Phi_sol"], data["Phi_HC"],
        )
        if len(self.output_states) == 1:
            return y_pred[:, :, self.output_states[0][0]]
        return np.stack(
            [y_pred[:, :, idx] for idx, _ in self.output_states], axis=-1,
        )

    def get_truth(self, data: dict) -> np.ndarray:
        if len(self.output_states) == 1:
            return data[self.output_states[0][1]]
        return np.stack(
            [data[key] for _, key in self.output_states], axis=-1,
        )

    def compute_dynamic_constraints(self, params: dict) -> list[float]:
        return []


class HeatDemandProblem(BaseProblem):
    REQUIRED_KEYS = (
        "t", "T_a", "Phi_sol", "Phi_HC", "Phi_HC_0",
        "T_i_set", "data_for_init_state",
    )
    OPTIONAL_KEYS = ("DR",)

    N_CONSTR = 2

    # ------------------------------------------------------------------ hooks
    def process_data(self):
        self.t = self.data["t"]
        self.T_a = self.data["T_a"]
        self.Phi_sol = self.data["Phi_sol"]
        self.Phi_HC = self.data["Phi_HC"]
        self.Phi_HC_0 = self.data["Phi_HC_0"]
        self.T_i_set = self.data["T_i_set"]
        self.data_for_init_state = self.data["data_for_init_state"]
        # Kept for backwards-compat with code that reads it off the problem.
        self.T_i = self.data.get("T_i")

        self.DR = self.data.get("DR")
        if self.DR is not None and not np.any(self.DR != 0):
            # No DR events in this batch -> fall back to uniform objective.
            self.DR = None

    def predict(self, params: dict, data: dict) -> np.ndarray:
        self.model.set_params(params)
        y_0 = self.model.get_initial_state(data=data["data_for_init_state"])
        Phi_HC_pred, _ = self.model.forecast_heat_demand_fast(
            y_0,
            data["Phi_HC_0"],
            data["t"],
            data["T_a"],
            data["Phi_sol"],
            data["T_i_set"],
        )
        return Phi_HC_pred

    def get_truth(self, data: dict) -> np.ndarray:
        return data["Phi_HC"]

    def compute_dynamic_constraints(self, params: dict) -> list[float]:
        """Ordering of the two nodes: the envelope is the slow, massive one
        (C_e >= C_i) and its ambient path the resistive one (R_ea >= R_ie).
        Each entry must be >= 0 for a feasible candidate."""
        return [
            params["C_e"] - params["C_i"],
            params["R_ea"] - params["R_ie"],
        ]
