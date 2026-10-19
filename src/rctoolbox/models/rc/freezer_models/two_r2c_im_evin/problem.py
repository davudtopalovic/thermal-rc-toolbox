"""Identification problem for the standalone 2R2C internal-mass model driven
by the measured evaporator temperature.

- ``TemperatureProblem`` — T_a, T_ev and the on-fraction are known inputs; the
  predicted air temperature is compared against the measured ``"T_i"``.
  ``output_states`` lists the ``(state_index, truth_key)`` pairs to supervise;
  the default is ``[(0, "T_i")]`` (T_m is hidden).

The inverse (heat-demand) problem is not implemented for this model.
"""
import numpy as np

from rctoolbox.identification.base_problem import BaseProblem


class TemperatureProblem(BaseProblem):
    REQUIRED_KEYS = (
        "t", "T_a", "T_ev", "on", "data_for_init_state",
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
        self.T_ev = self.data["T_ev"]
        self.on = self.data["on"]
        self.data_for_init_state = self.data["data_for_init_state"]
        for _, key in self.output_states:
            setattr(self, key, self.data[key])

    def predict(self, params: dict, data: dict) -> np.ndarray:
        self.model.set_params(params)
        y_0 = self.model.get_initial_state(data=data["data_for_init_state"])
        y_pred = self.model.forecast_temperature(
            y_0, data["t"], data["T_a"], data["T_ev"], data["on"],
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
