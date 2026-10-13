"""Identification problem shared by all freezer models.

Same contract as the 1R1C ``TemperatureProblem``: the model is simulated over
each window with the known inputs and the supervised states listed in
``output_states`` (default: the measured air ``T_i``) are compared with the
truth. The inputs are passed by name (``model.all_inputs``), so one problem
class serves every freezer model.
"""
import numpy as np

from rctoolbox.identification.base_problem import BaseProblem


class FreezerTemperatureProblem(BaseProblem):
    REQUIRED_KEYS = ("t", "data_for_init_state")
    N_CONSTR = 0

    def __init__(self, model, parameter_space, data, objective_function, n_constr, output_states=None):
        if output_states is None:
            output_states = [(model.T_i_idx, "T_i")]
        self.output_states = [tuple(x) for x in output_states]
        missing = [k for k in model.all_inputs if k not in data] + \
                  [k for _, k in self.output_states if k not in data]
        if missing:
            raise KeyError(f"{type(self).__name__}: missing data keys {missing}")
        super().__init__(model=model, parameter_space=parameter_space, data=data,
                         objective_function=objective_function, n_constr=n_constr)

    def process_data(self):
        self.t = self.data["t"]

    def predict(self, params: dict, data: dict) -> np.ndarray:
        self.model.set_params(params)
        y_0 = self.model.get_initial_state(data=data["data_for_init_state"])
        y = self.model.forecast_temperature(y_0, data["t"], **{k: data[k] for k in self.model.all_inputs})
        if len(self.output_states) == 1:
            return y[:, :, self.output_states[0][0]]
        return np.stack([y[:, :, i] for i, _ in self.output_states], axis=-1)

    def get_truth(self, data: dict) -> np.ndarray:
        if len(self.output_states) == 1:
            return data[self.output_states[0][1]]
        return np.stack([data[k] for _, k in self.output_states], axis=-1)

    def compute_dynamic_constraints(self, params: dict) -> list:
        return []
