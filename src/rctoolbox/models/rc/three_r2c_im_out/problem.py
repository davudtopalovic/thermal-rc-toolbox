"""Identification problem for models driven by T_a, T_out and Phi_HC
(``ThreeR2CInternalMassOut``, ``FiveR4CInternalMass``).

Same as the 1R1C ``TemperatureProblem`` (only the states listed in
``output_states`` are supervised, default the measured air T_i), with the
outdoor temperature ``T_out`` as an additional required input.

The inverse (heat-demand) problem is not implemented for these models.
"""
import numpy as np

from rctoolbox.models.rc.one_r1c.problem import TemperatureProblem as _TemperatureProblem1R1C


class TemperatureProblem(_TemperatureProblem1R1C):
    REQUIRED_KEYS = (
        "t", "T_a", "T_out", "Phi_HC", "data_for_init_state",
    )

    def process_data(self):
        super().process_data()
        self.T_out = self.data["T_out"]

    def predict(self, params: dict, data: dict) -> np.ndarray:
        self.model.set_params(params)
        y_0 = self.model.get_initial_state(data=data["data_for_init_state"])
        y_pred = self.model.forecast_temperature(
            y_0, data["t"], data["T_a"], data["Phi_HC"], data["T_out"],
        )
        if len(self.output_states) == 1:
            return y_pred[:, :, self.output_states[0][0]]
        return np.stack(
            [y_pred[:, :, idx] for idx, _ in self.output_states], axis=-1,
        )
