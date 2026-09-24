"""Abstract base for GA problems used by the identification pipeline.

Subclasses fill four hooks (process_data, predict, get_truth, compute_dynamic_constraints).
``_evaluate`` calls them in order, caches y_pred on ``self._last_y_pred`` so
subclasses can reuse it (e.g. for DR weighting) without re-simulating, and
writes ``out["F"]`` and ``out["G"]``.
"""
from abc import abstractmethod

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from rctoolbox.identification.parameter_space import BaseParameterSpace


class BaseProblem(ElementwiseProblem):
    REQUIRED_KEYS: tuple[str, ...] = ()
    OPTIONAL_KEYS: tuple[str, ...] = ()

    def __init__(
        self,
        model,
        parameter_space: BaseParameterSpace,
        data: dict,
        objective_function,
        n_constr: int,
    ):
        super().__init__(
            n_var=parameter_space.n_free_params,
            n_obj=1,
            n_constr=n_constr,
            xl=parameter_space.xl,
            xu=parameter_space.xu,
        )
        self._validate_data_keys(data)
        self.model = model
        self.parameter_space = parameter_space
        self.data = data
        self.objective_function = objective_function
        self._last_y_pred = None
        self.process_data()

    def _validate_data_keys(self, data: dict) -> None:
        missing = set(self.REQUIRED_KEYS) - set(data.keys())
        if missing:
            raise KeyError(
                f"{type(self).__name__} requires data keys "
                f"{sorted(missing)} which are missing."
            )

    @abstractmethod
    def process_data(self) -> None: ...

    @abstractmethod
    def predict(self, params: dict, data: dict) -> np.ndarray: ...

    @abstractmethod
    def get_truth(self, data: dict) -> np.ndarray: ...

    @abstractmethod
    def compute_dynamic_constraints(self, params: dict) -> list[float]: ...

    def _evaluate(self, x, out, *args, **kwargs):
        params = dict(zip(self.parameter_space.free_param_names, x))
        y_pred = self.predict(params, self.data)
        y_true = self.get_truth(self.data)
        self._last_y_pred = y_pred
        out["F"] = self.objective_function(y_true, y_pred)
        constraints = self.compute_dynamic_constraints(params)
        if constraints:
            out["G"] = -np.array(constraints)
