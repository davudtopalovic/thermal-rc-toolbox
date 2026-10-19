"""Termination criteria for the GA identification.

``FitnessPlateauTermination`` stops the run when the best fitness has stopped
improving: the improvement over the last ``patience`` generations is below
``tol``. The improvement is measured over the whole window, not per
generation, so a run that keeps improving slowly is not stopped. ``n_max_gen``
is always an upper limit.

Configured through ``optimization_config``:

    {"pop_size": 100, "n_gen": 300,
     "early_stopping": {"patience": 20, "tol": 1e-3, "relative": False, "min_gen": 20}}

``tol`` is in the units of the objective (e.g. K for an RMSE of temperatures);
with ``relative=True`` it is a fraction of the current best fitness. Without
``early_stopping`` the GA runs for exactly ``n_gen`` generations.
"""
from pymoo.core.termination import Termination


class FitnessPlateauTermination(Termination):

    def __init__(self, n_max_gen, patience=20, tol=1e-3, relative=False, min_gen=0):
        super().__init__()
        if patience < 1:
            raise ValueError("early_stopping.patience must be >= 1")
        if tol < 0:
            raise ValueError("early_stopping.tol must be >= 0")
        self.n_max_gen = int(n_max_gen)
        self.patience = int(patience)
        self.tol = float(tol)
        self.relative = bool(relative)
        self.min_gen = int(min_gen)
        self.best_hist = []       # best fitness after each generation
        self.stop_reason = None

    @classmethod
    def from_config(cls, n_max_gen, early_stopping: dict):
        unknown = set(early_stopping) - {"patience", "tol", "relative", "min_gen"}
        if unknown:
            raise ValueError(f"Unknown early_stopping keys {sorted(unknown)}; "
                             "valid: patience, tol, relative, min_gen")
        return cls(n_max_gen, **early_stopping)

    def _update(self, algorithm):
        F = algorithm.opt.get("F") if algorithm.opt is not None else None
        best = float(F.min()) if F is not None and len(F) else float("inf")
        self.best_hist.append(best)
        n = algorithm.n_gen

        if n >= self.n_max_gen:
            self.stop_reason = f"reached n_gen = {self.n_max_gen}"
            return 1.0
        if n >= max(self.min_gen, self.patience + 1) and len(self.best_hist) > self.patience:
            gain = self.best_hist[-self.patience - 1] - best
            limit = self.tol * abs(best) if self.relative else self.tol
            if gain < limit:
                self.stop_reason = (f"best fitness improved by {gain:.3g} (< {limit:.3g}) over the last "
                                    f"{self.patience} generations")
                return 1.0
        return n / self.n_max_gen
