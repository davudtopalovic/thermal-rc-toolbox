"""Batch prediction helper of ``Freezer1R1C``."""
from rctoolbox.models.rc.freezer_models._base import predict_for_batch as _predict


def predict_for_batch(model, batch, raw_dfs, column_map=None):
    return _predict(model, batch, raw_dfs, column_map)
