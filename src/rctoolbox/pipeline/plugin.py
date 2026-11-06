"""Model plugins: built-in (registry) or local (user code next to the config).

A plugin pairs a model factory with the matching ``BaseParameterSpace`` and
``BaseProblem`` subclasses. ``model_config`` selects it in one of two ways:

- ``"model_type": "<key>"`` — a model registered in
  ``rctoolbox.pipeline.registry.MODEL_REGISTRY``;
- ``"module": "<spec>"`` — user code, e.g. a model under development that
  lives in the training folder rather than in the toolbox. ``<spec>`` is

  * ``"plugin.py"`` — a single file (relative to the identification config,
    or absolute); its directory is put on ``sys.path`` so it can import
    sibling helpers, but only the file itself is copied into the results;
  * ``"my_model"`` — a package directory (with ``__init__.py``), relative or
    absolute; its modules can use relative imports (``from .model import X``)
    and the whole directory is copied into the results;
  * ``"package.module"`` — an importable module (nothing is copied);

  optionally followed by ``":NAME"`` for the attribute holding the plugin
  (default ``PLUGIN``). The module defines, for example::

      from rctoolbox.pipeline.plugin import ModelPlugin

      def build_model(building_config, model_config):
          model = MyModel()
          model.set_initialization(init_config=model_config["init_config"])
          return model

      PLUGIN = ModelPlugin(
          model_factory=build_model,
          parameter_space_cls=MyParameterSpace,
          problem_cls=MyProblem,
      )

  ``model_type`` may be given as well and is then only a label.

Local code is copied into the results folder (``snapshot_module``) and the
copied spec is what ``results.json`` records, so a run can be rebuilt from its
results folder alone (``rctoolbox.pipeline.run.load_run``).
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

# Prefix of the sys.modules names given to local modules. Classes pickled with
# the optimizer refer to these names; ``load_optimizer`` remaps them.
LOCAL_MODULE_PREFIX = "_rct_local_"
DEFAULT_PLUGIN_ATTR = "PLUGIN"


@dataclass(frozen=True)
class ModelPlugin:
    # (building_config, model_config) -> model
    model_factory: Callable[[dict, dict], object]
    parameter_space_cls: type
    problem_cls: type


def build_parameter_space(plugin: ModelPlugin, building_config: dict, model_config: dict):
    """The plugin's parameter space for this building, with the bounds of
    ``model_config["free_param_bounds"]`` ({name: [lower, upper]}) replacing
    the default/derived ones of the parameters listed there."""
    building_config = building_config or {}
    if "free_param_bounds" in building_config:
        raise ValueError(
            "free_param_bounds moved from the building config to the identification config: "
            'set model_config["free_param_bounds"] (only the parameters to change)'
        )
    parameter_space = plugin.parameter_space_cls.from_building_config(building_config)
    parameter_space.override_bounds(model_config.get("free_param_bounds"))
    return parameter_space


def _split_spec(spec: str) -> tuple[str, str]:
    target, sep, attr = spec.partition(":")
    return target, (attr if sep else DEFAULT_PLUGIN_ATTR)


def _local_path(target: str, base_dir) -> Path | None:
    """The file/directory ``target`` names, or None for an importable module name."""
    path = Path(target)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    if target.endswith(".py") or path.is_dir():
        return path.resolve()
    return None


def load_module(target: str, base_dir=None) -> ModuleType:
    """Import ``file.py``, a package directory, or an importable ``package.module``."""
    path = _local_path(target, base_dir)
    if path is None:
        return importlib.import_module(target)
    if not path.exists():
        raise FileNotFoundError(f"model module not found: {path}")

    name = f"{LOCAL_MODULE_PREFIX}{path.stem}_{hashlib.md5(str(path).encode()).hexdigest()[:8]}"
    if path.is_dir():
        init = path / "__init__.py"
        if not init.exists():
            raise FileNotFoundError(f"model package {path} has no __init__.py")
        module_spec = importlib.util.spec_from_file_location(
            name, init, submodule_search_locations=[str(path)]
        )
    else:
        # Siblings of the file (helpers it imports) must be importable.
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))
        module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    # Registered before exec so that relative imports and pickling work.
    sys.modules[name] = module
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def resolve_plugin(model_config: dict, base_dir=None) -> tuple[ModelPlugin, ModuleType | None]:
    """The plugin ``model_config`` selects, plus its module (None for registry models)."""
    spec = model_config.get("module")
    if spec:
        target, attr = _split_spec(spec)
        module = load_module(target, base_dir)
        try:
            plugin = getattr(module, attr)
        except AttributeError:
            raise AttributeError(
                f"model module {target!r} has no attribute {attr!r}; define "
                f"{attr} = ModelPlugin(model_factory=..., parameter_space_cls=..., "
                f"problem_cls=...)"
            ) from None
        missing = [a for a in ("model_factory", "parameter_space_cls", "problem_cls")
                   if not hasattr(plugin, a)]
        if missing:
            raise TypeError(f"{spec!r} is not a ModelPlugin (missing {missing})")
        return plugin, module

    model_type = model_config.get("model_type")
    if not model_type:
        raise ValueError(
            "model_config needs 'model_type' (a registered model) or 'module' "
            "(local model code, e.g. 'plugin.py')."
        )
    from rctoolbox.pipeline.registry import get_plugin
    return get_plugin(model_type), None


def snapshot_module(spec: str, base_dir, results_dir) -> tuple[str, str | None]:
    """Copy local model code into ``results_dir``.

    Returns the spec relative to ``results_dir`` and the absolute source path
    (None, and the spec unchanged, for an importable module).
    """
    target, sep, attr = spec.partition(":")
    src = _local_path(target, base_dir)
    if src is None:
        return spec, None
    results_dir = Path(results_dir).resolve()
    dst = results_dir / src.name
    if src.is_dir():
        if results_dir == src or src in results_dir.parents:
            raise ValueError(
                f"results directory {results_dir} lies inside the model package {src}"
            )
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    elif src != dst:
        shutil.copy2(src, dst)
    return src.name + (sep + attr if sep else ""), str(src)


class _LocalModuleUnpickler(pickle.Unpickler):
    """Unpickler that maps pickled local-module names onto ``module``.

    The name of a local module depends on its path, so an optimizer pickled
    from the training folder refers to a different name than the copy in the
    results folder (or a moved training folder) gets.
    """

    def __init__(self, file, module: ModuleType | None):
        super().__init__(file)
        self._module = module

    def find_class(self, module_name, name):
        if self._module is not None and module_name.startswith(LOCAL_MODULE_PREFIX):
            _, dot, rest = module_name.partition(".")
            module_name = self._module.__name__ + (dot + rest if dot else "")
            if module_name not in sys.modules:
                importlib.import_module(module_name)
        return super().find_class(module_name, name)


def load_optimizer(optimizer_path, module: ModuleType | None = None):
    """Unpickle ``optimizer.pkl``; ``module`` is the local model module, if any."""
    with open(optimizer_path, "rb") as f:
        return _LocalModuleUnpickler(f, module).load()
