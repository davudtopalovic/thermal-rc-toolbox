"""Registry of (Model, ParameterSpace, Problem) plugins for the GA pipeline.

Each entry pairs a model factory (so plugin-specific setup like
``set_heat_power_range`` stays with the plugin, not in the runner) with the
matching ``BaseParameterSpace`` and ``BaseProblem`` subclasses. The runner
selects an entry by the ``model_type`` field in the identification config.
"""
from dataclasses import dataclass
from typing import Callable


# Each model folder ships its two identification problems in ``problem.py``:
# ``TemperatureProblem`` (Phi_HC known, states supervised) and
# ``HeatDemandProblem`` (setpoint-to-power, demand supervised).

# --- 5R1C EN ISO 13790
from rctoolbox.models.rc.five_r1c_eniso13790.model import FiveR1C_ENISO13790
from rctoolbox.models.rc.five_r1c_eniso13790.parameter_space import RCParameterSpace
from rctoolbox.models.rc.five_r1c_eniso13790 import problem as problem_5r1c

# --- 6R3C EN ISO 13790
from rctoolbox.models.rc.six_r3c_eniso13790.model import SixR3C_ENISO13790
from rctoolbox.models.rc.six_r3c_eniso13790.parameter_space import ParameterSpace_6R3C_ENISO13790
from rctoolbox.models.rc.six_r3c_eniso13790 import problem as problem_6r3c


# --- 3R3C
from rctoolbox.models.rc.three_r3c.model import ThreeR3C
from rctoolbox.models.rc.three_r3c.parameter_space import ParameterSpace_3R3C
from rctoolbox.models.rc.three_r3c import problem as problem_3r3c

# --- 1R1C
from rctoolbox.models.rc.one_r1c.model import OneR1C
from rctoolbox.models.rc.one_r1c.parameter_space import ParameterSpace_1R1C
from rctoolbox.models.rc.one_r1c import problem as problem_1r1c

# --- 2R2C with internal mass (freezer / cold store)
from rctoolbox.models.rc.two_r2c_im.model import TwoR2CInternalMass
from rctoolbox.models.rc.two_r2c_im.parameter_space import ParameterSpace_2R2C_IM
from rctoolbox.models.rc.two_r2c_im import problem as problem_2r2c_im

# --- 3R2C with internal mass and outdoor link (freezer / cold store)
from rctoolbox.models.rc.three_r2c_im_out.model import ThreeR2CInternalMassOut
from rctoolbox.models.rc.three_r2c_im_out.parameter_space import ParameterSpace_3R2C_IM_OUT
from rctoolbox.models.rc.three_r2c_im_out import problem as problem_3r2c_im_out

# --- 5R4C with two envelope nodes and internal mass (freezer / cold store)
from rctoolbox.models.rc.five_r4c_im.model import FiveR4CInternalMass
from rctoolbox.models.rc.five_r4c_im.parameter_space import ParameterSpace_5R4C_IM
from rctoolbox.models.rc.five_r4c_im import problem as problem_5r4c_im

# --- 2R2C with internal mass, driven by the measured evaporator temperature (freezer, standalone)
from rctoolbox.models.rc.freezer_models.two_r2c_im_evin.model import TwoR2CInternalMassEvIn
from rctoolbox.models.rc.freezer_models.two_r2c_im_evin.parameter_space import ParameterSpace_2R2C_IM_EVIN
from rctoolbox.models.rc.freezer_models.two_r2c_im_evin import problem as problem_2r2c_im_evin

# --- freezer models (configurations of freezer_models.network.FreezerNetwork)
import importlib as _importlib
from rctoolbox.models.rc.freezer_models._problem import FreezerTemperatureProblem

FREEZER_PACKAGES = {
    "fz_1R1C_temperature":            ("fz_1r1c", "Freezer1R1C"),
    "fz_2R2C_im_temperature":         ("fz_2r2c_im", "Freezer2R2C_IM"),
    "fz_3R3C_im_ev_temperature":      ("fz_3r3c_im_ev", "Freezer3R3C_IM_EV"),
    "fz_3R3C_im_ev_ta_temperature":   ("fz_3r3c_im_ev_ta", "Freezer3R3C_IM_EV_TA"),
    "fz_3R3C_im_ev_out_temperature":  ("fz_3r3c_im_ev_out", "Freezer3R3C_IM_EV_OUT"),
    "fz_3R3C_im_ev_cop_temperature":  ("fz_3r3c_im_ev_cop", "Freezer3R3C_IM_EV_COP"),
    "fz_1R1C_evin_temperature":       ("fz_1r1c_evin", "Freezer1R1C_EVIN"),
    "fz_2R2C_im_evin_temperature":    ("fz_2r2c_im_evin", "Freezer2R2C_IM_EVIN"),
}

# --- 2R2C
from rctoolbox.models.rc.two_r2c.model import TwoR2C
from rctoolbox.models.rc.two_r2c.parameter_space import ParameterSpace_2R2C
from rctoolbox.models.rc.two_r2c import problem as problem_2r2c

# --- 3R2C (with gains)
from rctoolbox.models.rc.three_r2c.model import ThreeR2C
from rctoolbox.models.rc.three_r2c.parameter_space import RC_3R2CParameterSpace
from rctoolbox.models.rc.three_r2c import problem as problem_3r2c


@dataclass(frozen=True)
class ModelPlugin:
    # (building_config, model_config) -> model
    model_factory: Callable[[dict, dict], object]
    parameter_space_cls: type
    problem_cls: type

def build_5r1c_model(building_config: dict, model_config: dict):
    model = FiveR1C_ENISO13790()
    model.set_heat_power_range([0, building_config["max_HP_power"]])
    model.set_initialization(init_config=model_config["init_config"])
    return model

def build_1r1c_temp_model(building_config: dict, model_config: dict):
    model = OneR1C()
    if "init_config" in model_config:
        model.set_initialization(init_config=model_config["init_config"])
    model.set_integration(model_config.get("integration"))
    return model

def build_2r2c_im_temp_model(building_config: dict, model_config: dict):
    model = TwoR2CInternalMass()
    if "init_config" in model_config:
        model.set_initialization(init_config=model_config["init_config"])
    model.set_integration(model_config.get("integration"))
    return model

def build_3r2c_im_out_temp_model(building_config: dict, model_config: dict):
    model = ThreeR2CInternalMassOut()
    if "init_config" in model_config:
        model.set_initialization(init_config=model_config["init_config"])
    model.set_integration(model_config.get("integration"))
    return model

def build_5r4c_im_temp_model(building_config: dict, model_config: dict):
    model = FiveR4CInternalMass()
    if "init_config" in model_config:
        model.set_initialization(init_config=model_config["init_config"])
    model.set_integration(model_config.get("integration"))
    return model

def build_2r2c_im_evin_temp_model(building_config: dict, model_config: dict):
    model = TwoR2CInternalMassEvIn()
    if "init_config" in model_config:
        model.set_initialization(init_config=model_config["init_config"])
    model.set_integration(model_config.get("integration"))
    return model

def _freezer_factory(model_cls):
    def build(building_config: dict, model_config: dict):
        model = model_cls()
        if "init_config" in model_config:
            model.set_initialization(init_config=model_config["init_config"])
        model.set_integration(model_config.get("integration", "zoh"))
        return model
    return build


def _freezer_plugins():
    plugins = {}
    for key, (pkg, cls_name) in FREEZER_PACKAGES.items():
        base = f"rctoolbox.models.rc.freezer_models.{pkg}"
        model_cls = getattr(_importlib.import_module(f"{base}.model"), cls_name)
        ps_cls = getattr(_importlib.import_module(f"{base}.parameter_space"), f"ParameterSpace_{cls_name}")
        plugins[key] = (model_cls, ps_cls)
    return plugins


def build_2r2c_temp_model(building_config: dict, model_config: dict):
    model = TwoR2C()
    if "init_config" in model_config:
        model.set_initialization(init_config=model_config["init_config"])
    model.set_integration(model_config.get("integration"))
    return model

def build_2r2c_heat_demand_model(building_config: dict, model_config: dict):
    model = TwoR2C()
    model.set_heat_power_range([0, building_config["max_HP_power"]])
    model.set_initialization(init_config=model_config.get("init_config", {"method": "default"}))
    model.set_integration(model_config.get("integration"))
    return model

def build_6r3c_model(building_config: dict, model_config: dict):
    model = SixR3C_ENISO13790()
    if "init_config" in model_config:
        model.set_initialization(init_config=model_config["init_config"])
    return model

def build_3r3c_model(building_config: dict, model_config: dict):
    model = ThreeR3C()
    if "init_config" in model_config:
        model.set_initialization(init_config=model_config["init_config"])
    return model

# def build_6r2c_model(building_config: dict, model_config: dict):
#     model = SixR2C_ENISO13790()
#     model.set_heat_power_range([0, building_config["max_HP_power"]])
#     model.set_initialization(init_config=model_config.get("init_config", {"method": "default"}))
#     return model

def build_3r2c_heat_demand_model(building_config: dict, model_config: dict):
    model = ThreeR2C()
    model.set_heat_power_range([0, building_config["max_HP_power"]])
    model.set_initialization(init_config=model_config.get("init_config", {"method": "default"}))
    model.set_integration(model_config.get("integration"))
    return model


MODEL_REGISTRY: dict[str, ModelPlugin] = {
    "5R1C_heat_demand": ModelPlugin(
        model_factory=build_5r1c_model,
        parameter_space_cls=RCParameterSpace,
        problem_cls=problem_5r1c.HeatDemandProblem,
    ),
    "1R1C_temperature": ModelPlugin(
        model_factory=build_1r1c_temp_model,
        parameter_space_cls=ParameterSpace_1R1C,
        problem_cls=problem_1r1c.TemperatureProblem,
    ),
    "2R2C_im_temperature": ModelPlugin(
        model_factory=build_2r2c_im_temp_model,
        parameter_space_cls=ParameterSpace_2R2C_IM,
        problem_cls=problem_2r2c_im.TemperatureProblem,
    ),
    "3R2C_im_out_temperature": ModelPlugin(
        model_factory=build_3r2c_im_out_temp_model,
        parameter_space_cls=ParameterSpace_3R2C_IM_OUT,
        problem_cls=problem_3r2c_im_out.TemperatureProblem,
    ),
    "5R4C_im_temperature": ModelPlugin(
        model_factory=build_5r4c_im_temp_model,
        parameter_space_cls=ParameterSpace_5R4C_IM,
        problem_cls=problem_5r4c_im.TemperatureProblem,
    ),
    "2R2C_im_evin_temperature": ModelPlugin(
        model_factory=build_2r2c_im_evin_temp_model,
        parameter_space_cls=ParameterSpace_2R2C_IM_EVIN,
        problem_cls=problem_2r2c_im_evin.TemperatureProblem,
    ),
    "2R2C_temperature": ModelPlugin(
        model_factory=build_2r2c_temp_model,
        parameter_space_cls=ParameterSpace_2R2C,
        problem_cls=problem_2r2c.TemperatureProblem,
    ),
    "2R2C_heat_demand": ModelPlugin(
        model_factory=build_2r2c_heat_demand_model,
        parameter_space_cls=ParameterSpace_2R2C,
        problem_cls=problem_2r2c.HeatDemandProblem,
    ),

    "6R3C_temperature": ModelPlugin(
        model_factory=build_6r3c_model,
        parameter_space_cls=ParameterSpace_6R3C_ENISO13790,
        problem_cls=problem_6r3c.TemperatureProblem,
    ),
    "3R3C_temperature": ModelPlugin(
        model_factory=build_3r3c_model,
        parameter_space_cls=ParameterSpace_3R3C,
        problem_cls=problem_3r3c.TemperatureProblem,
    ),
    
    # "2R2C_heat_demand_unconstrained": ModelPlugin(
    #     model_factory=build_2r2c_heat_demand_model,
    #     parameter_space_cls=RC2R2CHeatDemandParameterSpace,
    #     problem_cls=HeatDemandProblemUnconstrained,
    # ),
    "3R2C_heat_demand": ModelPlugin(
        model_factory=build_3r2c_heat_demand_model,
        parameter_space_cls=RC_3R2CParameterSpace,
        problem_cls=problem_3r2c.HeatDemandProblem,
    ),
    # "3R2C_temperature": ModelPlugin(
    #     model_factory=build_3r2c_heat_demand_model,
    #     parameter_space_cls=RC_3R2CParameterSpace,
    #     problem_cls=problem_3r2c.TemperatureProblem,
    # ),
    # "6R2C_heat_demand": ModelPlugin(
    #     model_factory=build_6r2c_model,
    #     parameter_space_cls=RC6R2CParameterSpace,
    #     problem_cls=problem_6r2c.HeatDemandProblem,
    # )
}

for _key, (_cls, _ps) in _freezer_plugins().items():
    MODEL_REGISTRY[_key] = ModelPlugin(
        model_factory=_freezer_factory(_cls),
        parameter_space_cls=_ps,
        problem_cls=FreezerTemperatureProblem,
    )


def get_plugin(model_type: str) -> ModelPlugin:
    try:
        return MODEL_REGISTRY[model_type]
    except KeyError as e:
        raise KeyError(
            f"Unknown model_type {model_type!r}. "
            f"Registered model_types: {sorted(MODEL_REGISTRY)}"
        ) from e
