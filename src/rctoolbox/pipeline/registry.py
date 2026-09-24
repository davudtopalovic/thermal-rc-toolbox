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

def get_plugin(model_type: str) -> ModelPlugin:
    try:
        return MODEL_REGISTRY[model_type]
    except KeyError as e:
        raise KeyError(
            f"Unknown model_type {model_type!r}. "
            f"Registered model_types: {sorted(MODEL_REGISTRY)}"
        ) from e
