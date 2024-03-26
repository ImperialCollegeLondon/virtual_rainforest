"""Module for all variables."""

import inspect
import pkgutil
from collections.abc import Hashable
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import cast

import tomli_w

import virtual_ecosystem.core.axes as axes
import virtual_ecosystem.core.base_model as base_model
from virtual_ecosystem.core.logger import LOGGER


@dataclass
class Variable:
    """Simulation variable, containing static and runtime metadata."""

    name: str
    """Name of the variable. Must be unique."""
    model_name: str
    """Name of the model defining the variable."""
    description: str
    """Description of what the variable represents."""
    units: str
    """Unites the variable should be represented in."""
    var_type: str
    """Type of the variable."""
    axis: tuple[str, ...]
    """Axes the variable is defined on."""
    initialised_by: str = field(default_factory=str, init=False)
    """Model that initialised the variable."""
    required_init_by: list[str] = field(default_factory=list, init=False)
    """Models that requires the variable to be initialised."""
    updated_by: list[str] = field(default_factory=list, init=False)
    """Models that update the variable."""
    required_update_by: list[str] = field(default_factory=list, init=False)
    """Models that use the variable."""

    def __post_init__(self) -> None:
        """Register the variable in the known variables.

        Raises:
            ValueError: If a variable is already in the known variables registry.
        """
        if self.name in KNOWN_VARIABLES:
            raise ValueError(
                f"Variable {self.name} already in the known variables registry."
            )

        LOGGER.info(f"Variable registered for model '{self.model_name}': {self.name}")
        KNOWN_VARIABLES[self.name] = self


RUN_VARIABLES_REGISTRY: dict[str, Variable] = {}
"""The global registry of variables used in a run."""

KNOWN_VARIABLES: dict[str, Variable] = {}
"""The global known variable registry."""


def register_all_variables() -> None:
    """Registers all variables provided by the models."""
    import virtual_ecosystem.models as models

    for mod in pkgutil.iter_modules(models.__path__):
        if not mod.ispkg:
            continue

        try:
            import_module(f"{models.__name__}.{mod.name}.variables")
        except ImportError:
            LOGGER.debug(
                f"No 'variables' module found for model {models.__name__}.{mod.name}."
            )


def _discover_all_variables_usage() -> None:
    """Discover the usage of variables in the models."""
    import virtual_ecosystem.models as models

    models_found = []
    for mod in pkgutil.iter_modules(models.__path__):
        if not mod.ispkg:
            continue

        try:
            module = import_module(f"{models.__name__}.{mod.name}.{mod.name}_model")
        except ImportError:
            LOGGER.warning(
                f"No model file found for model {models.__name__}.{mod.name}."
            )
            continue

        models_found.extend(
            [
                obj
                for _, obj in inspect.getmembers(module)
                if inspect.isclass(obj)
                and issubclass(obj, base_model.BaseModel)
                and obj is not base_model.BaseModel
            ]
        )

    setup_variables(models_found, [])


def output_known_variables(output_file: Path) -> None:
    """Output the known variables to a file.

    For the variables to be output, the variables must be registered and the usage of
    the variables must be discovered, assigning the appropriate models to the variables.

    Args:
        output_file: The file to output the known variables to.
    """
    register_all_variables()
    _discover_all_variables_usage()
    vars = {
        var.name: asdict(var)
        for var in sorted(KNOWN_VARIABLES.values(), key=lambda x: x.name)
    }
    with open(output_file, "bw") as f:
        tomli_w.dump(vars, f)


def _collect_initialise_by_vars(models: list[type[base_model.BaseModel]]) -> None:
    """Initialise the runtime variable registry.

    Args:
        models: The list of models that are initialising the variables.

    Raises:
        ValueError: If a variable required by a model is not in the known variables
            registry or if it is already initialised by another model.
    """
    for model in models:
        for var in model.vars_initialised:
            if var not in KNOWN_VARIABLES:
                raise ValueError(
                    f"Variable {var} required by {model.model_name} is not in the"
                    " known variables registry."
                )
            if var in RUN_VARIABLES_REGISTRY:
                raise ValueError(
                    f"Variable {var} already in registry, initialised by"
                    f"{RUN_VARIABLES_REGISTRY[var].initialised_by}."
                )

            KNOWN_VARIABLES[var].initialised_by = model.model_name
            RUN_VARIABLES_REGISTRY[var] = KNOWN_VARIABLES[var]


def _collect_updated_by_vars(models: list[type[base_model.BaseModel]]) -> None:
    """Verify that all variables updated by models are in the runtime registry.

    Args:
        models: The list of models to check.

    Raises:
        ValueError: If a variable required by a model is not in the known variables
            registry or the runtime registry.
    """
    for model in models:
        for var in model.vars_updated:
            if var not in KNOWN_VARIABLES:
                raise ValueError(
                    f"Variable {var} required by {model.model_name} is not in the known"
                    " variables registry."
                )
            if var not in RUN_VARIABLES_REGISTRY:
                raise ValueError(
                    f"Variable {var} required by {model.model_name} is not initialised"
                    " by any model."
                )
            if len(RUN_VARIABLES_REGISTRY[var].updated_by):
                LOGGER.warning(
                    f"Variable {var} updated by {model.model_name} is already updated"
                    f" by {RUN_VARIABLES_REGISTRY[var].updated_by}."
                )
            RUN_VARIABLES_REGISTRY[var].updated_by.append(model.model_name)


def _collect_required_update_vars(models: list[type[base_model.BaseModel]]) -> None:
    """Verify that all variables required by the update methods are in the registry.

    Args:
        models: The list of models to check.

    Raises:
        ValueError: If a variable required by a model is not in the known variables
            registry or the runtime registry.
    """
    for model in models:
        for var in model.required_update_vars:
            if var not in KNOWN_VARIABLES:
                raise ValueError(
                    f"Variable {var} required by {model.model_name} is not in the known"
                    " variables registry."
                )
            if var not in RUN_VARIABLES_REGISTRY:
                raise ValueError(
                    f"Variable {var} required by {model.model_name} is not initialised"
                    " by any model neither provided as input."
                )
            RUN_VARIABLES_REGISTRY[var].required_update_by.append(model.model_name)


def _collect_required_init_vars(models: list[type[base_model.BaseModel]]) -> None:
    """Verify that all variables required by the init methods are in the registry.

    Args:
        models: The list of models to check.

    Raises:
        ValueError: If a variable required by a model is not in the known variables
            registry or the runtime registry.
    """
    for model in models:
        for v in model.required_init_vars:
            # TODO In the future, var will be a string, so this won't be necessary
            var = v[0]
            if var not in KNOWN_VARIABLES:
                raise ValueError(
                    f"Variable {var} required by {model.model_name} is not in the known"
                    " variables registry."
                )
            if var not in RUN_VARIABLES_REGISTRY:
                raise ValueError(
                    f"Variable {var} required by {model.model_name} during "
                    "initialisation is not initialised by any model neither provided as"
                    "input."
                )
            RUN_VARIABLES_REGISTRY[var].required_init_by.append(model.model_name)


def _collect_initial_data_vars(vars: list[str]) -> None:
    """Collects the variables defined in the data object.

    Args:
        vars: The list of variables defined in the data object.
    """
    for var in vars:
        if var not in KNOWN_VARIABLES:
            raise ValueError(f"Variable {var} defined in data object is not known.")

        if var in RUN_VARIABLES_REGISTRY:
            raise ValueError(
                f"Variable {var} already in registry, initialised by"
                f"{RUN_VARIABLES_REGISTRY[var].initialised_by}."
            )

        KNOWN_VARIABLES[var].initialised_by = "data"
        RUN_VARIABLES_REGISTRY[var] = KNOWN_VARIABLES[var]


def setup_variables(
    models: list[type[base_model.BaseModel]], data_vars: list[Hashable]
) -> None:
    """Setup the runtime variable registry, running some validation.

    Args:
        models: The list of models to setup the registry for.
        data_vars: The list of variables defined in the data object.

    Raises:
        ValueError: If a variable required by a model is not in the known variables
            registry or the runtime registry.
    """
    _collect_initial_data_vars(cast(list[str], data_vars))
    _collect_initialise_by_vars(models)
    _collect_required_init_vars(models)
    _collect_updated_by_vars(models)
    _collect_required_update_vars(models)


def verify_variables_axis() -> None:
    """Verify that all required variables have valid, available axis."""
    for var in RUN_VARIABLES_REGISTRY.values():
        unknown_axes = set(var.axis).difference(axes.AXIS_VALIDATORS)

        if unknown_axes:
            to_raise = ValueError(
                f"Variable {var.name} uses unknown axis: {','.join(unknown_axes)}"
            )
            LOGGER.error(to_raise)
            raise to_raise
