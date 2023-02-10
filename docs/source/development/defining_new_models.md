# How to add a new model to the Virtual Rainforest

The Virtual Rainforest is designed to be modular, meaning that the set of models to be
used in a particular run is configurable at the start of the simulation. We are starting
out by defining a core set of models (`abiotic`, `animals`, `plants` and `soil`), which
will generally all be included for the vast majority of simulations. In the future, it
might be desirable to include models for other aspects of rainforests (e.g.
`freshwater`, `disturbance`), or to include multiple modelling approaches for a process.
When this happens a new model should be created. This page will set out the process for
adding a new model to the Virtual Rainforest in a manner that allows it to be used
appropriately by the `core` simulation functionality.

## Define a new `Model` class

You should first start by defining a new folder for your model (within
`virtual_rainforest/models/`).

```bash
mkdir virtual_rainforest/models/freshwater
```

Within this folder a `python` script defining the model should be created. This script
should be called "{MODEL_NAME}_model.py".

```bash
touch virtual_rainforest/models/freshwater/freshwater_model.py
```

This script must import a number of things to be able to set up a new `Model` class
correctly.

```python
# One of the member functions of the Model class returns a class instance. mypy doesn't
# know how to handle this unless annotations are imported from __future__
from __future__ import annotations

# Any needed for type hints of the config dictionary as the values are of various types
from typing import Any

# Used by the timing loop to handle time units
import pint
# Used by timing loop to store date times, and time intervals, respectively
from numpy import datetime64, timedelta64

# Logging of relevant information handled by Virtual Rainforest logger module
from virtual_rainforest.core.logger import LOGGER
# New model class will inherit from BaseModel.
# InitialisationError is a custom exception, for case where a `Model` class cannot be
# properly initialised based on the data contained in the configuration
from virtual_rainforest.core.model import BaseModel, InitialisationError
```

The new model class is created using a class method, which means that the model is
automatically be added to the model registry when class inheritance occurs. The model is
added to the registry under the name set by the `model_name` attribute. This means that
this attribute should be populated (as a string) for every new model, and cannot be the
`model_name` of an already existing model.

```python
class FreshWaterModel(BaseModel):
    """Docstring describing model.

    Args:
        Describe arguments here
    """

    model_name = "freshwater"
    """The model name for use in registering the model and logging."""
```

With the basic class now defined an `__init__` function should be added to the class.
This should do a few things. Firstly, it should check that the provided initialisation
values are sane (e.g. number of ponds is not negative). Secondly, any model specific
attributes (e.g. number of ponds) should be stored, and general attributes should be
passed to the `__init__` of the base class. Finally, names of model specific attributes
should be appended to `__repr` so that they can be found (and printed) by `BaseModel`'s
`__repr__` function.

```python
def __init__(
    self,
    update_interval: timedelta64,
    start_time: datetime64,
    no_of_ponds: int,
    **kwargs: Any,
):
        
    # Sanity checking of input variables goes here
    if no_of_ponds < 0:
        to_raise = InitialisationError(
                "There has to be at least one pond in the freshwater model!"
            )
        LOGGER.error(to_raise)
        raise to_raise
        
    # Provide general attributes to the __init__ of the base class
    super().__init__(update_interval, start_time, **kwargs)
    # Store model specific details as attributes.
    self.no_of_ponds = int(no_of_ponds)
    # Save attribute names to be used by the __repr__
    self._repr.append("no_of_ponds")
```

Generally, `__init__` functions should only handle the most basic sanity checking and
should take in fairly simple variables. However, our configuration is stored in a
complex configuration object (a dictionary at the moment), which needs to be unpacked.
Furthermore, some of the information contained in the configuration has to be converted
to a more usable form (particularly time/date information). To accomplish this a
`from_config` factory method has to be defined. This function unpacks the configuration,
performs more complex sanity checks, and if necessary converts the information from the
configuration to a more usable form. This is a class method, and so creates a
initialised instance of the `Model` (and if it can't is raises an error).

```python
@classmethod
def from_config(cls, config: dict[str, Any]) -> FreshWaterModel:
    """Factory function to initialise the freshwater model from configuration.

    This function unpacks the relevant information from the configuration file, and
    then uses it to initialise the model. If any information from the config is
    invalid rather than returning an initialised model instance an error is raised.

    Args:
        config: The complete (and validated) Virtual Rainforest configuration.

    Raises:
        InitialisationError: If configuration data can't be properly converted
    """

    # Assume input is valid until we learn otherwise
    valid_input = True
    # Convert configuration values to the required format for the __init__
    try:
        raw_interval = pint.Quantity(config["freshwater"]["model_time_step"]).to(
            "minutes"
        )
        # Round raw time interval to nearest minute
        update_interval = timedelta64(int(round(raw_interval.magnitude)), "m")
        start_time = datetime64(config["core"]["timing"]["start_time"])
        no_of_pools = config["freshwater"]["no_of_pools"]
    # Catch cases where Values or dimensions are wrong
    except (
        ValueError,
        pint.errors.DimensionalityError,
        pint.errors.UndefinedUnitError,
    ) as e:
        valid_input = False
        LOGGER.error(
            "Configuration types appear not to have been properly validated. This "
            "problem prevents initialisation of the freshwater model. The first "
            "instance of this problem is as follows: %s" % str(e)
        )
    
    # If everything is fine initialise a class instance (using cls)
    if valid_input:
        LOGGER.info(
            "Information required to initialise the freshwater model successfully "
            "extracted."
        )
        return cls(update_interval, start_time, no_of_pools)
    else:
        # Otherwise raise an InitialisationError
        raise InitialisationError()
```

Every model class needs to include a function to update its Model state. The exact
details of what should be in this `update` function are yet to be decided, apart from
how to update the internal model timing loop.

```python
def update(self) -> None:
    """Function to update the freshwater model (only updates time currently)."""

    # Update internal model timing loop
    self.next_update += self.update_interval
```

In addition to the above, there are three other functions that must be included as part
of the model class. The names and roles of these functions might well change as the
Virtual Rainforest model develops, but that kind of API change is something that would
require significant discussion. These functions are not actually used at present, so
while they have to be included, there's no need to include any particular content within
them (i.e. they can just be function definitions with docstrings).

```python
def setup(self) -> None:
    """Placeholder function to spin up the freshwater model."""

def spinup(self) -> None:
    """Placeholder function to spin up the freshwater model."""

def cleanup(self) -> None:
    """Placeholder function for freshwater model cleanup."""
```

## Including a configuration schema

A detailed description of the configuration system can be found
[here](../virtual_rainforest/core/config.md). The key thing to note is that a `JSON`
schema file should be saved within your model folder. This file should have a name of
the format "{MODEL_NAME}_schema.json". In order for this schema to be generally
accessible, it needs to be registered in the model's `__init__.py` (i.e. the
`__init__.py` in the model folder). This means that when the model is imported, it's
schema is automatically added to the schema registry.

```python
from virtual_rainforest.core.config import register_schema

@register_schema("freshwater")
def schema() -> dict:
    """Defines the schema that the freshwater module configuration should conform to."""

    schema_file = Path(__file__).parent.resolve() / "freshwater_schema.json"

    with schema_file.open() as f:
        config_schema = json.load(f)

    return config_schema
```

## Ensuring that schema and models are always added to the registry

At the moment, a configuration schema only gets added to the schema registry when the
model it belongs to is imported, and a `Model` class only gets added to the registry
when the class itself is imported. This is a problem because the script that runs the
main Virtual Rainforest simulation does not import these things directly. To circumvent
this these imports needed to be placed in the top level `__init__.py` file (the one in
the same folder as `main.py`). This won't pass the `pre-commit` checks unless `flake8`
checks are turned off for the relevant lines. It's only strictly necessary to import the
`Model` class, as this implicitly entails importing the specific model as a whole.
However, for the sake of clarity we currently include both imports.

```python
# Import all module schema here to ensure that they are added to the registry
from virtual_rainforest.models.freshwater import schema  # noqa

# Import models here so that they also end up in the registry
from virtual_rainforest.models.freshwater.model import FreshWaterModel  # noqa
```