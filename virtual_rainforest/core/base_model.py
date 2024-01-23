"""The :mod:`~virtual_rainforest.core.base_model` module defines the high level API for
the different models within the Virtual Rainforest. The module creates the
:class:`~virtual_rainforest.core.base_model.BaseModel` abstract base class (ABC) which
defines a consistent API for subclasses defining an actual model. The API defines
abstract methods for each of the key stages in the workflow of running a model:
individual subclasses are **required** to provide model specific implementations for
each stage, although the specific methods may simply do nothing if no action is needed
at that stage. The stages are:

* Creating a model instance (:class:`~virtual_rainforest.core.base_model.BaseModel`).
* Setup a model instance (:meth:`~virtual_rainforest.core.base_model.BaseModel.setup`).
* Perform any spinup required to get a model state to equilibrate
  (:meth:`~virtual_rainforest.core.base_model.BaseModel.spinup`).
* Update the model from one time step to the next
  :meth:`~virtual_rainforest.core.base_model.BaseModel.update`).
* Cleanup any unneeded resources at the end of a simulation
  (:meth:`~virtual_rainforest.core.base_model.BaseModel.cleanup`).

The :class:`~virtual_rainforest.core.base_model.BaseModel` class also provides default
implementations for the :meth:`~virtual_rainforest.core.base_model.BaseModel.__repr__`
and :meth:`~virtual_rainforest.core.base_model.BaseModel.__str__` special methods.

The :class:`~virtual_rainforest.core.base_model.BaseModel` has four class attributes
that must be defined in subclasses:

* The :attr:`~virtual_rainforest.core.base_model.BaseModel.model_name` attribute and
* The :attr:`~virtual_rainforest.core.base_model.BaseModel.required_init_vars`
  attribute.
* The :attr:`~virtual_rainforest.core.base_model.BaseModel.model_update_bounds`
  attribute
* The :attr:`~virtual_rainforest.core.base_model.BaseModel.vars_updated`
  attribute


The usage of these four attributes is described in their docstrings and each is
validated when a new subclass is create using the following private
methods of the class:
:meth:`~virtual_rainforest.core.base_model.BaseModel._check_model_name`,
:meth:`~virtual_rainforest.core.base_model.BaseModel._check_required_init_vars`,
:meth:`~virtual_rainforest.core.base_model.BaseModel._check_model_update_bounds` and
:meth:`~virtual_rainforest.core.base_model.BaseModel._check_vars_updated`.

Model checking
--------------

The :class:`~virtual_rainforest.core.base_model.BaseModel` abstract base class defines
the :func:`~virtual_rainforest.core.base_model.BaseModel.__init_subclass__` class
method. This method is called automatically whenever a subclass of the ABC is imported
and validates the class attributes for the new model class.

The ``BaseModel.__init__`` method
----------------------------------

The ``__init__`` method for subclasses **must** call the ``BaseModel``
:meth:`~virtual_rainforest.core.base_model.BaseModel.__init__` method as shown below.
This method carries out some core initialisation steps: see the method description for
details.

.. code-block:: python

    super().__init__(data, update_interval, start_time, **kwargs)


The ``from_config`` factory method
----------------------------------

The ABC also defines the abstract class method
:func:`~virtual_rainforest.core.base_model.BaseModel.from_config`. This method must be
defined by subclasses and must be a factory method that takes a
:class:`~virtual_rainforest.core.data.Data` instance and  a model specific configuration
dictionary and returns an instance of the subclass. For any given model, the method
should provide any code to validate the configuration and then use the configuration to
initialise and return a new instance of the class.

Model registration
------------------

Models have three core components: the
:class:`~virtual_rainforest.core.base_model.BaseModel` subclass itself (``model``),
a JSON schema for validating the model configuration (``schema``) and an optional set of
user modifiable constants classes (``constants``, see
:class:`~virtual_rainforest.core.constants_class.ConstantsDataclass`). All model
modules must register these components when they are imported: see the
:mod:`~virtual_rainforest.core.registry` module.
"""  # noqa: D205, D415

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pint
import xarray as xr

from virtual_rainforest.core.axes import AXIS_VALIDATORS
from virtual_rainforest.core.config import Config
from virtual_rainforest.core.data import Data
from virtual_rainforest.core.exceptions import ConfigurationError
from virtual_rainforest.core.logger import LOGGER


class BaseModel(ABC):
    """A superclass for all Virtual Rainforest models.

    This abstract base class defines the shared common methods and attributes used as an
    API across all Virtual Rainforest models. This includes functions to setup, spin up
    and update the specific model, as well as a function to cleanup redundant model
    data.

    The base class defines the core abstract methods that must be defined in subclasses
    as well as shared helper functions.

    Args:
        data: A :class:`~virtual_rainforest.core.data.Data` instance containing
            variables to be used in the model.
        start_time: Point in time that the model simulation should be started.
        update_interval: Time to wait between updates of the model state.
    """

    model_name: str
    """The model name.

    This class attribute sets the name used to refer to identify the model class in
    the :data:`~virtual_rainforest.core.registry.MODULE_REGISTRY`, within the
    configuration settings and in logging messages.
    """

    model_update_bounds: tuple[pint.Quantity, pint.Quantity]
    """Bounds on model update frequencies.

    This class attribute defines two time intervals that define a lower and upper bound
    on the update frequency that can reasonably be used with a model. Models updated
    more often than the lower bound may fail to capture transient dynamics and models
    updated more slowly than the upper bound may fail to capture important temporal
    patterns.
    """

    required_init_vars: tuple[tuple[str, tuple[str, ...]], ...]
    """Required variables for model initialisation.

    This class property defines a set of variable names that must be present in the
    :class:`~virtual_rainforest.core.data.Data` instance used to initialise an
    instance of this class. It is a tuple containing zero or more tuples, each
    providing a variable name and then a tuple of zero or more core axes that the
    variable must map onto.

    For example: ``(('temperature', ('spatial', 'temporal')),)``
    """

    vars_updated: tuple[str, ...]
    """Variables that are updated by the model.

    At the moment, this tuple is used to decide which variables to output from the
    :class:`~virtual_rainforest.core.data.Data` object, i.e. every variable updated
    by a model used in the specific simulation. In future, this could also be used
    to prevent multiple models from updating the same variable and similar problems.
    """

    def __init__(
        self,
        data: Data,
        update_interval: pint.Quantity,
        static: bool = False,
        static_data: Optional[xr.Dataset] = None,
        **kwargs: Any,
    ):
        """Performs core initialisation for BaseModel subclasses.

        This method should be called by the ``__init__`` method of all subclasses and
        performs the following core steps:

        * It populates the shared instance attributes
          :attr:`~virtual_rainforest.core.base_model.BaseModel.data` and
          :attr:`~virtual_rainforest.core.base_model.BaseModel.update_interval`.
        * It uses the
          :meth:`~virtual_rainforest.core.base_model.BaseModel.check_init_data`
          to confirm that the required variables for the model are present in the
          :attr:`~virtual_rainforest.core.base_model.BaseModel.data` attribute.
        """
        self.data = data
        """A Data instance providing access to the shared simulation data."""
        self.update_interval = self._check_update_speed(update_interval)
        """The time interval between model updates."""
        self._repr = ["update_interval"]
        """A list of attributes to be included in the class __repr__ output"""
        self._static = static
        """Flag indicating if the model is static, i.e. does not change with time."""
        self._static_data = static_data
        """Pre-set data for the whole simulation provided as input."""

        # Check the required init variables
        self.check_init_data()

    @abstractmethod
    def setup(self) -> None:
        """Function to use input data to set up the model."""

    @abstractmethod
    def spinup(self) -> None:
        """Function to spin up the model."""

    def update(self, time_index: int, **kwargs: Any) -> None:
        """Function to update the model.

        Args:
            time_index: The index representing the current time step in the data object.
        """
        if not self._static:
            self._update(time_index, **kwargs)
            return

        self._update_static(time_index, **kwargs)

    @abstractmethod
    def _update(self, time_index: int, **kwargs: Any) -> None:
        """Function to update the model.

        Args:
            time_index: The index representing the current time step in the data object.
        """

    def _update_static(self, time_index: int, **kwargs: Any) -> None:
        """Function to update the model in the static case.

        Args:
            time_index: The index representing the current time step in the data object.
        """
        if self._static_data is None:
            self._update(time_index, **kwargs)
            self._build_static_data(time_index)
            return

        self._update_with_static_data(time_index)

    def _build_static_data(self, time_index: int) -> None:
        """Populates the static data attribute with the data for time index.

        It extracts :attr:`~virtual_rainforest.core.base_model.BaseModel.vars_updated`
        from :attr:`~virtual_rainforest.core.base_model.BaseModel.data` at `time_index`
        to create :attr:`~virtual_rainforest.core.base_model.BaseModel._static_data`,
        which then is used at any future times.

        Args:
            time_index: The index representing the current time step in the data object.
        """

    def _update_with_static_data(self, time_index: int) -> None:
        """Updates data object with with static data at time_index.

        It updates :attr:`~virtual_rainforest.core.base_model.BaseModel.data` with the
        :attr:`~virtual_rainforest.core.base_model.BaseModel._static_data` at
        `time_index`.

        Args:
            time_index: The index representing the current time step in the data object.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Function to delete objects within the class that are no longer needed."""

    @classmethod
    @abstractmethod
    def from_config(
        cls, data: Data, config: Config, update_interval: pint.Quantity
    ) -> BaseModel:
        """Factory function to unpack config and initialise a model instance."""

    @classmethod
    def _check_model_name(cls, model_name: str) -> str:
        """Check the model_name attribute is valid.

        Args:
            model_name: The
                :attr:`~virtual_rainforest.core.base_model.BaseModel.model_name`
                attribute to be used for a subclass.

        Raises:
            ValueError: the model_name is not a string.

        Returns:
            The provided ``model_name`` if valid
        """

        if not isinstance(model_name, str):
            excep = TypeError(
                f"Class attribute model_name in {cls.__name__} is not a string"
            )
            LOGGER.error(excep)
            raise excep

        return model_name

    @classmethod
    def _check_required_init_vars(
        cls, required_init_vars: tuple[tuple[str, tuple[str, ...]], ...]
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Check the required_init_vars property is valid.

        Args:
            required_init_vars: The
                :attr:`~virtual_rainforest.core.base_model.BaseModel.required_init_vars`
                attribute to be used for a subclass.

        Raises:
            TypeError: the value of required_init_vars has the wrong type structure.
            ValueError: required_init_vars uses unknown core axis names.

        Returns:
            The provided ``required_init_vars`` if valid
        """

        to_raise: Exception

        # Check the structure
        required_init_vars_ok = True
        unknown_axes: list[str] = []

        if not isinstance(required_init_vars, tuple):
            required_init_vars_ok = False
        else:
            for entry in required_init_vars:
                # entry is a 2 tuple
                if not (isinstance(entry, tuple) and len(entry) == 2):
                    required_init_vars_ok = False
                    continue

                # and entry contains (str, tuple(str,...))
                vname, axes = entry
                if not (
                    isinstance(vname, str)
                    and isinstance(axes, tuple)
                    and all([isinstance(a, str) for a in axes])
                ):
                    required_init_vars_ok = False
                else:
                    # Add any unknown axes
                    unknown_axes.extend(set(axes).difference(AXIS_VALIDATORS))

        if not required_init_vars_ok:
            to_raise = TypeError(
                f"Class attribute required_init_vars has the wrong "
                f"structure in {cls.__name__}"
            )
            LOGGER.error(to_raise)
            raise to_raise

        if unknown_axes:
            to_raise = ValueError(
                f"Class attribute required_init_vars uses unknown core "
                f"axes in {cls.__name__}: {','.join(unknown_axes)}"
            )
            LOGGER.error(to_raise)
            raise to_raise

        return required_init_vars

    @classmethod
    def _check_model_update_bounds(
        cls, model_update_bounds: tuple[str, str]
    ) -> tuple[pint.util.Quantity, pint.util.Quantity]:
        """Check that the model_update_bounds attribute is valid.

        This is used to validate the class attribute
        :attr:`~virtual_rainforest.core.base_model.BaseModel.model_update_bounds`, which
        describes the lower and upper bounds on model update frequency. The lower bound
        must be less than the upper bound.

        Args:
            model_update_bounds: A tuple of two strings representing time periods that
                can be parsed using :class:`pint.Quantity`.


        Raises:
            ValueError: If the provided model_update_bounds cannot be parsed as
                :class:`pint.Quantity` with time units or if the lower bound is not less
                than the upper bound.

        Returns:
            The validated model_update_bounds, converted to a tuple of
            :class:`pint.Quantity` values.
        """

        # Check the conversion
        try:
            model_update_bounds_pint: tuple[pint.util.Quantity, pint.util.Quantity] = (
                pint.Quantity(model_update_bounds[0]),
                pint.Quantity(model_update_bounds[1]),
            )
        except pint.errors.UndefinedUnitError:
            to_raise = ValueError(
                f"Class attribute model_update_bounds for {cls.__name__} "
                "contains undefined units."
            )
            LOGGER.error(to_raise)
            raise to_raise

        # Check time units
        if not all(val.check("[time]") for val in model_update_bounds_pint):
            to_raise = ValueError(
                f"Class attribute model_update_bounds for {cls.__name__} "
                "contains non-time units."
            )
            LOGGER.error(to_raise)
            raise to_raise

        # Check lower less than upper bound
        if model_update_bounds_pint[0] >= model_update_bounds_pint[1]:
            to_raise = ValueError(
                f"Lower time bound for {cls.__name__} is not less than the upper "
                f"bound."
            )
            LOGGER.error(to_raise)
            raise to_raise

        return model_update_bounds_pint

    def _check_update_speed(self, update_interval: pint.Quantity) -> pint.Quantity:
        """Function to check that the update speed of a specific model is within bounds.

        Args:
            update_interval: A :class:`pint.Quantity` giving the update interval for the
                model.

        Returns:
            The update interval for the overall model

        Raises:
            ConfigurationError: If the update interval does not fit with the model's
                time bounds
        """

        # Check if either bound is violated
        if update_interval < pint.Quantity(self.model_update_bounds[0]):
            to_raise = ConfigurationError(
                "The update interval is faster than the model update bounds."
            )
            LOGGER.error(to_raise)
            raise to_raise

        if update_interval > pint.Quantity(self.model_update_bounds[1]):
            to_raise = ConfigurationError(
                "The update interval is slower than the model update bounds."
            )
            LOGGER.error(to_raise)
            raise to_raise

        return update_interval

    @classmethod
    def _check_vars_updated(cls, vars_updated: tuple[str, ...]) -> tuple[str, ...]:
        """Check that vars_updated is valid.

        Returns:
            The provided value if valid.
        """
        # TODO - currently no validation.
        return vars_updated

    @classmethod
    def __init_subclass__(
        cls,
        model_name: str,
        model_update_bounds: tuple[str, str],
        required_init_vars: tuple[tuple[str, tuple[str, ...]], ...],
        vars_updated: tuple[str, ...],
    ) -> None:
        """Initialise subclasses deriving from BaseModel.

        This method runs when a new BaseModel subclass is imported. It adds the new
        subclasses to the model registry and populates the values of the class
        attributes.

        Subclasses of the BaseModel need to provide the values for class attributes in
        their signatures. Those values are defined by the arguments to this method,
        which validates and sets the class attributes for the subclass. See
        :class:`~virtual_rainforest.core.base_model.BaseModel` for details on the class
        attributes. For example:

        .. code-block:: python

            class ExampleModel(
                BaseModel,
                model_name='example',
                model_update_bounds= ("30 minutes", "3 months"),
                required_init_vars=(("required_variable", ("spatial",)),),
                vars_updated=("updated_variable"),
            ):
                ...

        Args:
            model_name: The model name to be used
            model_update_bounds: Bounds on update intervals handled by the model
            required_init_vars: A tuple of the variables required to create a model
                instance.
            vars_updated: A tuple of the variable names updated by the model.

        Raises:
            ValueError: If the model_name or required_init_vars properties are not
                defined
            TypeError: If model_name is not a string
        """

        try:
            cls.model_name = cls._check_model_name(model_name=model_name)
            cls.required_init_vars = cls._check_required_init_vars(
                required_init_vars=required_init_vars
            )
            cls.vars_updated = cls._check_vars_updated(vars_updated=vars_updated)
            cls.model_update_bounds = cls._check_model_update_bounds(
                model_update_bounds=model_update_bounds
            )

        except (NotImplementedError, TypeError, ValueError) as excep:
            LOGGER.critical(
                f"Errors in defining {cls.__name__} class attributes: see log"
            )
            raise excep

    def __repr__(self) -> str:
        """Represent a Model as a string."""

        # Add all args to the function signature
        func_sig = ", ".join([f"{k} = {getattr(self, k)}" for k in self._repr])

        return f"{self.__class__.__name__}({func_sig})"

    def __str__(self) -> str:
        """Inform user what the model type is."""
        return f"A {self.model_name} model instance"

    def check_init_data(self) -> None:
        """Check the init data contains the required variables.

        This method is used to check that the set of variables defined in the
        :attr:`~virtual_rainforest.core.base_model.BaseModel.required_init_vars` class
        attribute are present in the :attr:`~virtual_rainforest.core.data.Data` instance
        used to create a new instance of the class.

        Raises:
            ValueError: If the Data instance does not contain all the required variables
                or if those variables do not map onto the required axes.
        """

        # Sentinel variables
        all_axes_ok: bool = True
        all_vars_found: bool = True

        # Loop over the required  and axes
        for var, axes in self.required_init_vars:
            # Record when a variable is missing
            if var not in self.data:
                LOGGER.error(
                    f"{self.model_name} model: init data missing required var '{var}'"
                )
                all_vars_found = False
                continue

            # Get a list of missing axes
            bad_axes = []
            # Could use try: here and let on_core_axis report errors but easier to
            # provide more clearly structured feedback this way
            for axis in axes:
                if not self.data.on_core_axis(var, axis):
                    bad_axes.append(axis)

            # Log the outcome
            if bad_axes:
                LOGGER.error(
                    f"{self.model_name} model: required var '{var}' "
                    f"not on required axes: {','.join(bad_axes)}"
                )
                all_axes_ok = False
            else:
                LOGGER.debug(f"{self.model_name} model: required var '{var}' checked")

        # Raise if any problems found
        if not (all_axes_ok and all_vars_found):
            error = ValueError(
                f"{self.model_name} model: error checking required_init_vars, see log."
            )
            LOGGER.error(error)
            raise error
