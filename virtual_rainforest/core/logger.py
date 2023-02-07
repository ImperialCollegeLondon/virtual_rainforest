"""The :mod:`~virtual_rainforest.core.logging` module is used to setup the extend the
standard logging setup to provide additional functionality relevant to the virtual
rainforest model.

At the moment the module simply sets up the logger so that other modules can access it.
It is very likely to be further extended in future.

All logging messages are emitted with a specified logging level, which essentially
indicates the importance of the logging message. At the moment we use the 5 standard
logging levels, though we might extend this by using custom logging levels at some
point. The five logging levels we use are as follows:

=============  =========================================================================
Logging level  Use case
=============  =========================================================================
``CRITICAL``   Something has gone so wrong that the model run has to stop immediately.
``ERROR``      | Something has definitely gone wrong, but there is still a value in
                 continuing the execution
               | of the model. This is mainly to check if other errors crop
                 up, so that all relevant errors
               | can be reported at once.
``WARNING``    | Something seems a bit off, so the user should be warned, but the model
                 might actually be
               | fine.
``INFO``       | Something expected has happened, and it's useful to give the user
                 information about it,
               | e.g. configuration has been validated, or an output
                 file is being saved to a specific
               | location.
``DEBUG``      | Something has happened that is generally of minimal interest, but might
                 be relevant when
               | attempting to debug issues.
=============  =========================================================================

These logging levels can then be used to filter the messages the user receives, by
setting the logging level such that only messages above a certain level (of importance)
are displayed. In practice, we are likely to generally set the logging level to ``INFO``
so that ``DEBUG`` messages are suppressed, except when we are actively trying to debug
the model.

Logging and exceptions
----------------------

When an exception is allowed to halt the code, it is important for the reason to be
written to the log, as well as producing any traceback to the console. So, exception
handling should always include a LOGGER call, using one of the following patterns.

#. A test in the code indicates that we should raise an exception:

  .. code-block:: python

    if thing_has_gone_wrong:
        to_raise = ValueError("It went wrong!")
        LOGGER.critical(to_raise)
        raise to_raise

#. A ``try`` block results in an exception:

  .. code-block:: python

    try:
        doing_something_that_raises()
    except ValueError as excep:
        LOGGER.critical(excep)
        raise

#. A ``try`` block results in an exception and we want to change the exception type:

  .. code-block:: python

    try:
        doing_something_that_raises()
    except ValueError as excep:
        LOGGER.critical(excep)
        raise ValueError("Bad input") from excep
"""  # noqa: D205, D415

import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] - %(module)s - %(funcName)s(%(lineno)d) - %(message)s",
)

LOGGER = logging.getLogger("virtual_rainforest")
