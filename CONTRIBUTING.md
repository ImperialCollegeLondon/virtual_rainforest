# Contributing to the Virtual Ecosystem

We're really happy that you are thinking about contributing to the Virtual Ecosystem
project. The whole point of the project is to generate a tool that can be used widely by
the community and the best way for that to happen is for the community to build it.

To make contributing as seamless as possible, please note these developer guidelines.

## Development setup

Further notes are available [here](source/docs/develop/developer_setup.md) but - once
you have cloned the Virtual Ecosystem repository to your own machine - you will need to
set up our development toolchain on your machine to contribute to the project.

* We use [`poetry`](https://https://python-poetry.org/) to manage the package
  development. Once you have installed `poetry`, you can use `poetry install` within
  the repo to start using it.
* We use [`pre-commit`](https://pre-commit.com/) to maintain code quality on
  submitted code. Before changing the code, install `pre-commit` and then use
  `pre-commit install` to make sure that your code is being checked.

## Contributing code

We expect all contributors to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). The
steps for contributing code are:

* Code contributed to the Virtual Ecosystem should usually address a documented issue
  on the [issue
  tracker](https://github.com/ImperialCollegeLondon/virtual_ecosystem/issues). Very
  minor changes - typos or simple one-line fixes - may not need to have an issue.
* If you are fixing an existing issue then that is great, but please do ask to be
  assigned to the issue to avoid duplicating effort!
* If this is something new, please raise an issue describing the contribution you want
  to make and **do wait for feedback** on your suggestion before spending time and
  effort coding.
* Once you are ready to contribute code, create a new feature branch in your local
  repository and develop your code in that branch.
* When the issue is solved, do include the text `Closes #nnn` in your final commit
  message body, to tie your pull request to the original issue.
* Obviously, that code needs to pass the `pre-commit` checks!
* Now submit a **pull request** to merge your branch into the `develop` branch of the
  Virtual Ecosystem project.
* We will then review the contributed code and merge it once any problems have been
  resolved.

## The `pytest` framework

We use [`pytest`](https://docs.pytest.org/) to run continuous integration and other
testing on the code in the Virtual Ecosystem. If you are adding new functionality or
fixing errors in existing implementations, please also add new tests or amend any
existing tests.
