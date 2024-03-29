---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: vr_python3
  language: python
  name: vr_python3
---

# Package release process

The package release process has three stages:

* Merge changes from the `develop` branch onto `main` that will form the new release
  version.
* Publish a release on GitHub - this is basically just a specific tagged commit on
  `main` that has some associated release notes.
* Publish the built code packages to PyPI - this is the packaged version of the code
  that users will install and use. The `virtual_ecosystem` uses the trusted publishing
  mechanism to make it easy to add new release to PyPI.

## Generate the code commit to be released

The release process for new versions of the `virtual_ecosystem` package is managed using
pull requests to the `main` branch to create a specific commit that will be released.
The steps of the process are:

1. **Create a new release branch** from the `develop` branch called `release/X.Y.Z` ,
   where `X.Y.Z` is the expected release version number.

1. Update the `pyproject.toml` file to use the expected release versions number and
   commit that change. You can use `poetry version` command to increment the major,
   minor and patch version but it is almost as easy to edit the file by hand.

1. **Start a pull request against the `main` branch**. The PR will transfer all of the
   changes to the `develop` branch since the last release on to the `main` branch. The
   PR description should provide a good explanation of the functionality that is being
   changed or added in this version, and an explanation of the suggested version number
   increment. For example, "This PR fixes a bug in calculating plant growth and so is a
   patch release from v.0.1.8 to v0.1.9".

1. **The CI testing obviously now needs to pass**. Any issues need to be resolved by
   commits or PRs onto the `release/x.y.x` branch.

1. **The PR also must be reviewed**. The code itself has already gone through the
   review process to be merged into `develop`, so this is not a code review so much as a
   review of the justification for a release.

1. **The branch can then be merged into `main`**. Do _not_ delete the release branch at
   this point.

1. **Create a second PR to merge the release branch into `develop`.** This is to
   synchronise any release changes (including the version number change) between the
   `main` and `develop` branches.

## Create the GitHub release

The head of the `main` branch is now at the commit that will be released as version
`X.Y.Z`. The starting point is to **go to the [draft new release
page](https://github.com/ImperialCollegeLondon/virtual_ecosystem/releases/new)**. The
creation of a new release is basically attaching notes and files to a specific commit on
a target branch. The steps are:

1. On that release page, the **release target** dropdown should essentially always be
   set to `main`: the whole point in this branch is to act as a release branch.

1. You need to provide a tag for the commit to be released - so you need to **tag the
   commit on the `main` branch** using the format `vX.Y.Z`. You can:

   * Create the tag locally using `git tag vX.Y.Z` and then push the tag using `git push
     --tags`. You can then select the existing tag from the drop down on the release
     page.
   * Alternatively, you can simply type the tag name into that drop down and the tag
     will be created alongside the draft release.

1. You can create release notes automatically - this is basically a list of the commits
   being added since the last release - and can also set the version as a pre-release.
   This is different from having an explicit release version number (e.g. `X.Y.Za1`) -
   it is just a marker used on GitHub.

   At this point, you can either save the draft or simply publish it. It is probably
   good practice to save the draft and then have a discussion with the other developers
   about whether to publish it.

1. Once everyone is agreed **publish the release**.

## Publish the package on PyPI

We publish to _two_ package servers:

* The
  [TestPyPI](https://test.pypi.org/project/virtual-ecosystem/) server is a final check
  to make sure that the package build and publication process is working as expected.
* The package builds are then published to the main
  [PyPI](https://pypi.org/project/virtual-ecosystem/) server for public use.

The `virtual_ecosystem` automates the publication process but the process can also be
carried out manually.

### Manual publication

The publication process _can_ be carried out from the command line. The manual process
looks like this:

```sh
# Use poetry to create package builds in the dist directory
poetry build
# Building virtual_ecosystem (0.1.1a4)
#  - Building sdist
#  - Built virtual_ecosystem-0.1.1a4.tar.gz
#  - Building wheel
#  - Built virtual_ecosystem-0.1.1a4-py3-none-any.whl

# Use twine to validate publication to TestPyPI
twine upload --repository testpypi --config-file .pypirc dist/*

# Use twine to publish to PyPI
twine upload --repository pypi --config-file .pypirc dist/*
```

The tricky bit is that you need to provide a config file containining authentication
tokens to permit publication. Those tokens **must not be included in the repository**
and so need to be carefully shared with developers who make releases. If this is being
used the `.pypirc` file contents will look something like:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-longAuthToken


[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-anotherLongAuthToken
```

At this point, you should also upload the built package wheel and source files to the
assets section of the GitHub release.

### Trusted publishing

The `virtual_ecosystem` repository is set up to use trusted publishing through a Github
Actions workflow. The workflow details are shown below, along with comments, but the
basic flow is:

1. When a GitHub release is published, the PyPI publication workflow is triggered.
1. The standard continuous integration tests are run again, just to be sure!
1. If the tests pass, the package is built and the wheel and source code are stored as
   job artefacts.
1. The built files are automatically added to the release assets.
1. The job artefacts are published to the Test PyPI server, which is configured to
   automatically trust publications from this GitHub repository.
1. As long as all the steps above succeed, the job artefacts are now published to the
   main PyPI site, which is also configured to trust publications from the repository.

```{eval-rst}
.. include:: ../../../.github/workflows/publish.yml
    :code: yaml
```
