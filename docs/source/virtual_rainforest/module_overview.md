# The Virtual Rainforest modules

This document provides a brief overview of the modules that make up the Virtual
Rainforest.

## Core Module

The `core` module is responsible for:

- **Model configuration**: running a model requires a configuration file to set the
  various options to be used. The `core` module provides loading and validation routines
  for this configuration.

- **Logger configuration**: the various modules in the model can emit a lot of logging
  information and the `core` module is used to set up the logging depth and log files.

- **Spatial grid setup**: a model typically contains individual cells to capture spatial
  heterogeneity and establish landscape scale processes. The `core` module supports the
  configuration of those cells and potentially mapping of habitats to cells.

- **Input validation**: once a model is configured, the `core` module is able to
  validate the various inputs to the model to make sure that they are consistent with
  the spatial grid configuration and each other.

- **Cell initiation and timekeeping**: each cell contains instances of the various
  modules used to simulate behaviour within that cell. The `core` module sets up those
  instances.

- **Timekeeping**: the `core` module is also responsible for the timekeeping of the
  simulation - ensuring that the modules execute the right commands at the right time.

## Plant Module

The Plant Module models the primary production from plants in the Virtual Rainforest. We
use the P Model ({cite}`Prentice:2014bc,Wang:2017go`), to estimate the optimal balance
between water loss and photosynthetic productivity and hence gross primary productivity
(GPP). The P Model requires estimates of the following drivers:

- Air temperature (°C)
- Vapour pressure deficit (VPD, Pa)
- Atmospheric pressure (Pa)
- Atmospheric CO2 concentration (parts per million)
- Fraction of absorbed photosynthetically active radiation ($F\_{APAR}$, unitless)
- Photosynthetic photon flux density (PPFD, $\\mu \\text{mol}, m^{-2}, s^{-1}$)

GPP is then allocated to plant maintenance, respiration and growth using the T Model
({cite}`Li:2014bc`).

This growth model is used to simulate the demographics of cohorts of key plant
functional types (PFTs) under physiologically structured population models developed in
the [Plant-FATE](https://jaideep777.github.io/libpspm/) framework. The framework uses
the perfect-plasticity approximation (PPA, {cite}`purves:2008a`) to model the canopy
structure of the plant community, the light environments of different PFTs and hence the
change in the size-structured demography of each PFT through time.

## Soil Module

The principal function of the Soil Module is to model the cycling of nutrients. This
cycling is assumed to be primarily driven by microbial activity, which in turn is
heavily impacted by both environmental and soil conditions. Plant-microbe interactions
are taken to principally be either exchanges of or competition for nutrients, and so are
modelled within the same nutrient cycling paradigm. Three specific nutrient cycles are
incorporated into this module:

### Carbon cycle

The Carbon cycle uses as its basic structure a recently described soil-pool model termed
the Millennial model ({cite}`abramoff_millennial_2018`). This model splits carbon into
five separate pools: particulate organic matter, low molecular weight carbon (LMWC),
mineral associated organic matter, aggregates and microbial biomass. Though plant root
exudates feed directly into the LMWC pool, most biomass input will less direct and occur
via litter decomposition. Thus, we utilize a common set of litter pools
({cite}`kirschbaum_modelling_2002`), that are divided between above- and below-ground
pools, and by biomass source (e.g. deadwood).

### Nitrogen cycle

The Nitrogen cycle is strongly coupled to the carbon cycle, therefore tracking the
stoichiometry of the carbon pools is key to modelling it correctly. In addition,
specific forms of nitrogen are explicitly modelled. They are as follows: a combined
$\\ce{NH\_{3}}$ and $\\ce{NH\_{4}^{+}}$ pool to represent the products of nitrogen
fixation and ammonification, a $\\ce{NO\_{3}^{-}}$ pool to represent the products of
nitrification, and a $\\ce{NO\_{2}^{-}}$ pool to capture the process of denitrification.

### Phosphorous cycle

The Phosphorus cycle is similarly coupled to the carbon cycle. The additional inorganic
pools tracked in this case are as follows: primary phosphorus in the form of weatherable
minerals, mineral phosphorus which can be utilized by plants and microbes, secondary
phosphorus which is mineral associated but can be recovered as mineral phosphorus, and
occluded phosphorus which is irrecoverably bound within a mineral structure.

### Further details

Further theoretical background for the soil module can be found [here](./soil/soil_details.md).

## Animal Module

## Abiotic Module

The abiotic module provides the microclimate and hydrology for the Virtual Rainforest.
Using a small set of input variables from external sources such as WFDE5
({cite}:`WFDE5-2020`) or regional climate models, the module calculates atmospheric and
soil parameters that drive the dynamics of plants, animals, and microbes at different
vertical levels. Two subroutines, the energy balance and the water balance, provide the
following variables at required levels:

- Net radiation ($R_N$) and Photosynthetic photon flux density ($PPFD$)
- Air Temperature ($T\_{air}$)
- Relative humidity ($RH$) and vapor pressure deficit ($VPD$)
- Soil Temperature ($T\_{soil}$)
- Soil moisture ($W\_{soil}$)
- Runoff ($RO$), mean vertical flow ($VF$) and drainage ($D$)

### Vertical structure of atmosphere and soil

The atmosphere is divided in four vertical layers:

1. the top of the canopy which links the external driver to the module,
1. the upper canopy where most photosynthetic activity occurs,
1. the understorey where most large animal are active, and
1. the near surface which homes ground-dwelling organisms and links the atmosphere to
   the top soil layer.

The soil is represented by three vertical layers:

1. the top soil where most microbial activity occurs,
1. the root zone where plant water extraction is the prevalent process, and
1. the deep soil where changes in water storage and subsurface drainage are modeled.

### The Energy balance

The Energy balance subroutine uses incoming solar radiation and vegetation structure to
calculate vertical profiles of $R_N$ and $PPFD$. Based on the vertical profile of $R_N$,
the subroutine derives sensible and latent heat fluxes from leaves and soil to the
atmosphere and updates $T\_{air}$, $RH$, and $VPD$ at each level. The vertical mixing
between levels is assumed to be driven by heat conductance because turbulence is
typically low below the canopy ({cite}:`MACLEAN2021`). Part of the $R_N$ is converted
into soil heat flux. The vertical exchange of heat between soil layers follows that same
approach as the atmospheric mixing.

### The Water balance

The Water balance subroutine is based on the soil moisture 'bucket' scheme of the SPLASH
model ({cite}:`Davis:2017`). The scheme uses rainfall and soil moisture of the previous
timestep to calculate runoff, evaporation, condensation, and soil moisture. We extend
the SPLASH scheme to derive soil moisture at different vertical levels, mean vertical
flow, and drainage.

## Disturbance Module

Introducing disturbances (e.g. logging) into the model will usually require making
alterations to the state of multiple modules. As such, different disturbance models are
collected in a separate Disturbance Module. This module will be capable of altering the
state of all the other modules, and will do so in a manner that allows the source of the
changes to be explicitly identified.
