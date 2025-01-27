# Frequency dynamics predict viral fitness, antigenic relationships and epidemic growth

**Marlin Figgins**<sup>1,2</sup>, **Trevor Bedford**<sup>1,3</sup>

<sup>1</sup> *Vaccine and Infectious Disease Division, Fred Hutchinson Cancer Research Center, Seattle, WA, USA*
<sup>2</sup> *Department of Applied Mathematics, University of Washington, Seattle, WA, USA*
<sup>3</sup> *Howard Hughes Medical Institute, Seattle, WA, USA*

## Abstract

During the COVID-19 pandemic, SARS-CoV-2 variants drove large waves of infections, fueled by increased transmissibility and immune escape.
Current models focus on changes in variant frequencies without linking them to underlying transmission mechanisms of intrinsic transmissibility and immune escape.
We introduce a framework connecting variant dynamics to these mechanisms, showing how host population immunity interacts with viral transmissibility and immune escape to determine relative variant fitness.
We advance a selective pressure metric that provides an early signal of epidemic growth using genetic data alone, crucial with current underreporting of cases.
Additionally, we show that a latent immunity space model approximates immunological distances, offering insights into population susceptibility and immune evasion.
These insights refine real-time forecasting and lay the groundwork for research into the interplay between viral genetics, immunity, and epidemic growth.

## Citation
Figgins and Bedford. 2024. Frequency dynamics predict viral fitness, antigenic relationships and epidemic growth. medRxiv: 2024.12.02.24318334. DOI: [10.1101/2024.12.02.24318334](https://doi.org/10.1101/2024.12.02.24318334).

## Data

The data files used for the analyses can be found in `./data`.
Details on the files and how they are obtained and generated are found in `./data/README.md`.

## Relative fitness mechanims package

This repository also includes a code base which our analyses are based on.
This includes simulation tools, the approximate Gaussian process model, latent factor model, methods for computing selective pressure, and plotting utilites.
These can be found within `./relative_fitness_mechanisms`.

## Notebooks and reproducing figures

Notebooks for reproducing the major results, main figures, and supplementary figures are found in `./notebooks/`.
Details on the outputs of these notebooks can be found in `./notebooks/`.

## Package management and reproducibility

All notebooks are run within a [poetry](https://python-poetry.org/) environment.
Instructions for downloading poetry and can found [here](https://python-poetry.org/docs/#installing-with-the-official-installer).
Once poetry is installed, you can clone this repository:

```bash
git clone https://github.com/blab/relative-fitness-mechanisms.git
```

You can then navigate to the cloned repository and install the project dependencies:

```bash
cd relative-fitness-mechanisms
poetry install
```

This will install the project's dependencies as defined in `./pyproject.toml` and create a virtual enviroment.
A shell session with this environment can then be opened with

```bash
poetry shell
```

This shell will enable you to reproduce any of the major results within jupyter notebooks using `jupyter notebook`.
After running `poetry install`, you can also directly launch the jupyter environment with

```bash
poetry run jupyter notebook
```

### Further development

If you'd like to develop this further, you can add dependencies with

```bash
poetry add <package-name>
```

For development dependencies, you can should use

```bash
poetry add --group dev <package-name>
```
