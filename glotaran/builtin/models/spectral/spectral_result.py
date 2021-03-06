from __future__ import annotations

import xarray as xr

from glotaran.analysis.problem import Problem
from glotaran.builtin.models.spectral.spectral_megacomplex import SpectralMegacomplex


def finalize_spectral_result(model, problem: Problem, data: dict[str, xr.Dataset]):

    for label, dataset in data.items():

        dataset_model = problem.filled_dataset_descriptors[label]

        retrieve_spectral_data(problem.model, dataset, dataset_model)


def retrieve_spectral_data(model, dataset, dataset_model):
    spectral_compartments = []
    for megacomplex in dataset_model.megacomplex:
        if isinstance(megacomplex, SpectralMegacomplex):
            spectral_compartments += [
                compartment
                for compartment in megacomplex.shape
                if compartment not in spectral_compartments
            ]

    dataset.coords["species"] = spectral_compartments
    dataset["species_spectra"] = (
        (
            dataset_model.get_model_dimension(),
            "species",
        ),
        dataset.matrix.sel(clp_label=spectral_compartments).values,
    )
    dataset["species_associated_concentrations"] = (
        (
            dataset_model.get_global_dimension(),
            "species",
        ),
        dataset.clp.sel(clp_label=spectral_compartments).data,
    )
