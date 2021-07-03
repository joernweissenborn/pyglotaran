import numpy as np
import xarray as xr

from glotaran.analysis.optimize import optimize
from glotaran.analysis.simulation import simulate
from glotaran.analysis.util import calculate_matrix
from glotaran.builtin.megacomplexes.coherent_artifact import CoherentArtifactMegacomplex
from glotaran.builtin.megacomplexes.decay import DecayMegacomplex
from glotaran.model import Model
from glotaran.parameter import ParameterGroup
from glotaran.project import Scheme


def test_coherent_artifact():
    model_dict = {
        "initial_concentration": {
            "j1": {"compartments": ["s1"], "parameters": ["2"]},
        },
        "megacomplex": {
            "mc1": {"type": "decay", "k_matrix": ["k1"]},
            "mc2": {"type": "coherent-artifact", "order": 3},
        },
        "k_matrix": {
            "k1": {
                "matrix": {
                    ("s1", "s1"): "1",
                }
            }
        },
        "irf": {
            "irf1": {
                "type": "spectral-gaussian",
                "center": "2",
                "width": "3",
            },
        },
        "dataset": {
            "dataset1": {
                "initial_concentration": "j1",
                "megacomplex": ["mc1", "mc2"],
                "irf": "irf1",
            },
        },
    }
    model = Model.from_dict(
        model_dict.copy(),
        megacomplex_types={
            "decay": DecayMegacomplex,
            "coherent-artifact": CoherentArtifactMegacomplex,
        },
    )

    parameters = ParameterGroup.from_list(
        [
            101e-4,
            [10, {"vary": False, "non-negative": False}],
            [20, {"vary": False, "non-negative": False}],
            [30, {"vary": False, "non-negative": False}],
        ]
    )

    time = xr.DataArray(np.arange(0, 50, 1.5))
    spectral = xr.DataArray([0])
    coords = {"time": time, "spectral": spectral}

    dataset_model = model.dataset["dataset1"].fill(model, parameters)
    dataset_model.overwrite_global_dimension("spectral")
    dataset_model.set_coordinates(coords)
    matrix = calculate_matrix(dataset_model, {})
    compartments = matrix.coords["clp_label"].values

    print(compartments)
    assert len(compartments) == 4
    for i in range(1, 4):
        assert compartments[i - 1] == f"coherent_artifact_{i}"

    assert matrix.shape == (time.size, 4)

    clp = xr.DataArray(
        [[1, 1, 1, 1]],
        coords=[
            ("spectral", [0]),
            (
                "clp_label",
                [
                    "s1",
                    "coherent_artifact_1",
                    "coherent_artifact_2",
                    "coherent_artifact_3",
                ],
            ),
        ],
    )
    axis = {"time": time, "spectral": clp.spectral}
    data = simulate(model, "dataset1", parameters, axis, clp)

    dataset = {"dataset1": data}
    scheme = Scheme(
        model=model, parameters=parameters, data=dataset, maximum_number_function_evaluations=20
    )
    result = optimize(scheme)
    print(result.optimized_parameters)

    for label, param in result.optimized_parameters.all():
        assert np.allclose(param.value, parameters.get(label).value, rtol=1e-1)

    resultdata = result.data["dataset1"]
    assert np.array_equal(data.time, resultdata.time)
    assert np.array_equal(data.spectral, resultdata.spectral)
    assert data.data.shape == resultdata.data.shape
    assert data.data.shape == resultdata.fitted_data.shape
    assert np.allclose(data.data, resultdata.fitted_data, rtol=1e-2)

    assert "coherent_artifact_concentration" in resultdata
    assert resultdata["coherent_artifact_concentration"].shape == (time.size, 3)

    assert "coherent_artifact_associated_spectra" in resultdata
    assert resultdata["coherent_artifact_associated_spectra"].shape == (1, 3)