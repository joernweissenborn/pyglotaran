from copy import deepcopy

import pytest

from glotaran.analysis.problem_grouped import GroupedProblem
from glotaran.analysis.problem_ungrouped import UngroupedProblem
from glotaran.analysis.simulation import simulate
from glotaran.analysis.test.models import TwoCompartmentDecay as suite
from glotaran.model import ZeroConstraint
from glotaran.project import Scheme


@pytest.mark.parametrize("index_dependent", [True, False])
@pytest.mark.parametrize("grouped", [True, False])
def test_constraint(index_dependent, grouped):
    model = deepcopy(suite.model)
    model.megacomplex["m1"].is_index_dependent = index_dependent
    model.constraints.append(ZeroConstraint.from_dict({"target": "s2"}))

    print("grouped", grouped, "index_dependent", index_dependent)
    dataset = simulate(
        suite.sim_model,
        "dataset1",
        suite.wanted_parameters,
        {"e": suite.e_axis, "c": suite.c_axis},
    )
    scheme = Scheme(model=model, parameters=suite.initial_parameters, data={"dataset1": dataset})
    problem = GroupedProblem(scheme) if grouped else UngroupedProblem(scheme)

    reduced_clps = problem.reduced_clps["dataset1"]
    if index_dependent:
        reduced_matrix = (
            problem.reduced_matrices[0] if grouped else problem.reduced_matrices["dataset1"][0]
        )
    else:
        reduced_matrix = problem.reduced_matrices["dataset1"]
    matrix = problem.matrices["dataset1"][0] if index_dependent else problem.matrices["dataset1"]
    clps = problem.clps["dataset1"]

    assert "s2" not in reduced_clps.coords["clp_label"]
    assert "s2" not in reduced_matrix.coords["clp_label"]
    assert "s2" in clps.coords["clp_label"]
    assert clps.sel(clp_label="s2") == 0
    assert "s2" in matrix.coords["clp_label"]