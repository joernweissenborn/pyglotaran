import numpy as np
from typing import Literal
import xarray as xr
from numbers import Number


from glotaran.model import DatasetGroup
from glotaran.project import Scheme


def align_index(
    index: Number,
    target_axis: np.typing.ArrayLike,
    tolerance: Number,
    method: Literal["nearest", "backward", "forward"],
) -> Number:
    diff = target_axis - index

    if method == "forward":
        diff = diff[diff >= 0]
    elif method == "backward":
        diff = diff[diff <= 0]

    diff = np.abs(diff)

    if diff.min() <= tolerance:
        index = target_axis[diff.argmin()].values
    return index


class LinkedDataProvider:
    @property
    def group_definitions(self) -> dict[str, list[str]]:
        return self._group_definitions

    @property
    def aligned_group_labels(self) -> list[str]:
        return self._aligned_group_labels

    @property
    def aligned_weights(self) -> list[np.typing.ArrayLike | None]:
        return self._aligned_weights

    @property
    def aligned_dataset_indices(self) -> list[list[float]]:
        return self._aligned_dataset_indices

    @property
    def aligned_data(self) -> list[np.typing.ArrayLike]:
        return self._aligned_data

    def __init__(
        self,
        dataset_group: DatasetGroup,
        scheme: Scheme,
    ):

        aligned_axis_values = None
        aligned_datasets = {}
        aligned_dataset_labels = {}
        aligned_dataset_weights = {}
        aligned_dataset_indices = {}

        for label, dataset_model in dataset_group.dataset_models.items():
            dataset_model = dataset_model.fill(scheme.model, scheme.parameters).set_data(
                scheme.data[label]
            )
            data = dataset_model.get_data()

            aligned_global_axis = dataset_model.get_global_axis()
            if not aligned_axis_values:
                aligned_axis_values = aligned_global_axis
            else:
                aligned_global_axis = [
                    align_index(
                        index,
                        aligned_axis_values,
                        scheme.clp_link_tolerance,
                        scheme.clp_link_method,
                    )
                    for index in aligned_global_axis
                ]
                if len(np.unique(aligned_global_axis)) != len(aligned_global_axis):
                    raise ValueError(
                        "Cannot link datasets, aligning is ambiguous. \n\n"
                        "Try to lower link tolerance or change the alignment method."
                    )
                aligned_axis_values = np.unique(
                    np.concatenate([aligned_axis_values, aligned_global_axis])
                )

            aligned_datasets[label] = xr.DataArray(
                data, dims=["model", "global"], coords={"global": aligned_global_axis}
            )
            aligned_dataset_labels[label] = xr.DataArray(
                np.full_like(aligned_global_axis, label),
                dims=["global"],
                coords={"global": aligned_global_axis},
            )
            aligned_dataset_indices[label] = xr.DataArray(
                dataset_model.get_global_axis(),
                dims=["global"],
                coords={"global": aligned_global_axis},
            )
            if weight := dataset_model.get_weight():
                aligned_dataset_weights[label] = xr.DataArray(
                    weight, dims=["model", "global"], coords={"global": aligned_global_axis}
                )

        aligned_data = xr.concatenate(aligned_datasets.values(), dim="model")
        self._aligned_global_axis = aligned_data.coords["global"].data
        self._aligned_data = [
            aligned_data.isel({"global": i}).dropna(dim="model").data
            for i in range(self._aligned_global_axis)
        ]
        aligned_indices = xr.concatenate(aligned_dataset_indices.values(), dim="dataset")
        self._aligned_dataset_indices = [
            aligned_indices.isel({"global": i}).dropna(dim="dataset").data
            for i in range(self._aligned_global_axis)
        ]

        aligned_group_labels = xr.concat(
            aligned_dataset_labels.values(), dim="dataset", fill_value=""
        )
        self._aligned_group_labels = aligned_group_labels.str.join(dim="dataset").data

        self._group_definitions = {}
        self._aligned_weights = [None] * self._aligned_global_axis.size()
        for i, group_label in enumerate(self._aligned_group_labels):
            if group_label not in self._group_definitions:
                self._group_definitions[group_label] = list(
                    filter(lambda l: l != "", aligned_group_labels.isel({"global": i}).data)
                )

            group_dataset_labels = self._group_definitions[group_label]
            if any(label in aligned_dataset_weights for label in group_dataset_labels):
                weights = []
                for label in group_dataset_labels:
                    if weight := aligned_dataset_weights.get(label, None):
                        weights.append(weight.sel({"global": self._aligned_global_axis[i]}).data)
                    else:
                        size = aligned_datasets[label].coords["model"].size
                        weights.append(np.ones(size))

                self._aligned_weights[i] = np.concatenate(weights)
