import numpy as np
import xarray as xr


class LinkedDataset:
    def __init__(self, dataset_models):

        # create aligned datasets

        aligned_axis_values = None
        aligned_datasets = {}
        aligned_dataset_labels = {}
        aligned_dataset_weights = {}
        aligned_dataset_indices = {}

        for label, dataset_model in dataset_models.items():
            data = dataset_model.get_data()

            aligned_global_axis = dataset_model.get_global_axis()
            if not aligned_axis_values:
                aligned_axis_values = aligned_global_axis
            else:
                aligned_global_axis = [
                    align_index(index, aligned_axis_values, tolerance)
                    for index in aligned_global_axis
                ]
                if len(np.unique(aligned_global_axis)) != len(aligned_global_axis):
                    raise ValueError(
                        "Cannot link datasets, aligning is ambiguous. Try lower link tolerance."
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
