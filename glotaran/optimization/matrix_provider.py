import numpy as np

from glotaran.optimization.optimization_group import OptimizationGroup
from glotaran.optimization.util import CalculatedMatrix
from glotaran.optimization.util import calculate_matrix
from glotaran.optimization.util import reduce_matrix


class MatrixProvider:
    def __init__(self, group: OptimizationGroup):
        self._group = group

    def calculate(self):
        raise NotImplementedError

    def get_matrix(self, index: int) -> CalculatedMatrix:
        raise NotImplementedError

    def get_full_clp_label(self, index: int) -> list[str]:
        raise NotImplementedError


class MatrixProviderIndexIndependent(MatrixProvider):
    def __init__(
        self,
        group: OptimizationGroup,
        group_definitions: dict[str, list[str]],
        group_labels: list[str],
        group_weights: dict[str, np.typing.ArrayLike],
    ):
        super().__init__(group)

        self._group_definitions = group_definitions
        self._group_labels = group_labels
        self._group_weights = group_weights

        self._aligned_matrices = [None] * len(group_labels)
        self._aligned_full_clp_label = [None] * len(group_labels)

    def calculate_dataset_matrices(self):
        for label, dataset_model in self._group.dataset_models.items():
            self._group._matrices[label] = calculate_matrix(
                dataset_model,
                {},
            )
            self._group_clp_labels[label] = self._group._matrices[label].clp_labels
            self._group._reduced_matrices[label] = reduce_matrix(
                self._group._matrices[label],
                self._group.model,
                self._group.parameters,
                None,
            )
            if scale := dataset_model.scale:
                self._group._reduced_matrices[label] *= scale

    def create_aligned_matrices(self):
        grouped_clp_labels = {}
        grouped_matrices = {}
        grouped_full_clp_label = {}
        for group_label, dataset_labels in self._group_definitions.items():
            grouped_full_clp_label[group_label] = []
            for label in dataset_labels:
                grouped_full_clp_label += [
                    c
                    for c in self._group.matrices[dataset_labels].clp_labels
                    if c not in grouped_full_clp_label
                ]

            grouped_matrices[group_label] = combine_matrices(
                [self._group._reduced_matrices[label] for label in dataset_labels]
            )

        for i, group_label in enumerate(self._group_labels):
            group_matrix = grouped_matrices[group_label]
            if weight := self._group_weights[group_label]:
                group_matrix.matrix = group_matrix.matrix.copy()
                apply_weight(group_matrix.matrix, weight[:, i])
            self._aligned_matrices[i] = group_matrix
            self._aligned_full_clp_label[i] = grouped_full_clp_label[group_label]

    def calculate(self):

        self.calculate_dataset_matrices()

        self.create_aligned_matrices()

    def get_matrix(self, index: int) -> CalculatedMatrix:
        return self._aligned_matrices[index]

    def get_full_clp_label(self, index: int) -> list[str]:
        return self._aligned_full_clp_label[index]


class MatrixProviderIndexDependent(MatrixProvider):
    def __init__(
        self,
        group: OptimizationGroup,
        group_definitions: dict[str, list[str]],
        aligned_group_labels: list[str],
        aligned_weights: list[np.typing.ArrayLike | None],
        aligned_dataset_indices: list[list[float]],
    ):
        super().__init__(group)

        self._aligned_dataset_labels = [
            group_definitions[group_label] for group_label in aligned_group_labels
        ]
        self._aligned_dataset_indices = aligned_dataset_indices
        self._aligned_weights = aligned_weights

        self._aligned_matrices = [None] * len(aligned_group_labels)
        self._aligned_full_clp_label = [None] * len(aligned_group_labels)

    def clear(self):
        pass

    def calculate(self):
        self.clear()

        for i, dataset_labels, indices in enumerate(
            zip(self._aligned_dataset_labels, self._aligned_dataset_indices)
        ):
            group_matrices = []
            for label, index in zip(dataset_labels, indices):
                dataset_model = self._group.dataset_models[label]
                matrix = calculate_matrix(dataset_model, index)
                reduced_matrix = reduce_matrix(
                    matrix, self._group.model, self._group.parameters, index
                )
                if scale := dataset_model.scale:
                    reduced_matrix *= scale

                if label not in self._group.matrices:
                    self._group.matrices[label] = []
                    self._group.reduced_matrices[label] = []
                self._group.matrices.append(matrix)
                self._group.reduced_matrices.append(reduced_matrix)
                group_matrices.append(reduced_matrix)
            self._aligned_matrices[i] = combine_matrices(group_matrices)
            if weight := self._aligned_weights[i]:
                self._aligned_matrices[i].matrix = self._aligned_matrices[i].matrix.copy()
                apply_weight(self._aligned_matrices[i].matrix, weight[:, i])

    def get_matrix(self, index: int) -> CalculatedMatrix:
        return self._aligned_matrices[index]

    def get_full_clp_label(self, index: int) -> list[str]:
        return self._aligned_full_clp_label[index]


def combine_matrices(matrices: list[CalculatedMatrix]) -> CalculatedMatrix:
    if len(matrices) == 1:
        return matrices[0]
    masks = []
    full_clp_labels = None
    sizes = []
    dim1 = 0
    for matrix in matrices:
        clp_labels = matrix.clp_labels
        model_axis_size = matrix.matrix.shape[0]
        sizes.append(model_axis_size)
        dim1 += model_axis_size
        if full_clp_labels is None:
            full_clp_labels = clp_labels.copy()
            masks.append([i for i, _ in enumerate(clp_labels)])
        else:
            mask = []
            for c in clp_labels:
                if c not in full_clp_labels:
                    full_clp_labels.append(c)
                mask.append(full_clp_labels.index(c))
            masks.append(mask)
    dim2 = len(full_clp_labels)
    full_matrix = np.zeros((dim1, dim2), dtype=np.float64)
    start = 0
    for i, m in enumerate(matrices):
        end = start + sizes[i]
        full_matrix[start:end, masks[i]] = m.matrix
        start = end

    return CalculatedMatrix(full_clp_labels, full_matrix)
