from lmfit import Parameter, Parameters
from lmfit_varpro import SeparableModel

from .matrix_group_generator import MatrixGroupGenerator
from .result import Result


class FitModel(SeparableModel):

    def __init__(self, model):
        self._model = model
        self._generator = None
        self._dataset_group = None

    def get_initial_fitting_parameter(self):
        self._prepare_parameter()
        return self._fit_params

    def data(self, **kwargs):
        return self._dataset_group

    def fit(self, *args, **kwargs):
        self._generator = MatrixGroupGenerator.for_model(self._model,
                                                         self._model.
                                                         calculated_matrix())
        self._dataset_group = self._generator.create_dataset_group()
        result = Result(self, self.get_initial_fitting_parameter(),
                        *args, **kwargs)
        result.fit(*args, **kwargs)
        return result

    def c_matrix(self, parameter, **kwargs):
        if "dataset" in kwargs:
            label = kwargs["dataset"]
            gen = MatrixGroupGenerator.for_dataset(self._model, label,
                                                   self._model.
                                                   calculated_matrix())
            return gen.calculate(parameter)
        else:
            return self._generator.calculate(parameter)

    def e_matrix(self, parameter, **kwargs):
        dataset = kwargs["dataset"]
        gen = MatrixGroupGenerator.for_dataset(self._model, dataset,
                                               self._model.estimated_matrix(),
                                               calculated=True)
        return gen.calculate(parameter)

    def get_all_parameter(self):
        parameters = Parameters()
        for p in self.model.parameter:
            pass


    def _prepare_parameter(self):
        self._fit_params = Parameters()

        relations = [] # a list param indices which have relations
        fixed = [] # a list of param indices which are fixed
        bound = [] # list of tuples (index_relation, list[param_indices])

        # Collect Relations
        if self._model.relations is not None:
            relations = [r.parameter for r in self._model.relations]

        # Collect constraints
        if self._model.parameter_constraints is not None:
            i = 0
            for constraint in self._model.parameter_constraints:
                if isclass(constraint, "FixedConstraint"):
                    for p in constraint.parameter:
                        fixed.append(p)
                elif isclass(constraint, "BoundConstraint"):
                    bound.append((i, constraint.parameter))
                i += 1

        # add parameter from list
        for p in self._model.parameter:
            self._add_fit_param(p, fixed, bound, relations)

        # add parameter blocks

        for _, block in self._model.parameter_blocks.items():
            pass

    def _add_fit_block(self, block):
            if block.fit:
                for p in block.parameter:
                    self._add_fit_param(p, fixed, bound, relations)
                if block.sub_blocks is not None:
                    for sblock in block.sub_blocks:
                        self._add_fit_block(sblock)

    def _add_fit_param(self, p, fixed, bound, relations):
        # NaN means ignore param
        if not p.value == 'NaN':
            # collect onstraints for the param
            vary = p.index not in fixed
            min, max = None, None
            expr = None
            val = p.value
            for i in range(len(bound)):
                if p.index in bound[i][1]:
                    b = self._model.relations[bound[i][0]]
                    if b.min != 'NaN':
                        min = b.min
                    if b.max != 'NaN':
                        max = b.max
            # translate relation if necessary
            if p.index in relations:
                r = self._model.relations[relations.index(p.index)]
                vary = False
                val = None
                first = True
                expr = ''
                for target in r.to:
                    if not first:
                        expr += "+"
                    first = False
                    if target == 'const':
                        expr += "{}".format(r.to[target])
                    else:
                        expr += "p{}*{}".format(target, r.to[target])

            self._fit_params.add("p{}".format(p.index), val,
                                 vary=vary, min=min, max=max, expr=expr)


def isclass(obj, classname):
    return obj.__class__.__name__ == classname
