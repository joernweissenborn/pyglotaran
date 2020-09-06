from glotaran.model import model_attribute_typed

from .irf_gaussian import IrfGaussian
from .irf_gaussian import IrfMultiGaussian
from .irf_measured import IrfMeasured


@model_attribute_typed(
    types={
        "gaussian": IrfGaussian,
        "multi-gaussian": IrfMultiGaussian,
        "measured": IrfMeasured,
    }
)
class Irf:
    """Represents an IRF."""
