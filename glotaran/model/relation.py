""" Glotaran Relation """
from __future__ import annotations

from glotaran.model import model_attribute
from glotaran.model.interval_property import IntervalProperty
from glotaran.parameter import Parameter


@model_attribute(
    properties={
        "source": str,
        "target": str,
        "parameter": Parameter,
    },
    no_label=True,
)
class Relation(IntervalProperty):
    """Applies a relation between clps as

    :math:`target = parameter * source`.
    """