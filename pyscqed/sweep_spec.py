""""""
from .parameters import ParamCollection


class SweepSpec:
    def __init__(self, collection: ParamCollection):
        self._collection = collection
