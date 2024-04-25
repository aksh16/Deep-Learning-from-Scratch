import numpy as np
from typing import TypeVar

T = TypeVar("T", bound="Tensor")


class Tensor(object):
    def __init__(self, shape: tuple[int, int] = (1, 1), mat: np.ndarray = None):
        self.shape = shape
        self.mat = mat

    @classmethod
    def from_default(cls, default: float, rows: int, cols: int = 1):
        ndimarray = np.full((rows, cols), fill_value=default, dtype=float)
        mat = cls((rows, cols), ndimarray)
        return mat

    @classmethod
    def from_array(cls, result: np.ndarray):
        mat = cls((result.shape[0], result.shape[1]), result)
        return mat

    def __add__(self: T, other: T):
        x = np.add(self.mat, other.mat)
        t = Tensor()
        result = t.from_array(x)
        return result

    def __mul__(self: T, other: T):
        x = np.matmul(self.mat, other.mat)
        t = Tensor()
        result = t.from_array(x)
        return result

    def transpose(self) -> None:
        self.shape = (self.shape[1], self.shape[0])
        self.mat = np.transpose(self.mat)
        return

    def __repr__(self) -> str:
        return np.array2string(self.mat, formatter={'float_kind': lambda x: "%.2f" % x})
