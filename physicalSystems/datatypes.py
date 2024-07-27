from abc import ABC, abstractmethod
from typing import Callable, NewType, Union, Optional
from numpy import ndarray

Number = NewType('Number', Union[int, float])


class DataSet(ABC):

    @abstractmethod
    def apply_filed(self, h, damping_factor=0.9, maxiterations=100):
        """

        :param damping_factor:
        :param h: applied filed
        :param maxiterations: number of iteration
        :return:
        """
        pass

    @abstractmethod
    def regenerate(self, temperature=None):
        pass

    @abstractmethod
    def generate_positions_no_overlap(self):
        pass

    @staticmethod
    def make_jacobian_3d(positions):
        pass

    @abstractmethod
    def easy_axis_orientation(self, orientation):
        pass

    @abstractmethod
    def get_info(self):
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, filename):
        pass


class ParticlesSetType(DataSet, ABC):
    dimension = ...  # nano-meter
    no_particles = ...
    no_cluster = ...
    temperature = ...

    def __init__(self):
        self.__name: Optional[str] = None
        self.dist_func: Callable[[int], ndarray] = ...
        self.particle_size: ndarray = ...
        self.positions: ndarray = ...
        self.jacobian: ndarray = ...
        self.easy_axis_unit_vectors: ndarray = ...
        self.cubic_axes: tuple[ndarray, ndarray, ndarray] = ...
        self.hs = ...
        "Unknown functionality"
        self.volumes: Number = ...
        "volume of particles"
        self.ms: ndarray = ...
        "saturation magnetization of each particle"
        self.hk = ...
        "intrinsic switching field of the particle (in Tesla)"


class PlateType:
    temperature = ...


Sample = NewType('Sample', Union[ParticlesSetType])
