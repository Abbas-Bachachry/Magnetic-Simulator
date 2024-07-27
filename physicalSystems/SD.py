"""
this file contains the models of different kind single domain systems
"""
from matplotlib import pyplot as plt
from .GLL_v2 import make_jacobian_3d, do_llg_start
from .distribution_functions import get_distribution
from .datatypes import ParticlesSetType
from .display import render_box_and_particles, render_in_background
import numpy as np
from typing import Callable
import configparser
import ast


def b(tt=20):
    tc = 580  # C.
    t0 = 203  # C.
    # Moon and Merrill 1988 temperature dependence of Ms for magnetite (as used by Sherbakov)
    if tt < tc:
        modelms = ((tc - tt) / (tc - t0)) ** 0.42
    else:
        modelms = 0

    return modelms


def generate_cubic_axes(easy_axis_unit_vectors):
    no_particles = easy_axis_unit_vectors.shape[0]
    """number of particles"""

    cubic_axes_z_vectors = easy_axis_unit_vectors
    """the easy axis"""

    # normalization
    mag = np.sum(cubic_axes_z_vectors ** 2, axis=1) ** 0.5
    mag = np.array([mag, mag, mag]).T
    cubic_axes_z_vectors /= mag

    random_vector = np.random.random([no_particles, 3])
    mag = np.sum(random_vector ** 2, axis=1) ** 0.5
    mag = np.array([mag, mag, mag]).T
    random_vector /= mag

    cubic_axes_x_vectors = np.cross(cubic_axes_z_vectors, random_vector)
    mag = np.sum(cubic_axes_x_vectors ** 2, axis=1) ** 0.5
    mag = np.array([mag, mag, mag]).T
    cubic_axes_x_vectors /= mag

    cubic_axes_y_vectors = np.cross(cubic_axes_z_vectors, cubic_axes_x_vectors)
    mag = np.sum(cubic_axes_y_vectors ** 2, axis=1) ** 0.5
    mag = np.array([mag, mag, mag]).T
    cubic_axes_y_vectors /= mag

    return cubic_axes_z_vectors, cubic_axes_x_vectors, cubic_axes_y_vectors


def switching_field(x):
    x *= np.pi / 180  # radian
    tt = np.tan(x) ** (1 / 3)
    return ((1 - tt ** 2 + tt ** 4) ** 0.5) / (1 + tt ** 2)


class Box:
    """
    a Box that contain the magnetic system
    """
    dimension = np.array([581.224, 581.224, 581.224])  # nano-meter
    no_particles = 150
    no_cluster = 20
    temperature = 25

    @staticmethod
    def from_configuration(arg):
        if isinstance(arg, configparser.ConfigParser):
            config = arg
        else:
            filename = arg
            config = configparser.ConfigParser()
            config.read(filename)
        sample_type = config.get('sample', 'type')
        if config.has_option('sample', 'name'):
            name = config.get('sample', 'name')
        else:
            name = None
        if sample_type == 'Clusters':
            sample = Clusters.from_config(config)
        elif sample_type == 'RandomParticles':
            sample = RandomParticles.from_config(config)
        else:
            raise ValueError
        sample.name = name
        return sample

    @classmethod
    def random_particles(cls, dist_func: Callable = None, particle_size: float = 100, **kwargs):
        """
        this method generates random particles in the box

        :param dist_func:
        :param particle_size: the size of particles
        :key array of [float, float, float] dimension: the dimension of the box
        :key int no_particles: the number of particles in a box.
        :key float temperature: temperature of the system which is a number between 20 and 579.99. the default is 25.
        :return:
        """
        dimension = kwargs.get('dimension', cls.dimension.copy())
        no_particles = kwargs.get('no_particles', cls.no_particles)
        temperature = kwargs.get('temperature', cls.temperature)
        if dist_func is None:
            dist_func = get_distribution()

        return RandomParticles(dist_func, temperature, dimension, particle_size, no_particles)

    @classmethod
    def clusters(cls, packing_intra, dist_func: Callable = None, particle_size: float = 100, **kwargs):
        """
        this method generates random particles in the box

        :param packing_intra:
        :param particle_size:
        :param dist_func:
        :key array of [float, float, float] dimension: the dimension of the box
        :key int no_particles: the number of particles in a box.
        :key float temperature: temperature of the system which is a number between 20 and 579.99. the default is 25.
        :key int no_cluster:
        :key int particle_per_cluster:
        :return:
        """
        dimension = kwargs.get('dimension', cls.dimension.copy())
        temperature = kwargs.get('temperature', cls.temperature)
        no_cluster = kwargs.get('no_cluster', cls.no_cluster)
        particle_per_cluster = kwargs.get('particle_per_cluster', cls.no_particles // no_cluster)
        if dist_func is None:
            dist_func = get_distribution()

        return Clusters(dist_func, packing_intra, no_cluster, particle_per_cluster,
                        temperature, dimension, particle_size)


class _ParticlesSet(Box, ParticlesSetType):

    def __init__(self):
        super().__init__()
        self.__name: str | None = None

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if name is None:
            self.__name = name
        else:
            self.__name = str(name)

    def apply_filed(self, h, damping_factor=0.9, maxiterations=100):
        """

        :param damping_factor:
        :param h: applied filed
        :param maxiterations: number of iteration
        :return:
        """

        strayfield = np.zeros_like(self.positions)
        """interaction field among particles"""
        moment_vectors = np.zeros(self.positions.size)
        """moment vectors in one row
            put the moment vectors in one row to calculate the stray filed.
            """
        moment_unit_vectors = np.zeros_like(self.positions)
        moment_unit_vectors[:, 2] = 1  # directed in the z-direction
        moment_vectors[::3] = moment_unit_vectors[:, 0] * (self.volumes * self.ms).T
        moment_vectors[1::3] = moment_unit_vectors[:, 1] * (self.volumes * self.ms).T
        moment_vectors[2::3] = moment_unit_vectors[:, 2] * (self.volumes * self.ms).T
        "number of positions (particles)"

        # Repeat the loop for a maximum number of 'max iterations'
        jacob = self.jacobian
        for _ in np.arange(maxiterations):
            # step 1 calculate stray filed
            strayfield[:, 0] = (jacob @ moment_vectors)[::3].T
            strayfield[:, 1] = (jacob @ moment_vectors)[1::3].T
            strayfield[:, 2] = (jacob @ moment_vectors)[2::3].T
            moment_unit_vectors, residuals = do_llg_start(self.no_particles, h, damping_factor, moment_unit_vectors,
                                                          self.easy_axis_unit_vectors, strayfield, self.hk)
            residual_results = residuals / self.no_particles

            if residual_results < 1e-4:
                break
        return moment_unit_vectors

    def regenerate(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        assert 20 <= temperature <= 579.99
        self.temperature = temperature

        # step 1 generate position of particles
        # no overlap
        self.positions = self.generate_positions_no_overlap()

        # step 2 calculate the jacobian
        self.jacobian = self.make_jacobian_3d(self.positions)

        # step 3 generate cubic axes
        self.easy_axis_unit_vectors = np.random.random(self.positions.shape)
        # normalize the random generated easy axis
        len_ = np.sum(self.easy_axis_unit_vectors ** 2, axis=1) ** 0.5
        len_ = np.array([len_, len_, len_]).T
        self.easy_axis_unit_vectors /= len_
        self.cubic_axes = generate_cubic_axes(self.easy_axis_unit_vectors)

        # step 4
        # angle between the easy axis unit vectors and the axis 'z'
        phi = np.arccos(self.easy_axis_unit_vectors[:, 2]) * 180 / np.pi  # degrees

        hk = self.dist_func(self.no_particles)
        self.hs = switching_field(phi)
        "????????"

        self.volumes = (4 / 3) * np.pi * (self.particle_size / 2) ** 3  # nm^3
        ms = np.ones(self.no_particles) * 480000  # A/m

        self.ms = ms * b(temperature)
        self.hk = hk * b(temperature)
        "intrinsic switching field of the particle (in Tesla)"

    @staticmethod
    def make_jacobian_3d(positions):
        jacob = np.zeros([int(3 * positions.shape[0]), int(3 * positions.shape[0])], dtype=np.float64)
        pos = positions.T.reshape(positions.size)
        return make_jacobian_3d(positions.shape[0], pos, jacob)

    def easy_axis_orientation(self, orientation):
        if orientation is None:
            self.easy_axis_unit_vectors = np.random.random(self.positions.shape)
        else:
            theta, phi = orientation
            assert theta is not None and phi is not None
            easy_axis_unit_vectors = np.zeros_like(self.positions)
            easy_axis_unit_vectors[:] = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
            self.easy_axis_unit_vectors = easy_axis_unit_vectors
        # normalize the random generated easy axis
        len_ = np.sum(self.easy_axis_unit_vectors ** 2, axis=1) ** 0.5
        len_ = np.array([len_, len_, len_]).T
        self.easy_axis_unit_vectors /= len_

    def display(self, field=0, **kwargs):
        moment = self.apply_filed(field)
        background = kwargs.get('background', True)
        filename = kwargs.get('filename')
        frame = kwargs.get('frame')
        type_ = kwargs.get('type')
        if background:
            if frame:
                print("frame argument is ignored when background is True")
            return render_in_background(self.dimension,
                                        (
                                            self.positions,
                                            moment,
                                            self.easy_axis_unit_vectors
                                        ), filename=filename)
        else:
            render_box_and_particles(self.dimension,
                                     (
                                         self.positions,
                                         moment,
                                         self.easy_axis_unit_vectors
                                     ), filename=filename, frame=frame)

    def draw_coercivity_dist(self):
        # Plot the distributions
        plt.figure(figsize=(12, 6))
        plt.hist(self.hk, bins=50, density=True, label=self.name)
        plt.title("Histogram of coercivity distribution")
        plt.legend()
        plt.xlabel("Hc")
        plt.ylabel("Frequency")
        plt.show()


class RandomParticles(_ParticlesSet):
    def __init__(self, dist_func, temperature: float = 25,
                 box_size: np.ndarray = None, particle_size: float = 100, no_particles: int = 50):
        super().__init__()
        if box_size is None:
            box_size = Box.dimension.copy()
        assert no_particles < np.prod(box_size / particle_size), \
            """the number of particles is too large.
                the box cannot contain this meny."""
        self.dist_func = dist_func
        self.dimension: np.ndarray = box_size
        self.particle_size = particle_size
        self.no_particles = no_particles
        self.temperature = temperature
        self.regenerate(temperature)

    def generate_positions_no_overlap(self):
        """
        generate random position of the particles without any overlap

        :return: and array of the position of created particles.
        """
        positions_no_overlap = np.zeros((self.no_particles, 3))
        """stores the position of random particles"""
        positions_no_overlap[0] = (self.dimension - 2 * self.particle_size) * np.random.random(
            3) + self.particle_size  # position of first particle

        for i in range(1, self.no_particles):
            trial = (self.dimension - 2 * self.particle_size) * np.random.random(3) + self.particle_size
            "trial position of next random particle"

            # calculate the distance
            distances = np.sum((positions_no_overlap[:i] - trial) ** 2, axis=1) ** 0.5
            "distances between particles and the new one"

            # check if the distance between existence particles and the new is not less than the particle size
            # if it is, try another one.
            for tempt in range(1000):
                if (distances <= self.particle_size).any():
                    break

                trial = (self.dimension - 2 * self.particle_size) * np.random.random(
                    3) + self.particle_size  # position of next random particle
                distances = np.sum((positions_no_overlap[:i] - trial) ** 2, axis=1) ** 0.5

            positions_no_overlap[i] = trial
        return positions_no_overlap

    def get_info(self):
        return {
            'dist_func': self.dist_func.__name__,
            'no_particles': self.no_particles,
            'temperature': self.temperature,
            'box_size': self.dimension.tolist(),
            'particle_size': self.particle_size
        }

    @classmethod
    def from_config(cls, arg):
        if isinstance(arg, configparser.ConfigParser):
            config = arg
        else:
            filename = arg
            config = configparser.ConfigParser()
            config.read(filename)
        section = 'generation_data'
        """dist_func, temperature: float = 25,
                 box_size: np.ndarray = None, particle_size: float = 100, no_particles: int = 50"""
        generation_data = {
            'dist_func': get_distribution(config.get(section, 'dist_func')),
            'no_particles': config.getint(section, 'no_particles'),
            'temperature': config.getfloat(section, 'temperature'),
            'box_size': np.array(ast.literal_eval(config.get(section, 'box_size'))),
            'particle_size': config.getfloat(section, 'particle_size')
        }
        return cls(**generation_data)

    def __str__(self):
        return 'RandomParticles'


class Clusters(_ParticlesSet):
    def __init__(self, dist_func, packing_intra, no_cluster, particle_per_cluster, temperature: float = 25,
                 box_size: np.ndarray = None, particle_size: float = 100):
        super().__init__()
        if box_size is None:
            box_size = Box.dimension.copy()
        no_particles = no_cluster * particle_per_cluster
        volume_part_in_cluster = particle_per_cluster * particle_size ** 3
        cluster_radius = (volume_part_in_cluster / packing_intra) ** (1 / 3) / 2
        assert no_particles < np.prod(box_size / particle_size), \
            """the number of particles is too large.
                the box cannot contain this meny."""
        assert no_cluster < np.prod(box_size / cluster_radius), \
            """the number of custer is too large.
                the box cannot contain this meny."""
        self.dist_func = dist_func
        self.dimension: np.ndarray = box_size
        self.particle_size = particle_size
        self.no_particles = no_particles
        self.particle_per_cluster = particle_per_cluster
        self.no_cluster = no_cluster
        self.cluster_size = particle_size * particle_per_cluster
        self.packing_intra = packing_intra
        self.temperature = temperature
        self.regenerate(temperature)

    def generate_positions_no_overlap(self):
        """
        generate random position of the particles without any overlap

        :return: and array of the position of created particles.
        """
        positions_no_overlap = []
        volume_part_in_cluster = self.particle_per_cluster * (4 / 3) * np.pi * (self.particle_size / 2) ** 3
        cluster_radius = (volume_part_in_cluster / self.packing_intra) ** (1 / 3) / 2
        diameter = self.particle_size
        for j in range(self.no_cluster):
            attempts = 0
            while True:
                trial = (self.dimension - 2 * diameter) * np.random.random(3) + diameter
                if j == 0: break
                distances = np.sqrt(
                    np.sum((np.array(positions_no_overlap) - trial) ** 2, 1))
                if np.all(distances >= diameter): break
                attempts += 1
                if attempts > 1000:
                    print("Failed to place cluster after 1000 attempts.")
                    return None
            cluster_positions = [trial]
            for i in range(1, self.particle_per_cluster):
                attempts = 0
                while True:
                    trial2 = trial + np.random.normal(0, cluster_radius, 3)
                    distances = np.sqrt(
                        np.sum((np.array(cluster_positions) - trial2) ** 2, axis=1))
                    if np.all(distances >= diameter):
                        cluster_positions.append(trial2)
                        break
                    attempts += 1
                    if attempts > 1000:
                        print("Failed to place particle after 1000 attempts. Restarting this cluster.")
                        break
            if attempts > 1000:
                j -= 1
                break
            positions_no_overlap.extend(cluster_positions)
        return np.array(
            positions_no_overlap)

    def get_info(self):
        return {
            'dist_func': self.dist_func.__name__,
            'packing_intra': self.packing_intra,
            'no_cluster': self.no_cluster,
            'particle_per_cluster': self.particle_per_cluster,
            'temperature': self.temperature,
            'box_size': self.dimension.tolist(),
            'particle_size': self.particle_size
        }

    @classmethod
    def from_config(cls, arg):
        if isinstance(arg, configparser.ConfigParser):
            config = arg
        else:
            filename = arg
            config = configparser.ConfigParser()
            config.read(filename)
        section = 'generation_data'
        generation_data = {
            'dist_func': get_distribution(config.get(section, 'dist_func')),
            'packing_intra': config.getfloat(section, 'packing_intra'),
            'no_cluster': config.getint(section, 'no_cluster'),
            'particle_per_cluster': config.getint(section, 'particle_per_cluster'),
            'temperature': config.getfloat(section, 'temperature'),
            'box_size': np.array(ast.literal_eval(config.get(section, 'box_size'))),
            'particle_size': config.getfloat(section, 'particle_size')
        }
        return cls(**generation_data)

    def __str__(self):
        return 'Clusters'
