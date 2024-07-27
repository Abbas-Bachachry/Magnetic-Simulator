from time import time
from .SD import (
    Box,
    RandomParticles as _rd,
    Clusters as _c
)
from .distribution_functions import get_distribution
from .datatypes import Sample
from .GLL import llg_forc, prepare_llg_history
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import configparser


class FORCSimulator:  # fixme
    """
    First-Order Reversal Curve measurements Simulator
    """

    @classmethod
    def from_config(cls, filename):
        config = configparser.ConfigParser()
        config.read(filename)
        sample = Box.from_configuration(config)
        if config.has_option('simulation_data', 'dumping_factor'):
            df = config.get('simulation_data', 'dumping_factor')
        else:
            df = 0.9
        return cls(sample, df)

    def __init__(self, data: Sample, df=0.9):
        self.data: Sample = data
        self.h_sat = 0.0
        self.no = 0
        self.filed = np.array([])
        self.step = 0.0
        self.loops = 0
        self.max_iteration = 0
        self.moment_vectors_history = np.zeros([self.data.no_particles, self.no, 3])
        self.matrix_FORCs = np.zeros([self.no, self.no])
        self.__df = df
        self.__initiated = False

    @property
    def dumping_factor(self):
        return self.__df

    @dumping_factor.setter
    def dumping_factor(self, value):
        if 0 < value < 1:
            self.__df = value
        else:
            raise ValueError(f"value {value} is out of bound for dumping factor.")

    def initiate_simulator_by_config(self, filename):
        config = configparser.ConfigParser()
        config.read(filename)
        if not config.has_section('simulation_data'):
            raise ValueError("missing 'simulation_data' section.")

        h_sat = float(config.get('simulation_data', 'h_sat'))
        no = int(config.get('simulation_data', 'no'))
        loops = int(config.get('simulation_data', 'loops'))
        if config.has_option('simulation_data', 'max_iteration'):
            max_iteration = config.get('simulation_data', 'max_iteration')
        else:
            max_iteration = 500
        self.initiate_simulator(h_sat, no, loops, max_iteration)

    def initiate_simulator(self, h_sat: float, no: int, loops: int, max_iteration: int):
        print("Simulator Initiated ...")
        self.h_sat = h_sat
        self.no = no
        self.filed = np.linspace(self.h_sat, -self.h_sat, no)
        self.step = self.filed[0] - self.filed[1]
        self.loops = loops
        self.max_iteration = max_iteration
        self.moment_vectors_history = np.zeros([self.data.no_particles, no, 3])
        self.matrix_FORCs = np.zeros([self.no, self.no])
        self.__initiated = True

    def average_forc(self, orientation=None):
        assert self.__initiated, "Simulation is not initiated!"
        forc_average = np.zeros([self.no, self.no])
        plate = np.zeros_like(forc_average)
        plate[:, :] = np.nan
        t = 0
        for i in range(self.loops):
            t1 = time()
            self.data.regenerate()
            if orientation is not None:
                assert len(orientation) == 2
                self.data.easy_axis_orientation(orientation)

            forc_average += self.do_llg_mt(plate)
            t2 = time()
            timing = t2 - t1
            t += timing
            print(f"time of {i + 1}th average: {timing:.2f}")

        print(f"time: {t}")
        self.matrix_FORCs = forc_average / self.loops

    def prepare_llg_history(self):
        moment_unit_vectors = np.zeros(self.data.positions.shape)
        moment_unit_vectors[:, 2] = 1  # directed in the z-direction
        self.moment_vectors_history = prepare_llg_history(self.no, self.step, self.h_sat,
                                                          self.data.positions,
                                                          moment_unit_vectors, self.data.jacobian,
                                                          self.data.hk, self.max_iteration,
                                                          self.dumping_factor,
                                                          self.data.easy_axis_unit_vectors,
                                                          self.data.volumes,
                                                          self.data.ms, self.moment_vectors_history)

    def do_llg_mt(self, matrix2):
        self.prepare_llg_history()
        results = llg_forc(self.no, self.h_sat, self.step, self.moment_vectors_history, self.data.jacobian,
                           np.zeros((self.data.no_particles, 3)), self.data.no_particles,
                           self.data.easy_axis_unit_vectors, self.data.hk, self.data.volumes,
                           np.zeros((self.data.no_particles, 4)), self.data.ms, matrix2, self.max_iteration,
                           self.dumping_factor)
        return results

    def plot_curves(self):

        # get x and y from the data
        x = np.linspace(-self.h_sat, self.h_sat, self.no)
        y = self.matrix_FORCs

        # get keyword arguments
        xlabel = "H (Oe)"
        ylabel = "M (a.u.)"

        # get the upper and lower rage of axes
        # X-axis
        lower_x = 1.2 * x.min()
        upper_x = 1.2 * x.max()
        # Y-axis
        finite_index = np.where(np.isfinite(y))
        lower_y = 1.2 * y[finite_index].min()
        upper_y = 1.2 * y[finite_index].max()

        # plot the data
        fig, axs = plt.subplots()
        axs: Axes
        fig: Figure
        for curve in y:
            line = axs.plot(x, curve, 'b')
        # **** adjust axes ****
        # adjust X-axis
        axs.set_xlim(lower_x, upper_x)
        axs.hlines(0, lower_x, upper_x, 'k')
        # adjust Y-axis
        axs.set_ylim(lower_y, upper_y)
        axs.vlines(0, lower_y, upper_y, 'k')
        # **** adjust the graph ****
        axs.set_title('First-Order Reversal Curves')
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel)
        if self.data.name:
            line[0].set_label(self.data.name)
            axs.legend()
        # **** show the graph ****
        plt.show()

    def plot_FORC(self):
        pass

    def save(self, path=None, **kwargs):  # fixme
        name = self.data.name
        if path is not None:
            filename = path + name
        else:
            filename = name

        data = kwargs.get('data', self.matrix_FORCs)
        info = kwargs.get('info', False)

        x = np.flip(self.filed)

        with open(filename + '.frc', 'w') as frc_file:
            text = ''
            """content of the frc file."""
            for row in data:
                points = [x, row]
                points = np.array(points).T
                """[h,m] in every row"""
                for p in points:
                    if not np.isnan(p[1]):
                        text += f'{p[0]},{p[1]}\n'
                text += '\n'
            text += 'END'
            frc_file.write(text)
        print(f"{filename}.frc is saved")

        if info:  # fixme
            with open(filename + '.gd', 'w') as gd_file:
                if isinstance(self.data, _c):
                    text = f"""
                    [gen
                    {self.data.no_particles}
                    """
                    gd_file.write(text)

    def write_info(self):
        config = configparser.ConfigParser()
        config['sample'] = {'type': str(self.data)}
        config['generation_data'] = self.data.get_info()
        if self.__initiated:
            config['simulation_data'] = {
                "h_sat": str(self.h_sat),
                'no': self.no,
                'loops': self.loops,
                'max_iteration': self.max_iteration
            }
        with open(self.data.name + '.ini', 'w') as config_file:
            config.write(config_file)
        print(f"configuration {self.data.name}.ini has been saved!")
