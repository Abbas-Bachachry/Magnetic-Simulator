import sys
import time
from physicalSystems import Box, FORCSimulator
from configuration import parse_args

configs = []
display = []


def forcSimulator():
    global configs, display
    print('starting the simulation ...')
    for config in configs:
        simulator = FORCSimulator.from_config(config)
        simulator.initiate_simulator_by_config(config)
        # ['box', 'dist', 'curves', 'plate']
        print(display)
        for task in display:
            if task == 'box':
                pass
            elif task == 'dist':
                pass
            elif task == 'curves':
                pass
            elif task == 'plate':
                print('development')
            else:
                raise ValueError


if __name__ == '__main__':
    # _, packing_intra, name = sys.argv
    args = parse_args()
    if args.configs:
        configs = args.configs
    if args.no_display:
        display = []
    else:
        display = args.display

    forcSimulator()
    # packing_intra = float(packing_intra)
    # print(type(packing_intra), packing_intra)
    # print(type(name), name)
    #
    # sample = Box.clusters(packing_intra, no_cluster=20, particle_per_cluster=20, particle_size=20)
    # sample = Box.random_particles()
    # sample.name = name
    # print(sample.name)
    # orientation = (0, 0)
    # sample.easy_axis_orientation(orientation)
    #
    # simulator = FORCSimulator(sample)
    # simulator = FORCSimulator.from_config(f'{name}.ini')
    # print(simulator.data.no_particles)
    # sample.draw_coercivity_dist()
    # sample.display(.5, background=False, filename=sample.name)

    # simulator.initiate_simulator(0.3, 200, 100, 1000)
    # print(f"started at: {time.strftime('%H:%M:%S')}")
    # simulator.average_forc(orientation)
    #
    # simulator.plot_FORC()
    # simulator.plot_curves()
    print("finished")

    # simulator.save(info=True)
    # simulator.write_info()
