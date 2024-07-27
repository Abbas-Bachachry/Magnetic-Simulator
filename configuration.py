import argparse
import configparser
import os


def check_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    assert config.has_section('sample'), f'invalid configuration file {filename}.'
    assert config.has_option('sample', 'type'), f'invalid configuration file {filename}. missing type option'
    assert config.has_section('generation_data'), f'invalid configuration file {filename}.'
    has_simulation_data = config.has_section('simulation_data')

    sample_type = config.get('sample', 'type').capitalize()
    if sample_type == 'Clusters':
        assert config.has_option('generation_data', 'packing_intra'), f'invalid configuration file {filename}. ' \
                                                                      f'missing packing_intra option'
        assert config.has_option('generation_data', 'no_cluster'), f'invalid configuration file {filename}. ' \
                                                                   'missing no_cluster option'
        assert config.has_option('generation_data', 'particle_per_cluster'), f'invalid configuration file {filename}. ' \
                                                                             'missing particle_per_cluster option'
    elif sample_type == 'Random':
        pass  # fixme

    if has_simulation_data:
        assert config.has_option('simulation_data', 'h_sat'), f'invalid configuration file {filename}. ' \
                                                              'missing h_sat option'
        assert config.has_option('simulation_data', 'no'), f'invalid configuration file {filename}. ' \
                                                           'missing no (Number of FORCs) option'
        assert config.has_option('simulation_data', 'loops'), f'invalid configuration file {filename}. ' \
                                                              'missing loops option'


def parse_args():
    parser = argparse.ArgumentParser(description='Magnetic Simulator')

    # Config
    parser.add_argument('-c', '--configs', dest='configs', nargs='*', required=False, help='config file to use')
    #todo: add anme

    # Load arguments
    # parser.add_argument('--load-file', dest='load_file', required=False,
    #                     help='/path/to/population that you want to load individuals from')
    # parser.add_argument('--load-inds', dest='load_inds', required=False,
    #                     help='[start,stop] (inclusive) or ind1,ind2,... that you wish to load from the file')
    # No display
    display_choices = ['box', 'dist', 'curves', 'plate']
    parser.add_argument('--display', dest='display', required=False, default=['box', 'dist'],
                        choices=display_choices,
                        help=f'set what need to be display. options: {display_choices}')
    parser.add_argument('--no-display', dest='no_display', required=False, default=False, action='store_true',
                        help='If set, there will be no graphics displayed.')
    # Debug
    parser.add_argument('--debug', dest='debug', required=False, default=False, action='store_true',
                        help='If set, certain debug messages will be printed.')

    args = parser.parse_args()

    # check config
    if bool(args.configs):
        configs = args.configs
        for config in configs:
            if not os.path.exists(config):
                parser.error(f'invalid filename {config}')
            check_config(config)

    return args
