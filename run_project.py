import argparse
from mlops.Experiment import Experiment


def run_project(in_args):
    # create Experiment
    exp = Experiment(config_path=in_args.config_path, use_localhost=in_args.use_localhost)

    # run Experiment
    exp.run(docker_args={},
            entry_point=in_args.entry_point)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MLflow project directory')
    parser.add_argument('-e', dest='entry_point', action='store', default='main',
                        help='entry point to project')
    parser.add_argument('-c', dest='config_path', action='store', default='config/config.cfg',
                        help='entry point to project')
    parser.add_argument('-l', dest='use_localhost', action='store_true', default=False,
                        help='use localhost flag')
    args = parser.parse_args()

    run_project(args)
