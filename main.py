import argparse
from mlops.Experiment import Experiment


def run_project(args):
    exp = Experiment(config_path='config/config.cfg')
    exp.run(docker_args={}, entry_point=args.entry_point)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MLflow project directory')
    parser.add_argument('-e', dest='entry_point', action='store',
                        const=sum, default='main',
                        help='entry point to project')
    args = parser.parse_args()
    run_project(args)
