import argparse
from params import params_point, params_boxer, params_heijn

def load_params():
    # Read arguments from the command line
    parser = argparse.ArgumentParser(prog='Reactive TAMP', description='pass args')
    parser.add_argument('--robot', type=str, default='point', help='Robot to start')
    parser.add_argument('--task', type=str, default='simple', help='Task to start')
    args = parser.parse_args()
    print("The specified robot is a", args.robot, "robot")
    # Choose which parameter file to load
    if args.robot == "point":
        params = params_point
    elif args.robot == "boxer":
        params = params_boxer
    elif args.robot == "heijn":
        params = params_heijn
    params.task = args.task
    return params