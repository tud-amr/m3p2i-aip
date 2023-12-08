from pathlib import Path
import os
import yaml

# def get_module_path():
#     path = os.path.dirname(__file__)
#     return path

def get_package_path():
    curr_path = Path().absolute()
    list = str(curr_path).split('/')
    package_path = ''
    for i in list:
        package_path += i + "/"
        if i == "m3p2i-aip":
            break
    return package_path

def get_assets_path():
    package_path = get_package_path()
    path = os.path.join(package_path,'assets')
    return path

def get_scripts_path():
    package_path = get_package_path()
    path = os.path.join(package_path,'scripts')
    return path

def get_params_path():
    package_path = get_package_path()
    path = os.path.join(package_path,'params')
    return path

def get_plot_path():
    scripts_path = get_scripts_path()
    path = os.path.join(scripts_path,'plot')
    return path

def load_yaml(file_path):
    with open(file_path) as file:
        yaml_params = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_params