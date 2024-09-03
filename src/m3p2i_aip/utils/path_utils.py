import os, yaml
import m3p2i_aip

def get_package_path():
    package_path = os.path.dirname(m3p2i_aip.__file__)
    return package_path

def get_assets_path():
    package_path = get_package_path()
    path = os.path.join(package_path,'assets')
    return path

def get_config_path():
    package_path = get_package_path()
    path = os.path.join(package_path,'config/')
    return path

def get_plot_path():
    scripts_path = get_package_path()
    path = os.path.join(scripts_path,'plot')
    return path

def load_yaml(file_path):
    with open(file_path) as file:
        yaml_params = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_params