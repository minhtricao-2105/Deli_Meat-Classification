import os
import json

def load_tune_config():
    config_path = os.path.join(os.getcwd(), "utils", "tune_config.json")

    if not os.path.exists(config_path):
        raise Exception(f"{config_path} does not exists")
    
    with open(config_path, "r") as config_file:
        tune_config = json.load(config_file)

    return tune_config

def load_main_config():
    config_path = os.path.join(os.getcwd(), "config.json")
    config = load_config_file(config_path)

    return config

def load_config_file(config_path):
    if not os.path.exists(config_path):
        raise Exception(f"{config_path} does not exists")
    
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    return config

def save_config_file(config, json_path):
    with open(json_path, "w") as json_file:
        json.dump(config, json_file)

