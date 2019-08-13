# !/usr/bin/env python

import yaml

def load_config():
    """ Load config data from config.yaml """

    with open("config.yaml", 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
