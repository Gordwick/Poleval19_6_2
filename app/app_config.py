import os
import torch

# Config that serves all environment
GLOBAL_CONFIG = {
    "MODEL_PATH": "../model.bin",
}

# Environment specific config, or overwrite of GLOBAL_CONFIG
ENV_CONFIG = {
    "development": {
        "DEBUG": True
    },

    "staging": {
        "DEBUG": True
    },

    "production": {
        "DEBUG": False,
    }
}


MODEL_CONFIG = {
"MAX_LEN": 92,
"PRETRAINED_MODEL": 'allegro/herbert-base-cased',
"CATEGORIZATION_CLASSES": ("non-harmful", "cyberbullying", "hate-speech"),
"DEVICE": 'cpu'
}


def get_config() -> dict:
    # Determine running environment
    ENV = os.environ['PYTHON_ENV'] if 'PYTHON_ENV' in os.environ else 'development'
    ENV = ENV or 'development'

    # raise error if environment is not expected
    if ENV not in ENV_CONFIG:
        raise EnvironmentError(f'Config for envirnoment {ENV} not found')

    config = GLOBAL_CONFIG.copy()
    config.update(ENV_CONFIG[ENV])
    config.update(MODEL_CONFIG)

    config['ENV'] = ENV

    return config

# load config for import
CONFIG = get_config()