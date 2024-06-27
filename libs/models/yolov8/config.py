import hashlib
from collections import OrderedDict

from libs.common.config import load_config


def create_model_config(config, model_type, options=None):
    common_config = None if 'common' not in config else config.common

    if options is not None:
        model_config_str = create_model_config_with_options(options)
        uid = get_model_uid(options)
    else:
        VALID_MODEL_CONFIGS = {
            'yolov8-nano': yolov8_nano_config_str,
            'yolov8-small': yolov8_small_config_str,
            'yolov8-medium': yolov8_medium_config_str,
            'yolov8-large': yolov8_large_config_str
        }
        model_config_str = VALID_MODEL_CONFIGS[model_type]
        uid = get_model_uid(get_default_options()[model_type])
    model_config = load_config(model_config_str, common_config)

    return model_config, uid, model_config_str


def get_model_uid(model_options):
    key_strs = '#'.join([
        str(model_options['common']['depth']),
        str(model_options['common']['width']),
        model_options['common']['feat_channels'],
        str(model_options['backbone']['depthwise']),
        model_options['backbone']['act'],
        str(model_options['fpn']['depthwise']),
        model_options['fpn']['act'],
        str(model_options['head']['depthwise']),
        model_options['head']['act']
        ])
    m = hashlib.sha1()
    m.update(key_strs.encode('utf-8'))
    return m.hexdigest()


def get_model_options():
    return {
        'common': {
            'depth': 0.33,
            'width': 0.25,
            'feat_channels': '[128, 256, 512, 1024]',
        },
        'backbone': {
            'depthwise': False,
            'act': ['silu', 'relu', 'relu6', 'lrelu', 'sigmoid', 'hardsigmoid']
        },
        'fpn': {
            'depthwise': False,
            'act': ['silu', 'relu', 'relu6', 'lrelu', 'sigmoid', 'hardsigmoid']
        },
        'head': {
            'depthwise': False,
            'act': ['silu', 'relu', 'relu6', 'lrelu', 'sigmoid', 'hardsigmoid']
        }
    }


yolov8_nano_relu = {
    'common': {
        'depth': 0.33,
        'width': 0.25,
        'feat_channels': '[128, 256, 512, 1024]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'relu'
    },
    'fpn': {
        'depthwise': False,
        'act': 'relu'
    },
    'head': {
        'depthwise': False,
        'act': 'relu'
    }
}

yolov8_nano = {
    'common': {
        'depth': 0.33,
        'width': 0.25,
        'feat_channels': '[128, 256, 512, 1024]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'silu'
    },
    'fpn': {
        'depthwise': False,
        'act': 'silu'
    },
    'head': {
        'depthwise': False,
        'act': 'silu'
    }
}

yolov8_small_relu = {
    'common': {
        'depth': 0.33,
        'width': 0.50,
        'feat_channels': '[128, 256, 512, 1024]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'relu'
    },
    'fpn': {
        'depthwise': False,
        'act': 'relu'
    },
    'head': {
        'depthwise': False,
        'act': 'relu'
    }
}

yolov8_small = {
    'common': {
        'depth': 0.33,
        'width': 0.50,
        'feat_channels': '[128, 256, 512, 1024]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'silu'
    },
    'fpn': {
        'depthwise': False,
        'act': 'silu'
    },
    'head': {
        'depthwise': False,
        'act': 'silu'
    }
}


yolov8_medium_relu = {
    'common': {
        'depth': 0.67,
        'width': 0.75,
        'feat_channels': '[128, 256, 512, 1024]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'relu'
    },
    'fpn': {
        'depthwise': False,
        'act': 'relu'
    },
    'head': {
        'depthwise': False,
        'act': 'relu'
    }
}

yolov8_medium = {
    'common': {
        'depth': 0.67,
        'width': 0.75,
        'feat_channels': '[128, 256, 512, 1024]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'silu'
    },
    'fpn': {
        'depthwise': False,
        'act': 'silu'
    },
    'head': {
        'depthwise': False,
        'act': 'silu'
    }
}

yolov8_large_relu = {
    'common': {
        'depth': 1,
        'width': 1,
        'feat_channels': '[128, 256, 512, 1024]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'relu'
    },
    'fpn': {
        'depthwise': False,
        'act': 'relu'
    },
    'head': {
        'depthwise': False,
        'act': 'relu'
    }
}

yolov8_large = {
    'common': {
        'depth': 1,
        'width': 1,
        'feat_channels': '[128, 256, 512, 1024]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'silu'
    },
    'fpn': {
        'depthwise': False,
        'act': 'silu'
    },
    'head': {
        'depthwise': False,
        'act': 'silu'
    }
}

def get_recoment_options():
    return OrderedDict({
        'yolov8_nano_relu': yolov8_nano_relu,
        'yolov8_small_relu': yolov8_small_relu,
        'yolov8_medium_relu': yolov8_medium_relu,
        'yolov8_large_relu': yolov8_large_relu,
    })


def get_default_options():
    return OrderedDict({
        'yolov8-nano': yolov8_nano,
        'yolov8-small': yolov8_small,
        'yolov8-medium': yolov8_medium,
        'yolov8-large': yolov8_large,
    })


def create_model_config_with_options(options):
    return f"""
phase: phase
task_type: task_type
catenum: catenum
input_channel: 3
depth: {options['common']['depth']}
width: {options['common']['width']}
feat_channels: {options['common']['feat_channels']}
out_idxes: [1, 2, 3]
backbone:
    depthwise: {options['backbone']['depthwise']}
    act: {options['backbone']['act']}
fpn:
    depthwise: {options['fpn']['depthwise']}
    act: {options['fpn']['act']}
head:
    depthwise: {options['head']['depthwise']}
    act: {options['head']['act']}
"""



yolov8_nano_config_str = """
phase: phase
task_type: task_type
catenum: catenum
input_channel: 3
depth: 0.33
width: 0.25
feat_channels: [128, 256, 512, 1024]
out_idxes: [1, 2, 3]
backbone:
    depthwise: False
    act: silu
fpn:
    depthwise: False
    act: silu
head:
    depthwise: False
    act: silu
"""


yolov8_small_config_str = """
phase: phase
task_type: task_type
catenum: catenum
input_channel: 3
depth: 0.33
width: 0.50
feat_channels: [128, 256, 512, 1024]
out_idxes: [1, 2, 3]
backbone:
    depthwise: False
    act: silu
fpn:
    depthwise: False
    act: silu
head:
    depthwise: False
    act: silu

"""



yolov8_medium_config_str = """
phase: phase
task_type: task_type
catenum: catenum
input_channel: 3
depth: 0.67
width: 0.75
feat_channels: [128, 256, 512, 1024]
out_idxes: [1, 2, 3]
backbone:
    depthwise: False
    act: silu
fpn:
    depthwise: False
    act: silu
head:
    depthwise: False
    act: silu
"""



yolov8_large_config_str = """
phase: phase
task_type: task_type
catenum: catenum
input_channel: 3
depth: 1.00
width: 1.00
feat_channels: [128, 256, 512, 1024]
out_idxes: [1, 2, 3]
backbone:
    depthwise: False
    act: silu
fpn:
    depthwise: False
    act: silu
head:
    depthwise: False
    act: silu
"""



