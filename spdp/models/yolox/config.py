import hashlib
from collections import OrderedDict

from spdp.common.config import load_config


def create_model_config(config, model_type, options=None):
    common_config = None if 'common' not in config else config.common
    if options is not None:
        model_config_str = create_model_config_with_options(options)
        uid = get_model_uid(options)
    else:
        VALID_MODEL_CONFIGS = {
            'yolox-nano': yolox_nano_config_str,
            'yolox-tiny': yolox_tiny_config_str,
            'yolox-small': yolox_small_config_str,
            'yolox-medium': yolox_medium_config_str,
            'yolox-large': yolox_large_config_str
        }
        model_config_str = VALID_MODEL_CONFIGS[model_type]
        uid = get_model_uid(get_default_options()[model_type])
    model_config = load_config(model_config_str, common_config)
    return model_config, uid, model_config_str


def get_model_uid(model_options):
    key_strs = '#'.join([
        str(model_options['common']['depth']),
        str(model_options['common']['width']),
        model_options['common']['out_idxes'],
        model_options['common']['feat_channels'],
        str(model_options['backbone']['depthwise']),
        model_options['backbone']['act'],
        model_options['fpn']['name'],
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
            'out_idxes': '[1, 2, 3]',
        },
        'backbone': {
            'depthwise': False,
            'act': ['silu', 'relu', 'relu6', 'lrelu', 'sigmoid', 'hardsigmoid']
        },
        'fpn': {
            'name': ['YOLOPAFPN', 'YOLOPAN'],
            'depthwise': False,
            'act': ['silu', 'relu', 'relu6', 'lrelu', 'sigmoid', 'hardsigmoid']
        },
        'head': {
            'depthwise': False,
            'act': ['silu', 'relu', 'relu6', 'lrelu', 'sigmoid', 'hardsigmoid']
        }
    }


yolox_nano = {
    'common': {
        'depth': 0.33,
        'width': 0.25,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[1, 2, 3]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'silu'
    },
    'fpn': {
        'name': 'YOLOPAFPN',
        'depthwise': False,
        'act': 'silu'
    },
    'head': {
        'depthwise': False,
        'act': 'silu'
    }
}

yolox_nano_relu = {
    'common': {
        'depth': 0.33,
        'width': 0.25,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[1, 2, 3]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'relu'
    },
    'fpn': {
        'name': 'YOLOPAFPN',
        'depthwise': False,
        'act': 'relu'
    },
    'head': {
        'depthwise': False,
        'act': 'relu'
    }
}

yolox_nano_relu_viz = {
    'common': {
        'depth': 0.33,
        'width': 0.25,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[0, 1, 2, 3]',
    },
    'backbone': {
        'depthwise': True,
        'act': 'relu'
    },
    'fpn': {
        'name': 'YOLOPAN',
        'depthwise': True,
        'act': 'relu'
    },
    'head': {
        'depthwise': True,
        'act': 'relu'
    }
}

yolox_tiny = {
    'common': {
        'depth': 0.33,
        'width': 0.375,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[1, 2, 3]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'silu'
    },
    'fpn': {
        'name': 'YOLOPAFPN',
        'depthwise': False,
        'act': 'silu'
    },
    'head': {
        'depthwise': False,
        'act': 'silu'
    }
}

yolox_tiny_relu = {
    'common': {
        'depth': 0.33,
        'width': 0.375,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[1, 2, 3]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'relu'
    },
    'fpn': {
        'name': 'YOLOPAFPN',
        'depthwise': False,
        'act': 'relu'
    },
    'head': {
        'depthwise': False,
        'act': 'relu'
    }
}

yolox_tiny_relu_viz = {
    'common': {
        'depth': 0.33,
        'width': 0.375,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[0, 1, 2, 3]',
    },
    'backbone': {
        'depthwise': True,
        'act': 'relu'
    },
    'fpn': {
        'name': 'YOLOPAN',
        'depthwise': True,
        'act': 'relu'
    },
    'head': {
        'depthwise': True,
        'act': 'relu'
    }
}

yolox_small = {
    'common': {
        'depth': 0.33,
        'width': 0.5,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[1, 2, 3]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'silu'
    },
    'fpn': {
        'name': 'YOLOPAFPN',
        'depthwise': False,
        'act': 'silu'
    },
    'head': {
        'depthwise': False,
        'act': 'silu'
    }
}

yolox_small_relu = {
    'common': {
        'depth': 0.33,
        'width': 0.5,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[1, 2, 3]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'relu'
    },
    'fpn': {
        'name': 'YOLOPAFPN',
        'depthwise': False,
        'act': 'relu'
    },
    'head': {
        'depthwise': False,
        'act': 'relu'
    }
}

yolox_small_relu_viz = {
    'common': {
        'depth': 0.33,
        'width': 0.5,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[0, 1, 2, 3]',
    },
    'backbone': {
        'depthwise': True,
        'act': 'relu'
    },
    'fpn': {
        'name': 'YOLOPAN',
        'depthwise': True,
        'act': 'relu'
    },
    'head': {
        'depthwise': True,
        'act': 'relu'
    }
}

yolox_medium = {
    'common': {
        'depth': 0.67,
        'width': 0.75,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[1, 2, 3]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'silu'
    },
    'fpn': {
        'name': 'YOLOPAFPN',
        'depthwise': False,
        'act': 'silu'
    },
    'head': {
        'depthwise': False,
        'act': 'silu'
    }
}
yolox_medium_relu = {
    'common': {
        'depth': 0.67,
        'width': 0.75,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[1, 2, 3]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'relu'
    },
    'fpn': {
        'name': 'YOLOPAFPN',
        'depthwise': False,
        'act': 'relu'
    },
    'head': {
        'depthwise': False,
        'act': 'relu'
    }
}

yolox_medium_relu_viz  = {
    'common': {
        'depth': 0.67,
        'width': 0.75,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[1, 2, 3]',
    },
    'backbone': {
        'depthwise': True,
        'act': 'relu'
    },
    'fpn': {
        'name': 'YOLOPAN',
        'depthwise': True,
        'act': 'relu'
    },
    'head': {
        'depthwise': True,
        'act': 'relu'
    }
}

yolox_large_relu = {
    'common': {
        'depth': 1,
        'width': 1,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[1, 2, 3]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'relu'
    },
    'fpn': {
        'name': 'YOLOPAFPN',
        'depthwise': False,
        'act': 'relu'
    },
    'head': {
        'depthwise': False,
        'act': 'relu'
    }
}

yolox_large = {
    'common': {
        'depth': 1,
        'width': 1,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[1, 2, 3]',
    },
    'backbone': {
        'depthwise': False,
        'act': 'silu'
    },
    'fpn': {
        'name': 'YOLOPAFPN',
        'depthwise': False,
        'act': 'silu'
    },
    'head': {
        'depthwise': False,
        'act': 'silu'
    }
}

yolox_large_relu_viz  = {
    'common': {
        'depth': 1,
        'width': 1,
        'feat_channels': '[128, 256, 512, 1024]',
        'out_idxes': '[0, 1, 2, 3]',
    },
    'backbone': {
        'depthwise': True,
        'act': 'relu'
    },
    'fpn': {
        'name': 'YOLOPAN',
        'depthwise': True,
        'act': 'relu'
    },
    'head': {
        'depthwise': True,
        'act': 'relu'
    }
}


def get_recoment_options():
    return OrderedDict({
        'yolox_nano_relu': yolox_nano_relu,
        'yolox_nano_relu_viz': yolox_nano_relu_viz,
        'yolox_tiny_relu': yolox_tiny_relu,
        'yolox_tiny_relu_viz': yolox_tiny_relu_viz,
        'yolox_small_relu': yolox_small_relu,
        'yolox_small_relu_viz': yolox_small_relu_viz,
        'yolox_medium_relu': yolox_medium_relu,
        'yolox_medium_relu_viz': yolox_medium_relu_viz,
        'yolox_large_relu': yolox_large_relu,
        'yolox_large_relu_viz': yolox_large_relu_viz,
    })


def get_default_options():
    return OrderedDict({
        'yolox-nano': yolox_nano,
        'yolox-tiny': yolox_tiny,
        'yolox-small': yolox_small,
        'yolox-medium': yolox_medium,
        'yolox-large': yolox_large,
    })


def create_model_config_with_options(options):
    return f"""
phase: phase
catenum: catenum
input_channel: 3
depth: {options['common']['depth']}
width: {options['common']['width']}
feat_channels: {options['common']['feat_channels']}
out_idxes: {options['common']['out_idxes']}
backbone:
    name: CSPDarknet
    depthwise: {options['backbone']['depthwise']}
    act: {options['backbone']['act']}
fpn:
    name: {options['fpn']['name']}
    depthwise: {options['fpn']['depthwise']}
    act: {options['fpn']['act']}
head:
    name: YOLOXHead
    depthwise: {options['head']['depthwise']}
    act: {options['head']['act']}
"""



yolox_nano_config_str = """
phase: phase
catenum: catenum
input_channel: 3
depth: 0.33
width: 0.25
feat_channels: [128, 256, 512, 1024]
out_idxes: [1, 2, 3]
backbone:
    name: CSPDarknet
    depthwise: False
    act: silu
fpn:
    name: YOLOPAFPN
    depthwise: False
    act: silu
head:
    name: YOLOXHead
    depthwise: False
    act: silu
"""

yolox_tiny_config_str = """
phase: phase
catenum: catenum
input_channel: 3
depth: 0.33
width: 0.375
feat_channels: [128, 256, 512, 1024]
out_idxes: [1, 2, 3]
backbone:
    name: CSPDarknet
    depthwise: False
    act: silu
fpn:
    name: YOLOPAFPN
    depthwise: False
    act: silu
head:
    name: YOLOXHead
    depthwise: False
    act: silu
"""

yolox_small_config_str = """
phase: phase
catenum: catenum
input_channel: 3
depth: 0.33
width: 0.5
feat_channels: [128, 256, 512, 1024]
out_idxes: [1, 2, 3]
backbone:
    name: CSPDarknet
    depthwise: False
    act: silu
fpn:
    name: YOLOPAFPN
    depthwise: False
    act: silu
head:
    name: YOLOXHead
    depthwise: False
    act: silu
"""

yolox_medium_config_str = """
phase: phase
catenum: catenum
input_channel: 3
depth: 0.67
width: 0.75
feat_channels: [128, 256, 512, 1024]
out_idxes: [1, 2, 3]
backbone:
    name: CSPDarknet
    depthwise: False
    act: silu
fpn:
    name: YOLOPAFPN
    depthwise: False
    act: silu
head:
    name: YOLOXHead
    depthwise: False
    act: silu
"""

yolox_large_config_str = """
phase: train
catenum: catenum
input_channel: 3
depth: 1.0
width: 1.0
feat_channels: [128, 256, 512, 1024]
out_idxes: [1, 2, 3]
backbone:
    name: CSPDarknet
    depthwise: False
    act: silu
fpn:
    name: YOLOPAFPN
    depthwise: False
    act: silu
head:
    name: YOLOXHead
    depthwise: False
    act: silu
"""
