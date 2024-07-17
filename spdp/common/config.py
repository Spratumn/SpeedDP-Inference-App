import copy
import io
import os
import numpy as np
from ast import literal_eval
import yaml
from Crypto.Util.Padding import pad, unpad
from Crypto.Cipher import AES


_YAML_EXTS = {"", ".yaml", ".yml"}
_EXTS = {".py"}

_FILE_TYPES = (io.IOBase,)
_VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}
import importlib.util


class CfgNode(dict):
    IMMUTABLE = "__immutable__"
    DEPRECATED_KEYS = "__deprecated_keys__"
    RENAMED_KEYS = "__renamed_keys__"
    NEW_ALLOWED = "__new_allowed__"

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        init_dict = self._create_config_tree_from_dict(init_dict, key_list)
        super(CfgNode, self).__init__(init_dict)
        self.__dict__[CfgNode.IMMUTABLE] = False
        self.__dict__[CfgNode.DEPRECATED_KEYS] = set()
        self.__dict__[CfgNode.RENAMED_KEYS] = {}
        self.__dict__[CfgNode.NEW_ALLOWED] = new_allowed

    @classmethod
    def _create_config_tree_from_dict(cls, dic, key_list):
        dic = copy.deepcopy(dic)
        for k, v in dic.items():
            if isinstance(v, dict):
                dic[k] = cls(v, key_list=key_list + [k])
            else:
                _assert_with_logging(
                    _valid_type(v, allow_cfg_node=False),
                    "Key {} with value {} is not a valid type; valid types: {}".format(
                        ".".join(key_list + [k]), type(v), _VALID_TYPES
                    ),
                )
        return dic

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if self.is_frozen():
            raise AttributeError(
                "Attempted to set {} to {}, but CfgNode is immutable".format(
                    name, value
                )
            )

        _assert_with_logging(
            name not in self.__dict__,
            "Invalid attempt to modify internal CfgNode state: {}".format(name),
        )
        _assert_with_logging(
            _valid_type(value, allow_cfg_node=True),
            "Invalid type {} for key {}; valid types = {}".format(
                type(value), name, _VALID_TYPES
            ),
        )
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())

    def dump(self, **kwargs):
        def convert_to_dict(cfg_node, key_list):
            if not isinstance(cfg_node, CfgNode):
                _assert_with_logging(
                    _valid_type(cfg_node),
                    "Key {} with value {} is not a valid type; valid types: {}".format(
                        ".".join(key_list), type(cfg_node), _VALID_TYPES
                    ),
                )
                return cfg_node
            else:
                cfg_dict = dict(cfg_node)
                for k, v in cfg_dict.items():
                    cfg_dict[k] = convert_to_dict(v, key_list + [k])
                return cfg_dict

        self_as_dict = convert_to_dict(self, [])
        return yaml.safe_dump(self_as_dict, **kwargs)

    def merge_from_file(self, cfg_filename):
        """Load a yaml config file and merge it this CfgNode."""
        with open(cfg_filename, "r") as f:
            cfg = self.load_cfg(f)
        self.merge_from_other_cfg(cfg)

    def merge_from_yml_str(self, cfg_str):
        """Load a yaml config str and merge it this CfgNode."""
        cfg = self.load_cfg(cfg_str)
        self.merge_from_other_cfg(cfg)

    def merge_from_other_cfg(self, cfg_other):
        """Merge `cfg_other` into this CfgNode."""
        _merge_a_into_b(cfg_other, self, self, [])

    def merge_from_list(self, cfg_list):
        """Merge config (keys, values) in a list (e.g., from command line) into
        this CfgNode. For example, `cfg_list = ['FOO.BAR', 0.5]`.
        """
        _assert_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; it must be a list of pairs".format(
                cfg_list
            ),
        )
        root = self
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            if root.key_is_deprecated(full_key):
                continue
            if root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            key_list = full_key.split(".")
            d = self
            for subkey in key_list[:-1]:
                _assert_with_logging(
                    subkey in d, "Non-existent key: {}".format(full_key)
                )
                d = d[subkey]
            subkey = key_list[-1]
            _assert_with_logging(subkey in d, "Non-existent key: {}".format(full_key))
            value = self._decode_cfg_value(v)
            value = _check_and_coerce_cfg_value_type(value, d[subkey], subkey, full_key)
            d[subkey] = value

    def freeze(self):
        """Make this CfgNode and all of its children immutable."""
        self._immutable(True)

    def defrost(self):
        """Make this CfgNode and all of its children mutable."""
        self._immutable(False)

    def is_frozen(self):
        """Return mutability."""
        return self.__dict__[CfgNode.IMMUTABLE]

    def _immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested CfgNodes.
        """
        self.__dict__[CfgNode.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, CfgNode):
                v._immutable(is_immutable)
        for v in self.values():
            if isinstance(v, CfgNode):
                v._immutable(is_immutable)

    def clone(self):
        """Recursively copy this CfgNode."""
        return copy.deepcopy(self)

    def register_deprecated_key(self, key):
        _assert_with_logging(
            key not in self.__dict__[CfgNode.DEPRECATED_KEYS],
            "key {} is already registered as a deprecated key".format(key),
        )
        self.__dict__[CfgNode.DEPRECATED_KEYS].add(key)

    def register_renamed_key(self, old_name, new_name, message=None):
        _assert_with_logging(
            old_name not in self.__dict__[CfgNode.RENAMED_KEYS],
            "key {} is already registered as a renamed cfg key".format(old_name),
        )
        value = new_name
        if message:
            value = (new_name, message)
        self.__dict__[CfgNode.RENAMED_KEYS][old_name] = value

    def key_is_deprecated(self, full_key):
        """Test if a key is deprecated."""
        if full_key in self.__dict__[CfgNode.DEPRECATED_KEYS]:
            return True
        return False

    def key_is_renamed(self, full_key):
        """Test if a key is renamed."""
        return full_key in self.__dict__[CfgNode.RENAMED_KEYS]

    def raise_key_rename_error(self, full_key):
        new_key = self.__dict__[CfgNode.RENAMED_KEYS][full_key]
        if isinstance(new_key, tuple):
            msg = " Note: " + new_key[1]
            new_key = new_key[0]
        else:
            msg = ""
        raise KeyError(
            "Key {} was renamed to {}; please update your config.{}".format(
                full_key, new_key, msg
            )
        )

    def is_new_allowed(self):
        return self.__dict__[CfgNode.NEW_ALLOWED]

    @classmethod
    def load_cfg(cls, cfg_file_obj_or_str):
        _assert_with_logging(
            isinstance(cfg_file_obj_or_str, _FILE_TYPES + (str,)),
            "Expected first argument to be of type {} or {}, but it was {}".format(
                _FILE_TYPES, str, type(cfg_file_obj_or_str)
            ),
        )
        if isinstance(cfg_file_obj_or_str, str):
            return cls._load_cfg_from_yaml_str(cfg_file_obj_or_str)
        elif isinstance(cfg_file_obj_or_str, _FILE_TYPES):
            return cls._load_cfg_from_file(cfg_file_obj_or_str)
        else:
            raise NotImplementedError("Impossible to reach here (unless there's a bug)")

    @classmethod
    def _load_cfg_from_file(cls, file_obj):
        """Load a config from a YAML file or a Python source file."""
        _, file_extension = os.path.splitext(file_obj.name)
        if file_extension in _YAML_EXTS:
            return cls._load_cfg_from_yaml_str(file_obj.read())
        elif file_extension in _EXTS:
            return cls._load_cfg_source(file_obj.name)
        else:
            raise Exception(
                "Attempt to load from an unsupported file type {}; "
                "only {} are supported".format(file_obj, _YAML_EXTS.union(_EXTS))
            )

    @classmethod
    def _load_cfg_from_yaml_str(cls, str_obj):
        """Load a config from a YAML string encoding."""
        cfg_as_dict = yaml.safe_load(str_obj)
        return cls(cfg_as_dict)

    @classmethod
    def _load_cfg_source(cls, filename):
        """Load a config from a Python source file."""
        module = _load_module_from_file("yacs.config.override", filename)
        _assert_with_logging(
            hasattr(module, "cfg"),
            "Python module from file {} must have 'cfg' attr".format(filename),
        )
        VALID_ATTR_TYPES = {dict, CfgNode}
        _assert_with_logging(
            type(module.cfg) in VALID_ATTR_TYPES,
            "Imported module 'cfg' attr must be in {} but is {} instead".format(
                VALID_ATTR_TYPES, type(module.cfg)
            ),
        )
        return cls(module.cfg)

    @classmethod
    def _decode_cfg_value(cls, value):
        if isinstance(value, dict):
            return cls(value)
        # All remaining processing is only applied to strings
        if not isinstance(value, str):
            return value
        try:
            value = literal_eval(value)
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value


def _valid_type(value, allow_cfg_node=False):
    return (type(value) in _VALID_TYPES) or (
        allow_cfg_node and isinstance(value, CfgNode)
    )


def _merge_a_into_b(a, b, root, key_list):
    _assert_with_logging(
        isinstance(a, CfgNode),
        "`a` (cur type {}) must be an instance of {}".format(type(a), CfgNode),
    )
    _assert_with_logging(
        isinstance(b, CfgNode),
        "`b` (cur type {}) must be an instance of {}".format(type(b), CfgNode),
    )

    for k, v_ in a.items():
        full_key = ".".join(key_list + [k])

        v = copy.deepcopy(v_)
        v = b._decode_cfg_value(v)

        if k in b:
            v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)
            # Recursively merge dicts
            if isinstance(v, CfgNode):
                try:
                    _merge_a_into_b(v, b[k], root, key_list + [k])
                except BaseException:
                    raise
            else:
                b[k] = v
        elif b.is_new_allowed():
            b[k] = v
        else:
            if root.key_is_deprecated(full_key):
                continue
            elif root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            else:
                raise KeyError("Non-existent config key: {}".format(full_key))


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None
    casts = [(tuple, list), (list, tuple)]

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def _assert_with_logging(cond, msg):
    assert cond, msg


def _load_module_from_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config_from_file(config:CfgNode, cfg_path):
    config.defrost()
    config.merge_from_file(cfg_path)
    config.freeze()

def load_config_from_yml_str(config:CfgNode, cfg_str):
    config.defrost()
    config.merge_from_yml_str(cfg_str)
    config.freeze()



def make_project_config(settings, phase):
    config = load_config(project_config_str, None)
    config.defrost()
    config.catenames = list(settings['category_info'].keys())
    config.colormaps = [get_colormap(settings['category_info'][catename]['id']) for catename in config.catenames]
    config.common.phase = phase
    config.common.catenum = len(config.catenames)
    config.common.inputsize = settings['inputsize']
    config.common.task_type = settings.get('task_type', 'det')
    config.common.image_type = settings['image_type']
    config.common.segment_length = settings.get('segment_length', 1000)
    config.freeze()
    merge_dataset_config(config, settings)
    return config


def merge_dataset_config(config, settings):
    config.defrost()
    config.dataset = create_dataset_config(config.common, settings)
    config.dataset.defrost()
    if 'augmenters' in settings:
        config.dataset.targets_padding = settings['augmenters']['targets_padding']
        config.dataset.remove_empty_sample = settings['augmenters']['remove_empty_sample']
        config.dataset.min_size = settings['augmenters']['min_size']
        config.dataset.max_scale = -1 if settings['augmenters']['max_scale'] == 0 else 1 / settings['augmenters']['max_scale']
    config.dataset.manual_cate_dict = [{catename: settings['category_info'][catename]} for catename in settings['category_info']]
    if 'dataset_dirs' not in settings or settings['dataset_dirs'] is None or not settings['dataset_dirs']:
        config.dataset.freeze()
        config.freeze()
        return
    config.dataset.dataset_dirs = settings['dataset_dirs']
    config.dataset.freeze()
    config.freeze()


def load_config(cfg_str, common_params=None, from_file=False):
    cfg = CfgNode(new_allowed=True)
    if from_file:
        load_config_from_file(cfg, cfg_str)
    else:
        load_config_from_yml_str(cfg, cfg_str)
    if common_params is None: return cfg

    cfg.defrost()
    cfg = process_config(cfg, common_params)
    cfg.freeze()
    return cfg


def process_config(cfg_node, common_cfg=dict()) -> CfgNode:
    if common_cfg is None: return cfg_node
    for cfg_key, v in cfg_node.items():
        if isinstance(v, str):
            if v in common_cfg:
                cfg_node[cfg_key] = common_cfg[v]
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], dict):
                    for mainK in v[i]:
                        v[i][mainK] = process_config(v[i][mainK], common_cfg)
        elif isinstance(v, CfgNode):
            cfg_node[cfg_key] = process_config(cfg_node[cfg_key], common_cfg)
    return cfg_node


class Colors:
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

__colormaps = Colors()

def get_colormap(cateid):
    if cateid < 20: return __colormaps(cateid, True)
    idx = cateid * 3 + 1
    return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)


def create_dataset_config(common_config, settings):
    if 'augmenters' in settings and 'data_process' in settings['augmenters']:
        augmenters = create_augmenters_from_settings(settings['augmenters']['data_process'])
    else:
        augmenters = []
    if common_config.phase == 'train':
        if common_config.image_type == 'vl':
            dataset_config = load_config(vl_trainset_config_str, common_config)
        else:
            dataset_config = load_config(ir_trainset_config_str, common_config)
    else:
        dataset_config = load_config(evalset_config_str, common_config)
        augmenters = dataset_config.augmenters + [augmenters[-1]] if augmenters else []
    dataset_config.defrost()
    dataset_config.augmenters = augmenters
    dataset_config.freeze()
    return dataset_config


def create_augmenters_from_settings(settings):
    augmenters = []
    for option in settings:
        if not option.get('enabled', True): continue
        augmenters.append(
            {
               option['name']: {
                   key: value for key, value in option['kwargs']
               }
            }
        )
    return augmenters


def save_settings(filepath, settings, aes_key=b'SPEEDDPHF1234567'):
    cipher = AES.new(aes_key, AES.MODE_CBC, aes_key)
    encrypt_data = cipher.encrypt(pad(str(settings).encode('utf-8'), 16))
    with open(filepath, 'wb') as f:
        f.write(encrypt_data)


def load_settings(filepath, aes_key=b'SPEEDDPHF1234567'):
    with open(filepath, 'rb') as f:
        enc_data = f.read()
    cipher = AES.new(aes_key, AES.MODE_CBC, aes_key)
    assert isinstance(enc_data, bytes)
    decrypt_data = unpad(cipher.decrypt(enc_data), 16)
    return literal_eval(decrypt_data.decode('utf-8'))

#######################################################################################################
########################################### config string #############################################


project_config_str = """
    catenames: None
    colormaps: None
    common:
        phase: None
        inputsize: None
        task_type: None
        image_type: None
        catenum: -1
        xyxy_to_xywh: True
        segment_length: -1
    dataset: -1
"""


vl_trainset_config_str = """
    inputsize: inputsize
    task_type: task_type
    image_type: vl
    min_size: None
    max_scale: None
    remove_empty_sample: None
    targets_padding: None
    xyxy_to_xywh: xyxy_to_xywh
    segment_length: segment_length
    manual_cate_dict: None
    dataset_dirs: -1
    augmenters: []
"""


ir_trainset_config_str = """
    inputsize: inputsize
    task_type: task_type
    image_type: ir
    min_size: None
    max_scale: None
    remove_empty_sample: None
    targets_padding: None
    xyxy_to_xywh: xyxy_to_xywh
    segment_length: segment_length
    manual_cate_dict: None
    dataset_dirs: -1
    augmenters: []
"""

evalset_config_str = """
    inputsize: inputsize
    task_type: task_type
    image_type: vl
    min_size: 1
    max_scale: -1
    remove_empty_sample: False
    targets_padding: 500
    xyxy_to_xywh: xyxy_to_xywh
    segment_length: segment_length
    manual_cate_dict: None
    dataset_dirs: -1
    augmenters:
    - resize:
        outsize: inputsize
        min_size: 1
        max_scale: -1
"""
