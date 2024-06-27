import os
import shutil
import re
import platform
import sys


if platform.system() == 'Windows':
    LIBRARY_SUFFIX = f"cp{''.join(sys.version.split('.')[:2])}-win_amd64.pyd"
else:
    LIBRARY_SUFFIX = f"cpython-{''.join(sys.version.split('.')[:2])}-x86_64-linux-gnu.so"


def str_alter(file, old_str, new_str):
    with open(file, "r", encoding="utf-8") as f1, open("%s.bak" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            f2.write(re.sub(old_str, new_str, line))
    os.remove(file)
    os.rename("%s.bak" % file, file)


def merge_src_code(src_paths, dst_path, preffix=''):
    if os.path.exists(dst_path): os.remove(dst_path)
    with open(dst_path, 'a', encoding='utf-8') as f:
        if preffix: f.write(preffix)
        for src_path in src_paths:
            if os.path.exists(src_path):
                with open(src_path, 'r', encoding='utf-8') as src_f:
                    f.write(src_f.read())


def compile_pyxes(pyx_dir, dst_lib_dir):
    # create and run setup.py
    os.chdir(pyx_dir)
    for pyx_file in os.listdir('./'):
        if not pyx_file.endswith('pyx'): continue
        libname = pyx_file.split('.pyx')[0]
        setup_str = f"""
from distutils.core import setup
from Cython.Build import cythonize

lib_names = ["{libname}.pyx"]

setup(ext_modules=cythonize(lib_names))
    """
        with open('./setup.py', 'w', encoding='utf-8') as f:
            f.write(setup_str)
        os.system('python ./setup.py build_ext --inplace')
        # # update lib files
        shutil.move(f'./{libname}.{LIBRARY_SUFFIX}', dst_lib_dir)
        shutil.rmtree('./build')


def prepare_common(src_dir, pyx_dir):
    os.chdir(src_dir)
    print('Preparing config..')
    libname = 'lib0x00'
    pyx_path = os.path.join(pyx_dir, f'./{libname}.pyx')
    if os.path.exists(pyx_path): os.remove(pyx_path)
    merge_src_code(['config.py'], pyx_path, preffix='# cython:language_level=3\n')

    print('Preparing dataset..')
    libname = 'lib0x01'
    pyx_path = os.path.join(pyx_dir, f'./{libname}.pyx')
    if os.path.exists(pyx_path): os.remove(pyx_path)
    merge_src_code(['dataset.py'], pyx_path, preffix='# cython:language_level=3\n')


def prepare_api(src_dir, pyx_dir):
    print('Preparing api..')
    libname = 'lib2x00'
    # copy and adjust lib name used in py file
    os.chdir(src_dir)
    pyx_path = os.path.join(pyx_dir, f'./{libname}.pyx')
    if os.path.exists(pyx_path): os.remove(pyx_path)
    merge_src_code(['detect.py'
                    ],
                   pyx_path,
                   preffix='# cython:language_level=3\n')
    str_alter(pyx_path, 'from libs.common.config', 'from .lib0x00')
    str_alter(pyx_path, 'from libs.models.creator', 'from .lib3x00')


def prepare_models(src_dir, pyx_dir):
    os.chdir(src_dir)
    print('Preparing model creator..')
    libname = 'lib3x00'
    pyx_path = os.path.join(pyx_dir, f'./{libname}.pyx')
    if os.path.exists(pyx_path): os.remove(pyx_path)
    merge_src_code(['creator.py'], pyx_path, preffix='# cython:language_level=3\n')
    str_alter(pyx_path, 'from .yolox.config', 'from .lib3x02')
    str_alter(pyx_path, 'from .yolox.loss', 'from .lib3x02')
    str_alter(pyx_path, 'from .yolox.predictor', 'from .lib3x02')
    str_alter(pyx_path, 'from .yolov8.config', 'from .lib3x03')
    str_alter(pyx_path, 'from .yolov8.loss', 'from .lib3x03')
    str_alter(pyx_path, 'from .yolov8.predictor', 'from .lib3x03')

    print('Preparing model base module..')
    libname = 'lib3x01'
    pyx_path = os.path.join(pyx_dir, f'./{libname}.pyx')
    if os.path.exists(pyx_path): os.remove(pyx_path)
    merge_src_code(['module.py'], pyx_path, preffix='# cython:language_level=3\n')

    print('Preparing model yolox..')
    libname = 'lib3x02'
    pyx_path = os.path.join(pyx_dir, f'./{libname}.pyx')
    if os.path.exists(pyx_path): os.remove(pyx_path)
    merge_src_code(['./yolox/config.py',
                    './yolox/model.py',
                    './yolox/predictor.py'
                    ], pyx_path,
                   preffix='# cython:language_level=3\n')
    str_alter(pyx_path, 'from libs.common.config', 'from .lib0x00')
    str_alter(pyx_path, 'from ..module', 'from .lib3x01')
    str_alter(pyx_path, 'from .config import create_model_config', '')
    str_alter(pyx_path, 'from .model import YOLOX', '')
    str_alter(pyx_path, 'from libs.common.dataset', 'from .lib0x01')

    print('Preparing model yolov8..')
    libname = 'lib3x03'
    pyx_path = os.path.join(pyx_dir, f'./{libname}.pyx')
    if os.path.exists(pyx_path): os.remove(pyx_path)
    merge_src_code(['./yolov8/config.py',
                    './yolov8/loss.py',
                    './yolov8/model.py',
                    './yolov8/predictor.py'
                    ], pyx_path,
                   preffix='# cython:language_level=3\n')
    str_alter(pyx_path, 'from libs.common.config', 'from .lib0x00')
    str_alter(pyx_path, 'from ..module', 'from .lib3x01')
    str_alter(pyx_path, 'from .loss import dist2bbox, make_anchors', '')
    str_alter(pyx_path, 'from .loss import non_max_suppression', '')
    str_alter(pyx_path, 'from .config import create_model_config', '')
    str_alter(pyx_path, 'from .model import YOLOV8', '')
    str_alter(pyx_path, 'from libs.common.dataset', 'from .lib0x01')


def build_libs(src_dir, dst_dir):
    pyx_dir = os.path.join(dst_dir, 'pyx')
    if os.path.exists(pyx_dir): shutil.rmtree(pyx_dir)
    os.mkdir(pyx_dir)
    prepare_common(os.path.join(src_dir, 'libs', 'common'), pyx_dir)
    prepare_api(os.path.join(src_dir, 'libs', 'api'), pyx_dir)
    prepare_models(os.path.join(src_dir, 'libs', 'models'), pyx_dir)

    # return
    dst_lib_dir = os.path.abspath(os.path.join(dst_dir, 'spdp'))
    if os.path.exists(dst_lib_dir): shutil.rmtree(dst_lib_dir)
    os.makedirs(dst_lib_dir)
    compile_pyxes(pyx_dir, dst_lib_dir)
    os.chdir(src_dir)
    if os.path.exists(pyx_dir): shutil.rmtree(pyx_dir)



if __name__ == '__main__':
    src_dir = os.path.abspath('./')
    dst_dir = os.path.abspath('./packages')
    if not os.path.exists(dst_dir): os.makedirs(dst_dir)

    build_libs(src_dir, dst_dir)
