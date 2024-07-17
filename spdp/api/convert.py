import os
import sys
import argparse


sys.path.append('../..')
from spdp.common.config import load_settings, save_settings


parser = argparse.ArgumentParser(description='')
parser.add_argument('--workdir', default='', type=str, help='')
parser.add_argument('--aes_key', default='', type=str, help='')

args = parser.parse_args()


def convert_settings(filepath, aes_key):
    try:
        save_settings(filepath,
                      load_settings(filepath, aes_key=bytes(aes_key, encoding='utf8')),
                      aes_key=b'SPEEDDPHF1234567')
        print(f'convert succeed with {filepath}')
    except:
        print(f'convert failed with {filepath}')

def convert(workdir, aes_key):
    prj_path = os.path.join(workdir, '.prj')
    convert_settings(prj_path, aes_key)
    for expname in os.listdir(workdir):
        expdir = os.path.join(workdir, expname)
        if not os.path.isdir(expdir): continue
        train_path = os.path.join(expdir, '.train')
        convert_settings(train_path, aes_key)


if __name__ == '__main__':
    convert(args.workdir, args.aes_key)