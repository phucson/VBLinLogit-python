"""Run all vb_{linear,logit}_example_* scripts."""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import parse_args  # noqa: E402

EXAMPLE_SCRIPTS = [
    'vb_linear_example',
    'vb_linear_example_highdim',
    'vb_linear_example_sparse',
    'vb_linear_example_modelsel',
    'vb_logit_example',
    'vb_logit_example_coeff',
    'vb_logit_example_highdim',
    'vb_logit_example_modelsel',
]


def main(save_dir=None, show=True):
    for name in EXAMPLE_SCRIPTS:
        print(f'Running {name}')
        importlib.import_module(name).main(save_dir=save_dir, show=show)
        print()


if __name__ == '__main__':
    main(**parse_args(__doc__))
