## script to run all vb_{linar,logit}_example_* scripts
import os
import runpy

import matplotlib.pyplot as plt

example_scripts = [
    'vb_linear_example.py',
    'vb_linear_example_highdim.py',
    'vb_linear_example_sparse.py',
    'vb_linear_example_modelsel.py',
    'vb_logit_example.py',
    'vb_logit_example_coeff.py',
    'vb_logit_example_highdim.py',
    'vb_logit_example_modelsel.py']

for i in range(len(example_scripts)):
    script_name = example_scripts[i]
    print('Running %s' % script_name)
    ws = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name))
    if 'f1' in ws:
        nf1 = ws['f1'].number
        if 'f2' in ws:
            nf2 = ws['f2'].number
            print('Figures %d and %d\n' % (nf1, nf2))
        else:
            print('Figure %d\n' % nf1)

plt.show()
