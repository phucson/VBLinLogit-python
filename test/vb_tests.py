## script to run all vb_{linar,logit}_*_test scripts
import os
import runpy

test_scripts = [
    'vb_linear_fit_test.py',
    'vb_linear_fit_ard_test.py',
    'vb_linear_pred_test.py',
    'vb_logit_fit_test.py',
    'vb_logit_fit_ard_test.py',
    'vb_logit_fit_iter_test.py',
    'vb_logit_pred_test.py',
    'vb_logit_pred_incr_test.py']

for i in range(len(test_scripts)):
    script_name = test_scripts[i]
    print('Running %s' % script_name)
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name),
                   run_name='__main__')
    print()
