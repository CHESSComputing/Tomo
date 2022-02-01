import os
import sys
work_dir = os.path.realpath(os.getcwd())
if work_dir[-1] != '/':
    work_dir += '/'

import unittest
import filecmp
from tomo import runTomo

class TomoTestCase(unittest.TestCase):

    def test_tomo(self):
        '''
        Run tomo on the config.txt file for testing.
        Compare results to the saved expected output.
        '''

        # Run the tomo test
        print(f'Generating test data for comparison...')
        runTomo(config_file='config.txt', output_folder='tests/output/actual/', test_mode=True)

        # Compare each file of the test output with the expected output.
        out_dir = os.path.join(os.path.dirname(work_dir), 'tests/output/')
        exp_dir = os.path.join(out_dir, 'expected/')
        act_dir = os.path.join(out_dir, 'actual/')
        files = [f for f in os.listdir(exp_dir) if os.path.isfile(os.path.join(exp_dir, f)) and
                f.endswith('.txt')]
        for f in files:
            exp_fpath = os.path.join(exp_dir, f)
            act_fpath = os.path.join(act_dir, f)
            self.assertTrue(os.path.isfile(os.path.join(act_dir, f)))
            with open(exp_fpath) as exp_f:
                exp_fstr = exp_f.read()
            with open(act_fpath) as act_f:
                act_fstr = act_f.read()
            print(f'Comparing {exp_fpath}\n    and {act_fpath}...')
            self.assertEqual(exp_fstr, act_fstr)

if __name__ == '__main__':
    tomo_test_case = TomoTestCase()
