import os
import sys
import re
import statistics
import unittest
from ast import literal_eval

from tomo import runTomo

work_dir = os.path.realpath(os.getcwd())
if work_dir[-1] != '/':
    work_dir += '/'

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
            with open(exp_fpath) as f:
                exp_items = re.findall(r'\S+', f.read())
            with open(act_fpath) as f:
                act_items = re.findall(r'\S+', f.read())
            self.assertEqual(len(act_items), len(exp_items))
            err = []
            for (exp_item, act_item) in zip(exp_items, act_items):
                try:
                    e = literal_eval(exp_item)
                    a = literal_eval(act_item)
                except (ValueError, TypeError, SyntaxError, MemoryError):
                    continue
                if e and a:
                    err.append((a-e)/e)
                else:
                    err.append(a-e)
                #self.assertAlmostEqual(err, 0, 2)
            print(f'\nfile: {f}')
            print(f'min {min(err)}')
            print(f'max {max(err)}')
            print(f'mean {statistics.mean(err)}')
            print(f'median {statistics.median(err)}')
            print(f'stdev {statistics.stdev(err)}\n')
            self.assertEqual(statistics.median(err), 0.0)
            self.assertAlmostEqual(statistics.mean(err), 0, 6)


if __name__ == '__main__':
    unittest.main()
