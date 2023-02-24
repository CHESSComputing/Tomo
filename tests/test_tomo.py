from ast import literal_eval
from os import getcwd, listdir, path
from re import findall
import statistics
import unittest
from yaml import safe_load

from sys import path as syspath
syspath.append('workflow')
from run_tomo import run_tomo

work_dir = path.realpath(getcwd())
if work_dir[-1] != '/':
    work_dir += '/'

class TomoTestCase(unittest.TestCase):

    def test_tomo(self):
        '''
        Test tomo on the config.yaml's in each of the test directories.
        Compare results to the saved expected outputs.
        '''

        for i in [1, 2, 3]:
            with self.subTest(i=i):
                name = f'test{str(i)}'
                # Run test
                print(f'Generating {name} data for comparison...')
                run_tomo(f'tests/{name}/config.yaml', f'tests/{name}/output/actual/output.nxs',
                        ['all'], output_folder=f'tests/{name}/output/actual/', save_figs='only',
                        test_mode=True)

                # Compare the test's output with the expected output
                out_dir = path.join(path.dirname(work_dir), f'tests/{name}/output/')
                exp_dir = path.join(out_dir, 'expected/')
                act_dir = path.join(out_dir, 'actual/')
                files = [f for f in listdir(exp_dir) if path.isfile(path.join(exp_dir, f))
                        and f.endswith('.yaml')]
                for f in files:
                    exp_fpath = path.join(exp_dir, f)
                    act_fpath = path.join(act_dir, f)
                    self.assertTrue(path.isfile(path.join(act_dir, f)))
                    with open(exp_fpath) as f:
                        exp_dict = safe_load(f)
                    with open(act_fpath) as f:
                        act_dict = safe_load(f)
                    self.assertEqual(act_dict, exp_dict)
                    print(f'\nfile: {f}\n\t{exp_dict}\n\t{act_dict}')
                files = [f for f in listdir(exp_dir) if path.isfile(path.join(exp_dir, f))
                        and f.endswith('.txt')]
                for f in files:
                    exp_fpath = path.join(exp_dir, f)
                    act_fpath = path.join(act_dir, f)
                    self.assertTrue(path.isfile(path.join(act_dir, f)))
                    with open(exp_fpath) as f:
                        exp_items = findall(r'\S+', f.read())
                    with open(act_fpath) as f:
                        act_items = findall(r'\S+', f.read())
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
                    #self.assertEqual(statistics.median(err), 0.0)
                    self.assertAlmostEqual(statistics.median(err), 0, 4)
                    self.assertAlmostEqual(statistics.mean(err), 0, 4)


if __name__ == '__main__':
    unittest.main()
