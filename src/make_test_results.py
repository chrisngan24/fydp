"""
Python script for running code against all the test cases
and then saving the results so it can be committed.
"""
import test

import os
import datetime
import shutil



def make_dir_name(base='test_results'):
    now = datetime.datetime.now()
    return base + '/' + now.strftime('%Y-%m-%d_%H-%M')


def copy_test_plots(testing_dir, output_dir, plot_file='fused_plot.png'):
    test_case_list = sorted(next(os.walk(testing_dir))[1])
    for test in test_case_list:
        for fi in os.listdir(testing_dir + test):
            if fi == plot_file:
                print 'copying', test,plot_file
                shutil.copy(
                        testing_dir + test + '/' + fi, 
                        output_dir + '/%s-%s' % (test, plot_file),
                        )


if __name__ == "__main__":
    print "Are you sure you want to rerun the entire test suite? (y/n)"
    print "You will create a lot of images and it may take a while"
    res = raw_input()
    if res == 'y':
        print "RUNNING ALL TEST CASES AND SAVING IT"
        test.main() 
        dir_name = make_dir_name()
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        print 'Copying data to ' + dir_name
        shutil.copy('test_results/test_results.html', dir_name)
        # so that it renders in github
        shutil.copy('test_results/test_results.html', dir_name + '/' + 'README.md')
        copy_test_plots('test_suite/test_cases/', dir_name)
        
    else:
        print 'Canelling the process'


