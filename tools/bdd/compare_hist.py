import os
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt

def load_hist(fpath):
    return np.load(fpath)

def chi_squared(y1,y2):
    return np.sum( (y1-y2)**2/(y1+y2+1e-6) )


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--task',
        required=True,
        help='task -- confusion_matrix or CDF')
    parser.add_argument(
        '--hist_files',
        required=True,
        nargs='*',
        help='list of histogram NumPy saves to compare')
    parser.add_argument(
        '--show_TP_only',
        action='store_true',
        help='show histogram of only True Positive probabilities')
    parser.add_argument(
        '--show_FP_only',
        action='store_true',
        help='show histogram of only False Positive probabilities')

    return parser.parse_args()


#### Show a matrices with pairwise chi-squared distances ####

def confusion_matrices(hist_files):
    n_files = len(hist_files)
    print('\n\nHist Files:')
    for h,hist in enumerate(hist_files):
        print(str(h)+':',hist)
    tp_cm = np.zeros((n_files,n_files))
    fp_cm = np.zeros((n_files,n_files))
    for i in range(n_files):
        for j in range(n_files):
            x1,tp1,fp1 = load_hist(hist_files[i])
            x2,tp2,fp2 = load_hist(hist_files[j])
            tp_cm[i,j] = chi_squared(tp1,tp2)
            fp_cm[i,j] = chi_squared(fp1,fp2)
    print('\n\nShowing pairwise chi-squared distances between the TP histograms:')
    print(tp_cm)
    print('\n\nShowing pairwise chi-squared distances between the FP histograms:')
    print(fp_cm)

################################


#### Plot and compare CDFs ####

def compare_cdfs(hist_files,
                 show_TP_only=False,
                 show_FP_only=False):
    for h,hist in enumerate(hist_files):
        x,tp,fp = load_hist(hist)
        lab = os.path.split(str(hist))[-1].split('.')[0]
        if not show_FP_only:
            tp_cdf = np.cumsum(tp)
            plt.plot(x,tp_cdf,label=lab+'_TP')
        if not show_TP_only:
            fp_cdf = np.cumsum(fp)
            plt.plot(x,fp_cdf,label=lab+'_FP')
    plt.legend()
    plt.xlabel('scores')
    plt.show()

###############################



if __name__ == '__main__':
    
    args = parse_args()

    if args.task == 'confusion_matrix':
        confusion_matrices(args.hist_files)
    
    elif args.task == 'CDF':
        compare_cdfs(args.hist_files,
                      show_TP_only=args.show_TP_only,
                      show_FP_only=args.show_FP_only)


