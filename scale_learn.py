"""It appears that when using a gaussian as a neighbourhood function
PINK does not normalise the peak to one. This cause the weighting
updates to behave in a difficult to predict manner, especially in
combination with the learning rate. This script given a desired 
sigma and learning rate for a set of stages, will scale the learning
rate to account for 'off normalised' peak
"""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_update(sigma: list, lr: list, scale_lr: list):
    """Plot the before and after of the scaled neighborhood
    update
    
    Arguments:
        sig {list} -- Set of sigmas
        lr {list} -- Set of desired learning rates
        scale_lr {list} -- Set of effective learning rate
    """

    x = np.linspace(-7, 7, 200)

    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax2 = ax.twinx()

    for s, l, sl in zip(sigma, lr, scale_lr):
        n = norm.pdf(x,0,s)

        ax.plot(x, n*l, label=f"Sigma={s}")
        ax2.plot(x, n*sl, ls='--', label=f"Sigma={s}")
        ax.axhline(l, ls='-')
        ax2.axhline(l, ls='--')

    leg = ax.legend(loc='upper left', title='Uncorrected', framealpha=1.)
    leg.remove()
    ax2.legend(loc='upper right', title='Corrected', framealpha=1.)
    ax2.add_artist(leg)
    
    ax.set(ylabel='Uncorrected Weighting', xlabel='Neuron distance')
    ax2.set(ylabel='Corrected Weighting')

    fig.tight_layout()
    fig.savefig('Effective_LR.pdf')



def scale(sigma: list, lr: list, plot: bool=False):
    """Driver for when script is called as program. Will output
    the effective learning rate term to used given the desired 
    sigma and learning rate 
    
    Arguments:
        sig {list} -- List of desired sigma for each stage  
        lr {list} -- List of desired learning rates
    
    Keyword Arguments:
        plot {bool} -- [description] (default: {False})
    """
    scale_lr = []
    sigma = [float(s) for s in sigma]
    lr = [float(l) for l in lr]

    for s, l in zip(sigma, lr):
        peak = norm.pdf(0,0,s)
        scale_lr.append( l / peak)
    
    if plot:
        print('plotting')
        plot_update(sigma, lr, scale_lr)

    print("Desired_LR: sigma effective_LR")
    for s, l, sl in zip(sigma, lr, scale_lr):
        print(f"{l:.3f} : {s:.3f} {sl:.5f}")


if __name__ == '__main__':
    arg = argparse.ArgumentParser(description="Convert from a set of desired sigma "\
                                 "and learning rates to effective sigma and learning rates "\
                                "after accounting for correct normalisation of the gaussian.")
    arg.add_argument('--sigma', nargs='+', help="Desired sigma stages")
    arg.add_argument('--learn', nargs='+', help='Desired learning rates for each stage')
    arg.add_argument('--plot', default=False, action='store_true')

    pargs = vars(arg.parse_args())

    if len(pargs['sigma']) != len(pargs['learn']):
        print("Ensure lengths of stages are equal")
        sys.exit(1)
    
    scale(pargs['sigma'], pargs['learn'], plot=pargs['plot'])
