import gnpy as gn
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def main(args=None):
    """Required for setup.py install gui script."""
    if args is None:
        args = sys.argv[1:]

    num_ch = 95
    if num_ch % 2 == 1:  # odd number of channels
        fch = np.arange(-np.floor(num_ch/2), np.floor(num_ch/2)+1, 1) * 0.05
        fch_NLI = [0]
    else:
        fch = (np.arange(0,num_ch)-(num_ch/2)+0.5)*0.05
        fch_NLI = [-.5 * 0.05, .5 * 0.05]
    # fch_NLI = np.concatenate((fch,fch+.01,fch-.01))
    # fch_NLI = sorted(fch_NLI)
    rs = np.ones(num_ch) * 0.032
    roll_off = np.ones(num_ch) * 0.05
    model_param = {'min_FWM_inv': 30.0, 'n_grid': 500, 'n_grid_min': 4,
                   'f_array': np.array(fch_NLI, copy=True)}
    power = np.ones(num_ch) * 0.001
    beta2 = 21.27
    Lspan = 100
    loss = 0.2
    gam = 1.27
    t = time.time()
    nli = gn.GN_integral(beta2, Lspan, loss, gam, fch, rs, roll_off, power, num_ch,model_param)
    print('Elapsed: %s' % (time.time() - t))
    f1_array = np.linspace(np.amin(fch), np.amax(fch), 1e3)
    Gtx = gn.raised_cosine_comb(f1_array, rs, roll_off, fch, power)
    Gtx = Gtx + 10**-6  # To avoid log10 issues.
    plt.figure(1)
    plt.plot(f1_array, 10*np.log10(Gtx), '-b', label='WDM comb')
    plt.plot(fch_NLI,10*np.log10(nli),'ro',label='GNLI')
    plt.ylabel('PSD [dB(W/THz)]')
    plt.xlabel('f [THz]')
    plt.legend(loc='upper left')
    plt.grid()
    plt.draw()
    plt.show()


if __name__ == '__main__':
    main()