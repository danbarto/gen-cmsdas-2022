#!/usr/bin/env python3
import numpy as np
import awkward as ak
import uproot
import boost_histogram as bh
import argparse
import matplotlib.pyplot as plt
import mplhep as hep

def calc_mt(pt1, phi1, pt2, phi2):
    '''
    Calculate the transverse mass
    '''
    return np.sqrt(2*pt1*pt2 * (1 - np.cos(phi1-phi2)))

def add_systematic(ax, up, down, color='gray', hatch='///', label='uncertainty'):
    opts = {
        'step': 'post',
        'label': label,
        'hatch': hatch,
        'facecolor': 'none',
        'edgecolor': color,
        'linewidth': 0,
    }
    ax.fill_between(x=mt_h.axes[0].edges, y1=np.r_[down, down[-1]], y2=np.r_[up, up[-1]], **opts)

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--input', action='store', default='NanoGEN.root', help="Input file")
    argParser.add_argument('--logy', action='store_true', default=False, help="Set y-axis to log scale")
    args = argParser.parse_args()

    f_in = args.input

    t = uproot.open(f"{f_in}")["Events"].arrays()  # events tree
    r = uproot.open(f"{f_in}")["Runs"].arrays()  # run tree with event weight sums

    print (f'Opened file {f_in}')

    xsec = 61526.7  # cross section in pb
    lumi = 60 * 1000  # luminosity in 1/pb
    weights = lumi*xsec*t["genWeight"]/sum(r['genEventSumw'])

    scale_weight_norms = sum(r['genEventSumw'])/sum(r['genEventSumw']*r['LHEScaleSumw'])

    selection = t["nGenDressedLepton"]==1
    mt = calc_mt(
        t[selection]["GenMET_pt"],
        t[selection]["GenMET_phi"],
        ak.flatten(t[selection]["GenDressedLepton_pt"]),
        ak.flatten(t[selection]["GenDressedLepton_phi"]),
    )

    print (f'Filling histograms.')

    mt_h = bh.Histogram(
        bh.axis.Regular(10, 0, 200),
        storage=bh.storage.Weight(),
    )

    mt_muf_d_mur_n_h = mt_h.copy()  # mu_f down, mu_r nominal
    mt_muf_u_mur_n_h = mt_h.copy()  # mu_f up, mu_r nominal

    mt_alphas_u_h = mt_h.copy()
    mt_alphas_d_h = mt_h.copy()

    mt_h.fill(mt, weight=weights[selection])

    # get factorization scale variations
    mt_muf_d_mur_n_h.fill(
        mt,
        weight=weights[selection]*t['LHEScaleWeight'][selection][:,3],  # can multiply with respective norm
    )

    mt_muf_u_mur_n_h.fill(
        mt,
        weight=weights[selection]*t['LHEScaleWeight'][selection][:,5],  # can multiply with respective norm
    )

    # alphaS variations not kept in this example, so omitted
    #mt_alphas_u_h.fill(
    #    mt,
    #    weight=weights[selection]*t['LHEPdfWeight'][selection][:,101],
    #)

    #mt_alphas_d_h.fill(
    #    mt,
    #    weight=weights[selection]*t['LHEPdfWeight'][selection][:,102],
    #)

    # make the plot
    plt.style.use(hep.style.CMS)

    f, ax = plt.subplots()

    hep.histplot(
        [mt_h.values()],
        mt_h.axes[0].edges,
        w2 = mt_h.variances(),
        histtype="step",
        stack=False,
        ax=ax)

    ax.set_ylabel(r'Events')
    ax.set_xlabel(r'$M_{T}\ (GeV)$')

    if args.logy:
        ax.set_yscale('log')
    ax.set_xlim(min(mt_h.axes[0].edges), max(mt_h.axes[0].edges))

    add_systematic(
        ax,
        up      = mt_muf_u_mur_n_h.values(),
        down    = mt_muf_d_mur_n_h.values(),
        color   = 'gray',
        label   = r'$\mu_{f}\ uncertainty$'
    )

    #add_systematic(
    #    ax,
    #    up      = mt_alphas_u_h.values(),
    #    down    = mt_alphas_d_h.values(),
    #    color   = 'blue',
    #    hatch   = 'XXX',
    #    label   = r'$\alpha_{S}\ uncertainty$'
    #)

    plt.legend()
    f.savefig('./mt.pdf')

    print (f'Done')
