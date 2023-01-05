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

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--input', action='store', default='NanoGEN.root', help="Input file")
    args = argParser.parse_args()

    f_in = args.input

    t = uproot.open(f"{f_in}")["Events"].arrays()

    xsec = 61526.7  # cross section in pb
    lumi = 60 * 1000  # luminosity in pb
    weights = lumi*xsec*t["genWeight"]/sum(t["genWeight"])

    selection = t["nGenDressedLepton"]==1
    mt = calc_mt(
        t[selection]["GenMET_pt"],
        t[selection]["GenMET_phi"],
        ak.flatten(t[selection]["GenDressedLepton_pt"]),
        ak.flatten(t[selection]["GenDressedLepton_phi"]),
    )

    mt_h = bh.Histogram(
        bh.axis.Regular(10, 0, 200),
        storage=bh.storage.Weight(),
    )
    mt_h.fill(mt, weight=weights[selection])

    # make the plot
    plt.style.use(hep.style.CMS)

    f, ax = plt.subplots()

    hep.histplot(
        [mt_h.values()],
        mt_h.axes[0].edges,
        w2 = mt_h.variances(),
        histtype="step",
        stack=False,
        #label=[r'SM', r'cpt=6'],
        #color=['blue', 'red'],
        ax=ax)

    ax.set_ylabel(r'Events')
    ax.set_xlabel(r'$M_{T}\ (GeV)$')

    #plt.legend()
    f.savefig('./mt.pdf')
