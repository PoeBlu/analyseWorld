#!/usr/bin/env python

import numpy as np
import pandas as pd
import easygui as gui
from opterator import opterate
import pyperclip

def fileList(foldPath, included=['*traj.bag_df.pickle'], excluded=[]):
    import os
    import fnmatch

    matches = []
    for root, dirnames, filenames in os.walk(foldPath):

        for include in included:
            for filename in fnmatch.filter(filenames, include):
                toKeep = True

                for exclude in excluded:
                    if exclude in filename:
                        toKeep = False

                if toKeep:
                    matches.append(os.path.join(root, filename))

    matches.sort()
    return matches


#make auto cli
@opterate
def htmlGen(foldPath=None, defaultPath='/media/rhagoletis/6db8b2b2-ebe2-4555-9b79-93b10ef9dec31/2017_08',
            start='<ol>', stop='</ol>', bstart='<li>', bstop='</li>',
            lstart='<a href="', lstop='</a>',
            lpre='https://labnotes.ncbs.res.in/files/users/pavankk/2017/2017_08/', lpost='">',
            istart='<iframe ', istop='</iframe>',
            h=1100, w=1100, scroll='yes', ipost='">',verbose=False):
    '''
    
    :param foldPath: Dir path of html files 
    :param defaultPath: default foalder path
    :param start: starting header 
    :param stop: -S footer
    :param bstart: --bstart
    :param bstop: --bstop
    :param lstart: --lstart
    :param lstop: --lstop
    :param lpre: --lpre
    :param lpost: --lpost
    :param istart: --istart
    :param istop: --istop
    :param h: -H --height
    :param w: -W --widtt
    :param scroll: --scroll
    :param ipost: --ipost
    :param verbose: 

    :return: 
    '''

    ipre = 'frameborder="1" height="' + str(h) + '" width="' + str(w) + \
           '" scrolling="' + scroll + '" src="' + lpre

    if foldPath is None:
        foldPath = gui.diropenbox(default=defaultPath)
    files = fileList(foldPath, included=['*.html'])
    print foldPath.split('/')[-1]
    txt = start
    for item in files:
        pth = item.split('/')[-1]
        txt += (bstart +
                lstart +
                lpre + pth + lpost + pth + lstop +
                istart + ipre + pth + ipost + istop +
                bstop)

    txt += stop
    # print txt
    pyperclip.copy(txt)
    if verbose:
        print txt
    return txt

if __name__ =='__main__':
    htmlGen()
