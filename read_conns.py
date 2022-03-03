# -*- coding: utf8 -*-
"""
Converst suballconns_*.dat files into list of afferents and efferents
(Only internal connections, no projections)
author: András Ecker, last update: 02.2020
"""

import os
import numpy as np
import multiprocessing as mp


def _read_suballconns_subprocess(f_name):
    """Subprocess used by multiprocessing pool -  see: `read_allconns()`"""

    pairs = np.zeros((2, int(1e7)), dtype=np.int32)

    with open(f_name, "rb") as f:
        l = next(f)
        preID = int(l.split()[0])
        postID = int(l.split()[1])
        pairs[0, 0] = preID
        pairs[1, 0] = postID
        prev_preID = preID
        prev_postID = postID
        i = 1
        for l in f:
            preID = int(l.split()[0])
            postID = int(l.split()[1])
            # count evey connection once
            if preID == prev_preID and postID == prev_postID:
                continue
            # ignore projections
            if preID <= 338739:  # hard codedID of last "real cell"
                pairs[0, i] = preID
                pairs[1, i] = postID
                prev_preID = preID
                prev_postID = postID
                i += 1

    # cut arrays after the last entry
    pairs = pairs[:, :i]

    return pairs


def read_suballconns(dir_path, N):
    """Reads in all suballconn_*.dat into preID-postID array"""

    pool = mp.Pool(processes=N)
    # 3007: hard coded total number of files
    pairs = pool.map(_read_suballconns_subprocess,
                     [os.path.join(dir_path, "suballconns_%i.dat"%i) for i in range(3007)])
    pool.terminate()

    # concatenate them into 1 array
    all_pairs = pairs[0]
    for i in range(1, len(pairs)):
        all_pairs = np.concatenate((all_pairs, pairs[i]), axis=1)

    return all_pairs


def save_conns(all_pairs):
    """Sorts and saves connections"""

    pre = all_pairs[0, :].reshape(1, all_pairs.shape[1])
    post = all_pairs[1, :].reshape(1, all_pairs.shape[1])

    # sort by preID
    idx = np.argsort(pre)
    efferents = np.concatenate((pre[0, idx], post[0, idx]), axis=0)
    f_name = os.path.join(base_path, "efferents.npz")
    np.savez(f_name, efferents)

    # sort by postID
    idx = np.argsort(post)
    afferents = np.concatenate((pre[0, idx], post[0, idx]), axis=0)
    f_name = os.path.join(base_path, "afferents.npz")
    np.savez(f_name, afferents)


if __name__ == "__main__":

    base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-1])
    dir_path = os.path.join(base_path, "ca1connections")

    all_pairs = read_suballconns(dir_path, mp.cpu_count())
    save_conns(all_pairs)