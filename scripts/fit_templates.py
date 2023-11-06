#!/usr/bin/env python3
import fitsio
import numpy as np
import cupy as cp

from nearly_nmf import nmf

import time
import argparse
p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=100921, help="Random seed")
p.add_argument('-i', '--in_file', type=str, help="Input filename")
p.add_argument('-o', '--out_suffix', type=str, help="Output template name suffix")
p.add_argument('-p', '--in_suffix', type=str, help="Input template name suffix")
p.add_argument('--n_templates', type=int, default=5, help="Number of templates to train")
p.add_argument('--validate', action="store_true", help="Whether to fit the validation set or not.")

args = p.parse_args()

n_qsos = 130000
with fitsio.FITS(args.in_file, "r") as h:
    X = h["FLUX"][:, n_qsos:]
    V = h["IVAR"][:, n_qsos:]
    w_grid = h["WAVELENGTH"].read()
    z = h["Z"].read()
    
rng = np.random.default_rng(args.seed)

M = V != 0
n_cover = np.sum(M, axis=1)

# Cut pixels off the bottom to avoid edge effects
cut_lower = 50
# Find the upper index at which the same number of spectra are in the last
# pixel as the new (post cut) first pixel
n_spec_lower = n_cover[cut_lower]
cut_upper = np.argmax(n_cover[500:] < n_spec_lower) + 500

# Round the upper limit so that we have a nice round number
# of pixels in the output
diff = cut_upper - cut_lower
diff = int(np.round(diff / 50) * 50)
cut_upper = cut_lower + diff

keep_range = np.s_[cut_lower:cut_upper]

print("Keep range:", cut_lower, cut_upper)

X = cp.asarray(X[keep_range, :])
V = cp.asarray(V[keep_range, :])

print(X.shape)

# Helpful to abstract these shapes for later
H_shape = (args.n_templates, X.shape[1])

# Sequential start so we can use a smooth flat W start
# for every template
H_start = rng.uniform(0, 1, H_shape)

# Sequentially construct the n_templates templates
H_nearly = cp.array(H_start, copy=True)
W_nearly = cp.load(f"templates/W_nearly_{args.in_suffix}.npy")

H_shift = cp.array(H_start, copy=True)
W_shift = cp.load(f"templates/W_shift_{args.in_suffix}.npy")

print(f"templates/W_shift_{args.in_suffix}.npy")
print(W_shift.shape)
print(H_shift.shape)


print("Starting fit")
# Refining the templates
t1 = time.time()
H_nearly, W_nearly = nmf.nearly_NMF(X, V, H_nearly, W_nearly, n_iter=100)
t2 = time.time()
H_shift, W_shift = nmf.shift_NMF(X, V, H_shift, W_shift, n_iter=100)
t3 = time.time()
print(t2 - t1, t3 - t2)

print("Saving templates")
cp.save(f"templates/H_nearly_{args.out_suffix}_test.npy", H_nearly)
cp.save(f"templates/H_shift_{args.out_suffix}_test.npy", H_shift)
