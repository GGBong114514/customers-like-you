from pathlib import Path
import argparse
import numpy as np
import scipy.sparse as sp
p=argparse.ArgumentParser()
p.add_argument("recommendations", type=Path)
p.add_argument("--matrix", type=Path, default=Path(__file__).resolve().parent/"A_train_log.npz")
a=p.parse_args()
A=sp.load_npz(a.matrix)
r=np.load(a.recommendations,allow_pickle=False)
assert r.shape==(A.shape[0],10), f"Wrong shape: {r.shape}"
assert np.issubdtype(r.dtype,np.integer), "Integer dtype required"
assert ((r>=0)&(r<A.shape[1])).all(), "Out-of-range product indexes"
assert (np.diff(np.sort(r,axis=1),axis=1)!=0).all(), "Duplicate recommendations"
rows=np.repeat(np.arange(A.shape[0]),10)
assert (np.asarray(A[rows,r.ravel()]).ravel()==0).all(), "Already purchased items recommended"
print("PASS: shape, dtype, range, uniqueness, and exclusion of training purchases")
