"""Extract 50-D PCA (implicitly centred sparse SVD, same subspace as sc.pp.pca(n_comps=50, mask_var=None)),
n_genes_by_counts and obs labels from data/crc_epithelial.h5ad into _ricci_tmp/crc_pca50.npz
(input for scripts/ricci_curvature_pilot.py). Needs only numpy/scipy/h5py."""
import h5py, numpy as np, scipy.sparse as sp, time
from scipy.sparse.linalg import svds, LinearOperator
P='/'.join([__import__('os').environ['HOME'],'mnt/Documents/gft-scrna/data/crc_epithelial.h5ad'])
t=time.time()
f=h5py.File(P,'r')
def csr(g):
    shp=tuple(g.attrs['shape']) if 'shape' in g.attrs else None
    return sp.csr_matrix((g['data'][:],g['indices'][:],g['indptr'][:]),shape=shp)
Xg=f['X']; nobs=Xg['indptr'].shape[0]-1
X=sp.csr_matrix((Xg['data'][:].astype(np.float64),Xg['indices'][:],Xg['indptr'][:]),shape=tuple(Xg.attrs['shape']))
print('X',X.shape,time.time()-t,flush=True)
R=f['raw/X']; rd=R['data'][:]; rip=R['indptr'][:]
ng=np.add.reduceat((rd>0).astype(np.int32), rip[:-1]) if len(rd) else None
ng[np.diff(rip)==0]=0
del rd
def cat(col):
    g=f['obs'][col]
    if isinstance(g,h5py.Group):
        cats=g['categories'][:].astype(str); codes=g['codes'][:]
        return cats[codes]
    return g[:].astype(str)
donor=cat('donor_id'); disease=cat('disease'); ctype=cat('cell_type')
mu=np.asarray(X.mean(axis=0)).ravel()
n,p=X.shape
op=LinearOperator((n,p),matvec=lambda v: X@np.ravel(v) - (mu@np.ravel(v)), rmatvec=lambda u: X.T@np.ravel(u) - mu*np.ravel(u).sum(), dtype=np.float64)
U,S,Vt=svds(op,k=50,random_state=0)
o=np.argsort(-S); U,S=U[:,o],S[o]
pcs=U*S
np.savez_compressed(f"{__import__('os').environ['HOME']}/mnt/Documents/gft-scrna/_ricci_tmp/crc_pca50.npz",pcs=pcs.astype(np.float32),S=S,n_genes=ng,donor=donor,disease=disease,cell_type=ctype)
print('done',time.time()-t,flush=True)
