import numpy as np,torch,pytest
from ccda_phase3.phase314b_r22_geometry import *
def future():
 x=np.zeros((8,4,87),np.float32);xy=np.zeros((8,4,24,2),np.float32);xy[...,0]=np.linspace(.3,.53,24);x[...,:48]=xy.reshape(8,4,48);x[...,83:87]=[0,0,0,1];return x
def test_normalizers():fit_geometry_normalizers(future()).validate()
def test_permutation_ordered_loss():
 x=future()[:2];y=x.copy();xy=y[...,:48].reshape(2,4,24,2);xy[:]=xy[...,list(range(0,24,2))+list(range(1,24,2)),:];n=fit_geometry_normalizers(x);c=ordered_geometry_components(torch.tensor(y),torch.tensor(x),n);assert c['ordered_xy'].mean()>0 and c['edge_vector'].mean()>0
def test_translation_edge_invariant():
 x=future()[:2];y=x.copy();y[...,:48].reshape(2,4,24,2)[...,:]+=.1;n=fit_geometry_normalizers(x);c=ordered_geometry_components(torch.tensor(y),torch.tensor(x),n);assert c['ordered_xy'].mean()>0 and c['edge_vector'].mean()<1e-6
def test_bad_shape():
 with pytest.raises(ValueError):torch_ordered_xy(torch.zeros(2,3))
