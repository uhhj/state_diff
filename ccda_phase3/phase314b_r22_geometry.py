"""Torch/NumPy ordered cable geometry for Phase3.14b-r2.2."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, torch
import torch.nn.functional as F
from ccda_phase3.schema_v2 import DEFAULT_TF,STATE_DIM
from ccda_phase3.phase314a_contract import BEAD_XY_DIM,N_BEADS
from ccda_phase3.phase314a_metrics import chamfer_xy
from ccda_phase3.phase314b_r21_geometry import CalibratedGeometryContract,calibrated_validity,ordered_xy
from ccda_phase3.phase314b_r21_contract import conformal_quantile

@dataclass(frozen=True)
class GeometryNormalizers:
 ordered_xy_scale:float; edge_vector_scale:float; segment_length_scale:float; chain_length_scale:float; temporal_edge_scale:float
 def to_json(self): return self.__dict__.copy()
 def validate(self):
  if not all(np.isfinite(v) and v>0 for v in self.__dict__.values()): raise ValueError('invalid geometry normalizers')

def _future(x,name='future'):
 a=np.asarray(x,dtype=np.float32)
 if a.ndim!=3 or a.shape[1:]!=(4,87) or not np.isfinite(a).all(): raise ValueError(f'{name} must be finite [N,4,87]')
 return a

def fit_geometry_normalizers(fit_future):
 a=_future(fit_future); xy=a[...,:48].reshape(-1,4,24,2); e=xy[...,1:,:]-xy[...,:-1,:]; seg=np.linalg.norm(e,axis=-1); chain=seg.sum(-1); te=e[:,1:]-e[:,:-1]
 n=GeometryNormalizers(max(float(np.median(chain)),.05),max(float(np.median(seg)),1e-3),max(float(np.median(seg)),1e-3),max(float(np.median(chain)),.05),max(float(np.percentile(np.abs(te),90)),1e-3)); n.validate(); return n

def contract_from_json(d):
 return CalibratedGeometryContract(np.asarray(d['coordinate_lower'],np.float32),np.asarray(d['coordinate_upper'],np.float32),np.asarray(d['segment_center'],np.float32),np.asarray(d['segment_scale'],np.float32),float(d['segment_score_threshold']),np.asarray(d['chain_center'],np.float32),np.asarray(d['chain_scale'],np.float32),float(d['chain_score_threshold']),float(d.get('quaternion_norm_lower',.9)),float(d.get('quaternion_norm_upper',1.1)))
def normalizers_from_json(d): n=GeometryNormalizers(**{k:float(v) for k,v in d.items()}); n.validate(); return n

def torch_inverse_standardize(z,mean,scale):
 if z.shape[-2:]!=(4,87): raise ValueError('z shape')
 return z*scale[None]+mean[None]
def torch_ordered_xy(x):
 if x.shape[-2:]!=(4,87) or not torch.isfinite(x).all(): raise ValueError('future shape/finite')
 return x[...,:48].reshape(*x.shape[:-2],4,24,2)
def torch_edges(x):
 xy=torch_ordered_xy(x); return xy[...,1:,:]-xy[...,:-1,:]
def _smooth(a,b,scale): return F.smooth_l1_loss(a/scale,b/scale,reduction='none').flatten(1).mean(1)
def ordered_geometry_components(predicted_raw,target_raw,normalizers):
 normalizers.validate(); pxy=torch_ordered_xy(predicted_raw); txy=torch_ordered_xy(target_raw); pe=pxy[...,1:,:]-pxy[...,:-1,:]; te=txy[...,1:,:]-txy[...,:-1,:]; ps=torch.linalg.norm(pe,dim=-1); ts=torch.linalg.norm(te,dim=-1); pc=ps.sum(-1); tc=ts.sum(-1); ptemp=pe[:,1:]-pe[:,:-1]; ttemp=te[:,1:]-te[:,:-1]
 return {'ordered_xy':_smooth(pxy,txy,normalizers.ordered_xy_scale),'edge_vector':_smooth(pe,te,normalizers.edge_vector_scale),'segment_length':_smooth(ps,ts,normalizers.segment_length_scale),'chain_length':_smooth(pc,tc,normalizers.chain_length_scale),'temporal_edge':_smooth(ptemp,ttemp,normalizers.temporal_edge_scale)}

def final_ordered_rmse_pool(pool,target):
 p=np.asarray(pool,np.float32); t=_future(target); pxy=p[...,-1,:48].reshape(p.shape[0],p.shape[1],24,2); txy=t[:,-1,:48].reshape(t.shape[0],24,2); return np.sqrt(np.mean((pxy-txy[None])**2,axis=(-2,-1)))
def trajectory_ordered_rmse_pool(pool,target):
 p=np.asarray(pool,np.float32); t=_future(target); return np.sqrt(np.mean((p[...,:48]-t[None,...,:48])**2,axis=(-2,-1)))
def last_repeat_future(paper):
 a=np.asarray(paper,np.float32); last=a.reshape(-1,3,87)[:,-1]; return np.repeat(last[:,None],4,axis=1)
def nearest_index_metrics(pred,target):
 p=ordered_xy(_future(pred))[:,-1]; t=ordered_xy(_future(target))[:,-1]; d=((p[:,:,None]-t[:,None])**2).sum(-1); idx=d.argmin(-1); inv=np.mean(np.diff(idx,axis=1)<0,axis=1); return {'nearest_inversion':inv,'nearest_unique_fraction':np.asarray([len(np.unique(x))/24 for x in idx])}
def fit_prediction_reference_thresholds(reference,target,alpha=.10):
 ref=_future(reference); tgt=_future(target); ordered=final_ordered_rmse_pool(ref[None],tgt)[0]; inv=nearest_index_metrics(ref,tgt)['nearest_inversion']; return {'reference':'train_calibration_last_repeat','alpha':float(alpha),'final_ordered_rmse_threshold':conformal_quantile(ordered,alpha=alpha),'nearest_index_inversion_threshold':conformal_quantile(inv,alpha=alpha),'calibration_row_count':int(ref.shape[0])}
def evaluate_validation_pool(pool_z,pool_raw,target,physical_contract,ordered_threshold,inversion_threshold):
 p=np.asarray(pool_raw,np.float32); t=_future(target); valid=calibrated_validity(p,physical_contract); ordered=final_ordered_rmse_pool(p,t); best=ordered.min(0); cham=np.stack([[chamfer_xy(x[-1,:48].reshape(24,2),y[-1,:48].reshape(24,2)) for x,y in zip(pk,t)] for pk in p]); inv=np.stack([nearest_index_metrics(pk,t)['nearest_inversion'] for pk in p]); xy=p[...,:48].reshape(*p.shape[:-1],24,2); seg=np.linalg.norm(xy[...,1:,:]-xy[...,:-1,:],axis=-1); center=np.asarray(physical_contract.segment_center); ratio=seg/np.maximum(center[None,None],1e-8); z=np.abs(np.asarray(pool_z)); diversity=float(np.mean(np.std(p[...,:48],axis=0)))
 return {'finite':bool(np.isfinite(p).all()),'calibrated':{k:v for k,v in valid.items() if not isinstance(v,np.ndarray)},'k1_chamfer':float(cham[0].mean()),'best8_chamfer':float(cham.min(0).mean()),'best8_ordered_rmse':float(best.mean()),'ordered_support_rate':float(np.mean(best<=ordered_threshold)),'permutation_gap':float(np.mean(ordered.min(0)-cham.min(0))),'nearest_inversion_p95':float(np.percentile(inv,95)),'gross_stretch_fraction':float(np.mean(ratio>=4)),'gross_compression_fraction':float(np.mean(ratio<=.25)),'segment_stretch_p95':float(np.percentile(ratio,95)),'z_abs_p99':float(np.percentile(z,99)),'z_abs_max':float(z.max()),'pool_diversity':diversity}
