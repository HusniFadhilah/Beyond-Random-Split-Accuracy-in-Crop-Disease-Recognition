import os,json,hashlib,time
from PIL import Image
import numpy as np
from torchvision import datasets
from sklearn.model_selection import train_test_split
T=time.time(); ROOT='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color'; raw=datasets.ImageFolder(ROOT); y=np.array([s[1] for s in raw.samples]); idx=np.arange(len(raw)); tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y); va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy); split={i:('train' if i in set(tr) else 'validation' if i in set(va) else 'test') for i in idx}; paths=[p for p,_ in raw.samples]
def phash(path):
 im=Image.open(path).convert('L').resize((32,32),Image.Resampling.LANCZOS); a=np.asarray(im,dtype=np.float32)
 from scipy.fftpack import dct
 d=dct(dct(a,axis=0,norm='ortho'),axis=1,norm='ortho')[:8,:8]; bits=d>d[1:,1:].mean(); return ''.join('1' if x else '0' for x in bits.ravel())
md5={}; ph={}; failures=[]
for i,p in enumerate(paths):
 try:
  h=hashlib.md5(open(p,'rb').read()).hexdigest();md5.setdefault(h,[]).append(i);ph.setdefault(phash(p),[]).append(i)
 except Exception as e: failures.append({'index':i,'path':p,'error':repr(e)})
def summarize(groups):
 out=[]
 for g in groups:
  ss=sorted(set(split[i] for i in g))
  if len(ss)>1: out.append({'splits':ss,'n_images':len(g),'split_counts':{s:sum(split[i]==s for i in g) for s in ss},'paths':[paths[i] for i in g]})
 return out
md5groups=[g for g in md5.values() if len(g)>1]; phgroups=[g for g in ph.values() if len(g)>1]
report={'phase':'54_cross_split_duplicate_audit','dataset_root':ROOT,'split_protocol':{'seed':42,'train':len(tr),'validation':len(va),'test':len(te),'stratified':True},'n_images':len(paths),'n_failures':len(failures),'md5':{'total_groups':len(md5groups),'cross_split_groups':len(summarize(md5groups)),'cross_split':summarize(md5groups)},'phash_exact':{'total_groups':len(phgroups),'cross_split_groups':len(summarize(phgroups)),'cross_split':summarize(phgroups)},'sha256_self':None,'elapsed_sec':round(time.time()-T,2)}
report['sha256_self']=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest();open('/kaggle/working/phase54_cross_split_duplicate_audit.json','w').write(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['phase','n_images','n_failures','md5','phash_exact']},indent=2))