"""Phase 25: EfficientNet-B0 end-to-end fine-tuning and augmentation ablation."""
import subprocess
subprocess.run(["pip","install","--quiet","torch==2.5.0","torchvision==0.20.0","--index-url","https://download.pytorch.org/whl/cu121"],check=False,timeout=300)
import json,time,hashlib,random
import numpy as np,torch,torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
T0=time.time(); seed=42; random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
device='cuda' if torch.cuda.is_available() else 'cpu'; print('device:',device)
PV='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color'
mean=[.485,.456,.406]; std=[.229,.224,.225]
basic=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean,std)])
strong=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.75,1.0)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.2,.2,.2,.1),transforms.ToTensor(),transforms.Normalize(mean,std)])
raw=datasets.ImageFolder(PV); y=np.array([s[1] for s in raw.samples]); idx=np.arange(len(raw));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy); ncls=len(raw.classes)
def metric(y,p):return {'accuracy':round(float(accuracy_score(y,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),6),'macro_f1':round(float(f1_score(y,p,average='macro',zero_division=0)),6)}
def run(mode,aug,include_test=False):
  tf=strong if aug=='strong' else basic; ds=datasets.ImageFolder(PV,transform=tf)
  train=DataLoader(Subset(ds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True); val=DataLoader(Subset(ds,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True); test=DataLoader(Subset(ds,te.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True)
  m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1); m.classifier[1]=nn.Linear(m.classifier[1].in_features,ncls)
  if mode=='frozen':
    for p in m.features.parameters(): p.requires_grad=False
  elif mode=='last_block':
    for p in m.features.parameters(): p.requires_grad=False
    for p in m.features[-1].parameters(): p.requires_grad=True
  m=m.to(device); params=[p for p in m.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=3e-4 if mode!='frozen' else 1e-2,weight_decay=1e-4); crit=nn.CrossEntropyLoss(); best=(-1,None)
  for ep in range(8):
    m.train()
    for x,yb in train:
      opt.zero_grad(); loss=crit(m(x.to(device)),yb.to(device)); loss.backward();opt.step()
    m.eval(); yy=[];pp=[]
    with torch.no_grad():
      for x,yb in val: pp.extend(m(x.to(device)).argmax(1).cpu().numpy());yy.extend(yb.numpy())
    score=f1_score(yy,pp,average='macro',zero_division=0)
    print(mode,aug,'epoch',ep+1,'val_macro_f1',round(float(score),6))
    if score>best[0]:best=(score,ep+1)
  if include_test:
    m.eval(); yy=[];pp=[]
    with torch.no_grad():
      for x,yb in test: pp.extend(m(x.to(device)).argmax(1).cpu().numpy());yy.extend(yb.numpy())
    test_metrics=metric(np.array(yy),np.array(pp))
  else: test_metrics=None
  return {'mode':mode,'augmentation':aug,'best_val_macro_f1':round(float(best[0]),6),'best_epoch':best[1],'test':test_metrics,'trainable_params':sum(p.numel() for p in params)}
report={'phase':'25_finetune_ablation','captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':{'model':'efficientnet_b0','modes':['frozen','last_block','full'],'augmentations':['basic','strong'],'epochs':8,'seed':42,'selection':'validation macro_f1','test_policy':'test evaluated only once for locked final configuration'},'split_sizes':{'train':len(tr),'val':len(va),'test':len(te),'n_classes':ncls},'results':[]}
for mode in ['frozen','last_block','full']:
 for aug in ['basic','strong']: report['results'].append(run(mode,aug,False))
report['results'].sort(key=lambda x:x['best_val_macro_f1'],reverse=True); best_cfg=report['results'][0]; final=run(best_cfg['mode'],best_cfg['augmentation'],True); best_cfg['test']=final['test']; report['locked_final']=best_cfg; report['elapsed_sec']=round(time.time()-T0,2); rawj=json.dumps(report,indent=2).encode();report['sha256_self']=hashlib.sha256(rawj).hexdigest()
with open('/kaggle/working/phase25_finetune_ablation_report.json','w') as f:f.write(json.dumps(report,indent=2))
print(json.dumps({'phase':report['phase'],'elapsed_sec':report['elapsed_sec'],'locked_final':report['locked_final']},indent=2))
