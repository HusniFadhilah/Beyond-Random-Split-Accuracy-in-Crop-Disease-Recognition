"""Phase 33: input-resolution ablation for the selected ResNet-50 family."""
import subprocess
subprocess.run(["pip","install","--quiet","--retries","10","--timeout","60","torch==2.3.1","torchvision==0.18.1"],check=True,timeout=900)
import json,time,hashlib,random,copy
import numpy as np,torch,torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
T0=time.time();device='cuda' if torch.cuda.is_available() else 'cpu';print('device:',device)
if device!='cuda':raise RuntimeError('CUDA GPU required; refusing CPU fallback')
PV='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color';mean=[.485,.456,.406];std=[.229,.224,.225]
def make_tf(res,train):
 size=int(res*256/224)
 return transforms.Compose(([transforms.Resize(size),transforms.RandomCrop(res),transforms.RandomHorizontalFlip()] if train else [transforms.Resize(size),transforms.CenterCrop(res)])+[transforms.ToTensor(),transforms.Normalize(mean,std)])
raw=datasets.ImageFolder(PV);y=np.array([s[1] for s in raw.samples]);idx=np.arange(len(raw));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy);ncls=len(raw.classes)
def metric(y,p):return {'accuracy':round(float(accuracy_score(y,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),6),'macro_f1':round(float(f1_score(y,p,average='macro',zero_division=0)),6)}
def evalm(m,dl):
 m.eval();yy=[];pp=[]
 with torch.no_grad():
  for x,yb in dl:pp.extend(m(x.to(device)).argmax(1).cpu().numpy());yy.extend(yb.numpy())
 return metric(np.array(yy),np.array(pp))
def build():
 m=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2);m.fc=nn.Linear(m.fc.in_features,ncls)
 return m.to(device)
resolutions=[224,256];results=[]
for res in resolutions:
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);trds=datasets.ImageFolder(PV,transform=make_tf(res,True));evads=datasets.ImageFolder(PV,transform=make_tf(res,False));trl=DataLoader(Subset(trds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);val=DataLoader(Subset(evads,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);m=build();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);crit=nn.CrossEntropyLoss(label_smoothing=.1);best=-1;be=0
 for ep in range(6):
  m.train()
  for x,yb in trl:opt.zero_grad();loss=crit(m(x.to(device)),yb.to(device));loss.backward();opt.step()
  sch.step();vm=evalm(m,val);print('resolution',res,'epoch',ep+1,vm['macro_f1'])
  if vm['macro_f1']>best:best=vm['macro_f1'];be=ep+1
 results.append({'resolution':res,'best_val_macro_f1':round(float(best),6),'best_epoch':be,'trainable_params':sum(p.numel() for p in m.parameters())})
results.sort(key=lambda z:z['best_val_macro_f1'],reverse=True);winner=results[0];res=winner['resolution'];random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);trds=datasets.ImageFolder(PV,transform=make_tf(res,True));evads=datasets.ImageFolder(PV,transform=make_tf(res,False));trl=DataLoader(Subset(trds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);val=DataLoader(Subset(evads,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);test=DataLoader(Subset(evads,te.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);m=build();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);crit=nn.CrossEntropyLoss(label_smoothing=.1);bestv=-1;state=None
for ep in range(6):
 m.train()
 for x,yb in trl:opt.zero_grad();loss=crit(m(x.to(device)),yb.to(device));loss.backward();opt.step()
 sch.step();vm=evalm(m,val)
 if vm['macro_f1']>bestv:bestv=vm['macro_f1'];state=copy.deepcopy(m.state_dict())
m.load_state_dict(state);winner['locked_test']=evalm(m,test);report={'phase':'33_resolution_ablation','captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':{'model':'resnet50','resolutions':resolutions,'mode':'full_finetune','augmentation':'basic_resize_crop_flip','optimizer':'adamw','loss':'cross_entropy','label_smoothing':.1,'scheduler':'cosine','epochs':6,'selection':'validation macro_f1','test_policy':'test only locked winner'},'split_sizes':{'train':len(tr),'val':len(va),'test':len(te)},'screening':results,'locked_winner':winner};report['elapsed_sec']=round(time.time()-T0,2);rawj=json.dumps(report,indent=2).encode();report['sha256_self']=hashlib.sha256(rawj).hexdigest();open('/kaggle/working/phase33_resolution_ablation_report.json','w').write(json.dumps(report,indent=2));print(json.dumps({'winner':winner,'elapsed_sec':report['elapsed_sec']},indent=2))
