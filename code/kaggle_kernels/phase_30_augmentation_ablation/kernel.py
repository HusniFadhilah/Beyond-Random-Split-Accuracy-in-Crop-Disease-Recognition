"""Phase 30: augmentation ablation under the locked Phase 28 training protocol."""
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
evaltf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean,std)])
def make_tf(name):
 base=[transforms.ToTensor(),transforms.Normalize(mean,std)]
 if name=='no_aug':return transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224)]+base)
 if name=='basic':return transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),transforms.RandomHorizontalFlip()]+base)
 if name=='strong':return transforms.Compose([transforms.RandomResizedCrop(224,scale=(.75,1.0)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.2,hue=0.0)]+base)
 if name=='randaugment':return transforms.Compose([transforms.RandomResizedCrop(224,scale=(.75,1.0)),transforms.RandomHorizontalFlip(),transforms.RandAugment(num_ops=2,magnitude=9)]+base)
 raise ValueError(name)
raw=datasets.ImageFolder(PV);y=np.array([s[1] for s in raw.samples]);idx=np.arange(len(raw));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy);ncls=len(raw.classes)
def metric(y,p):return {'accuracy':round(float(accuracy_score(y,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),6),'macro_f1':round(float(f1_score(y,p,average='macro',zero_division=0)),6)}
def evalm(m,dl):
 m.eval();yy=[];pp=[]
 with torch.no_grad():
  for x,yb in dl:pp.extend(m(x.to(device)).argmax(1).cpu().numpy());yy.extend(yb.numpy())
 return metric(np.array(yy),np.array(pp))
def soft_loss(logits,ya,yb,lam):return lam*nn.functional.cross_entropy(logits,ya,label_smoothing=.1)+(1-lam)*nn.functional.cross_entropy(logits,yb,label_smoothing=.1)
def train_batch(m,x,yb,mode):
 if mode=='mixup':
  lam=float(np.random.beta(.4,.4));perm=torch.randperm(x.size(0),device=device);log=m(lam*x+(1-lam)*x[perm]);return soft_loss(log,yb,yb[perm],lam)
 if mode=='cutmix':
  lam=float(np.random.beta(1.0,1.0));perm=torch.randperm(x.size(0),device=device);_,_,h,w=x.shape;cx=random.randint(0,w-1);cy=random.randint(0,h-1);rw=int(w*np.sqrt(1-lam));rh=int(h*np.sqrt(1-lam));x1=max(cx-rw//2,0);x2=min(cx+rw//2,w);y1=max(cy-rh//2,0);y2=min(cy+rh//2,h);mixed=x.clone();mixed[:,:,y1:y2,x1:x2]=x[perm,:,y1:y2,x1:x2];lam=1-((x2-x1)*(y2-y1)/(w*h));log=m(mixed);return soft_loss(log,yb,yb[perm],lam)
 return nn.functional.cross_entropy(m(x),yb,label_smoothing=.1)
configs=[{'name':n,'transform':n,'mix_mode':n if n in ['mixup','cutmix'] else 'standard'} for n in ['no_aug','basic','strong','randaugment','mixup','cutmix']]
results=[]
for cfg in configs:
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);name=cfg['name'];trtf=make_tf(cfg['transform']) if name not in ['mixup','cutmix'] else make_tf('strong');trds=datasets.ImageFolder(PV,transform=trtf);evads=datasets.ImageFolder(PV,transform=evaltf);trl=DataLoader(Subset(trds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);val=DataLoader(Subset(evads,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier[1]=nn.Linear(m.classifier[1].in_features,ncls);m=m.to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);best=-1;be=0
 for ep in range(6):
  m.train()
  for x,yb in trl:opt.zero_grad();loss=train_batch(m,x.to(device),yb.to(device),name);loss.backward();opt.step()
  sch.step();vm=evalm(m,val);print(name,'epoch',ep+1,vm['macro_f1'])
  if vm['macro_f1']>best:best=vm['macro_f1'];be=ep+1
 results.append({'config':cfg,'best_val_macro_f1':round(float(best),6),'best_epoch':be,'trainable_params':sum(p.numel() for p in m.parameters())})
results.sort(key=lambda z:z['best_val_macro_f1'],reverse=True);winner=results[0];name=winner['config']['name'];random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);trtf=make_tf(winner['config']['transform']) if name not in ['mixup','cutmix'] else make_tf('strong');trds=datasets.ImageFolder(PV,transform=trtf);evads=datasets.ImageFolder(PV,transform=evaltf);trl=DataLoader(Subset(trds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);val=DataLoader(Subset(evads,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);test=DataLoader(Subset(evads,te.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier[1]=nn.Linear(m.classifier[1].in_features,ncls);m=m.to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);bestv=-1;state=None
for ep in range(6):
 m.train()
 for x,yb in trl:opt.zero_grad();loss=train_batch(m,x.to(device),yb.to(device),name);loss.backward();opt.step()
 sch.step();vm=evalm(m,val)
 if vm['macro_f1']>bestv:bestv=vm['macro_f1'];state=copy.deepcopy(m.state_dict())
m.load_state_dict(state);winner['locked_test']=evalm(m,test);report={'phase':'30_augmentation_ablation','captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':{'model':'efficientnet_b0','mode':'full_finetune','optimizer':'adamw','loss':'cross_entropy','label_smoothing':0.1,'scheduler':'cosine','configs':configs,'epochs':6,'selection':'validation macro_f1','test_policy':'test only locked winner'},'split_sizes':{'train':len(tr),'val':len(va),'test':len(te)},'screening':results,'locked_winner':winner};report['elapsed_sec']=round(time.time()-T0,2);rawj=json.dumps(report,indent=2).encode();report['sha256_self']=hashlib.sha256(rawj).hexdigest();open('/kaggle/working/phase30_augmentation_ablation_report.json','w').write(json.dumps(report,indent=2));print(json.dumps({'winner':winner,'elapsed_sec':report['elapsed_sec']},indent=2))
