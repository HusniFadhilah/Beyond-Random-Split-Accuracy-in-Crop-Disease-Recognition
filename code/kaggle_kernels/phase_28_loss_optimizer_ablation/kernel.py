"""Phase 28: full fine-tuning loss/optimizer/scheduler/class-weight ablation."""
import subprocess
subprocess.run(["pip","install","--quiet","--retries","10","--timeout","60","torch==2.3.1","torchvision==0.18.1"],check=True,timeout=900)
import json,time,hashlib,random,copy
import numpy as np,torch,torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score
T0=time.time();device='cuda' if torch.cuda.is_available() else 'cpu';print('device:',device)
if device!='cuda':raise RuntimeError('CUDA GPU required')
PV='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color';mean=[.485,.456,.406];std=[.229,.224,.225]
trtf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.75,1)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.2,hue=0),transforms.ToTensor(),transforms.Normalize(mean,std)]);evtf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean,std)])
raw=datasets.ImageFolder(PV);y=np.array([s[1] for s in raw.samples]);idx=np.arange(len(raw));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy);ncls=len(raw.classes);counts=np.bincount(y,minlength=ncls)
def metric(y,p):return {'accuracy':round(float(accuracy_score(y,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),6),'macro_f1':round(float(f1_score(y,p,average='macro',zero_division=0)),6)}
def evalm(m,dl):
 m.eval();yy=[];pp=[]
 with torch.no_grad():
  for x,yb in dl:pp.extend(m(x.to(device)).argmax(1).cpu().numpy());yy.extend(yb.numpy())
 return metric(np.array(yy),np.array(pp))
def focal(logits,t,gamma=2):
 ce=nn.functional.cross_entropy(logits,t,reduction='none');return (((1-torch.exp(-ce))**gamma)*ce).mean()
configs=[{'name':'adamw_ce_cosine','loss':'ce','optimizer':'adamw','scheduler':'cosine','label_smoothing':0,'weighted':False},{'name':'adamw_label_smoothing','loss':'ce','optimizer':'adamw','scheduler':'cosine','label_smoothing':.1,'weighted':False},{'name':'adamw_focal','loss':'focal','optimizer':'adamw','scheduler':'cosine','label_smoothing':0,'weighted':False},{'name':'adamw_weighted_ce','loss':'ce','optimizer':'adamw','scheduler':'cosine','label_smoothing':0,'weighted':True},{'name':'sgd_ce','loss':'ce','optimizer':'sgd','scheduler':'cosine','label_smoothing':0,'weighted':False}]
results=[]
for cfg in configs:
 random.seed(42);np.random.seed(42);torch.manual_seed(42);trds=datasets.ImageFolder(PV,transform=trtf);evads=datasets.ImageFolder(PV,transform=evtf);trl=DataLoader(Subset(trds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);val=DataLoader(Subset(evads,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier[1]=nn.Linear(m.classifier[1].in_features,ncls);m=m.to(device);params=m.parameters();opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-4) if cfg['optimizer']=='adamw' else torch.optim.SGD(params,lr=1e-2,momentum=.9,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);w=torch.tensor((counts.sum()/(ncls*np.maximum(counts,1))),dtype=torch.float32,device=device);crit=nn.CrossEntropyLoss(weight=w if cfg['weighted'] else None,label_smoothing=cfg['label_smoothing']);best=-1;be=0
 for ep in range(6):
  m.train()
  for x,yb in trl:
   opt.zero_grad();log=m(x.to(device));loss=focal(log,yb.to(device)) if cfg['loss']=='focal' else crit(log,yb.to(device));loss.backward();opt.step()
  sch.step();vm=evalm(m,val);print(cfg['name'],'epoch',ep+1,vm['macro_f1'])
  if vm['macro_f1']>best:best=vm['macro_f1'];be=ep+1
 results.append({'config':cfg,'best_val_macro_f1':round(float(best),6),'best_epoch':be,'trainable_params':sum(p.numel() for p in m.parameters())})
results.sort(key=lambda z:z['best_val_macro_f1'],reverse=True);best=results[0];cfg=best['config'];# retrain locked winner, then test once
random.seed(42);np.random.seed(42);torch.manual_seed(42);trds=datasets.ImageFolder(PV,transform=trtf);evads=datasets.ImageFolder(PV,transform=evtf);trl=DataLoader(Subset(trds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);val=DataLoader(Subset(evads,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);test=DataLoader(Subset(evads,te.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier[1]=nn.Linear(m.classifier[1].in_features,ncls);m=m.to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4) if cfg['optimizer']=='adamw' else torch.optim.SGD(m.parameters(),lr=1e-2,momentum=.9,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);w=torch.tensor((counts.sum()/(ncls*np.maximum(counts,1))),dtype=torch.float32,device=device);crit=nn.CrossEntropyLoss(weight=w if cfg['weighted'] else None,label_smoothing=cfg['label_smoothing']);bestv=-1;state=None
for ep in range(6):
 m.train()
 for x,yb in trl:opt.zero_grad();log=m(x.to(device));loss=focal(log,yb.to(device)) if cfg['loss']=='focal' else crit(log,yb.to(device));loss.backward();opt.step()
 sch.step();vm=evalm(m,val)
 if vm['macro_f1']>bestv:bestv=vm['macro_f1'];state=copy.deepcopy(m.state_dict())
m.load_state_dict(state);best['locked_test']=evalm(m,test);report={'phase':'28_loss_optimizer_ablation','captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':{'model':'efficientnet_b0','mode':'full_finetune','augmentation':'strong_no_hue','configs':configs,'epochs':6,'selection':'validation macro_f1','test_policy':'test only locked winner'},'split_sizes':{'train':len(tr),'val':len(va),'test':len(te)},'screening':results,'locked_winner':best};report['elapsed_sec']=round(time.time()-T0,2);rawj=json.dumps(report,indent=2).encode();report['sha256_self']=hashlib.sha256(rawj).hexdigest();open('/kaggle/working/phase28_loss_optimizer_ablation_report.json','w').write(json.dumps(report,indent=2));print(json.dumps({'winner':best,'elapsed_sec':report['elapsed_sec']},indent=2))
