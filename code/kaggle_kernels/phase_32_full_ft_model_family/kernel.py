"""Phase 32: fair full-fine-tuning model-family comparison."""
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
trtf=transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)]);evtf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean,std)])
raw=datasets.ImageFolder(PV);y=np.array([s[1] for s in raw.samples]);idx=np.arange(len(raw));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy);ncls=len(raw.classes)
def metric(y,p):return {'accuracy':round(float(accuracy_score(y,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),6),'macro_f1':round(float(f1_score(y,p,average='macro',zero_division=0)),6)}
def evalm(m,dl):
 m.eval();yy=[];pp=[]
 with torch.no_grad():
  for x,yb in dl:pp.extend(m(x.to(device)).argmax(1).cpu().numpy());yy.extend(yb.numpy())
 return metric(np.array(yy),np.array(pp))
def build(name):
 if name=='resnet50':m=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2);m.fc=nn.Linear(m.fc.in_features,ncls)
 elif name=='mobilenet_v3_large':m=models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2);m.classifier[3]=nn.Linear(m.classifier[3].in_features,ncls)
 elif name=='efficientnet_b0':m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier[1]=nn.Linear(m.classifier[1].in_features,ncls)
 return m.to(device)
models_to_run=['resnet50','mobilenet_v3_large','efficientnet_b0'];results=[]
for name in models_to_run:
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);trds=datasets.ImageFolder(PV,transform=trtf);evads=datasets.ImageFolder(PV,transform=evtf);trl=DataLoader(Subset(trds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);val=DataLoader(Subset(evads,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);m=build(name);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);crit=nn.CrossEntropyLoss(label_smoothing=.1);best=-1;be=0
 for ep in range(6):
  m.train()
  for x,yb in trl:opt.zero_grad();loss=crit(m(x.to(device)),yb.to(device));loss.backward();opt.step()
  sch.step();vm=evalm(m,val);print(name,'epoch',ep+1,vm['macro_f1'])
  if vm['macro_f1']>best:best=vm['macro_f1'];be=ep+1
 results.append({'model':name,'best_val_macro_f1':round(float(best),6),'best_epoch':be,'trainable_params':sum(p.numel() for p in m.parameters())})
results.sort(key=lambda z:z['best_val_macro_f1'],reverse=True);winner=results[0];name=winner['model'];random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);trds=datasets.ImageFolder(PV,transform=trtf);evads=datasets.ImageFolder(PV,transform=evtf);trl=DataLoader(Subset(trds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);val=DataLoader(Subset(evads,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);test=DataLoader(Subset(evads,te.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);m=build(name);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);crit=nn.CrossEntropyLoss(label_smoothing=.1);bestv=-1;state=None
for ep in range(6):
 m.train()
 for x,yb in trl:opt.zero_grad();loss=crit(m(x.to(device)),yb.to(device));loss.backward();opt.step()
 sch.step();vm=evalm(m,val)
 if vm['macro_f1']>bestv:bestv=vm['macro_f1'];state=copy.deepcopy(m.state_dict())
m.load_state_dict(state);winner['locked_test']=evalm(m,test);report={'phase':'32_full_finetune_model_family','captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':{'models':models_to_run,'mode':'full_finetune','augmentation':'basic_resize_crop_flip','optimizer':'adamw','loss':'cross_entropy','label_smoothing':.1,'scheduler':'cosine','epochs':6,'selection':'validation macro_f1','test_policy':'test only locked winner'},'split_sizes':{'train':len(tr),'val':len(va),'test':len(te)},'screening':results,'locked_winner':winner};report['elapsed_sec']=round(time.time()-T0,2);rawj=json.dumps(report,indent=2).encode();report['sha256_self']=hashlib.sha256(rawj).hexdigest();open('/kaggle/working/phase32_full_finetune_model_family_report.json','w').write(json.dumps(report,indent=2));print(json.dumps({'winner':winner,'elapsed_sec':report['elapsed_sec']},indent=2))
