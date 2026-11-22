"""Phase 38: calibrated selective prediction for the locked EfficientNet-B0 pipeline."""
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
random.seed(42);np.random.seed(42);torch.manual_seed(42);raw=datasets.ImageFolder(PV);y=np.array([s[1] for s in raw.samples]);idx=np.arange(len(raw));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy);ncls=len(raw.classes)
trds=datasets.ImageFolder(PV,transform=trtf);evads=datasets.ImageFolder(PV,transform=evtf);trl=DataLoader(Subset(trds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);val=DataLoader(Subset(evads,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);test=DataLoader(Subset(evads,te.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True)
def logits(m,dl):
 m.eval();ls=[];ys=[]
 with torch.no_grad():
  for x,yb in dl:ls.append(m(x.to(device)).cpu());ys.append(yb)
 return torch.cat(ls),torch.cat(ys)
def metric(y,p):return {'accuracy':round(float(accuracy_score(y,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),6),'macro_f1':round(float(f1_score(y,p,average='macro',zero_division=0)),6)}
def ece(p,y,bins=15):
 conf=p.max(1).values.numpy();pred=p.argmax(1).numpy();yy=y.numpy();edges=np.linspace(0,1,bins+1);out=0
 for a,b in zip(edges[:-1],edges[1:]):
  q=(conf>a)&(conf<=b)
  if q.any():out+=q.mean()*abs((pred[q]==yy[q]).mean()-conf[q].mean())
 return round(float(out),6)
m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier[1]=nn.Linear(m.classifier[1].in_features,ncls);m=m.to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);crit=nn.CrossEntropyLoss(label_smoothing=.1);best=-1;state=None
for ep in range(6):
 m.train()
 for x,yb in trl:opt.zero_grad();loss=crit(m(x.to(device)),yb.to(device));loss.backward();opt.step()
 sch.step();vl,vy=logits(m,val);vf=f1_score(vy.numpy(),vl.argmax(1).numpy(),average='macro');print('epoch',ep+1,'val_macro_f1',vf)
 if vf>best:best=vf;state=copy.deepcopy(m.state_dict())
m.load_state_dict(state);vl,vy=logits(m,val);tl,ty=logits(m,test);temperature=torch.nn.Parameter(torch.ones(1));to=torch.optim.LBFGS([temperature],lr=.1,max_iter=50,line_search_fn='strong_wolfe');nll=nn.CrossEntropyLoss()
def closure():
 to.zero_grad();loss=nll(vl/temperature.clamp_min(.05),vy);loss.backward();return loss
to.step(closure);temp=float(temperature.detach().clamp_min(.05));vp=torch.softmax(vl/temp,1);tp=torch.softmax(tl/temp,1);thresholds=[.50,.60,.70,.80,.90,.95,.99];selective=[]
for q in thresholds:
 keep=tp.max(1).values.numpy()>=q; pred=tp.argmax(1).numpy(); yy=ty.numpy(); selective.append({'threshold':q,'coverage':round(float(keep.mean()),6),'n_accepted':int(keep.sum()),'accuracy_on_accepted':round(float((pred[keep]==yy[keep]).mean()),6) if keep.any() else None,'macro_f1_on_accepted':round(float(f1_score(yy[keep],pred[keep],average='macro',zero_division=0)),6) if keep.any() else None})
def corruption_metrics(kind):
 base=datasets.ImageFolder(PV,transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()]));dl=DataLoader(Subset(base,te.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);ls=[];ys=[];mu=torch.tensor(mean,device=device).view(1,3,1,1);sd=torch.tensor(std,device=device).view(1,3,1,1);torch.manual_seed(2026)
 m.eval()
 with torch.no_grad():
  for x,yb in dl:
   x=x.to(device)
   if kind=='gaussian_noise': x=(x+0.08*torch.randn_like(x)).clamp(0,1)
   elif kind=='brightness': x=(x*0.55).clamp(0,1)
   elif kind=='contrast': x=((x-0.5)*1.8+0.5).clamp(0,1)
   elif kind=='blur': x=transforms.functional.gaussian_blur(x,[9,9],[3.0,3.0])
   ls.append(m((x-mu)/sd).cpu());ys.append(yb)
 l=torch.cat(ls);y=torch.cat(ys);p=torch.softmax(l/temp,1);return {'accuracy':accuracy_score(y.numpy(),p.argmax(1).numpy()),'balanced_accuracy':balanced_accuracy_score(y.numpy(),p.argmax(1).numpy()),'macro_f1':f1_score(y.numpy(),p.argmax(1).numpy(),average='macro',zero_division=0),'ece':ece(p,y)}
corruptions={k:corruption_metrics(k) for k in ['gaussian_noise','brightness','contrast','blur']}
report={'phase':'40_corruption_robustness_efficientnet_b0_basic','captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':{'model':'efficientnet_b0','mode':'full_finetune','augmentation':'basic_resize_crop_flip','corruptions':{'gaussian_noise':'sigma_0.08_pixel_space','brightness':'factor_0.55','contrast':'factor_1.8','blur':'gaussian_kernel_9_sigma_3'},'checkpoint':'best validation macro_f1','temperature_fit':'validation NLL only','test_policy':'clean and corrupted test evaluated after checkpoint and temperature are locked'},'split_sizes':{'train':len(tr),'val':len(va),'test':len(te)},'best_val_macro_f1':round(float(best),6),'temperature':round(temp,6),'clean_test':metric(ty.numpy(),tp.argmax(1).numpy())|{'ece':ece(tp,ty)},'corruptions':corruptions};report['elapsed_sec']=round(time.time()-T0,2);rawj=json.dumps(report,indent=2).encode();report['sha256_self']=hashlib.sha256(rawj).hexdigest();open('/kaggle/working/phase40_corruption_robustness_report.json','w').write(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
