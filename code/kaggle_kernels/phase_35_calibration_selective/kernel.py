"""Phase 35: calibration and selective prediction for ResNet-50 224px."""
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
def collect(m,dl):
 m.eval();ps=[];ys=[]
 with torch.no_grad():
  for x,yb in dl:ps.append(torch.softmax(m(x.to(device)),1).cpu().numpy());ys.append(yb.numpy())
 return np.concatenate(ps),np.concatenate(ys)
def met(y,p):return {'accuracy':round(float(accuracy_score(y,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),6),'macro_f1':round(float(f1_score(y,p,average='macro',zero_division=0)),6)}
def ece(p,y,bins=15):
 conf=p.max(1);pred=p.argmax(1);edges=np.linspace(0,1,bins+1);out=0
 for a,b in zip(edges[:-1],edges[1:]):
  q=(conf>a)&(conf<=b)
  if q.any():out+=q.mean()*abs((pred[q]==y[q]).mean()-conf[q].mean())
 return float(out)
m=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2);m.fc=nn.Linear(m.fc.in_features,ncls);m=m.to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);crit=nn.CrossEntropyLoss(label_smoothing=.1);best=-1;state=None
for ep in range(6):
 m.train()
 for x,yb in trl:opt.zero_grad();loss=crit(m(x.to(device)),yb.to(device));loss.backward();opt.step()
 sch.step();p,yy=collect(m,val);vf=f1_score(yy,p.argmax(1),average='macro');print('epoch',ep+1,'val_macro_f1',vf)
 if vf>best:best=vf;state=copy.deepcopy(m.state_dict())
m.load_state_dict(state);vp,vy=collect(m,val);tp,ty=collect(m,test);base=met(ty,tp.argmax(1));thresholds=[.50,.60,.70,.80,.90,.95,.99];selective=[]
for q in thresholds:
 keep=tp.max(1)>=q
 selective.append({'threshold':q,'coverage':round(float(keep.mean()),6),'n_accepted':int(keep.sum()),'accuracy_on_accepted':round(float((tp.argmax(1)[keep]==ty[keep]).mean()),6) if keep.any() else None,'macro_f1_on_accepted':round(float(f1_score(ty[keep],tp.argmax(1)[keep],average='macro',zero_division=0)),6) if keep.any() else None})
report={'phase':'35_calibration_selective_resnet50','captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':{'model':'resnet50','resolution':224,'mode':'full_finetune','augmentation':'basic_resize_crop_flip','optimizer':'adamw','loss':'cross_entropy','label_smoothing':.1,'scheduler':'cosine','epochs':6,'seed':42,'selection':'validation macro_f1','test_policy':'test only locked validation checkpoint','ece_bins':15},'split_sizes':{'train':len(tr),'val':len(va),'test':len(te)},'best_val_macro_f1':round(float(best),6),'validation_ece':round(ece(vp,vy),6),'test_ece':round(ece(tp,ty),6),'test_metrics':base,'selective_test':selective};report['elapsed_sec']=round(time.time()-T0,2);rawj=json.dumps(report,indent=2).encode();report['sha256_self']=hashlib.sha256(rawj).hexdigest();open('/kaggle/working/phase35_calibration_selective_report.json','w').write(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
