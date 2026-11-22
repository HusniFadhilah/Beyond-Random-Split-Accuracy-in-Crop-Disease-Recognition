"""Phase 26: matched-seed confirmation of full EfficientNet fine-tuning."""
import subprocess
subprocess.run(["pip","install","--quiet","--retries","10","--timeout","60","torch==2.3.1","torchvision==0.18.1"],check=True,timeout=900)
import json,time,hashlib,random,copy
import numpy as np,torch,torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
T0=time.time(); device='cuda' if torch.cuda.is_available() else 'cpu'; print('device:',device)
if device!='cuda': raise RuntimeError('CUDA GPU required; refusing CPU fallback')
PV='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color'; mean=[.485,.456,.406];std=[.229,.224,.225]
tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.75,1.0)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.2,hue=0.0),transforms.ToTensor(),transforms.Normalize(mean,std)])
evaltf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean,std)])
raw=datasets.ImageFolder(PV); y=np.array([s[1] for s in raw.samples]); idx=np.arange(len(raw));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy);ncls=len(raw.classes)
def met(y,p):return {'accuracy':round(float(accuracy_score(y,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),6),'macro_f1':round(float(f1_score(y,p,average='macro',zero_division=0)),6)}
def evaluate(m,loader):
 m.eval();yy=[];pp=[]
 with torch.no_grad():
  for x,yb in loader:pp.extend(m(x.to(device)).argmax(1).cpu().numpy());yy.extend(yb.numpy())
 return met(np.array(yy),np.array(pp))
results=[]
for seed in [7,42,123]:
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 tra=datasets.ImageFolder(PV,transform=tf); eva=datasets.ImageFolder(PV,transform=evaltf)
 trl=DataLoader(Subset(tra,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);val=DataLoader(Subset(eva,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);test=DataLoader(Subset(eva,te.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True)
 m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier[1]=nn.Linear(m.classifier[1].in_features,ncls);m=m.to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);crit=nn.CrossEntropyLoss();best=-1;best_ep=0;best_state=None
 for ep in range(8):
  m.train()
  for x,yb in trl:opt.zero_grad();loss=crit(m(x.to(device)),yb.to(device));loss.backward();opt.step()
  vm=evaluate(m,val);print('seed',seed,'epoch',ep+1,vm['macro_f1'])
  if vm['macro_f1']>best:best=vm['macro_f1'];best_ep=ep+1;best_state=copy.deepcopy(m.state_dict())
 m.load_state_dict(best_state); tm=evaluate(m,test);results.append({'seed':seed,'best_epoch':best_ep,'best_val':{'macro_f1':round(float(best),6)},'test':tm})
report={'phase':'26_multiseed_full_finetune','captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':{'model':'efficientnet_b0','mode':'full_finetune','augmentation':'strong','seeds':[7,42,123],'epochs':8,'selection':'best validation macro_f1 checkpoint per seed','test_policy':'test evaluated only after validation checkpoint is locked'},'split_sizes':{'train':len(tr),'val':len(va),'test':len(te),'n_classes':ncls},'per_seed':results,'aggregates':{}}
for split in ['best_val','test']:
 for metric in (['macro_f1'] if split=='best_val' else ['accuracy','balanced_accuracy','macro_f1']):
  vals=[r[split][metric] for r in results];report['aggregates'].setdefault(split,{})[metric]={'mean':round(float(np.mean(vals)),6),'sd':round(float(np.std(vals,ddof=1)),6),'values':vals}
report['elapsed_sec']=round(time.time()-T0,2);rawj=json.dumps(report,indent=2).encode();report['sha256_self']=hashlib.sha256(rawj).hexdigest()
with open('/kaggle/working/phase26_multiseed_finetune_report.json','w') as f:f.write(json.dumps(report,indent=2))
print(json.dumps({'phase':report['phase'],'elapsed_sec':report['elapsed_sec'],'aggregates':report['aggregates']},indent=2))
