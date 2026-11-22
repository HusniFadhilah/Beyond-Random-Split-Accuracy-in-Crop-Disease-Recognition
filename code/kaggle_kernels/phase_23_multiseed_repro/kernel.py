"""Phase 23: multi-seed reproducibility for the frozen model-family benchmark."""
import json,time,hashlib,random,subprocess
subprocess.run(["pip","install","--quiet","torch==2.5.0","torchvision==0.20.0","--index-url","https://download.pytorch.org/whl/cu121"],check=False,timeout=300)
import torch,torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset,TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
SEEDS=[7,42,123]; MODELS=['resnet50','mobilenet_v3_large','efficientnet_b0']; random.seed(42);np.random.seed(42);torch.manual_seed(42)
t0=time.time();device='cuda' if torch.cuda.is_available() else 'cpu';print('device:',device)
PV='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color';tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);ds=datasets.ImageFolder(PV,transform=tf);y=np.array([s[1] for s in ds.samples]);idx=np.arange(len(ds));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy);ncls=len(ds.classes)
def loader(a):return DataLoader(Subset(ds,a.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True)
def build(name):
 if name=='resnet50':m=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2);m.fc=nn.Identity()
 elif name=='mobilenet_v3_large':m=models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2);m.classifier=nn.Identity()
 else:m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier=nn.Identity()
 return m.to(device).eval()
def extract(m,dl):
 z=[];yy=[]
 with torch.no_grad():
  for x,yb in dl:z.append(m(x.to(device)).flatten(1).cpu());yy.append(yb)
 return torch.cat(z),torch.cat(yy)
def metrics(y,p):return {'accuracy':round(float(accuracy_score(y,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),6),'macro_f1':round(float(f1_score(y,p,average='macro',zero_division=0)),6)}
report={'phase':'23_multiseed_reproducibility','captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':{'seeds':SEEDS,'backbone_mode':'ImageNet pretrained frozen features','head':'linear classifier','head_epochs':10,'models':MODELS,'split_seed':42,'aggregate':'mean and sample SD over seeds'}}
tr_dl,va_dl,te_dl=loader(tr),loader(va),loader(te);results={}
for name in MODELS:
 print('model',name);m=build(name);Xtr,Ytr=extract(m,tr_dl);Xva,Yva=extract(m,va_dl);Xte,Yte=extract(m,te_dl);per=[]
 for seed in SEEDS:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);head=nn.Linear(Xtr.shape[1],ncls).to(device);opt=torch.optim.AdamW(head.parameters(),lr=1e-2,weight_decay=1e-4);crit=nn.CrossEntropyLoss();dl=DataLoader(TensorDataset(Xtr,Ytr),batch_size=256,shuffle=True)
  for _ in range(10):
   head.train()
   for xb,yb in dl:opt.zero_grad();crit(head(xb.to(device)),yb.to(device)).backward();opt.step()
  head.eval()
  with torch.no_grad():pv=head(Xva.to(device)).argmax(1).cpu().numpy();pt=head(Xte.to(device)).argmax(1).cpu().numpy()
  per.append({'seed':seed,'val':metrics(Yva.numpy(),pv),'test':metrics(Yte.numpy(),pt)})
 results[name]={'feature_dim':int(Xtr.shape[1]),'per_seed':per}
report['split_sizes']={'train':int(len(tr)),'val':int(len(va)),'test':int(len(te)),'n_classes':ncls};report['results']=results
for name in MODELS:
 for split in ['val','test']:
  for metric in ['accuracy','balanced_accuracy','macro_f1']:
   vals=[x[split][metric] for x in results[name]['per_seed']];report.setdefault('aggregates',{}).setdefault(name,{}).setdefault(split,{})[metric]={'mean':round(float(np.mean(vals)),6),'sd':round(float(np.std(vals,ddof=1)),6),'values':vals}
report['elapsed_sec']=round(time.time()-t0,2);raw=json.dumps(report,indent=2).encode();report['sha256_self']=hashlib.sha256(raw).hexdigest()
with open('/kaggle/working/phase23_multiseed_reproducibility_report.json','w') as f:f.write(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
