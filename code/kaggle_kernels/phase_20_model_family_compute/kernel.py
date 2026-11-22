"""Phase 20 — frozen model-family and computational benchmark.
Compare ResNet-50, MobileNetV3-Large and EfficientNet-B0 using the same
PlantVillage split, frozen ImageNet features and linear classifier head.
Report accuracy metrics and basic compute measurements.
"""
import json,time,hashlib,random,subprocess
subprocess.run(["pip","install","--quiet","torch==2.5.0","torchvision==0.20.0","--index-url","https://download.pytorch.org/whl/cu121"],check=False,timeout=300)
import torch,torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset,TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score
random.seed(42);np.random.seed(42);torch.manual_seed(42)
t0=time.time();device="cuda" if torch.cuda.is_available() else "cpu"
report={"phase":"20_model_family_compute","captured_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"protocol":{"seed":42,"backbone_mode":"ImageNet pretrained frozen features","head":"linear classifier","head_epochs":10,"models":["resnet50","mobilenet_v3_large","efficientnet_b0"]}}
print("device:",device)
PV="/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color";tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);ds=datasets.ImageFolder(PV,transform=tf);y=np.array([s[1] for s in ds.samples]);idx=np.arange(len(ds));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy);ncls=len(ds.classes)
def loader(indices):return DataLoader(Subset(ds,indices.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True)
tr_dl=loader(tr);va_dl=loader(va);te_dl=loader(te)
def build(name):
 if name=="resnet50":m=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2);m.fc=nn.Identity()
 elif name=="mobilenet_v3_large":m=models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2);m.classifier=nn.Identity()
 else:m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier=nn.Identity()
 return m.to(device).eval()
def extract(m,dl):
 z=[];yy=[];t=time.perf_counter()
 with torch.no_grad():
  for x,yb in dl:z.append(m(x.to(device)).flatten(1).cpu());yy.append(yb)
 return torch.cat(z),torch.cat(yy),time.perf_counter()-t
def metrics(head,X,Y):
 head.eval();
 with torch.no_grad():p=head(X.to(device)).argmax(1).cpu().numpy()
 yy=Y.numpy();return {"accuracy":round(float(accuracy_score(yy,p)),4),"balanced_accuracy":round(float(balanced_accuracy_score(yy,p)),4),"macro_f1":round(float(f1_score(yy,p,average="macro",zero_division=0)),4)}
results={}
for name in report["protocol"]["models"]:
 print("model",name);m=build(name);params=sum(p.numel() for p in m.parameters());
 Xtr,Ytr,tt=extract(m,tr_dl);Xva,Yva,tv=extract(m,va_dl);Xte,Yte,te_time=extract(m,te_dl);dim=Xtr.shape[1]
 head=nn.Linear(dim,ncls).to(device);opt=torch.optim.AdamW(head.parameters(),lr=1e-2,weight_decay=1e-4);crit=nn.CrossEntropyLoss();dl=DataLoader(TensorDataset(Xtr,Ytr),batch_size=256,shuffle=True)
 for ep in range(10):
  head.train()
  for x,yb in dl:opt.zero_grad();loss=crit(head(x.to(device)),yb.to(device));loss.backward();opt.step()
 # latency on a fixed test batch, after warmup.
 xb=Xte[:128].to(device); 
 with torch.no_grad():
  for _ in range(3):head(xb)
  if device=="cuda":torch.cuda.synchronize()
  a=time.perf_counter();
  for _ in range(10):head(xb)
  if device=="cuda":torch.cuda.synchronize()
  latency_ms=(time.perf_counter()-a)*1000/(10*len(xb))
 results[name]={"total_backbone_params":int(params),"feature_dim":int(dim),"feature_extract_seconds":{"train":round(tt,3),"val":round(tv,3),"test":round(te_time,3)},"linear_head_params":int(dim*ncls+ncls),"linear_head_inference_ms_per_image":round(float(latency_ms),4),"val":metrics(head,Xva,Yva),"test":metrics(head,Xte,Yte)}
 print(results[name])
report["split_sizes"]={"train":len(tr),"val":len(va),"test":len(te),"n_classes":ncls};report["results"]=results;report["elapsed_sec"]=round(time.time()-t0,2);raw=json.dumps(report,indent=2).encode();report["sha256_self"]=hashlib.sha256(raw).hexdigest()
with open("/kaggle/working/phase20_model_family_compute_report.json","w") as f:f.write(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
