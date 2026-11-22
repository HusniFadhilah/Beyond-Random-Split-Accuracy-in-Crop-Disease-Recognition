"""Phase 19 — frozen-feature data-efficiency analysis.
Extract ImageNet ResNet-50 features once, then train independent linear heads
using 1/5/10/25/50/100 percent of the PlantVillage training split. Validation
and test metrics are reported separately; test is not used for selection.
"""
import json,time,hashlib,random,subprocess
subprocess.run(["pip","install","--quiet","torch==2.5.0","torchvision==0.20.0","--index-url","https://download.pytorch.org/whl/cu121"],check=False,timeout=300)
import torch,torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score
random.seed(42);np.random.seed(42);torch.manual_seed(42)
t0=time.time();device="cuda" if torch.cuda.is_available() else "cpu"
report={"phase":"19_data_efficiency","captured_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"protocol":{"seed":42,"backbone":"ResNet-50 ImageNet pretrained frozen features","head":"linear softmax classifier","head_epochs":10,"fractions":[0.01,0.05,0.10,0.25,0.50,1.0]}}
print("device:",device)
PV="/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color";tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
ds=datasets.ImageFolder(PV,transform=tf);y=np.array([s[1] for s in ds.samples]);idx=np.arange(len(ds));tr,tmp,tr_y,tmp_y=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmp_y,test_size=.5,random_state=42,stratify=tmp_y);ncls=len(ds.classes)
backbone=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2);backbone.fc=nn.Identity();backbone=backbone.to(device).eval()
def feats(indices):
 dl=DataLoader(torch.utils.data.Subset(ds,indices.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);out=[];yy=[]
 with torch.no_grad():
  for x,t in dl:out.append(backbone(x.to(device)).cpu());yy.append(t)
 return torch.cat(out),torch.cat(yy)
print("extracting frozen features")
Xtr,Ytr=feats(tr);Xva,Yva=feats(va);Xte,Yte=feats(te);dim=Xtr.shape[1];print("feature_shapes",tuple(Xtr.shape),tuple(Xva.shape),tuple(Xte.shape))
def metrics(model,X,Y):
 model.eval()
 with torch.no_grad():p=model(X.to(device)).argmax(1).cpu().numpy()
 yy=Y.numpy();return {"accuracy":round(float(accuracy_score(yy,p)),4),"balanced_accuracy":round(float(balanced_accuracy_score(yy,p)),4),"macro_f1":round(float(f1_score(yy,p,average="macro",zero_division=0)),4)}
results=[]
for frac in report["protocol"]["fractions"]:
 n=max(ncls,int(round(len(Xtr)*frac)));rng=np.random.RandomState(42);chosen=[]
 # stratified fractional subset, preserving at least one sample per class.
 for c in range(ncls):
  ci=np.where(Ytr.numpy()==c)[0];k=max(1,int(round(len(ci)*frac)));chosen.extend(rng.choice(ci,size=min(k,len(ci)),replace=False).tolist())
 chosen=np.array(sorted(set(chosen)));head=nn.Linear(dim,ncls).to(device);opt=torch.optim.AdamW(head.parameters(),lr=1e-2,weight_decay=1e-4);crit=nn.CrossEntropyLoss();loader=DataLoader(TensorDataset(Xtr[chosen],Ytr[chosen]),batch_size=256,shuffle=True)
 for ep in range(10):
  head.train()
  for x,t in loader:opt.zero_grad();loss=crit(head(x.to(device)),t.to(device));loss.backward();opt.step()
 vm=metrics(head,Xva,Yva);tm=metrics(head,Xte,Yte);row={"fraction":frac,"n_train":int(len(chosen)),"val":vm,"test":tm};results.append(row);print(row)
report["n_split"]={"train":len(tr),"val":len(va),"test":len(te),"n_classes":ncls,"feature_dim":int(dim)};report["results"]=results;report["elapsed_sec"]=round(time.time()-t0,2);raw=json.dumps(report,indent=2).encode();report["sha256_self"]=hashlib.sha256(raw).hexdigest()
with open("/kaggle/working/phase19_data_efficiency_report.json","w") as f:f.write(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
