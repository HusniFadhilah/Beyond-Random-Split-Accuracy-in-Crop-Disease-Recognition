"""Phase 16 — selective prediction under uncertainty.
MC-dropout ResNet-50 on the PlantVillage held-out test set. Compute
risk-coverage curves and AURC for confidence, predictive entropy, mutual
information and variation ratio. External domains are retained only for
score-distribution context because their disease labels are incompatible.
"""
import json,time,hashlib,random,subprocess
subprocess.run(["pip","install","--quiet","torch==2.5.0","torchvision==0.20.0","--index-url","https://download.pytorch.org/whl/cu121"],check=False,timeout=300)
import torch, torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
random.seed(42); np.random.seed(42); torch.manual_seed(42)
t0=time.time(); device="cuda" if torch.cuda.is_available() else "cpu"
report={"phase":"16_selective_prediction","captured_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"protocol":{"seed":42,"mc_passes":4,"dropout_p":0.30,"training_epochs":2}}
print("device:",device)
PV="/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"; PD="/kaggle/input/datasets/nirmalsankalana/plantdoc-dataset/test"; CV="/kaggle/input/datasets/nirmalsankalana/cassava-leaf-disease-classification/data"
eval_tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
train_tf=transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
base=datasets.ImageFolder(PV,transform=eval_tf); y=np.array([s[1] for s in base.samples]); idx=np.arange(len(base)); tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y); va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy)
def make(root,tf,indices=None,bs=128,shuffle=False):
 d=datasets.ImageFolder(root,transform=tf)
 if indices is not None:d=torch.utils.data.Subset(d,indices.tolist())
 return DataLoader(d,batch_size=bs,shuffle=shuffle,num_workers=2,pin_memory=True)
train=make(PV,train_tf,tr,64,True); test=make(PV,eval_tf,te); ncls=len(base.classes)
model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); model.fc=nn.Sequential(nn.Dropout(.30),nn.Linear(model.fc.in_features,ncls)); model=model.to(device)
w=1/np.maximum(np.bincount(y,minlength=ncls),1); w=w/w.sum()*ncls; crit=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device)); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
for ep in range(2):
 model.train(); correct=total=0
 for bi,(x,t) in enumerate(train):
  x,t=x.to(device),t.to(device); opt.zero_grad(); out=model(x); loss=crit(out,t); loss.backward(); opt.step(); correct+=(out.argmax(1)==t).sum().item(); total+=len(t)
  if bi%100==0: print(f"ep{ep+1} batch{bi}/{len(train)} loss={loss.item():.4f}")
 print(f"epoch {ep+1}: train_acc={correct/total:.4f}")
def mc(dl,K=4):
 model.eval()
 for m in model.modules():
  if isinstance(m,nn.Dropout):m.train()
 passes=[]; labels=None
 with torch.no_grad():
  for k in range(K):
   out=[]; yy=[]
   for x,t in dl: out.append(torch.softmax(model(x.to(device)),1).cpu().numpy()); yy.append(t.numpy())
   passes.append(np.concatenate(out)); labels=np.concatenate(yy)
 model.eval(); return np.stack(passes),labels
mc_p,labels=mc(test); meanp=mc_p.mean(0); pred=meanp.argmax(1); correct=(pred==labels).astype(float)
def H(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(-1)
pe=H(meanp); mi=pe-H(mc_p).mean(0); vote=mc_p.argmax(2); mode=np.apply_along_axis(lambda z:np.bincount(z,minlength=ncls).argmax(),0,vote); vr=1-(vote==mode[None,:]).mean(0)
def curve(score,lower_is_better=False):
 order=np.argsort(score) if lower_is_better else np.argsort(-score); c=correct[order]; cum=np.cumsum(c); rows=[]
 for cov in np.linspace(.1,1.0,10):
  n=max(1,int(round(cov*len(c)))); rows.append({"coverage":round(float(n/len(c)),4),"n_retained":n,"accuracy":round(float(cum[n-1]/n),4),"selective_risk":round(float(1-cum[n-1]/n),4)})
 # AURC: mean risk over every retained-prefix coverage, lower is better.
 risks=1-cum/np.arange(1,len(c)+1); aurc=float(risks.mean())
 return {"aurc":round(aurc,6),"curves":rows}
report["in_domain"]={"n":int(len(labels)),"base_accuracy":round(float(correct.mean()),4),"base_ece":round(float(np.abs(meanp.max(1)-correct).mean()),4),"confidence_mean":round(float(meanp.max(1).mean()),4),"confidence":curve(meanp.max(1)),"predictive_entropy":curve(pe,True),"mutual_information":curve(mi,True),"variation_ratio":curve(vr,True)}
# External score context only; no risk/accuracy because class spaces are incompatible.
for name,root in [("plantdoc",PD),("cassava",CV)]:
 dl=make(root,eval_tf); model.eval(); conf=[]; ent=[]
 with torch.no_grad():
  for x,_ in dl:
   p=torch.softmax(model(x.to(device)),1).cpu().numpy(); conf.append(p.max(1)); ent.append(H(p))
 conf=np.concatenate(conf); ent=np.concatenate(ent); report.setdefault("external_score_context",{})[name]={"n":int(len(conf)),"max_conf_mean":round(float(conf.mean()),4),"predictive_entropy_mean":round(float(ent.mean()),4)}
report["elapsed_sec"]=round(time.time()-t0,2); raw=json.dumps(report,indent=2).encode(); report["sha256_self"]=hashlib.sha256(raw).hexdigest()
with open("/kaggle/working/phase16_selective_prediction_report.json","w") as f:f.write(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
