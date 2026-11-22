"""Phase 15 — uncertainty quantification under distribution shift.
Train the same PlantVillage ResNet-50 reference model, then use MC dropout
for predictive entropy, expected entropy, mutual information and variation
ratio on PV test, PlantDoc and Cassava. OOD scores are evaluated only for
PV-vs-external-domain discrimination; no class-level external accuracy claim.
"""
import os, json, sys, time, hashlib, random, subprocess
print("Installing GPU torch...")
subprocess.run(["pip","install","--quiet","torch==2.5.0","torchvision==0.20.0","--index-url","https://download.pytorch.org/whl/cu121"], check=False, timeout=300)
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

random.seed(42); np.random.seed(42); torch.manual_seed(42)
t0 = time.time()
report = {"phase":"15_uncertainty", "captured_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "protocol": {"seed":42, "mc_passes":4, "dropout_p":0.30, "training_epochs":2}}
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
DATA_ROOT = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"
PD_ROOT = "/kaggle/input/datasets/nirmalsankalana/plantdoc-dataset/test"
CV_ROOT = "/kaggle/input/datasets/nirmalsankalana/cassava-leaf-disease-classification/data"
eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
train_tf = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
full = datasets.ImageFolder(DATA_ROOT, transform=eval_tf)
classes = full.classes; n_classes = len(classes); targets = np.array([s[1] for s in full.samples]); idx=np.arange(len(full))
tr_idx, tmp_idx, tr_y, tmp_y = train_test_split(idx, targets, test_size=.2, random_state=42, stratify=targets)
va_idx, te_idx, _, _ = train_test_split(tmp_idx, tmp_y, test_size=.5, random_state=42, stratify=tmp_y)
print(f"train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}")
def loader(root, tf, indices=None, bs=128, shuffle=False):
    ds=datasets.ImageFolder(root, transform=tf)
    if indices is not None: ds=torch.utils.data.Subset(ds, indices.tolist())
    return DataLoader(ds,batch_size=bs,shuffle=shuffle,num_workers=2,pin_memory=True)
train_loader=loader(DATA_ROOT,train_tf,tr_idx,64,True); val_loader=loader(DATA_ROOT,eval_tf,va_idx); test_loader=loader(DATA_ROOT,eval_tf,te_idx)
model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc=nn.Sequential(nn.Dropout(.30),nn.Linear(model.fc.in_features,n_classes)); model=model.to(device)
counts=np.bincount(targets,minlength=n_classes); w=(1/np.maximum(counts,1)); w=w/w.sum()*n_classes
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); crit=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device))
for ep in range(2):
    model.train(); correct=total=0
    for bi,(x,y) in enumerate(train_loader):
        x,y=x.to(device),y.to(device); opt.zero_grad(); out=model(x); loss=crit(out,y); loss.backward(); opt.step(); correct+=(out.argmax(1)==y).sum().item(); total+=len(y)
        if bi%100==0: print(f"  ep{ep+1} batch{bi}/{len(train_loader)} loss={loss.item():.4f}")
    print(f"epoch {ep+1}: train acc={correct/total:.4f}")

def deterministic(dl):
    model.eval(); z=[]; y=[]
    with torch.no_grad():
        for x,t in dl: z.append(model(x.to(device)).cpu().numpy()); y.append(t.numpy())
    return np.concatenate(z),np.concatenate(y)

def mc_probs(dl, passes=4):
    # Keep batch-normalization frozen while enabling only dropout stochasticity.
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()
    allp=[]; labels=None
    with torch.no_grad():
        for k in range(passes):
            pp=[]; yy=[]
            for x,t in dl:
                p=torch.softmax(model(x.to(device)),1).cpu().numpy(); pp.append(p); yy.append(t.numpy())
            allp.append(np.concatenate(pp)); labels=np.concatenate(yy)
    model.eval()
    return np.stack(allp),labels

def entropy(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(-1)
def summarize(name, mc, labels, has_labels):
    meanp=mc.mean(0); pe=entropy(meanp); ee=entropy(mc).mean(0); mi=pe-ee; preds=mc.argmax(2)
    mode=preds[0].copy()
    for i in range(len(mode)):
        mode[i]=np.bincount(preds[:,i],minlength=n_classes).argmax()
    vr=1-(preds==mode[None,:]).mean(0)
    out={"n":int(len(meanp)),"predictive_entropy_mean":round(float(pe.mean()),4),"predictive_entropy_std":round(float(pe.std()),4),"expected_entropy_mean":round(float(ee.mean()),4),"mutual_information_mean":round(float(mi.mean()),4),"variation_ratio_mean":round(float(vr.mean()),4),"max_conf_mean":round(float(meanp.max(1).mean()),4)}
    if has_labels:
        acc=float((meanp.argmax(1)==labels).mean()); out.update({"accuracy":round(acc,4),"ece":round(float(ece(meanp,labels)),4)})
    return out,pe,mi,vr

def ece(p,y,bins=15):
    c=p.max(1); pred=p.argmax(1); val=0
    for lo,hi in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
        mask=(c>lo)&(c<=hi)
        if mask.any(): val += mask.mean()*abs((pred[mask]==y[mask]).mean()-c[mask].mean())
    return val
print("MC dropout: PlantVillage test")
pv_mc,pv_y=mc_probs(test_loader,4); pv_s,pv_pe,pv_mi,pv_vr=summarize("pv",pv_mc,pv_y,True)
print("MC dropout: PlantDoc")
pd_mc,pd_y=mc_probs(loader(PD_ROOT,eval_tf),4); pd_s,pd_pe,pd_mi,pd_vr=summarize("plantdoc",pd_mc,pd_y,False)
print("MC dropout: Cassava")
cv_mc,cv_y=mc_probs(loader(CV_ROOT,eval_tf),4); cv_s,cv_pe,cv_mi,cv_vr=summarize("cassava",cv_mc,cv_y,False)
# OOD scores: 0=PV, 1=external; labels are domain labels, not disease labels.
for score_name, pv, pd, cv in [("predictive_entropy",pv_pe,pd_pe,cv_pe),("mutual_information",pv_mi,pd_mi,cv_mi),("variation_ratio",pv_vr,pd_vr,cv_vr)]:
    y=np.r_[np.zeros(len(pv)),np.ones(len(pd)+len(cv))]; s=np.r_[pv,pd,cv]
    report.setdefault("ood_auroc",{})[score_name]=round(float(roc_auc_score(y,s)),4)
report["results"]={"plantvillage_test":pv_s,"plantdoc_ood":pd_s,"cassava_ood":cv_s}
report["elapsed_sec"]=round(time.time()-t0,2)
raw=json.dumps(report,indent=2).encode(); report["sha256_self"]=hashlib.sha256(raw).hexdigest()
with open("/kaggle/working/phase15_uncertainty_report.json","w") as f: f.write(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
