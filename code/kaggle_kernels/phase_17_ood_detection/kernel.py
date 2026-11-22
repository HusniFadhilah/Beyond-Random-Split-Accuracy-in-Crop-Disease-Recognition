"""Phase 17 — validation-calibrated OOD detection.
Train a diagnostic PV ResNet-50, reserve PV validation and half of each
external dataset for threshold calibration, then evaluate on held-out PV,
PlantDoc and Cassava. No external class-accuracy claims are made.
"""
import json,time,hashlib,random,subprocess
subprocess.run(["pip","install","--quiet","torch==2.5.0","torchvision==0.20.0","--index-url","https://download.pytorch.org/whl/cu121"],check=False,timeout=300)
import torch,torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,average_precision_score
random.seed(42);np.random.seed(42);torch.manual_seed(42)
t0=time.time();device="cuda" if torch.cuda.is_available() else "cpu"
report={"phase":"17_ood_detection","captured_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"protocol":{"seed":42,"training_epochs":2,"threshold_calibration":"PV validation + 50% of each external dataset","external_labels_used_for_class_accuracy":False}}
print("device:",device)
PV="/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color";PD="/kaggle/input/datasets/nirmalsankalana/plantdoc-dataset/test";CV="/kaggle/input/datasets/nirmalsankalana/cassava-leaf-disease-classification/data"
eval_tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);train_tf=transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
base=datasets.ImageFolder(PV,transform=eval_tf); y=np.array([s[1] for s in base.samples]); idx=np.arange(len(base));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy)
def make(root,tf,indices=None,bs=128,shuffle=False):
 d=datasets.ImageFolder(root,transform=tf)
 if indices is not None:d=torch.utils.data.Subset(d,indices.tolist())
 return DataLoader(d,batch_size=bs,shuffle=shuffle,num_workers=2,pin_memory=True)
train=make(PV,train_tf,tr,64,True);val=make(PV,eval_tf,va);test=make(PV,eval_tf,te);ncls=len(base.classes)
model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2);model.fc=nn.Linear(model.fc.in_features,ncls);model=model.to(device);w=1/np.maximum(np.bincount(y,minlength=ncls),1);w=w/w.sum()*ncls;crit=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
for ep in range(2):
 model.train();correct=total=0
 for bi,(x,t) in enumerate(train):
  x,t=x.to(device),t.to(device);opt.zero_grad();o=model(x);loss=crit(o,t);loss.backward();opt.step();correct+=(o.argmax(1)==t).sum().item();total+=len(t)
  if bi%100==0:print(f"ep{ep+1} batch{bi}/{len(train)} loss={loss.item():.4f}")
 print(f"epoch {ep+1}: train_acc={correct/total:.4f}")
@torch.no_grad()
def logits(dl):
 model.eval();z=[];yy=[]
 for x,t in dl:z.append(model(x.to(device)).cpu().numpy());yy.append(t.numpy())
 return np.concatenate(z),np.concatenate(yy)
def split_external(root):
 ds=datasets.ImageFolder(root,transform=eval_tf);a=np.arange(len(ds));rng=np.random.RandomState(42);rng.shuffle(a);k=len(a)//2;return DataLoader(torch.utils.data.Subset(ds,a[:k].tolist()),batch_size=128,shuffle=False,num_workers=2),DataLoader(torch.utils.data.Subset(ds,a[k:].tolist()),batch_size=128,shuffle=False,num_workers=2),len(ds)
print("extracting PV validation/test and external calibration/evaluation")
va_z,va_y=logits(val);te_z,te_y=logits(test);pd_cal,pd_eval,pd_n=split_external(PD);cv_cal,cv_eval,cv_n=split_external(CV);pdcal_z,_=logits(pd_cal);pdeval_z,_=logits(pd_eval);cvcal_z,_=logits(cv_cal);cveval_z,_=logits(cv_eval)
def softmax(z):z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
def score(z):
 p=softmax(z);s=np.sort(p,axis=1);return {"msp":-p.max(1),"entropy":-(p*np.log(np.clip(p,1e-12,1))).sum(1),"margin":-(s[:,-1]-s[:,-2]),"energy":-np.log(np.exp(z-z.max(1,keepdims=True)).sum(1))-z.max(1)}
sc_va=score(va_z);sc_te=score(te_z);sc_pdc=score(pdcal_z);sc_pde=score(pdeval_z);sc_cvc=score(cvcal_z);sc_cve=score(cveval_z)
report["split_sizes"]={"pv_val":len(va_z),"pv_test":len(te_z),"plantdoc_calibration":len(pdcal_z),"plantdoc_evaluation":len(pdeval_z),"cassava_calibration":len(cvcal_z),"cassava_evaluation":len(cveval_z)}
report["methods"]={}
for name in ["msp","entropy","margin","energy"]:
 cal_ood=np.r_[sc_pdc[name],sc_cvc[name]];test_ood=np.r_[sc_pde[name],sc_cve[name]];test_id=sc_te[name]
 ytest=np.r_[np.zeros(len(test_id)),np.ones(len(test_ood))];stest=np.r_[test_id,test_ood];threshold=float(np.quantile(cal_ood,.05));pred_ood=stest>=threshold;id_fpr=float((test_id>=threshold).mean());ood_tpr=float((test_ood>=threshold).mean());det_err=float(min((id_fpr+(1-ood_tpr))/2,1))
 report["methods"][name]={"auroc":round(float(roc_auc_score(ytest,stest)),4),"auprc":round(float(average_precision_score(ytest,stest)),4),"threshold_at_cal_ood_tpr_0.95":round(threshold,6),"test_ood_tpr":round(ood_tpr,4),"test_id_fpr":round(id_fpr,4),"detection_error":round(det_err,4)}
report["per_domain"]={}
for name in ["msp","entropy","margin","energy"]:
 th=report["methods"][name]["threshold_at_cal_ood_tpr_0.95"];report["per_domain"][name]={"plantdoc_ood_tpr":round(float((sc_pde[name]>=th).mean()),4),"cassava_ood_tpr":round(float((sc_cve[name]>=th).mean()),4)}
report["elapsed_sec"]=round(time.time()-t0,2);raw=json.dumps(report,indent=2).encode();report["sha256_self"]=hashlib.sha256(raw).hexdigest()
with open("/kaggle/working/phase17_ood_detection_report.json","w") as f:f.write(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
