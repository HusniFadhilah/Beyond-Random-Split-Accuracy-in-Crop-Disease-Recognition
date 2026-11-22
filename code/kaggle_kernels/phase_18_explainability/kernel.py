"""Phase 18 — explainability and shortcut diagnostics.
Train a diagnostic ResNet-50 and compute Grad-CAM plus coarse occlusion
sensitivity on bounded samples from PlantVillage, PlantDoc and Cassava.
No lesion masks exist, so no lesion-localization score is claimed.
"""
import json,time,hashlib,random,subprocess
subprocess.run(["pip","install","--quiet","torch==2.5.0","torchvision==0.20.0","--index-url","https://download.pytorch.org/whl/cu121"],check=False,timeout=300)
import torch,torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset
import numpy as np
from sklearn.model_selection import train_test_split
random.seed(42);np.random.seed(42);torch.manual_seed(42)
t0=time.time();device="cuda" if torch.cuda.is_available() else "cpu"
report={"phase":"18_explainability","captured_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"protocol":{"seed":42,"training_epochs":2,"samples_per_domain":50,"occlusion_grid":"3x3","lesion_masks_available":False}}
print("device:",device)
PV="/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color";PD="/kaggle/input/datasets/nirmalsankalana/plantdoc-dataset/test";CV="/kaggle/input/datasets/nirmalsankalana/cassava-leaf-disease-classification/data"
eval_tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);train_tf=transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
base=datasets.ImageFolder(PV,transform=eval_tf);y=np.array([s[1] for s in base.samples]);idx=np.arange(len(base));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy)
def make(root,tf,indices=None,bs=64,shuffle=False):
 d=datasets.ImageFolder(root,transform=tf)
 if indices is not None:d=Subset(d,indices.tolist())
 return DataLoader(d,batch_size=bs,shuffle=shuffle,num_workers=2,pin_memory=True)
train=make(PV,train_tf,tr,64,True);ncls=len(base.classes)
model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2);model.fc=nn.Linear(model.fc.in_features,ncls);model=model.to(device);w=1/np.maximum(np.bincount(y,minlength=ncls),1);w=w/w.sum()*ncls;crit=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
for ep in range(2):
 model.train();correct=total=0
 for bi,(x,t) in enumerate(train):
  x,t=x.to(device),t.to(device);opt.zero_grad();o=model(x);loss=crit(o,t);loss.backward();opt.step();correct+=(o.argmax(1)==t).sum().item();total+=len(t)
  if bi%100==0:print(f"ep{ep+1} batch{bi}/{len(train)} loss={loss.item():.4f}")
 print(f"epoch {ep+1}: train_acc={correct/total:.4f}")
# Hooks for Grad-CAM on the final residual block.
acts=[];grads=[]
def fh(m,i,o):acts.clear();acts.append(o)
def bh(m,gi,go):grads.clear();grads.append(go[0])
h1=model.layer4[-1].register_forward_hook(fh);h2=model.layer4[-1].register_full_backward_hook(bh)

def cam_one(x):
 model.eval();model.zero_grad(set_to_none=True);out=model(x[None].to(device));pred=out.argmax(1);out[0,pred].backward();a=acts[-1].detach();g=grads[-1].detach();weights=g.mean((2,3),keepdim=True);c=torch.relu((weights*a).sum(1,keepdim=True));c=torch.nn.functional.interpolate(c,size=(224,224),mode="bilinear",align_corners=False)[0,0].cpu().numpy();c=(c-c.min())/(c.max()-c.min()+1e-12);return c,float(torch.softmax(out,1).max().item()),int(pred.item())
def occlusion_one(x):
 model.eval()
 with torch.no_grad():basep=torch.softmax(model(x[None].to(device)),1);pred=basep.argmax(1);baseconf=basep[0,pred].item()
 drops=[];H,W=224,224;step=H//3;patch=step
 for r in range(3):
  for c in range(3):
   z=x.clone();z[:,r*step:min(H,(r+1)*step),c*step:min(W,(c+1)*step)]=0
   with torch.no_grad():p=torch.softmax(model(z[None].to(device)),1)[0,pred].item()
   drops.append(baseconf-p)
 return np.array(drops),baseconf,int(pred.item())
def domain_stats(name,root,indices=None):
 ds=datasets.ImageFolder(root,transform=eval_tf);allidx=np.arange(len(ds));
 if indices is None:indices=allidx[:min(50,len(allidx))]
 else:indices=indices[:min(50,len(indices))]
 cams=[];drops=[];confs=[];correct=[];cam_center=[];cam_border=[];occ_center=[];occ_border=[]
 for i in indices:
  x,t=ds[int(i)];c,conf,p=cam_one(x);d,bc,_=occlusion_one(x);cams.append(c);drops.append(d);confs.append(conf);correct.append(int(p==t))
  yy,xx=np.mgrid[:224,:224];center=((xx>=56)&(xx<168)&(yy>=56)&(yy<168));border=~center;cam_center.append(float(c[center].sum()/(c.sum()+1e-12)));cam_border.append(float(c[border].sum()/(c.sum()+1e-12)));occ_center.append(float(d.reshape(3,3)[1,1]));occ_border.append(float(np.delete(d,4).mean()))
 arr=np.array(drops)
 return {"n":len(indices),"mean_confidence":round(float(np.mean(confs)),4),"accuracy_if_labels_compatible":round(float(np.mean(correct)),4) if name=="plantvillage" else None,"gradcam_center_mass_mean":round(float(np.mean(cam_center)),4),"gradcam_border_mass_mean":round(float(np.mean(cam_border)),4),"occlusion_max_drop_mean":round(float(arr.max(1).mean()),4),"occlusion_center_drop_mean":round(float(np.mean(occ_center)),4),"occlusion_border_drop_mean":round(float(np.mean(occ_border)),4),"occlusion_center_minus_border":round(float(np.mean(occ_center)-np.mean(occ_border)),4)}
print("PlantVillage explainability")
pv_stats=domain_stats("plantvillage",PV,te[:50]);print(pv_stats)
print("PlantDoc explainability")
pd_stats=domain_stats("plantdoc",PD);print(pd_stats)
print("Cassava explainability")
cv_stats=domain_stats("cassava",CV);print(cv_stats)
report["domains"]={"plantvillage":pv_stats,"plantdoc":pd_stats,"cassava":cv_stats};report["interpretation_boundary"]="Without lesion masks, Grad-CAM/occlusion values are attribution diagnostics only; center/border concentration is not lesion localization or causal evidence. External-domain labels are not mapped to PlantVillage classes."
report["elapsed_sec"]=round(time.time()-t0,2);raw=json.dumps(report,indent=2).encode();report["sha256_self"]=hashlib.sha256(raw).hexdigest()
with open("/kaggle/working/phase18_explainability_report.json","w") as f:f.write(json.dumps(report,indent=2))
print(json.dumps(report,indent=2));h1.remove();h2.remove()
