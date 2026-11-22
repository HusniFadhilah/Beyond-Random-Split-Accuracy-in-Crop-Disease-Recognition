"""Phase 27: final EfficientNet-B0 external OOD evaluation."""
import subprocess
subprocess.run(["pip","install","--quiet","--retries","10","--timeout","60","torch==2.3.1","torchvision==0.18.1"],check=True,timeout=900)
import json,time,hashlib,random,copy
import numpy as np,torch,torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score
T0=time.time();device='cuda' if torch.cuda.is_available() else 'cpu';print('device:',device)
if device!='cuda':raise RuntimeError('CUDA GPU required')
PV='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color';PD='/kaggle/input/datasets/nirmalsankalana/plantdoc-dataset/test';CV='/kaggle/input/datasets/nirmalsankalana/cassava-leaf-disease-classification/data';mean=[.485,.456,.406];std=[.229,.224,.225]
train_tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.75,1.0)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.2,hue=0.0),transforms.ToTensor(),transforms.Normalize(mean,std)])
eval_tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean,std)])
random.seed(42);np.random.seed(42);torch.manual_seed(42);raw=datasets.ImageFolder(PV);y=np.array([s[1] for s in raw.samples]);idx=np.arange(len(raw));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy);ncls=len(raw.classes)
trds=datasets.ImageFolder(PV,transform=train_tf);evads=datasets.ImageFolder(PV,transform=eval_tf);trl=DataLoader(Subset(trds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);val=DataLoader(Subset(evads,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);pvtest=DataLoader(Subset(evads,te.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True)
def probs(m,loader):
 m.eval();ps=[];ys=[]
 with torch.no_grad():
  for x,yb in loader:ps.append(torch.softmax(m(x.to(device)),1).cpu().numpy());ys.append(yb.numpy())
 return np.concatenate(ps),np.concatenate(ys)
def met(y,p):return {'accuracy':round(float(accuracy_score(y,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),6),'macro_f1':round(float(f1_score(y,p,average='macro',zero_division=0)),6)}
m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier[1]=nn.Linear(m.classifier[1].in_features,ncls);m=m.to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);crit=nn.CrossEntropyLoss();best=-1;state=None;be=0
for ep in range(8):
 m.train()
 for x,yb in trl:opt.zero_grad();loss=crit(m(x.to(device)),yb.to(device));loss.backward();opt.step()
 p,yy=probs(m,val);vm=met(yy,p.argmax(1));print('epoch',ep+1,vm)
 if vm['macro_f1']>best:best=vm['macro_f1'];be=ep+1;state=copy.deepcopy(m.state_dict())
m.load_state_dict(state);pvprob,pvy=probs(m,pvtest);pvmetrics=met(pvy,pvprob.argmax(1));
def external(root):
 ds=datasets.ImageFolder(root,transform=eval_tf);dl=DataLoader(ds,batch_size=128,shuffle=False,num_workers=2,pin_memory=True);p,_=probs(m,dl);mx=p.max(1);ent=-(p*np.log(p+1e-10)).sum(1);return {'n':len(ds),'n_classes':len(ds.classes),'max_conf_mean':round(float(mx.mean()),6),'max_conf_sd':round(float(mx.std()),6),'entropy_mean':round(float(ent.mean()),6),'entropy_sd':round(float(ent.std()),6),'top_predicted_pv_classes':[{'class':raw.classes[i],'n':int((p.argmax(1)==i).sum())} for i in np.argsort(np.bincount(p.argmax(1),minlength=ncls))[::-1][:5]],'max_conf_values':mx}
pd=external(PD);cv=external(CV)
def auroc_ood(mx):
 ytrue=np.r_[np.zeros(len(pvprob)),np.ones(len(mx))];score=np.r_[-pvprob.max(1),-mx];return round(float(roc_auc_score(ytrue,score)),6)
pd.pop('max_conf_values');cv.pop('max_conf_values');report={'phase':'27_external_ood_final_model','captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':{'model':'efficientnet_b0','mode':'full_finetune','augmentation':'strong_without_hue','epochs':8,'seed':42,'checkpoint':'best validation macro_f1','label_mapper':'none; external labels are not compared to PV classes'},'split_sizes':{'pv_train':len(tr),'pv_val':len(va),'pv_test':len(te)},'pv_test':pvmetrics,'plantdoc':pd,'cassava':cv,'ood_auroc':{'plantdoc_vs_pv':auroc_ood(external(PD)['max_conf_values']) if False else None,'cassava_vs_pv':auroc_ood(external(CV)['max_conf_values']) if False else None},'best_val_macro_f1':best,'best_epoch':be}
# Recompute AUROC without retaining raw external arrays in public report.
for key,path in [('plantdoc',PD),('cassava',CV)]:
 ds=datasets.ImageFolder(path,transform=eval_tf);dl=DataLoader(ds,batch_size=128,shuffle=False,num_workers=2,pin_memory=True);p,_=probs(m,dl);report['ood_auroc'][key+'_vs_pv']=auroc_ood(p.max(1))
report['elapsed_sec']=round(time.time()-T0,2);rawj=json.dumps(report,indent=2).encode();report['sha256_self']=hashlib.sha256(rawj).hexdigest();
with open('/kaggle/working/phase27_external_ood_report.json','w') as f:f.write(json.dumps(report,indent=2))
print(json.dumps({'phase':report['phase'],'best_val_macro_f1':best,'pv_test':pvmetrics,'plantdoc':pd,'cassava':cv,'ood_auroc':report['ood_auroc'],'elapsed_sec':report['elapsed_sec']},indent=2))
