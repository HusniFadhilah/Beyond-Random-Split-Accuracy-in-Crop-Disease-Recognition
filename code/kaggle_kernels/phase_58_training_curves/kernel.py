"""Phase 38: calibrated selective prediction for the locked EfficientNet-B0 pipeline."""
import subprocess
subprocess.run(["pip","install","--quiet","--retries","10","--timeout","60","torch==2.3.1","torchvision==0.18.1"],check=True,timeout=900)
import json,time,hashlib,random,copy
import numpy as np,torch,torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score
T0=time.time();device='cuda' if torch.cuda.is_available() else 'cpu';print('device:',device)
if device!='cuda':raise RuntimeError('CUDA GPU required; refusing CPU fallback')
PV='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color';mean=[.485,.456,.406];std=[.229,.224,.225]
trtf=transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)]);evtf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean,std)])
random.seed(42);np.random.seed(42);torch.manual_seed(42);raw=datasets.ImageFolder(PV);y=np.array([s[1] for s in raw.samples]);idx=np.arange(len(raw));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy);ncls=len(raw.classes)
trds=datasets.ImageFolder(PV,transform=trtf);evads=datasets.ImageFolder(PV,transform=evtf);trl=DataLoader(Subset(trds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);val=DataLoader(Subset(evads,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);test=DataLoader(Subset(evads,te.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True)
def logits(m,dl):
 m.eval();ls=[];ys=[]
 with torch.no_grad():
  for x,yb in dl:ls.append(m(x.to(device)).cpu());ys.append(yb)
 return torch.cat(ls),torch.cat(ys)
def metric(y,p):return {'accuracy':round(float(accuracy_score(y,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),6),'macro_f1':round(float(f1_score(y,p,average='macro',zero_division=0)),6)}
def ece(p,y,bins=15):
 conf=p.max(1).values.numpy();pred=p.argmax(1).numpy();yy=y.numpy();edges=np.linspace(0,1,bins+1);out=0
 for a,b in zip(edges[:-1],edges[1:]):
  q=(conf>a)&(conf<=b)
  if q.any():out+=q.mean()*abs((pred[q]==yy[q]).mean()-conf[q].mean())
 return round(float(out),6)
m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier[1]=nn.Linear(m.classifier[1].in_features,ncls);m=m.to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);crit=nn.CrossEntropyLoss(label_smoothing=.1);nll=nn.CrossEntropyLoss();history=[];best=-1;state=None
for ep in range(6):
 m.train();tloss=[];ttrue=[];tpred=[]
 for x,yb in trl:
  opt.zero_grad();out=m(x.to(device));loss=crit(out,yb.to(device));loss.backward();opt.step();tloss.append(float(loss.detach().cpu()));ttrue.extend(yb.numpy());tpred.extend(out.detach().cpu().argmax(1).numpy())
 sch.step();vl,vy=logits(m,val);vf=f1_score(vy.numpy(),vl.argmax(1).numpy(),average='macro');vprob=torch.softmax(vl,1);vll=float(nll(vl,vy).detach().cpu());history.append({'epoch':ep+1,'train_loss':round(float(np.mean(tloss)),6),'train_accuracy':round(float(accuracy_score(ttrue,tpred)),6),'validation_loss':round(vll,6),'validation_accuracy':round(float(accuracy_score(vy.numpy(),vl.argmax(1).numpy())),6),'validation_macro_f1':round(float(vf),6)});print('epoch',ep+1,history[-1])
 if vf>best:best=vf;state=copy.deepcopy(m.state_dict())
m.load_state_dict(state);vl,vy=logits(m,val);tl,ty=logits(m,test);temperature=torch.nn.Parameter(torch.ones(1));to=torch.optim.LBFGS([temperature],lr=.1,max_iter=50,line_search_fn='strong_wolfe');nll=nn.CrossEntropyLoss()
def closure():
 to.zero_grad();loss=nll(vl/temperature.clamp_min(.05),vy);loss.backward();return loss
to.step(closure);temp=float(temperature.detach().clamp_min(.05));vp=torch.softmax(vl/temp,1);tp=torch.softmax(tl/temp,1);thresholds=[.50,.60,.70,.80,.90,.95,.99];selective=[]
for q in thresholds:
 keep=tp.max(1).values.numpy()>=q; pred=tp.argmax(1).numpy(); yy=ty.numpy(); selective.append({'threshold':q,'coverage':round(float(keep.mean()),6),'n_accepted':int(keep.sum()),'accuracy_on_accepted':round(float((pred[keep]==yy[keep]).mean()),6) if keep.any() else None,'macro_f1_on_accepted':round(float(f1_score(yy[keep],pred[keep],average='macro',zero_division=0)),6) if keep.any() else None})
def ood_eval(path):
 ds=datasets.ImageFolder(path,transform=evtf);dl=DataLoader(ds,batch_size=128,shuffle=False,num_workers=2,pin_memory=True);el,ey=logits(m,dl);pr=torch.softmax(el/temp,1);return {'n':len(ds),'n_classes':len(ds.classes),'max_conf_mean':round(float(pr.max(1).values.mean()),6),'max_conf_sd':round(float(pr.max(1).values.std()),6),'entropy_mean':round(float((-(pr*pr.clamp_min(1e-8).log()).sum(1)).mean()),6),'entropy_sd':round(float((-(pr*pr.clamp_min(1e-8).log()).sum(1)).std()),6)},pr.max(1).values.numpy()
pv_conf=tp.max(1).values.numpy();
plantdoc,plantdoc_conf=ood_eval('/kaggle/input/datasets/nirmalsankalana/plantdoc-dataset/test');cassava,cassava_conf=ood_eval('/kaggle/input/datasets/nirmalsankalana/cassava-leaf-disease-classification/data')
def auc(ext):return round(float(roc_auc_score(np.r_[np.zeros(len(pv_conf)),np.ones(len(ext))],np.r_[-pv_conf,-ext])),6)
report={'phase':'39_external_ood_efficientnet_b0_basic','captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':{'model':'efficientnet_b0','mode':'full_finetune','augmentation':'basic_resize_crop_flip','optimizer':'adamw','loss':'cross_entropy','label_smoothing':0.1,'scheduler':'cosine','learning_rate':3e-4,'weight_decay':1e-4,'resolution':224,'selection':'validation macro_f1','temperature_fit':'validation NLL only','ood_score':'negative temperature-scaled max confidence; external=positive class','test_policy':'test evaluated after checkpoint and temperature are locked'},'split_sizes':{'train':len(tr),'val':len(va),'test':len(te)},'best_val_macro_f1':round(float(best),6),'temperature':round(temp,6),'validation_ece_after':ece(vp,vy),'test_ece_after':ece(tp,ty),'test_metrics':metric(ty.numpy(),tp.argmax(1).numpy()),'plantdoc':plantdoc,'cassava':cassava,'ood_auroc':{'plantdoc_vs_pv':auc(plantdoc_conf),'cassava_vs_pv':auc(cassava_conf)}};report['training_history']=history;report['elapsed_sec']=round(time.time()-T0,2);rawj=json.dumps(report,indent=2).encode();report['sha256_self']=hashlib.sha256(rawj).hexdigest();open('/kaggle/working/phase39_external_ood_report.json','w').write(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
