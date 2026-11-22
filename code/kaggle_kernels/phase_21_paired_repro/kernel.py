"""Phase 21: paired reproducibility and uncertainty for model-family comparison."""
import json,time,hashlib,random,subprocess
subprocess.run(["pip","install","--quiet","torch==2.5.0","torchvision==0.20.0","--index-url","https://download.pytorch.org/whl/cu121"],check=False,timeout=300)
import torch,torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset,TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
random.seed(42);np.random.seed(42);torch.manual_seed(42)
t0=time.time();device='cuda' if torch.cuda.is_available() else 'cpu';print('device:',device)
PV='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color'
tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
ds=datasets.ImageFolder(PV,transform=tf);y=np.array([s[1] for s in ds.samples]);idx=np.arange(len(ds))
tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy);ncls=len(ds.classes)
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
def predict(head,X):
 with torch.no_grad():return head(X.to(device)).argmax(1).cpu().numpy()
def metric(y,p):return {'accuracy':round(float(accuracy_score(y,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),6),'macro_f1':round(float(f1_score(y,p,average='macro',zero_division=0)),6)}
report={'phase':'21_paired_reproducibility','captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':{'seed':42,'backbone_mode':'ImageNet pretrained frozen features','head':'linear classifier','head_epochs':10,'bootstrap_replicates':1000,'bootstrap_seed':123,'models':['resnet50','mobilenet_v3_large','efficientnet_b0'],'paired_test_split':True}}
tr_dl,va_dl,te_dl=loader(tr),loader(va),loader(te);preds={};results={}
for name in report['protocol']['models']:
 print('model',name);m=build(name);Xtr,Ytr=extract(m,tr_dl);Xva,Yva=extract(m,va_dl);Xte,Yte=extract(m,te_dl);head=nn.Linear(Xtr.shape[1],ncls).to(device);opt=torch.optim.AdamW(head.parameters(),lr=1e-2,weight_decay=1e-4);crit=nn.CrossEntropyLoss();dl=DataLoader(TensorDataset(Xtr,Ytr),batch_size=256,shuffle=True)
 for _ in range(10):
  head.train()
  for xb,yb in dl:opt.zero_grad();crit(head(xb.to(device)),yb.to(device)).backward();opt.step()
 p=predict(head,Xte);yy=Yte.numpy();preds[name]=p.tolist();results[name]={'val':metric(Yva.numpy(),predict(head,Xva)),'test':metric(yy,p),'test_correct_count':int((yy==p).sum()),'test_n':int(len(yy)),'feature_dim':int(Xtr.shape[1])}
 del m,Xtr,Xva,Xte,head
# paired bootstrap over the identical test examples
rng=np.random.default_rng(123);n=len(te);boot={};pairs=[]
for i,a in enumerate(report['protocol']['models']):
 for b in report['protocol']['models'][i+1:]:
  ca=np.array(preds[a])==y[te];cb=np.array(preds[b])==y[te];d=ca.astype(float)-cb.astype(float);means=[]
  for _ in range(1000):means.append(float(d[rng.integers(0,n,n)].mean()))
  q=np.quantile(means,[.025,.5,.975]);pairs.append({'a':a,'b':b,'accuracy_delta_a_minus_b':round(float(d.mean()),6),'bootstrap_ci95':{'low':round(float(q[0]),6),'median':round(float(q[1]),6),'high':round(float(q[2]),6)},'discordant_a_only':int(((ca==1)&(cb==0)).sum()),'discordant_b_only':int(((ca==0)&(cb==1)).sum())})
report['split_sizes']={'train':int(len(tr)),'val':int(len(va)),'test':int(len(te)),'n_classes':ncls};report['results']=results;report['paired_bootstrap']=pairs;report['prediction_alignment']={'labels_sha256':hashlib.sha256(y[te].tobytes()).hexdigest(),'test_indices_sha256':hashlib.sha256(te.tobytes()).hexdigest()};report['elapsed_sec']=round(time.time()-t0,2)
raw=json.dumps(report,indent=2).encode();report['sha256_self']=hashlib.sha256(raw).hexdigest()
with open('/kaggle/working/phase21_paired_reproducibility_report.json','w') as f:f.write(json.dumps(report,indent=2))
with open('/kaggle/working/phase21_per_sample_predictions.json','w') as f:f.write(json.dumps({'test_indices':te.tolist(),'labels':y[te].tolist(),'predictions':preds},separators=(',',':')))
print(json.dumps(report,indent=2))
