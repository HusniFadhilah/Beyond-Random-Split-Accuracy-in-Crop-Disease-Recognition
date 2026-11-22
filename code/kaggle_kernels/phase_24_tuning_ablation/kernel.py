"""Phase 24: validation-only tuning and component ablation."""
import subprocess
subprocess.run(["pip","install","--quiet","torch==2.5.0","torchvision==0.20.0","--index-url","https://download.pytorch.org/whl/cu121"],check=False,timeout=300)
import json,time,hashlib,random
import numpy as np, torch, torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score

T0=time.time(); SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device='cuda' if torch.cuda.is_available() else 'cpu'; print('device:',device)
PV='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color'
tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
ds=datasets.ImageFolder(PV,transform=tf); y=np.array([s[1] for s in ds.samples]); idx=np.arange(len(ds))
tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y)
va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy); ncls=len(ds.classes)
def dl(a,bs=128): return DataLoader(Subset(ds,a.tolist()),batch_size=bs,shuffle=False,num_workers=2,pin_memory=True)
def build(name):
    if name=='resnet50': m=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc=nn.Identity()
    elif name=='mobilenet_v3_large': m=models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2); m.classifier=nn.Identity()
    else: m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1); m.classifier=nn.Identity()
    return m.to(device).eval()
def extract(m,a):
    z=[]; yy=[]
    with torch.no_grad():
        for x,yb in dl(a): z.append(m(x.to(device)).flatten(1).cpu()); yy.append(yb)
    return torch.cat(z),torch.cat(yy)
def met(y,p): return {'accuracy':round(float(accuracy_score(y,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(y,p)),6),'macro_f1':round(float(f1_score(y,p,average='macro',zero_division=0)),6)}
def train_eval(Xtr,Ytr,Xva,Yva,cfg,return_test=None):
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    head=nn.Sequential(nn.Dropout(cfg['dropout']),nn.Linear(Xtr.shape[1],ncls)).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['wd']) if cfg['optimizer']=='AdamW' else torch.optim.SGD(head.parameters(),lr=cfg['lr'],momentum=.9,weight_decay=cfg['wd'])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs']) if cfg['scheduler']=='cosine' else None
    crit=nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    train_dl=DataLoader(TensorDataset(Xtr,Ytr),batch_size=256,shuffle=True)
    best=(-1,None); patience=0
    for ep in range(cfg['epochs']):
        head.train()
        for xb,yb in train_dl:
            opt.zero_grad(); loss=crit(head(xb.to(device)),yb.to(device)); loss.backward(); opt.step()
        if sch: sch.step()
        head.eval()
        with torch.no_grad(): pv=head(Xva.to(device)).argmax(1).cpu().numpy()
        score=f1_score(Yva.numpy(),pv,average='macro',zero_division=0)
        if score>best[0]+1e-5: best=(score,ep+1); patience=0
        else: patience+=1
        if patience>=cfg['patience']: break
    out={'config':cfg,'best_val_macro_f1':round(float(best[0]),6),'best_epoch':best[1]}
    if return_test is not None:
        Xt,Yt=return_test; head.eval()
        with torch.no_grad(): pt=head(Xt.to(device)).argmax(1).cpu().numpy()
        out['test']=met(Yt.numpy(),pt)
    return out

base={'optimizer':'AdamW','lr':.01,'wd':.0001,'dropout':0.0,'scheduler':'none','label_smoothing':0.0,'epochs':20,'patience':5,'seed':42}
# Deliberately compact, pre-registered screening grid; every choice uses validation only.
grid=[]
for opt in ['AdamW','SGD']:
  for lr in [.001,.003,.01]:
    grid.append({**base,'optimizer':opt,'lr':lr})
for wd in [0.0,.001]: grid.append({**base,'wd':wd})
for drop in [.2,.5]: grid.append({**base,'dropout':drop})
for sch in ['cosine']: grid.append({**base,'scheduler':sch})
for ls in [.05,.1]: grid.append({**base,'label_smoothing':ls})
report={'phase':'24_tuning_optimization_ablation','captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol':{'selection_metric':'validation macro_f1','grid_size':len(grid),'test_policy':'test evaluated only for locked best configuration','seed':42,'models':['resnet50','mobilenet_v3_large','efficientnet_b0']},'split_sizes':{'train':len(tr),'val':len(va),'test':len(te),'n_classes':ncls},'results':{}}
for name in report['protocol']['models']:
  print('MODEL',name); m=build(name); Xtr,Ytr=extract(m,tr); Xva,Yva=extract(m,va); Xte,Yte=extract(m,te)
  runs=[]
  for i,cfg in enumerate(grid):
    r=train_eval(Xtr,Ytr,Xva,Yva,cfg); runs.append(r); print(name,'run',i+1,'/',len(grid),r['best_val_macro_f1'])
  runs.sort(key=lambda r:r['best_val_macro_f1'],reverse=True); best=runs[0]
  final=train_eval(Xtr,Ytr,Xva,Yva,best['config'],(Xte,Yte))
  # Fixed pre-tuning baseline and selected optimum are both retained for ablation comparison.
  baseline=train_eval(Xtr,Ytr,Xva,Yva,base,(Xte,Yte))
  report['results'][name]={'feature_dim':int(Xtr.shape[1]),'tuning_runs':runs,'locked_best':final,'fixed_baseline':baseline,'ablation_delta_macro_f1':round(final['test']['macro_f1']-baseline['test']['macro_f1'],6)}
raw=json.dumps(report,indent=2).encode(); report['sha256_self']=hashlib.sha256(raw).hexdigest();
with open('/kaggle/working/phase24_tuning_ablation_report.json','w') as f:f.write(json.dumps(report,indent=2))
print(json.dumps({'phase':report['phase'],'sha256_self':report['sha256_self'],'elapsed_sec':round(time.time()-T0,2),'results':{k:{'best_val_macro_f1':v['locked_best']['best_val_macro_f1'],'test':v['locked_best']['test'],'baseline_test':v['fixed_baseline']['test']} for k,v in report['results'].items()}},indent=2))
