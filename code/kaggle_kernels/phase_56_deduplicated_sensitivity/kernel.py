import subprocess;subprocess.run(['pip','install','--quiet','--retries','10','--timeout','60','torch==2.3.1','torchvision==0.18.1','pillow'],check=True,timeout=900)
import os,json,hashlib,time,random,copy
import numpy as np,torch,torch.nn as nn
from PIL import Image
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
T=time.time();seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;dev='cuda' if torch.cuda.is_available() else 'cpu'
if dev!='cuda':raise RuntimeError('CUDA GPU required')
ROOT='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color';raw=datasets.ImageFolder(ROOT);paths=[p for p,_ in raw.samples];y=np.array([v for _,v in raw.samples]);idx=np.arange(len(paths));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy);split={i:('train' if i in set(tr) else 'validation' if i in set(va) else 'test') for i in idx}
md={};
for i,p in enumerate(paths):md.setdefault(hashlib.md5(open(p,'rb').read()).hexdigest(),[]).append(i)
# exact MD5 groups are sufficient for primary sensitivity; pHash exact is reported from the audit but not used to double-remove same groups
cross=[g for g in md.values() if len(set(split[i] for i in g))>1];affected=set(i for g in cross for i in g);clean={s:[i for i in (tr if s=='train' else va if s=='validation' else te) if i not in affected] for s in ['train','validation','test']}
mean=[.485,.456,.406];std=[.229,.224,.225];tf=transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)]);ev=tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean,std)]);ds=datasets.ImageFolder(ROOT,transform=tf);eds=datasets.ImageFolder(ROOT,transform=ev)
def train_model():
 dl=DataLoader(Subset(ds,clean['train']),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);vd=DataLoader(Subset(eds,clean['validation']),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier[1]=nn.Linear(m.classifier[1].in_features,len(raw.classes));m=m.to(dev);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);crit=nn.CrossEntropyLoss(label_smoothing=.1);best=-1;state=None
 for ep in range(6):
  m.train()
  for x,z in dl:opt.zero_grad();loss=crit(m(x.to(dev)),z.to(dev));loss.backward();opt.step()
  sch.step();m.eval();pp=[];yy=[]
  with torch.no_grad():
   for x,z in vd:pp.append(m(x.to(dev)).cpu());yy.append(z)
  f=f1_score(torch.cat(yy).numpy(),torch.cat(pp).argmax(1).numpy(),average='macro')
  if f>best:best=f;state=copy.deepcopy(m.state_dict())
 m.load_state_dict(state);return m,best
def evaluate(m,ids):
 dl=DataLoader(Subset(eds,ids),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);p=[];z=[];m.eval()
 with torch.no_grad():
  for x,y0 in dl:p.append(m(x.to(dev)).cpu());z.append(y0)
 yy=torch.cat(z).numpy();pp=torch.cat(p).argmax(1).numpy();return {'n_images':len(ids),'accuracy':round(float(accuracy_score(yy,pp)),6),'balanced_accuracy':round(float(balanced_accuracy_score(yy,pp)),6),'macro_f1':round(float(f1_score(yy,pp,average='macro')),6)}
m,best=train_model();report={'phase':'56_deduplicated_sensitivity','seed':seed,'original_split':{'train':len(tr),'validation':len(va),'test':len(te)},'md5_cross_split_groups':len(cross),'affected_images_total':len(affected),'clean_split':{s:len(v) for s,v in clean.items()},'excluded_by_split':{s:len(([i for i in (tr if s=='train' else va if s=='validation' else te) if i in affected])) for s in ['train','validation','test']},'best_clean_validation_macro_f1':round(float(best),6),'deduplicated_test_metrics':evaluate(m,clean['test']),'protocol':'same seed-42 stratified split; exact MD5 cross-split groups removed; ImageNet-normalized EfficientNet-B0 basic augmentation; six epochs; validation macro-F1 checkpoint','elapsed_sec':round(time.time()-T,2),'sha256_self':None};report['sha256_self']=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest();open('/kaggle/working/phase56_deduplicated_sensitivity_report.json','w').write(json.dumps(report,indent=2));print(json.dumps(report,indent=2))