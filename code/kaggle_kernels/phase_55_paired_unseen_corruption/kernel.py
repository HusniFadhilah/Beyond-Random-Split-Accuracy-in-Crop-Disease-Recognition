import subprocess;subprocess.run(['pip','install','--quiet','--retries','10','--timeout','60','torch==2.3.1','torchvision==0.18.1','pillow'],check=True,timeout=900)
import json,time,copy,random,hashlib
import numpy as np,torch,torch.nn as nn
from PIL import Image
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
T=time.time();dev='cuda' if torch.cuda.is_available() else 'cpu'
if dev!='cuda':raise RuntimeError('CUDA GPU required')
PV='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color';mean=[.485,.456,.406];std=[.229,.224,.225]
base=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()]); evds=datasets.ImageFolder(PV,transform=base);raw=datasets.ImageFolder(PV);y=np.array([s[1] for s in raw.samples]);idx=np.arange(len(raw));tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y);va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy)
def train(aware):
 aug=[transforms.Resize(256),transforms.RandomCrop(224),transforms.RandomHorizontalFlip()]
 if aware: aug += [transforms.RandomApply([transforms.GaussianBlur(9,(3,3))],p=.3),transforms.ToTensor(),transforms.Normalize(mean,std)]
 else: aug += [transforms.ToTensor(),transforms.Normalize(mean,std)]
 if aware: tf=transforms.Compose(aug)
 else: tf=transforms.Compose(aug)
 ds=datasets.ImageFolder(PV,transform=tf);dl=DataLoader(Subset(ds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True);vd=DataLoader(Subset(evds,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True);m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier[1]=nn.Linear(m.classifier[1].in_features,len(raw.classes));m=m.to(dev);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);crit=nn.CrossEntropyLoss(label_smoothing=.1);best=-1;state=None
 for ep in range(6):
  m.train()
  for x,yy in dl:opt.zero_grad();z=m(x.to(dev));loss=crit(z,yy.to(dev));loss.backward();opt.step()
  sch.step();m.eval();ls=[];ys=[]
  with torch.no_grad():
   for x,yy in vd:ls.append(m(x.to(dev)).cpu());ys.append(yy)
  vf=f1_score(torch.cat(ys).numpy(),torch.cat(ls).argmax(1).numpy(),average='macro')
  if vf>best:best=vf;state=copy.deepcopy(m.state_dict())
 m.load_state_dict(state);return m
mu=torch.tensor(mean,device=dev).view(1,3,1,1);sd=torch.tensor(std,device=dev).view(1,3,1,1);dl=DataLoader(Subset(evds,te.tolist()),batch_size=128,shuffle=False,num_workers=2)
def evaluate(m,kind):
 ls=[];ys=[];m.eval()
 with torch.no_grad():
  for x,yy in dl:
   if kind=='jpeg':
    aa=[]
    for im in x:
     q='/tmp/x.jpg';Image.fromarray((im.permute(1,2,0).numpy()*255).astype('uint8')).save(q,quality=25);aa.append(transforms.ToTensor()(Image.open(q).convert('RGB')))
    x=torch.stack(aa)
   elif kind=='motion_blur':x=transforms.functional.gaussian_blur(x,[15,15],[5.,1.])
   elif kind=='occlusion':x=x.clone();x[:,:,70:150,70:150]=0
   elif kind=='color_shift':x=(x*torch.tensor([1.35,.75,.75]).view(1,3,1,1)).clamp(0,1)
   ls.append(m((x.to(dev)-mu)/sd).cpu());ys.append(yy)
 p=torch.cat(ls).argmax(1).numpy();yy=torch.cat(ys).numpy();return {'accuracy':round(float(accuracy_score(yy,p)),6),'balanced_accuracy':round(float(balanced_accuracy_score(yy,p)),6),'macro_f1':round(float(f1_score(yy,p,average='macro',zero_division=0)),6)}
models_out={};
for name,aware in [('basic',False),('corruption_aware',True)]:
 m=train(aware);models_out[name]={k:evaluate(m,k) for k in ['clean','jpeg','motion_blur','occlusion','color_shift']}
report={'phase':'55_paired_unseen_corruption_basic_vs_corruption_aware','split':{'train':len(tr),'validation':len(va),'test':len(te),'seed':42},'training':{'basic':'resize-crop-flip','corruption_aware':'resize-crop-flip plus GaussianBlur p=0.3; this run does not add Gaussian noise'},'unseen_corruptions':['jpeg_quality_25','motion_blur_kernel_15_sigma_5_1','square_occlusion_80px','red_color_shift'],'results':models_out,'elapsed_sec':round(time.time()-T,2),'sha256_self':None};report['sha256_self']=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest();open('/kaggle/working/phase55_paired_unseen_corruption_report.json','w').write(json.dumps(report,indent=2));print(json.dumps(report,indent=2))