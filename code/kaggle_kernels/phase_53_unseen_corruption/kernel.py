import subprocess; subprocess.run(['pip','install','--quiet','--retries','10','--timeout','60','torch==2.3.1','torchvision==0.18.1','pillow'],check=True,timeout=900)
import json,time,copy,random,hashlib
import numpy as np, torch, torch.nn as nn
from PIL import Image,ImageFilter
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
T=time.time(); device='cuda' if torch.cuda.is_available() else 'cpu'
if device!='cuda': raise RuntimeError('CUDA GPU required')
PV='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color'; mean=[.485,.456,.406]; std=[.229,.224,.225]
trtf=transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)])
evtf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean,std)])
random.seed(42);np.random.seed(42);torch.manual_seed(42);raw=datasets.ImageFolder(PV); y=np.array([s[1] for s in raw.samples]); idx=np.arange(len(raw)); tr,tmp,_,tmpy=train_test_split(idx,y,test_size=.2,random_state=42,stratify=y); va,te,_,_=train_test_split(tmp,tmpy,test_size=.5,random_state=42,stratify=tmpy)
trds=datasets.ImageFolder(PV,transform=trtf); evds=datasets.ImageFolder(PV,transform=evtf); trl=DataLoader(Subset(trds,tr.tolist()),batch_size=128,shuffle=True,num_workers=2,pin_memory=True); val=DataLoader(Subset(evds,va.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True); test=DataLoader(Subset(evds,te.tolist()),batch_size=128,shuffle=False,num_workers=2,pin_memory=True)
def collect(m,dl):
 m.eval();ls=[];ys=[]
 with torch.no_grad():
  for x,yb in dl: ls.append(m(x.to(device)).cpu());ys.append(yb)
 return torch.cat(ls),torch.cat(ys)
m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1);m.classifier[1]=nn.Linear(m.classifier[1].in_features,len(raw.classes));m=m.to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6);crit=nn.CrossEntropyLoss(label_smoothing=.1);best=-1;state=None
for ep in range(6):
 m.train()
 for x,yb in trl: opt.zero_grad(); loss=crit(m(x.to(device)),yb.to(device));loss.backward();opt.step()
 sch.step(); vl,vy=collect(m,val); vf=f1_score(vy.numpy(),vl.argmax(1).numpy(),average='macro')
 if vf>best:best=vf;state=copy.deepcopy(m.state_dict())
m.load_state_dict(state); tl,ty=collect(m,test); p=torch.softmax(tl,1); mu=torch.tensor(mean,device=device).view(1,3,1,1); sd=torch.tensor(std,device=device).view(1,3,1,1)
# Apply perturbations after unnormalized center-crop tensor and before normalization.
base=datasets.ImageFolder(PV,transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])); dl=DataLoader(Subset(base,te.tolist()),batch_size=128,shuffle=False,num_workers=2)
def run(kind):
 ls=[];ys=[]
 with torch.no_grad():
  for x,yb in dl:
   if kind=='jpeg':
    arr=[]
    for im in x: 
     q=Image.fromarray((im.permute(1,2,0).numpy()*255).astype('uint8')).save('/tmp/a.jpg',quality=25); arr.append(transforms.ToTensor()(Image.open('/tmp/a.jpg').convert('RGB')))
    x=torch.stack(arr)
   elif kind=='motion_blur': x=transforms.functional.gaussian_blur(x,[15,15],[5.0,1.0])
   elif kind=='occlusion': x=x.clone(); x[:,:,70:150,70:150]=0
   elif kind=='color_shift': x=(x*torch.tensor([1.35,.75,.75]).view(1,3,1,1)).clamp(0,1)
   z=m((x.to(device)-mu)/sd).cpu();ls.append(z);ys.append(yb)
 l=torch.cat(ls);yy=torch.cat(ys).numpy();pp=l.argmax(1).numpy();return {'accuracy':round(float(accuracy_score(yy,pp)),6),'balanced_accuracy':round(float(balanced_accuracy_score(yy,pp)),6),'macro_f1':round(float(f1_score(yy,pp,average='macro',zero_division=0)),6)}
report={'phase':'53_unseen_corruption_evaluation','model':'EfficientNet-B0 full fine-tuning','split':{'train':len(tr),'validation':len(va),'test':len(te),'seed':42},'checkpoint':'best validation macro-F1','training_corruptions':'basic resize-crop-flip only','unseen_corruptions':{'jpeg_quality_25':run('jpeg'),'motion_blur_kernel_15_sigma_5_1':run('motion_blur'),'square_occlusion_80px':run('occlusion'),'red_color_shift':run('color_shift')},'clean_test':{'accuracy':round(float(accuracy_score(ty.numpy(),p.argmax(1).numpy())),6),'balanced_accuracy':round(float(balanced_accuracy_score(ty.numpy(),p.argmax(1).numpy())),6),'macro_f1':round(float(f1_score(ty.numpy(),p.argmax(1).numpy(),average='macro')),6)},'elapsed_sec':round(time.time()-T,2),'sha256_self':None}; report['sha256_self']=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest();open('/kaggle/working/phase53_unseen_corruption_report.json','w').write(json.dumps(report,indent=2));print(json.dumps(report,indent=2))