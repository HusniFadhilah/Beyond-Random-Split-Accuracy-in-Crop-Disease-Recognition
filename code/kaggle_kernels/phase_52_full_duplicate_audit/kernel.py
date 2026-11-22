import os,json,hashlib,time
from PIL import Image
import numpy as np
T=time.time(); ROOT='/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color'
paths=[]
for r,ds,fs in os.walk(ROOT):
    for f in fs:
        if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')): paths.append(os.path.join(r,f))
paths.sort()
def phash(path):
    im=Image.open(path).convert('L').resize((32,32),Image.Resampling.LANCZOS)
    a=np.asarray(im,dtype=np.float32)
    try:
        from scipy.fftpack import dct
        d=dct(dct(a,axis=0,norm='ortho'),axis=1,norm='ortho')[:8,:8]
    except Exception:
        d=np.fft.fft2(a).real[:8,:8]
    bits=d>d[1:,1:].mean() if d.shape==(8,8) else d>d.mean()
    return ''.join('1' if x else '0' for x in bits.ravel())
md5={}; ph={}; failures=[]
for i,p in enumerate(paths):
    try:
        b=open(p,'rb').read(); h=hashlib.md5(b).hexdigest(); md5.setdefault(h,[]).append(p)
        ph.setdefault(phash(p),[]).append(p)
    except Exception as e: failures.append({'path':p,'error':repr(e)})
def groups(d): return [v for v in d.values() if len(v)>1]
exact=groups(md5); pexact=groups(ph)
# Count close perceptual hashes without O(n^2): bucket by first 16 bits, then compare nearby buckets.
items=list(ph.keys()); near=[]
for i,a in enumerate(items):
    for b in items[i+1:]:
        if a[:16]!=b[:16]: continue
        dist=sum(x!=y for x,y in zip(a,b))
        if dist<=4: near.append({'distance':dist,'paths':ph[a]+ph[b]})
report={'phase':'52_full_plantvillage_duplicate_audit','dataset_root':ROOT,'n_images':len(paths),'n_failures':len(failures),'failures':failures[:20],'md5_duplicate_groups':len(exact),'md5_duplicate_images':sum(len(x) for x in exact),'phash_exact_groups':len(pexact),'phash_exact_images':sum(len(x) for x in pexact),'phash_near_pairs_hamming_le_4':len(near),'scope':'all image files under color/','elapsed_sec':round(time.time()-T,2),'sha256_self':None}
raw=json.dumps(report,sort_keys=True).encode();report['sha256_self']=hashlib.sha256(raw).hexdigest();open('/kaggle/working/phase52_full_plantvillage_duplicate_audit.json','w').write(json.dumps(report,indent=2));print(json.dumps(report,indent=2))