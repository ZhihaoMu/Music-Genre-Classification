# 🎵 Music Genre Classifier — Demo App

Streamlit app that classifies music clips into 16 FMA genres using a
ConvNeXt-Tiny trained on mel-spectrograms. Three pages: **Classify**
(with Grad-CAM "what the model heard" overlay), **Find similar**
(embedding nearest-neighbors), and **How it works** (confusion matrix +
F1).

The app is **pure PyTorch + timm — no fastai at inference** — so there
is no pickle/version coupling between Kaggle and your laptop or the
Hugging Face Space. Only `genre_model_plain.pt` is required; everything
else degrades gracefully with an "add me" note.

---

## Step 1a — Export the plain model (REQUIRED)

Paste this cell at the end of your Kaggle notebook and run it while the
trained `learn` is in memory. (Fresh session? First recover the learner:
attach the saved notebook version's output and run
`from fastai.vision.all import *; learn = load_learner(<path to the .pkl>)`
— unpickling works fine *on Kaggle* because it's the same environment.)

The cell rebuilds the model in plain PyTorch, **numerically verifies**
it gives identical outputs, and only then saves. If the assert fails,
send the printed head layout to Claude.

```python
# ============ PLAIN EXPORT — pure-torch model file ============
ARCH = 'convnext_tiny.fb_in22k'      # the timm arch you trained

import torch, torch.nn as nn, torch.nn.functional as F, timm

model = learn.model.eval().float().cpu()
body_fastai, head_fastai = model[0], model[1]
body_raw = getattr(body_fastai, 'model', body_fastai)   # unwrap TimmBody

# --- describe the head as a plain spec ---
spec = []
for m in head_fastai.children():
    n = type(m).__name__
    if n == 'AdaptiveConcatPool2d':      spec.append(['concat_pool'])
    elif n == 'Flatten':                 spec.append(['flatten'])
    elif isinstance(m, nn.BatchNorm1d):  spec.append(['bn', m.num_features])
    elif isinstance(m, nn.Dropout):      spec.append(['dropout', m.p])
    elif isinstance(m, nn.Linear):       spec.append(['linear', m.in_features,
                                                      m.out_features,
                                                      m.bias is not None])
    elif isinstance(m, nn.ReLU):         spec.append(['relu'])
    else:
        print(head_fastai); raise ValueError(f'unexpected head layer: {n}')

# --- rebuild in plain torch and VERIFY ---
class ConcatPool(nn.Module):
    def forward(self, x):
        return torch.cat([F.adaptive_max_pool2d(x, 1),
                          F.adaptive_avg_pool2d(x, 1)], dim=1)

def build_head(spec):
    L = []
    for s in spec:
        k = s[0]
        if k == 'concat_pool': L.append(ConcatPool())
        elif k == 'flatten':   L.append(nn.Flatten(1))
        elif k == 'bn':        L.append(nn.BatchNorm1d(s[1]))
        elif k == 'dropout':   L.append(nn.Dropout(s[1]))
        elif k == 'linear':    L.append(nn.Linear(s[1], s[2], bias=s[3]))
        elif k == 'relu':      L.append(nn.ReLU(inplace=True))
    return nn.Sequential(*L)

class FeatureBody(nn.Module):
    """fastai's TimmBody calls forward_features(), NOT the full forward()
    — the full forward would also apply timm's own head norm/pool."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        return self.m.forward_features(x)

body2 = timm.create_model(ARCH, pretrained=False, num_classes=0)
body2.load_state_dict(body_raw.state_dict())
head2 = build_head(spec); head2.load_state_dict(head_fastai.state_dict())
plain = nn.Sequential(FeatureBody(body2), head2).eval()

x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    diff = (plain(x) - model(x)).abs().max().item()
print('max output diff:', diff)
assert diff < 1e-4, 'rebuild mismatch — send Claude the printed head above'

torch.save({'arch': ARCH, 'vocab': list(learn.dls.vocab), 'head_spec': spec,
            'body_sd': body2.state_dict(), 'head_sd': head2.state_dict()},
           '/kaggle/working/genre_model_plain.pt')
print('SAVED genre_model_plain.pt | genres:', list(learn.dls.vocab))
# ===============================================================
```

Download `genre_model_plain.pt` from the Output sidebar / Output tab.

## Step 1b — Export the presentation extras (optional)

Run after Step 1a (uses `plain`, `learn`, and `SPEC_DIR` from earlier
cells). Produces the confusion matrix, per-genre F1, embeddings for
"find similar", and sample clips, zipped as `app_artifacts.zip`.

```python
# ============ ARTIFACTS EXPORT ============
import os, glob, shutil, random, zipfile
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report
from fastai.vision.all import ClassificationInterpretation, get_image_files

OUT = '/kaggle/working/app_artifacts'; os.makedirs(OUT, exist_ok=True)

# 1. Confusion matrix + per-class F1 (validation set)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10, 10))
plt.savefig(f'{OUT}/confusion_matrix.png', bbox_inches='tight'); plt.close('all')

preds, targs = learn.get_preds()
vocab = list(learn.dls.vocab)
rep = classification_report(targs.numpy(), preds.argmax(1).numpy(),
                            target_names=vocab, output_dict=True)
pd.DataFrame([{'genre': g, 'precision': rep[g]['precision'],
               'recall': rep[g]['recall'], 'f1': rep[g]['f1-score'],
               'support': int(rep[g]['support'])} for g in vocab]
             ).round(3).to_csv(f'{OUT}/per_class_f1.csv', index=False)

# 2. Embeddings with the PLAIN model (must match app.py's embed_clip)
#    Names use artist/title from tracks.csv (`tracks` from the labels cell).
import torch
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

meta_a, meta_t = tracks[('artist', 'name')], tracks[('track', 'title')]
def track_name(tid, genre):
    a, t = meta_a.get(tid), meta_t.get(tid)
    a = str(a) if pd.notna(a) else 'Unknown artist'
    t = str(t) if pd.notna(t) else f'track {tid}'
    return f'{a} — {t}  ({genre})'
def prep(png):
    img = Image.open(png).convert('RGB').resize((224, 224))
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.
    return (t - IMAGENET_MEAN) / IMAGENET_STD

plain_gpu = plain.cuda().eval()
val_pngs = get_image_files(SPEC_DIR/'validation')
embs, names = [], []
with torch.no_grad():
    for i in range(0, len(val_pngs), 64):
        batch = torch.stack([prep(p) for p in val_pngs[i:i+64]]).cuda()
        f = plain_gpu[0](batch)
        v = ConcatPool()(f).flatten(1).cpu().numpy()
        embs.append(v)
        names += [track_name(int(p.stem), p.parent.name)
                  for p in val_pngs[i:i+64]]
embs = np.concatenate(embs).astype(np.float32)
np.savez_compressed(f'{OUT}/embeddings.npz', emb=embs,
                    names=np.array(names, dtype=object))
plain = plain.cpu()

# 3. Sample clips: 2 mp3s per genre, named "Artist - Title.mp3"
#    (uses df + id_to_path from cells 0/2)
import re
def safe(s, n=60):
    return re.sub(r'[^\w\- .,()]', '', str(s))[:n].strip() or 'unknown'

for g in df['genre'].unique():
    ids = df[df['genre'] == g]['track_id'].sample(2, random_state=0)
    d = f"{OUT}/samples/{str(g).replace('/', '-')}"; os.makedirs(d, exist_ok=True)
    for tid in ids:
        tid = int(tid)
        dst = f"{d}/{safe(f'{meta_a.get(tid)} - {meta_t.get(tid)}')}.mp3"
        if os.path.exists(dst):
            dst = dst[:-4] + f' [{tid}].mp3'
        shutil.copy(id_to_path[tid], dst)

# 4. Zip
with zipfile.ZipFile('/kaggle/working/app_artifacts.zip', 'w') as z:
    for root, _, files in os.walk(OUT):
        for f in files:
            fp = os.path.join(root, f)
            z.write(fp, os.path.relpath(fp, OUT))
print('DONE -> download /kaggle/working/app_artifacts.zip')
# ==========================================
```

## Step 2 — Test locally (recommended)

Put `app.py`, `requirements.txt`, and `genre_model_plain.pt` in one
folder, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Any Python 3.10–3.12 environment works — no version matching needed.

## Step 3 — Deploy to Hugging Face Spaces (free)

1. huggingface.co → **New Space** → SDK: **Streamlit** → hardware:
   **CPU basic (free)**.
2. **Files → Add file → Upload files**: `app.py`, `requirements.txt`,
   `genre_model_plain.pt` (~110 MB — the web uploader handles Git LFS
   automatically), plus the unzipped artifacts and `samples/` folder
   when you have them.
3. Wait for the build (~5–10 min; check the Logs tab on failure). Your
   public URL: `https://huggingface.co/spaces/<you>/<space-name>`.

## Sanity checks

- **Spectrogram settings** in `app.py` (`SR=22050, N_MELS=128,
  N_FFT=2048, HOP=512`, per-clip min-max grayscale, flipud, Squish
  resize to 224) match your notebook's `make_spectrogram` + DataBlock
  exactly. If you ever change the notebook recipe, change `app.py`.
- **Grad-CAM** auto-targets ConvNeXt (`stages[-1]`) or ResNet
  (`layer4`); anything else just skips the overlay gracefully.
- `embeddings.npz` must be generated by the Step 1b cell above (concat-
  pooled backbone features) so the app's query embedding lives in the
  same space.

## Presentation tips

- Open on **Classify** with a bundled sample — one click, no file
  hunting on stage.
- Show the **Grad-CAM overlay** and explain one example ("the model
  focused on the low-frequency rhythm here").
- End on **How it works** — the confusion matrix and F1 table show
  rigor, and being honest about which genres confuse the model always
  lands well with reviewers.
