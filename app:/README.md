# 🎵 Music Genre Classifier

Upload a music clip, get a genre — plus a picture of *why*.

This app converts audio into a mel-spectrogram (an image of sound) and classifies it into
**16 genres** with a ConvNeXt-Tiny network fine-tuned on the
[Free Music Archive](https://github.com/mdeff/fma) dataset. It also shows you the
spectrogram the model actually looked at, a Grad-CAM heatmap of the time–frequency regions
that drove the decision, and the ten most acoustically similar tracks in the reference set.

**[▶ Try the live demo](https://huggingface.co/spaces/<your-username>/<your-space>)** ·
Runs on CPU. No GPU needed.

<!-- Add a screenshot here: docs/screenshot-classify.png -->

---

## Contents

- [What it does](#what-it-does)
- [Genres](#genres)
- [How well does it work?](#how-well-does-it-work)
- [Run it locally](#run-it-locally)
- [Deploy your own copy](#deploy-your-own-copy)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Known limitations](#known-limitations)
- [Training and export](#training-and-export)
- [Credits and license](#credits-and-license)

---

## What it does

The app has three pages.

**🎵 Classify.** Upload an `mp3`, `wav`, `m4a`, `ogg`, or `flac` file — or click one of the
bundled sample clips — and the app returns the predicted genre, a confidence score, and the
full probability distribution over all 16 genres. Alongside the prediction you see:

- the **mel-spectrogram** the network received as input, and
- a **Grad-CAM overlay** marking the regions of that spectrogram the network weighted most
  heavily. Bright areas low on the frequency axis usually mean the model keyed on rhythm and
  bass; bright areas up top usually mean it keyed on cymbals, vocals, or synth texture.

**🔎 Find similar.** Embeds your clip using the network's pooled backbone features (a
1,536-dimensional vector) and returns the nearest neighbours by cosine similarity from a
reference set of 2,504 FMA validation tracks, listed with artist and title. This is genre
classification's more interesting cousin: the model has learned an acoustic space, and you can
browse it.

**📊 How it works.** The pipeline description, the confusion matrix, and the per-genre
precision/recall/F1 table — including the genres the model is bad at.

Only the model file is required. Every other feature (samples, similarity search, confusion
matrix, F1 table) degrades gracefully to an "add this file" note if its artifact is missing.

## Genres

`Blues` · `Classical` · `Country` · `Easy Listening` · `Electronic` · `Experimental` ·
`Folk` · `Hip-Hop` · `Instrumental` · `International` · `Jazz` · `Old-Time / Historic` ·
`Pop` · `Rock` · `Soul-RnB` · `Spoken`

These are FMA's top-level genre labels, assigned by the uploading artists.

## How well does it work?

Scores on FMA-medium's **official, artist-disjoint test split** (2,572 clips the model never
saw during training or checkpoint selection):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Random guess | 0.066 | 0.048 |
| Always predict "Rock" | 0.276 | 0.027 |
| Logistic regression on mel-band statistics | 0.564 | 0.293 |
| SimpleCNN, trained from scratch (~1M params) | 0.636 | 0.433 |
| **ConvNeXt-Tiny, pretrained (~28M params)** ← *shipped in this app* | **0.623** | **0.423** |

Accuracy is ~9× better than chance. Macro-F1 (the unweighted mean of per-genre F1) is the
number to trust: the dataset is dominated by Rock and Electronic, so accuracy alone flatters
any model that learns to guess those two.

Per-genre performance varies enormously, and the app is upfront about it:

| Strong | Weak |
|---|---|
| Rock (F1 0.83), Classical (0.82), Old-Time/Historic (0.80), Electronic (0.72), Hip-Hop (0.72) | Pop (0.11), Soul-RnB (0.07), Blues (0.00), Easy Listening (0.00) |

The weak genres fail for two different reasons. Blues (58 training clips) and Easy Listening
(13) fail because there is nothing to learn from. Pop fails despite having 945 clips, because
"Pop" is a fuzzy human category that overlaps Rock, Electronic, and Folk — its predictions
scatter across those three.

> **Note on the leaderboard.** The from-scratch CNN edged out the pretrained ConvNeXt in the
> final run. Run-to-run variance is ±1–2 points, and an earlier ConvNeXt run scored 0.648 /
> 0.445. The honest reading is that ImageNet features transfer only *weakly* to spectrograms,
> which are not natural images; pretraining bought convergence speed and training stability
> rather than a decisive accuracy win. We ship ConvNeXt because its training curve is stable
> and reproducible, not because it is dramatically better.

## Run it locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501>.

**Requirements.** Python 3.10–3.12. Any recent PyTorch works — the model is stored as plain
weights, not a pickled training object, so there is no version coupling to worry about.
Inference on CPU takes about a second per clip.

**The model file.** `genre_model_plain.pt` (~110 MB) must sit next to `app.py`. If the repo
uses Git LFS, `git lfs pull` will fetch it; otherwise download it from the
[Releases](https://github.com/<your-username>/<your-repo>/releases) page. Without it, the app
starts and tells you what's missing.

## Deploy your own copy

The app is built for [Hugging Face Spaces](https://huggingface.co/spaces) on the free CPU tier.

1. **New Space** → SDK: **Streamlit** → Hardware: **CPU basic (free)**.
2. Upload `app.py`, `requirements.txt`, and `genre_model_plain.pt`. The web uploader handles
   Git LFS for the model file automatically.
3. Optionally upload `confusion_matrix.png`, `per_class_f1.csv`, `embeddings.npz`, and the
   `samples/` folder to enable the methodology page, similarity search, and one-click demo
   clips.
4. Wait ~5–10 minutes for the build. Check the **Logs** tab if it fails.

Your Space will be live at `https://huggingface.co/spaces/<you>/<space-name>`.

## How it works

```
audio  →  mono @ 22,050 Hz  →  128-band log-mel spectrogram  →  224×224 grayscale image
                                (n_fft 2048, hop 512)             (per-clip min–max norm)
                                                                          │
                                                    ConvNeXt-Tiny backbone (ImageNet-22k)
                                                                          │
                                              concat-pool → 1536-d ──┬──→ classifier head
                                                                     │         │
                                                                     │    16 genre probs
                                                                     └──→ similarity search
```

Some deliberate choices worth knowing:

- **Mel scale + log (dB) compression.** Mel bands space frequency the way human hearing does;
  the log makes quiet spectral structure visible instead of letting a few loud partials
  dominate.
- **Per-clip normalization.** Every spectrogram is min–max scaled to 0–255 independently, so
  the model learns spectral *patterns* rather than mastering loudness.
- **Squish, not crop.** Images are resized anisotropically to 224×224 with no cropping. A
  spectrogram's coordinates *mean* something — position encodes time and frequency — so the
  training and inference transforms must be identical, and random crops would slice off real
  information.
- **Center-cropping at inference.** Uploads longer than 30 s are cropped to the middle 30 s.
  FMA's training clips are mid-song excerpts, so classifying a song's intro would be a
  train/serve mismatch.
- **No fastai at inference.** The model was trained with fastai, but fastai's pickled export is
  coupled to exact library versions and breaks outside its home environment. Instead, the
  training notebook exports raw weights plus a declarative description of the classifier head,
  rebuilds the network in plain PyTorch + timm, and *numerically verifies* — before saving —
  that the rebuild produces bit-identical outputs. The app therefore depends only on `torch`
  and `timm`.

## Repository layout

```
app.py                    Streamlit application (all three pages)
requirements.txt          Python dependencies
genre_model_plain.pt      REQUIRED — weights + head spec + genre vocabulary
confusion_matrix.png      optional — methodology page
per_class_f1.csv          optional — methodology page
embeddings.npz            optional — "Find similar" page (keys: 'emb', 'names')
samples/<Genre>/*.mp3     optional — one-click demo clips
```

Anything marked optional can be absent; the app will say so and keep working.

## Known limitations

Read these before trusting a prediction.

**Domain shift.** The model is trained on Creative-Commons indie music. Commercially mastered
tracks — anything ripped from a streaming service or YouTube — are out of distribution, and
their predictions tend to skew toward `Experimental`, which acts as the model's de facto
catch-all class.

**Rare genres are not learnable here.** `Easy Listening` has 13 training clips and `Blues` has
58. Their F1 is effectively zero. No loss function fixes that; the app reports the failure
rather than hiding it.

**Fuzzy labels.** `Pop` is a marketing category, not an acoustic one. A 0.11 F1 on Pop is
partly the model's fault and partly the label's.

**Rhythm is compressed away by the resize.** A 30-second clip yields a 128 × 1,292
spectrogram, but the network consumes 224 × 224 — the time axis is squished ~5.8×, so each
pixel column spans roughly 135 ms. Percussive transients (~20 ms) are smeared into their
neighbours, and fine rhythmic detail — groove, swing, drum-attack sharpness — is attenuated
before the model ever sees it. Because the resize is a fixed squish rather than a fixed
pixels-per-second rate, a clip shorter than 30 s is compressed less, so the same tempo maps to
different pixel spacings and the model has no consistent time scale to learn from. This pushes
the network toward timbral rather than rhythmic evidence, which is consistent with a logistic
regression on purely time-averaged mel statistics already reaching 56.4% accuracy.

**Confidence scores are uncalibrated.** The number the app prints next to the genre is a
softmax output, not a probability you should bet on. Treat it ordinally.

**Grad-CAM is a heuristic.** It tells you where gradient-weighted activations were large. It
does not tell you what the model would have predicted had that region been different.

## Training and export

The full training pipeline — data loading, spectrogram rendering, the five-model comparison
ladder, and the plain-PyTorch export cell that produces `genre_model_plain.pt` — lives in the
Kaggle notebook:

**[fma-music-genre-classification.ipynb](notebooks/fma-music-genre-classification.ipynb)**

Headline training details: FMA-medium (25,000 clips, 19,909 / 2,504 / 2,572 official
artist-disjoint split), class-weighted cross-entropy with square-root-dampened inverse-frequency
weights capped at 3×, label smoothing 0.1, MixUp, mixed precision, 2 frozen + 24 unfrozen
epochs on one T4, checkpointed on validation macro-F1. About 50 minutes of GPU time.

If you retrain with different spectrogram settings, you **must** update the constants at the
top of `app.py` (`SR`, `N_MELS`, `N_FFT`, `HOP`, `IMG_SIZE`) to match. A model only understands
inputs prepared exactly the way its training data was.

## Credits and license

- **Dataset:** [FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma) — Defferrard,
  Benzi, Vandergheynst, and Bresson, ISMIR 2017. All audio is Creative Commons licensed; see
  the FMA repository for per-track terms.
- **Backbone:** `convnext_tiny.fb_in22k` via [timm](https://github.com/huggingface/pytorch-image-models).
  ConvNeXt: *A ConvNet for the 2020s*, Liu et al., CVPR 2022.
- **Built with:** PyTorch, timm, librosa, Streamlit.

Code in this repository is released under the MIT License. The bundled sample clips retain
their original Creative Commons licenses from FMA.

---

<sub>Built by Xiaoyu Ma and Zhihao Mu with the assistant of Claude Code. Questions, bug reports, and genre disagreements welcome in the
[issues](https://github.com/<your-username>/<your-repo>/issues).</sub>
