"""
Music Genre Classifier — Streamlit demo app
Deploy on Hugging Face Spaces (free CPU tier is enough).

Pure PyTorch + timm — NO fastai needed at inference, so there is no
pickle/version coupling with the training environment. The model file is
produced by the "plain export" cell in README.md Step 1a (it saves the
weights + head spec + genre vocab, and numerically verifies the round-trip
inside Kaggle before saving).

Required file:   genre_model_plain.pt
Optional files:  confusion_matrix.png        (methodology page)
                 per_class_f1.csv            (methodology page)
                 embeddings.npz              (find-similar page: keys 'emb', 'names')
                 samples/<Genre>/<clip>.mp3  (one-click demo clips)

Spectrogram settings below MUST match how training spectrograms were made.
"""

import glob
import io
import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- settings
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP = 512
CLIP_SECONDS = 30          # length used per training spectrogram
MAX_DECODE_SECONDS = 300   # decode up to this much, then keep the center
IMG_SIZE = 224             # training used Resize(224, method=Squish)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

_pts = sorted(glob.glob("genre_model_plain*.pt"))
MODEL_PATH = _pts[0] if _pts else "genre_model_plain.pt"
SAMPLES_DIR = Path("samples")

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="wide")


# ---------------------------------------------------------------- model
class ConcatPool(nn.Module):
    """fastai's AdaptiveConcatPool2d: [max-pool, avg-pool] concatenated."""
    def forward(self, x):
        return torch.cat([F.adaptive_max_pool2d(x, 1),
                          F.adaptive_avg_pool2d(x, 1)], dim=1)


class FeatureBody(nn.Module):
    """Run the timm backbone the way fastai's TimmBody does:
    forward_features() only — skip timm's own head norm/pool."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_features(x)


def build_head(spec):
    """Rebuild the classifier head from the exported layer spec."""
    layers = []
    for s in spec:
        kind = s[0]
        if kind == "concat_pool":
            layers.append(ConcatPool())
        elif kind == "flatten":
            layers.append(nn.Flatten(1))
        elif kind == "bn":
            layers.append(nn.BatchNorm1d(s[1]))
        elif kind == "dropout":
            layers.append(nn.Dropout(s[1]))
        elif kind == "linear":
            layers.append(nn.Linear(s[1], s[2], bias=s[3]))
        elif kind == "relu":
            layers.append(nn.ReLU(inplace=True))
        else:
            raise ValueError(f"unknown head layer: {kind}")
    return nn.Sequential(*layers)


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import timm
    d = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    body = timm.create_model(d["arch"], pretrained=False, num_classes=0)
    body.load_state_dict(d["body_sd"])
    head = build_head(d["head_spec"])
    head.load_state_dict(d["head_sd"])
    model = nn.Sequential(FeatureBody(body), head).eval()
    return model, list(d["vocab"])


@st.cache_resource
def load_embeddings():
    if not os.path.exists("embeddings.npz"):
        return None
    d = np.load("embeddings.npz", allow_pickle=True)
    emb = d["emb"].astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb, list(d["names"])


# ---------------------------------------------------------------- audio -> image
def _decode_with_av(audio_bytes: bytes):
    """Fallback decoder (m4a/aac and friends) using PyAV's bundled FFmpeg."""
    import av
    container = av.open(io.BytesIO(audio_bytes))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
    chunks, total, limit = [], 0, SR * MAX_DECODE_SECONDS
    for frame in container.decode(audio=0):
        out = resampler.resample(frame)
        for f in (out if isinstance(out, list) else [out]):
            arr = f.to_ndarray().astype(np.float32).ravel() / 32768.0
            chunks.append(arr)
            total += arr.size
        if total >= limit:
            break
    container.close()
    if not chunks:
        return None
    return np.concatenate(chunks)[:limit]


def _center_crop(y):
    """Keep the middle CLIP_SECONDS — training clips are mid-song excerpts,
    so classifying a song's intro would be a train/serve mismatch."""
    n = SR * CLIP_SECONDS
    if y is not None and len(y) > n:
        start = (len(y) - n) // 2
        y = y[start:start + n]
    return y


def load_audio(audio_bytes: bytes):
    """Decode to mono float32 @ SR. librosa/soundfile first, PyAV fallback."""
    import librosa
    try:
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True,
                            duration=MAX_DECODE_SECONDS)
        return _center_crop(y)
    except Exception:
        try:
            return _center_crop(_decode_with_av(audio_bytes))
        except Exception:
            return None


def audio_to_melspec(audio_bytes: bytes):
    """Decode audio -> mel-spectrogram (dB), same recipe as training."""
    import librosa
    y = load_audio(audio_bytes)
    if y is None or len(y) < SR:  # decode failure, or under 1 s
        return None
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP)
    return librosa.power_to_db(mel, ref=np.max)


def melspec_to_image(mel_db):
    """dB mel-spectrogram -> PIL image, EXACTLY as the training notebook:
    per-clip min-max to uint8 grayscale, flipped so low freqs are at the
    bottom. RGB conversion mirrors what the training image loader did."""
    from PIL import Image
    x = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
         * 255).astype(np.uint8)
    return Image.fromarray(np.flipud(x)).convert("RGB")


def preprocess(img):
    """PIL RGB image -> normalized 1x3x224x224 tensor (Squish resize +
    ImageNet stats, matching the training DataBlock)."""
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)


# ---------------------------------------------------------------- inference
def predict(model, vocab, xb):
    with torch.no_grad():
        probs = torch.softmax(model(xb), dim=1)[0].cpu().numpy()
    probs = dict(zip(vocab, probs.tolist()))
    pred = max(probs, key=probs.get)
    return pred, probs


def find_cam_layer(body):
    body = getattr(body, "m", body)     # unwrap FeatureBody
    if hasattr(body, "stages"):    # ConvNeXt and many timm models
        return body.stages[-1]
    if hasattr(body, "layer4"):    # ResNet family
        return body.layer4
    return None


def gradcam(model, xb):
    """Grad-CAM heatmap in [0,1] over the input, or None."""
    try:
        target = find_cam_layer(model[0])
        if target is None:
            return None
        acts, grads = [], []
        h1 = target.register_forward_hook(lambda m, i, o: acts.append(o))
        h2 = target.register_full_backward_hook(
            lambda m, gi, go: grads.append(go[0]))
        try:
            out = model(xb)
            out[0, out.argmax(dim=1)].backward()
        finally:
            h1.remove(); h2.remove()
        a, g = acts[0][0], grads[0][0]
        w = g.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((w * a).sum(0)).detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    except Exception:
        return None


def overlay_cam(img, cam):
    import matplotlib.cm as cm
    from PIL import Image
    heat = Image.fromarray(
        (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)).resize(img.size)
    return Image.blend(img.convert("RGB"), heat, alpha=0.35)


def embed_clip(model, xb):
    """Concat-pooled backbone features (matches the README embeddings cell)."""
    with torch.no_grad():
        f = model[0](xb)
        v = ConcatPool()(f).flatten().cpu().numpy().astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------- ui helpers
def list_samples():
    out = {}
    if SAMPLES_DIR.exists():
        for gdir in sorted(SAMPLES_DIR.iterdir()):
            if gdir.is_dir():
                clips = sorted(gdir.glob("*.mp3")) + sorted(gdir.glob("*.wav"))
                if clips:
                    out[gdir.name] = clips
    return out


def get_audio_bytes():
    """Uploader + bundled sample picker. Returns (bytes, label) or (None, None)."""
    up = st.file_uploader("Upload a clip (mp3/wav, a few seconds is enough)",
                          type=["mp3", "wav", "ogg", "flac", "m4a"])
    if up is not None:
        return up.read(), up.name

    samples = list_samples()
    if samples:
        st.caption("…or try a bundled sample:")
        c1, c2 = st.columns(2)
        genre = c1.selectbox("Genre", list(samples))
        clip = c2.selectbox("Clip", samples[genre], format_func=lambda p: p.name)
        if st.button("Use this sample"):
            return clip.read_bytes(), f"{genre}/{clip.name}"
    return None, None


# ---------------------------------------------------------------- pages
def page_classify(model, vocab):
    st.header("🎵 Classify a clip")
    audio, label = get_audio_bytes()
    if audio is None:
        st.info("Upload a clip or pick a sample to get started.")
        return

    st.audio(audio)
    with st.spinner("Analyzing…"):
        mel_db = audio_to_melspec(audio)
        if mel_db is None:
            st.error("Couldn't decode this file (or it's under ~3 seconds). "
                     "Try mp3, wav, m4a, ogg, or flac.")
            return
        img = melspec_to_image(mel_db)
        xb = preprocess(img)
        pred, probs = predict(model, vocab, xb)
        cam = gradcam(model, xb)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("What the model saw")
        st.image(img, caption="Mel-spectrogram", use_container_width=True)
        if cam is not None:
            st.image(overlay_cam(img, cam),
                     caption="Grad-CAM — regions that drove the prediction",
                     use_container_width=True)
    with right:
        st.subheader("Prediction")
        st.metric("Genre", pred, f"{probs[pred]*100:.1f}% confidence")
        top3 = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        st.write(" · ".join(f"**{g}** {p*100:.1f}%" for g, p in top3))
        st.subheader("All genres")
        st.bar_chart(probs, horizontal=True)


def page_similar(model):
    st.header("🔎 Find similar tracks")
    data = load_embeddings()
    if data is None:
        st.info("Add `embeddings.npz` (see README export snippet) to enable "
                "similar-track search.")
        return
    ref_emb, names = data

    audio, _ = get_audio_bytes()
    if audio is None:
        return
    st.audio(audio)
    with st.spinner("Embedding…"):
        mel_db = audio_to_melspec(audio)
        if mel_db is None:
            st.error("Clip too short.")
            return
        v = embed_clip(model, preprocess(melspec_to_image(mel_db)))
    sims = ref_emb @ v
    order = np.argsort(-sims)[:10]
    st.subheader("Closest tracks in the reference set")
    for i in order:
        st.write(f"**{names[i]}** — similarity {sims[i]:.3f}")


def page_methodology():
    st.header("📊 How it works")
    st.markdown(
        f"""
**Pipeline.** Audio → mel-spectrogram ({N_MELS} mels, {N_FFT} FFT,
hop {HOP}, {SR} Hz) → rendered as a grayscale image → **ConvNeXt-Tiny**
(transfer learning from ImageNet-22k, fine-tuned) → genre probabilities.

**Dataset.** [FMA — Free Music Archive](https://github.com/mdeff/fma),
16 top-level genres, official train/validation/test split.

**Why spectrograms?** They turn audio into images, letting us reuse
powerful pretrained vision models — genre cues (rhythm, timbre,
instrumentation) show up as visual texture.
        """)
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Confusion matrix")
        else:
            st.info("Add `confusion_matrix.png` to show the confusion matrix.")
    with c2:
        if os.path.exists("per_class_f1.csv"):
            import pandas as pd
            st.dataframe(pd.read_csv("per_class_f1.csv"),
                         use_container_width=True)
        else:
            st.info("Add `per_class_f1.csv` to show per-genre F1 scores.")


# ---------------------------------------------------------------- main
def main():
    st.sidebar.title("Music Genre Classifier")
    page = st.sidebar.radio("Page", ["Classify", "Find similar", "How it works"])
    st.sidebar.markdown("---")
    st.sidebar.caption("ConvNeXt on mel-spectrograms · FMA dataset · "
                       "PyTorch + Streamlit")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file `{MODEL_PATH}` not found. Run the plain-export "
                 "cell from README.md Step 1a in your Kaggle notebook and "
                 "put `genre_model_plain.pt` next to app.py.")
        return
    model, vocab = load_model()

    if page == "Classify":
        page_classify(model, vocab)
    elif page == "Find similar":
        page_similar(model)
    else:
        page_methodology()


if __name__ == "__main__":
    main()
