# Music Genre Classification from Audio Spectrograms

**Erdős Institute Deep Learning Boot Camp — Summer 2026**
**Team:** Xiaoyu Ma, Zhihao Mu

🎵 **Live Demo:** [huggingface.co/spaces/XiaoyuMa94/Music_Genre_Classification](https://huggingface.co/spaces/XiaoyuMa94/Music_Genre_Classification)

---

## What This Project Does

We built a deep learning pipeline that takes a music clip as input and predicts its genre. The idea is pretty simple: convert raw audio into a mel-spectrogram (basically a visual representation of sound), then treat it as an image classification problem using a CNN.

The app has three pages:
- **Classify** — upload any music clip and get a genre prediction with confidence scores and a Grad-CAM heatmap showing what the model focused on
- **Find Similar** — upload a clip and find the closest matching tracks from the FMA validation set using embedding similarity
- **How It Works** — confusion matrix and per-genre F1 scores from the trained model

---

## Dataset

We used the [Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset:

- **FMA-Small:** 8,000 clips, 30 seconds each, 8 balanced genres (used for our custom CNN experiments)
- **FMA-Medium:** 25,000 clips, 16 genres with heavy class imbalance (used for the ResNet34 model)

The 16 genres in FMA-Medium are: Rock, Electronic, Experimental, Hip-Hop, Folk, Instrumental, Pop, International, Classical, Old-Time/Historic, Jazz, Country, Soul-RnB, Spoken, Blues, Easy Listening.

---

## How It Works

### Step 1 — Audio to Mel-Spectrogram
Each 30-second `.mp3` clip is loaded using `librosa` and converted to a log-mel spectrogram (128 mel bins, sr=22050, n_fft=2048, hop_length=512). The spectrogram is saved as a PNG image. This step is slow the first time (~1 hour for 25,000 clips) but cached so reruns are instant.

### Step 2 — Image Classification
We treated each spectrogram as a 224×224 RGB image and trained two types of models:

**Custom CNN (trained from scratch on FMA-Small):**
- 4 conv blocks with BatchNorm and MaxPooling
- Global Average Pooling + 2 fully connected layers
- Test accuracy: ~47%, Macro F1: 0.44

**ResNet34 fine-tuned via FastAI (trained on FMA-Medium):**
- Pretrained ImageNet weights, fine-tuned with `learn.fine_tune(8)`
- Label smoothing (0.1) and macro-F1 metric to handle class imbalance
- Validation accuracy: ~65%, Macro F1: ~0.40

The ResNet model benefited from more data (25k vs 8k) but the class imbalance in FMA-Medium made it harder to get strong F1 scores across all genres.

### Step 3 — Web App
The final model is exported as a plain PyTorch checkpoint (`genre_model_plain.pt`) with no FastAI dependency at inference time, and served via a Streamlit app deployed on Hugging Face Spaces.

---

## Results

### Custom CNN on FMA-Small (8 genres)

| Genre | F1-Score |
|---|---|
| Hip-Hop | 0.75 |
| Electronic | 0.63 |
| International | 0.52 |
| Instrumental | 0.41 |
| Experimental | 0.38 |
| Pop | 0.37 |
| Rock | 0.35 |
| Folk | 0.14 |

Overall test accuracy: **47%** — not amazing but a reasonable baseline for a custom CNN trained from scratch on 6,400 samples.

### ResNet34 on FMA-Medium (16 genres)
Best validation accuracy: **65%**, Macro F1: **0.40**

The lower macro F1 here reflects the imbalance problem — genres like Easy Listening (only 21 training clips) are very hard to learn.

---

## Repository Structure

```
├── app.py                                  # Streamlit demo app
├── requirements.txt                        # Python dependencies
├── Dockerfile                              # Hugging Face Spaces deployment
├── genre_model_plain.pt                    # Exported PyTorch model (no FastAI needed)
├── embeddings.npz                          # Precomputed track embeddings for similarity search
├── confusion_matrix.png                    # Validation confusion matrix
├── per_class_f1.csv                        # Per-genre F1 scores
├── samples/                                # Sample audio clips (2 per genre)
├── fma-music-genre-classification.ipynb    # Training notebook (runs on Kaggle)
├── kaggle_cnn.py                           # Custom CNN training script
└── README.md
```

---

## How to Run

### Training (Kaggle)
1. Open `fma-music-genre-classification.ipynb` on Kaggle
2. Add Data → attach *"FMA - Free Music Archive - Small & Medium"* by imsparsh
3. Set Accelerator to GPU T4
4. Run all cells

### Demo App (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

### Demo App (Online)
Just visit: https://huggingface.co/spaces/XiaoyuMa94/Music_Genre_Classification

---

## Dependencies

```
torch, timm, streamlit, librosa, soundfile, av, matplotlib, pandas, numpy
```

No FastAI needed at inference — the model is exported as a plain PyTorch checkpoint.

---

## Lessons Learned

A few things we ran into that are worth noting:

- **FMA has corrupted files.** Some mp3s throw decoding errors. We handled this with try/except and by filtering out files under 1KB.
- **Precomputing spectrograms makes a huge difference.** Loading mp3s on the fly during training is extremely slow even with a GPU. Saving them as `.png` files first made each epoch ~10x faster.
- **Class imbalance is a real problem.** FMA-Medium has 7,000 Rock clips vs 21 Easy Listening clips. Inverse-frequency class weighting helped but didn't fully solve it.
- **Transfer learning from ImageNet doesn't transfer perfectly to spectrograms.** We saw consistent overfitting where training accuracy hit 99% while validation plateaued around 60%.
- **num_workers in DataLoader causes issues on Windows.** Set `num_workers=0` when running locally on Windows to avoid multiprocessing errors with the audio backend.

---

## What We'd Do Differently

- Use audio-native pretrained models like `wav2vec2` or `AST` (Audio Spectrogram Transformer) instead of vision models
- Augment training data with pitch shifting, time stretching, and background noise injection
- Train on FMA-Medium with balanced sampling to fix the class imbalance properly
