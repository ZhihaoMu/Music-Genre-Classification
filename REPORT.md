# Music Genre Classification from Mel-Spectrograms

**Author:** Xiaoyu Ma, Zhihao Mu · **Date:** July 2026 · **Dataset:** FMA (Free Music Archive)

## Abstract

This project builds an end-to-end music genre classifier: raw audio is converted into
mel-spectrogram images and classified into 16 genres by a fine-tuned ConvNeXt-Tiny
convolutional network. On the official held-out test split of the FMA *medium* dataset
(25,000 clips), the model reaches **62.3% accuracy and 0.423 macro-F1**, roughly ten times
the 6.6% random-guess baseline. A ladder of five models — from random guessing to a
pretrained deep network — quantifies where the performance comes from. The classifier is
deployed as a public web application (Streamlit on Hugging Face Spaces) featuring Grad-CAM
explanations, probability distributions, and embedding-based similar-track search.

## 1. Introduction

Music genre classification is a canonical audio machine-learning task: given a short audio
clip, predict its genre. The central design idea of this project is to convert the audio
problem into an image problem. A mel-spectrogram renders sound as a picture — time on the
horizontal axis, frequency on the vertical, brightness encoding energy — and genre cues
such as rhythm, timbre, and instrumentation appear as visual texture. This allows us to
reuse powerful pretrained image models rather than training an audio model from scratch.

Beyond the model itself, the project delivers a working product: a web application where
anyone can upload a clip, see the spectrogram the model analyzed, inspect which
time-frequency regions drove the prediction (Grad-CAM), and browse acoustically similar
tracks.

## 2. Data

We use the FMA *medium* subset: 25,000 Creative-Commons licensed 30-second clips across 16
top-level genres, with the official artist-disjoint train/validation/test split (19,909 /
2,504 / 2,572 clips). The class distribution is severely imbalanced: Rock (7,103 clips) and
Electronic (6,314) together account for more than half the data, while Blues has 74 clips
and Easy Listening only 21. This imbalance shapes both our training choices and our
evaluation: we report macro-F1 (the unweighted average of per-genre F1) alongside accuracy,
because accuracy alone can be inflated by performing well on the two dominant genres.

## 3. Method

### 3.1 Preprocessing

Each clip is loaded as 30 seconds of mono audio at 22,050 Hz and converted to a 128-band
log-mel spectrogram (FFT window 2048, hop 512), normalized per clip to an 8-bit grayscale
image. Images are stored in genre-named folders, resized to 224×224, and normalized with
ImageNet statistics at load time. The identical preprocessing code runs in the deployed web
application, which is essential: a model only understands inputs prepared exactly as its
training data was.

### 3.2 Model ladder

To attribute performance honestly, we evaluate five models of increasing capability on the
same test split: (1) uniform random guessing; (2) a majority-class baseline that always
predicts Rock; (3) multinomial logistic regression on 256 hand-crafted features (per-mel-band
mean and standard deviation); (4) **SimpleCNN**, a hand-built five-block convolutional
network (~1M parameters) trained from random initialization; and (5) **ConvNeXt-Tiny**
(~28M parameters), pretrained on ImageNet-22k and fine-tuned.

### 3.3 Training

Both deep models minimize class-weighted cross-entropy with label smoothing 0.1. The
weights are square-root-dampened inverse frequencies capped at 3×, a compromise we arrived
at empirically: uncapped inverse-frequency weights (up to ~437× for the rarest genre)
caused catastrophic failure (9% accuracy), while no weighting sacrificed rare-genre F1.
ConvNeXt is fine-tuned with the fastai one-cycle schedule (2 frozen + 24 unfrozen epochs),
MixUp augmentation, and mixed-precision training; SimpleCNN trains for 25 epochs with
weight decay 0.1. For both models we checkpoint on validation macro-F1 and restore the best
epoch, so overfitting past the optimum cannot degrade the final model.

## 4. Results

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| Random guess | 0.066 | 0.048 |
| Majority class (Rock) | 0.276 | 0.027 |
| Logistic regression (mel statistics) | 0.564 | 0.293 |
| SimpleCNN (from scratch) | 0.636 | 0.433 |
| ConvNeXt-Tiny (pretrained) | 0.623 | 0.423 |

Per-genre results (validation split) span an enormous range. The model is strong on genres
with distinctive acoustic signatures and ample data: Rock (F1 0.83), Classical (0.82),
Old-Time/Historic (0.80), Electronic (0.72), and Hip-Hop (0.72). It fails almost completely
on data-starved genres — Blues (F1 0.00, 58 training clips), Easy Listening (0.00, 13
clips), Soul-RnB (0.07, 94 clips) — and struggles with Pop (0.11) despite Pop having 945
training clips. The confusion matrix shows Pop predictions scattering into Rock and
Electronic, and Experimental absorbing errors from many genres.

## 5. Analysis

**Performance tracks data availability.** Plotting per-genre F1 against training-clip count
shows a near-monotonic relationship. The exceptions are informative: Old-Time/Historic
scores 0.80 from only 408 clips because vintage recordings have an unmistakable spectral
signature, while Pop scores 0.11 from 945 clips because "Pop" is a fuzzy human category
that overlaps Rock, Electronic, and Folk — a labeling problem, not a data-volume problem.

**Transfer learning: faster and more stable, but not decisively better here.** The headline
surprise is that SimpleCNN (63.6%) slightly outperformed the pretrained ConvNeXt (62.3%) in
the final run. Three factors explain this. First, run-to-run variance is roughly ±1–2
percentage points — an earlier ConvNeXt run on identical data scored 64.8% / 0.445, above
SimpleCNN. Second, checkpoint selection: we restore the best *macro-F1* epoch, and
ConvNeXt's F1 peaked at epoch 7 (65.8% validation accuracy) even though later epochs
reached 67.1% accuracy; optimizing one metric costs the other. Third, and most
interesting: ImageNet features transfer only weakly to spectrograms, which are not natural
images. Where pretraining clearly wins is *convergence behavior*: ConvNeXt reached 65%
validation accuracy within 5 epochs and improved smoothly, while SimpleCNN needed 20+
epochs and oscillated wildly (dropping to 34% accuracy mid-training). The fair conclusion
is that pretraining buys speed and stability, and a modest quality edge on average, rather
than a categorical advantage on this domain.

**Class weighting is a dial, not a switch.** The project's most instructive failure was the
original uncapped inverse-frequency loss, which drove the model to over-predict rare
genres and collapsed accuracy to 9%. The final square-root-capped weighting recovers
rare-genre performance (macro-F1 0.42 vs. 0.38 unweighted) at a cost of about one point of
accuracy.

## 6. The product: a web application

The classifier is deployed as a Streamlit application on Hugging Face Spaces (free CPU
tier). A visitor can upload a clip (mp3/wav/m4a/ogg/flac) or pick a bundled sample, and the
app displays the mel-spectrogram, a Grad-CAM overlay highlighting the time-frequency
regions that drove the prediction, the predicted genre with its confidence, and the full
16-genre probability distribution. A second page performs similar-track search by embedding
the clip with the network's penultimate features and retrieving nearest neighbors (cosine
similarity) among the validation tracks, displayed with artist and title. A third page
documents the methodology with the confusion matrix and per-genre F1 table.

Deployment required solving a real MLOps problem: fastai's pickled model export could not
be loaded outside the training environment (library-version coupling). The solution
exports raw weights plus a declarative head specification, rebuilds the network in plain
PyTorch + timm, and numerically verifies inside the training environment that the rebuilt
model's outputs match to zero difference before saving. The app therefore has no fastai
dependency and no version coupling. Inference robustness additions include a PyAV decoding
fallback for m4a/AAC files and center-cropping uploads to 30 seconds, since training clips
are mid-song excerpts rather than intros.

## 7. Limitations and future work

**Domain shift.** The model is trained on Creative-Commons indie music; commercially
mastered tracks (e.g., from YouTube) are out-of-distribution, and their predictions skew
toward Experimental — the model's de facto catch-all class. 

**Rare genres.** No training
technique can learn Easy Listening from 13 examples; options are collecting data, merging
related genres, or honestly reporting the failure (our choice). 

**Temporal resolution loss from resizing.** A 30-second clip produces a 128 × 1,292 spectrogram, but the network consumes a 224 × 224 image, so the time axis is squished by a factor of roughly 5.8 and each output column spans about 135 ms of audio. Percussive transients (on the order of 20 ms) are smeared into their neighbours, and the fine rhythmic detail — groove, swing, drum-attack sharpness — that distinguishes genres such as Jazz, Blues, and Hip-Hop is attenuated before the model ever sees it. Compounding this, the resize is a fixed squish rather than a fixed pixels-per-second rate, so clips shorter than 30 seconds are compressed less and an identical tempo maps to different pixel spacings across the dataset, leaving the model no consistent time scale to learn from. This likely pushes the network toward timbral rather than rhythmic evidence — consistent with the 56.4% accuracy already reached by logistic regression on purely time-averaged mel statistics, which discards rhythm entirely. Preserving the time axis (for example, a 224 × 448 input, or the windowed inference proposed below) is the natural remedy.



**Future work**, in order
of expected value: training on multiple 10-second windows per track with per-track
probability voting (estimated +3–5% accuracy); audio-native pretrained backbones (AST,
PANNs) that would test the ImageNet-transfer hypothesis directly; and probability
calibration so the app's confidence numbers are trustworthy.






## 8. Conclusion

A pretrained image network fine-tuned on mel-spectrograms classifies 16 music genres at
62–65% accuracy — ten times chance — and the complete ladder from random guessing through
logistic regression to deep networks shows exactly what each increment of model capability
buys. The project's most valuable lessons were not the headline number but the failures on
the way: a loss function that silently optimized the wrong objective, a serialization
format that broke outside its home environment, and a deployment that revealed
train/serving skew and domain shift. The result is a working, explainable, publicly
deployed product with honestly reported limitations.

## Appendix: Reproducibility

Kaggle notebook, NVIDIA T4 GPU. Python 3.12.13, PyTorch 2.10.0, fastai 2.8.7, timm 1.0.26,
librosa 0.11.0. Dataset: FMA — Free Music Archive (Defferrard et al., 2017), medium subset,
official splits. Total training time: ~50 min (ConvNeXt) + ~25 min (SimpleCNN) + ~10 min
(baselines). Model export verified with max output difference 0.0 between the fastai
learner and the deployed plain-PyTorch rebuild.
