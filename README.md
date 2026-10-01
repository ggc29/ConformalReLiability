# Conformal Reliability

Official code release for **Conformal Reliability: A New Evaluation Metric for Conditional Generation**, accepted to ICML 2026.

[Project page](https://ggc29.github.io/CReL/) | [Paper](http://arxiv.org/abs/2605.30807)

## Installation

Create a dedicated environment, then install the dependencies from the repository root:

```bash
conda create -n crel-synthetic python=3.11 -y
conda activate crel-synthetic
pip install -r requirements.txt
```

For CUDA 12.4, install the matching PyTorch wheel before installing the requirements:

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Synthetic Experiments

The synthetic pipeline includes data generation, VAE training, directional quantile
regression (DQR), conformal calibration, and decoded-set evaluation.

Run the following commands from the repository root. The example uses GPU 0:

```bash
cd synthetic
export CUDA_VISIBLE_DEVICES=0
python 1-gen_syn_data.py
python 2-train_vae.py
python 3-train_dqr.py
python 4-calibrate_test_crel.py --alpha 0.1 --compute-y-space
python 4-calibrate_test_crel.py --alpha 0.02 --compute-y-space
```

Omit `--compute-y-space` for the faster latent-space coverage evaluation only.
The full decoded-set evaluation uses 4,000 test samples and approximately 20,000 grid
points in each space. Its cost depends on the number of points inside each region.
Use `--num-test`, `--z-grid-total`, and `--y-eval-grid-total` for smaller exploratory
runs; changing these values changes the evaluation setting.

### Configuration and data splits

Edit `synthetic/configs/train.yaml` before training. The default nonlinear
simulation uses `x_dim=38`, `y_dim=2`, noise standard deviation `0.3`,
VAE latent dimension `2`, MSE reconstruction, and KL weight `0.001`.
Only MSE and a single DQR model (`num_ens: 1`) are supported here.

The generator creates 100,000 samples, and the experiment uses the last 50,000:
30,000 for VAE training, 12,000 for DQR training, 4,000 for calibration, and 4,000
for testing. Generating the full sequence preserves the seeded data generation;
the first 50,000 samples are excluded from training and evaluation. The last
20,000 samples also serve as VAE validation data, and the calibration fold is
used for DQR early stopping.

Each stage fits separate StandardScaler instances for X and Y on its training
split. DQR uses a training quantile level of 0.07 with the default configuration;
`--alpha` controls the target miscoverage level during evaluation.

### Checkpoints, caches, and outputs

Data and results are saved under `synthetic/generated_data/` and
`synthetic/results/`, which are ignored by git.

Each training stage writes `best_checkpoint.json` to identify the checkpoint selected
by that run, its configuration, and input hashes. Downstream stages resolve and
validate the selected checkpoint through this manifest. Retrain downstream stages
when their inputs change. Use the supplied training scripts to create checkpoints
and manifests compatible with this implementation.

The distance cache records its model, data, evaluation code, and runtime identity.
Mismatching or legacy caches are recomputed automatically. Both alpha values can reuse
a matching cache. To force recomputation, pass `--recompute-distances`.

Under the DQR result directory, each alpha has its own
`crel_calibration_test_alpha_<alpha>.json` containing the radius and latent-space
coverage. With `--compute-y-space`, a separate directory named by alpha, test count,
and both grid sizes contains `coverage_and_size_statistics.json`. It reports
uncalibrated/calibrated Y-space coverage and the mean/std of region sizes.
**Region size is a count of evaluation-grid points, not continuous geometric area.**
Both decoded membership and region size are estimated on finite evaluation grids.

## Real-World Feature Experiments

The `realworld/` pipeline trains a VAE on generated-output features, trains DQR on
its sampled latents, calibrates prediction sets, and minimizes cosine similarity
inside each calibrated set to estimate reliability. Data and pretrained feature
extractors are not included. Prepare features using the folder structure below
and supply one `--data-root` path.

### Feature folders

The supplied presets use these directories beneath the same data root:

```text
MS-COCO2014/
  val2014-224_clip/                    # I2T condition images (CLIP)
  val2014_text_clip/                   # I2T reference text / T2I condition text
  val2014-224-gitlarge-len_50-clip/     # I2T generated captions (CLIP)
  val2014_512_dinov2/                  # T2I reference images (DINOv2)
  kandinsky-2-2_512_dinov2/            # T2I generated images (DINOv2)
```

Each directory contains one tensor per sample, saved using `torch.save`, for example
`COCO_val2014_000000000042.pt`. The three directories used by a given experiment
must contain the same samples in matching lexicographic filename order. Keep the
same sample identifier in all three directories. The evaluated COCO dataset has
40,504 samples with feature shape `[1, 768]`; the loader flattens each tensor.

The loader constructs paths using these exact naming rules; hyphens and underscores
are significant:

| Task | Condition X | Reference Y | Generated output Y_hat |
| --- | --- | --- | --- |
| I2T | `val2014-{img_res}_clip` | `val2014_text_{metric_name}` | `val2014-{img_res}-{model}-len_{text_len}-{metric_name}` |
| T2I | `val2014_text_clip` | `val2014_{img_res}_{metric_name}` | `{model}_{img_res}_{metric_name}` |

Choose `clip` or `bert` for I2T, and `clip` or `dinov2` for T2I. When changing the
model, resolution, or feature extractor, edit the corresponding preset. Only the
three feature folders selected by that preset are needed by training/evaluation.
Raw images and caption JSON files are used when preparing features, not by this pipeline.

The experiments use CLIP ViT-L/14, BERT `bert-base-uncased` CLS
embeddings, or DINOv2-base CLS embeddings. Reference and generated outputs must use
the same feature extractor. CLIP compares generated-output features with condition
features; BERT/DINOv2 compare with reference-target features. This choice is derived
from `metric_name`. Matching dimensions alone do not establish a shared embedding space.

The evaluated feature files store CLIP tensors as float16 and BERT/DINOv2 tensors
as float32. The loader applies L2 normalization in the stored dtype, then converts
to float32 and flattens each tensor. Supply features in their stored precision
without additional normalization before loading.

### Run I2T or T2I

Use the same environment and root `requirements.txt` as synthetic. Run the following
commands from the repository root:

```bash
cd realworld
export CUDA_VISIBLE_DEVICES=0

python 1-train_vae.py --config configs/i2t.yaml --data-root /path/to/MS-COCO2014
python 2-train_dqr.py --config configs/i2t.yaml --data-root /path/to/MS-COCO2014
python 3-test_fast.py --config configs/i2t.yaml --data-root /path/to/MS-COCO2014
```

For T2I, use `--config configs/t2i.yaml` in all three commands. Alternatively, set
`dataset.data_root` once in the YAML and omit `--data-root`. The I2T preset uses
GIT-large, resolution 224, maximum caption length 50, and CLIP. The T2I preset uses
Kandinsky 2.2, resolution 512, and DINOv2.

The test command includes reliability optimization by default.
It can be substantially slower than coverage evaluation. Use `--coverage-only` to
skip optimization, or `--num-test 20` to optimize only the first 20 test samples;
calibration and latent test coverage still use their complete splits. These flags
are for exploratory runs and do not represent the full default reliability evaluation.

### Configuration and data splits

Both presets use a 50-dimensional latent space and cosine reconstruction loss plus
KL with weight 0.001. DQR uses a training quantile level of 0.07 with the default
configuration. `reliability.alpha` sets the target miscoverage level to 0.1.

Samples are ordered lexicographically by filename. The first 75% train the VAE; the final
25% serve as its validation data and are divided 60%/20%/20% for DQR training,
calibration, and testing. For 40,504 samples, these counts are 30,378 / 6,075 /
2,025 / 2,026. The calibration fold is also used for DQR early stopping.

### Saved artifacts

Each experiment writes under `realworld/results/`, with task, resolution, model,
caption length, and feature type in its path. This directory is ignored by git.

- `best_checkpoint.json` identifies each stage's selected model, configuration,
  and input hashes. Downstream stages use it to resolve and validate the checkpoint.
- `Z_hat.pt` is tied to the VAE checkpoint and ordered feature contents. DQR training
  rebuilds stale latent caches; changed inputs require retraining downstream stages.
- `dist_list.pt` is tied to the models, latents, code, and runtime. Stale caches are
  recomputed. `--recompute-distances` forces a fresh calculation. Cache reuse also
  restores random-number state so subsequent computations match a cache miss.
- `crel_calibration_test_alpha_<alpha>.json` reports the calibrated radius and coverage.
- Reliability subdirectories contain `metric_value.json` and `reliability.json`,
  each with `avg`, `std`, and per-sample `values`, plus `best_z0s.pt`.

Retrain when checkpoint signatures no longer match the configuration, input data,
or implementation. Incompatible distance caches are recomputed automatically.

## Notes

Our code is based on [MQR](https://github.com/Shai128/mqr).

Results may vary across different GPUs, CUDA versions, cuDNN versions, and hardware-specific kernels.

## Citation

```bibtex
@InProceedings{gao2026conformal,
  title     = {Conformal Reliability: A New Evaluation Metric for Conditional Generation},
  author    = {Gao, Yachen and Sun, Xinwei and Wang, Yikai and Shi, Ye and Wang, Jingya and Feng, Jianfeng and Fu, Yanwei},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages     = {33632--33658},
  year      = {2026},
  volume    = {306},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v306/gao26aa.html}
}
```
