# CsMr-Grasp

**CsMr-Grasp: Classifier-Steered and Multiscale-Refined Intent-based Grasp Generation**

**Zhenhua Tang**, **Yanbin Hao**, **Richang Hong**, and **Chi-Man Pun** (corresponding author)

- Zhenhua Tang and Chi-Man Pun: Faculty of Science and Technology, University of Macau, Macau SAR, China.
- Yanbin Hao and Richang Hong: School of Computer and Information Engineering, Hefei University of Technology, Anhui, China.

Research code for the manuscript submitted to *IEEE Transactions on Image Processing*. The manuscript is under review; this repository does not claim publication or acceptance.

## Method

CsMr-Grasp generates 3D hand grasps conditioned on object geometry and a requested manipulation intent such as **use** or **hold**. It has two stages:

1. A classifier-steered conditional VAE generates a coarse grasp. A multi-scale encoder and decoder model geometric context, while classifier-steered latent adjustment guides the requested intent.
2. A multi-scale refinement network uses residual KAN blocks to improve hand-object geometry and contact.

The repository also contains OakInk-Shape baseline and comparison code inherited from the research codebase. The CsMr-specific coarse and refinement implementations are `lib/models/grabnet/coarse_intent_cs.py` and `lib/models/grabnet/refine_msrn.py`.

## Repository layout

| Path | Contents |
| --- | --- |
| `lib/models/grabnet/` | CsMr and baseline generation models |
| `lib/datasets/` | OakInk-Shape dataset loaders |
| `scripts/` | Training, generation, and evaluation scripts |
| `config/oishape_bm/intent/test_CSMR_OIShape.yml` | CsMr evaluation configuration |
| `config/oishape_bm/intent/test_GrabIntentNet_OIShape_trigger.yml` | Legacy RAGG comparison configuration |

## Setup and evaluation

Create the environment from `environment.yml`, then install `requirements.txt` and `requirements@git.txt`. Install the [OakInk toolkit](https://github.com/oakink/OakInk) and provide the OakInk-Shape data, MANO assets, and pretrained weights. Data, MANO assets, checkpoints, classifier weights, and generated results are **not** distributed in this repository.

Place or link the inputs at `data/`, `assets/mano_v1_2/`, and `checkpoints/`. Set the two `PRETRAINED` paths in `config/oishape_bm/intent/test_CSMR_OIShape.yml` to your CsMr coarse and refinement checkpoints. Some older scripts and configurations contain machine-specific paths and require adjustment for another environment.

The current public repository is a research-code snapshot. The reviewer-response experiments, including independent semantic-classifier evaluation, are being organized separately and should not be inferred from the baseline configuration or from the manuscript's original classifier accuracy.

## Citation and provenance

If you use CsMr-Grasp, cite the manuscript by **Zhenhua Tang, Yanbin Hao, Richang Hong, and Chi-Man Pun**, *“CsMr-Grasp: Classifier-Steered and Multiscale-Refined Intent-based Grasp Generation.”* Publication details will be added when available.

This work builds on the OakInk-Shape grasp-generation codebase. Please also cite the relevant [OakInk paper](https://arxiv.org/abs/2203.15709) and the dataset or baseline methods you use. The OakInk project authors listed in the upstream README are credited for that foundational work; they are not the author list of the CsMr-Grasp manuscript.
