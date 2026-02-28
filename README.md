# Beyond the Known: An Unknown-Aware Large Language Model for Open-Set Text Classification

| [Overview](#overview)
| [Installation](#installation)
| [Dataset](#dataset)
| [Folder Structure](#folder-Structure)
| [How to Run Model](#how-to-run-model)

## Overview
Official code for article "[Beyond the Known: An Unknown-Aware Large Language Model for Open-Set Text Classification (ICLR-26)](https://openreview.net/forum?id=BqLGlQF46f)".

Open-set text classification (OSTC) requires models to correctly classify in-distribution (ID) samples while reliably rejecting out-of-distribution (OOD) inputs—an essential capability for real-world NLP systems. 
However, existing approaches largely follow the post-hoc OOD detection paradigm after a closed-world training, optimizing predictive distributions solely over known label spaces and thus producing overconfident and biased predictions on OOD inputs.
In this work, we present **UnLLM**, an Unknown-aware Large Language Model for OSTC. Instead of fixing classification to the entire known label space, we reformulate it into a subset-conditioned classification task: the LLM is prompted with sampled subsets of known labels, and any instance outside the candidate set is explicitly assigned as an “unknown” class. This reformulation transforms OOD detection from a post-hoc procedure into an intrinsic modeling capability.
Grounded in this formulation, the modeling of the “unknown” is further systematically realized through a unified \textbf{representation–probability–inference optimization, which progressively strengthens the model’s capacity to capture open-set risk.
Extensive experiments across six benchmarks show that **UnLLM** consistently outperforms state-of-the-art (SOTA) baselines.
Code and datasets are available at: https://github.com/cx9941/UnLLM.

## Installation

Create a python 3.12 environment and install dependencies:

```
  conda create -n python3.12 UnLLM
  source activate UnLLM
```

Install library

```
  pip install -r requirements.txt
```

Note that pytorch >= 2.3.1

## Dataset
The datasets used in all experiments are derived from previously published scientific papers.
In this repository, only the BANKING dataset is displayed for the time being. Other datasets can be obtained by going to the relevant links. We provide the links from which we obtain the datasets.

[BANKING](https://github.com/fanolabs/NID_ACLARR2022), [CLINC](https://github.com/fanolabs/NID_ACLARR2022), [StackOverflow](https://github.com/fanolabs/NID_ACLARR2022), [THUCNews](http://thuctc.thunlp.org), [NewsGroup](https://github.com/leishu02/EMNLP2017_DOC?tab=readme-ov-file), [Reviews](https://github.com/leishu02/EMNLP2017_DOC?tab=readme-ov-file)


## Folder Structure

```tex
└── data
    └── banking                   # BANKING Dataset
        ├── banking_0.25          # 0.25 known ratio dataset
        ├── banking_0.5           # 0.5 known ratio dataset
        ├── banking_0.75          # 0.75 known ratio dataset
        └── origin_data           # 1.0 known ratio dataset
    ├── clinc                     # CLINC Dataset
    ├── stackoverflow             # StackOverflow Dataset
    ├── ele                       # Reviews Dataset
    ├── news                      # Newsgroups Dataset
    └── thucnews                  # THUCNews Dataset
└── models
    ├── modeling_llama.py         # The model
    └── utlis.py                  # The tools for modeling
└── scripts                       # The running command
    └── run.sh                    # The pipeline training and inference of UnLLM
└── utils
    ├── bert_embedding.py         # For retrival in stage3: analogy-augmented self-reflection
    ├── dataset_utils.py          # The tools for dataset and dataloader
    ├── interfere.py              # For stage2: OOD parameter cabliation
    ├── metric.py                 # Compute the metrics incling ACC, F1, K-F1 and N-F1
    └── utils.py                  # The tools
└── configs
    ├── yaml                      # accelerate configs
    └── args.json                 # Default parameters
├── args.py                       # Set the parameters
├── main.py                       # For training and test
├── trainer.py                    # Train and Test functions
├── README.md                     # This document
└── requirements.txt              # The dependencies
```

## How to Run Model

Train and test the UnLLM:

```
sh scripts/run.sh
```


## Citation

If you find our work is useful for your research, please consider citing:

```
@inproceedings{
chen2026beyond,
title={Beyond the Known: An Unknown-Aware Large Language Model for Open-Set Text Classification},
author={Xi Chen and Chuan Qin and Ziqi Wang and Shasha Hu and Chao Wang and Hengshu Zhu and Hui Xiong},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=BqLGlQF46f}
}
```

## License

This project is licensed under the MIT License.
