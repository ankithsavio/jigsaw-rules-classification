## Jigsaw Rule Competition

The repository currently uses the code from the following [kaggle notebook](https://www.kaggle.com/code/hiranorm/offline-install-vllm-0-10-0-i-qwenemdding-llama)

Allows user to interact with dataset, training and inference scripts through the jigsaw_rules package

## Codebase Outline

    jigsaw-rules-classification/
    |-- README.md
    |-- config/
    |   |-- instruct_config.yaml
    |   `-- roberta_config.yaml
    |-- dataset/
    |-- kaggle_kernel/
    |   |-- jigsaw_rules.ipynb
    |   `-- kernel-metadata.json
    |-- notebooks/
    |-- pyproject.toml
    |-- requirements.txt
    |-- src/
    |   `-- jigsaw_rules/
    |       |-- __init__.py
    |       |-- configs.py
    |       |-- dataset.py
    |       |-- evaluate.py     # evaluation script
    |       |-- inference.py    # inference script
    |       |-- semantic.py     # semantic infernce script
    |       |-- train.py        # training script
    |       `-- utils.py
    `-- uv.lock

## Usage

### Installation

    git clone https://github.com/ankithsavio/jigsaw-rules-classification.git

    cd jigsaw-rules-classification

    uv pip install -e .

### Training

    !accelerate launch --config_file config/accelerate_config.yaml -m jigsaw_rules.train "instruct" # distributed data parallel

    !python -m jigsaw_rules.train <type> # data parallel

### Inference

    !python -m jigsaw_rules.inference <type>

    !python -m jigsaw_rules.semantic <type>

### Ensemble generation

    import pandas as pd
    import numpy as np
    from jigsaw_rules.configs import InstructConfig, ChatConfig, EmbeddingConfig

    q = pd.read_csv(InstructConfig.out_file)
    l = pd.read_csv(ChatConfig.out_file)
    m = pd.read_csv(EmbeddingConfig.out_file)


    rq = q['rule_violation'].rank(method='average') / (len(q)+1)
    rl = l['rule_violation'].rank(method='average') / (len(l)+1)
    rm = m['rule_violation'].rank(method='average') / (len(m)+1)


    blend = 0.5*rq + 0.3*rl + 0.2*rm
    q['rule_violation'] = blend
    q.to_csv('/kaggle/working/submission.csv', index=False)

## Docs

## Training

Supported models

- Qwen 2.5 0.5b : Uses SFTTrainer from trl - Instruct
- Roberta Base : Uses Trainer from transformers - RobertaBase
- Deberta Base : Uses Trainer from transformers - DebertaBase
- E5 Base V3 : Uses SentenceTransformerTrainer from sentence_transformers - E5Base
- Bge Base : Uses SentenceTransformerTrainer from sentence_transformers - BgeBase
- ModernBERT : Uses Trainer from transformers - ModernBERTBase

## Inference

Supported models

- Qwen 2.5 0.5b - InstructEngine
- Qwen 2.5 14b - ChatEngine
- Roberta Base - RobertaEngine
- Deberta Base - DebertaEngine
- ModernBERT Base - ModernBERTEngine

## Semantic Search

Supported models

- Qwen 3 0.6b - Qwen3EmbEngine
- E5 Base V3 - E5BaseEngine
- Bge Base - BgeBaseEngine
