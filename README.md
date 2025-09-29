The repository currently uses the code from the following [kaggle notebook](https://www.kaggle.com/code/hiranorm/offline-install-vllm-0-10-0-i-qwenemdding-llama)

Allows user to interact with dataset, training and inference scripts through the jigsaw_rules package

## Usage

### Installation

    uv pip install -e .

### Training

    !accelerate launch --config_file config/accelerate_config.yaml -m jigsaw_rules.train "instruct"

    !python -m jigsaw_rules.train <type>

### Inference

    !python -m jigsaw_rules.inference <type>

    !python -m jigsaw_rules.semantic

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
