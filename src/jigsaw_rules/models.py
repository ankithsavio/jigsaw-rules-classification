import os
import torch
import torch.nn as nn
from transformers import (
    DebertaV2Config,
    DebertaV2ForSequenceClassification,
    PreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F

class DebertaWithFusion(PreTrainedModel):
    """
    Wraps DeBERTa (2-class) + small fusion head that consumes:
      - base logits (shape: [B, 2])
      - extra_feats (shape: [B, 2])  # your two probabilities
    Returns fused 2-class logits.
    """
    config_class = DebertaV2Config

    def __init__(
        self,
        config: DebertaV2Config,
        base_model_name: str,
        freeze_base: bool = False,
        extra_size: int = 2,
        hidden_size: int = 32
    ):
        super().__init__(config)
        # base classifier produces 2 logits already
        self.base = DebertaV2ForSequenceClassification.from_pretrained(
            base_model_name, num_labels=config.num_labels
        )

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        # fusion MLP: [2 (base logits) + extra_size] -> hidden -> 2 logits
        self.fc1 = nn.Linear(config.num_labels + extra_size, hidden_size)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.fc2 = nn.Linear(hidden_size, config.num_labels)

        # save init weights
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        extra_feats=None,   # <-- expect shape [B, 2]
        labels=None,
        **kwargs
    ):
        if extra_feats is None:
            raise ValueError("`extra_feats` (shape [batch, 2]) must be provided.")

        base_out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,
            return_dict=True,
        )

        base_logits = base_out.logits  # [B, 2]
        x = torch.cat([base_logits, extra_feats], dim=1)  # [B, 4]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        fused_logits = self.fc2(x)  # [B, 2]

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(fused_logits.view(-1, self.config.num_labels),
                                         labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=fused_logits,
            hidden_states=base_out.hidden_states,
            attentions=base_out.attentions,
        )
