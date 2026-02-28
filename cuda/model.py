"""
MistralForSequenceClassification — PyTorch classification wrapper.

This module wraps a PEFT-adapted Mistral model with a classification head.
The key insight: causal LMs process tokens left-to-right, so the *last*
non-padding token has attended to the entire input and holds the richest
representation. We extract that hidden state and project it through a
linear layer to produce class logits.

Also provides load_trained_model() — the single source of truth for
reconstructing a trained model from a checkpoint. Used by evaluate.py
and inference.py.

Used by: train.py, evaluate.py, inference.py
"""

import logging
from pathlib import Path

import safetensors.torch
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM

import config

log = logging.getLogger(__name__)


class MistralForSequenceClassification(nn.Module):
    """Wraps a PEFT Mistral model with dropout + linear classification head.

    Architecture:
        input_ids -> Mistral (frozen + LoRA) -> hidden_states
        -> extract last non-padding token -> dropout -> Linear(4096, num_classes)
        -> logits

    The last-token extraction uses attention_mask to find the correct position,
    handling variable-length sequences within a dynamically-padded batch.
    """

    def __init__(self, base_model: nn.Module, num_labels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.hidden_size = base_model.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        # Instantiate once in __init__ rather than per forward pass — CE loss
        # has no learnable parameters, but re-creating it each call is wasteful
        self.loss_fn = nn.CrossEntropyLoss()

        # Small std initialization keeps initial logits near zero, so the
        # model starts with roughly uniform class probabilities
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        # Move the classification head to the same device as the base model.
        # The base model is on GPU via device_map="auto", but nn.Linear
        # defaults to CPU — without this, the forward pass would crash with
        # a device mismatch error when the classifier receives GPU tensors.
        device = next(base_model.parameters()).device
        self.classifier = self.classifier.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass: base model -> last-token extraction -> classification.

        Args:
            input_ids: (batch_size, seq_len) token IDs.
            attention_mask: (batch_size, seq_len) binary mask, 1 for real tokens.
            labels: (batch_size,) integer class labels. If provided, computes loss.

        Returns:
            Dict with 'loss' (if labels given) and 'logits' (batch_size, num_classes).
        """
        # output_hidden_states=True gives us access to the final layer's
        # hidden states, which we need for classification
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Shape: (batch_size, seq_len, hidden_size)
        hidden_states = outputs.hidden_states[-1]

        if attention_mask is not None:
            # For each sequence, find the index of the last real (non-padding)
            # token. attention_mask.sum(dim=1) gives the count of real tokens,
            # subtract 1 for 0-based indexing.
            # Example: mask=[1,1,1,0,0] -> sum=3 -> last_token_idx=2
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(
                hidden_states.shape[0], device=hidden_states.device
            )
            last_hidden = hidden_states[batch_indices, sequence_lengths]
        else:
            # No padding — just use the final position
            last_hidden = hidden_states[:, -1, :]

        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Checkpoint Loading — single source of truth for evaluate.py & inference.py
# ---------------------------------------------------------------------------

def load_trained_model(
    num_classes: int,
    checkpoint_dir: Path,
    device: str,
) -> MistralForSequenceClassification:
    """Reconstruct the model and load saved LoRA adapters + classification head.

    Steps:
        1. Load base Mistral 7B in float16
        2. Re-apply LoRA structure (empty adapters)
        3. Load trained LoRA weights from checkpoint
        4. Wrap with classification head
        5. Load trained head weights from checkpoint

    Args:
        num_classes: Number of output categories (5 for Phase 1, 20 for Phase 2).
        checkpoint_dir: Path to the checkpoint directory containing
            lora_adapters/ and classifier_head.pt.
        device: Target device ("cuda" or "cpu").

    Returns:
        Model in eval mode, ready for inference.
    """
    log.info(f"Loading base model: {config.MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias=config.LORA_BIAS,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    base_model = get_peft_model(base_model, lora_config)

    # Load trained LoRA weights
    lora_path = checkpoint_dir / "lora_adapters"
    log.info(f"Loading LoRA adapters from {lora_path}")
    adapter_weights = safetensors.torch.load_file(
        str(lora_path / "adapter_model.safetensors")
    )
    set_peft_model_state_dict(base_model, adapter_weights)

    # Build classification wrapper and load head weights
    model = MistralForSequenceClassification(
        base_model, num_labels=num_classes, dropout_rate=config.CLASSIFIER_DROPOUT,
    )

    head_path = checkpoint_dir / "classifier_head.pt"
    log.info(f"Loading classification head from {head_path}")
    head_state = torch.load(head_path, map_location="cpu", weights_only=True)
    model.classifier.weight.data = head_state["classifier.weight"].to(device)
    model.classifier.bias.data = head_state["classifier.bias"].to(device)

    model.eval()
    return model
