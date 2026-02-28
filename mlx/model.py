"""
MistralClassifier — MLX classification wrapper for Apple Silicon.

Wraps the inner Mistral transformer (which returns hidden states) with a
linear classification head. Functionally equivalent to the CUDA
MistralForSequenceClassification but uses MLX's nn.Module system.

Key difference from CUDA: MLX uses `model.model` (the inner transformer)
rather than `model` (which includes the LM head). The inner transformer
returns hidden states directly — no need for output_hidden_states=True.

Used by: train.py, evaluate.py, inference.py
"""

import mlx.core as mx
import mlx.nn as nn


class MistralClassifier(nn.Module):
    """Classification wrapper around MLX Mistral transformer.

    Architecture:
        input_ids -> Mistral transformer (frozen + LoRA) -> hidden_states
        -> extract last real token -> dropout -> Linear(4096, num_classes)
        -> logits

    Args:
        transformer: The inner transformer model (model.model from mlx_lm),
            which returns hidden states of shape (batch, seq_len, hidden_size).
        num_classes: Number of output categories.
        hidden_size: Transformer hidden dimension (4096 for Mistral 7B).
        dropout_rate: Dropout between hidden state and classification head.
    """

    def __init__(
        self,
        transformer: nn.Module,
        num_classes: int,
        hidden_size: int = 4096,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def __call__(
        self,
        input_ids: mx.array,
        lengths: mx.array,
    ) -> mx.array:
        """Forward pass: transformer -> last-token extraction -> classification.

        Args:
            input_ids: (batch_size, seq_len) token IDs, right-padded.
            lengths: (batch_size,) actual sequence lengths (before padding).

        Returns:
            logits: (batch_size, num_classes) classification scores.
        """
        # The inner transformer returns hidden states directly
        # Shape: (batch_size, seq_len, hidden_size)
        hidden_states = self.transformer(input_ids)

        # Extract the last real (non-padding) token for each sequence.
        # lengths[i] is the count of real tokens, so lengths[i] - 1 is the
        # 0-based index of the last real token.
        # mx.take_along_axis selects one position per sequence from the
        # hidden states tensor.
        last_token_indices = (lengths - 1).reshape(-1, 1, 1)
        last_token_indices = mx.broadcast_to(
            last_token_indices,
            (input_ids.shape[0], 1, self.hidden_size),
        )
        last_hidden = mx.take_along_axis(hidden_states, last_token_indices, axis=1)
        last_hidden = last_hidden.squeeze(1)  # (batch_size, hidden_size)

        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)

        return logits
