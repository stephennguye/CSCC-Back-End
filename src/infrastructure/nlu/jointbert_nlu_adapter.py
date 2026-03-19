"""JointBERT NLU adapter — real PhoBERT model inference.

Loads the trained JointBERT (JointIDSF + PhoBERT) checkpoint and performs
joint intent classification + slot filling on Vietnamese text.

Falls back to the keyword-based PhoBERTNLUAdapter if the model fails to load
(e.g., missing checkpoint, missing torch/transformers).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import structlog

from src.domain.entities.dialogue_state import NLUResult, SlotValue

logger = structlog.get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

_MODELS_DIR = Path(
    os.environ.get(
        "JOINTBERT_MODEL_DIR",
        str(Path(__file__).resolve().parent.parent.parent.parent / "models" / "jointbert"),
    )
)
_CHECKPOINT_NAME = os.environ.get("JOINTBERT_CHECKPOINT", "best_jointbert.pt")
_MODEL_NAME = os.environ.get("JOINTBERT_HF_MODEL", "vinai/phobert-base-v2")
_MAX_SEQ_LEN = int(os.environ.get("JOINTBERT_MAX_SEQ_LEN", "128"))


class JointBERTNLUAdapter:
    """NLU adapter backed by a trained JointBERT + PhoBERT model.

    Implements the ``NLUPort`` Protocol.

    On construction, lazily loads the model on first ``understand()`` call
    to avoid blocking the application startup for too long.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: Any | None = None
        self._id2intent: dict[int, str] = {}
        self._id2slot: dict[int, str] = {}
        self._loaded = False
        self._load_failed = False
        self._fallback: Any | None = None  # lazy keyword adapter

    # ------------------------------------------------------------------ #
    # Model loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_model(self) -> bool:
        """Synchronously load the JointBERT model + label mappings.

        Returns True on success, False if any dependency is missing.
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            logger.error("jointbert_missing_deps", detail="torch or transformers not installed")
            return False

        checkpoint_path = _MODELS_DIR / _CHECKPOINT_NAME
        intent2id_path = _MODELS_DIR / "intent2id.json"
        slot2id_path = _MODELS_DIR / "slot2id.json"

        if not checkpoint_path.exists():
            logger.error("jointbert_checkpoint_not_found", path=str(checkpoint_path))
            return False

        # Load label mappings
        try:
            with open(intent2id_path, encoding="utf-8") as f:
                intent2id: dict[str, int] = json.load(f)
            self._id2intent = {v: k for k, v in intent2id.items()}

            with open(slot2id_path, encoding="utf-8") as f:
                slot2id: dict[str, int] = json.load(f)
            self._id2slot = {v: k for k, v in slot2id.items()}
        except Exception as exc:
            logger.error("jointbert_label_load_failed", error=str(exc))
            return False

        # Load tokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        except Exception as exc:
            logger.error("jointbert_tokenizer_load_failed", error=str(exc))
            return False

        # Determine device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location=self._device, weights_only=False)
            model_config = checkpoint.get("model_config", {})
            num_intents = model_config.get("num_intents", len(self._id2intent))
            num_slots = model_config.get("num_slots", len(self._id2slot))
            use_crf = model_config.get("use_crf", False)
        except Exception as exc:
            logger.error("jointbert_checkpoint_load_failed", error=str(exc))
            return False

        # Build model architecture
        try:
            from torch import nn

            class _JointBERTModel(nn.Module):
                def __init__(self, encoder: nn.Module, n_intents: int, n_slots: int, dropout: float = 0.1):
                    super().__init__()
                    self.encoder = encoder
                    hidden = encoder.config.hidden_size
                    self.dropout = nn.Dropout(dropout)
                    self.intent_classifier = nn.Linear(hidden, n_intents)
                    self.slot_classifier = nn.Linear(hidden, n_slots)
                    self.num_intents = n_intents
                    self.num_slots = n_slots

                def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
                    out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                    seq = self.dropout(out.last_hidden_state)
                    cls = self.dropout(out.last_hidden_state[:, 0, :])
                    return {
                        "intent_logits": self.intent_classifier(cls),
                        "slot_logits": self.slot_classifier(seq),
                    }

            encoder = AutoModel.from_pretrained(_MODEL_NAME)
            model = _JointBERTModel(encoder, num_intents, num_slots)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.to(self._device)
            model.eval()
            self._model = model
        except Exception as exc:
            logger.error("jointbert_model_build_failed", error=str(exc))
            return False

        logger.info(
            "jointbert_loaded",
            device=str(self._device),
            intents=num_intents,
            slots=num_slots,
            use_crf=use_crf,
            checkpoint=str(checkpoint_path),
        )
        return True

    def _ensure_loaded(self) -> bool:
        """Ensure the model is loaded, attempting once if not yet tried."""
        if self._loaded:
            return True
        if self._load_failed:
            return False
        ok = self._load_model()
        if ok:
            self._loaded = True
        else:
            self._load_failed = True
        return ok

    def _get_fallback(self) -> Any:
        if self._fallback is None:
            from src.infrastructure.nlu.phobert_nlu_adapter import PhoBERTNLUAdapter
            self._fallback = PhoBERTNLUAdapter()
        return self._fallback

    # ------------------------------------------------------------------ #
    # NLUPort implementation                                               #
    # ------------------------------------------------------------------ #

    async def understand(self, text: str) -> NLUResult:
        """Parse Vietnamese text into intent and slots using JointBERT."""
        if not self._ensure_loaded():
            logger.warning("jointbert_fallback_to_keyword")
            return await self._get_fallback().understand(text)

        # Run inference in a thread pool to avoid blocking the event loop
        return await asyncio.get_event_loop().run_in_executor(
            None, self._predict, text,
        )

    def _predict(self, text: str) -> NLUResult:
        """Synchronous inference — called from executor."""
        import torch

        encoded = self._encode(text)
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self._device)
        attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=self._device)

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)

            # Intent
            intent_probs = torch.softmax(outputs["intent_logits"], dim=-1)
            intent_idx = torch.argmax(intent_probs, dim=-1).item()
            intent_conf = intent_probs[0, intent_idx].item()
            intent_label = self._id2intent.get(intent_idx, "UNK")

            # Slots
            slot_preds = torch.argmax(outputs["slot_logits"], dim=-1)[0].cpu().numpy()

        # Extract BIO slot values
        slots = self._extract_slots(encoded, slot_preds)

        return NLUResult(
            intent=intent_label,
            intent_confidence=intent_conf,
            slots=slots,
            raw_text=text,
        )

    # ------------------------------------------------------------------ #
    # Tokenization & slot extraction                                       #
    # ------------------------------------------------------------------ #

    def _encode(self, text: str) -> dict[str, Any]:
        """Tokenize text with subword tracking."""
        words = text.split()
        all_ids: list[int] = []
        word_ids: list[int] = []

        for widx, word in enumerate(words):
            sub_tokens = self._tokenizer.tokenize(word)
            if not sub_tokens:
                sub_tokens = [self._tokenizer.unk_token]
            sub_ids = self._tokenizer.convert_tokens_to_ids(sub_tokens)
            all_ids.extend(sub_ids)
            word_ids.extend([widx] * len(sub_ids))

        # Truncate
        max_tok = _MAX_SEQ_LEN - 2
        all_ids = all_ids[:max_tok]
        word_ids = word_ids[:max_tok]

        cls_id = self._tokenizer.cls_token_id
        sep_id = self._tokenizer.sep_token_id
        pad_id = self._tokenizer.pad_token_id

        input_ids = [cls_id] + all_ids + [sep_id]
        word_ids = [-1] + word_ids + [-1]
        attn_mask = [1] * len(input_ids)

        pad_len = _MAX_SEQ_LEN - len(input_ids)
        input_ids += [pad_id] * pad_len
        word_ids += [-1] * pad_len
        attn_mask += [0] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "word_ids": word_ids,
            "words": words,
        }

    def _extract_slots(self, encoded: dict[str, Any], slot_preds: Any) -> list[SlotValue]:
        """Extract SlotValue list from BIO predictions."""
        words: list[str] = encoded["words"]
        word_ids: list[int] = encoded["word_ids"]

        # Map first subword → slot label per word
        word_labels: dict[int, str] = {}
        for tok_idx, (widx, sid) in enumerate(zip(word_ids, slot_preds)):
            if widx >= 0 and widx not in word_labels:
                word_labels[widx] = self._id2slot.get(int(sid), "O")

        # Group BIO tags
        slots: list[SlotValue] = []
        cur_type: str | None = None
        cur_value: list[str] = []

        for widx, word in enumerate(words):
            label = word_labels.get(widx, "O")
            if label.startswith("B-"):
                if cur_type and cur_value:
                    slots.append(SlotValue(name=cur_type, value=" ".join(cur_value), confidence=0.90))
                cur_type = label[2:]
                cur_value = [word]
            elif label.startswith("I-") and cur_type == label[2:]:
                cur_value.append(word)
            else:
                if cur_type and cur_value:
                    slots.append(SlotValue(name=cur_type, value=" ".join(cur_value), confidence=0.90))
                cur_type = None
                cur_value = []

        if cur_type and cur_value:
            slots.append(SlotValue(name=cur_type, value=" ".join(cur_value), confidence=0.90))

        return slots
