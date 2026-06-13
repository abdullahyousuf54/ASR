from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import torch
import inspect
from datasets import DatasetDict, load_from_disk
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from .audio import load_audio_mono_16k
from .metrics import compute_asr_metrics
from .utils import current_gpu_memory_mb, get_logger, set_seed


LOGGER = get_logger(__name__)


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if labels.shape[1] > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def _load_processor_and_model(cfg: dict[str, Any]) -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    model_name = cfg["model"]["whisper_name"]
    processor = WhisperProcessor.from_pretrained(model_name, language=cfg["model"]["language"], task=cfg["model"]["task"])
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = cfg["model"]["language"]
    model.generation_config.task = cfg["model"]["task"]
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    return processor, model


def _maybe_apply_lora(model: WhisperForConditionalGeneration, cfg: dict[str, Any]):
    if cfg["training"].get("mode", "lora") != "lora":
        return model
    from peft import LoraConfig, get_peft_model

    lora_cfg = cfg["training"]["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def _prepare_features(dataset_dict: DatasetDict, processor: WhisperProcessor, cfg: dict[str, Any]) -> DatasetDict:
    label_col = cfg["text"].get("train_label", "transcript_indic")
    sampling_rate = int(cfg["audio"].get("sampling_rate", 16000))

    def prepare(example: dict) -> dict:
        audio = example["audio"]
        if isinstance(audio, str):
            start = float(example.get("start_seconds") or 0.0)
            end = example.get("end_seconds")
            duration = float(end) - start if end is not None else None
            audio = load_audio_mono_16k(audio, sampling_rate=sampling_rate, offset=start, duration=duration)
        example["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio.get("sampling_rate", sampling_rate),
        ).input_features[0]
        example["labels"] = processor.tokenizer(example[label_col]).input_ids
        return example

    remove_columns = dataset_dict[next(iter(dataset_dict.keys()))].column_names
    return dataset_dict.map(
        prepare,
        remove_columns=remove_columns,
        num_proc=1,
        desc="extract Whisper features",
    )


def train_whisper(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(cfg["project"].get("seed", 1337)))
    processed_dir = Path(cfg["paths"]["processed_dir"])
    dataset_dict: DatasetDict = load_from_disk(str(processed_dir))
    if "train" not in dataset_dict or "dev" not in dataset_dict:
        raise ValueError("Processed dataset must contain train and dev splits.")

    processor, model = _load_processor_and_model(cfg)
    model = _maybe_apply_lora(model, cfg)
    features = _prepare_features(dataset_dict, processor, cfg)

    train_cfg = cfg["training"]
    train_ds = features["train"]
    dev_ds = features["dev"]
    if train_cfg.get("max_train_samples"):
        train_ds = train_ds.select(range(min(int(train_cfg["max_train_samples"]), len(train_ds))))
    if train_cfg.get("max_dev_samples"):
        dev_ds = dev_ds.select(range(min(int(train_cfg["max_dev_samples"]), len(dev_ds))))

    data_collator = WhisperDataCollator(processor=processor)
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        result = compute_asr_metrics(label_str, pred_str, mode="indic").to_dict()
        result["wer_eval_lib"] = metric.compute(predictions=pred_str, references=label_str)
        return result

    training_kwargs = dict(
        output_dir=train_cfg.get("output_dir", "outputs/checkpoints/whisper-hi"),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 2)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        warmup_steps=int(train_cfg.get("warmup_steps", 500)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        fp16=bool(train_cfg.get("fp16", True)),
        bf16=bool(train_cfg.get("bf16", False)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        save_strategy="steps",
        eval_steps=int(train_cfg.get("eval_steps", 500)),
        save_steps=int(train_cfg.get("save_steps", 500)),
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        predict_with_generate=True,
        generation_max_length=int(cfg["model"].get("generation", {}).get("max_new_tokens", 225)),
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=train_cfg.get("report_to", "tensorboard"),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 2)),
        save_total_limit=3,
    )
    strategy_name = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_kwargs[strategy_name] = "steps"
    args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(train_cfg.get("early_stopping_patience", 3)))],
    )

    train_result = trainer.train()
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    metrics = dict(train_result.metrics)
    metrics["max_gpu_memory_mb"] = current_gpu_memory_mb()
    metrics = {k: v for k, v in metrics.items() if v is not None}
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    LOGGER.info("Training complete. Best checkpoint: %s", trainer.state.best_model_checkpoint)
    return {
        "output_dir": args.output_dir,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "metrics": metrics,
    }
