#!/usr/bin/env python3
"""Export sshleifer/tiny-gpt2 as a fixed-shape prefill ONNX fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "sshleifer/tiny-gpt2"
SEQUENCE_LENGTH = 16
PROMPT = "Once upon a time"


class FixedPrefill(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, attention_mask: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.sequence_length = int(attention_mask.shape[1])
        self.hidden_size = int(model.config.n_embd)
        self.register_buffer("fixed_attention_mask", attention_mask)

    def forward(self, flat_input_embeddings: torch.Tensor) -> tuple[torch.Tensor, ...]:
        input_embeddings = flat_input_embeddings.reshape(
            1, self.sequence_length, self.hidden_size
        )
        result = self.model(
            inputs_embeds=input_embeddings,
            attention_mask=self.fixed_attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return (
            result.logits,
            result.hidden_states[0],
            result.hidden_states[1],
            result.hidden_states[2],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--prompt", default=PROMPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.eval()
    model.config.use_cache = False
    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.sequence_length,
    )
    input_ids = encoded["input_ids"].to(torch.int64)
    attention_mask = encoded["attention_mask"].to(torch.int64)
    valid_tokens = int(attention_mask.sum().item())
    position_ids = torch.arange(args.sequence_length, dtype=torch.int64).unsqueeze(0)
    input_embeddings = (
        model.transformer.wte(input_ids) + model.transformer.wpe(position_ids)
    ).detach()
    with torch.no_grad():
        model.transformer.wpe.weight.zero_()
    wrapper = FixedPrefill(model, attention_mask).eval()
    flat_input_embeddings = input_embeddings.reshape(1, -1)

    with torch.no_grad():
        torch_outputs = tuple(value.cpu().numpy() for value in wrapper(flat_input_embeddings))
    torch_logits = torch_outputs[0]

    onnx_path = args.output_dir / "tiny_gpt2_prefill_s16.onnx"
    torch.onnx.export(
        wrapper,
        (flat_input_embeddings,),
        onnx_path,
        input_names=["input_embeddings"],
        output_names=["logits", "hidden_embedding", "hidden_block0", "hidden_block1"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    onnx_model = onnx.load(onnx_path)
    for node in onnx_model.graph.node:
        if "hidden_embedding" in node.output:
            node.op_type = "Identity"
            del node.input[1:]
            break
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_path)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_embeddings_fp32 = flat_input_embeddings.cpu().numpy().astype(np.float32)
    ort_outputs = session.run(None, {"input_embeddings": input_embeddings_fp32})
    output_errors = [
        float(np.max(np.abs(torch_value - ort_value)))
        for torch_value, ort_value in zip(torch_outputs, ort_outputs)
    ]
    max_abs_error = max(output_errors)
    ort_logits = ort_outputs[0]
    prediction_index = valid_tokens - 1
    torch_token = int(np.argmax(torch_logits[0, prediction_index]))
    ort_token = int(np.argmax(ort_logits[0, prediction_index]))
    if torch_token != ort_token or max_abs_error > 1.0e-4:
        raise RuntimeError(
            f"ONNX mismatch: max_abs_error={max_abs_error}, "
            f"torch_token={torch_token}, ort_token={ort_token}"
        )

    input_ids.numpy().astype(np.int32).tofile(args.output_dir / "tiny_gpt2_input_ids.bin")
    input_embeddings_fp32.tofile(args.output_dir / "tiny_gpt2_input_embeddings.bin")
    torch_logits.astype(np.float32).tofile(args.output_dir / "tiny_gpt2_logits_fp32.bin")
    tokenizer.save_pretrained(args.output_dir / "tokenizer")

    metadata = {
        "source_model": args.model_id,
        "prompt": args.prompt,
        "input_shape": list(input_embeddings_fp32.shape),
        "input_dtype": "float32",
        "token_ids": input_ids.numpy().astype(np.int32).tolist(),
        "valid_tokens": valid_tokens,
        "prediction_index": prediction_index,
        "output_shape": list(torch_logits.shape),
        "output_dtype": "float32",
        "diagnostic_output_shapes": {
            name: list(value.shape)
            for name, value in zip(
                ["hidden_embedding", "hidden_block0", "hidden_block1"],
                torch_outputs[1:]
            )
        },
        "next_token_id": torch_token,
        "next_token_text": tokenizer.decode([torch_token]),
        "pytorch_onnx_max_abs_error": max_abs_error,
        "pytorch_onnx_output_errors": output_errors,
        "onnx_bytes": onnx_path.stat().st_size,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
