import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_checkpoint(checkpoint_arg: str | None, output_dir: str) -> str:
    """
    --checkpoint 200        → <output_dir>/checkpoint-200
    --checkpoint /full/path → /full/path
    (omitted)               → <output_dir>  (final saved adapter)
    """
    if checkpoint_arg is None:
        return output_dir
    if checkpoint_arg.isdigit():
        return str(Path(output_dir) / f"checkpoint-{checkpoint_arg}")
    return checkpoint_arg


def build_prompt(tokenizer: AutoTokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         required=True,  help="Path to YAML config file")
    parser.add_argument("--checkpoint",     default=None,   help="Step number (e.g. 200), full path, or omit for final checkpoint")
    parser.add_argument("--system",         default=None,   help="NPC system prompt (overrides config default)")
    parser.add_argument("--prompt",         default=None,   help="Player message; omit to enter interactive mode")
    parser.add_argument("--max_new_tokens", type=int,   default=None, help="Override config max_new_tokens")
    parser.add_argument("--temperature",    type=float, default=None, help="Override config temperature")
    parser.add_argument("--top_p",          type=float, default=None, help="Override config top_p")
    args = parser.parse_args()

    cfg            = load_config(args.config)
    base_model     = cfg["model"]["base_model"]
    output_dir     = cfg["paths"]["output_dir"]
    infer_cfg      = cfg.get("inference", {})

    checkpoint_dir = resolve_checkpoint(args.checkpoint, output_dir)
    system         = args.system         or infer_cfg.get("default_system", "You are an NPC in a fantasy RPG.")
    max_new_tokens = args.max_new_tokens or infer_cfg.get("max_new_tokens", 200)
    temperature    = args.temperature    or infer_cfg.get("temperature", 0.9)
    top_p          = args.top_p          or infer_cfg.get("top_p", 0.9)

    print(f"Base model  : {base_model}")
    print(f"Checkpoint  : {checkpoint_dir}")
    print(f"System      : {system}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()

    if args.prompt:
        prompt   = build_prompt(tokenizer, system, args.prompt)
        response = generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p)
        print(response)
    else:
        print("Interactive mode — Ctrl+C or type 'quit' to exit.\n")
        while True:
            try:
                user_input = input("Player: ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            prompt   = build_prompt(tokenizer, system, user_input)
            response = generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p)
            print(f"NPC: {response}\n")


if __name__ == "__main__":
    main()
