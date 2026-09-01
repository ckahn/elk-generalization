from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from elk_generalization.datasets.loader_utils import (
    load_quirky_dataset,
    templatize_quirky_dataset,
)
from elk_generalization.utils import (
    DEVICE_CHOICES,
    DTYPE_CHOICES,
    resolve_device,
    resolve_dtype,
)

warned_about_choices = set()


def required_output_files(skip_contrast_pairs: bool) -> tuple[str, ...]:
    files = (
        "hiddens.pt",
        "labels.pt",
        "alice_labels.pt",
        "bob_labels.pt",
        "lm_log_odds.pt",
    )
    if skip_contrast_pairs:
        return files
    return files + ("ccs_hiddens.pt",)


def allocate_activation_buffers(
    num_examples: int,
    num_layers: int,
    hidden_size: int,
    dtype: torch.dtype,
    include_contrast_pairs: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None, torch.Tensor]:
    """Allocate persistent extraction outputs on CPU to conserve accelerator memory."""
    buffers = [
        torch.full([num_examples, hidden_size], torch.nan, dtype=dtype)
        for _ in range(num_layers)
    ]
    ccs_buffers = None
    if include_contrast_pairs:
        ccs_buffers = [
            torch.full([num_examples, 2, hidden_size], torch.nan, dtype=dtype)
            for _ in range(num_layers)
        ]
    log_odds = torch.full([num_examples], torch.nan, dtype=dtype)
    return buffers, ccs_buffers, log_odds


def encode_choice(text, tokenizer):
    global warned_about_choices

    c_ids = tokenizer.encode(text, add_special_tokens=False)

    # some tokenizers split off the leading whitespace character
    if tokenizer.decode(c_ids[0]).strip() == "":
        c_ids = c_ids[1:]
        assert c_ids == tokenizer.encode(text.lstrip(), add_special_tokens=False)

    c_ids = tuple(c_ids)
    if len(c_ids) != 1 and c_ids not in warned_about_choices:
        warned_about_choices.add(c_ids)
        print(f"Choice should be one token: {c_ids} -> {tokenizer.decode(c_ids)}")
    return c_ids[0]


if __name__ == "__main__":
    parser = ArgumentParser(description="Process and save model hidden states.")
    parser.add_argument("--model", type=str, help="Name of the HuggingFace model")
    parser.add_argument("--dataset", type=str, help="Name of the HuggingFace dataset")
    parser.add_argument(
        "--character",
        default="none",
        choices=["Alice", "Bob", "none"],
        help="Character in the context",
    )
    parser.add_argument(
        "--difficulty",
        default="none",
        choices=["easy", "hard", "none"],
        help="Difficulty of the examples",
    )
    parser.add_argument(
        "--standardize-templates",
        action="store_true",
        help="Standardize the templates",
    )
    parser.add_argument(
        "--templatization-method",
        default="random",
        choices=["random", "first", "all"],
        help="Method to use for standardizing the templates",
    )
    parser.add_argument("--save-path", type=Path, help="Path to save the hidden states")
    parser.add_argument("--seed", type=int, default=633, help="Random seed")
    parser.add_argument(
        "--max-examples",
        type=int,
        nargs="+",
        help="Max examples per split",
        default=[1000, 1000],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Execution device. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default="auto",
        help="Model dtype. 'auto' selects a suitable dtype for the device.",
    )
    parser.add_argument(
        "--skip-contrast-pairs",
        action="store_true",
        help=(
            "Skip choice-conditioned activation extraction. Suitable for "
            "diff-in-means, logistic regression, and LDA; incompatible with "
            "CCS, CRC, and on-pair probes."
        ),
    )
    args = parser.parse_args()

    required_files = required_output_files(args.skip_contrast_pairs)

    # check if all the results already exist
    if all(
        all((args.save_path / split / filename).exists() for filename in required_files)
        for split in args.splits
    ):
        print(f"Hiddens already exist at {args.save_path}")
        exit()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f"Loading {args.model} on {device} with dtype {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    assert len(args.max_examples) == len(args.splits)
    for split, max_examples in zip(args.splits, args.max_examples):
        root = args.save_path / split
        root.mkdir(parents=True, exist_ok=True)
        # skip if the results for this split already exist
        if all((root / filename).exists() for filename in required_files):
            print(f"Skipping because all required outputs already exist in '{root}'")
            continue

        print(f"Processing '{split}' split...")

        dataset = templatize_quirky_dataset(
            load_quirky_dataset(
                args.dataset,
                character=args.character,
                max_difficulty_quantile=0.25 if args.difficulty == "easy" else 1.0,
                min_difficulty_quantile=0.75 if args.difficulty == "hard" else 0.0,
                split=split,
            ).shuffle(seed=args.seed),
            ds_name=args.dataset,
            standardize_templates=args.standardize_templates,
            method=args.templatization_method,
        )
        assert isinstance(dataset, Dataset)
        try:
            dataset = dataset.select(range(max_examples))
        except IndexError:
            print(
                f"Using all {len(dataset)} examples for {args.dataset}/{split} "
                f"instead of {max_examples}"
            )

        buffers, ccs_buffers, log_odds = allocate_activation_buffers(
            num_examples=len(dataset),
            num_layers=model.config.num_hidden_layers,
            hidden_size=model.config.hidden_size,
            dtype=model.dtype,
            include_contrast_pairs=not args.skip_contrast_pairs,
        )

        for i, record in tqdm(enumerate(dataset), total=len(dataset)):
            assert isinstance(record, dict)

            prompt = tokenizer.encode(record["statement"])
            choice_toks = [
                encode_choice(record["choices"][0], tokenizer),
                encode_choice(record["choices"][1], tokenizer),
            ]

            with torch.inference_mode():
                outputs = model(
                    torch.as_tensor([prompt], device=device),
                    output_hidden_states=True,
                    use_cache=not args.skip_contrast_pairs,
                )

                if ccs_buffers is not None:
                    # FOR CCS: Gather hidden states for each of the two choices
                    ccs_outputs = [
                        model(
                            torch.as_tensor([[choice]], device=device),
                            output_hidden_states=True,
                            past_key_values=outputs.past_key_values,
                        ).hidden_states[1:]
                        for choice in choice_toks
                    ]
                    for j, (state1, state2) in enumerate(zip(*ccs_outputs)):
                        ccs_buffers[j][i, 0].copy_(
                            state1.squeeze().detach().to("cpu")
                        )
                        ccs_buffers[j][i, 1].copy_(
                            state2.squeeze().detach().to("cpu")
                        )

                logit1, logit2 = outputs.logits[0, -1, choice_toks]
                log_odds[i] = (logit2 - logit1).detach().to("cpu")

                # Extract hidden states of the last token in each layer
                for j, state in enumerate(outputs.hidden_states[1:]):
                    buffers[j][i].copy_(state[0, -1, :].detach().to("cpu"))

        # Sanity check
        assert all(buffer.isfinite().all() for buffer in buffers)
        if ccs_buffers is not None:
            assert all(buffer.isfinite().all() for buffer in ccs_buffers)
        assert log_odds.isfinite().all()

        # Save results to disk for later
        labels = torch.as_tensor(dataset["label"], dtype=torch.int32)
        alice_labels = torch.as_tensor(dataset["alice_label"], dtype=torch.int32)
        bob_labels = torch.as_tensor(dataset["bob_label"], dtype=torch.int32)
        torch.save(buffers, root / "hiddens.pt")
        if ccs_buffers is not None:
            torch.save(ccs_buffers, root / "ccs_hiddens.pt")
        torch.save(labels, root / "labels.pt")
        torch.save(alice_labels, root / "alice_labels.pt")
        torch.save(bob_labels, root / "bob_labels.pt")
        torch.save(log_odds, root / "lm_log_odds.pt")
