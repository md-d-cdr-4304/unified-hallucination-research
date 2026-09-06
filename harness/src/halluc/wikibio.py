"""Loader for the SelfCheckGPT annotated dataset (R1 reproduction).

potsawee/wiki_bio_gpt3_hallucination: 238 GPT-3 generated WikiBio passages,
each split into sentences with human factuality labels
(accurate / minor_inaccurate / major_inaccurate) and 20 stochastic samples
from the same prompt. Following Manakul et al. (2023), the non-factual
detection task treats {minor,major}_inaccurate as positive (label 1).
"""

from dataclasses import dataclass

from datasets import load_dataset

LABEL_MAP = {
    "accurate": 0,
    "minor_inaccurate": 1,
    "major_inaccurate": 1,
}


@dataclass
class Passage:
    idx: int
    sentences: list[str]
    labels: list[int]  # 1 = non-factual
    raw_labels: list[str]
    samples: list[str]  # stochastic re-generations of the same passage


def load_wikibio(n_passages: int | None = None, n_samples: int | None = None) -> list[Passage]:
    ds = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")
    passages = []
    for i, row in enumerate(ds):
        if n_passages is not None and i >= n_passages:
            break
        samples = row["gpt3_text_samples"]
        if n_samples is not None:
            samples = samples[:n_samples]
        passages.append(
            Passage(
                idx=i,
                sentences=row["gpt3_sentences"],
                labels=[LABEL_MAP[l] for l in row["annotation"]],
                raw_labels=row["annotation"],
                samples=samples,
            )
        )
    return passages
