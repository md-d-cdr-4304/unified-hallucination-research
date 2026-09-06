"""SelfCheckGPT-Prompt scorer (Manakul et al., EMNLP 2023, Sec. 3.5).

For each sentence and each stochastic sample of the same passage, ask a judge
model whether the sample supports the sentence. Inconsistency score =
fraction of samples that do NOT support it. Higher score => more likely
hallucinated. The original paper's Prompt variant (GPT-3.5 judge) is the
strongest reported detector on WikiBio (AUC-PR ~93.4 for non-factual).

Here the judge is configurable and deliberately includes small models,
since RQ2 concerns detection under the same constraints as generation.
"""

from .client import NimClient

PROMPT_TEMPLATE = (
    "Context: {context}\n\n"
    "Sentence: {sentence}\n\n"
    "Is the sentence supported by the context above? "
    "Answer Yes or No.\n\nAnswer:"
)


def _judge_answer_to_score(text: str) -> float:
    """Map judge output to inconsistency contribution: Yes->0, No->1, unparseable->0.5."""
    t = text.strip().lower()
    if t.startswith("yes"):
        return 0.0
    if t.startswith("no"):
        return 1.0
    # fall back to substring search in the first line
    first = t.splitlines()[0] if t else ""
    if "yes" in first and "no" not in first:
        return 0.0
    if "no" in first and "yes" not in first:
        return 1.0
    return 0.5


def score_sentence(
    client: NimClient, judge_model: str, sentence: str, samples: list[str]
) -> tuple[float, list[float]]:
    per_sample = []
    for sample in samples:
        prompt = PROMPT_TEMPLATE.format(context=sample.strip(), sentence=sentence.strip())
        out = client.chat(
            judge_model,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8,
        )
        per_sample.append(_judge_answer_to_score(out))
    return sum(per_sample) / len(per_sample), per_sample
