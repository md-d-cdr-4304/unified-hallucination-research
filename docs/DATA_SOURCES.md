# Source datasets, models, and what this repository does not redistribute

## The four source datasets

Each dataset carries one hallucination-prone failure mode in the core study.
None of them is redistributed here as a dataset: `build_items.py` downloads
them from the Hugging Face Hub at run time, and the archived item list refers
to the instances by their own identifiers (`items.ids.jsonl`).

| Dataset | Hub identifier | Config / split | Failure mode | Licence on the dataset card |
|---|---|---|---|---|
| TruthfulQA | `truthfulqa/truthful_qa` | `generation`, validation (817 questions) | factual fabrication | Apache-2.0 |
| SQuAD 2.0 | `rajpurkar/squad_v2` | validation | false confidence on unanswerable items | CC BY-SA 4.0 |
| FEVER (NLI form) | `pietrolesci/nli_fever` | dev | unsupported assertion | CC BY-SA 3.0 (Wikipedia text) |
| KILT | `facebook/kilt_tasks` | `nq`, validation | citation grounding | MIT (Wikipedia snapshot: CC BY-SA) |

Sampling: 200 items per dataset, one draw with seed `20260901`, frozen to
`harness/data/core_prompt/items.jsonl`. SQuAD is drawn half answerable, half
unanswerable; FEVER is drawn balanced over the three labels. Five items per
dataset are marked as sentinels.

**Verify each licence on the current dataset card before submission.** The
table records what the cards stated when this repository was assembled;
dataset cards are edited over time.

## Open question to settle before deposit

The thesis archive statement says the source datasets are not redistributed and
that the item lists refer to them by identifier. That is exactly true of
`items.ids.jsonl`. It is **not** true of `items.jsonl`, which embeds the item
text (questions, SQuAD passages, FEVER claims and evidence sentences, gold
answers) for the 800 sampled instances, and it is not true of the prompt
records inside `harness/results/**/responses.jsonl`, which quote those items.

Two of the four sources are share-alike, so releasing that text under CC BY 4.0
is not automatically permitted. Choose one before depositing:

1. **Deposit the identifier-only list.** Ship `items.ids.jsonl` and let users
   rebuild the full list with `build_items.py`. The archive statement then
   stands as written. Cost: the raw responses still quote item text, so the
   response files would need the prompt field reduced to an item id plus a
   template id.
2. **Deposit the full list and adjust the licence.** Keep `items.jsonl` and the
   full response records, and license the affected files under CC BY-SA 4.0
   with per-source attribution, keeping CC BY 4.0 for the purely derived
   material (scores, summaries, metrics). The sentence about
   non-redistribution then needs to say that item excerpts are quoted under the
   source licences.

Option 2 preserves the chain of evidence — a reader can see the exact prompt
that produced a response — and is the reason the repository is currently
arranged that way. `LICENSE-DATA` states the caveat explicitly.

## Models

Model weights are not redistributed. The exact version used in every run is
recorded in the run metadata:

- API arms: `model_version` in each record of `responses.jsonl` is the model
  identifier string returned by the provider, next to the endpoint and a UTC
  timestamp; `run_meta.json` repeats the requested model id and the decoding
  parameters.
- Local arms: the Hugging Face model id and the quantisation setting are in
  `run_meta.json`; the roster is in `harness/configs/models.yaml` and
  [MODEL_ROSTER_AND_COMPUTE.md](MODEL_ROSTER_AND_COMPUTE.md).

Silent provider-side updates are handled by design rather than by trust: the
20-item sentinel subset is re-run at the start of every session into
`sentinel_<session>.json`, and `analyze.py` runs a McNemar test between the
first and each later session.

## Personal data and confidentiality

No personal data was collected. All four datasets are public benchmark
corpora; the annotation sheets contain model outputs and item text only, with
annotators identified as A and B. No material in this repository is subject to
a non-disclosure agreement.
