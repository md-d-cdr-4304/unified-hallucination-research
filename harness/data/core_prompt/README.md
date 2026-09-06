# The frozen item list of the core study

`items.jsonl` — 800 items, 200 per dataset, sampled **once** with seed
`20260901` by `experiments/core_prompt/build_items.py` and frozen, so that
every model, every prompt class and every run sees exactly the same instances
in exactly the same order. The first 5 items of each dataset carry
`"sentinel": true` (20 items in total); that subset is re-run at the start of
every session and stored separately, which is what makes the silent
model-update check in `analyze.py` possible.

`items.ids.jsonl` — the identifier-only view of the same list
(`dataset`, `id`, `answerable`, `sentinel`, `position`), produced by
`experiments/core_prompt/export_item_ids.py`. It contains no text from the
source datasets and is the redistribution-safe form of the list.

| Dataset | Failure mode studied | Items | Source split |
|---|---|---|---|
| TruthfulQA | factual fabrication | 200 | `truthfulqa/truthful_qa`, generation, validation |
| SQuAD 2.0 | false confidence | 200 (half unanswerable) | `rajpurkar/squad_v2`, validation |
| FEVER | unsupported assertion | 200 (balanced 3-way) | `pietrolesci/nli_fever`, dev |
| KILT | citation grounding | 200 | `facebook/kilt_tasks`, nq, validation |

## Rebuilding

```
python experiments/core_prompt/build_items.py          # re-samples with SEED = 20260901
python experiments/core_prompt/export_item_ids.py      # regenerates items.ids.jsonl
```

`build_items.py` downloads the four datasets from the Hugging Face Hub. The
sample is a function of the seed and of the dataset versions, so a rebuild
reproduces `items.jsonl` byte for byte only against the same dataset revisions;
if a source dataset has been revised upstream, the archived `items.jsonl` is
the authoritative list and the rebuild is a check, not a replacement.

## Redistribution

`items.jsonl` embeds the item text — questions, SQuAD passages, FEVER claims
and evidence, gold answers — because the prompts and the scorer are built from
it. Those excerpts stay under the licence of their source dataset, two of which
are share-alike. See [../../../docs/DATA_SOURCES.md](../../../docs/DATA_SOURCES.md)
and `LICENSE-DATA` before publishing this file.
