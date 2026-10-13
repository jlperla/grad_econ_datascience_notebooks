# Data for the slides

Most input data are committed so that data examples render offline. The Toronto
and Argentine full census pages below load from their linked sources. The decks read
`data/<file>` when it exists and otherwise the copy published at
`https://jlperla.github.io/grad_econ_datascience/slides/data/<file>`.

The `utilities/build_<name>.py` scripts download, prepare, or generate these files.
They are published with the other Python scripts in the notebooks repository so
readers can inspect the preparation. Run them manually from the repository root
using the commands below; rendering and deployment do not execute them.

The original data extracts were built on 2026-09-02; generated files give their
own build dates below.

## Handwriting

### Toronto 1901: local model comparison

`handwriting/toronto/` contains Chandra OCR 2, Qwen3.8 27B, and Qwen3.5 2B
responses, parsed CSVs, run metadata, and aggregate comparisons for Canada's
1901 census. **26 labeled people on one development page; 103 on eight test
pages from eight other enumerators.** Each test page contributes all its
York-labeled people. All 400 visible test rows are transcribed, including rows
without reference labels. Pages were selected for public image availability
and verified record alignment before test inference, not randomly from Toronto.

Reference source: Gordon Darroch and the York CCRI Centre,
[Toronto Area Test Sample](https://www.yorku.ca/ccri/), 52,702 people from a
20% sample of dwellings. `source.json` links the original SPSS file, user guide,
and 2006 data-entry manual. The source's title does not denote an OCR evaluation
split. We use an enumerator-separated development/test split; exposure during
model pretraining cannot be ruled out.

Scans: **Library and Archives Canada, Census of Canada 1901, RG31-C-1**.
`pages.json` records each original image URL, source-page citation, York lookup
key, and manually annotated geometry. The build script uses `curl` to download
the approximately 104 MB York SPSS file and the nine scans into ignored
`build/handwriting/toronto/`. It writes `labels.csv`, image panels, and download
hashes there. Full source scans, York microdata, and detailed cell comparisons
are not mirrored in the public repository. York requests source acknowledgement
and supplies no explicit open-data licence. See the
[LAC reproduction terms](https://www.canada.ca/en/library-archives/terms-conditions.html).

`input_illustration.png` is one attributed teaching illustration: lines 1–10 of
[LAC image z000106253](https://central.bac-lac.gc.ca/.item/?id=z000106253&app=census1901&op=img),
enumerator James F. Hutchman. It is an adapted excerpt, with selected columns,
rectification, rescaling, and modern headings and line numbers; it is not an
official LAC version. Source-image terms are separate from this repository's
licence. The full-page slide loads the LAC image remotely.

**Input and scoring.** Manual page, column, and row localization produces five
ten-row panels per page, with name, age, occupation, and earnings. Every model
sees identical 1410 × 1070 PNGs. This is assisted transcription, not automatic
layout recovery; handwriting that crosses the annotated cell boundaries can be
clipped or enter an adjacent crop. Names and occupations are preserved as raw
predictions but not scored. York expands household surnames, and occupation
strings mix abbreviations, literal descriptions, and editorial expansions;
there is no adjudicated mapping to equivalent occupations. Score **age and
earnings** only on the matching physical page/line. Unique printed line numbers
align people independently of row order. A missing or duplicate identifier makes
that person's values unavailable, without discarding other rows. HTML headers
and spans identify columns; a missing or ambiguous column becomes `NaN` while
other columns remain usable. A two-column Earnings header is ambiguous, so we
do not choose either subcolumn. Complete JSON objects survive truncation;
unfinished objects and missing fields are unavailable. Correct line identifiers
alone do not guarantee that the values belong to the right person.

CSV files write unavailable output as the literal `NaN`; extracted blanks stay
empty strings. Load with `pd.read_csv(..., keep_default_na=False,
na_values=["NaN"])` to retain this distinction. Unavailable cells have no
correctness score. A completely unavailable column therefore has `NaN`
accuracy and zero coverage, rather than affecting other columns.

`protocol.json` records the numeric normalization: Unicode normalization,
currency formatting, whole-dollar earnings, and completed-year infant ages. York's
`*` denotes missing, while `!`, `?`, braces, and `=` flag uncertain readings;
uncertain references are counted separately. **Digit accuracy** is one minus the
sum of character edit distances divided by the number of reference digits,
computed within each available filled cell after numeric normalization. A
substitution, insertion, or deletion costs one: `53` read as `23` scores 50%,
and `350` read as `35` scores 66.7%. Unexpected nondigit characters also cost
edits. The measure can be negative if insertions exceed the reference length;
we do not clip it. Extracted empty strings count as deletions of every reference
digit. `NaN` cells are omitted from conditional accuracy and counted separately
in coverage. Whole-value agreement and blank agreement use their own available
cell denominators. These metrics compare standardized research labels,
not literal character transcriptions. The published labels remain unchanged,
including three flagged ages on the Maclennan and Murray pages (`audit.json`).
Occupation exclusion, independent column recovery, and digit scoring were
chosen after inspecting initial results. `protocol.json` records the amendments
and original/current script hashes. Raw responses, images, references, and numeric
normalization are unchanged; parsed availability and scoring denominators change.
`post_run_audit.json` records
numeric-format diagnostics and confirms that parsed Qwen cells contain no JSON
`null` values.

`scores/test_age_audit.json` separates Chandra's 35 whole-age disagreements into
21 on the Hutchman page, three flagged reference ages, and eleven other digit
disagreements. Eight correct ages are retained despite unavailable earnings.
All 103 labeled ages are available: 68 whole-value matches and 42 digit edits
among 186 reference digits, for 77.4% digit accuracy. One Hutchman panel emits
the invented age sequence 74–83. All pages remain in the benchmark.
The cause of the Hutchman failure has not been
isolated between source quality, image preparation, and model behavior.

The earnings diagnostic selects positive reference earnings with no alternative
pay-period code (annual under the manual's convention). It reports numeric
prediction coverage, reference and predicted means for the same available
people within each model, and paired mean absolute error. Models can have
different available subsets, so the means and errors are conditional on coverage.
Spaced cents strings such as `120 00` are nonnumeric for the earnings calculation;
their unexpected characters still contribute errors to digit scoring.
Chandra's saved `250 00` contributes three edits against the standardized `250`,
so this metric includes formatting errors. Qwen 2B's `- -` remains nonblank under
the fixed missing-value convention and accounts for its one blank disagreement.
Missing amounts are not zeros. This
illustrates measurement error in a small selected sample; it is not a Toronto
earnings estimate. `scores/test_page_scores.csv` exposes variation by page.
Chandra's four ambiguous earnings columns contain eight labeled people, including
two filled earnings references. Its earnings digit accuracy is 82.5% on 27/29
available filled cells (80 reference digits). Qwen 27B's token-limited panel has
two complete rows and eight unavailable rows, with no York-labeled people.
Qwen 2B completes all forty panels.

**Executed models:** Chandra OCR 2 revision
`af93b47dba1b47b6640c86ccf487ed2260ab9a09`, BF16 through MLX-VLM 0.7.6 / MLX
0.32.3; `qwen3.8:27b-mxfp8` through Ollama's MLX runtime; and `qwen3.5:2b`
in GGUF Q8_0, both through Ollama 0.35.0. Runs used an Apple M4 Max with 64 GB.
Chandra uses its native `OCR_PROMPT` and HTML; Qwen uses the same JSON prompt
for both sizes, with thinking disabled. Temperature is 0, output limit 4,096
tokens, and Qwen context 16,384. Each `test_run.json` records the exact model
digest/revision, prompt, input hashes, and software. This compares workflows,
including interfaces and quantization, not isolated model weights. No model was
fine-tuned. Built 2026-10-13 UTC.

Prepare the downloadable data:

```bash
uv run --extra handwriting python utilities/build_toronto_census.py
```

For Chandra, use the pinned download/server commands in the Paris section below.
Then run in another terminal:

```bash
OCR_MODEL=build/handwriting/model \
OCR_REVISION=af93b47dba1b47b6640c86ccf487ed2260ab9a09 \
uv run --extra handwriting python code/toronto_census_ocr.py \
  --workflow chandra --split test
```

With Ollama installed, stop the Chandra server to free memory, then run:

```bash
ollama pull qwen3.8:27b-mxfp8
ollama pull qwen3.5:2b
uv run --extra handwriting python code/toronto_census_ocr.py \
  --workflow qwen27b --split test
ollama stop qwen3.8:27b-mxfp8
uv run --extra handwriting python code/toronto_census_ocr.py \
  --workflow qwen2b --split test
uv run python utilities/build_toronto_census_scores.py
```

Use `--split development` for the development page. Extraction and scores go
to `build/handwriting/toronto/`; the raw HTML/JSON, CSV, and metadata are saved.
Ollama tags can change: each run records the installed digest and server version.
After preparing the downloadable data above, recompute scores for the committed
predictions without loading any model:

```bash
uv run python utilities/build_toronto_census_scores.py \
  --predictions slides/data/handwriting/toronto
```

To reparse saved raw responses with the current column rules, run the following
for each workflow (`chandra`, `qwen27b`, `qwen2b`) before scoring:

```bash
uv run python code/toronto_census_ocr.py --workflow chandra --split test \
  --output slides/data/handwriting/toronto/chandra --reparse
```

To update the lecture's saved outputs, copy each model's `*_responses.jsonl`,
`*_predictions.csv`, `*_structure.csv`, and `*_run.json` into its corresponding
`slides/data/handwriting/toronto/` directory. Copy only `*_field_scores.csv`,
`*_structure.csv`, `*_page_scores.csv`, `*_earnings.csv`, `*_sample.json`,
and `*_age_audit.json` from the scoring directory into
`slides/data/handwriting/toronto/scores/`.
Keep `*_cells.csv` in `build/`: it contains the downloaded York reference values.

### Paris census benchmark

`handwriting/` contains six lossless crops from the **1926 Paris census**: 60
development rows from two writers and 120 evaluation rows from four other writers,
with ten independently published labels per row. The original printed column
headers remain in each image. `pages.csv` records the image hashes and crop offsets;
`labels.csv` retains the original slash-separated annotation, parsed cells, and
ALTO row coordinates relative to the cropped image. No synthetic handwriting or
model-generated reference labels are used.

Source: Constum, Kempf, Paquet, Tranouez, Chatelain, Brée, and Merveille (2022),
"Recognition and Information Extraction in Historical Handwritten Tables:
Toward Understanding Early 20th Century Paris Census", *DAS*, pp. 143–157.
[POPP Generic release](https://doi.org/10.5281/zenodo.6581158), **CC BY 4.0**;
[annotations](https://github.com/constum-thomas-m/POPP-datasets/tree/580000d295c88e02bb7d8578ed203b7cde5deba7/Generic).
The original release's licence applies; the different MIT metadata on the
Hugging Face line mirror is not used. Built 2026-10-12.

The four test double pages were sampled with Python `random.Random(526).sample(ids, 4)` from
the sorted eight original test double-page IDs; only the left page is used.
The literal IDs are frozen in `utilities/build_handwriting_data.py`. Development
uses two manually selected pages from the original validation split, `275_0025`
and `301_0020`. These are six writers from one archive,
not a representative sample of historical handwriting. Public availability since
2022 means exposure during the base model's training cannot be ruled out.

The ten fields are surname, given names, birth year **as written**, birthplace,
nationality, marital status, relationship to the household head, education,
occupation, and employer/statistical code. A two-digit birth year is not an age.
`¤` is an empty cell; `/` separates columns; `!` and `?` mark text below or above
the baseline. Evaluation replaces the latter two with spaces, normalizes Unicode
to NFC, and collapses whitespace. It preserves case, accents, punctuation,
abbreviations, and ditto marks. No names or places are filled from neighboring rows.

Chandra makes separate page and row-crop predictions with its packaged `OCR_PROMPT`,
greedy decoding, 8,192 page tokens, and 256 row tokens. Page inputs are enlarged
2×; row crops are resized to height 128, preserving aspect ratio, with LANCZOS
interpolation. These settings were fixed using the development pages. Run metadata records the
exact model revision, prompt, package versions, and runtime. Raw responses are
retained in `chandra/`. The model is a modified OpenRAIL-M checkpoint; its licence
is separate from the census data's CC BY 4.0 licence.

`scores/` contains comparisons per cell and row, plus a wide `predictions.csv`.
Each field is scored independently. HTML spans retain column positions; a cell
spanning distinct people or fields is unavailable (`NaN`), while unaffected cells
survive. Read these CSVs with `keep_default_na=False, na_values=["NaN"]` to keep
extracted blanks distinct from unavailable output. Incorrect readings are not
converted to `NaN`. `--reject-field FIELD` explicitly rejects only that column;
the summary records the rejection and reduced coverage.

Character accuracy is one minus summed Levenshtein edits divided by available
filled reference characters, within cells. Digit accuracy applies the same rule
to birth-year references consisting entirely of digits, without stripping any
predicted characters. Case, punctuation, accents and ditto marks remain significant.
Scores are uncapped and can be negative. Report cell, filled-cell and character
coverage alongside conditional accuracy. Exact recovery over all reference cells
gives unavailable cells zero credit. Blank agreement is separate. The independent
row-crop run retains its CER/WER, including repetition and truncated text.

Automatic alignment uses physical order on a complete single table with matching
row counts; this is a count check, not proof of semantic layout correctness.
Both models emit 30, 30, 30 and 29 rows. A 2026-10-13 source-image audit found that
the final page has 29 people and a thirtieth annotation region containing only
employer code `437`. It possibly overlaps the last person, whose annotation also
contains that code. Preserve every published label, align the 29 visible people
in order, and leave the final region unavailable. This is **source-assisted
alignment**, not a missed-person claim or autonomous layout recovery. Chandra
also returns two brace marks spanning five blank surname cells; our per-person
projection leaves only those cells `NaN`. This affects conditional blank agreement,
but gives no extra credit in exact recovery over all reference cells.

`layout_audit.json` binds the structural maps to the original outputs and labels,
records image hashes, and flags the possible overlap. It contains no spelling
repairs or reference-text matching. `scoring_protocol.json` documents this
post-inference amendment. Automatic coverage is 895/1200 cells for Chandra and
900/1200 for Claude; source-audited coverage is 1185/1200 and 1190/1200. Both cover
99.9% of filled reference characters after the audit. Automatic exact recovery
is 47.1% versus 54.8%; source-audited exact recovery is 60.3% versus 71.5%.
Use `--automatic-only` to omit the recorded assistance. New output hashes need
their own source audit; the scorer never applies the old map to a new response.

For the frozen model version, download the pinned weights and start the server:

```bash
uv run hf download datalab-to/chandra-ocr-2 \
  --revision af93b47dba1b47b6640c86ccf487ed2260ab9a09 \
  --local-dir build/handwriting/model
uv run --extra handwriting-mac python -m mlx_vlm.server \
  --model build/handwriting/model --host 127.0.0.1 --port 8081
```

In a second terminal, regenerate the data and results used by the slides:

```bash
uv run python utilities/build_handwriting_data.py
export OCR_MODEL=build/handwriting/model
export OCR_REVISION=af93b47dba1b47b6640c86ccf487ed2260ab9a09
uv run --extra handwriting python utilities/build_handwriting_predictions.py \
  --split test --mode pages --output slides/data/handwriting/chandra
uv run --extra handwriting python utilities/build_handwriting_predictions.py \
  --split test --mode rows --output slides/data/handwriting/chandra
uv run --extra handwriting python utilities/build_handwriting_scores.py \
  slides/data/handwriting/chandra --output slides/data/handwriting/scores
uv run --extra handwriting python code/handwriting_ocr.py \
  --output slides/data/handwriting/chandra/demo
```

Use `--split development` for development runs. Data preparation reads selected
members of the 1.54 GB Zenodo ZIP through HTTP range requests. The committed data,
images, and results are published on the course site **and copied into the notebooks
artifact before upload and leak checking**; rerunning a notebook needs neither
Zenodo nor a running model for the Paris example.

`claude/` contains an independent transcription by `claude-opus-5-5` through
Claude Code, with the exact prompt in `claude_prompt.txt`. Each source image is
copied alone into a temporary directory as `census.png`; only the Read tool is
enabled, and no labels or Chandra predictions are supplied. Claude returned
ten-string JSON rows. `claude_scores/` applies the same independent-cell metrics
and recorded source audit. `[illegible]` is scored as an incorrect transcription,
including as a false fill on a blank cell. This compares two
workflows: Claude has an explicit column schema while Chandra uses its native
OCR prompt. It is not an isolated comparison of model weights.

With an authenticated Claude CLI, reproduce that comparison with:

```bash
uv run python utilities/build_handwriting_claude.py \
  --output slides/data/handwriting/claude
uv run --extra handwriting python utilities/build_handwriting_scores.py \
  slides/data/handwriting/claude --pages-only \
  --output slides/data/handwriting/claude_scores
```

The Chandra endpoint can also be supplied by the
[official NVIDIA/vLLM setup](https://github.com/datalab-to/chandra#usage).
Set `OCR_BASE_URL`, `OCR_MODEL`, and `OCR_API_KEY` for that server; the Apple
Silicon/MLX path is the one executed for the committed results. Set
`OCR_REVISION` and `OCR_RUNTIME` when rebuilding to record the loaded checkpoint
revision and server versions alongside the predictions.

### Argentine census fine-tune: download on demand

`handwriting/argentina/` contains our model response, parsed JSON, CSV, and run
metadata for one page of **Argentina's 1895 census**, Cañada de Álvarez,
Calamuchita, Córdoba, image 10 of 168: 15 predicted people. It contains no source image or reference
transcription. The script downloads the page into the ignored
`build/handwriting/argentina/source.jpg`; the slides link the same image remotely.

Source: Nils Ledin (2026), *Handwritten Text Recognition of Historical Argentinian
Census Documents*, [Chalmers thesis](https://odr.chalmers.se/items/8ba120cd-6603-4f65-aa77-773917e5fba2),
[author's image directory](https://github.com/Saucy-Stream/Argentina_1895_census_HTR_code/tree/1048ba9c12863283ad17b6cde11684e0ed42885d/test).
The pinned repository revision is `1048ba9c12863283ad17b6cde11684e0ed42885d`.
The thesis identifies FamilySearch as the image source and says its source images
cannot be shared publicly. We link the author's hosted example and do not mirror
the image collection or apply this repository's licence to it. `source.json`
records the exact download URL and SHA-256. Built 2026-10-12.

This is a **training-set page**. The author's [inference log](https://github.com/Saucy-Stream/Argentina_1895_census_HTR_code/blob/1048ba9c12863283ad17b6cde11684e0ed42885d/slurm-outputs/slurm-6653862.out)
reads it from `training_data`. The public repository's `test/` directory name
does not indicate a held-out split. The comparison below measures agreement
with its public transcription on this training page. We never feed that
transcription to a model.

The model is [`Saucystream/fine_tuned_chandra`](https://huggingface.co/Saucystream/fine_tuned_chandra),
revision `d94b4ca422dd87e06b55abdd955bb199879a545a`: a PEFT LoRA adapter for
**`datalab-to/chandra` (Chandra 1)**, pinned at
`bd40c21576564d31ba99f204e28ccd5c4c1751fa`. It is not a Chandra OCR 2 adapter.
Its configuration has rank 8, alpha 16, and targets query, key, and value
attention projections. The base model's card lists OpenRAIL; those terms are
separate from source-image rights. The adapter card gives no separate licence.

The demonstration uses the author's exact system/user prompts and made-up schema
example from `CensusDataset.py`; no reference transcription is supplied. It fits
the full page within 2048 by 1024 pixels with Pillow's default BICUBIC thumbnail,
as in `inference_no_transform`. The output schema uses letters printed on the
form: `A` is `[surname, given names]`, `C` age, `H` occupation, `I` literacy,
and `K` property ownership. `"*"` denotes an empty cell, not a negative answer.
The original JSON retains split columns and all other fields. The CSV contains
surname, given names, sex, age, occupation, literacy, and property ownership;
it does not repair spelling or expand ditto marks.

We merge the adapter into the BF16 base with PEFT, then serve through MLX on
Apple Silicon. The author's code trains the adapter against a 4-bit NF4 base;
merging into BF16 does not reproduce that model. Its inference pipeline samples
at temperature 0.7 with a 3000-token limit; ours decodes greedily with an 8192-token
limit. This comparison does not estimate unseen-page performance. We did not
train any model.

From the repository root, prepare the model once and start the server:

```bash
uv run --extra handwriting python utilities/build_handwriting_argentina_model.py
uv run --extra handwriting-mac python -m mlx_vlm.server \
  --model build/handwriting/argentina-model --host 127.0.0.1 --port 8084
```

The base download is about 17 GB; the merged copy uses another 17 GB. The verified
machine is an Apple M4 Max with 64 GB of memory. In a second terminal:

```bash
uv run --extra handwriting python utilities/build_handwriting_argentina.py
```

Results go to `build/handwriting/argentina/finetuned-json/`. `--download-only` downloads
just the page. To regenerate the committed outputs, keeping the image in `build/`:

```bash
OCR_RUNTIME='mlx-vlm 0.7.6; mlx 0.32.3; Apple M4 Max; BF16 PEFT merge' \
uv run --extra handwriting python utilities/build_handwriting_argentina.py \
  --output slides/data/handwriting/argentina
```

`OCR_BASE_URL`, `OCR_MODEL`, and `OCR_API_KEY` select another compatible server;
the checkpoint and prompt must match this adapter. The download requires access
to the author's GitHub repository; model preparation requires Hugging Face.
Quarto and the notebooks artifact exclude `build/`, so downloads are not published.

#### Comparing Chandra generations

`argentina/scores/` contains aggregate error counts and the models' projected
predictions. The scoring script downloads the author's pinned transcription
into `build/`; it does not publish the annotation or cell-level correctness masks.
The six fields are full name, sex, age, occupation, literacy, and property
ownership: 90 cells for 15 people, including 67 filled cells. Row numbers align
records. Names retain surname-first order; split names are joined without
guessing boundaries. This change follows observing Chandra 1's unsplit names.

Exact matching preserves case, accents, punctuation, and ditto marks after NFC
and whitespace normalization; empty text becomes `*`, integers become strings.
Thus `no` disagrees with the reference's `nó`. Separate sensitivity counts
equate these two spellings and exclude the uncertain row-14 occupation
(`Jomalero` in the reference, `Jornalero` in the fine-tune). The script leaves
both original readings intact. Unavailable fields and duplicate or absent row
IDs become `NaN`; readable blank `*` remains a scored prediction. Other rows and
fields survive. Complete structured records survive truncation. Report extra and
duplicate row IDs, finish reason, coverage, conditional character/digit accuracy,
and exact recovery over all 90 reference cells separately. Missing cells receive
zero credit in exact recovery, not a fictional handwriting error. Both scorers
accept `--reject-field FIELD` to reject a column without losing the rest, recording
the choice and coverage. See `argentina/comparison_protocol.json` for the protocol.

Character/digit accuracy uses the same uncapped within-cell Levenshtein formula
as Paris. Age digit accuracy retains predicted periods and other stray characters.
For the training-page HTML/adapter comparison, all three automatically recover
90/90 cells. Original Chandra, LoRA and OCR 2 have character accuracies of 68.3%,
96.6% and 71.7%, and age digit accuracies of 57.7%, 100% and 61.5%. Exact-cell
disagreements remain 63/90, 3/90 and 53/90. Reading accuracy concerns the 67 filled
cells; blank agreement is separate. This is still a training-page demonstration.

Post-run diagnostics also count disagreements differing only by periods or
ditto-mark typography. They do not change the score. Periods alone account for
24 original-Chandra HTML disagreements and 22 OCR-2 HTML disagreements; the
latter also has three double-quote versus two-single-quote dittos. Both base
models write ages such as `4.5` instead of the reference's `45`. Automatically
removing periods would change economic variables, so this is a diagnostic,
not a correction rule or a pure handwriting-recognition benchmark.

There are two comparisons. All three models receive the author's JSON prompt.
OCR 2 exhausts its 8192-token limit without a complete structured record:
0/90 available cells, undefined conditional reading accuracy, and 0% exact recovery.
The two base models also receive `chandra-ocr` 0.2's supported `OCR_PROMPT` to
produce HTML; the fine-tune retains its JSON prompt. This second comparison is
between extraction workflows, with different prompts. All runs use the same
image bytes, BF16, MLX, and greedy decoding. The adapter has seen both this page
and its prompt in training. Neither comparison measures generalization.

To run the original base model, stop the other server and use two terminals.
Set `OCR_MODEL` in both terminals; `hf download` prints the pinned cache path:

```bash
export OCR_MODEL=$(uv run hf download datalab-to/chandra \
  --revision bd40c21576564d31ba99f204e28ccd5c4c1751fa)
export OCR_RUNTIME='mlx-vlm 0.7.6; mlx 0.32.3; Apple M4 Max; BF16'
```

```bash
uv run --extra handwriting-mac python -m mlx_vlm.server \
  --model "$OCR_MODEL" --host 127.0.0.1 --port 8084
```

```bash
uv run --extra handwriting python utilities/build_handwriting_argentina.py \
  --checkpoint chandra1 --prompt html
uv run --extra handwriting python utilities/build_handwriting_argentina.py \
  --checkpoint chandra1 --prompt json
```

For OCR 2, repeat with `datalab-to/chandra-ocr-2`, revision
`af93b47dba1b47b6640c86ccf487ed2260ab9a09`, and `--checkpoint chandra2`.
Its JSON run saves the raw response before JSON parsing fails. To regenerate
the saved comparison, use `--output slides/data/handwriting/argentina/<run>`
where `<run>` is `chandra1`, `chandra2`, `chandra1-html`, or `chandra2-html`.
Then score the saved responses:

```bash
uv run --extra handwriting python utilities/build_handwriting_argentina_scores.py
```

The scoring output defaults to `build/handwriting/argentina/scores/`. Use
`--output slides/data/handwriting/argentina/scores` to update the slide data.
Use `--predictions build/handwriting/argentina` to score fresh runs from the
builder's `<checkpoint>-<prompt>` directories instead of the committed responses.

#### One held-out census page

`argentina/holdout/` contains four local model responses and aggregate scores for
**S3HT-62RW-C1Y**, 15 people, built 2026-10-12. Source: Nils Ledin (2026),
[*Handwritten Text Recognition of Historical Argentinian Census Documents*](https://odr.chalmers.se/items/8ba120cd-6603-4f65-aa77-773917e5fba2),
Figure 5.16 (PDF page 64); reference: Figure A.31 (PDF page 92). Figure A.30 is
the author's model output and is never used as ground truth. The thesis and
source scans retain their source rights; no redistribution licence is assumed.
The PDF, extracted image and reference stay in ignored `build/` and are not
included in the site or notebooks. Slides link the author's thresholded GitHub
copy of the same page; inference uses the grayscale thesis figure.

The thesis describes 14 pages excluded from training and tuning. Its author
selected this as the **best test page for the blur-trained model**, so this is not a random or
representative benchmark. We checked both GitHub revisions, its branch/release
list, the author's Hugging Face collection and datasets, and Chalmers attachments.
Three processed holdout images and one appendix reference were found, but no
complete labeled holdout. An older GitHub revision contains 110 concatenated JSON
pages, without this page's distinctive names. The logged training and final-test
page IDs do not overlap. See `holdout/search_sources.json` for the search record.
Held-out status concerns the author's fine-tuning, not unknown base-model
pretraining data.

`build_handwriting_argentina_holdout.py` downloads the thesis and uses Poppler's
`pdfimages` to extract its embedded 2037×1024 image. The figure has the author's
median blur; RGB conversion does not rescale it. `pdftotext -layout` extracts
A.31, replacing the typeset `„` glyph with two comma delimiters. All 15 rows
then have 20 fields. This extraction was checked against the rendered appendix.
Hashes identify the PDF, extracted reference and common model input.

The same six fields give 90 cells, including 67 filled cells. The primary score
retains the author's labels, case, punctuation and accents. Separate diagnostics
fold accents and exclude five visually uncertain cells (four names and one
occupation), giving 85 cells; their locations are in `comparison_protocol.json`.
These checks were specified from an independent image/reference review before
inspecting model responses. The HTML parser reads name cells spanning both name
columns once, instead of duplicating pandas' colspan expansion; this parsing
rule was added after inspecting Chandra 1's output. Automatic scores use the
fixed printed-column map. Source-audited scores separately apply the recorded
OCR 2 structural mapping, without changing field strings. Emitted rows, automatic
and assisted alignment, unmatched reference rows, and format failures remain visible.

The primary comparison uses the same three checkpoints as the training page.
The fourth run uses `Saucystream/fine_tuned_chandra_blur`, revision
`6a103e674149b2aed54e824fe41ab09838443e7e`, the adapter variant used for the thesis's
final evaluation. The original adapter was trained without blur; the blur adapter
was trained on median-blurred pages. The common input here is the blurred thesis
figure, and selection as the best page reflects the blur adapter's own errors.
Both base models output native HTML; both adapters output
JSON with the author's prompt. All runs use identical image bytes, BF16, MLX
and greedy decoding. No model is trained, and these runs do not reproduce the
author's quantized, sampled inference or full-test-set metrics.

| Workflow | Automatic cells | Source-audited cells | Automatic exact recovery | Source-audited character accuracy |
|:---|---:|---:|---:|---:|
| Chandra 1 / HTML | 90/90 | 90/90 | 62.2% | 83.7% |
| Original LoRA / JSON | 90/90 | 90/90 | 87.8% | 92.9% |
| Chandra OCR 2 / HTML | 36/90 | 90/90 | 31.1% | 84.5% |
| Blur LoRA / JSON | 90/90 | 90/90 | 87.8% | 93.5% |

OCR 2 inserts a margin column from row 7 and drops province F. Its HTML body
contradicts its header: the automatic map cannot align nine emitted rows, leaving
54 unavailable cells. The other 36 cells remain scored. A source-image audit
shifts only left-page fields and leaves the right side fixed. This retrospective,
output-specific map recovers those 54 cells; a uniform shift would damage the
right-page fields. `handwriting/layout_audit.json` binds it to output and reference
hashes; `scores/layout_diagnostic.json` records the assisted result. The models'
raw responses and reference labels are unchanged.

After this documented assistance, exact-cell disagreements are 34/90, 11/90,
28/90 and 11/90, respectively; filled-cell disagreements are 32/67, 11/67, 28/67
and 11/67. Age digit accuracy is 100%, 96.2%, 96.2% and 96.2%. Both base models
have 15 period-only disagreements (`V.`/`M.` versus `V`/`M`); neither adapter has
any. Excluding five uncertain reference cells gives 30/85, 6/85, 23/85 and 6/85.
Accent folding leaves these error counts unchanged. Chandra 1 still misplaces
some values, including both filled property cells; availability does not certify
correct column assignment, and these remain disagreements. No output values
are repaired or discarded just because they disagree with the reference.

These are diagnostics of reference agreement on one selected page. Assisted
accuracy is not automatic layout accuracy. `automatic_predictions.csv` keeps the
unassisted projection alongside the source-audited `predictions.csv`; neither
contains the reference text. Run with `--automatic-only` to disable the audit.
The thesis is downloaded on first use and reused from the local ignored cache.

Install Poppler (`brew install poppler` on macOS), then inspect the source and
reference without loading a model:

```bash
uv run python utilities/build_handwriting_argentina_holdout.py
```

Use each matching server from the preceding instructions, one at a time.
The runtime strings describe our Mac; change them to match your environment:

```bash
OCR_RUNTIME='mlx-vlm 0.7.6; mlx 0.32.3; Apple M4 Max; BF16 PEFT merge' \
uv run --extra handwriting python utilities/build_handwriting_argentina.py \
  --page holdout --checkpoint finetuned --prompt json
uv run --extra handwriting python utilities/build_handwriting_argentina.py \
  --page holdout --checkpoint chandra1 --prompt html
uv run --extra handwriting python utilities/build_handwriting_argentina.py \
  --page holdout --checkpoint chandra2 --prompt html
```

For the secondary blur adapter, stop the other server, unset `OCR_MODEL` if set,
then prepare and serve its published weights:

```bash
uv run --extra handwriting python utilities/build_handwriting_argentina_model.py \
  --adapter blur
uv run --extra handwriting-mac python -m mlx_vlm.server \
  --model build/handwriting/argentina-blur-model --host 127.0.0.1 --port 8084
```

In another terminal, also with `OCR_MODEL` unset:

```bash
OCR_RUNTIME='mlx-vlm 0.7.6; mlx 0.32.3; Apple M4 Max; BF16 PEFT merge' \
uv run --extra handwriting python utilities/build_handwriting_argentina.py \
  --page holdout --checkpoint blur --prompt json
uv run --extra handwriting python utilities/build_handwriting_argentina_scores.py \
  --page holdout --predictions build/handwriting/argentina/holdout
```

Results go to `build/handwriting/argentina/holdout/<checkpoint>-<prompt>/` and
`holdout/scores/`. Omit `--predictions` to rescore the committed responses.
To update slide data, copy only run outputs and aggregate scores into
`slides/data/handwriting/argentina/holdout/`; keep the PDF, image and labels in
`build/`. The notebook artifact includes the saved responses and scoring scripts.

### Thai bill fine-tune

`handwriting/appen/` demonstrates
[`fwgpiyawudk/ChandraOCR2_AppenDoc-merged`](https://huggingface.co/fwgpiyawudk/ChandraOCR2_AppenDoc-merged)
on `THAI_RECEIPT_003149.jpg` from the same author's
[`appen-thai-doc`](https://huggingface.co/datasets/fwgpiyawudk/appen-thai-doc).
The original source is **Appen Limited, OCR image data for Thai documents**,
[Kaggle release](https://www.kaggle.com/datasets/appenlimited/ocr-image-data-for-thai-documents),
**CC BY-SA 4.0** (also reported by Kaggle's public dataset API).
The original JPEG is unchanged. `input.png` is a crop to the printed item header
and first handwritten row, resized to width 1600; exact coordinates, image hashes,
model/data revisions, and the prompt are in `source.json`. Built 2026-10-12.
This example and adapted outputs retain CC BY-SA 4.0, separately from POPP's CC BY.

The author's paired LoRA configuration names `datalab-to/chandra-ocr-2` as its
base, with rank 32 and alpha 64. The merged checkpoint runs through the same MLX
API. This image is in the dataset's **train** split: treat it as a training example.
The model's full recipe and exact training membership are undocumented, as is the
origin of the mirror's `text` annotations. Those annotations are not bundled or
used for scoring. This demonstrates inference; performance on new pages and
improvement over the base model are unmeasured. No fine-tuning was performed here.
An exploratory full-page call repeated blank rows and reached its token limit;
the demonstrated crop returns one complete row and stops normally.

Download and serve the pinned fine-tune in one terminal:

```bash
uv run hf download fwgpiyawudk/ChandraOCR2_AppenDoc-merged \
  --revision dde20ac03d4dbc851af082576c7b34f095cf12e0 \
  --local-dir build/handwriting/appen-model
uv run --extra handwriting-mac python -m mlx_vlm.server \
  --model build/handwriting/appen-model --host 127.0.0.1 --port 8083
```

In another terminal:

```bash
uv run --extra handwriting python utilities/build_handwriting_finetune.py
```

The script downloads the source image, builds the
crop, and saves the model response, HTML, CSV, and JSON records under
`slides/data/handwriting/appen/`. Quantities and prices remain strings until an
analysis explicitly converts them. The original annotation is never in the prompt.

### Adapting a new corpus

Use the original POPP **training** split for learning, its validation writers for
choosing settings, and leave the lecture's test writers untouched. A training
example pairs a cropped line image with verified text wrapped in native HTML,
such as `<p>Vallet Etienne 1900 o cheminot</p>`; construct the full target from
all fields, with the same abbreviation and annotation conventions as evaluation.
Train with loss only on the assistant's target tokens, including the end token.
LoRA learns small matrix updates while the base weights stay frozen; full
fine-tuning updates the selected original weights. Choose learning rate, number
of epochs, LoRA rank, and image scale on development documents. Retain a base-model
run on the same test images to measure whether adaptation helps. Line recognition
and page-to-table extraction require separate evaluations.

For a page-to-table task, the Argentine example instead pairs a whole image with
verified JSON in a fixed schema. The prompt, image preprocessing, and target
format are part of that learned task; they must accompany a distributed adapter.
The [Barbados checkpoint](https://huggingface.co/Abdoul27/chandra-ocr-2-barbados)
also provides a detailed line-level training recipe, but its gated weights and
challenge data are not used by this example.

## `embedding_bank_comparison.csv`

Eight cosine similarities for four word pairs, generated with local
`qwen3-embedding:0.6b` embeddings and hosted OpenAI
`text-embedding-3-large` embeddings. The file lets `local_llms.qmd` render
without Ollama or an API key.

Columns: `pair_order, word_1, word_2, provider, model, dimensions,
cosine_similarity`.

Source: model outputs from
[`qwen3-embedding:0.6b`](https://ollama.com/library/qwen3-embedding:0.6b)
through Ollama and OpenAI's
[`text-embedding-3-large`](https://platform.openai.com/docs/models/text-embedding-3-large),
generated 2026-09-03. No third-party source data.
Licence: derived numerical measurements released under the repository's
CC BY-NC-SA licence.

Rebuild: start Ollama, run `ollama pull qwen3-embedding:0.6b`, set
`OPENAI_API_KEY`, then run
`uv run python utilities/build_embedding_comparison.py`.

## `eurosat_predictions.csv`

1,000 predictions for the first 100 chips of the EuroSAT-RGB test split
(`cm93/eurosat` on the Hugging Face Hub, `split="test[:100]"`): each chip
classified by the local vision-language models `qwen3.5:2b`, `qwen3.5:4b`,
`qwen3.8:27b-mlx`, and `qwen3-vl:4b-instruct` through Ollama with
`temperature=0` and `reasoning_effort="none"`, once zero-shot and once with
one labelled example chip per class (from `test[100:300]`) in the prompt; by
OpenAI's `gpt-4o-mini` zero-shot; and by the fine-tuned Vision Transformer
`Haifald/vit-eurosat` through `transformers`. `answer` is the model's reply;
`predicted` is the reply when it is one of the ten labels and empty otherwise.
The file lets `image_classification.qmd` render without Ollama or `torch`.

Columns: `index, truth, model, examples, answer, predicted`.

Source: labels from Helber, Bischke, Dengel, and Borth (2019), "EuroSAT: A
Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover
Classification", *IEEE JSTARS* 12(7), via
[`cm93/eurosat`](https://huggingface.co/datasets/cm93/eurosat) (MIT licence);
model outputs from [`qwen3.5`](https://ollama.com/library/qwen3.5),
[`qwen3.8`](https://ollama.com/library/qwen3.8), and
[`qwen3-vl`](https://ollama.com/library/qwen3-vl) through Ollama, OpenAI's
[`gpt-4o-mini`](https://platform.openai.com/docs/models/gpt-4o-mini), and
[`Haifald/vit-eurosat`](https://huggingface.co/Haifald/vit-eurosat) (Apache
2.0), generated 2026-09-04.
Licence: derived model outputs released under the repository's CC BY-NC-SA
licence.

Rebuild: start Ollama, run `ollama pull qwen3.5:2b`, `ollama pull qwen3.5:4b`,
`ollama pull qwen3.8:27b-mlx` (an Apple Silicon Mac with 48 GB), and
`ollama pull qwen3-vl:4b-instruct`, install the extra with
`uv sync --extra extras`, set `OPENAI_API_KEY`, then run
`uv run python utilities/build_eurosat_predictions.py`.

## `onet_titles.csv.gz`

8,189 reported job titles from the O*NET 31.0 database file
`sample_of_reported_titles.csv`, with the O*NET-SOC code, the SOC occupation
title, the two-digit SOC major group, and the major group's name. 22 of the 23
major groups appear (no military occupations).

Columns: `title, soc_code, soc_title, major_group, major_group_title`.

Source: O*NET 31.0 Database, National Center for O*NET Development,
https://www.onetcenter.org/database.html. O*NET is a trademark of the
U.S. Department of Labor, Employment and Training Administration (USDOL/ETA).
Licence: CC BY 4.0.

Rebuild: `uv run python utilities/build_onet_titles.py`.

## `chicago_payroll.csv.gz`

8,000 rows drawn at random (seed 526) from the 24,842 full-time salaried
positions in the City of Chicago's "Current Employee Names, Salaries, and
Position Titles" dataset (31,754 rows on 2026-09-02). Employee names are
dropped.

Columns: `job_title, department, annual_salary`.

Source: City of Chicago Data Portal, dataset `xzkq-xp2w`,
https://data.cityofchicago.org/Administration-Finance/Current-Employee-Names-Salaries-and-Position-Title/xzkq-xp2w,
accessed 2026-09-02. Public data released under the portal's terms of use.

Rebuild: `uv run python utilities/build_chicago_sample.py`.

## `financial_phrasebank.csv.gz`

2,264 sentences from financial news with a sentiment label, the subset on which
all annotators agreed (`Sentences_AllAgree.txt` in FinancialPhraseBank-v1.0):
1,391 neutral, 570 positive, 303 negative.

Columns: `sentence, label`.

Source: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., and Takala, P. (2014),
"Good debt or bad debt: Detecting semantic orientations in economic texts",
Journal of the Association for Information Science and Technology 65(4),
782-796. Archive hosted at https://huggingface.co/datasets/takala/financial_phrasebank.
Licence: CC BY-NC-SA 3.0.

Rebuild: `uv run python utilities/build_phrasebank.py`.

## `newswire_sample.csv.gz`

14,280 wire-service articles sampled from the Newswire dataset: 40 per year for
1900-1946 and 400 per year for 1947-1977 (seed 526). The yearly JSON files on
the Hub total 35 GB, so the sample is drawn by HTTP range requests: 1 MB
windows at uniformly random byte offsets, keeping every complete record inside
a window, de-duplicating on the article text, and then sampling down to the
yearly target. Records that straddle a window edge are dropped, which slightly
under-samples very long articles. Years have fixed counts, so pooled statistics
are not corpus-representative; report by year.

Columns: `year, date, text` (the `cleaned_article` field), `ca_topic`,
`wire_state, wire_country` (the dateline location), `n_papers` (number of
newspapers that ran the article), `paper_states` (the state of each of those
newspapers, `;`-joined with duplicates kept), and the seven topic flags
`politics, civil_rights, labor_movement, antitrust, govt_regulation, protests,
crime`. The flags are the output of the dataset's own topic classifiers, not
hand coding.

Source: Silcock, E., Arora, A., D'Amico-Wong, L., and Dell, M. (2024),
"Newswire: A Large-Scale Structured Database of a Century of Historical News",
arXiv:2406.09490. Dataset https://huggingface.co/datasets/dell-research-harvard/newswire,
DOI 10.57967/hf/2423. Licence: CC BY 4.0.

Rebuild: `uv run python utilities/build_newswire_sample.py` (a few minutes,
about 120 MB of range requests).

## `work_stoppages.csv`

Major work stoppages (1,000 workers or more) beginning in each year, 1947-1977,
and the workers involved in thousands, from the Bureau of Labor Statistics
annual listing.

Columns: `year, stoppages, workers_thousands`.

Source: U.S. Bureau of Labor Statistics, Work Stoppages program,
https://www.bls.gov/web/wkstp/annual-listing.htm. Public domain. The BLS site
refuses requests from outside the United States, so the script reads the
Internet Archive's capture of the page from 2025-12-31.

Rebuild: `uv run --with lxml python utilities/build_work_stoppages.py`.

## Spatial data

U.S. Census Bureau public-domain sources, downloaded 2026-09-09. The files
retain every data record. The build script converts formats, removes presentation
headers and footnotes, and gives columns short names. Sample selection, geographic
IDs, joins, projections, and measure construction run in `slides/spatial_data.qmd`.

- `spatial_commuting.csv.gz`: 122,335 rows by 10 columns from
  [2016–2020 ACS commuting flows, Table 1](https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2020/commuting-flows-2020/table1.xlsx).
  Excel converted to compressed CSV with named columns. Eight opening rows and
  four closing notes are omitted; geographic codes retain their leading zeros.
  Estimates cover workers age 16+; margins of error are at 90% confidence.
- `spatial_counties.zip`: the unchanged
  [2020 cartographic county boundaries, 1:500,000](https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip).
  Contains 3,234 features in NAD83 (`EPSG:4269`). These are simplified cartographic
  boundaries, not survey-grade boundaries.
- `spatial_population_centers.csv`: the
  [2020 county mean centers of population](https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt),
  with 3,221 rows and seven columns. Longitude and latitude columns are named
  `pop_lon` and `pop_lat`. Coordinates use NAD83 and represent residents, not jobs.
- `spatial_places.csv`: the
  [2020 Washington places Gazetteer](https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_gaz_place_53.txt),
  converted from tab-separated text to CSV, with 639 rows and 12 columns.
  Column names are trimmed, with `NAME`, `INTPTLONG`, and `INTPTLAT` renamed
  `city`, `lon`, and `lat`. Place names and NAD83 internal-point coordinates
  are retained.

Economic application: Monte, F., Redding, S. J., and Rossi-Hansberg, E. (2018),
"Commuting, Migration, and Local Employment Elasticities", American Economic
Review 108(12), 3855–3890, https://doi.org/10.1257/aer.20151507.
The lecture estimates the commuting-gravity specification on a newer regional
sample; it does not reproduce the paper's national estimates.

Maintainer commands, from the repository root:

```bash
uv run --extra spatial python utilities/build_spatial_data.py
uv run quarto render slides/spatial_data.qmd --no-cache
```

The script downloads about 18 MB and saves about 14 MB. Ordinary lecture
execution loads these files and writes `build/spatial_county_analysis.csv`.
