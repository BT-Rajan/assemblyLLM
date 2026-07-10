# assemblyLLM

A tiny from-scratch character-level Transformer (8 layers, 384-dim, ~90-char
vocab) trained to translate natural-language questions about a ballpoint pen's
assembly into a small routing JSON (`lookup`, `trace_downstream`,
`matrix_check`). The trained model is exported to ONNX and runs entirely
client-side in the browser via `onnxruntime-web` — no server, no API calls.

- `train.py` / `model.py` / `dataset.py` / `config.py` — training pipeline
- `inference.py` — reference Python inference + routing logic
- `export_onnx.py` — exports `checkpoints/best_model.pt` to `docs/assembly_router.onnx`
- `docs/` — the static demo site, published via GitHub Pages

## Test it live via GitHub Pages

### 1. Enable Pages (one-time, if not already on)

1. On GitHub, go to **Settings → Pages** for this repo.
2. Under **Build and deployment → Source**, choose **Deploy from a branch**.
3. Set **Branch** to `main` and the folder to **`/docs`**, then **Save**.
4. GitHub will publish the site at:

   ```
   https://bt-rajan.github.io/assemblyLLM/
   ```

   The first deploy can take a minute or two. You can check progress under
   the repo's **Actions** tab (or **Settings → Pages**, which shows a "your
   site is live at ..." banner once it's ready).

### 2. Use the demo

1. Open the Pages URL above.
2. Wait for the status bar to switch from *"Initializing WebAssembly Runtime
   & loading model configurations..."* to **"✓ ONNX WebAssembly Runtime
   Active."** — this means `assembly_router.onnx` (~57MB, loaded once and
   cached by the browser) and the tokenizer vocab have both loaded
   successfully.
3. Type a question about the pen assembly, e.g.:
   - `What is the thread_specification profile assigned to component P008?`
   - `If part P001 breaks down, what downstream units lose operational integrity?`
   - `Check the codified design matrix connection mapping from P005 to P006.`
4. Click **Analyze Assembly State**. Generation is genuinely slow — the
   model runs one full forward pass per output character (~75-80 passes for
   a typical answer), and GitHub Pages doesn't send the headers WASM
   threading needs, so it runs single-threaded. Expect **~15-30 seconds**
   per query. The button disables and the output panel streams live
   token/elapsed-time progress while it's working, so it won't look frozen
   — just give it a bit. You'll see two panels once it finishes:
   - **Neural Network Output Token** — the raw JSON the model generated
     (e.g. `{"action": "lookup", "target": "P008", "property": "thread_specification"}`)
   - **Interceptor Graph Registry Ground Truth** — the deterministic answer
     looked up from the hardcoded parts graph once the model's JSON is parsed

If step 2 never turns green, open the browser console (F12) — the error
there will point at the actual failure (e.g. a blocked/slow CDN fetch for
`onnxruntime-web`, or the model files not being present in `docs/`).

### 3. Test it locally instead

GitHub Pages isn't required — any static file server works, since
`docs/index.html` only fetches files relative to itself:

```bash
cd docs
python3 -m http.server 8000
```

Then open `http://localhost:8000/`. This is the fastest way to iterate on
`docs/index.html` without waiting on a Pages deploy — just refresh the page
after editing.

> Don't open `docs/index.html` directly as a `file://` URL — browsers block
> `fetch()` for local files, so the model and vocab won't load. It needs to
> be served over HTTP (locally or via Pages).

## Known limitation

This is a genuinely tiny model trained on a small synthetic dataset, so it
occasionally predicts a plausible-but-wrong value (e.g. the correct JSON
shape with the wrong `target`/`destination` field). That will surface as a
`[ROUTING ERROR]` or `[COMPLIANCE ALERT]` in the Ground Truth panel even
though the demo itself is working correctly — it's a model-accuracy limit,
not a bug in the inference pipeline. Retraining with more data/steps
(`train.py`) is the way to improve it, not the demo code.

## Retraining / re-exporting

```bash
pip install torch onnx networkx

python3 prepare_data.py   # regenerate data/train.jsonl, data/eval.jsonl if needed
python3 train.py          # writes checkpoints/best_model.pt + data/tokenizer_vocab.json
python3 export_onnx.py    # writes docs/assembly_router.onnx(.data)
```

If you retrain, remember to also refresh the copy of the vocab file the demo
actually reads:

```bash
cp data/tokenizer_vocab.json docs/tokenizer_vocab.json
```

(`docs/index.html` derives `VOCAB_SIZE` from this file directly, so it will
stay in sync automatically as long as this copy step isn't skipped.)
