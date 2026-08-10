# Muse Glimmer 30B on H100 with llama.cpp

Muse Glimmer can be used as an agent-provider model through `llm-serve`, but its
official quantized release is GGUF rather than a vLLM-native checkpoint. The
working stack is:

```text
OpenCode / Hermes / Pi -> llm-serve :11424 -> llama-server :8080 -> H100
```

The `llamacpp` backend is opt-in. Existing `vllm`, `ollama`, and `mock` presets
keep their previous runtime and validation paths.

## What is official

Meta publishes these files in
[`meta-models/Muse-Glimmer-30B-GGUF`](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF):

- `muse-glimmer-30B-kquant-dynamic.gguf`: 19.65 GB, the higher-quality official
  quantization intended for higher-VRAM systems. Meta reports 0.2% average
  degradation across its measured benchmarks.
- `muse-glimmer-30B-kquant-17gb.gguf`: 16.76 GB, intended for 24 GB systems.
- `dflash-kquant.gguf`: optional speculative-decoding drafter.
- `mmproj-kquant.gguf`: optional vision encoder. The two main GGUF files are
  text-only without it.

The H100 preset uses K-Quant-Dynamic, a 131,072-token context, one request slot,
and Q8 KV cache. Muse supports `low`, `medium`, `high`, and `xhigh` reasoning
strengths; `llm-serve` maps the OpenAI-compatible `reasoning_effort` field to the
model's `reasoning_strength` template variable.

## 1. Build a compatible llama.cpp

Muse support entered llama.cpp in
[`ggml-org/llama.cpp#26841`](https://github.com/ggml-org/llama.cpp/pull/26841).
Use a checkout containing merge commit `62bf73d25c53b8161f8a22894d4f90c4aebbd7d0`
or anything newer:

```bash
export LLAMA_CPP_DIR=/absolute/path/to/llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
git -C "$LLAMA_CPP_DIR" checkout 62bf73d25c53b8161f8a22894d4f90c4aebbd7d0
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP_DIR/build" --config Release --target llama-server -j
```

If the checkout already exists, fetch and check out a newer `main` instead of
cloning it again.

## 2. Download weights into this repository

Run these commands from the `llm-serve` root. Both the final files and the
Hugging Face cache remain under `models/`:

```bash
mkdir -p models/muse-glimmer-30b
HF_HOME="$PWD/models/.cache/huggingface" \
HF_HUB_CACHE="$PWD/models/.cache/huggingface/hub" \
HF_HUB_DISABLE_PROGRESS_BARS=0 \
uvx --from huggingface-hub hf download \
  meta-models/Muse-Glimmer-30B-GGUF \
  muse-glimmer-30B-kquant-dynamic.gguf \
  --local-dir models/muse-glimmer-30b
```

For optional DFlash acceleration, download its companion file the same way:

```bash
HF_HOME="$PWD/models/.cache/huggingface" \
HF_HUB_CACHE="$PWD/models/.cache/huggingface/hub" \
HF_HUB_DISABLE_PROGRESS_BARS=0 \
uvx --from huggingface-hub hf download \
  meta-models/Muse-Glimmer-30B-GGUF \
  dflash-kquant.gguf \
  --local-dir models/muse-glimmer-30b
```

## 3. Start llama-server on the H100

Start with the base model only. `--fit off` prevents llama.cpp from silently
reducing the requested context, and `--alias` must match the llm-serve preset:

```bash
"$LLAMA_CPP_DIR/build/bin/llama-server" \
  --model "$PWD/models/muse-glimmer-30b/muse-glimmer-30B-kquant-dynamic.gguf" \
  --alias meta-models/Muse-Glimmer-30B-GGUF \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 131072 \
  --parallel 1 \
  --n-gpu-layers all \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --fit off \
  --jinja \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 64 \
  --log-timestamps \
  --log-verbosity 3
```

Verify the exact API alias in another terminal:

```bash
curl -sS http://127.0.0.1:8080/v1/models
```

After the base configuration is stable, DFlash can be added to the same launch
command:

```text
--spec-draft-model models/muse-glimmer-30b/dflash-kquant.gguf
--spec-type draft-dflash
--spec-draft-ngl all
--spec-draft-n-max 16
```

Benchmark before keeping DFlash enabled; the best speculative settings can vary
with prompt length and sampling.

## 4. Start llm-serve

In a second terminal, from the repository root:

```bash
cp .env.example_muse_glimmer_30b_kquant_dynamic_h100_80gb .env
uv sync --extra dev
./run-server.sh
```

This backend only needs the normal project dependencies; the vLLM runtime extra
is not required. Startup verifies `GET /v1/models` but does not run a generation
self-test or concurrency profile in this preset.

Verify both the gateway and a tool-capable chat request:

```bash
curl -sS http://127.0.0.1:11424/healthz
curl -sS http://127.0.0.1:11424/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-models/Muse-Glimmer-30B-GGUF",
    "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
    "reasoning_effort": "low",
    "max_tokens": 64
  }'
```

## OpenCode model entry

Point OpenCode at `llm-serve`, not directly at port 8080:

```json
{
  "provider": {
    "llm-serve": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local llm-serve",
      "options": {
        "baseURL": "http://127.0.0.1:11424/v1",
        "apiKey": "local"
      },
      "models": {
        "meta-models/Muse-Glimmer-30B-GGUF": {
          "name": "Muse Glimmer 30B K-Quant H100",
          "limit": {"context": 131072, "output": 32768}
        }
      }
    }
  },
  "model": "llm-serve/meta-models/Muse-Glimmer-30B-GGUF"
}
```

Use Chat Completions for this gateway backend. The gateway's stateful Responses
API remains specific to its supervised vLLM workers.

## Memory and concurrency

The H100 has ample static-weight headroom for the official 19.65 GB quantized
model, but a 128K KV cache and long prefills still consume meaningful memory.
The preset and llama-server command therefore both use concurrency 1. Raising
only `PROMPT_MAX_PARALLEL` does not create more llama-server slots; raise
`--parallel` as well, then measure memory with real long-context prompts.

