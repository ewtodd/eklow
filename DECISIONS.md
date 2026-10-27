# DECISIONS

Dated decisions with the evidence behind them. Newest last.

## 2026-10-24 — a custom engine for the A770M

On fiona, llama.cpp's Vulkan backend gives Gemma 4 12B about 100-250 tok/s of prefill and ~22
tok/s of decode, and cannot finish a prompt past ~12k tokens (next entry). son-of-anton's first
reply took about two minutes on a cold cache. The owner chose a from-scratch engine in unholy's
style (DESIGN.md) over tuning llama.cpp: C++20 + SYCL on Level Zero, own XMX kernels, weights from
Google's QAT Q4_0 GGUFs, the 12B first, then the 26B A4B, keeping whichever wins the son-of-anton
tool-call eval. Prefill matters more than decode (>= 20 tok/s is enough). Name: eklow, for Eklow
Labs (Silicon Valley, season 5), where fiona came from.

## 2026-10-24 — the driver: xe, and why llama.cpp fails on long prompts

The A770M runs on xe (`i915.force_probe=!5690 xe.force_probe=5690`; the Iris Xe stays on i915).
xe kills any GPU job over 5 s (sysfs `job_timeout_ms`, capped at 10 s by
`CONFIG_DRM_XE_JOB_TIMEOUT_MAX`; no module parameter). llama.cpp Vulkan submits a whole prompt
chunk's graph as one job, so prompts past ~12k tokens are killed. Under i915 the same jobs ran
past its 10 s fence timeout and llama.cpp silently served `<unused49>` garbage. The owner chose
not to patch kernel timeouts before eklow; eklow keeps every submission short instead.

Level Zero under xe, checked with the fleet's nixpkgs (c3f15ff4: intel-llvm 7.1.1, level-zero
1.32.0, compute-runtime 26.31.39395.13, IGC 2.40.13): Level Zero and OpenCL see both GPUs; JIT and
AOT (`-fsycl-targets=intel_gpu_dg2_g10`) kernels run; single Level Zero kernels of 12-30 s complete
without an xe job timeout (not tried while another process used the GPU). Free-memory queries
need `UR_L0_ENABLE_ZESINIT_DEFAULT=1 UR_L0_ENABLE_SYSMAN_ENV_DEFAULT=0`; Sysman's environment
path fails under xe.

## 2026-10-24 — M0 in son-of-anton: the prompt diet stays, the off-the-shelf checker does not

The prompt diet (son-of-anton 85300ef3) cut the prefix before the user's message from 15,342 to
10,425 tokens with no loss on the Gemma probe (149 vs 155 calls, 11 vs 14 errors).

A tool-call checker was evaluated on 1,082 hand-labelled steps from Gemma Signal probes plus 201
synthetic mistakes. Generic "is this step ok" questions ranked a real bad call above a good one
69-76% of the time (AUROC) for Gemma 12B as a judge, decider-2b/4b and StartLux 4B. Targeted
questions (repeat, progress, valid arguments; claims an action it did not take, gives up while a
tool could help, unsupported facts) on Bespoke-Nimble-9B-v3, combined by a fitted logistic model,
reached 0.81 for calls and 0.86 for final replies, held out by scenario. Wired into son-of-anton
as advisory notes plus one re-prompt per turn (branch `step-checker`, 664274f2), a 54-run A/B gave
35/36 scored scenarios against 34/36 without it, within noise, for 18% more model calls and 11%
more wall time; it also pushed an honest "I can't book a table" into a pointless search. Parked.

So M5's checker is a model fine-tuned on son-of-anton's own labelled steps, not an off-the-shelf
decision model. The labelled set and the bake-off code are in `~/Software/eklow-data/checker-2026-10-24`
on e-desktop (not in this repo: the items carry snippets of scraped web pages).

## 2026-10-24 — toolchain and layout

- nixpkgs `intel-llvm.stdenv` (its wrapped clang) builds everything; the flake pins the fleet's
  nixpkgs so host and engine agree on the driver stack.
- AOT device code goes through `ocloc`, which dlopens `libigc.so.2` by soname; the build puts
  intel-graphics-compiler on `LD_LIBRARY_PATH` (nix/package.nix).
- fiona's system config carries intel-compute-runtime (`/etc/nixos` ee3be49b). nixpkgs'
  `libze_loader.so.1` has no runpath and dlopens the driver itself, so a runpath on eklow's
  binaries would not reach it; the package wraps them with `/run/opengl-driver/lib` on
  `LD_LIBRARY_PATH`. Checked on fiona: the store-built probe runs with no dev shell. The dev shell
  carries its own copy of the drivers, so it works on any host.
- Same working method as unholy: this repo is the source of truth; fiona builds an rsync of it
  without `.git` in `~/eklow`.

## 2026-10-24 — the GGUF loader

`src/model/gguf.cpp` maps the file read-only and checks everything it reads against the bytes
left: counts before allocating, alignment, tensor bounds and overlaps, known types, row lengths
against block sizes, size overflow. On `gemma-4-12b-it-qat-q4_0.gguf` (sha256 93567e57...)
`tools/ref/gguf_compare.py` finds all 50 metadata keys and 667 tensors identical to the gguf Python
package: Q4_0 everywhere except the Q6_K token embedding (0.77 GiB), F32 norms and scales.

## 2026-10-24 — the probe: what the A770M offers

`eklow-probe`, record in `docs/probe/fiona-2026-10-24.json` (Gemma's server stopped, the
embeddings server running on the card):

| | measured |
|---|---|
| device memory | 15.11 GiB reported; 15.00 GiB allocated in 256 MiB chunks, written and read back |
| host <-> card, pinned | 11.76 GB/s up, 11.33 GB/s down (PCIe Gen4 x8; ~74% of the link) |
| host <-> card, pageable | 10.5 GB/s up, 6.4 GB/s down |
| kernel launch | 5.4 µs submit+wait, 3.3 µs per kernel back to back, 1.15 µs on the device |
| host flag -> running kernel -> host | 3.9 µs median (atomic load), 2.9 µs (uncached load) |

The host-signal row has a rule attached: memory from `sycl::malloc_host` is never seen by a running
kernel under xe, whatever the load (300 of 300 samples lost with atomic loads; 100 of 100 with an
added system fence, and with an uncached load hint). Memory from `zeMemAllocHost` with
`ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED` works. Any host-to-GPU signal (the 26B's cold-expert
handshakes) must use the Level Zero allocation.

Two probe bugs on the way: an integrated GPU reports the host's RAM as device memory, so the first
version picked the Iris Xe and its allocation test ran fiona out of memory (the OOM killer took the
little-titles server, which restarted, and the fiona user manager). The probe now considers only
Level Zero GPUs, never runs the allocation test on an integrated one, caps it at the reported
size, and stops if the host's available memory falls by more than 512 MiB.

## 2026-10-24 — XMX: int8 activations times the GGUF's own 4-bit weights

`eklow-xmx-bench`, record in `docs/bench/xmx-fiona-2026-10-24.json` (register-resident loops, no
memory traffic, except the last row):

| inner loop | measured |
|---|---|
| fp16 / bf16 `dpas` | 98.8 TFLOPS |
| s8 `dpas` | 197.6 TOPS |
| s8 activations x s4 weights | 197.6 TOPS |
| s4 `dpas` | 395.3 TOPS |
| s8 x s4 plus the Q4_0 block scales after each K=32 `dpas` | 222.8 TOPS |
| Q4_0 unpacked to fp16 in registers, then fp16 `dpas` | 26.6 TFLOPS |
| VRAM read (block loads) | 399 GB/s |

The rates stand in exact 1:2:4 ratios and match 128 fp16 FLOP per clock per EU at about 1.5 GHz
sustained; the driver reports a 2050 MHz maximum, the A770M's rated boost is 1650 MHz.

Layouts, each checked against a CPU reference: the B (weight) operand of fp16 `dpas` packs pairs of
consecutive k per column, s8 groups of 4. With s8 activations and s4 weights, B packs groups of 8
consecutive k per column (one dword), not 4; groups of 4 give wrong sums.

Decision: prefill GEMMs quantize activations to int8 per 32-element block, as llama.cpp's MMQ path
does, and feed the Q4_0 nibbles to `dpas` as s4 (q - 8 is the nibble with bit 3 flipped), applying
the weight and activation block scales after each K=32 `dpas`. The scale work hides under the XMX
pipe; unpacking to fp16 is 8x slower. fp16/bf16 `dpas` stays for attention.

What it means for the targets: the 12B's weight GEMMs cost about 22 GFLOP per token, ~4,500 tok/s
at half the int8 rate, so the 1500 tok/s prefill target at 16k leaves room for attention. Decode
reads ~6.5 GB per token: 399 GB/s caps one stream near 60 tok/s.

Measurement notes: the first runs used 11-29 ms kernels and came out 25-50% low because the clock
was still ramping; each kernel now runs once untimed and is timed best of three at ~90 ms. A
`memset` of 0x01 had also made the fp16 inputs denormal.

## 2026-10-24 — the oracle: llama.cpp's CPU backend, and its noise floor

`eklow-oracle` (tools/ref/oracle) links the llama.cpp fiona serves (the fleet's ea0c4c2c), built
for AVX2, and runs one CPU forward pass over a prompt with an fp32 KV cache and no flash
attention, dumping any graph tensor by name prefix plus the logits of every position.

Two llama.cpp backends on the same GGUF set the scale for "correct": on a 25-token Gemma 4 chat
prompt the CPU and CUDA builds agree on the top-1 token at 24 of 25 positions, mean KL 4.7e-3
(max 3.9e-2); per-layer outputs drift from 2% relative L2 at layer 0 to 18% at layer 47, with
cosine never below 0.98. eklow against the oracle should land in that band.

A false alarm on the way: on the raw text "The capital of France is", both backends predict
digits ("011111111111"). That is the instruction-tuned model off its chat format, not a bug; with
the chat format both put probability 1.0 on "The". Checks use chat-formatted prompts.

## 2026-10-24 — the tokenizer follows Hugging Face, not llama.cpp

Google's `tokenizer.json` for Gemma 4 (in e-desktop's Hugging Face cache with the bf16 weights)
defines: the 24 added tokens split out first; spaces replaced by U+2581; a pre-tokenizer that
splits on spaces, of which none remain, so each stretch between added tokens is one BPE word; BPE
with byte fallback, lowest merge rank first and leftmost first among equal ranks. llama.cpp also
splits on newlines before merging; eklow does not. The GGUF holds everything needed: its 17
control and 7 user-defined tokens are exactly the 24 added tokens, and its 256 byte tokens the
fallback.

`tools/ref/tok_compare.py` runs Hugging Face's `tokenizers` on `tokenizer.json` against
`eklow-tok`: hand-written edge cases (whitespace runs, CRLF, every script at hand, emoji with ZWJ
and modifiers, combining marks, control characters, partial and adjacent special tokens), the same
wrapped in the chat template, 20,000 random strings over a mixed alphabet, and five real files
whole and line by line. 27,607 strings, 1.34M tokens: no difference in encoding or decoding.
Reversing the tie-break among equal ranks gives 9 differences in 3,142 strings, so the comparison
catches the rule that matters. Encoding runs at 1.5M tokens/s on one core; loading takes 0.10 s.

## 2026-10-24 — the chat template is ported, not interpreted

The plan was the official Jinja template through an embedded interpreter, most likely llama.cpp's
own engine (common/jinja, ~6,500 lines, MIT). Its record on Gemma 4 argued against it:

- #21547 (open since April): Gemma 4 requests with tools fail with "NotImplemented: map:
  filter-mapping not implemented", the `map` filter this template uses.
- #28249 (open): the engine can mark request text so it cannot become special tokens, but
  llama.cpp never wired it up; #24382 showed `A<|turn>B` in a user message tokenized as a real
  turn boundary.
- #30834 (open): deeply nested JSON in a request overflows the stack in the engine's value layer
  and kills the server; #29706: nested `range` loops ran 2^N iterations.
- The engine is from January 2026 (b7756) and still taking fixes in features this template uses
  (`select`/`selectattr`, #31059, after the pinned revision).

The tool-call bugs reported against Gemma 4 (#21384, #22786, #21660, #29089) were in llama.cpp's
output parser and grammar, which eklow replaces either way.

eklow serves one model family, as unholy serves one model; unholy ported DeepSeek's Python encoder
rather than embedding Python. So `src/text/chat_gemma4.cpp` ports Google's template (390 lines,
"published 2026-07-09"; the GGUF's copy is byte-identical to the Hugging Face repo's) statement by
statement, with the Jinja2 and Python semantics it depends on: `{{ x }}` is Python's `str()` (float
repr, list and dict repr for list-typed schemas through `| upper`), `dictsort` is case-insensitive
and stable, `trim` strips Python's Unicode whitespace, Undefined differs from None (`x['k']` versus
`x.get('k')`; Undefined iterates as empty and is a "sequence"), tool-response names resolve with
`None == None`. Case mapping covers Latin, Greek and Cyrillic; JSON numbers keep Python's int/float
distinction. JSON nesting is capped at 128 levels.

`tools/ref/chat_compare.py` renders with transformers 5.17 `apply_chat_template` (Jinja2 3.1.6)
and with `eklow-chat`: 264 hand-written combinations of each branch (system/developer, tools,
tool calls and responses, legacy `tool_responses`, thinking channels, reasoning fields, images,
continued assistant turns; generation prompt, `enable_thinking`, `preserve_thinking`), 4 inputs
both must reject, son-of-anton's real requests (the dieted and pre-diet prompts with their full tool
schemas) and probe transcripts, and 20,000 random conversations with random tool schemas and
argument values. 20,416 cases: identical text on every one that renders, both reject the 4.
Making `dictsort` case-sensitive gives 568 differences in 2,268 cases.

The render comes back in parts, template text and request text. Tokenization finds special tokens
only in template text and runs BPE over everything between them, so it equals Hugging Face's
tokenization of the joined prompt whenever no request text spells a special token; checked on
3.82M tokens against an independent reference built on Hugging Face's tokenizer. `A<|turn>B` now
tokenizes as A, <, |, turn, >, B. A GGUF whose `tokenizer.chat_template` is not this template
(FNV-1a fingerprint) is refused rather than rendered wrong; a new Google template means a new port
and a new comparison run.

## 2026-10-24 — the reference forward pass matches the oracle within its noise floor

`src/kernels/ref.cpp` holds plain SYCL reference kernels (fp32 activations and accumulation,
Q4_0 and Q6_K dequantized as ggml stores them, RoPE angles computed on the host in double since the
A770M has no fp64, every launch split well under xe's 5 s timeout) and `src/model/gemma4.cpp` the
12B's forward pass, following llama.cpp's gemma4 graph: embeddings times sqrt(3840); q/k norms with
weights, v norm without; V is K's projection, before K's norm, on the global layers; attention scale
1; NeoX RoPE, base 1e4 on sliding layers and 1e6 divided by `rope_freqs` on global ones; sliding
layers see the last 1024 positions; GeGLU with tanh gelu; post-attention and post-FFN norms;
`layer_output_scale`; tied Q6_K head with a 30 softcap. `eklow-ref-run` dumps in the oracle's format.

| | eklow reference vs CPU oracle | CUDA llama.cpp vs CPU oracle |
|---|---|---|
| 25-token chat prompt: layer outputs, relative L2 | 0.7-2.6% (cosine >= 0.9997) | 2-18% |
| 25-token chat prompt: top-1, mean KL | 25/25, 3.9e-3 | 24/25, 4.7e-3 |
| 2,616 tokens (DESIGN.md as a user turn): layer 35 / 46 | 10.8% / 8.8% | 9.8% / 8.5% |
| 2,616 tokens: top-1 over the last 32, mean KL | 31/32, 1.9e-2 | 30/32, 1.75e-2 |

The first matmul already differs by 0.4%, which is the oracle quantizing activations to Q8_0 for
Q4_0 matmuls where the reference stays in fp32. On the long prompt the error grows with depth and
with position for eklow and for CUDA alike, with no step at position 1024, so the sliding window and
long-range RoPE are right; the drift is the quantized model's own sensitivity, which two llama.cpp
backends show against each other. The reference runs 25 tokens in 12 s and 2,616 in 135 s on the
A770M; speed is M4's job.

## 2026-10-24 — the output parser reads tokens, not text

Gemma 4 writes `<|channel>thought\n...<channel|>`, then `<|tool_call>call:NAME{ARGS}<tool_call|>`
calls, then content, ending at `<turn|>`, `<eos>` or `<|tool_response>`. ARGS is the template's
format_argument with bare keys and strings between `<|"|>` tokens. llama.cpp parses the decoded
text and has broken on strings holding braces or commas (#21384, #29655); `src/text/output_parser.cpp`
works on token ids, where `<|"|>` is one special token, so string contents never confuse it. Text is
released only as whole UTF-8 characters; the template's newline before `<channel|>` is held back
until the next token shows whether the channel closes. Keys holding `:` or `,` cannot be recovered
(the template does not escape keys).

`tools/ref/parse_compare.py` renders random assistant turns after a user turn with the verified
template port, tokenizes them, feeds the generated part to `eklow-parse` one token at a time, and
compares: 2 x 20,000 turns (reasoning, 1-3 tool calls with nested arguments of every JSON kind,
content) recovered exactly. Keeping the template's trailing newline in the reasoning shows up as
974 differences in 2,000.

## 2026-10-24 — the first XMX matmuls, and what limits them

`src/kernels/xmx.cpp`: Q4_0 weights are repacked at load into dpas tiles (8 columns x 32 K, 128
bytes, signed nibbles in groups of 8 along K, K-block-major so that threads working on the same K
step read adjacent memory) with fp16 scales; activations are quantized to int8 per 32-block by
eklow's own kernel straight into 8-row tiles with fp32 scales. `eklow-kernel-test` runs real
layer weights: every sampled output matches a CPU emulation of the quantized arithmetic to ~1e-6.

- One register file only: the 256-register mode (`grf_size<256>`) is ignored on DG2 (IGC dump:
  `numGRF=128`), so 16 x 32 output tiles are the largest that fit without spilling; 32 x 32 spills
  3.3 KB and runs at 15 TOPS.
- Decode (1-8 rows), a K-split GEMV: 110-240 GB/s of weights (ffn_down 243 GB/s), against 399 the
  card reads. Interleaving the weight tiles K-block-major gained 10-20%.
- 16-256 rows, a column-group kernel (each thread applies one weight tile to up to 64 rows): ~11
  TOPS; the tile GEMM was ten times slower there, rereading weight tiles for every 16 rows.
- 2048 rows, the 16 x 32 tile GEMM: 28-53 TOPS. Enough for the weight GEMMs to run at ~1,200-2,000
  tok/s, short of the 1,500 tok/s prefill target with attention on top; the fix is the standard one,
  work-group tiles staged through shared local memory, in M4.

## 2026-10-24 — the serving forward pass

`src/engine/engine.cpp` runs the 12B over a step: a list of chunks, each a prefill run or one decode
token of one of the engine's slots, continuing where that slot left off. Q4_0 matrices go through
the XMX kernels; everything else is a batched kernel in `src/kernels/fast.cpp` (embedding, RMS norm,
NeoX RoPE from a position array with fp32 angles as llama.cpp and Hugging Face compute them, GELU,
residuals, softcap) or the attention kernels.

- KV cache: fp16 in 256-position pages. Global layers take pages from a pool through per-slot page
  tables. Sliding layers keep a ring per slot of window + max_chunk positions (2,048 by default),
  so a step writes its chunk's K/V before attending and every key comes from the cache; the
  alternative, reading the chunk's own keys from the step's buffers, made the kernel branch per key.
- The output head (the tied Q6_K embedding, 806 MB) is split at load into aligned arrays of low
  nibbles, high bits, scales and super-block scales: the same bits and values, but a lane's loads
  are aligned and contiguous. The first version, a thread per 256-weight block over the raw 210-byte
  blocks, took 69 ms of a 124 ms decode step; the split layout takes 3.4 ms.
- Logits only for the rows asked for (the last token of a prefill chunk, each decode token).

Against the llama.cpp oracle on the 2,616-token prompt (the CPU-vs-CUDA floor: top-1 30/32, mean KL
1.75e-2): one chunk 30/32, 1.49e-2; 512-token chunks 31/32, 1.40e-2; 1,000-token chunks then 32
decode steps 31/32, 1.30e-2; with a second slot running the same prompt a step behind, 30-31/32 and
0.99-2.43e-2 for the two slots. On the 25-token chat prompt 24/25, 4.9e-3. Runs that differ only in
how the work was batched differ from each other by about as much as from the oracle: an int8
activation that rounds the other way after a last-bit difference perturbs the next layer, and 48
layers amplify it. One sequence prefilled in 900- and 1,000-token chunks gives bit-identical logits
(the GEMM is independent of the chunk), so the spread comes only from different kernels (GEMV for 1
or 2 decode rows, K-splits that depend on the batch's rows).

The GEMM chose its K-split and its row tiles from the activation buffer's allocated rows, not the
rows of the call: every prefill chunk was computed at the engine's full batch. Fixed in
`xmx::quantize`, which now records the padded rows of each call.

## 2026-10-24 — attention on the matrix units, and what a DG2 thread waits on

`src/kernels/attention.cpp`. The cache stores K and V as fp16 dpas operands (K: 8 positions x 16
dims, V: 16 positions x 8 dims, VNNI), each a contiguous 256-byte tile. Query rows are (token, head)
pairs sharing a K/V head, 8 to a tile; a work-group of head_dim/64 threads takes a tile, each
thread a 64-dimension slice of Q and of the output (DG2 has 128 registers of 32 bytes per thread,
so an 8 x 256 fp32 output does not fit one thread). Per 16 keys: partial Q.K^T on each slice,
summed through shared local memory, the online softmax by one thread per row, P.V on each slice.
Long key ranges split across work-groups and merge (flash-decoding). `eklow-attn-test` drives it as
the engine does (two slots, chunked prefill then decode, ring and page table) against
`ref::attention`: relative L2 1.0e-4 (sliding) and 3.5e-4 (global).

What the A770M made of it, from IGC's ISA dumps and a copy of the block loop with parts switched
off (`eklow-attn-test`, and a scratch micro-benchmark):

- Integer division by a runtime value is a library call on DG2, and 64-bit `/` and `%` in the
  kernel's index math (`__vc_builtin_sdiv_i64`, `srem_i64`) cost more than the attention itself:
  decode with one or two key blocks took 100-217 us. In 32 bits, with the query group size a
  template parameter, 35 us.
- Loads are the cost. Each 256-byte operand load is waited on before its dpas: ~375 ns per load
  from VRAM, ~125 ns from cache, so a 16-key block takes a thread ~7 us when its tiles come from
  memory, 0.9 us with them in registers (41-85 TFLOPS of dpas with no loads). Loading a block's
  tiles up front spills; L1 cache hints and prefetching the next block changed nothing measurable.
  Barriers and the softmax are minor (0.4 us per exchange).
- A dependent chain of dpas runs at ~230 ns per dpas: accumulators need independent chains.
- So decode wants many short splits, and their merging costs: at least 4 blocks per split, at
  most 32 splits beyond that, measured best.

Now (docs/bench/attn-fiona-2026-10-24.json): decode 92 us per sliding layer, 84 / 148 / 314 us per
global layer at 1k / 4k / 16k; prefill of a 1,024-token chunk 6.1 TFLOPS (sliding), 7.2-8.9 TFLOPS
(global). The engine (docs/bench/engine-fiona-2026-10-24.json): prefill 1,141 / 1,057 / 867 tok/s
over 1k / 4k / 16k prompts; decode 22.2 / 20.8 / 19.4 tok/s at 0 / 4k / 16k. At 16k, attention is
35% of prefill and the weight GEMMs at ~33 TOPS most of the rest. The next attention step is the
standard one: a work-group stages each K/V block through shared local memory with double buffering,
so several query tiles share one load and the next block's loads overlap this block's dpas.

## 2026-10-24 — fp16 GEMM

Taking the int8 GEMM apart the same way as attention (a copy with parts switched off): with every
load removed it still ran at 28-46 TOPS, and 67 with the per-block scaling removed too. Each s8 x s4
dpas is one Q4_0 block, and its 64 int32 results need converting and scaling by an activation and a
weight scale, about as much vector work as the dpas is matrix work; the vector side is the limit.

Prefill (more than 32 rows) now multiplies fp16 activations by weights dequantized to fp16 in
shared local memory: each Q4_0 block's tiles once per work-group of 16 x 4 threads (256 rows x 128
columns), the dpas accumulating in fp32 over all of K, so the scaling cost is per weight per
work-group instead of per output per block. Global loads for block b+1 go out before block b's dpas.
In isolation 48-52 TFLOPS on all four of the 12B's shapes (no loads, no barrier: 95). The products
are also closer to fp32: 2e-4 relative against the reference, against 1e-2 for int8 activations.
Gemma's matmul inputs stay well inside fp16 (largest 2,488, on the chat prompt's first FFN; 1,573
on the 2,616-token prompt). Decode stays on int8, where the weights' bytes are the cost.

## 2026-10-24 — prefill attention: shared K/V blocks, and where it stands

The target is the rate at 16k depth (DESIGN: ~25 GFLOP a token counting attention at 16k), which
`eklow-engine-bench --depth` now measures: a sequence filled to the depth untimed, then two chunks
timed. A 16k prompt from empty averages 8k of depth and flatters the number.

Prefill attention now runs a work-group per group of query tiles of one chunk and K/V head, 12
tiles (head 256) or 6 (head 512) with 16 loader threads: 64 threads, each 16-key K/V block staged
once in shared local memory. The sizes come from two measurements: an Xe-core holds 128 thread
slots, and gives shared local memory out in power-of-two sizes from 128 KB (a 33-64 KB work-group
lets two fit, 32 KB four), so 64 threads under 64 KB fill a core twice. Loaders and compute
threads run separate loops with the same barriers (one loop kept both register sets live and
spilled). 10.4 TFLOPS on sliding layers and 10.9 on global (from 6.1 and 7-9).

Timing the compute threads' phases with rdtsc: Q.K^T about a quarter of a block, P.V a fifth, and
each of the score exchange's two barriers about a sixth, waiting. The remedies that suggest
themselves do not fit head 512's shared memory: a double-buffered score exchange (one barrier a
block) or 32-key blocks would need 80-112 KB. Q kept in registers spilled 3 KB, skipping the output
rescale when no maximum moved spilled 1.7 KB; both ran slower.

At 16k depth the engine prefills 962 tok/s (2,048 tokens in 2.13 s): global attention 0.88 s at
~10.6 TFLOPS, the weight GEMMs 0.96 s at ~46.5 TOPS. 1,500 tok/s needs both: global attention near
20 TFLOPS and the GEMMs near 60.

## 2026-10-25 — the server, against what son-of-anton sends

`eklow-server` (tools/server, src/serve): /v1/chat/completions renders the request with the
template port, samples on the host and parses the output token by token; a scheduler thread runs
continuous batching over the slots (one decode token per generating sequence, then prefill chunks)
and gives a new request the free slot whose cached tokens share the longest prefix with its prompt.

The compatibility surface comes from a read of son-of-anton's client, not from OpenAI's spec:
- thinking is on unless the request says otherwise (reasoning_effort "none", think or
  chat_template_kwargs.enable_thinking false); llama.cpp's server thinks by default too (231 of
  239 baseline probe calls carried reasoning), and the agent's main turns send no reasoning field;
- reasoning goes in delta.reasoning_content, tool calls as indexed deltas, usage in a final chunk
  with empty choices and prompt_tokens_details.cached_tokens;
- an over-long prompt fails as llama.cpp's 400 exceed_context_size_error with n_prompt_tokens and
  n_ctx (the client matches the wording and compacts), while a max_tokens past the window is
  capped (the agent asks for 65,536 on a 64k window);
- image parts are refused with "image_url is not supported" / "does not support image input", which
  the client takes as "strip images for this session";
- /v1/models lists context_length first: the client takes the first context key it finds in an
  entry, which on llama.cpp's listing is n_ctx_train.

EmbeddingGemma 2 (llama.cpp's gemma-embedding2 graph: bidirectional attention, a symmetric
sliding window of 512 positions either side, gated per-layer inputs, a 768-wide projection, mean
pooling, L2 norm) runs on the reference kernels on its own queue; the house's llama.cpp embedding
server on fiona gives the same vectors to cosine 0.9996-0.99995.

## 2026-10-25 — int8 for the global layers' cache, and constrained tool calls

2 x 262k does not fit in fp16: the eight global layers' K/V would be 8.6 GB beside 6.5 GB of
weights and 1.3 GB of sliding rings. `--kv-int8` keeps them as int8 in the places the fp16 tiles
use, with an fp16 scale per position and 16 dimensions (2 x 262k: ~4.8 GB). Against fp16 on the
2,616-token prompt with decode the logits differ by mean KL 6.5e-3, under the oracle's noise floor
(1.75e-2). A single scale per position over all 512 dimensions would let decode skip the
per-element multiply (the K scale factors out of each score, the V scale into each probability)
but cost KL 2.1e-2 against fp16: K and V have outlier dimensions. Decode is slower with int8 (470
against 320 us a global layer at 16k; threads wait on loads, not bytes, and the conversion adds to
the chain), so fp16 stays the default and int8 is for contexts fp16 cannot hold. Prefill is
unchanged: its loader threads dequantize each block into shared memory.

Tool calls are constrained. Inside <|tool_call>...<tool_call|> the scheduler samples only tokens the
tool-call grammar allows (src/serve/tool_grammar.cpp): a pushdown automaton over the syntax the
output parser accepts, with the name limited to the request's tools and nothing but <tool_call|>
ending a call. A token is allowed if its bytes run through the automaton; the 262k-token vocabulary
is bucketed by first byte, inside strings everything but other special tokens is allowed outright,
and allowed sets are cached by state (mode, open containers, and the name or word so far) across
requests. eklow-grammar-test on the vocabulary: 5,000 random well-formed calls accepted token by
token; 4,997 random walks through the allowed sets closed, and every one parsed into a call of a
declared tool.

## 2026-10-26 — 2 x 262k by default

`eklow-server` and the NixOS module now give both slots Gemma's full 262,144-token window, with
the global layers' K/V in int8 (`--kv-fp16` for fp16). fp16 cannot hold it: the global layers alone
would take 8.6 GB beside the weights. With int8 the card carries 6.48 GiB of weights, 5.75 GiB of
KV (both slots' global pages and sliding rings) and 0.27 GiB of EmbeddingGemma 2, 12.5 of the 15
GiB it allocates.

End to end, through the server as the module runs it (`tools/server/needle.py`,
docs/bench/needle-2x262k-fiona-2026-10-26.jsonl): two prompts of 259,623 and 258,916 tokens
(son-of-anton's Python in two orders, each with a one-line note giving a passphrase, 2% and 55% of
the way in), sent at once. Both prefilled together in 36 min (first tokens at 2,172 and 2,184 s:
237 tok/s for the two, from empty, so about 18 min for one), with no xe job timeout, and each
answered with its own passphrase and an accurate summary of the code. Decode at 259k is about 11
tok/s per sequence with both decoding (the first slot's 8.0 includes steps that also carried the
other's last prefill chunks): each token reads the whole global cache. What int8 costs where fp16
also fits (`eklow-engine-bench`, docs/bench/engine-int8-fiona-2026-10-25.json): decode 31.6
against 31.9 tok/s at 0, 26.8 against 28.0 at 16k; prefill at 16k depth 943 against 965.

maitred lists the model as `eklow/gemma-4-12b` with a 262,144 window (the id names the engine,
as `unholy/` and `radiance/` do). son-of-anton asks for "auto", which takes the strongest serving
model whose window fits the conversation, so a long conversation can now stay on Gemma: a cold one
near the full window costs ~18 min of prefill, a cached one only its new tokens. For scale, a
10.6k-token prompt, the size of son-of-anton's prefix, takes 8.5 s to the first token cold and 0.2
s with the prefix cached (`tools/server/ttft.py`, 2026-10-25).

The same run shows embeddings starving behind chat: the three /v1/embeddings requests sent during
the prefill each timed out after 600 s; on an idle card they take 50-100 ms, and maitred gives the
embed upstream 120 s. The embedding model runs on its own queue, which keeps it out of the
scheduler's steps but evidently not from waiting on them. Why, and the fix: the next entry.

## 2026-10-26 — embeddings beside a long prefill

Timing each phase of `EmbedGemma2::embed` while a 49,836-token prompt prefilled
(`tools/server/embed_contend.py`, an embedding request every 5 s) cleared the first suspect, the
per-call device allocations and frees (0.0 ms). The time was all in the kernels: each of the ~650
in the 24 layers was waited for from the host and took ~90 ms to come back (0.1 ms on an idle
card), so an input took 60 s and finished only when the prefill did. xe gives each queue 1 ms
timeslices on the card's one compute engine (`ccs_mode` 1), but taking the engine back from a
running prefill costs tens of milliseconds, and every host wait paid that again.

On the same prefill (embedding latency during it; chat alone on a 10.6k-token prompt, first token
and decode, 2 x 262k int8):

| | embedding during the prefill | chat alone |
|---|---|---|
| as deployed | 60 s, one in the whole prefill | 8.68 s, 27.3 tok/s |
| chat queue at low priority | median 60 ms, worst 109 ms | 8.92 s, 24.1 tok/s |
| embedding queue at high priority | 60 s (ignored, unprivileged) | |
| no waits per kernel | 0.70 s, 1.45 s; 32 inputs a request 22-36 s | unchanged |
| no waits, and the scheduler held | 0.69 s, 1.43 s; 32 inputs 2.3 s, 3.1 s | 8.71 s, 27.4 tok/s |

Low priority for chat is the simplest fix for embeddings but costs decode 12%, for reasons not
looked into, so chat stays at normal priority. Instead the reference kernels no longer wait (every
queue here is in-order; the callers already waited before reading results or freeing buffers), the
embedding model builds its RoPE tables once with its activation buffers and runs a request's
inputs back to back with a single wait, and the endpoint holds the scheduler while it embeds
(`Scheduler::Hold`): the step under way finishes, no new one starts until the embeddings are done,
and a step always runs after each hold, so requests back to back cannot starve chat. An embedding
then waits at most for the step under way: 0.06-3.8 s at 135k tokens of depth, where steps take
~4 s; at 2 x 262k, with steps near 15 s, it should stay under 20 s (estimated), inside maitred's
120 s. Chat pays the embeddings' own time on the card, ~50 ms an input: 352 inputs during the 50k
prefill made it 18 s longer (80.3 against 62.3 s).

Unchanged by it: `eklow-embed`'s vectors identical to the bit on the four inputs of the comparison
with llama.cpp's server, 50 ms each on an idle card; the reference forward bit-identical to its
2026-10-24 dump on the 25-token chat prompt (all 48 layer outputs and the last row of logits);
`eklow-attn-test` and `eklow-kernel-test` as recorded.

## 2026-10-26 — decode attention: more splits need a cheaper merge

Decode attention reads its cache at 25-107 GB/s of the card's 399, and with the int8 global cache
the server now uses it set decode at long contexts (~11 tok/s at 259k). The reading was that too
few threads run: a decode token on a global layer gave 2 tiles x 32 splits x 8 threads = 512
threads of 4,096, each walking 32 blocks at ~7 us a block. More splits alone made it worse (global8
at 16k: 470 us with today's cap of 32 splits, 526 with splits of 4 blocks, 780 with 2) and slowed
short contexts by up to 40%, because the merge's work-items each loop over every split: with 128
splits the merge was half of the decode time (a timing run with it skipped).

So splits stay as they were up to 4k keys (the sliding layers, 40 of 48, always have ~1k) and are
at most 8 blocks beyond, and past 32 splits a kernel folds each group of 8 into one partial in the
same layout before the merge. A single-kernel rewrite of the merge, with the per-split weights
computed once per row, was slower: 16 work-items looped over 128 splits. eklow-attn-test at 16k keys:
global8 470 -> 302 us a layer, global 311 -> 306; 1k and 4k unchanged; splits of at most 4 blocks
gave 366. Engine decode with the int8 cache at 16k 26.8 -> 27.8 tok/s, 24.8 at 64k. The decode
kernel is now most of those 302 us, at ~111 GB/s: what limits it is how much one thread keeps in
flight, not how many threads run.

The work was done by Qwen 3.8 27B through son-of-anton one-shot runs, from step-by-step specs with
acceptance checks (`~/eklow-perf` on fiona holds the benchmark and oracle-check scripts the runs
used, with a lock so that two never share the card).

## 2026-10-26 — what holds the prefill GEMM: shared-memory load latency

The fp16 GEMM runs at 36-47 TFLOPS of 98.8. Per 32-K block a thread reads 12 x 256 bytes of shared
local memory for 16 dpas, 192 bytes a dpas, which looked like more than SLM delivers. Measured
(`eklow-xmx-bench`, 8,192 threads reading 256-byte blocks): SLM ~214 bytes a clock per Xe-core at
1.5 GHz, L1 ~204, both at once ~173, so bandwidth is not the limit and moving activations to L1
cannot help (SLM and L1 share a port).

Knockouts, each removing one cost (results wrong on purpose; TFLOPS at 2,048 rows averaged over
the six shapes `eklow-kernel-test` now runs, from 41.4):

| removed | speed-up |
|---|---|
| the inner loop's SLM loads (one A pair and one B reused) | +45% (to 68 on qkv) |
| the next block's global loads | +28% |
| the dpas | +18% |
| the barrier | +15% |
| the dequantization | +13% |

Without any dpas the kernel reaches only ~48 TFLOPS of loads and stores: the matrix units wait on
the staging. Each B tile is loaded from SLM right before its dpas, so every one of the 12 loads a
block is waited on in turn; the global loads compete for the same port. Next: fewer, larger SLM
loads (both K halves of a tile in one 512-byte load) and the next tile's load issued before the
current dpas.

## 2026-10-26 — decode GEMVs in one kernel; the GEMM at the engine's 1,024 rows

Each int8 GEMV was a flat grid of ~8,192 threads, one per (K split, 8-column group), writing
partial sums, plus a second kernel adding them; for the small matrices (attention output, 8.9 MB,
~22 us of reading) the two launches and the partials made it ~77 us. Now a work-group of T threads
per column group splits K and adds the partials in shared local memory. T must follow the shape:
a fixed T = 8 lifted the narrow matrices (attention output 124 -> 166 GB/s) but took the stacked
gate|up, 3,840 column groups and ~10 ms of a decode step, from 302 to 245 GB/s, so decode got 2%
slower overall. With T the largest power of two under target / column groups, one row aiming at
4,096 threads and 2-32 rows at 8,192 (4,096 left the two-row gate|up at T = 1, 232 GB/s): decode
+3.1 / +2.7 / +2.2% at 0 / 4k / 16k with one slot, +4.4 / +4.1 / +3.3% with two (`ab.sh`, three
rounds alternating with main).

The GEMM's B tiles are now loaded one column tile ahead (+4% in kernel-test at 2,048 rows, nothing
measurable in the engine alone), and loading both K halves of a tile at once ran out of registers
(-51%). The engine prefills one sequence in 1,024-token chunks, which kernel-test did not measure
until now: there the 256 x 128 work-group tile gives the narrow matrices (N 3,840) 120 groups,
under two waves of the card's 64; a 128 x 128 tile below 256 groups gives prefill +1.1-1.5% on
every row of the engine bench (963.8 tok/s at 16k depth). The smaller tile on every shape lost 1-3%:
the doubled weight traffic costs more than the tail saves. Two lessons from the round: kernel-test
at 2,048 rows twice predicted gains the engine did not show, so keep rules now use `ab.sh`; and a
geometric mean over shapes weighs a 9 MB matrix like a 236 MB one.

## 2026-10-26 — decode: weight tiles column-group major, and what did not help

The one-row GEMV's knockouts (`eklow-kernel-test`, six shapes) put its time in the arithmetic chain
(dpas, int-to-float, scale, add: +29% without it) and in the non-weight loads (+15%). Splitting the
accumulator four ways gave decode +0.7-1.1% and a gather of four blocks' scales was mixed (this
ESIMD's `gather<uint32_t, 16, 4>` returns its 4-dword groups transposed), so neither was kept. The
knockout that mattered read four tiles in one 512-byte load: +68% on average (partly cache overlap
between neighbouring threads), because the K-block-major layout put a thread's consecutive tiles
ng * 128 bytes apart, every one a separate message. With the tiles column-group major ([N/8][K/32],
through one `tile_at(b, g, kb)` for the repack, the GEMM and both GEMVs), the layout alone gave
decode +3.0 / +3.7 / +2.8% at 0 / 4k / 16k and the 512-byte loads another +5.8 / +6.0 / +5.7%;
prefill unchanged. The same four-tile load in the 2-32-row GEMV gave two-slot decode +1-2%, not kept.

Also tried and not kept: a compact one-row activation form (one 128-byte and one 16-byte load per
four blocks): kernel-test faster, engine decode 3-4% slower. Several columns per sub-group for the
Q6 output head (3.4 ms a decode step at ~237 GB/s): register spills, +1.5% and +63% slower. A
32-dimension slice for the global layers' decode attention removed its 768-byte int8 spill and was
21-26% faster at 1-4k keys but 4-6% slower at 16k with the int8 cache, where it matters. For the
sliding layers (head 256) the same slice removed a 928-byte spill: 83 -> 69 us a layer (fp16), 114 ->
75 (int8), engine decode +0.6-2.1% (kept).

Decode at 4k context now spends ~27 of ~33 ms in GEMVs (gate|up 11, down 5.6, qkv 3.6, head 3.4,
attention output 3.1), so they remain the lever; the card's launch gaps are ~1.15 us x ~650 kernels,
~0.75 ms.

## 2026-10-26 — Gemma 4 26B A4B: the reference, its noise, its routing

`model::reference_forward` runs the 26B (30 layers; 25 sliding with 8 K/V heads, 5 global with 2;
every layer a dense GeGLU of 2,112 plus 128 experts of 704, 8 used) as llama.cpp's gemma4 graph
does: dense path normed by ffn_norm / post_ffw_norm_1; experts on rmsnorm(attn_out) *
pre_ffw_norm_2; router on rmsnorm(attn_out) / sqrt(n_embd) * ffn_gate_inp.scale, softmax over 128,
top 8 renormalized (floor 2^-14), each expert's output times ffn_down_exps.scale; the sum normed by
post_ffw_norm. Against llama.cpp's CPU backend: chat 23/25, mean KL 3.7e-2; 2,616 tokens 31/32,
2.5e-2. That is inside the 26B's own noise, which is ~8x the 12B's: llama.cpp CUDA (a 4090, the
oracle built against llama.cpp's cuda package) against CPU gives 25/25 at 3.6e-2 and 31/32 at
1.9e-2, final-norm rel L2 11.8% (eklow 12.2%), presumably near-tied routes going to different
experts. The 26B's band: long top-1 >= 30/32 and mean KL <= 3.5e-2, chat <= 5e-2.

Routing (`eklow-ref-run --routing`, `tools/ref/routing_stats.py`, ~15k tokens of eklow's notes, two
Python sources and a chat prompt) is skewed per layer: each layer's busiest 64 of 128 experts take
89% of its selections (76-94% by layer, 85-89% held out), 96 take 98% (97-98% held out), 112 take
99.6%. With weights at 13.4 GiB the card holds every expert only at short contexts; at 2 x 128k about
76% fit, so ~2-3% of expert work would go elsewhere, at 2 x 262k (~52%) 11-15%.

For that rest, `eklow::cpu::gemv_q4` (AVX-VNNI on fiona's i7-12700H, a persistent pool) computes one
expert's gate_up + GeGLU + down in ~100 us from DRAM (~33 GB/s, flat from ~6 threads) or ~60 us
cached, against ~290 us to copy it over PCIe.

The worker on these tasks moved from Qwen 3.8 27B to Sonnet subagents in the evening, to let the
rack's R9700s rest.

## 2026-10-26 — the 26B in the engine; its yardstick is the fp32 reference, not llama.cpp

The engine runs the 26B's MoE layers as the reference does, on a slow path for now: router logits
through `ref::matmul`, routing on the host, then one expert at a time (gather its tokens' rows,
GeGLU, down, scatter-add), every expert repacked on the card (13.4 GiB in all). An expert's rows
are padded to 33 so that its matmuls take the fp16 path. The int8 GeGLU takes any width that is a
multiple of 64 (sub-groups of 8; the experts are 704 wide, the dense FFN 2,112), and attention
dispatches (512, 8) for the global layers' 2 K/V heads. Speed, for the record: prefill 1,540 tok/s
at 1,024 tokens and 1,736 at 4,096; decode 18.9 / 18.3 tok/s at 0 / 4k (a wait per expert per layer).

Against llama.cpp's CPU dumps the long prompt missed the band set above (30/32, mean KL 4.1e-2
against <= 3.5e-2). The band was wrong, not the engine. llama.cpp quantizes activations to int8 per
32-block before every Q4_0 matmul (Q8_0 on the CPU, Q8_1 in CUDA's MMQ); the engine's prefill
GEMMs take fp16. At layer 0's attention output llama.cpp (CPU and CUDA) and the engine's own int8
path are 6.8e-3 from eklow's fp32 reference forward, its fp16 path 3e-4. The router turns such
differences into different experts wherever the 8th and 9th logits nearly tie (margins of
0.0005-0.008): at layer 0 of the 2,616-token prompt, 90 positions route differently between
llama.cpp CPU and the engine, but only 16 between llama.cpp CPU and CUDA, whose int8 rounding is
the same. So llama.cpp's two backends agree with each other far better than either does with fp32,
and their distance understated the 26B's noise. Over the long prompt's last 256 positions against
the fp32 reference (the engine's last 64 decoded):

| | top-1 | mean KL |
|---|---|---|
| llama.cpp CPU | 220/256 | 7.5e-2 |
| engine, fp16 expert matmuls (kept) | 249/256 | 1.7e-2 |
| engine, int8 expert matmuls up to 32 rows | 244/256 | 3.6e-2 |

(On its last 32 positions alone llama.cpp CPU happened to land at 2.5e-2.) The chat prompt's 25
positions tell these apart poorly: llama.cpp CPU 23/25 at 4.2e-2, CUDA 23/25 at 2.3e-2, the engine
23/25 at 6.4e-2 (int8 experts 22/25, 5.9e-2). Int8 experts are as close to fp32 as llama.cpp's at
layer 0 (ffn_moe-0 6.0e-3 for both; fp16 2.7e-3), so the decode GEMVs may stay int8: it doubles the
engine's distance from fp32 but keeps it at half llama.cpp's.

The 26B is judged from here by `check26.sh` (in the fiona harness): the engine against
`eklow-ref-run`'s dumps on those 256 positions, inside llama.cpp CPU's own distance (>= 220/256,
mean KL <= 7.5e-2; chat >= 22/25, <= 8e-2). The 12B keeps its check against llama.cpp: without
routing the same int8 difference stays small (long 31/32, 1.3e-2).

## 2026-10-26 — the 26B's decode MoE on the card

For steps of at most 4 tokens (T x 8 pairs <= 32) the MoE no longer touches the host:
- `fast::matmul_f32_small` computes the router logits. `ref::matmul`'s generic F32 path, one
  work-item per output, had taken 58% of a decode step.
- `fast::moe_route` does softmax, top 8, renormalization and the down scales, one sub-group per
  token; ties go to the lowest expert id.
- The expert input is normed and quantized to int8.
- `xmx::matmul_geglu_grouped` and `xmx::matmul_grouped` run the gate_up and down GEMVs over all
  pairs at once. They are gemv1 with each pair's weight looked up in per-layer device tables of
  the experts' tile and scale pointers (the experts stay separate allocations), and the pair's
  activation row at p / row_div.
- `fast::moe_combine` takes the weighted sum in a fixed order.

Prefill keeps the per-expert host path.

Decode measured with `bench.sh engine26`, against the host path in the same runs:

| | at 0 | at 4k |
|---|---|---|
| one slot, tok/s (host path) | 50.5 (11.5) | 46.4 (11.3) |
| two slots, tok/s per sequence | 41.4 | 38.1 |

The 18.9 tok/s quoted above came from different bench settings. With int8 experts at decode,
check26 gives 245/256 at mean KL 2.6e-2, inside the band.

A profiled decode step at 0 takes 23.9 ms; the marks cost ~20%. The MoE takes about half of it.
The biggest section is the grouped gate_up + GeGLU, at 5.8 ms over the 30 layers: ~190 us a layer,
or ~94 GB/s. The kernel test, with cold experts, gives 75-95 us (~185 GB/s), and the grouped down
runs in the engine as fast as in its test (~50 us). The gap is unexplained. A layer is now ~25
launches, of which the small MoE kernels take ~3 ms of the step.

Prefill: 1,014 tok/s at 1k and 973 at 4k depth, the per-expert loop with a host wait per expert.

## 2026-10-26 — the 26B's decode step in fewer kernels

The gate_up "gap" above was the profiler's doing. With the kernels around it fused, the grouped
gate_up profiles at ~92 us a layer, the same as the kernel test. `--profile`'s `mark()`s wait on
the queue after every section, so a short kernel's section also carries its neighbours' launch and
ramp. Small decode sections must be judged by ab.sh deltas, not by the profile. The grouped GEMV's
work-item count, swept 2,048-32,768: 14.9 / 14.6 / 14.8 / 15.4 / 17.3 ms a step. 8,192 is kept.

Decode steps of the 26B (T x 8 <= 32) now run 10 launches a layer outside attention, from 18:
- **`fast::residual_norm_quant3`:** one kernel takes the post-attention residual and one rmsnorm,
  then writes the dense FFN's int8 input, the experts' int8 input and the router's fp32 input.
- **The dense FFN's gate|up rides in the grouped GEMV as three more "experts".** Its 4,224
  interleaved columns are 3 x 1,408, and in the column-group-major layout each third is a
  contiguous range of tiles, so the thirds are just pointer offsets in the expert table. One GeGLU
  launch then serves the experts and the dense FFN.
- **`fast::moe_tail`:** one kernel per row does the dense norm, the experts' weighted sum, its norm,
  the add, the residual and the next layer's quantize.

| decode, tok/s | at 0 | at 4k |
|---|---|---|
| one slot (stage 2b) | 68.2 (51.0) | 61.9 (46.9) |
| two slots, per sequence (stage 2b) | 53.9 (41.3) | 48.3 (38.1) |

One slot at 0 is 14.7 ms a step, from 19.6. check26 gives 247/256 at mean KL 2.6e-2; the 12B is
unchanged. Folding the GeGLU into the grouped GEMV's epilogue gave wrong logits and no gain
(14.73 against 14.64 ms), so it was dropped.

The grouped GEMVs read at ~180 GB/s against the card's 399. A profiled step at 0 (18.3 ms with
marks): gate_up (experts and dense) 3.8 ms, head 2.8, qkv 1.9, wo 1.75, expert down 1.6.

## 2026-10-26 — the 26B's prefill MoE on the card: grouped expert GEMMs

Steps of more than 4 tokens now keep the MoE on the card too:
- `fast::matmul_f32` computes the router; it stays fp32, since routing is sensitive to it.
- `fast::moe_route` picks each token's experts.
- `fast::moe_group` stable-sorts the (token, slot) pairs by expert, pads each expert's segment to
  the row tile (64 rows), and builds the tile list.
- `fast::moe_gather` writes the expert inputs in sorted order (fp16).
- Two grouped fp16 GEMMs run: `Gemm16Kernel` with each work-group's row tile and expert taken from
  the list and its weights from the `ExpertTable`. One is gate_up with the GeGLU epilogue, the other
  down into fp32 rows.
- `fast::moe_combine` sums each token's 8 rows in a fixed order.

The host path and its buffers are gone. Against per-expert `xmx::matmul` the grouped GEMMs are
bit-identical.

The row tile was measured by prefill at 1k: TM 32 / 64 / 128 / 256 gave 2,450 / 3,126 / 3,044 /
2,465 tok/s, so 64 is kept. Padding rows are computed at full price, ~46% extra at TM 64. Two forms
of skipping idle work halved the kernel: an early return for idle work-groups before the K loop,
and an `if` over idle 8-row subtiles inside it. Idle groups now run zero blocks, and only the
padding rows' epilogue is skipped. The workspace is ~312 MB at max_batch 2,048; the fp32 down rows
share memory with the fp16 inputs.

A 1,024-token chunk profiles at ~330 ms. The expert GEMMs are 82 + 45 ms of it (~26 / 21 TFLOPS on
real rows in the kernel test), attention 44, qkv and wo 57, the router 17 (its weight reads are
uncoalesced), gather and sum 18.

| 26B, `bench.sh engine26` at main with stage 2d | one slot | two slots |
|---|---|---|
| prefill 1k / 4k, tok/s | 3,130 / 2,821 (1,014 / 1,003) | 3,123 / 2,821 |
| prefill at 4k depth, tok/s | 2,525 (970) | 2,521 |
| decode at 0 / 4k, tok/s per sequence | 68.1 / 61.4 | 53.9 / 48.2 |

check26 gives 242/256 at mean KL 2.8e-2 (chat 23/25, 6.9e-2); the 12B is unchanged.

Through `eklow-server` the 26B answers and makes tool calls at 2 x 8k (batch 1,024). At 2 x 32k
the first request fails with UR_RESULT_ERROR_OUT_OF_RESOURCES: 13.43 GiB of weights, 1.48 GiB of
KV and the workspace exceed the card. Longer contexts need the experts placed by the routing
profile, with the rest off the card.

## 2026-10-27 — the 26B with cold experts on the CPU (M6 stage 3a)

The routing profile is a file, `data/gemma-4-26b-a4b-experts.json` (`tools/ref/expert_profile.py`
over the four 2026-10-26 recordings: per-layer selection counts, 30 x 128). With
`Config::expert_profile` and `experts_on_card` = N, each layer's N busiest experts (ties to the lower
id) are repacked onto the card, rank by rank across the layers; the others stay in the GGUF mapping
(touched at load so a request does not wait on the page cache) and are computed by `cpu::gemv_q4`'s
AVX-VNNI kernels on a 16-thread pool. A cold expert is an `ExpertTable` entry with null tiles. With
N = 128 nothing changes: the decode path has no host wait (ab.sh against main: 26B -0.1 / +0.2% decode,
-0.1 / -0.0% prefill; 12B within 0.3%; check26 identical to main's).

Decode (up to 4 tokens). After `moe_route`, one kernel (`fast::host_pack`) writes the ids and the
experts' int8 input row (the card's own `act_exp_`, so the CPU runs the same quantization) straight into
pinned host memory; the hot experts' kernels are enqueued; then the host waits on the pack kernel's event,
computes the cold pairs (one expert's gate_up, GeGLU, int8 requantize, down; ~230 us a pair, 100 us being the DRAM floor,
split over the pool) and `fast::cold_scatter` copies the rows into `gx_` for `moe_tail`. The grouped
GEMVs skip cold pairs by an early return of the whole work-group (null tiles), leaving their output
rows alone. Memcpys (three for the routing and the input row, one per cold pair for the rows) instead of the two kernels cost ~0.5 ms a step (each is a serialization point
on the in-order queue). The pool's workers spin 400 us after a job during decode (a futex wake is
50-100 us, a whole expert's GEMV); that took the CPU's share of a step from 2.8 to 1.6 ms at N = 112.
Prefill turns the spin off: it only took cores from the host thread.

Prefill. `moe_group` leaves cold experts out of the tile list (`row_of_pair` = -1 for their pairs),
`moe_gather` and `moe_combine` skip them. The expert input rows go to the host as int8
(`fast::quantize_rows`, a quarter of the fp32 bytes; the D2H copy sits in the queue ahead of the hot
kernels and was 1-2 ms a layer at fp32). The host groups the cold pairs by expert and runs
`cpu::gemv_q4_multi`, which unpacks each weight block once for four inputs (4M MACs per input-expert
matrix in ~90 us on one thread; an expert's rows are read from DRAM once, not once a token), sums each
token's cold experts weighted in slot order, and uploads only the rows of tokens that have any; `moe_combine`
adds them. About 20 us a pair with 16 threads when an expert has dozens of rows, so N = 64 prefill is
CPU-bound: ~600 tok/s on real text. Streaming the busy cold experts over PCIe instead (3.3 MB, ~290 us,
against ~26 us a row on the CPU) would win above ~12 rows an expert; not done.

Fit. Allocations succeed past what the card holds and the first kernels then fail, so the engine reads the
driver's free memory (`rt::free_memory`, Level Zero Sysman; the nix shell and package set its variables)
before and after each rank of experts and stops when the free memory less the margin (1 GiB) cannot hold
another rank of the size the last one took (101 MiB; the weights are 96 MiB, allocation rounding the rest).
The server loads the embedding model first, with its activations reserved for 8,192 tokens, so the
placement sees it. Card: 15.11 GiB reported, 15.05 free at start. A 2 x 262k server: 6.41 GiB of KV, 6.08 GiB
free after the rest, N = 51 (the margin left 1.03 GiB). Margins tried at 2 x 262k with two long requests
(88k and 167k tokens): 1 GiB and 0.5 GiB run clean, and at 1 GiB (N = 51) so do 224k and 30k tokens; 0.04 GiB loses the device (DEVICE_LOST) once the
long request is deep, so the margin pays for what the first long requests allocate later (kernel scratch,
the attention partials) and the driver keeps.

## 2026-10-27 — streaming the busy cold experts to the card for prefill (M6 stage 3b)

Measured (tools/copy_bench, tools/copy_bench/tax.cpp; fiona, PCIe 4.0 x8): a cold expert is 3,345,408 bytes.
Host to card, per expert: 360 us from the GGUF's pageable mapping (9.3 GB/s), 307 us from pinned memory (10.9
GB/s), 358 us from the mapping "registered" with prepare_for_device_copy (no gain: the driver still stages it, and a
Level Zero copy-engine append from it blocks the host for the whole copy). Under the cold GEMVs' load the pageable
copy takes ~530 us, pinned stays at 300. The card has one compute queue and one copy-only queue (blitter): a SYCL
queue's memcpy runs on the compute engine and serializes with kernels; a Level Zero immediate command list on the
copy group (src/rt/copy_engine.cpp) runs by itself and its fence can be a SYCL dependency (make_event), so the host
only issues the copies.

The finding that shapes everything: PCIe traffic slows every kernel on the card, copy engine or not. A VRAM-streaming
kernel runs x1.07 / 1.26 / 1.98 / 5.8 beside DMA at 1.2 / 3.2 / 7 / 11.4 GB/s; the grouped GEMMs 2.5x; ALU-only
kernels and card-to-card copies are untouched; every section of a prefill chunk slows 1.3-1.9x under continuous DMA
(all-hot prefill 2,750 to 995 tok/s). The loss is roughly proportional to the bytes moved (~65 us of card time per
MB), so a streamed expert costs ~0.3 ms of DMA that is mostly not free; overlap with the hot GEMMs gained little
over serial. Streaming later (after the hot GEMMs) was worse than overlapped (1,685 against 1,939 tok/s at N = 64).

Design. With streaming, the cold experts are copied once into pinned host memory (7.2 GiB at N = 51); the CPU
computes from that copy too (decode unchanged). After the router, the host reads the ids, counts pairs per cold
expert, and picks the busiest (at least stream_min_rows = 16 pairs, at most stream_slots) up to the point where
slots x 300 us exceeds the CPU's 22 us a pair on the rest. The copy engine moves them into raw slots; the card
repacks them (35 us an expert; kept raw so that pinned memory is the only copy) into repacked slots, a second
fast::moe_group (group_of, row_base) numbers their pairs after the hot experts' rows, the same GEMM kernels run
them over a second tile list, and moe_combine takes both from one row table (the pairs' inputs are gathered into
att_, dead during the MoE). The CPU takes the other cold pairs meanwhile. A layer's experts also share one slab now.
Rejected: warm repacked pinned copies (reach 64: 2,070 against 1,940 tok/s at N = 64, but 2x host memory or
profile-limited coverage: reach 32/48 carried 61/73% of cold pairs); speculative prefetch of the next layer (more
DMA, and DMA is what costs).

Results (`engine26` on real text, prefill at 1k / 4k / at 4k depth in tok/s; decode unchanged within noise):

| N | before | streamed |
|---|---|---|
| 51 | 516 / 487 / 516 | 1,734 / 1,730 / 1,644 |
| 64 | 609 / 610 / 669 | 1,931 / 1,907 / 1,792 |
| 96 | 1,441 / 1,429 / 1,643 | 2,526 / 2,417 / 2,228 |
| 112 | 2,381 / 2,354 / 2,321 | 2,788 / 2,572 / 2,342 |

40 slots (255 MiB, paid for by the slab, so N stays 51 at 2 x 262k); 24 / 32 / 40 / 48 slots gave 1,467 / 1,692 /
1,730 / 1,733 tok/s at N = 51. A 1k chunk at N = 51:
- hot GEMMs 248 ms, ~3x their undisturbed time, the DMA tax;
- copy wait and repack 89 ms;
- streamed GEMM 39 ms;
- CPU wait 20 ms;
- attention and the dense layers ~210 ms.

The server at 2 x 262k with the embedding model takes a 106,708-token prompt to its first token in 140 s (298 s
before), held back by attention at depth. check26 at N = 51 with 40 slots gives 243/256 at 2.9e-2 (chat 22/25,
5.1e-2), inside the band; with every expert on the card it is unchanged. 2,048-token chunks would halve the DMA per
token (2,196 against 1,682 tok/s at N = 51 on a 4k prompt), but the sliding rings hold the window plus one chunk,
so they cost card memory (~0.4 GB at two slots, about 4 ranks of experts); the server keeps 1,024.

## 2026-10-27 — global prefill attention: where its time goes (measured, nothing kept yet)

At 32k depth the 26B's global attention is 61% of prefill (1,016 ms of 1.66 s for 2,048 tokens,
from 5 of 30 layers); in the 12B at 16k it is ~45%. `FlashGroup` (head 512) runs at ~11 TFLOPS of
the 98.8 the fp16 dpas offer. Copies of it with parts switched off (results wrong on purpose;
`eklow-attn-test`, a 1,024-token chunk, the 12B's group of 16, int8 cache) took:

| removed | 4k keys, ms | 16k keys, ms | at 16k |
|---|---|---|---|
| nothing | 10.9 | 47.2 | |
| the two barriers a block | 7.5 | 31.9 | -32% |
| barriers and the score exchange | 6.9 | 29.9 | -37% |
| the loaders' work | 9.3 | 39.4 | -17% |
| Q.K^T (dpas and K loads) | 8.8 | 43.9 | -7% |
| P.V (dpas and V loads) | 9.2 | 41.1 | -13% |
| Q.K^T and P.V together | 2.6 | 12.8 | -73% |
| all of the above | 0.8 | 2.5 | -95% |
| the softmax arithmetic | 11.1 | 47.3 | 0 |

The 26B's group of 8 behaves the same (barriers -32% at 16k), and the fp16 cache runs within 2% of
the int8 one, so the int8 dequantization costs nothing. What the table says:
- The kernel is latency-bound through chains of stages, not bandwidth-bound: shared memory runs at
  about a third of its bandwidth.
- The barriers are the largest single item, worth at most ~1.45x. The two halves of the matrix work
  are strongly super-additive, each hiding behind the other.
- The 128-register wall is what holds it: the output accumulators take 64. Each attempt to add
  independence spilled 2.6 KB and ran 2-2.5x slower:
  - Q.K^T's B operands in two 1 KB loads: 91.5 ms;
  - four Q.K^T accumulators: 116 ms;
  - both: 105 ms.
- The baseline already spills 288 B (fp16 cache) or 416 B (int8). A per-row rescale factor instead
  of a broadcast changed nothing (47.4 ms).

Reasoned out, not built:
- **One barrier a block** needs K and V double-buffered (64 KB), leaving no room for the scores.
  Staging int8 and dequantizing in the compute threads adds work to the threads that are already
  latency-bound.
- **Int8 Q.K^T** is worth at most its ~12% share, with an accuracy risk and a cache format change.

Left to try: a group of 12 query tiles (112 threads, partial scores exchanged in fp16, one
work-group a core, sharing the barriers and loads over twice the rows). It builds within 64 KB,
but its first run hung (2026-10-27). Beyond it, the likely remaining lever is a deeper restructure
that splits the 512 dimensions so each thread holds fewer than 64 accumulator registers.

## 2026-10-27 — production on the 26B, and the card is the service's

fiona's eklow.service serves the 26B at 2 x 262k from 09:03. N = 51 of 128 experts a layer are on the card, with
40 streaming slots and 7.2 GiB pinned, as the startup line reports. Through it, by unholy's method
(`tools/bench/serve_perf.py`; `docs/bench/serve-perf-26b-fiona-2026-10-27.txt`):
- prefill 1,578 / 1,257 / 493 / 352 tok/s at 8k / 32k / 128k / 252k (734 s to the first token at 252k);
- decode 37.6 / 34.9 / 30.9 / 26.7 tok/s;
- GSM8K 33.4 / 41.9 tok/s aggregate, one and two at a time.

Embeddings (60-78 ms) and llama.cpp's titles on the Iris Xe (368 ms idle, ~460 ms busy) run beside it.

A second eklow-server started beside the live one died at startup on pinned memory, but took card memory first.
The live service then answered every request with UR_RESULT_ERROR_OUT_OF_RESOURCES until a restart, 14 minutes later.
Once a running server has hit an out-of-resources launch, it stays broken. `bench.sh` and `tools/bench/serve-perf.sh`
now refuse to run while the service is active.

## 2026-10-27 — global prefill attention as two GEMMs

The global layers' prefill attention (head 512) now runs a chunk over the cache in segments of 2,048 keys, each as
two kernels in the shape of xmx.cpp's fp16 GEMM (src/kernels/attention.cpp, `run_segments`):
- SegQK: S = Q K^T, 256 rows x 128 keys a work-group, the int8 K tiles dequantized into shared local memory as
  the GEMM does its weights; its epilogue masks keys past each row's position, takes the row's maximum over the 128
  keys and its running maximum, and writes P = exp(S - m) in fp16 (the A layout) with that maximum and the sum;
- a combining step: per row, the segment's maximum, the running output's factor, each 128 keys' factor, the sum;
- SegPV: O = alpha O + (beta P) V, 256 rows x 128 dimensions, the probabilities scaled as they are staged and the
  running output kept in fp32 in device memory between segments;
then a division by the sums. Chunks under 64 tokens (decode, short prompts) keep the flash kernels, and a decode
token beside a long chunk in the same step runs through the split kernel on its own tile range.

Why. The micro-benchmarks behind it (scratch tools on fiona, `~/eklow-micro`):
- dpas with operands from shared local memory reach 81-92 TFLOPS when each thread holds 8 accumulators (2 x 4
  tiles, 192 bytes loaded per dpas), and 72-81 with a barrier every 8 dpas: neither bandwidth (~10 TB/s from
  shared memory or L1) nor barriers hold FlashGroup;
- a 256-byte load from shared local memory takes 170 ns idle, 250 ns under load; from L1, 70 / 100 ns; a dependent
  dpas 44 ns;
- FlashGroup's ISA waits on every load right before the dpas that uses it, and all 128 threads of a core reach
  the same phase together, so per 16-key block a thread runs a chain of load, dpas, store, barrier, exchange
  loads, softmax, store, barrier, load, dpas: ~13,500 clocks for 16 dpas.
Kept out: a version of the flash kernel with the work split by role (32 Q.K^T threads holding Q in registers, 64
P.V threads that also ran the softmax, 32 loaders; the next block's scores beside this block's P.V) ran at the
same ~9-10 TFLOPS: per block it still exchanged partial scores through shared memory with two barriers, the
loaders alone cost 45%, and a 128-thread work-group of it hung (a 128-thread micro-benchmark does not, so not the
work-group size; not found). Rescaling the output only when a correction is not 1 made the compiler keep two
copies of the 64 output registers and spill 1 KB; unconditional is free.

Measured (eklow-attn-test, a 1,024-token chunk, int8 cache; TFLOPS counting 4 x 512 per row and key):

| shape | keys | FlashGroup | segments |
|---|---|---|---|
| 26B (8 heads a K/V head) | 16k | 10.3 | 30.5 |
| 26B | 131k | 10.3 | 31.2 |
| 12B (16) | 16k | ~10.9 | 33.3 |

At 131k on the 26B's shape: Q.K^T 31 ms (~35 TFLOPS), combining 2 ms, P.V 37 ms (~29), the division 0.1.
Segments of 4,096 keys gave 5% more for twice the probabilities' memory (128 MB for the 26B), 1,024 10% less. P.V
as 64 rows x all 512 dimensions, so that each row's probabilities are read once rather than four times, ran 10%
slower. Rounding: within the noise of the old kernel (global rel L2 3.6e-4 / int8 9.0e-3 against ref::attention;
check26 long 245/256 at 2.38e-2, chat 23/25 at 6.87e-2; check 31/32 at 1.93e-2, chat 24/25 at 5.2e-3).

The buffers (probabilities 2 bytes per row and key of a segment, statistics and the running output ~2.2 KB per
row; ~100 MB for the 26B's 1,024-token chunks) are allocated when the engine starts, before the experts take the
card's free memory (`attn::reserve`): at 2 x 262k the server now places N = 50 experts a layer (51 before).

Engine (engine26 real text, N = 51, profile; tok/s and global attention's time per 2,048 tokens):

| depth | before | now |
|---|---|---|
| 32k | 957 (1,007 ms, 47%) | 1,398 (332 ms, 23%) |
| 128k | 396 (4,048 ms, 78%) | 787 (1,438 ms, 55%) |

Through the server with production flags (`docs/bench/serve-perf-26b-segments-fiona-2026-10-27.txt`), prefill as
time to first token on unique text:

| depth | before | now |
|---|---|---|
| 8k | 1,578 tok/s (5.2 s) | 1,631 (5.0 s) |
| 32k | 1,257 (26.1 s) | 1,529 (21.5 s) |
| 128k | 493 (266 s) | 1,040 (126 s) |
| 252k | 352 (734 s) | 703 (367 s) |

GSM8K is unchanged (33.2 / 41.8 tok/s aggregate, 1 / 2 at a time). Decode is untouched (a decode token beside a long
chunk still takes the split kernel): main and this tree, alternated on one card state, decode the same in the
engine (46.2 / 41.5 tok/s at 0 / 8k) and through the server (35.6 at 8k); the server run's decode, ~10% under the
morning's at every depth, came after hours of load on the card. N = 50 against 51 moves decode < 1%.

What is left at depth: global attention is still 55% of prefill at 128k. P.V (~29 TFLOPS) with its pieces knocked
out one at a time (the beta scaling, the V dequantization, the V loads, the running output's read) gives back at
most 5% each, so no one of them holds it. Beyond it: 2,048-token chunks (open), and the sliding layers' prefill
attention (FlashGroup, ~10 TFLOPS, 4-6% of prefill).

## 2026-10-27 — 2,048-token prefill chunks by default

eklow-server's `--chunk` default is now 2,048 (from 1,024); the NixOS module does not set it. A chunk is the prefill
tokens of one sequence in a step, and the step stays at most 2,048 tokens (`--batch`), decodes first, so a sequence
prefilling beside another's decode takes 2,047.

Cost: the sliding layers' rings hold the window and one chunk, ~0.4 GB more at two slots of 262k. The segmented
global attention now runs a chunk's K/V heads one after the other on one set of buffers, so they stay ~100 MB
rather than doubling (at 2,048-token chunks one head's rows fill the card; per head is +1.6% against both heads at
once there, -3.8% at 1,024-token chunks, engine A/B at 32k depth). The server at 2 x 262k places N = 46 experts a
layer (50 with 1,024-token chunks).

Gain: twice the rows per expert GEMM, and half the steps (and PCIe streaming) per token. Through the server with
production flags (`docs/bench/serve-perf-26b-chunk2048-fiona-2026-10-27.txt`), prefill as time to first token on
unique text, against today's 1,024-token run:

| depth | 1,024 (N = 50) | 2,048 (N = 46) |
|---|---|---|
| 8k | 1,631 tok/s (5.0 s) | 1,865 (4.4 s) |
| 32k | 1,529 (21.5 s) | 1,794 (18.3 s) |
| 128k | 1,040 (126 s) | 1,177 (111 s) |
| 252k | 703 (367 s) | 765 (338 s) |

Alternated on one card state at 32k: 1,741 / 1,758 against 1,535 / 1,534 tok/s (+14%), decode 32.6 against 33.0.
GSM8K two at a time 40.2 tok/s aggregate (41.8-41.9 at N = 50-51), the CPU's larger share of two streams' cold
experts. A long prefill step now takes about twice as long, so another slot's decode waits longer between tokens
while a prompt prefills. check26 with 2,048-token chunks is identical to 1,024 (245/256, 2.38e-2; chat 23/25,
6.87e-2): every kernel computes each row on its own.

## 2026-10-27 — EmbeddingGemma 2 in GEMMs

The embedding endpoint ran ~900 tok/s through production while Open WebUI reindexed (the server's log): the model ran
on the reference kernels, one input at a time, every matrix a per-output dot product over Q8_0 weights. Now
(`src/model/embed_gemma2.cpp`) a request's inputs go through together in passes of up to the reserved 8,192 tokens,
RoPE at each row's position within its input, attention input by input (still `ref::attention_bidir`), each input
averaged on the card. Every matrix runs on xmx's fp16 GEMM, which gains two weight forms (`xmx::repack`): Q8_0, tiles
of int8 already in B-operand order dequantized in shared local memory as Q4_0's are, and F16 (from F16, BF16 or F32
files; EmbeddingGemma's one BF16 matrix, the per-layer projection). Each layer's per-layer input is made from the
scaled embeddings when the layer needs it: precomputing all 24 layers' took 400 MB of the card at 8,192 tokens.

Through the server with production flags (`tools/bench/embed_bench.py`: unholy's 32k Pile document cut into
1,800-character chunks with EmbeddingGemma's document prefix, Open WebUI's shape; main against this, one card state):

| | before | now |
|---|---|---|
| 16 inputs a request (~470 tokens each) | 883 tok/s | 3,717 |
| 1 input a request | 890 | 2,335 |
| ~5,900-token inputs, 4 a request | 425 | 610 |

The vectors are the old ones to six places (cosine 1.000000, the least over 98 inputs), and check26 is unchanged
(the Q4_0 path is the same code). The activations at 8,192 tokens fell from ~0.9 GB to ~0.45, so the 26B at 2 x 262k
places N = 50 experts a layer again (46). What holds it now is attention on the reference kernel, a work-group
reduction per key (long inputs barely moved); the engine's attention kernels with a key range per token, which image
input needs too, are next.

## 2026-10-27 — a key range per token

Attention's tokens were causal: each saw the keys from its window's start (or 0) to its own position. Two things need
other ranges. An image's tokens on the 26B's sliding layers see the whole image (llama.cpp runs Gemma 4's vision
tokens bidirectionally on the sliding layers only), and EmbeddingGemma 2's inputs see themselves both ways (within
window/2 on its sliding layers), which still ran on `ref::attention_bidir`, a work-group reduction per key, after the
GEMMs had made the rest of the model fast. `attn::Tokens` now optionally carries each token's first and last key
(`lo`, `hi`; neither may decrease within a chunk), and `TileList::add_chunk` the chunk's key range. The kernels read
the last key where they read the position (the host passes `hi` or `pos`), the segment kernel masks keys before `lo`
as it masks those past `hi`, and a chunk's segments start at its first key. The split kernel (decode) takes the
ranges as a template parameter: reading `lo` there cost the int8 kernel registers (spill 768 -> 928 bytes) and
global8 decode 6%; without them it is the old kernel. `eklow-attn-test` runs each kind again (`+img`) with every token
carrying its range and a quarter of each slot's second chunk an image seeing to its end, those rows against a host
reference in double: within the kinds' bands (rel L2 1.0e-4 and 3.6e-4 fp16, 8.9e-3 and 9.0e-3 int8, as without).

EmbeddingGemma 2 now runs on these kernels: a pass's inputs back to back in one slot of an fp16 cache (identity page
table), each input in chunks of up to 2,048 tokens that see their own input, at most 256 inputs a pass (the bounds
keep the segment buffers and query tiles to ~120 MB with the cache at 8,192 tokens). Its global layers, one K/V head
of 4 query heads at head 512, get their own instantiation. Through the server with production flags (as in "in
GEMMs"):

| | before | now |
|---|---|---|
| 16 inputs a request (~470 tokens each) | 3,717 tok/s | 23,201 |
| 1 input a request | 2,335 | 4,374 |
| ~5,900-token inputs, 4 a request | 610 | 18,416 |

The vectors are main's to six places (cosine 1.000000, the least over 98 inputs). One input a request is now the
request's own overhead (497 tokens in 0.106 s). The cache and scratch cost the 26B one expert at 2 x 262k: N = 49
(50). check26 is unchanged (245/256, 2.377e-2; chat 23/25, 6.872e-2), check too (31/32, 1.93e-2; chat 24/25,
5.24e-3). Alternating `eklow-attn-test` runs against main: decode the same (global8 at 16k 302-306 us against
305-308), prefill the same but the int8 sliding layers' 1024-token chunk at 16k, 1.71 ms against 1.67-1.68 (the group
kernel reads `lo` at run time; ~0.15% of a prefill chunk).

## 2026-10-27 — Gemma 4's vision encoder, on the reference kernels

Image input for the 26B and EmbeddingGemma 2 starts from llama.cpp's projector `gemma4v` (both models' mmproj files:
the 26B's 27 layers of 1,152, 16 heads of 72, BF16, out 2,816; EmbeddingGemma 2's 16 of 768, 12 heads of 64, Q8_0,
out 512). `model::VisionGemma4V` (from `m7/vision`) follows `tools/mtmd/models/gemma4v.cpp` and `build_vit` at the
oracle's revision on the reference kernels: the patch embedding, x/y position tables, 2D NeoX RoPE (base 100, a head's
halves by column and row), unweighted V norm, attention scale 1, post norms, 3 x 3 average pooling, sqrt(n_embd),
std_bias / std_scale, an unweighted RMS norm and the projection. Neither file sets `use_gelu`, so the FFN is
llama.cpp's fallback, gelu_quick. `image::prepare` (from `image-input`) decodes with the stb_image llama.cpp vendors
and resizes as mtmd does (a token budget, 70 to 1,120, default 280; Pillow's bicubic in fixed point). The oracle gains
`--image-tokens N --encode-only` (mtmd's encoder on the CPU, rows and taps), `eklow-vision` encodes an image, and
`tools/ref/compare_rows.cpp` compares rows.

Token counts equal llama.cpp's on every image. Against the stock oracle the rows miss the gate on their tail rows
(26B min cosine 0.994-0.998, total rel L2 1.2-1.9e-2; EmbeddingGemma 2 to 0.32 on one row of one image, 3-9e-2): ggml's
CPU backend rounds activations to bf16 / q8_0 and gelu and im2col through fp16, and the stock oracle differs from the
same oracle on F32-converted weights by as much as eklow does from it. Against the F32-weight oracle (280 tokens):

| | cosine min / mean | rel L2 worst row / total |
|---|---|---|
| 26B, test-1 / tall / wide | 0.99715 / 0.999932, 0.99986 / 0.999983, 0.99954 / 0.999995 | 7.6e-2 / 1.2e-2, 1.7e-2 / 5.7e-3, 3.0e-2 / 3.3e-3 |
| EmbeddingGemma 2, the same | 0.99985 / 0.999997, 0.99994 / 0.999997, 0.99997 / 0.999999 | 1.7e-2 / 2.5e-3, 1.1e-2 / 2.7e-3, 8.2e-3 / 1.2e-3 |

Layer by layer (26B, tall) the difference grows smoothly from 1e-4 to 1.4e-2, no step. The encoder is judged against
the F32-weight oracle; the 26B's one row at 0.997 is left to the end-to-end check (the model's answers about images).
The reference kernels are correctness only: 7.5 s for 72 tokens of the 26B, 41 s for 270, EmbeddingGemma 2 148 s for
1,080 (llama.cpp's CPU: 4-19 s); one work-item per output and a serial attention.

## 2026-10-27 — the vision encoder on the matrix units

The reference encoder took 41 s for a 270-token image of the 26B. Now (`src/model/vision_gemma4v_fast.cpp`, the
default; `--ref` and the constructor flag keep the reference path) every matrix runs on xmx's fp16 GEMM (BF16 as F16,
Q8_0 as Q8_0; Q, K, V as one weight, gate and up as one, the FFN padded 4,304 -> 4,352), attention on a new dense
kernel (`src/kernels/vision_attn.*`: the long-prefill path's two GEMMs per segment of keys, cut for heads of 64 and
72, the latter padded to 96, several heads a launch, the segment length chosen so a launch has ~128 work-groups), and
the elementwise work in a few fused kernels (a prep kernel for the per-head norms and 2D RoPE writing the attention's
tiles; post-norm, residual and the next norm after each GEMM; pooling, std scale and norm together). The weights stay
in pinned host memory (1.1 GB for the 26B), converted once at load, and each layer is copied into one of two staging
buffers on the card while the layer before computes; the card holds the encoder only while it encodes (`release()`
frees it), and `reserve()` / `card_bytes()` let a server count it.

| | tokens | now | reference |
|---|---|---|---|
| 26B | 72 / 130 / 253 / 270 | 0.136 / 0.171 / 0.271 / 0.28 s | 7.5 / 13.2 / 34.7 / 41 s |
| 26B | 1,080 (budget 1,120) | 1.40-1.47 s | > 900 s |
| EmbeddingGemma 2 | 130 / 270 / 1,080 | 0.047 / 0.082 / 0.51 s | - / 6.8 / 148 s |

Against the F32-weight oracle the rows are closer than the reference path's (26B worst rows 1.4-2.4e-2 rel L2 from
1.7-7.6e-2, totals 3.3-5.3e-3; EmbeddingGemma 2 totals 1.4-1.9e-3): every mean cosine and total passes the gate.
Card memory while encoding: the 26B 282 MiB at the 280-token budget, 432 at 1,120 (staging 2 x 40.6, attention operands
94 and scratch ~82, residual 47, GEMM scratch 80); EmbeddingGemma 2 183 and 264; at rest 8.4 and 1.7 MiB. Where a
253-token 26B image's 270 ms go: attention 62-73 ms, GEMMs 76 (28-41 TFLOPS), elementwise ~55, and ~70 that is the
streaming: the copies (1.1 GB at ~12 GB/s) overlap compute in time, but DMA slows the kernels beside it (without the
copies, as a diagnostic, 195-210 ms; the same with the copies issued up front, 2 or 3 buffers ahead, on the copy engine
or a second queue). At 1,080 tokens attention is 61% (~14 TFLOPS counting the real 72 dims): its K blocks are 3 dpas
deep against 16 at head 512, so the epilogue weighs more; that and a resident-weights option are what is left.

## 2026-10-27 — image input through the server

`eklow-server --mmproj GGUF` (chat) and `--embed-mmproj GGUF` (embeddings), `--image-tokens N` (70-1,120, default 280). Chat takes OpenAI
`image_url` parts with base64 `data:` URLs (JPEG, PNG, GIF, BMP; WebP and http(s) refused with a 400, nothing is fetched); embeddings take an
input item that is an array of `text` / `image_url` parts or llama.cpp's `{"content": [parts]}`. What llama.cpp does, from its source at the
oracle's revision and checked by running it (llama-server and the oracle, CPU): the server turns an `image_url` part into a media marker in the
message's text (text parts joined by newlines, none around a marker), the template runs on that string, and mtmd tokenizes the text before and
after each marker on its own with `<|image>`, the rows and `<image|>` between, no newlines, BOS (and EOS for embeddings) added. Image rows are
fed unscaled, the 26B's per-layer inputs do not exist, EmbeddingGemma 2's are made from the rows like from any row. Gemma 4's `non_causal_type`
is SWA-only: on the sliding layers each image row sees the whole image (and the window back), the global layers stay causal.

eklow: `Chunk::images` (spans of rows in place of lookups; on the sliding layers those tokens' `hi` is the span's last position, through
`attn::Tokens::lo, hi` when a step holds any), the scheduler never splits an image across chunks and cuts a reused prefix back to an image's
start, and an image's positions hold ids below zero derived from a hash of its bytes (and budget) and the row, so that prefix reuse tells
images apart. Each image is encoded on its request's thread with the scheduler held (`Scheduler::Hold`), one at a time per encoder, and
the host rows of the last six are kept by hash (a conversation resends its images every turn). The encoders' card memory is reserved before the
experts are placed (26B 282 MiB at 280 rows; EmbeddingGemma 2's projector at 1,120 rows, 264 MiB: `reserve(280)` on it gave NaN embeddings
for some images, which `reserve(400)`, `reserve(1120)` and no reserve did not). On the 26B at production flags the encoders cost experts a
layer: N = 44 (49 before), 41 with `--image-tokens 1120`.

Against llama.cpp: token ids and image spans equal for one image, two, and a conversation with an image in an earlier turn; the logits of
the 26B over the positions after an image, from the oracle's own rows, mean KL 6.3e-2 and top-1 77/90 (one image, within check26's band),
and a causal image costs 0.25; a two-image prompt reads 0.25 (median 0.02) from either rows, and eklow's own two runs with the two row sets
differ by the same 0.26: that prompt is chaotic at this size, not a different computation. Embeddings of images against llama-server:
cosine 0.99966-0.99996, token counts equal; string inputs against main: 1.000000.
