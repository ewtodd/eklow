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
