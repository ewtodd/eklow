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
synthetic mistakes. Generic and `<|"|>`. The sampler drops them before anything else (from the tool grammar's allowed set too), so the model
takes its next choice instead. llama.cpp samples them and its parser copes as it can. `eklow-grammar-test` checks that
in each state the banned set is exactly what the parser refuses (96 of 96 state and token pairs).

The sampler took 1.2-1.8 ms a token (on the host, between two decode steps of ~27 ms): every one of the 262,144
candidates went into an array for `nth_element`. Dropping the bans from that array cost another 0.33 ms, as did
setting them to -inf (`nth_element`'s pivots on that data). Top-k now keeps the k best in a heap during one pass over
the logits, skipping the bans by comparing each id with the next one: 0.21 ms with or without bans, and the same
picks as before (900 of 900 against main's sampler over greedy, top-k 64 and none, top-p, min-p and penalties, ties
included).is this step ok" questions ranked a real bad call above a good one
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

`eklow-server --mmproj GGUF` (chat) and `--embed-mmproj GGUF` (embeddings), `--image-tokens N` (70-1,120, default 280,
Google's `max_soft_tokens`; llama.cpp's server defaults to the projector's 1,120). Chat takes OpenAI `image_url` parts
with base64 `data:` URLs (JPEG, PNG, GIF, BMP; WebP and http(s) refused with a 400, nothing is fetched); embeddings take
an input item that is an array of `text` / `image_url` parts or llama.cpp's `{"content": [parts]}`.

What llama.cpp does, from its source at the oracle's revision and checked by running it (llama-server and the oracle,
CPU): the server turns an `image_url` part into a media marker in the message's text (text parts joined by newlines,
none around a marker), the template runs on that string, and mtmd tokenizes the text before and after each marker on
its own with `<|image>`, the rows and `<image|>` between, no newlines, BOS (and EOS for embeddings) added. Image rows
are fed unscaled; the 26B has no per-layer inputs, and EmbeddingGemma 2 makes its per-layer inputs from the rows like
from any row. Gemma 4's `non_causal_type` is SWA-only: on the sliding layers each image row sees the whole image (and
the window back), the global layers stay causal.

eklow: `Chunk::images` (spans of rows in place of lookups; on the sliding layers those tokens' `hi` is the span's last
position, through `attn::Tokens::lo, hi` when a step holds any), the scheduler never splits an image across chunks and
cuts a reused prefix back to an image's start, and an image's positions hold ids below zero derived from a hash of its
bytes (and budget) and the row, so that prefix reuse tells images apart. Each image is encoded on its request's thread
with the scheduler held (`Scheduler::Hold`), one at a time per encoder, and the host rows of the last six are kept by
hash (a conversation resends its images every turn). The encoders' card memory is reserved before the experts are
placed: encoding inside the 1 GiB margin and releasing after each image would keep the experts, but the margin is what
long requests allocate into later (stage 3a: 0.04 GiB lost the device), and an image can arrive during one. 26B 282 MiB
at 280 rows; EmbeddingGemma 2's projector at 1,120 rows, 264 MiB, because `reserve(280)` on it once gave NaN embeddings
for some images (test-1, tall; `reserve(400)`, `reserve(1120)` and no reserve did not). That did not reproduce in three
later server runs at `reserve(280)` (fresh, and after chat images, test-1 / tall / wide twice): the workaround stays
until it is understood. On the 26B at production flags the encoders cost N = 44 experts a layer (49 without them), 41
with `--image-tokens 1120`.

Against llama.cpp: token ids and image spans equal for one image, two, and a conversation with an image in an earlier
turn (llama-server's `prompt_tokens` 160, 354, 109, as eklow's). The 26B's logits over the 90 positions after a
130-row image, from the oracle's own rows: top-1 77/90, mean KL 6.3e-2 (within check26's band); from eklow's rows
(1.6e-2 from the stock oracle's) 0.12; with the image forced causal 0.25. A two-image prompt reads 0.25 (median per
position 0.02) from either row set, and eklow's two runs with the two row sets differ from each other by the same
0.26: the 26B's routing is chaotic at this size, not a different computation. Embeddings of images against
llama-server: cosine 0.99966-0.99996, token counts equal; string inputs against main: 1.000000. check, check26 and the
attention test unchanged.

End to end at production flags: an image encodes in 0.14-0.18 s at 280 rows (1.44 s at 1,080), first token 0.3 s
after a 130-row request; a second turn reuses the prefix with the image (`prompt 297 (149 cached, 130 image)`); two
different same-size pictures are told apart. The 26B described test-1 as the New York Times front page of July 21,
1969, "MEN WALK ON MOON", and a 1,080-row IDE screenshot as a C++ file in an editor with a build log, both right.

## 2026-10-27 — structural tokens only where they parse

The 26B, through son-of-anton at 46k tokens of context (request `chatcmpl-48ecc99c`, 21:25), wrote `<channel|>` in the
middle of a word of its answer ("doesn'"): the end of a thought channel, with no channel open. The output parser
refuses a special token its state cannot take and the server shows the refusal in the content, so the answer ended in
"[malformed tool call: unexpected <channel|> in content]". The model does this by itself (the owner has seen such
slips under llama.cpp); eklow's arithmetic is within the oracle's noise.

Gemma 4 has 24 special tokens (17 control, 7 user-defined), all structural. `OutputParser::banned()` lists, for the
parser's state, the ones it would refuse: in content all but `<|channel>`, `<|tool_call>` and the turn's ends; in a
channel's name all but the ends; in reasoning all but `<channel|>` and the ends; in a tool call all but `<tool_call|>`
and `<|"|>`. The sampler drops them before anything else (from the tool grammar's allowed set too), so the model
takes its next choice instead. llama.cpp samples them and its parser copes as it can. `eklow-grammar-test` checks that
in each state the banned set is exactly what the parser refuses (96 of 96 state and token pairs).

The sampler took 1.2-1.8 ms a token (on the host, between two decode steps of ~27 ms): every one of the 262,144
candidates went into an array for `nth_element`. Dropping the bans from that array cost another 0.33 ms, as did
setting them to -inf (`nth_element`'s pivots on that data). Top-k now keeps the k best in a heap during one pass over
the logits, skipping the bans by comparing each id with the next one: 0.21 ms with or without bans, and the same picks
as before (900 of 900 against main's sampler over greedy, top-k 64 and none, top-p, min-p and penalties, ties
included).

## 2026-10-31 — Qwen3.6-35B-A3B in place of Gemma 4, with DFlash 2

The owner asked to drop Gemma 4 12B and 26B A4B for Qwen3.6-35B-A3B finetunes served with DFlash 2 speculative
decoding, trying several; Ornith-1.5-35B-A3B (ornith-ai, revision 10fbf86f) first. Its model card claims large gains
over stock Qwen3.6 on agentic coding (SWE-bench Verified 79 against 73.4, Terminal-Bench 2.1 67.8 against 52.5), but
every number there is the vendor's own, its Qwen3.6 baselines included, and no independent comparison or
contamination check existed on this date (one private 30-task test: 15 against 13). So both are served and measured
here: stock `Qwen/Qwen3.6-35B-A3B` (995ad96e) and Ornith, the same architecture (their configs differ only in metadata,
their tokenizers not at all), with eklow's fidelity checks and the owner's own evals, and the numbers published.
EmbeddingGemma 2 stays (production embeddings use it), and with it Gemma's tokenizer and vision encoder.

**The model** (llama.cpp's `qwen35moe`, which the oracle's pin ea0c4c2c runs, with DFlash 2 since PR 27342): 40 layers
of hidden 2048; every 4th (3, 7, ..., 39) gated attention with 16 query heads and 2 K/V heads of 256, IMRoPE over 64
dims at base 1e7, the output scaled by a sigmoid gate from the query projection; the other 30 Gated DeltaNet (16 key
heads and 32 value heads of 128, a 4-tap causal conv, a 128 x 128 fp32 state per value head); every layer 256 routed
experts of 512 (softmax, top 8, renormalized, SwiGLU) beside a shared expert of 512 under a sigmoid gate; an untied
head over 248,320 tokens. 35B parameters, 3B active a token; the routed experts are 32.2B of them.

**Memory at 2 x 262,144.** KV exists only for the 10 attention layers: 10 x 2 x 2 x 256 values a token, 5.6 GiB for
both slots in int8 with fp16 scales (the 26B's was 6.4). The DeltaNet state is ~60 MiB a slot. The experts are 18.1 GB
at 4.5 bits a weight, so after ~1.5 GB of dense weights (at 8.5), the head (0.4 GB, Q6_K), the KV, the draft (~0.5 GB)
and the 1 GiB margin, roughly 70-80 of each layer's 256 stay on the card, chosen by a routing profile as the 26B's
were, and the rest (~13 GB) are computed by the CPU from pinned memory: on fiona's 30 GB that fits, tightly.

**Weights.** The format is chosen by quality per byte among those the card runs well, and eklow gets the kernels for
it (the owner: what matters is what the hardware supports, not what eklow implements today). Decode reads every
weight it uses once a token at up to 399 GB/s, so bits per weight decide its speed; the matrix units take int8 and
int4 operands 32 deep, so formats with 32-weight sub-blocks and a scale (Q4_0, Q4_1, Q4_K, Q5_K, Q8_0) go straight
into the int8 decode path, a min costing one more term from the activations' block sums; Q6_K (16-weight
sub-blocks) and IQ4_XS (a 16-value table per nibble) need an unpack step first, which decode's spare arithmetic
covers; prefill dequantizes any of them to fp16 in shared memory. K-quant kernels also let eklow serve the published
Q4_K_M files of other finetunes as they are. The candidates (the published Q4_K_M, Q4_K_S, IQ4_XS and Q4_0, and
mixes made with the oracle's converter and quantizer under bartowski's importance matrix) are measured by KL against
Q8_0, with the experts' bytes (what fits on the card, what the CPU reads per cold expert) and the dense bytes (what a
decode token reads) counted apart (`tools/weights/`, `qwen/weights` branch). If the winner is a file of our own,
the owner hosts it on Hugging Face (uploaded from son-of-anton, which holds the key).

**DFlash 2** (Inco AI, August 2026; z-lab `dflash/model.py`): a 6-layer Qwen3 draft over the target's hidden states
from layers 1, 6, 11, 16, 22, 27, 32, 37 (concatenated, projected by `fc`, normed, written as K/V into the draft's
sliding cache once a token is committed), drafting a block of [last token, MASK x (B-1)] in one pass, non-causal within
the block, with a two-tap grouped dynamic convolution around each layer's attention and MLP, and a selector that walks
the top 16 candidates at each position with low-rank codebook scores (predecessor x hidden x successor). The target
verifies the block; acceptance is rejection sampling against the selector's distribution. Ornith's own draft
(ornith-ai/Ornith-1.5-35B-A3B-DFlash) is DFlash 1, not 2. Two DFlash 2 drafts fit: `incoai/Qwen3.6-35B-A3B-DFlash2`
(51ef7b69, the authors', trained on stock Qwen3.6; block 8, window 2,048; 5.21 / 4.90 / 4.41 tokens accepted a verify
on GSM8K / HumanEval / MT-Bench at temperature 1, SGLang) and `jzinno/Ornith-1.5-35B-A3B-DFlash2` (9b4852c0, a
community draft trained on Ornith's outputs; block 16, window 4,096; ~4.0 greedy at 10 drafted tokens). The owner: use
Inco's, and the community one if it is better; llama.cpp's acceptance for the four pairs is measured first as the
reference eklow's implementation will be held to. Both drafts use sliding attention in every layer, so their cache
stays a few tens of MB however long the context.

**The DeltaNet on the card.** Each column of a value head's state updates independently given that token's k, q, gate
and beta, so one recurrent kernel can serve prefill, decode and verify alike: a thread holds a slice of columns and
walks its slot's tokens in order. By estimate ~0.2-0.3 us a token, 30 layers of a 2,048-token chunk ~15 ms against
~400 ms for the rest of the chunk, so the chunked (WY) form is left until measurements ask for it (`qwen/gdn` measures).
A verify block must leave the state as if only its accepted tokens had run: rather than copying states, a step keeps
the inputs of tokens it leaves pending and the next step replays the accepted ones onto the committed state before its
own tokens. Prefix reuse cannot truncate a recurrent state, so the server will keep snapshots of a slot's state (~60
MiB, copied to pinned host memory in ~6 ms) where conversations diverge (the end of a prompt); Ornith's template
re-renders earlier reasoning exactly as generated, which helps.

**The card.** Development needs fiona's A770M for long stretches, and the owner chose to stop production for the
duration: `eklow.service` (the 26B) stopped 2026-10-31 17:40. The work happens on `qwen/*` branches merged into
`qwen/main`; main keeps the 26B until the new model serves, then the Gemma 4 chat code goes.

## 2026-10-31 — the Gated DeltaNet's serving kernels

`src/kernels/gdn.*` (`qwen/gdn`, merged): everything between a DeltaNet layer's input matmuls and `ssm_out`, for steps
holding several slots' tokens. Two forms with one layout and interface: for up to 16 new tokens a slot (decode, verify
blocks) one fused kernel, a work-group per (slot, value head) in 32 KB of shared memory (conv, silu, l2 norm, the
recurrence, the gated norm, the quantization for `ssm_out`); for longer steps three (prepare, scan, gated norm), the
scan's threads each holding two columns of a head's state in registers and walking the slot's tokens. The recurrence is
llama.cpp's with the decay folded in (`kv = eg S^T k; d = beta (v - kv); o = eg S^T q + d (k.q); S = eg S + k d^T`);
value head h reads key head h % 16 (the converter's tiled order).

Rollback: a verify block runs with `commit = 0` (outputs from the committed state walked through the block in
registers, nothing written back, the tokens' post-conv inputs and raw rows saved as pending); the slot's next step says
how many were accepted and the scan replays them first through the same loop. Against a twin that ran only the
accepted tokens: 1,524 of 1,524 comparisons bit-identical per form (blocks of 2-16, every acceptance count, 200-block
random chains), and a mutated replay fails it. A separate replay loop did not stay bit-identical (IGC fused the
multiply-adds differently), hence the one loop. The fused and split forms agree to a few ulp, not bitwise. Against a
double emulation in llama.cpp's op order over 4,200 tokens: rms error / rms signal 1.4e-7 on the output, 1.3e-7 on the
state, no drift; 4e-6 at the edges (decay from 0 to below -100, beta saturated).

Measured (`eklow-gdn-test`, fiona, GPU time for 30 layers): decode 0.59 ms (0.66 wall), two slots 1.0 ms; verify blocks
of 8 and 16 tokens 0.79 and 0.96 ms; a 2,048-token prefill chunk 84 ms (138 ms for two slots at once); a slot's
state (64 MiB) to or from pinned host memory in 6 ms, exact. Decode sits at about twice its state traffic (2 MiB read and
written a layer, 316 us for 30 layers at 399 GB/s). Prefill is the weak part: ~1 us a token and layer, not the 0.2-0.3
estimated, about 17% of the ~400 ms the rest of a chunk was estimated at. The scan is bound by its threads' loads of k and
q (each thread takes its own 1 KB a token; an Xe-core delivers 64 B a clock); with them removed it runs at 0.58 us.
Four columns a thread halve that traffic but leave one slot only 1,024 threads (slower for the common case). Untried,
in order: carrying the decay's running product instead of rescaling the state every token, k and q in fp16, the chunked
form on the matrix units.

## 2026-10-31 — Qwen's weight formats on the card and the CPU: what each costs

`qwen/mm`. Every format the `qwen/weights` candidates use now has kernels on both: Q8_0, Q4_K, Q5_K, Q6_K and IQ4_XS
beside Q4_0, each at the bytes per weight of ggml's own (8.5, 4.5, 5.5, 6.5625, 4.25 bits; Q4_0 4.5). Raw output of
every number below: `docs/bench/qwen-mm-*` and `xmx-fiona-2026-10-31.*`; the tools are `eklow-xmx-bench`,
`eklow-qfmt-test` (card) and `eklow-cpu-qgemv-test` (CPU).

**What the matrix units take** (`eklow-xmx-bench`, extended). Every integer operand pair runs and every one at the rate
of its larger operand, s8 x anything 197.6 TOPS, s4 x s4 and u4 x u4 395.3: s8 / u8 activations against s8, u8, s4,
u4, s2 and u2 weights (a 2-bit operand does not double the rate against s8). B's layout follows one rule, a dword
holds 32 / bits consecutive k of one column, dword (k / G) * 8 + n, for all of them (the s8 x s4 finding generalizes;
checked against a CPU reference for all 16 pairs). fp16 x bf16 mixed, and tf32, are refused by the compiler's
static_asserts; the 16-wide execution size compiles and runs on DG2 (native is 8; not checked further). This is what
makes the formats cheap: Q4_K and Q5_K feed the nibbles to the u4 dpas as they are (the minimum comes off with one
more dpas against a tile of ones, which sums the activation block), Q5_K's fifth bit goes through a u2 dpas whose
operand is two bit-masks of one stored word, and Q6_K's low 4 bits and high 2 bits through a u4 and an s2 dpas (the
xor of the stored field makes q - 32 = low + 16 s2), each on one 16-weight half of the tile (the other half of the B
operand zero), so no format unpacks to bytes. Only IQ4_XS needs a lookup.

**Decode GEMV** (int8 activations per 32-block as before; `Weight` repacked per format at load into tiles of 8 columns
x 32 K plus per-superblock records, `src/kernels/xmx_fmt.hpp`; one schedule for all: a work-group of T threads per
8-column group splits K and merges in shared local memory; 1 row dpas with repeat count 1, 2-32 rows 8-row tiles).
GB/s of weight read, one row (the card reads 399): attn_qkv 2048 -> 8192: Q4_0 279, Q8_0 303, Q4_K 269, Q5_K 286,
Q6_K 295, IQ4_XS 161; the head 2048 -> 248,320: 367, 364, 354, 358, 357, 218. Every format but IQ4_XS reads at the same rate as
Q4_0's, so a decode token's time follows the bits per weight; the small matrices (an expert's 0.6-1.2 MB) are
launch and latency bound at ~8-9 us a call whatever the format. The grouped GEMVs of the experts (256 experts,
distinct ones read, P = 8 pairs, gate|up + SwiGLU): 196 (Q4_0), 243 (Q8_0), 178 (Q4_K), 194 (Q5_K), 208 (Q6_K), 122
(IQ4_XS) GB/s, so per byte Q4_K costs 10% more than Q4_0 and per weight (bits / rate) Q5_K +23%, Q6_K +37%, Q8_0 +52%,
IQ4_XS +52%. At 16 and 32 rows
(a verify block) the K-quants' scale work shows: attn_qkv at 32 rows 98-107 GB/s against Q4_0's 134 and Q8_0's 176.
- **IQ4_XS** is bound by its table: the 16-value int8 table is looked up per packed byte (a 256-entry table of int8
  pairs in shared local memory, one gather lane for two weights, 16-bit entries +3-4% over 32-bit). With the gather
  taken out (wrong results) the kernel reads 238 / 341 GB/s (qkv / head) instead of 161 / 218. Register-indirect
  `iselect` as the lookup is slower still (211 GB/s as an arithmetic ceiling, in registers with free memory), a
  16-bit select tree 358, the gather 730; all the chunk's gathers issued together before the dpas: -8%. At 4.25 bits it
  costs as much time as Q8_0 does at 8.5 on the grouped GEMVs.
- The Q6_K head: the dpas GEMV (int8 activations) reads at 357 GB/s at 1-8 rows, `fast::matmul_q6` (fp32
  activations, one sub-group a column) at 253 GB/s at one row, 150 at two, 46 at eight: 1.4 / 2.4 / 7.6x faster. The
  price is int8 activations on the head (1e-2 of the output vs fp32 activations); not wired into the engine.

**fp16 GEMM** (dequantized to fp16 in shared memory per format, `GemmDec` in `xmx_fmt.hpp`; scales rounded to fp16 as
Q4_0's, so ~4e-4 of the fp32 product, against ~2e-4 for Q4_0): at 2,048 rows attn_qkv 54.5 (Q4_0), 51.2 (Q8_0), 49.8
(Q4_K), 49.1 (Q5_K), 49.6 (Q6_K), 47.8 (IQ4_XS) TFLOPS; the grouped expert GEMMs (2,048 tokens x 8 of 256 experts,
tile 128, gate|up + SwiGLU epilogue, TFLOPS on real pairs) 27.1, 24.8, 26.5, 23.9, 24.1, 16.4. The K-quants cost 8-9% of the
GEMM at the big shapes, IQ4_XS 12% there and 40% on the experts. The 32-row-tile grouped kernel spills 256-1,216 bytes
for the new formats (the Meta registers); tiles 64 and 128 do not.

**SwiGLU**: `matmul_geglu`, `matmul_geglu_grouped`, `matmul_geglu_grouped_dense` and `matmul_geglu_grouped16` take a
`Glu` (default Gemma's GeGLU, existing callers unchanged); the GEMM epilogue and the int8 requantizing kernels branch
on it. Checked against the fp32 silu(gate) * up of the dequantized halves.

**CPU** (fiona's i7-12700H, AVX-VNNI, ggml's bytes read in place; `src/cpu/qgemv.cpp`; the same int8 activations):
its two DDR4 channels read 43 GB/s (8 threads and up; 25 on one). One cold expert's FFN (gate|up GEMV, SwiGLU,
requantize, down GEMV; 1.67-3.34 MB), best of 1-16 threads, one input: Q4_0 44 us (40 GB/s), Q4_K 45, IQ4_XS 42, Q5_K
53, Q6_K 64, Q8_0 77 us (43.5 GB/s): every format reads at the DRAM rate, so time follows bytes (the pshufb lookup of
IQ4_XS costs nothing here). With 8 and 16 inputs (tokens of a verify block that pick the same expert) it is arithmetic
bound: 122-156 us and 233-290 us, 15-19 us a token against 42-77 for one: sharing the read makes a block of eight cost ~3
singles. Q4_0 is the fastest there (233 us at 16 inputs), Q8_0 next (251: its sign trick has fewer ops than the K-quants'
scale work), then Q4_K 267, IQ4_XS 278, Q6_K 280, Q5_K 290.

**Correctness**: every kernel against ggml's own dequantization at the fp32 activations (1e-2 up to 32 rows, 2-5e-4
above) and an exact emulation of the arithmetic on the card's own int8 activations (<= 1 float epsilon of the sum of
absolute terms, ~1e-6), on real tensors (Gemma 12B requantized by llama-quantize to each format) and random valid
blocks, plus grouped, interleaved gate|up, `rows_of` views, and the grouped GEMMs against the dense path (bit-equal).
An early emulation that quantized the activations on the host showed 2-4e-4 outliers: a quantization tie rounds either
way by an ulp of the reciprocal, the same as the old 5.8e-4 at `attn_output` 16 rows. Gates: `bench.sh kernels`
unchanged (same numbers as main), `attn` all ok.

**Which formats are cheap enough** (card and CPU costs alone): the card reads Q4_K, Q5_K, Q6_K and Q8_0 at Q4_0's
rate, so an expert in Q4_K costs 10% more decode time than in Q4_0 at the same bytes and the CPU none; Q5_K and Q6_K
cost about their bytes (+22% and +46% on the CPU, +23% and +37% on the card's grouped GEMVs); Q8_0 is for the dense matrices
(1.9x the bytes of 4.5 bits) and is the fastest per byte on the card; IQ4_XS saves 6% of the bytes and costs 1.7x the
card's decode time and 40% of its expert prefill, though the CPU reads it as fast as Q4_0. The dense matrices and
the head are 1.5 GB of the ~20, read once a token at 300-360 GB/s whatever the format: their cost is their bytes.

## 2026-10-31 — Qwen3.6's text layer

`qwen/text` (merged): the tokenizer (`src/text/tokenizer_qwen.*`), both chat templates (`chat_qwen36.*`: stock
Qwen3.6's and Ornith's, chosen by the template's exact text, anything else refused), the output parser
(`output_parser_qwen.*`) and the tool-call grammar (`src/serve/tool_grammar_qwen.*`), held to Hugging Face as the
Gemma ones were (transformers 5.17, tokenizers 0.23.2). The rule from the Gemma port stays: request text never becomes
an added token, special or not (`<think>` or `<tool_call>` typed by a user are characters).

- **Tokenizer**: added tokens, NFC, the pre-tokenizer's regex written by hand over code points, byte-level BPE. Against
  `tokenizers`: 648,044 strings / 14.4M tokens encoded and 1.36M decodes, 0 differences; every code point in 17
  contexts (18.9M strings), 0. 26-31M tokens/s on one core. Hugging Face's NFC is Unicode 9 (the Rust crate's data),
  not 16: with Python's full tables 371 of 10.2M probes differed. The generator leaves out what the installed
  `tokenizers` does not know (120 combining classes, 21 decompositions, 21 compositions), so eklow matches Hugging
  Face; regenerate if `tokenizers` moves (`tools/ref/qwen_classes_check.py` shows it). The GGUF marks 33 tokens
  special against tokenizer.json's 26 (7 audio / TTS ones, plus 243 unused `[PAD...]`): those are plain text in encode,
  nothing in decode, and never sampled.
- **Templates**: 39,177 cases per variant (5,177 written for the branches, 34,000 random), 0 differences, and 7.8M
  tokens of ids, 0. Ornith's repo ships stock's `tokenizer_config.json` byte for byte, stock template inside, while its
  own is `chat_template.jinja`, which transformers uses; llama.cpp's converter takes the former, so **an Ornith GGUF
  converted as-is carries stock's template** (it drops earlier reasoning that Ornith's keeps). And the official Ornith
  GGUFs (`ornith-ai/Ornith-1.5-35B-A3B-GGUF`) carry a third, Unsloth's edit of Ornith's (a `developer` role, two
  system messages merged, arguments by mapping test). So the server takes the template variant as an option when the
  file's is not one eklow knows, and renders the model repo's own template.
- **Parser**: generation starts in the reasoning (the prompt ends `<think>\n`), then content, then
  `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>` calls. Values go
  back to JSON by the request's schema, after vLLM's `qwen3_coder` parser with three differences: a string parameter's
  "null" stays the string, 5.0 stays 5.0, a value that does not parse as its type stays text (vLLM makes a bad boolean
  False). 40,000 rendered turns (60,078 runs, 72,022 calls) parse back exactly. One hole no parser can close: a string
  value containing `</parameter>` followed by `</function>` or `<parameter=`.
- **Grammar**: a byte automaton keeping calls in the template's shape: declared tools and parameters (each once,
  required ones before `</function>`), free text for string parameters, JSON of the declared kinds for the rest, enums
  exact. 20,432 valid calls accepted token by token as Hugging Face tokenizes them, 12,000 mutated ones refused at
  their first bad token, 40,000 random walks closed into calls the parser accepts; bans equal refusals on 828 of 828
  (state, special token) pairs.

## 2026-10-31 — Qwen3.6's reference forward, and what the engine is held to

`qwen/ref` (merged): `src/model/qwen35.*`, `eklow-ref-run` on qwen35moe, and the reference kernels it needs
(`ref.*`: the causal conv, l2 norm, the gated delta rule, the gated norm, SwiGLU, the MoE's gathers; `ref::matmul`
dequantizes Q4_K, Q5_K and IQ4_XS as ggml does, bit-identical to the gguf package on real tensors for Q4_0, Q4_K, Q6_K,
Q8_0). A layer's weights (~450 MB) go to the card as the layer runs: the file is larger than the card.
`eklow-oracle` runs qwen35moe unchanged; `nix/oracle-cuda.nix` builds it against llama.cpp CUDA for e-desktop's 4090.

Layout facts (llama.cpp's graph and converter): attention layers are those with (l + 1) % 4 == 0; `attn_q` interleaves
256 query and 256 gate columns head by head; DeltaNet value head h reads key head h % 16; the GGUF's norms carry the
+1 except `ssm_norm`; "attn_post_norm" is the pre-MoE norm (`x2 = x + attn(rmsnorm(x)); x = x2 + moe(m) + shared(m)`,
m = rmsnorm(x2) * post_attention_norm), no norm on sub-layer outputs; RoPE is NeoX over the first 64 of 256 dims at
base 1e7 (IMRoPE with equal position components); the MoE renormalizes its top 8 by their sum (floored at 2^-14), the
shared expert is scaled by sigmoid(m . ffn_gate_inp_shexp) and added.

Checks (prompts tokenized by the oracle: `long`, DESIGN.md as a user turn with thinking on, 2,573 tokens; `chat`, 27):
- A tiny random qwen35moe in F32 (4 layers, the real tensor names): every tensor of every layer within 1e-7..5e-6 of
  llama.cpp CPU, logits 27/27 at KL 1.7e-11: no head order, norm or rope convention is off.
- The real files: layer 0's dense tensors within fp32 rounding of llama.cpp CPU (4e-7..5e-6 at `long`, where its CPU
  matmuls run in float); the first difference is the routed experts (8e-3, ggml's int8 activations), and `l_out` then
  grows smoothly to ~9e-2 at the last layer, with no step at a layer-type boundary.
- Logits, last 256 positions of `long` / all of `chat` (stock EK8): llama.cpp CPU against CUDA 231/256 top-1, mean KL
  9.7e-2 (median 6.8e-3) / 26/27; CPU against the reference 243/256, 1.18e-1 (4.1e-3) / 26/27, 3.5e-4; CUDA against the
  reference 228/256, 1.51e-1. Ornith EK8, a less certain model on the same text: 203 / 214 / 203 of 256. The mean is a
  few positions' (the digits of dates in DESIGN.md, 1-9 nats of entropy, where another set of experts flips a digit).

So, as with the 26B, llama.cpp's two backends are no closer to each other than either is to the reference, and the
fp32 reference is the engine's ground truth. The gate for the engine against it (from the CPU-CUDA distance, stock
EK8): `long` top-1 >= 225/256, mean KL <= 1.2e-1, median <= 1e-2; `chat` top-1 >= 25/27, mean KL <= 2e-3; Ornith
`long` >= 195/256, mean KL <= 2e-1. Judge by top-1 and median: three positions move the mean ~5x. Dumps on fiona:
`~/eklow-oracle-qwen36/`, `~/eklow-ref-qwen36/` (`NOTES.txt` has the files' sha256). The reference takes 19 s on
`chat` and 139-163 s on `long`.

## 2026-10-31 — the DFlash 2 draft on the card, and lossless acceptance

`qwen/dflash`. `src/model/dflash2.*` (config, weights from the draft GGUF, `Dflash2Ref` the fp32 reference forward,
`Dflash2` the serving forward in `dflash2_fast.cpp`), `src/kernels/ref_dflash.*` and `dflash.*`, `src/serve/spec.*`
(walk over the selector's lattice, greedy and rejection-sampling verify), `tools/dflash_test/` (`eklow-dflash-test`
torch|fast|accept|matmul|time, `eklow-spec-test`, `ref_torch.py`). Shared files touched: `attn::attend` gets head 128 with
4 query heads per K/V head (fp16 ring only: three lines), `Sampler` gets `distribution()` (sample() draws the same
tokens as before: 55,000 of 55,000 equal to the previous implementation).

**Checks.** Reference vs z-lab's model.py in PyTorch fp32 on real target features (eklow-oracle on stock Q8_0, a 523-token
chat, layers 1, 6, ..., 37; BF16 draft file): hidden rel error 3-6e-6, lattice 1e-6, 8 of 8 greedy paths equal, for Inco's
draft at B = 8 and 16, jzinno's at B = 16, and with the window cut to 64 and 100 (ring and window logic). model.py's
draft predicts the text well (7 of 7 tokens at several positions), which also confirms the layer indices: the GGUF's
`target_layers` are the layer outputs plus one (llama.cpp reads layer inputs). Serving vs reference: the matmuls on int8
activations equal the reference on the same rounded activations to 2e-7; the whole forward is chaotic at int8 (a 1e-7
perturbation of the input moves the logits ~5%, the same as serving vs exact), so judged by what the draft does: mean
tokens agreeing with the text, 100 blocks of 8: exact fp32 3.570, serving 3.580; B = 16 Inco 4.393 vs 4.518, jzinno 4.143
vs 4.143, window 100 2.375 vs 2.375; whole greedy drafts equal in 41-97% of blocks, the rest differ at near-ties.
**Acceptance:** chi-square over 11 sampler configurations (temperature, top-k, top-p, min-p, penalties, bans, grammar,
all of them) with a poor and a near-target drafter: all pass; two deliberately broken verifies (accept-all, no residual)
fail every time; greedy sequences equal plain greedy in 20,000 of 20,000 per case.

**Timings** (A770M, Inco Q8_0, 780 MiB on the card with two slots): context injection 3.7 us a row at 1,024-2,048 rows,
0.4-0.5 ms for 1-16 rows; a block of B = 8 5.6 ms (layers 3.1, head 1.6, selector 0.8), B = 16 6.3 ms; two slots at B = 8
6.4 ms, at B = 16 11.5 ms (the head's 32-row GEMV 5.8).

**Where model.py and llama.cpp differ.** llama.cpp drafts n_max tokens from a block of n_max + 1 (the block size is the
caller's), applies an optional p_min cut on the walk, and runs its sliding window in a unified KV cache; the target-layer
ids in its GGUF are +1. Same math otherwise (the conv taps, the lattice layout, ties to the first candidate).
**Not done:** a token-for-token comparison with llama.cpp's draft (its draft graph needs a target context and the
private `llama_get_embeddings_nextn`); BF16/F16 draft files only run in the reference (xmx::matmul_geglu cannot form
fp16 outputs for few rows).

## 2026-11-01 — the serving engine for Qwen3.6-35B-A3B

`qwen/engine` (merged): `src/engine/engine.*` rewritten for qwen35moe; the Gemma 4 chat engine and its reference
(`src/model/gemma4.*`) are gone (EmbeddingGemma 2, the vision encoder and Gemma's text code stay). Steps of prefill
chunks, decode tokens and verify blocks (B <= 16 tokens with every logits row, the DeltaNet tokens left pending until the
slot's next chunk says how many were accepted, the K/V past them dropped); capture of layers 1, 6, 11, 16, 22, 27, 32,
37's outputs for the draft; `settle`, `snapshot` / `restore` of a slot's recurrent state for prefix reuse. Weights in
their file's formats (EKK8: Q4_K experts, Q8_0 dense, Q6_K head through the dpas GEMV); experts placed rank by rank by a
routing profile (`data/qwen36-experts.json`, `data/ornith-1.5-experts.json`) in the card's free memory less the 1 GiB
margin and a 1.2 GiB reservation for the draft; the cold ones on the CPU from pinned memory, the busiest streamed for
prefill. `eklow.service` stayed stopped; `server` is out of the build until the next task rewires it.

**Fidelity** (`checkq.sh`, against the fp32 reference on the 2,573-token prompt's last 256 positions, six ways: chunks of
1,024 and 2,048, the last 64 decoded, verify blocks of 8 accepting 3, mixed acceptances, blocks of 16 accepting 5): stock
EK8 238-245/256, median KL 2.2-4.5e-3; stock EKK8 237-246, 1.7-3.5e-3; Ornith EK8 201-221, 2.2-3.6e-2; `chat` 27/27,
27/27, 25/27 (mean KL 8.6e-4, 9.8e-4, 1.4e-2). Verify with replay is within the noise of plain decode. An apparent 10x
gap between EKK8's and EK8's `chat` KL (2.2e-3 against 2.2e-4) came from an earlier build and from placement: on 27
tokens the KL moves 10x with how many experts sit on the card (EK8 1.5e-4 to 1.6e-3 over 0-96 of them), in both files.

**At the production shape** (2 x 262,144, int8 KV): 6.46 GiB of weights and 5.84 GiB of KV and DeltaNet state on the
card, 7.16 GiB free before the experts, **71 of 256 experts a layer on the card**, ~12.2 GiB of cold experts in pinned host
memory. `eklow-engine-bench`, no profiling (prefill: 4,096 tokens after the depth; decode per sequence):

| | stock EKK8 | stock EK8 | the 26B in production |
|---|---|---|---|
| prefill from empty (2,048) | 1,297 tok/s | 1,402 | |
| prefill at 8k / 32k / 128k | 1,071 / 848 / 393 | 1,189 / 808 / 408 | 1,865 / 1,794 / 1,177 |
| decode, one slot, at 0 / 8k / 32k / 128k | 41.3 / 39.6 / 36.8 / 31.4 | 41.7 / 40.3 / 39.1 / 31.9 | 36.6 at 8k, 34.0 at 32k, 30.5 at 128k |
| decode, two slots, per sequence | 32.4 / 32.1 / 28.6 / 24.5 | 33.0 / 32.9 / 29.8 / 24.6 | |
| verify block of 8 / 16 at 8k, one slot | 70.9 / 112.8 ms | 67.6 / 106.4 ms | |

Decode matches or beats the 26B; prefill is behind it everywhere and a third of it at 128k. Where the time goes
(`--profile`): decode is launches plus the round trip to the CPU's cold experts (half the pairs are cold at 71 a layer:
9.3 ms of CPU a step, and the card waits 9.4 ms for routing); a verify block of 8 costs 2.8 decode steps and is bound by
the CPU (43.8 ms of cold experts; the card finished first in 38 of 40 layers), with `moe_sum` 35% of the card's part;
prefill is the streamed experts' copy and repack (41% from empty, 36% at 8k) and, deeper, the attention layers' prefill
attention (37% at 32k, 63% at 128k: the two-GEMM segments, DECISIONS 2026-10-27, exist only for head 512, and Qwen's
heads are 256). Those four are the levers.

## 2026-11-01 — prefill attention at depth for Qwen3.6: the segment path for head 256, group 8

The 26B's two-GEMM segment path (2026-10-27, "global prefill attention as two GEMMs") now also serves Qwen3.6's 10
attention layers (16 query heads over 2 K/V heads of 256, int8 or fp16 pages): `attn::reserve` takes the group, and
`run` takes the segment branch for D = 256, G = 8 when a chunk has 64 tokens or more; shorter chunks and decode keep the
flash kernels. The engine's change is the one `reserve` argument: its scratch (128 MB for a 2,048-token chunk) is
allocated before the experts are placed, so the placement's free-memory reading counts it.

- `eklow-attn-test` `qwen8` / `qwen`, 2,048-row chunk (was flash): 13.4 -> 27.6-28.3 TFLOPS at 16k, 64k, 128k and 256k
  keys; two slots' chunks at once 27.6-27.7. 4,096-key segments (kSeg, shared with head 512, whose numbers did not
  move: 27.4) against 2,048: 27.6 vs 25.6 at 16k, 27.7 vs 23.8 at 128k, so 4,096 for both heads (twice the
  probabilities' memory: 128 MB). P.V's fp16 tail mask as scalar dword masks per row removed a spill. SegQK still
  reports 32 bytes of spill, with no measurable cost.
- `bench.sh engineq` (2 x 262,144, int8 KV, 2,048-token chunks), prefill at depth 8k / 32k / 128k, tok/s:
  1,091 / 851 / 390 -> 1,180 / 1,032 / 539 (+8% / +21% / +38%); the 128k chunk's attention is ~3.9 s of 8.2 s. Decode at
  0 / 8k / 32k / 128k: 24.5 / 25.4 / 25.6 / 31.5 ms -> 24.3 / 24.9 / 25.4 / 31.2 ms; verify blocks unchanged. Cold-expert
  waits (5-7 s of the 128k prefill) are now the larger share and are the MoE worker's.
- Decode attention of these layers (split-K): the first level's blocks were fine (main kernel 585 us of 685 at 128k
  keys, ~460 GB/s of K/V), the cost was the fold between the split levels, 92 us of 750 at 128k: its work-items now run
  the dimensions fastest (128-byte reads of the partials) and folds by 8 from 8 splits up, level by level between the two
  partial buffers (fold and merge 12 us each). `qwen8` decode 128k 755 -> 690 us, 256k 1,196 -> 1,124 us, 16k 275 -> 126 us
  (391 / 478 GB/s of K/V counting the merges); a step's saving ~0.7 ms of 31 at 128k (10 layers), so the decode step at
  depth is the experts', not the attention's. Splits' count and blocks per split (kPerSplitMax 8 vs 4..16) were not
  better elsewhere.
- Gates: `bench.sh attn` every kind ok; `checkq ekk8` long 248/256 (1,024 chunks), mean KL 3.5e-2, median 2.0e-3, chat
  27/27, 9.8e-4; `checkq oek8` long 216/256, mean KL 1.08e-1, chat 25/27; `kernels` unchanged. Raw output in
  `docs/bench/*-fiona-2026-11-01.txt` (the "before" 128k profile's attn row overflowed; its engineq rows are good).

## 2026-11-01 — the MoE's speed for Qwen3.6: what bounds it (measured; one CPU kernel change kept)

`qwen/moeperf`. Raw output in `docs/bench/qwen-moeperf-*-2026-11-01.txt`.

**The cold experts' GEMVs are at the DDR wall in decode and verify.** A multithreaded AVX2 read of 2 GiB gives 38-41 GB/s
on fiona's i7-12700H at 4-20 threads (`mb/bw.cpp`: 41.2 at 8, 38.8 at 16), and the microbenchmark of a verify layer's cold
work (18 distinct experts, 33 pairs, `mb`) runs at 41 GB/s. In the engine a verify block of 8 reads 712 distinct cold experts a
step (1.26 GB) in 41.7 ms of CPU: 30 GB/s; decode 160 experts in 9 ms, 33 GB/s. So the CPU phase is ~75-80% of the wall and
what is left is barriers and imbalance between the gate|up, SwiGLU and down phases. The software prefetch and the 2-3 input
passes of 708e456 already took what the cores' own latency left. Consequences: (1) copying a block's busiest cold experts
to the card does not remove bytes from DDR (the copy reads them too, once, as the CPU does), it moves the compute; (2) nothing
that is a kernel makes a verify block's CPU phase much shorter, only fewer distinct cold experts (more on the card) does;
(3) `moe_sum` in the verify profile (33-45%) is not the combine: it is the mark after `cold_prefill`, i.e. the card
waiting for the host (the combine is a few rows).

**Prefill's CPU phase is compute-bound, not bandwidth-bound**: the pairs that are not streamed (~1.8k a layer over ~120
experts of ~15 rows each) run at 8 GB/s and ~15 us a pair. The Q4_K dot now builds the superblock's 8 scales and minimums as
vectors once and takes each input's scale products through the stack as broadcasts, with the minimum terms accumulated as a
vector: **14.7 to 10.2-11.8 us a pair** (`mb`, 120 experts x 1,800 pairs; 18.9-19.2 to 16.2-16.5 at 400 pairs), results
unchanged (`eklow-cpu-qgemv-test` vs emul 1e-6; checkq ekk8 241/256 median KL 3.5e-3, ek8 238/256 4.5e-3, oek8 201-203/256,
all inside the DECISIONS 2026-11-01 ranges). In `engineq` it moves prefill from empty within noise (1,297 before, 1,315
after; CPU phase 1.07 to 0.93 s, the card's wait for the routing up by the same): at this operating point the CPU and
the card paths are balanced, so a faster CPU alone shifts the balance rather than lifting the throughput.

**The streaming cost model is at its optimum** (prefill 2,048 from empty, tok/s): `stream_cpu_us` 5 / 7 / 10 / 15 / 20 / 30 =
1,085 / 1,172 / 1,239 / 1,202 / 1,144 / 1,119; `stream_copy_us` 200 / 270 / 350 = 1,224 / ~1,240 / 1,160; `stream_min_rows` 8
the same as 16. Streaming more makes the card's path (copies, their DMA tax on every kernel, repack) the longer one
(waiting on the routing 0.4 to 1.0 s), streaming less the CPU's (1.1 to 1.4 s). Chunks of 4,096 do not help either:
the activations cost ten ranks of experts (61 of 256 on the card instead of 71) and the run gives 1,225 tok/s (CPU 2.43 of 3.34 s).
Prefill from empty: the copy of 63 experts a layer (~4.5 GB a chunk, 410 ms at the link's 10.9 GB/s) plus the repack is the
41%; the DMA tax of DECISIONS 2026-10-27 makes it also a cost on every other kernel, so the copy bytes are the lever. A
repacked pinned copy would remove the repack and the raw slots but needs a second 12 GiB of pinned memory; fiona has 30 GB.

**Not done / left**: (a) the iGPU of the 12700H (96 EU, int8 dp4a, same DDR) as a second compute device for the
non-streamed pairs; (b) a SYCL command graph or fused kernels for the decode layer's ~25 launches (the card's critical chain
between two CPU phases is ~0.25 ms a layer, 10 ms a step); (c) a smaller hot KV footprint for more experts on the card, which
is the only thing that shortens verify (each rank is 40 experts, ~71 MB, and one fewer distinct cold expert a layer); (d) the
profile `--profile` of a verify block with timeline per layer.

`engineq` after (no profile; before = the 2026-11-01 table): prefill 1,315 (1,297) / 8k 1,135 (1,071) / 32k 867 (848) / 128k
395 (393) tok/s; decode 41.3 / 39.5 / 37.4 / 31.3 (41.3 / 39.6 / 36.8 / 31.4); verify 8 at 8k 72.2 (70.9) ms, 16: 113.6 (112.8). All within run-to-run noise (fiona's
CPU is shared with the other workers' server tests): the kernel change is a CPU saving that these paths do not turn into tok/s.

## 2026-11-01 — eklow-server on Qwen3.6, with DFlash 2 and snapshots of the DeltaNet state

`qwen/serve`: `eklow-server` and `eklow-prompt` are back in the build on the qwen35moe engine, linking `eklow_spec` (sampler
and acceptance). Chat uses Qwen's tokenizer, template (stock or Ornith, from the GGUF or `--chat-template`), parser and
tool grammar; images are refused with a clear 400; sampling defaults come from `general.sampling.*`. Engine changes: none
beyond reading `Engine::head_weight` for the draft (which shares the target's head) and a 1.3 GiB `draft_reserve_bytes`.

**Speculation.** A generating slot's step is a verify chunk of its sampled token and B-1 drafted ones; `serve::verify`
decides with the slot's sampler and a grammar and parser cloned per row; two slots draft in one pass. The block is a
trade the card sets: a verify row beyond the first costs ~8.5 ms (the cold experts' CPU time grows with the distinct
experts the rows touch). GSM8K, one stream at 8k: plain 35.4 tok/s, blocks of 4 / 6 / 8 / 12 / 16 56.3 / 54.5 / 53.1 /
50.1 / 42.7 (3.5 / 4.6 / 5.3 / 6.6 / 6.5 tokens a verify); prose at 8k: 4 beats 8 by a third. Default block 4 (or the
draft's if smaller); `"speculative": N` per request (0 or false: plain). With two streams the blocks of 4 and 8 finish
the same 16 requests in the same time (449 against 428 s on GSM8K at temperature 1), so 4 costs nothing there.
Acceptance with blocks of 8, temperature 1, against llama.cpp's on the same pair: GSM8K 5.75 (5.24), HumanEval 5.43
(4.98), MT-Bench 5.02 (4.53); eklow's is higher, not lower. Through the server, 3,000 completions at temperature 1 with the
draft against 3,000 without: chi-square 17.1 on 32 degrees of freedom, p = 0.985. At temperature 0 the outputs agree for
hundreds of characters and then part at a near-tie (the int8 activations depend on the step's shape; a request alone and
beside another do the same): that is the engine's noise, not the acceptance's.

**Snapshots.** 64 MiB each in pinned host memory under `--snapshot-mib`, at the starts of the request's messages (the
prefix a next turn shares), before the generation prompt, at the last two paragraph breaks before the last message (a
document and then another question), every `--snapshot-every` tokens of a long prefill, and at the end of a generation;
a new request takes the slot and snapshot that leave the least to prefill. A 10k conversation: 9.2 s cold, next turn 0.61 s
(stock; 105 tokens prefilled) or 0.26 s (Ornith, which keeps the reasoning); a question on a read 100k document 0.28-0.31 s
against 146-159 s. Finding: the end-of-generation snapshot is unusable when the template's re-rendering of the answer
tokenizes differently from what was sampled (a sampled " D" + "amas" against the tokenizer's " Damas"; Ornith's third turn
in the bench lost 677 tokens, +1.1 s). The server now logs where a slot's sequence and a prompt part. A snapshot every
~256 generated tokens would bound the loss; not done (each needs the verify's pending state settled mid-generation).

Gates and the tables are in README "Performance" and `docs/bench/serve-*-qwen-*`: whole and streamed chat, reasoning,
tool calls, stop strings, seeds, two at once, cancel, reuse (`serve_test.py basic`, all ok with the draft, without it,
and with Ornith); needle at 88k and 167k found with two requests and embeddings at once (embeddings 91-238 ms beside the
prefills).

## 2026-11-01 — the draft for both models, and the files on Hugging Face

**The draft.** llama.cpp's acceptance (tokens per verify, temperature 1, top-p 0.95, top-k 20, thinking on, 32
requests each; `docs/bench/qwen36-dflash2-llamacpp-2026-11-01.txt`): on stock Qwen3.6 (EKK8) Inco's draft gives 5.24 /
5.08 / 4.59 on GSM8K / HumanEval / MT-Bench at block 8 and 6.58 / 6.14 / 5.49 at block 16; jzinno's (trained on Ornith)
4.59 / 3.76 / 3.87 and 5.49 / 4.08 / 4.23: Inco's is clearly better there. On Ornith (EK8) the two tie at block 8 (Inco
5.21 / 5.01 / 4.01, jzinno 5.23 / 5.00 / 3.92) and jzinno's leads a little at block 16 on two of three (6.51 / 6.27 /
4.49 against 6.43 / 6.04 / 4.59). The protocol check reproduces Inco's own numbers (5.24 / 4.98 / 4.53 against 5.21 /
4.90 / 4.41 on SGLang). On this card a verify row costs ~8.5 ms of the CPU's cold experts, so eklow runs short blocks
(the server's sweep: block 4 best, 56.3 tok/s on GSM8K at 8k against 53.1 at 8 and 42.7 at 16), where the drafts tie on
Ornith. The owner's rule ("use Inco's, the community one if it is better") gives **Inco's draft for both models**: one
draft file. Worth revisiting if verify rows get cheap enough for blocks of 16.

**The files** (the owner: no "eklow" in the names, the quant style instead; public, so fiona's Nix fetch needs no token;
uploaded from fiona, the draft's repo created by the owner):
- `ethanwtodd/Qwen3.6-35B-A3B-Q4_K-Q8_0-GGUF` @ 8db5016e: `Qwen3.6-35B-A3B-Q4_K-Q8_0.gguf` (EKK8), sha256 48cf98f4...
- `ethanwtodd/Ornith-1.5-35B-A3B-Q4_K-Q8_0-GGUF` @ a89886e0: `Ornith-1.5-35B-A3B-Q4_K-Q8_0.gguf` (EKK8 with Ornith's
  own chat template), sha256 25a68129...
- `ethanwtodd/Qwen3.6-35B-A3B-DFlash2-Q8_0-GGUF` @ 65aaab95: `Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf` (Inco's draft through
  the pinned converter), sha256 986b8f24...

The model cards carry the KL tables: each file beats the published Q4_K_M of its model (stock 0.0134 against bartowski's
0.0168 at 4k; Ornith 0.0354 against bartowski's 0.0460 and the official 0.0596). Working copies live on fiona
(`~/models`, `~/models/drafts`); e-desktop's home filesystem was at 99% and the owner had them deleted there after their
sha256 matched fiona's.

## 2026-11-02 — Qwen3.6's vision encoder (Ornith-1.5), checked against llama.cpp

Projector `qwen3vl_merger` (`model::VisionQwen3VL`, `src/model/vision_qwen3vl*.cpp`, `src/image/qwen3vl_image.*`,
`eklow-vision-qwen`): 27 layers of 1,152 (16 heads of 72), LayerNorm with bias, fused qkv with bias, "vision M-RoPE"
(pairs (i, i + 36) of a head turn by the patch's row for i < 18 and by its column for i >= 18, base 10,000), scale
1/sqrt(72), gelu (tanh) FFN of 4,304, the learned 48 x 48 table bilinearly resized to the grid with align_corners (ggml's
float arithmetic), a post LayerNorm, then four neighbours side by side (4,608) -> mm.0 -> gelu -> mm.2 -> 2,048-wide rows.
The two conv kernels (two frames) see the same picture for a still image, so they are one weight, their sum. Patches are
kept in the model's merge order (a 2 x 2 block consecutive), so output row r is the merged grid row by row.

**The projector.** `convert_hf_to_gguf.py --mmproj --outtype bf16` at the oracle's revision on Ornith-1.5-35B-A3B @
10fbf86fed7e (only `model-00001-of-00016.safetensors`, sha256 250b3b09..., holds the 333 `model.visual.*` tensors; the
converter was given a directory of the configs, that shard and an index cut to it). `~/models/mmproj-Ornith-1.5-35B-A3B-
bf16.gguf` on fiona, 902,821,920 B, sha256 5a1ad8ff16aa66f4e538b34976b36c6551c0279b06112e61e26658e110871b23 (334
tensors: the patch kernel is split in two; BF16 matrices, F32 vectors, patch kernels and position table). Against bartowski's
`mmproj-Ornith-1.5-35B-A3B-bf16.gguf` @ 64b0493d34: all 334 tensors byte-identical, only `general.*` metadata differs
(name, basename, license, tags). An F32 conversion (1.79 GB) is the oracle's projector for the checks. No `image_min/max_pixels`
keys are written: llama.cpp's own limits apply (8 to 4,096 tokens).
Stock Qwen/Qwen3.6-35B-A3B (@995ad96e, visual tensors in its shards 1-2 of 26): all 333 `model.visual.*` tensors are byte-identical to Ornith's (compared by range requests, `tools/vision/cmp_towers.py`): Ornith did not touch the tower, so one projector serves both models.

**Preprocessing** is mtmd's `dyn_size`: stb decode to RGB, `calc_size_preserved_ratio` with align 32 and the pixel limits
min/max tokens x 1,024 (llama.cpp's `--image-min-tokens` 8 / `--image-max-tokens` 4,096; eklow's tool defaults to a max of
1,024), Pillow bicubic into the target centred on black, then `(x / 255 - 0.5) / 0.5` in float. `image::qwen3vl_size` gives the
size (and so the token count) from the dimensions alone. The token grid equals llama.cpp's on all 17 images x budgets tried (up
to 4,096 tokens): 1 x 1, 17 x 9, 48 x 36, 2400 x 60, 5000 x 20, 20 x 3000, 70 x 1500, 6000 x 4000, grayscale, RGBA PNG, 16-bit
PNG, palette GIF, JPEGs. Patch embeddings agree with the oracle's to 1-2.5e-4 (its im2col rounds to fp16), so the pixels are the same.

**Encoder.** Reference: fp32 on `ref::` kernels, all weights on the card (861 MiB). Fast: xmx fp16 GEMMs (BF16 as fp16; qkv,
o, up (4,304 padded to 4,352), down (K padded to 4,320), and the merger as three more streamed pieces: mm.0's halves and mm.2),
`vattn` attention with heads of 72 padded to 96 and the 1/sqrt(72) folded into Q, fused LayerNorm/bias/residual/gelu kernels, the
picture cut and normalized on the card from the RGB bytes, the position table (pinned host) interpolated on the card. Weights are
streamed a unit at a time from pinned host memory into two 30.6 MiB staging buffers as Gemma's (`--resident` keeps all 30
units on the card instead).

| budget (big.jpg) | tokens | patches | streamed | resident | card, streamed / resident |
|---|---|---|---|---|---|
| 256 | 238 | 952 | 146 ms | 63 ms | 210 / 1,027 MiB |
| 1,024 | 986 | 3,944 | 387 ms | 299 ms | 267 / 1,084 MiB |
| 4,096 | 4,071 | 16,284 | 2.90 s | 2.83 s | 456 / 1,273 MiB |

(A run of the same binary on another day varied by 10%: 132 ms at 256 earlier; read the table as +-10%.) Streaming costs a near
constant ~65-85 ms (0.9 GB at ~13 GB/s) and saves 817 MiB, i.e. ~11 experts a layer of 256: it is the default. At 1,024 tokens
(profile, streamed): attention 136 ms (~14 TFLOPS on the real 72 dims), GEMMs 118 (27 TFLOPS), elementwise ~60, copy waits 78.
GEMM chunks of 2,048 rows beat 1,024 (+8%) and 4,096 (+4%). The card memory while encoding is mostly attention scratch (~75 MiB),
Q/K/V tiles, residual, GEMM output and the staging; the encoder frees all but 3 MiB on `release()`.

**Checks** against `eklow-oracle --encode-only` with the F32 projector (`tools/vision/check-qwen.sh`; the oracle now loads
only the vocabulary in that mode, 1.3 s instead of 20 s and gigabytes of repacked experts). The encoder is chaotic in some
rows: shifting every pixel of an image by one level (+0.4%) moves eklow's own rows by 0.08-0.12 total relative L2, with single
rows at 0.5-1.0; the stock BF16 projector in the oracle moves the oracle's rows from its F32 run by 2.0e-2 (tall, 64 tokens)
and 8.7e-2 (test-1, 256), worst row cosine 0.78. Layer by layer (tall, reference) the difference grows smoothly from 2e-4
to 6.6e-3, no step. Against that: reference vs F32 oracle, 64 tokens: total relative L2 1.1-5.6e-3, mean cosine >= 0.99999,
worst row cosine 0.9997; fast: 2.1-4.5e-3 (mean cos >= 0.99999); at 256 tokens the reference 1.6e-3..3.3e-2 and fast
2.2e-3..1.3e-2; fast at 1,024: 2.2e-3..3.6e-2 (gray.png and the alpha PNG 3.1e-2 and 3.6e-2, one row each at cosine 0.88-0.92);
at 4,096 tokens (6000 x 4000, 4,056 rows) 4.3e-3, mean cosine 0.999992. The band for the fast encoder: mean cosine >= 0.9993 and
total relative L2 <= 4e-2 against the F32-weight oracle, inside the oracle's own BF16-vs-F32 spread and its sensitivity to
a pixel level. Fast against reference: 1.9e-3 (test-1, 64 tokens) to 4.4e-2 (test-1, 256, the chaotic image).
Not done: the model's answers about images (the text side is `qwen/imgserve2`'s).

**Interface for the server**: `model::VisionQwen3VL enc(q, gguf::File::open(mmproj))`; `spec = enc.spec(max_tokens, min_tokens)`;
`image::qwen3vl_size(w, h, spec)` -> pixels, tokens = (W/32) x (H/32); `QwenImageRows r = enc.encode(bytes, spec)` (or
`encode(prepare_qwen3vl(decode(bytes), spec))`) -> `r.rows` (`DeviceRows`: fp32 `[n][2048]` on the card, shared_ptr) and
`r.grid_w, r.grid_h` (merged patches; n = w * h); `reserve(max_tokens)`, `card_bytes(max_tokens)`, `release()` as VisionGemma4V's.

## 2026-11-02 — image input for Qwen3.6 through the engine and the server

`qwen/imgserve2`. The language-model side of images (the encoder is `qwen/vision`'s, DECISIONS above), checked against
llama.cpp at the oracle's revision on Ornith-1.5 (EKK8) with bartowski's mmproj.

**What llama.cpp does** (mtmd.cpp, mtmd-helper, llama-kv-cache.cpp, ggml-cpu rope): the media marker becomes
`<|vision_start|>`, the rows, `<|vision_end|>` (the template's `<|vision_start|><|image_pad|><|vision_end|>` gives the same
ids). An image of nx x ny rows takes n = nx * ny K/V cells but max(nx, ny) positions; row i has the three rotary
components (p, p + i / nx, p + i % nx) and the text after continues at p + max(nx, ny); text tokens are (p, p, p). IMRoPE
sections [11, 11, 10, 0] interleave the 32 rotary pairs: pair j takes the time component when j % 3 == 0, the row's when
j % 3 == 1, the column's when j % 3 == 2. Qwen3-VL images are decoded causally (`mtmd_decode_use_non_causal` is false for
it; the rows share a time position, and the mask then falls back to raster order), the DeltaNet layers just see the rows.

**Engine.** `Chunk::images` (spans with their grid, rows from the host, never split across chunks) and `Chunk::rope_delta`
(sequence index minus rotary position at the chunk's start); K/V index stays the sequence index (`pos_`), the rotary
positions go to `attn_heads` as `rpos_` [3][tokens] with the config's sections. Text-only steps are bit-identical to before
(checkq ekk8: long 248/244/245/244/245/242 of 256, medians 1.9-3.6e-3, chat 27/27 at 9.5e-4). `Config::vision_reserve_bytes`
for an encoder loaded after the engine (one loaded before is in the free memory the placement reads).

**Draft.** As llama.cpp's DFlash path: the draft's positions are the target's rotary ones (time component), image rows are
not injected, and a block is drafted at the rotary position. Our draft restarts its context at the gap an image leaves
(`Dflash2::inject` with a later position), where llama.cpp's windowed cache keeps the text before the image; acceptance
after an image is therefore from the text after it (2.4-2.75 tokens per verify, blocks of 4, in the server tests below).

**Server.** `image_url` parts (base64 data: URLs, JPEG/PNG/GIF/BMP; http(s), file: and WebP refused) become the
template's image item; each `<|image_pad|>` is expanded to the image's rows, named in the prompt by ids below zero
derived from the image's hash, its budget and the row (`image_row_id`), so prefix reuse and snapshots tell images apart;
snapshot points never fall inside an image, a prefix never ends inside one, a prefill chunk never splits one (an image
needs `--chunk`/`--batch` >= its rows). Images are encoded on the request's thread with the scheduler held, the last six by
hash cached. `--mmproj GGUF` turns it on (checked: qwen3vl_merger, projection width = the model's), `--image-max-tokens`
(1,024 by default, 8-4,096, at most the chunk) is the encoder's budget, `--vision-reserve-mib` for a late encoder;
without `--mmproj` an image part gets a 400. The encoder is loaded before the engine, so its weights and scratch (267 MiB
at 1,024 rows) come out of the experts' share (N = 144 of 256 on a 32k-context two-slot test server). The module has
`mmproj` and `imageMaxTokens`. `--image-rows-dir` serves llama.cpp's own rows for known pictures (test source).

**Checks against llama.cpp** (`tools/ref/image_logits_compare.py`; text positions after the first image; llama.cpp's own
oracle with M-RoPE positions for the image chunks; 256-row budget): prompts `one` (1 image, 16 positions after it), `two`
(2 images, 27) and `turn` (an image in an earlier turn, 42). Token ids and image spans of eklow-prompt equal the oracle's
in all three (254, 401, 291 tokens).

| top-1 / median KL after the image | llama.cpp CPU vs CUDA (the band) | engine, CPU's rows | engine, eklow's encoder |
|---|---|---|---|
| one | 13/16, 1.0e-2 | 16/16, 8.5e-3 | 15/16, 1.0e-2 |
| two | 21/27, 5.9e-2 | 23/27, 5.2e-2 | 21/27, 6.5e-2 |
| turn | 38/42, 2.8e-2 | 34/42, 9.5e-3 | 36/42, 1.0e-2 |

Chunks of 300, 20 tokens decoded after the prompt, and verify blocks of 8 (accepting 3) after it stay in the same range
(turn 36/42 and 38/42, medians 7.8e-3 to 9.5e-3). The 4090 run needed the userspace driver of the loaded kernel module
(595.112.07; the 595.104.02 in the task text no longer matches). The CPU oracle against itself (a rerun with other
threading) lands in the same range, so these are the noise of this model on short texts.

**Server tests** (Ornith EKK8 + Inco's draft, 2 slots, 32k, through the HTTP API): descriptions of the NYT front page
(correct headline, date, Aldrin photo) and of a llama.cpp logo, a comparison of two images, a second turn that cites
the picture's colours, one text-only follow-up after an image turn, two image requests at once on the two slots, tall and
wide pictures told apart. Prefix reuse with images: a second turn of a 480-token prompt cached 461, a repeat 315 of 322
(prompts under snapshot_min = 256 tokens reuse nothing, as for text). Encoding 0.12-0.31 s at 120-800 rows.
Errors: http(s) 400, undecodable bytes 400, no `--mmproj` 400 (`image input is off`).

## 2026-11-03 — layer-major prefill: a long prompt's chunks run layer by layer, so each cold expert crosses the link once per group

`qwen/layermajor`. Raw output in `docs/bench/layermajor/` (`console-excerpts-lost-logs.txt` holds the runs whose files an
`rsync --delete` removed). Tokens from `~/eklow-qe/prof/prose.json`, stock EKK8, 2 x 262,144 with int8 KV.

**What it is.** `Config::max_group` (a multiple of `max_chunk`; `eklow-engine-bench --group N`, `eklow-server --group N`): a lone
non-verify prefill chunk of up to that many tokens is cut into near-equal sub-chunks of at most `max_chunk` and run layer by
layer (`Engine::step_group`). Each layer, every sub-chunk in turn does its mixer, router, hot experts and shared expert and
adds them into its rows of the residual stream in place (no tail kernel; the next layer's input norm is made at the start of
its mixer). The layer's cold experts then go once for the whole group (`Engine::cold_group`): the streamed set is chosen from the
group's counts with no slot cap, it passes through the 80 slots in waves (~144 experts a layer at G = 8, 2.1 waves), and for
each wave every sub-chunk's pairs of those experts run before the next wave's copy (started as soon as the wave's repack has read
the raw slots) lands. The CPU takes what is not streamed (~3,800 pairs a layer at 16k tokens, against ~80,000 streamed) on a
thread of its own. Kept per sub-chunk beyond the 2,048-token scratch: the residual stream in fp32, the MoE input in fp16 for the
waves, the routing (~25 MB per 2,048 tokens). The attention and DeltaNet kernels are unchanged; a sub-chunk's layer l runs after
the previous sub-chunk's, so the recurrent state and K/V are what plain chunks leave. New kernels: `qwenk::moe_accumulate`,
`qwenk::to_half`, an fp16-source `fast::moe_gather`, `xmx::repack_format_batch_into`.

**Two things found on the way**
- The streamed experts were repacked by a launch per expert and matrix (~225 us for 1.8 MB: 1,024 work-items, far less than
  the card). One launch per layer (plain path) or wave (group) took the repack from 1.29 s to 0.49 s of a 16k group, and is
  worth **+19%** on plain prefill, with no group involved: 2,048 from empty 1,333 -> 1,576 tok/s, 4,096 at depth 8k 1,247 ->
  1,456 (medians of 3). It ships with or without groups.
- Starting a layer's copies once the whole group is routed and enqueueing the hot experts after (`group_early_copy`) beats finishing
  each sub-chunk's layer first at G = 4 (1,878 / 1,742 / 1,663 against 1,637 / 1,499 / 1,466 at 8k / 16k / 32k, single runs) and
  ties it at G = 8; kept.

**Prefill** (`bench.sh engineq`, tok/s, medians of 3 alternating rounds; plain includes the batched repack; the depth rows time
4,096 tokens for plain and one group of 16,384 for G = 8):

| | 8k | 16k | 32k | depth 32k | depth 128k |
|---|---|---|---|---|---|
| plain | 1,486 | 1,438 | 1,292 | 1,028 | 582 |
| G = 8 | 1,805 | 1,786 | 1,577 | 1,103 | 727 |
| gain | +21% | +24% | +22% | +7% | +25% |

The plain path before the repack change was 1,253 / 1,215 / 1,151 / 981 / 497 (same method). Through `eklow-server` at the release
shape (README method, `serve_perf.py depth`, one cold request, Inco's draft, EmbeddingGemma 2): plain 1,216 / 1,284 / 784 / 570 tok/s at
8k / 32k / 128k / 252k (README before: 1,124 / 1,151 / 749 / 564), `--group 8` **1,605 / 1,855 / 997 / 708** (TTFT 5.1 s / 17.6 s /
131 s / 356 s; 252k was 442 s). The gain over batched plain is +32% / +44% / +27% / +24%, over the README rows +43% / +61% / +33% /
+26%. The server rows are single cold requests (the engine rows are medians of 3), so the 32k gap (+44% against +22%) is partly noise. Grouped runs move up to 25% between rounds while plain moves ~10%; it
follows the card (the host's wait for the group's routing is 218-245 ms a layer in the slow rounds and 154-173 in the fast, the CPU
task is 45 +- 2 ms and its wait after the waves under 10 ms), and plain was slow in the same round, so I read clocks, not
the host threads.

**Where the time goes** (profile, G = 8, 16k, after the repack fix, card time of 7.43 s): stream_gemm 1.59 s (21%), copy and repack
0.49 s (6.5%), attention 0.88 s, DeltaNet projections and kernel 1.4 s, router 0.53 s (the fp32 256-expert matmul: 6%), moe_group 0.40 s,
hot GEMMs 0.78 s. Copies are no longer the cost; the card's own GEMMs are. At 128k, attention is 35% of a group.

**Memory and decode.** Group buffers: 185 MB at G = 8, ~25 MB per 2,048 tokens more than plain. Experts a layer on the card at the
production flags (Ornith, draft, projector, EmbeddingGemma 2): 51 plain and at G = 2, 50 at G = 4, **48 at G = 8**; at the engine bench
69 / 68 / 67 / 66. Decode with the smaller placement, one slot, 0 / 8k / 32k / 128k: plain 41.1 / 41.1 / 40.0 / 32.3, G = 8 40.8 / 40.7 /
38.9 / 32.6 tok/s; verify blocks of 8 at 8k 70.9 against 68.4 ms. The loss is within noise (0-3%); through the server at the
release shape (57 experts a layer with `--group 8`, 59 plain) decode with the draft at 8k / 32k / 128k / 252k was 44.2 / 42.6 / 37.3 / 33.7 tok/s grouped against 41.5 / 44.1 / 38.7 / 30.0 plain: the acceptance moves both ways.

**Fidelity.** `checkq` ekk8 (top-1 of 256, median KL against the fp32 reference): plain chunks of 1,024 / 2,048 248 / 244, 1.9e-3 /
2.2e-3; groups of 2 sub-chunks 246, 1.8e-3; a prompt as one group of 3 sub-chunks 247, 2.1e-3 (decoding its last 64: 247, 2.0e-3);
oek8 216 / 208 plain, 216 and 215 grouped (medians 2.3e-2 to 2.6e-2 against 2.4e-2). A 23,477-token prompt, plain chunks of 2,048
against one 16,384-token group step, last 256 positions: top-1 agreement 1.000, mean KL 3.7e-4. `eklow-gdn-test`, `eklow-qfmt-test`,
`eklow-cpu-qgemv-test`, `eklow-spec-test`, `bench.sh attn` pass; `serve_test.py basic` and `image_test.py` pass against a server at the
production shape with `--group 8` (both `--group-busy` settings).

**The scheduler** (`--group N`, `--group-busy TOKENS`; off by default): a step whose only work is one slot's prefill with more than a chunk
left takes up to N chunks of prompt as a group; a group is the step's only chunk. While another slot generates, the group is at
most `group_busy` tokens (0: plain chunks, decode first) and a generating slot gets a step between two groups. A first version that
skipped decode on every group step starved the stream for the whole prefill (longest gap 5.8 s at 8k, 21.9 s at 32k). With the
alternation, a 1,102-token stream plus an 8k / 32k document arriving 4 s in (`tools/bench/serve_stall.py`): plain TTFT 8.2 s / 32.5 s,
the stream's longest gap 2.1 / 2.3 s; `--group-busy 4096` 5.9 s / 24.4 s and 4.0 / 4.9 s; `--group-busy 16384` 5.6 s / 18.5 s and
4.6 / 9.6 s. A group costs the generating slot its whole time; 4,096 buys 25-28% of the document's TTFT for a stall twice as long.

**Did not pay / left.** Starting the first wave's copy from the counts of the first sub-chunks (the routing wait is 70-600 ms a
layer and the first wave's copy begins after it) is not done; the copies are hidden anyway once the hot experts are enqueued behind them. The
profile's next costs are the card's GEMMs, the fp32 router (6%) and, deep, attention. Groups pay least where attention
dominates (the 4,096-token depth rows), and cost 3 experts a layer at G = 8. Whether to ship them (and with what `--group-busy`) is the owner's call.

## 2026-11-03 — Gemma 4 26B A4B back beside Qwen3.6: one `eklow-server`, either model

`gemma/restore` (off `main` 39aa4df; host-side work done on e-desktop, the card checks are listed below and not run yet).
The owner wants the 26B servable again next to the Qwen3.6 finetunes, switched on fiona by `llm-mode` (two units of one
package, as unholy's one engine serves DeepSeek-V4-Flash and GLM-5.3-Flash). The 26B's last state is `76d8588` (2026-10-27,
production with image input); the Qwen port had rewritten Gemma's files in place and retired the rest (`81b3154`/`f15d04b`,
`80fd39b`, `d8006ec`).

**Beside, never instead.** Every restored file is 76d8588's, under its own name, and nothing of Qwen's changed:
`engine/engine_gemma4.*` (`eklow::engine::gemma4::Engine`), `serve/scheduler_gemma4.*` and `serve/openai_gemma4.*`
(`eklow::serve::gemma4`), `serve/tool_grammar_gemma4.*`, `text/output_parser_gemma4.*`, `text/chat_gemma4.*`,
`model/gemma4.*` (config, weights, the fp32 reference forward), `kernels/gemma_fast.*` (the six kernels `fast.cpp` lost:
`embed`, `qkv_heads`, `residual_norm_quant`, `softcap`, `residual_norm_quant3`, `moe_tail`, in `eklow::fast`, whose names
nothing else uses), `data/gemma-4-26b-a4b-experts.json`. The types Qwen's port had pulled out of Gemma's headers are used
as they are (`text::ChatPart`/`ChatError`/`join` in `text/chat.hpp`, `text::ToolCall`/`OutputDelta` in `text/output.hpp`,
`model::DumpSink` in `model/dump.hpp`); the sampler, the BPE tokenizer, `embed_gemma2`, `vision_gemma4v`, `image/gemma4_image`,
attention, xmx, `cpu/q4_gemv` and `rt` are shared. Engine, scheduler and request layer are the three that collide with
Qwen's names, hence the `gemma4` inner namespaces; the rest has distinct names.

**One binary.** `eklow-server --model X.gguf ...`: `main()` in `tools/server/main.cpp` reads `general.architecture`
(`gguf::File::open` only parses the header) before any flag is parsed: `qwen35moe` runs today's server (its `main` is now
`qwen_server_main`, untouched), `gemma4` runs `gemma4_server_main` (76d8588's `main`, in `tools/server/gemma4_main.cpp`),
anything else is an error naming both. Without `--model` the Qwen server runs and says what it always said; a file whose
header cannot be read is an error before either starts. The Gemma side names the Qwen-only flags it refuses (`--draft`: "Gemma 4 has no DFlash draft";
`--draft-block`, `--chat-template`, `--snapshot-*`, `--group`, `--group-busy`, `--image-*`, `--vision-reserve-mib`). The
Gemma flag set is a subset of Qwen's, so the Qwen side has nothing of Gemma's to refuse.

**Tools** (own binaries, `-gemma4`, installed): `eklow-engine-run-gemma4`, `eklow-engine-bench-gemma4`,
`eklow-grammar-test-gemma4`, `eklow-chat-gemma4`, `eklow-parse-gemma4`, `eklow-prompt-gemma4`;
`tools/ref/gemma4_chat_compare.py` and `gemma4_parse_compare.py` call them. `serve_test.py` takes `--family auto|qwen|gemma`
(auto: "gemma" in `/v1/models`' id) and reports instead of failing the one check Gemma's grammar cannot meet (typed
tool arguments: its grammar fixes the structure and the name, not the schema's types); `image_test.py` ran unchanged.

**Host checks** (no card; the 26B's GGUF header from Hugging Face at d1c082be, vocab-only copy in `~/eklow-qwen-data/gemma4-header/`,
Google's tokenizer files and `chat_template.jinja` (18,683 bytes, the one `is_gemma4_template` hashes) in
`~/eklow-qwen-data/gemma4-hf/`; transformers 5.17.0, Jinja2 3.1.6):
- `eklow-grammar-test-gemma4`: 2,000 of 2,000 random well-formed calls accepted token by token; 1,997 random walks closed (3
  hit the length cap), 0 failed to parse or got stuck; 96 of 96 (state, special token) pairs banned exactly when refused.
- `gemma4_chat_compare.py --random 20000 --gguf`: 20,268 cases (264 hand-written, 4 both must reject, 20,000 random): 20,264
  identical text, 4 rejected by both, 0 differences; 3,184,441 tokens of 20,264 prompts, 0 differences. (76d8588 added 148
  cases from son-of-anton's real requests, files that are not in this tree.)
- `gemma4_parse_compare.py`, seeds 5 and 6, 20,000 turns each: all recovered exactly.
- The dispatch: both vocab-only GGUFs reach "no Level Zero GPUs" (the card is the next thing they need); an EmbeddingGemma 2
  GGUF gets "architecture gemma-embedding2; eklow serves qwen35moe ... and gemma4 ..."; `--draft` and `--group` on the Gemma GGUF
  are refused by name.

**Shared code that changed since 76d8588 on the 26B's paths** (the checks on the card judge them):
- `kernels/attention.cpp` (2b0b48b, be6e2b2, be4bcb0): the key segments for head 512 went from 2,048 to **4,096 keys**
  (`kSeg`), so the global layers' long-prefill probabilities buffer doubles (32,768 rows x 4,096 halves = 268 MB at the 2 x
  2,048-token chunks, was 134 MB; `attn::reserve` allocates it before the experts are placed, so the placement gives up ~134 MB
  of experts instead of failing) and the softmax is accumulated over different segments (fp32 rounding only); the decode
  split-K fold now folds by 8 at more than 8 splits instead of 32 (a different grouping of the same merge, dimension-fast
  reads), for sliding and global layers; the fp16 P.V tail mask is a scalar dword mask per row (same values, no spill; only
  `--kv-fp16`). `attn::reserve` gained a `group` parameter (default 0: the old test).
- `kernels/xmx.cpp`: `WeightFormat` has five more members and `Gemm16Kernel`/`geglu` take `silu` (defaults: Gemma's GeGLU, the
  same arithmetic as before; the old formats' instantiations carry an empty placeholder where the new formats keep their metadata). `matmul_geglu*` take `Glu act = Gelu`.
- `kernels/fast.cpp`: `moe_route` takes `ld` (0: n_expert) and a null `down_scale` (Gemma passes both as before);
  `matmul_f32_small` goes to 16 tokens (for 1-4 tokens the same code as before, instantiated for 4); an fp16 `moe_gather` overload.
- `serve/sampler.cpp`: the top-k pass skips ids that cannot enter (same set), greedy returns the first of the sorted candidates
  (same token); `ref.cpp`, `image_rows.hpp` (`nx`/`ny` default 0), `gguf.cpp` (a file with no tensors): additive.
- Unchanged: `cpu/q4_gemv`, `rt`, `embed_gemma2`, `vision_gemma4v*`, `image/gemma4_image`, `text/tokenizer`, `pyfmt`.

**Fixes since 76d8588 ported to the Gemma side**: the prefix divergence log line (655209e, in `admit`; image rows have no bytes,
so ids below zero are skipped); the two test fixes (66e29b3 broken image part, 655209e stop string from the model's own answer)
were already in `serve_test.py`. Left: the `developer` role as `system` (Gemma's template already reads `developer` as
`system` in first place and 76d8588 had that test), the image error wording for video and audio parts (Gemma's 400 for any
other part type stands), `general.sampling.*` from the GGUF (the 26B's file says temperature 1, top-k 64, top-p 0.95: what
`SamplingParams` already defaults to), per-request `speculative`, `usage.speculation`, snapshots, groups (Qwen only).

**Not run on the card.** The owner chose to try the 26B in production instead of a card window ("all them tests ain't
necessary, let me just try it"). What a window would run, should something look wrong (fiona `~/gemma-window.sh` does it all,
detached, from `~/eklow-wt-gemma`): `bench.sh check26` with and without the profile (band: long >= 220/256 and mean KL <=
7.5e-2; chat >= 22/25 and <= 8e-2; 76d8588: 243/256 at 2.9e-2 with N = 51, 242/256 at 2.8e-2 with every expert on the card),
`bench.sh attn` and `kernels`, `serve_test.py basic` and `image_test.py` against the Gemma server at the production shape and
`basic` against the Qwen server, `serve_perf.py` at 8k and 32k against `docs/bench/serve-perf-26b-chunk2048-fiona-2026-10-27.txt`.

**Serving both.** The NixOS module serves named instances (`services.eklow.instances.<name>`, unit `eklow-<name>.service`,
each conflicting with the others; the embedding model shared on one port), and the house's `llm-mode` switches fiona between
`ornith` (8081) and `gemma` (8083), as it switches son-of-anton between unholy's instances.

## 2026-11-03 — speculative decoding for the 26B: verify blocks and capture in the Gemma engine, the draft path in its server

**Built and checked on e-desktop only** (it compiles, the host checks pass; nothing of it has run on a card, which production
holds). The card commands, and what each should show, are at the end.

**The engine** (`engine_gemma4.*`, no kernel changed). `Config::capture_layers` / `max_capture_rows` and `Chunk::capture_rows`
are Qwen's: after layer l's tail the residual stream (`l_out`, the buffer `fast::moe_tail` or `residual_norm_quant` just wrote)
of the rows a chunk asks for (its last `capture_rows`) is gathered by `qwenk::gather_rows` into a device buffer
[row][layers x 2816] fp32 in step order, `Engine::capture()` with the rows' slot and position. Layer numbers are the GGUF's
`target_layers` as Qwen's engine takes them (`Dflash2Config::target_layers`, passed through unchanged). A verify chunk
(`Chunk::verify`, 2 to 16 tokens) is a chunk with logits for every row; Gemma has no recurrent state, so there is no pending
state: the caller rewinds with `truncate(slot, start + 1 + accepted)` after the step (the Qwen engine's `accept` field has no
use here). Why the rewind is safe: a sliding ring holds `ring_pages x kPage` positions, window + max_chunk or more; a block of
B tokens writes position p + j over p + j - ring, which a query at p + a + 1 (a accepted) has left behind the window if
ring - window >= B - a - 2, and `ring_slack()` is >= max_chunk; `step` refuses a verify chunk longer than slack + 1 (it only
matters for a max_chunk under 15). The global layers' pages past the rewound length are freed and taken again by the next
chunk; their stale K/V is overwritten before it is read (a causal read never goes past the chunk).

*Which path a verify block takes.* No new one: `step` picks by rows. T x 8 pairs <= 32 (`kFp16Rows`), that is a block of up to 4
tokens of one slot, takes the fused decode path (grouped GEMVs, routing and the weighted sum on the card, the cold experts
overlapped by the host); more rows (two slots' blocks of 4, blocks of 8 or 16) the prefill path (sort by expert, 64-row
tiles, the cold pairs on the CPU: streaming needs 16 pairs for one expert, which 8 to 32 rows rarely give). The pool spins as at
decode for a verify step. Cost I expect: a block of 4 is a decode step plus three rows of the cold experts' CPU time (the
Qwen 35B measured 8.5 ms a row beyond the first; the 26B has 8 of 128 experts per token and N = 51 on the card, so its share
that is cold is lower, I would guess 2 to 5 ms), the attention and dense matmuls of 4 rows costing little more than 1; two
slots' blocks (8 rows) leave the fused path, which may cost more than two fused steps. `eklow-engine-bench-gemma4 --verify
1,2,4,8,16` measures exactly this (per row, per row beyond the first, with the cold experts' time) at 8k and 32k with 1 or 2
slots; if 8 rows cost more than 2 x the 4-row step, the cheaper path is to run each slot's block in its own fused pass (the MoE
section looping over groups of 4 tokens); not done, since it cannot be judged without the card.

*The head.* The draft needs the target's head as an `xmx::Weight`; Gemma's was `fast::Q6` (fp32 activations). With
`Config::head_dpas` (off by default, so plain serving is exactly as before; `--draft` turns it on) the engine's head is
`xmx::WeightFormat::Q6_K`, one copy (605 MB) shared with the draft through `Engine::head_weight()`; the final norm output is
quantized to int8 per 32 and the logits come from `xmx::matmul` as Qwen's head does. The embedding lookup then reads the
GGUF's mapped rows (a gather of T x 2,310 bytes per step into pinned memory, `ref::embed` on the card), so the table is not
on the card either, and the draft gets the table as host memory (`Engine::embedding_table()`, `Dflash2Target::host_table`). What
this costs in accuracy (int8 activations on the head, where the fp32 head was kept for the 26B's softcapped logits) and in speed
is for `check26` with `--head-dpas` and the decode bench to say; if the accuracy is outside the band, the fallback is the
draft carrying its own copy (605 MB, ~6 experts a layer).

*The draft's card memory.* `Config::draft_reserve_bytes` is kept from the experts' placement as in Qwen's engine; the server's
`--draft-reserve-mib` (default 1,200: the draft's 5 layers at hidden 2816 and its cache, to be read off the startup line and set
tighter) and the capture buffer (min(chunk, draft window) + slots x 16 rows x 6 x 2816 fp32) are counted before the placement.

**The server** (`scheduler_gemma4.*`, `openai_gemma4.*`, `gemma4_main.cpp`) ports Qwen's speculative path: the draft step (one
pass per block size, 32 rows at most), the verify chunks, `serve::verify` with the sampler's every feature and a clone of the
output parser and tool grammar per row (`RowState`), `on_token` returning why a request ends so that a stop string, an end token
or `max_tokens` in an accepted block ends it at that token (the block's tail is dropped), the rewind and the draft's context
injection of the accepted rows, per-request `"speculative"`, `usage.speculation`, the log's draft fields and per-step timings,
`--draft`, `--draft-block` (default min(trained block, 4)), `--draft-reserve-mib`; the draft's context follows prefix reuse
(`truncate` at admission, at the end of a request, `reset` after a failed step). Not ported, because Gemma has none of it:
snapshots, the pending-accept protocol, layer-major groups, rotary deltas of images. Differences: the draft context takes an
image's rows like any others (their hidden states are real); the block's rejected rows are cut by `Engine::truncate` in the step
that verified them, not by the next chunk.

**Needs `gemma/draft`.** The target's embedding for the Gemma draft is scaled by sqrt(2816) and its head's logits are softcapped
(30), neither of which `model::Dflash2Target` on main says; `gemma4_main.cpp` builds the target as Qwen's (`{table, head, true}`)
and must set the two new fields when that branch lands. Until it does, a Gemma draft run would embed and score wrongly.

**Host checks** (e-desktop): the tree and `nix build path:.` build; `eklow-grammar-test-gemma4` (2,000 calls, 1,997 walks closed,
96/96 bans), `eklow-grammar-test-qwen` (2,000/2,000, 828/828), `gemma4_chat_compare.py` (368 cases, 0 differences, 33,969 tokens),
`gemma4_parse_compare.py` (100/100), `eklow-spec-test` all ok.

**Card commands, in order** (fiona, in a window, from `~/eklow-wt-gspec` after building there; GGUFs and the profile as in
`~/gemma-window.sh`): (1) `bash ~/eklow-perf/bench.sh check26` with and without `--expert-profile data/gemma-4-26b-a4b-experts.json
--experts-on-card 51`, unchanged code path (band: long >= 220/256 at mean KL <= 7.5e-2; chat >= 22/25 at <= 8e-2; 76d8588: 243/256 at
2.9e-2); then the same with the engine-run call inside check26 given `--head-dpas` (the shared head's accuracy; the harness script is on fiona and I could not see whether it passes extra flags through). (2) `eklow-engine-run-gemma4 --model <26B> --out <dir>
--tokens-json <a 2-3k prompt> --chunk 1024 --verify-last 64 --verify-block 4 --accept 1,2,4,3 --capture-layers 1,6,11,17,22,27
--head-dpas [--expert-profile ... --experts-on-card 51]`, and again with `--verify-block 16 --accept 5,16,2` and `--verify-block 8`
(the rewinds and the 8- to 16-row prefill-shaped path): expect mean KL under 0.05, top-1 agreement over 95%, captures identical
for the prefill and relative RMS under 0.05 for verify rows. (3) `eklow-engine-bench-gemma4 --model <26B> --prefill 1024 --decode-ctx
8192 --verify 1,2,3,4,6,8,16 --verify-ctx 8192,32768 --slots 1 [--capture --head-dpas]` and `--slots 2`, with the profile and N = 51
(the per-row cost; compare B = 1 with the plain decode step, and with `--head-dpas` off for the head's speed). (4) `serve_test.py
basic` against the Gemma server without `--draft` (unchanged behaviour) and with `--draft <gemma draft> --draft-block 4`
(after `gemma/draft`). (5) the lossless check through the server: `serve_test.py sample --n 3000 --out A.json` against the server
without the draft and with it, then `serve_test.py chi2 A.json B.json` (Qwen's was 3,000 completions each); `basic --save` on
both and `compare`.

## 2026-11-03 — z-lab's DFlash draft for the 26B: llama.cpp's acceptance, the reference's, and the draft in eklow's code

TASK gemma/draft (branch `gemma/draft`; the speculative path itself is `gemma/spec`'s). The owner wants speculative decoding for
the 26B; the plan is z-lab's published `gemma-4-26B-A4B-it-DFlash` first, no training, and its acceptance at the short blocks this
card favours before deciding on a draft of our own. The card was not touched: everything ran on e-desktop (the 4090 for
llama.cpp, the CPU for the rest).

**Files** (`~/eklow-qwen-data/gemma/`, `MANIFEST.txt` has the sha256s). Draft: `z-lab/gemma-4-26B-A4B-it-DFlash` @ 77d42027
(plain DFlash, arXiv 2602.06036, not DFlash 2: 5 layers, 4 sliding (2,048) then 1 full, hidden 2,816, 32 heads / 8 KV of 128, ff
5,632 silu, block 16, 0.43 B parameters, BF16). Target: `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` @ d1c082be, sha256 as pinned.
`~/eklow-qwen-data/gemma4-hf/config.json` is byte-identical to google/gemma-4-26B-A4B-it's (main @ 4d7ae498), not only the
tokenizer. `tools/weights/make-draft.sh gemma` converts with the pinned converter (BF16 and Q8_0, 875 / 472 MB). The GGUF's
metadata against config.json: `dflash.target_layers` [2, 7, 12, 18, 23, 28] (each id + 1, as for Qwen), `block_size` 16,
`attention.sliding_window` 2,048, `sliding_window_pattern` [T, T, T, T, F], `embedding_scale` 53.066 (= sqrt(2,816)),
`final_logit_softcapping` 30, `tokenizer.ggml.mask_token_id` 4, rope base 1e6, eps 1e-6; no `attention.causal` key, no convolution
or selector keys, no embedding or output tensor (the target's, tied: the 26B has no `output.weight`, its `token_embd` is Q6_K).

**What z-lab's code does that the GGUF cannot say.** model.py makes a layer causal exactly when it is sliding
(`is_causal = layer_type == "sliding_attention"` when the config has no `is_causal`), and vLLM's Gemma 4 DFlash PR (#41703) does
the same (`causal_attn_metadata` for the sliding layers): the four sliding layers see the block's earlier positions only, the full
layer sees all of it, the draft embeds with the target's table times sqrt(hidden) and applies the final softcap. llama.cpp at the
pin has one flag for the whole model, unset here, so it runs all five layers non-causal. Its draft is also deterministic: it takes
the top candidate of each position and the target's own sampler decides (`common_sampler_sample_and_accept_n`; the rejection variant is for drafters that fill a q, this one does not), so at
temperature > 0 a drafted token is kept with probability p(d).

**Acceptance, llama.cpp** (pin ea0c4c2c CUDA on the 4090; 26B Q4_0 QAT target, draft Q8_0 with `dflash.block_size` rewritten for 8
and 4 as the Qwen matrix did; z-lab's GSM8K / HumanEval / MT-Bench prompt sets via `dflash2_accept.py`, 16 requests each, thinking
on, 4,096 new tokens; tokens per verification step, `(n - 1) / steps`; `tools/bench/dflash-gemma-matrix.sh`; raw in
`docs/bench/gemma-dflash/`). **One request at a time** (these are the numbers to use):

| sampling | block | GSM8K | HumanEval | MT-Bench |
|---|---:|---:|---:|---:|
| greedy | 16 | 6.18 | 6.38 | 4.93 |
| greedy | 8 | 5.16 | 5.07 | 4.08 |
| greedy | 4 | 3.42 | 3.42 | 2.98 |
| T=1, top-k 64, top-p 0.95 | 16 | 5.28 | 5.18 | 4.55 |
| T=1, top-k 64, top-p 0.95 | 8 | 4.94 | 4.43 | 3.85 |
| T=1, top-k 64, top-p 0.95 | 4 | 3.24 | 3.17 | 2.91 |

Four requests at a time (what Qwen's matrix did) gave much less, 5.20 / 4.46 / 2.86 at block 16 greedy and 4.06 / 3.47 / 2.63 at
temperature 1: llama.cpp's multi-slot DFlash loses a quarter to a third of the acceptance (single requests fall to ~1.0 outright;
the mechanism is not looked into). Those are the `gemma26b-zlab-b*` files without `-conc1`; the Qwen matrix's numbers may carry
the same loss. z-lab's card (vLLM, bf16 target on a B300, greedy, block 16): 7.71 GSM8K, 7.80 HumanEval, 4.33 MT-Bench. Decode
tok/s in the files is for the 4090 with this target fully on the card (330 median for GSM8K, one request, block 16); there is no
no-draft run to divide by.

**Acceptance, the reference** (z-lab's model.py in fp32 on the CPU, `tools/dflash_test/accept_torch_plain.py`; teacher forced on
the 26B's own greedy answers to one GSM8K, one HumanEval and one prose prompt, 360 positions at stride 5, the target's real
hidden states from `eklow-oracle` on the same GGUF, answers of 400 to 700 tokens, so short context): block 16: GSM8K 7.76,
HumanEval 7.01, prose 3.28; block 8: 5.99 / 5.19 / 3.24; block 4: 3.60 / 3.39 / 2.64 (the same with every layer non-causal, as
llama.cpp runs them: 7.64 / 6.78 / 3.15 at 16, no difference at 8 and 4). So the draft does what z-lab's card says on this QAT
target, llama.cpp with one request at a time gets 80 to 90% of the reference on code and math (more on MT-Bench-like prose), the
non-causal sliding layers cost 2 to 4% at block 16, and what eklow's forward must reproduce is model.py's, which the code now does
(below). Not measured: the serving forward's int8 activations (the card check), more than three prompts.

**The full layer at depth.** The one full-attention layer sees the slot's whole context; eklow's draft cache for the other
layers is a 2,048 ring. llama.cpp can cap only by making every layer sliding at 2,048 (the pattern all true; `dflash-gemma-depth.sh`):
prompts of about 7,000 and 24,000 tokens, greedy, thinking, 1,024 tokens, 8 each, block 16: acceptance 2.60 -> 2.53 at 7k and 2.77
-> 2.68 at 24k (-3%). In model.py with the full layer's mask limited to the last W positions (`capW`; the 6.3k-token prompts of
`accept_torch_plain.py`, 142 positions): 3.69 for the whole context, 3.48 at W = 2,048 (paired -0.21, se 0.10), 3.39 at W = 512
(-0.30). The layer is worth 3 to 6% at these depths in prose analysis and quoting; deeper than 24k is not measured. Memory at the
production shape (2 slots), per full-attention layer, the ring including a block and a page of slack: fp16 K/V 2.00 GiB at 262,144
positions, 1.00 at 131,072, 0.50 at 65,536, 0.25 at 32,768; int8 (the engine's K/V format: a scale per position and 16 dims)
1.13 / 0.57 / 0.28 / 0.14 GiB; the four sliding layers 80 MiB in fp16; the weights 0.44 GiB and the scratch ~0.15 GiB (computed from
the allocations, not measured on the card). **Proposal: the full layer as an int8 ring of 32,768 positions per slot** (0.14 GiB,
the default of `Options::full_context`), which keeps what the measurements say it is worth for contexts into the tens of thousands;
262,144 in int8 (1.13 GiB, the whole context) only if a long-context acceptance measurement on the card says so, and it also needs
the target's features for every prompt row (67.6 KB a row in fp32, 150 MFLOP a row through `fc` and the K/V projections), where
the capped ring needs the last `reach()` rows.

**The draft in eklow's code.** `Dflash2Config::from_gguf`, `Dflash2Weights`, `Dflash2Ref` and `Dflash2` take plain DFlash drafts
(`Dflash2Config::plain`): no convolutions, no selector; per-layer sliding/full attention with their own key ranges (sliding layers
causal in the block, the full layer not); the target's embedding times `embed_scale`; the head's logits, top 16 per position by
`dflash::topk16`, then the softcap on the 16 on the host (monotone, so the same 16); a tied head (`token_embd.weight` when the
target has no `output.weight`); the vocabulary from the tokenizer's length, checked against the target's table. The lattice has the
same row of scores for every predecessor (`plain_lattice`), so `serve::walk` and `serve::verify` are unchanged: a position is
sampled from the softmax at the request's temperature over its 16 candidates, `q` is that distribution, acceptance is the lossless
rule of `spec.hpp` for every sampler feature (penalties, bans, grammar enter p only). One thing the spec worker should try: at
temperature > 0 llama.cpp's deterministic draft keeps p(d) per token; sampling from q keeps sum min(p, q), which is lower when the
draft is sharper than the target; `draft(..., temperature)` with a small value (1e-3: q is one-hot, `verify` then accepts with
probability p(d)) is the same lossless rule with llama.cpp's draft, and the acceptance of each can be read on the card. Never
temperature 0 with a sampling request: a greedy block carries no q. The full layer's cache is a ring of `Options::full_context`
positions (default 32,768), int8 (`full_int8`) or fp16; a context beyond it is seen through its last positions (a capped window).
`Dflash2::reach()` is the rows a caller must capture for the draft: that, not `window()`. `Options::sliding_causal` (-1 the draft's
own) runs the sliding layers non-causally as llama.cpp does, for an A/B on the card. `attn::attend` takes an int8 cache for
128-wide heads (it was fp16 only; the instantiation compiles for dg2). The public API of `model::Dflash2` is unchanged (additive
fields and methods only). The DFlash 2 path is untouched in arithmetic: the reference forward on the host (SYCL native_cpu,
`tools/dflash_test/host-build.sh`, 4 sources, no card) gives the same bytes before and after on Inco's draft (a random Q8_0 target
for its embedding and head; two configurations, sha256 `2af90f70...` and `c273c410...`); the serving forward makes the same
calls in the same order for a draft without a full layer (`r0 = 0`, one tile list, one cache shape), checked by reading and by the
card check below, not run.

**Checked against z-lab's model.py** (`tools/dflash_test/ref_torch_plain.py`, fp32 PyTorch on the CPU, the draft's BF16 weights, fixed
random features, the 26B's Q6_K embedding dequantized as embedding and head): `Dflash2Ref` (BF16 GGUF) on the same inputs: final
hidden rows 7e-6 relative (ctx 60, 100 with window 32 so that the sliding mask cuts, and 2,500 with the draft's own 2,048: 7e-5),
softcapped top-16 scores 1e-4 absolute, greedy tokens equal in every block, 2 of 720 candidates swapped at a near tie; with the
window ignored the same comparison is 0.83 relative (the test sees the masks). The card side compares the serving forward with the
reference: `eklow-dflash-test plain`.

**Card commands** (fiona, `~/eklow-wt-gdraft`; nothing was run there). Build: `nix develop path:. -c bash -c 'cmake -S . -B build -G
Ninja -DCMAKE_CXX_COMPILER=clang++ && cmake --build build --target eklow-dflash-test eklow-server'`. Inputs from e-desktop's
`~/eklow-qwen-data/gemma/` (the two GGUFs under `work/gguf/drafts/`, `target/`, and `host/`):
1. Qwen identical, serving forward: build `main` (95d2912) and this branch, then with each
   `eklow-dflash-test dump --draft Qwen3.6-35B-A3B-DFlash2-inco-Q8_0.gguf --target <extract_head.py's GGUF of the Qwen target> --ctx 200 --block 8`
   and `diff` the two outputs (hashes of the lattices).
2. Plain draft, serving against reference, random features: `eklow-dflash-test plain --draft gemma-4-26B-A4B-it-DFlash-zlab-Q8_0.gguf
   --target gemma-4-26B_q4_0-it.gguf --feats host/t3/feats.f32 --anchors 106,5555,200000 --ctx 2500 --block 16` (the draft's own window,
   so the ring wraps), and `... --feats host/t2/feats.f32 --anchors 2,1000,77777 --ctx 100 --window 32`.
3. The same on the target's real features: `--feats host/real-gsm8k-ctx300.f32 --anchors 236812,236743,578 --ctx 300`, and
   `--feats host/real-deep-analysis-ctx4500.f32 --anchors 669,26886,528 --ctx 4500 --full-context 1024` (the capped ring against the
   whole one; `--no-int8` for fp16). Expect the three pairs of `fast` (serving vs exact within int8-activation noise, as for Qwen),
   and every greedy token equal at the capped/whole pair only where the cap does not matter.
4. `eklow-dflash-test dump` on the Gemma draft before and after any further change to the full layer's cache; and `Options::sliding_causal`
   0 against -1 on real acceptance.

## 2026-11-03 — z-lab's draft on the card: correct, lossless, and not worth its memory here

`gemma/dflash` (the merge of `gemma/spec` and `gemma/draft`), measured on fiona 19:28-20:31 with production paused
(`fiona:~/gdflash-window.sh`, log `~/eklow-perf/log/gdflash-window*`). The 26B at the production shape unless said: 2 x
262,144, int8 K/V, both image encoders, EmbeddingGemma 2; z-lab's draft in Q8_0, blocks of 4.

**Correct.**
- Qwen's draft is unchanged: `eklow-dflash-test fast` on Inco's draft (fiona's saved inputs) prints the same bytes from
  main's build and this one.
- The head on the matrix units (`--head-dpas`, what `--draft` turns on) keeps the 26B in its band: `check26` at 51 experts
  a layer gives long 248/256 at mean KL 1.92e-2, chat 22/25 at 5.13e-2. On 2026-10-27: 243/256 at 2.9e-2, chat 23/25.
- Verify blocks against the same tokens decoded one at a time (`eklow-engine-run-gemma4 --verify-last 64`):
  - blocks of 4: top-1 90/100, mean KL 0.064 (worst 1.88);
  - blocks of 8: 107/117, 0.034;
  - blocks of 16: 115/125, 0.037.
  - Captured rows drift from 0.4% (layer 1) to 9-14% (layer 27) relative RMS.
  This is the 26B's routing magnifying a step's shape, not a wrong computation: the acceptance and the lossless test
  through the server match the reference. It was not compared with the fp32 reference directly (the tool compares
  verify with decode).
- Acceptance through the server (`serve_perf.py accept`, z-lab's prompts, 8 each, two at a time, temperature 1), tokens a
  verify against llama.cpp's one-request numbers:

  | block | GSM8K | HumanEval | MT-Bench | llama.cpp |
  |---|---:|---:|---:|---|
  | 4 | 3.26 | 3.14 | 2.90 | 3.24 / 3.17 / 2.91 |
  | 8 | 4.74 | 4.33 | 3.50 | 4.94 / 4.43 / 3.85 |

- Lossless: 1,500 sampled completions with the draft against 1,500 without, chi-square 6.3 on 10 degrees of freedom,
  p = 0.787.
- `serve_test.py basic` passes with and without the draft, except "same seed, same output" in both. That test is new since
  the 26B last served. The second request reuses the first's cached prompt, so its first step has another shape, and at
  temperature 1 a near-tie flips; Qwen passes it.

**Not worth it on this card.**
- The draft takes the 26B from 42 experts a layer on the card to 27:
  - the draft is 788 MiB of the 1,200 MiB reserved for it;
  - the capture buffer (2,048 rows x 6 layers, 138 MB) and the head's move to the matrix units take the rest;
  - cold experts pinned: 9.44 GiB instead of 8.04.
- Without the draft, 42 is 2 fewer than on 2026-11-02, the cost of the 4,096-key attention segments inherited from Qwen's
  port.
- Through the server, one stream, Pile text (`serve_perf.py depth`, median of 3; tokens a verify in brackets):

  | | 8k | 32k |
  |---|---|---|
  | decode, plain | 34.9 tok/s | 32.4 tok/s |
  | decode, with the draft | 33.9 (2.37) | 30.3 (2.26) |
  | prefill, plain | 1,593 tok/s (5.1 s) | 1,612 (20.3 s) |
  | prefill, with the draft | 966 (8.5 s) | 1,040 (31.5 s) |

- Two streams at block 4 decode 21.9-23.7 tok/s each on the acceptance prompts (block 8: 18.5-23.8); the 26B without a
  draft did ~20 each on GSM8K (2026-10-27, 46 experts). Each step of two such streams:
  - engine 112-127 ms, against 74 ms in `eklow-engine-bench-gemma4` at 51 experts;
  - the draft 4.6 ms;
  - host sampling 10-16 ms.
- What a verify costs at 51 experts (`eklow-engine-bench-gemma4 --verify`):
  - one slot at 8k: decode 23.3 ms; 2 / 3 / 4 / 6 / 8 / 16 rows 29.2 / 36.5 / 41.9 / 65.0 / 73.5 / 104.0 ms, 6.5-8.6 ms a row
    beyond the first;
  - one slot at 32k: decode 26.0 ms, 4 rows 45.8;
  - two slots at 8k: decode 32.2 ms, 4 rows each 74.0, 8 each 105.3.

So the draft's acceptance is what llama.cpp gets and its output is exact, but its card memory costs the 26B more than the
draft saves. It is a wash on decode at an acceptance of ~2.3 (prose), and prefill loses ~40%. The owner (2026-11-03):
drop z-lab's draft, and "take the sliding window draft as the win". What stays is the 26B's verify, capture and speculative
path, and plain DFlash with per-layer sliding/full attention in the draft code, all off without `--draft`. A draft of our
own would be shaped for this card: every layer sliding, small (z-lab's full layer is worth 3-6% at <= 24k, its 0.44 GB of
weights about five experts a layer), and trained for blocks of 4. The reserve should be fitted to what the draft takes
(1,200 MiB kept, 788 used: four experts a layer).

Not run: the draft's serving forward against its reference (`eklow-dflash-test plain` stopped at once: `OwnedTarget` wanted an
`output.weight`, fixed in e950282, a tied head being the table) and the sliding layers' causal A/B (skipped). The Gemma
server does not exit on SIGTERM while a request runs or while it loads. systemd killed production after 90 s when the
window began. A test server still loading at the end held the pinned memory, so production's first start failed ("out of
pinned host memory") and its restart 6 s later came up clean at 42 experts.

## 2026-11-07 — a keep-alive on the streams, so a long tool call does not look like a dead connection

fiona's eklow logged a request "cancelled" at 10:26:47: 125k prompt, 120k cached, 17,911 tokens of completion at 29.3 tok/s over ~10.3 minutes, then the client hung up (1 of 65 requests in 24 hours; no restart, no kernel error). The server wrote an SSE chunk only when a token added text, and a tool call goes out whole once it closes (`ChatResponse::chunks`), so a long call, a file through `write_file` say, sent nothing from its first token to its last. son-of-anton abandons a stream after 10 minutes without a byte (`provider.go`, `IdleTimeout`), maitred after its idle timeout (300 s by default). A cold long prefill was silent the same way (~6 minutes at 252k). Strong inference, not proven: the client's log is not on this account, and eklow does not log what it generated.
The fix: `tools/server/keepalive.hpp`, both servers. While a stream has sent nothing for `--keepalive` seconds (default 15, 0 off), it writes an SSE comment, `: keep-alive`. Every client skips such lines (son-of-anton keeps only `data: ` lines), and any byte resets their timers (son-of-anton's `idleReader`, maitred's `relay`). Streaming a tool call's arguments as they are generated, as OpenAI does, would also do it, but touches the parser and the grammar.
Checked on the host (`eklow-keepalive-test`): the timer's rules, and a real HTTP stream on loopback with 3.5 s of events that add no text. The longest gap between bytes was 1.00 s with a 1 s keep-alive and 3.60 s without; a data-only parser gets the same payloads both ways. `eklow-server` builds. Not run on the card: `~/eklow-qwen-data/keepalive/card-check.sh` (a test server with `--keepalive 2`, a cold ~40k-token prompt, `serve_test.py basic`) is ready for a window.

## 2026-11-07 — Sage-A1.5's chat template, a third next to stock and Ornith

The owner is putting InternScience/Sage-A1.5 (a Qwen3.6-35B-A3B finetune, HF revision `0e3205cf`; config and tokenizer are stock's) in Ornith's place on fiona. eklow renders chat templates itself and refuses a GGUF whose template it does not know, so `QwenTemplate::Sage` joins Stock and Ornith. Sage's `chat_template.jinja` (9,025 bytes, no final newline; InternScience's GGUFs embed exactly these bytes, checked on the Q4_K_M) is stock's plus a `verification` switch (the `diff` is the three hunks the task named: the block after the macros, and one line in each system branch). `qwen36_template_of` recognises it by size and FNV-1a (also the 9,026-byte `tokenizer_config.json` copy with a trailing newline); `eklow-chat check` on the Q4_K_M says "sage".
Semantics, as Jinja has them: `verification` is on when defined and truthy (true, a non-empty string, a non-zero number, a non-empty list or object; null, "", 0, [] and {} are off). On, a string other than true/1/yes/on (any case) raises ("false" is on, and then raises); a missing first system message, or one blank after trimming, raises; otherwise the trimmed system content gets "\n\nVerification protocol: enabled" unless it already ends with that text, in both system branches. The server maps the raises to a 400 as for every template error. `QwenChatOptions::verification` is an optional JSON value (nullopt is Undefined), so the string and number cases render and raise as Jinja would. The request reads `verification` at the top level and in `chat_template_kwargs` (the latter wins), for stock and Ornith it is ignored as their templates ignore it. `--chat-template stock|ornith|sage`; `--verification on|off` (default off) is the value for a request that sends none, a request may still send false, and `on` with another template is refused at startup; the startup line names the choice.
Compared against Hugging Face's `apply_chat_template` (transformers, `tools/ref/qwen_chat_compare.py`, 8,017 hand-written cases, of them 2,840 new: 27 conversations x 26 verification values (absent, true, false, "true", "True", "YES", "On", "1", "false", "maybe", "", "0", "no", "off", " true", 1, 0, 2, -1, 0.0, 1.5, null, [], [0], {}, {"a": 1}) x with and without tools x preserve_thinking, over plain, padded, blank, None, missing, late, part-list, image and already-triggered system messages and reasoning turns, plus the other options with the switch on; and 20,000 random conversations, half carrying a random verification value). Sage: 28,017 cases, 14,076 identical, 13,941 rejected by both, 0 differences. The same cases on the stock template (18,257 identical, 9,760 rejected by both) and on Ornith's (same counts), 0 differences each, so the old comparisons and the ignoring of `verification` hold. With tokens (3,000 random, stock's vocabulary GGUF): 6,066 prompts, 1,503,451 tokens, 0 differences. Nothing is deployed and nothing ran on the card.
Found on the way: InternScience's Q4_K_M has no `tokenizer.ggml.normalizer.nfc` key, so `QwenTokenizer::from_gguf` (the server's loader too) refuses it ("does not say it normalizes to NFC"). A Sage GGUF made for fiona (the EKK8 script quantizes from InternScience's Q8_0) needs that key set to true, as the stock and Ornith files have it; not changed here.
Addendum, same day (found on fiona's card): `--verification on` made every request without a system message a 400 ("verification=true requires a non-empty system message"), so serve_perf.py's depth and gsm runs, which send none, failed. A server default must not reject requests the client never asked to verify. The rule now: the default applies only when the conversation's first message is a system message whose content, rendered and trimmed, is non-empty (`text::qwen36_has_system_text`, checked in `parse_chat` where the default is filled in); for any other conversation it is as if absent (rendered like stock, no error). A value the request sends itself, top level or in `chat_template_kwargs`, keeps the template's exact semantics, both raises included, so a request's `true` without system text is still a 400. `render_qwen36` is unchanged and stays byte-for-byte Jinja. Host test `eklow-verification-test VOCAB_GGUF` (tools/prompt/verification_test.cpp; there was no harness for `parse_chat` before), 18 checks, 0 failures: default on with system text (trigger), with none and with a blank one (as without), with the request sending false (top level and kwargs; off), a request's true without system text (400, default on and off), its string "false" (400). The Hugging Face comparison is unchanged: sage 28,017 cases, 14,076 identical, 13,941 rejected by both, 0 differences; stock and Ornith 18,257 / 9,760 / 0.

## 2026-11-07 — Sage-A1.5 on eklow: the behaviour test, the file, the draft, the speed

The owner, before more work on Flash-Next: try dropping in Sage-A1.5, a Qwen3.6-35B-A3B finetune meant for orchestration, in Ornith's
place on the engine as it is. This entry is the case for it, measured; the template is the entry above. Raw output:
`docs/bench/sage-2026-11-07/`.

**Behaviour, on fiona's own work.** `replay_fiona.py` (outside the repo) replays one real session of fiona's at three decision points:
the first move, and two turns after the owner corrected how work is handed to a worker. Each request is the chat as it stood (stored
reasoning dropped or replayed), 3 samples each, 3 passes, so 18 requests a point. A move is right if it is the one the owner wanted.
Most runs were on the 4090 under one llama.cpp server (0.8.0-dev, CUDA), Ornith being production's own file. Counts are right
moves out of 18 at the three points:

| model | first move | point 2 | point 3 | ran out at 8,000 tokens (point 2) | median tokens, points 2 / 3 |
|---|--:|--:|--:|--:|--:|
| Ornith-1.5 EKK8 | 18 | 2 | 6 | 6 | 7,508 / 4,214 |
| Sage Q4_K_M (InternScience) | 8 | 4 | 8 | 0 | 2,506 / 2,094 |
| the same, verification on | 4 | 2 | 14 | 0 | 1,977 / 2,636 |
| Sage Q8_0, verification on | 7 | 2 | 8 | 0 | 1,741 / 2,108 |
| **Sage EKK8 in eklow on fiona, verification on** | 6 | 2 | 9 | 0 | 1,298 / 2,187 |

Fisher exact, two-sided:
- **Sage never thought itself into the cap** (0/72 against Ornith's 6/18, p = 1.2e-4 on the 54 llama.cpp runs) and writes about a
  third of the tokens.
- **Ornith's first move is better** (18/18 against 19/54, p = 5e-7).
- **At point 3 Sage leads but not significantly**: 30/54 against 6/18, p = 0.17. Q4_K_M's 14/18 against Q8_0's 8/18 (p = 0.09) is
  noise, and so is any step-3 gain from verification.
- **The quant does not show**: Q8_0 is not better than Q4_K_M here, so no case for spending decode speed on bits.
- **eklow's Sage behaves as llama.cpp's**: no errors, the same prompt length (5,901 tokens with the trigger) and the same spread.

**Bits against speed, measured** (`expert-formats/`, `eklow-engine-bench` on fiona, stock Qwen3.6 at 8k, 2 slots of 16k, experts on
the card at the bytes production's card budget or a halved K/V leaves): Q4_K 48 / 92 experts 36.1 / 40.4 tok/s; Q5_K 39 / 75 31.8 /
36.0; Q6_K 32 / 63 29.6 / 32.8. Q8_0's streamed prefill has no batched repack (`xmx: batched repack is for the K-quant formats`), and its
cold experts would not fit fiona's 32 GB of pinned RAM at production's context anyway. With no gain in behaviour, the experts stay Q4_K.

**The file** (`tools/weights/make-sage.sh`): EKK8, from InternScience's Q8_0 (`1cbe2925`, sha256 `f7247f6f`) with mradermacher's
importance matrix for Sage (`6dd4c710`). Only the 123 expert tensors (with the MTP block's) and the head are requantized; the other 629
come out byte-identical to the Q8_0, checked tensor by tensor. Two keys are added: `tokenizer.ggml.normalizer.nfc` (eklow's tokenizer
requires it) and `general.sampling.*` (temperature 1.0, top-p 0.95, top-k 20; without them eklow defaulted to top-k 64). The MTP block
stays; eklow skips it. `ethanwtodd/Sage-A1.5-Q4_K-Q8_0-GGUF` @ `d89116d8`, 21,170,203,232 bytes, sha256 `6d25f141`; `make-sage.sh`
rebuilds it byte for byte. The first upload (`b89cbc0f`, sha256 `d50f428b`) had the same tensors, but its header's
`quantize.imatrix.file` held the imatrix's path on the machine that made it (a home directory). llama-quantize records the path it
is given, so the script now runs from the inputs' directory with the bare file name. The file was replaced the same day; the
repository's history still holds the first one.

KL against that Q8_0, tools/weights/kl.sh's protocol (wikitext-2 at 4k, 20 chunks; a Pile document at 32k, 2 chunks):

| file | mean KL 4k / 32k | 99th pct 4k / 32k | same top token 4k / 32k |
|---|---|---|---|
| this EKK8 | 0.0127 / 0.0524 | 0.119 / 0.652 | 95.5% / 94.4% |
| InternScience's Q4_K_M | 0.0442 / 0.1169 | 0.365 / 2.498 | 91.4% / 90.1% |

**The draft: Inco's DFlash 2 stays** (`draft/`). Sage ships an MTP head trained on itself; Inco's draft was trained on stock. Measured
in llama.cpp on the EKK8 (tokens a verify, temperature 1.0, 16 requests a dataset, GSM8K / HumanEval / MT-Bench):

| draft | tokens a verify |
|---|---|
| DFlash, blocks of 4 | 3.29 / 2.93 / 3.06 |
| MTP, 3 drafted | 2.83 / 2.63 / 2.74 |
| DFlash, blocks of 8 | 4.73 / 4.05 / 4.49 |
| MTP, 7 drafted | 3.75 / 3.18 / 3.44 |

DFlash accepts 11-30% more. MTP would also cost eklow an MoE layer and the head for each drafted token, where DFlash drafts a block in
one pass.

**Images:** InternScience's `Sage-A1.5-mmproj.gguf` holds Qwen3.6's tower. Every tensor equals the projector production serves with
Ornith within F16 rounding (222 bit for bit, the rest within 4e-7 relative), so that projector stays.

**Through the server on fiona** (production's shape: 2 x 262k int8, draft, both image encoders, EmbeddingGemma 2; `--chat-template
sage --verification on`, 48 experts a layer):
- Decode with the draft: 43.2 tok/s at 8k, 41.3 at 32k.
- Prefill: 1,636 tok/s at 8k, 1,922 at 32k.
- GSM8K one at a time: 50.9 tok/s.
- With `--kv-pool 262144` (branch `serve/kv-pool`): 52.9, 45.9 and 60.8.

`serve_test.py basic` fails six checks with Sage that pass with stock. They are the test's greedy decoding against Sage's longer
thinking: "why is the sky blue, in one sentence" thinks past the 700-token cap, and greedy tool calls repeat until 1,500. They are the
same with verification on and off, and none happened in the 162 sampled replay requests. Its developer-role check did find a bug in the default: a `developer`
message got no trigger, because the default looked at the roles before `parse_chat` maps developer to system (22 tokens against
27). Fixed, with two more checks in `eklow-verification-test` (20, 0 failures; 2 without the fix).

**A dense model for comparison, the owner's curiosity, same day:** Qwen3.8-27B at 3 bits (unsloth's UD-Q3_K_XL, `4ca72078`, 13.1 GB) on
the same replay, 18 requests a point, on the 4090.
- Right moves: 10 / 6 / 7 at the three points (Sage, all 72 of its runs: 25 / 10 / 39 out of 72; Ornith 18 / 2 / 6).
- Its best showing is point 2 (6/18 against Sage's 10/72, p = 0.081). It thinks like Ornith: medians 6,362 and 5,668 tokens, and 6 of
  36 at the 8,000-token cap against Sage's 0 of 144 (p = 4.5e-5).
- A dense model also reads all its weights every token. On fiona's A770M in the fleet's Vulkan llama.cpp (`149f564`; Mesa calls the
  Xe kernel driver experimental), with every layer on the card, it ran at 3.4 tok/s decode and ~110 tok/s prefill, so it was moved to
  the 4090.
- The 4090 gave 59-63 tok/s at 6k and 50-52 at 45-60k, ~83% of what its 1 TB/s allows for 13.1 GB of weights plus the K/V.

## 2026-11-07 — one K/V pool for the slots: `--kv-pool`, idle slots give pages back, requests wait, a starved sequence ends with "length"

**Why.** On fiona the Qwen3.6 engine gives each of its 2 slots 262,144 positions of int8 K/V: 5.84 GiB of the card's 15.1. That memory decides how many routed experts a layer stay on the card; the rest run on the CPU from DDR4, the slow part of decode. Measured today on fiona with `eklow-engine-bench` (stock Qwen3.6, decode at 8k; logs in `~/eklow-qwen-data/sage/fiona-bench-2026-11-07/`): Q4_K 48 experts on the card 36.1 tok/s, 92 experts 40.4; Q6_K 32 experts 29.6, 63 experts 32.8. Halving the K/V is what buys each second row. Production's log says it can be halved: both slots together held at most 187k positions (median 58k) over two days, one request at a time in every busy sample. The owner chose this shape for Flash-Next (2026-11-05, "the context shape: two slots, one pool", branch qwen/flash-next); this is the same change on the Qwen3.6 engine, where it can be measured now.

**What.** `engine::Config::kv_pool` (positions for all slots together, rounded up to a page of 256, at most `n_slots x max_ctx`; 0 = `n_slots x max_ctx`, today's allocation bit for bit) sets `n_pages_`; `max_ctx` stays the per-slot limit and the page table stays `n_slots x max_pages`. `Engine::free_pages_` was already one free list for every slot, so only its size changed. The K/V is allocated before `load_experts()` reads the card's free memory (`experts_on_card = -2`), so the freed memory goes to experts with no placement change; the startup line reports the pool, and the experts line the count. `server --kv-pool N`; the Gemma 4 server refuses it by name (its engine has sliding rings beside the global pages and its own scheduler: not the same few lines).
- The engine: `pool_pages()`, `free_pages()`, `slot_pages()`, and `pages_short(chunks)`: the pages a step's chunks take beyond the free ones, counting those that settling each slot's pending verify tokens gives back, in the order `step()` takes them. `step()` asks it after validation and throws `KvPoolExhausted` (a `runtime_error`) before anything changes, where `ensure_pages` used to throw in the middle of a step with the slots half advanced. The scheduler asks first, so that never fires; if it does, `run()` fails the requests as for any engine error.
- The scheduler, policy: (a) before a step, if its chunks need pages the pool does not have free, idle slots (no request: their content is only a prefix cache) give pages back, least recently used first, each cut from its end by as little as will do. The DeltaNet state exists only at snapshots, so a cut keeps the longest snapshot position that still leaves enough and puts that state back (`Engine::restore`, which frees the pages past it); with none that fits the slot is emptied (`release`). Snapshots past the kept position are dropped. The draft is not touched: the next admission truncates its context as before. (b) A request is admitted only if its prompt (less what its slot keeps) plus a step's growth (16 positions with a draft, else 1) fits the free pages, the other idle slots' pages, and its own slot's pages past what it keeps, less what running requests still need to finish their prompts; else it waits at the head of the queue, one log line naming the numbers, and goes in once pages free up. A prompt that cannot fit the whole pool is refused at `submit` as over the context (it would wait forever). Growth in generation is not reserved. (c) If the chunks still do not fit (both slots running, nothing idle to take from), the last chunk whose removal helps is dropped and its sequence ends with `finish_reason` "length" and a log line naming the pool (`stats.kv_ended`); the others go on. A verify block's rows are in the chunk, so a block of 16 is accounted like any other chunk; `spec.cpp` (acceptance) and the capture rows (the draft's context, a buffer of their own) take no pages. The stats and the periodic line report the pool: positions held, free, reclaimed since the start, and requests that waited / sequences ended.
- `src/serve/kv_pool.hpp` holds the decisions as functions over numbers (`plan_reclaim`, `admission_fits`), so they test on the host: `eklow-kvpool-test` (plans for one and two idle slots, the edges of a snapshot's room, a running slot never touched, admission with a running prompt's unfinished part reserved, and 20,000 random plans against the invariants).

**Not done.** Writing an idle slot's pages, pooled keys and snapshot to NVMe instead of dropping them (a re-prefill of a long conversation costs minutes, 3.4 GiB of Flash-Next's pool is seconds to write, estimated in its plan; its `spill.{hpp,cpp}` row). A slot whose content is cut keeps a prefix only at a snapshot, so a conversation without a snapshot before the cut is emptied; snapshots every 16k tokens and at message boundaries make the cut cheaper, but an extra snapshot at the cut point itself is not taken (the live state is at the slot's end). A verify chunk that does not fit is dropped, not shrunk to a plain token. Admission is first come first served: a big prompt at the head blocks a small one behind it.

**Estimates, not measurements.** The memory freed by a pool of 262,144 for two slots is about half of the 5.84 GiB (the K/V is linear in pages; the DeltaNet state is per slot and unchanged): ~2.9 GiB, and the table above (measured, engine bench, halved K/V) is what it buys: Q4_K 48 to 92 experts on the card and 36.1 to 40.4 tok/s, Q6_K 32 to 63 and 29.6 to 32.8. The server places fewer (59 against the bench's 71), so the gain through the server will differ. Not run yet: everything on the card. The checks, `tools/server/kv_pool_test.py` (record at the default pool, check at `--kv-pool`): a pool of 2 x 4096 positions over two slots with A trimmed and re-prefilled, a pair outgrowing the pool, a request that waits; and `serve_test.py basic` at `--kv-pool 262144 --slots 2 --ctx 262144`.

**Measured on fiona's card, same day** (`docs/bench/kv-pool-2026-11-07/`; this branch merged with `sage/template` as `test/sage-pool`, built
on fiona; the owner had the card free):
- `checkq ekk8` at the default pool (stock EKK8, eklow-engine-run): every path inside the gate. Long top-1 242-247/256, median KL
  1.8e-3 to 3.6e-3; chat 27/27, mean KL 9.5e-4. The engine is unchanged at the default.
- `kv_pool_test.py` with Sage-A1.5's EKK8 and the draft, a pool of 8,192 for two slots of 8,192 against the default pool: all checks ok.
  - trim: an idle slot was cut back to its 1,490-position system-prompt snapshot and re-prefilled the rest correctly.
  - exhaust: two 2,600-token generations outgrew the pool. One ended with "length" after 1,165 characters, a prefix of its
    default-pool text; the other finished with its default-pool text.
  - wait: a prompt that did not fit waited 7.6 s for the running one, then gave the default pool's text.
  - The log has the cut, wait and end lines.
- Production shape (`--slots 2 --ctx 262144 --kv-pool 262144`, the draft, both image encoders and EmbeddingGemma 2, Sage-A1.5 EKK8):
  K/V and state 3.03 GiB against 5.84, and **91 experts a layer on the card against 48** (the bench's halved K/V gave 92). Through
  the server (`serve_perf.py`), one stream with the draft:

  | | default pool (48) | one pool of 262k (91) | |
  |---|--:|--:|--:|
  | decode at 8k | 43.2 tok/s | 52.9 tok/s | +22% |
  | decode at 32k | 41.3 tok/s | 45.9 tok/s | +11% |
  | prefill at 8k / 32k | 1,636 / 1,922 tok/s | 1,825 / 2,005 tok/s | +12% / +4% |
  | GSM8K, 48 requests one at a time | 50.9 tok/s | 60.8 tok/s | +19% |

  Tokens a verify are the same (2.81-2.83 at 8k, 3.37-3.41 on GSM8K).
- `serve_test.py basic` at that shape: every check ok with stock Qwen3.6 EKK8. Sage fails six checks with the pool and without
  (the same six with verification on and off). They come from Sage's longer thinking under the test's greedy decoding and tight caps
  (700 tokens on "why is the sky blue, in one sentence"; repeated get_weather calls up to the 1,500-token cap), not from the server.

## 2026-11-07 — Gemma 4 out of eklow again: Qwen3.6 and its finetunes only

The owner: "we can remove gemma entirely from here. just qwen 3.6 and finetunes for now." Gemma 4 26B A4B had come back
beside Qwen on 2026-11-03 (`d9ffaf0`, the restore) and gained its speculative path the same day (`9775e98`); fiona now
serves only Sage-A1.5, a Qwen3.6 finetune, and /etc/nixos keeps one instance (`ornith`) with the instances pattern. No
fault in the 26B was found. `d9ffaf0` and `9775e98` hold its last state; the entries of 2026-11-03 ("Gemma 4 26B A4B
back beside Qwen3.6", "z-lab's draft on the card") and `docs/bench/**` stay as they were.

What went (branch `cleanup/gemma-out`; the first removal, `f15d04b`, `80fd39b`, `d8006ec` of 2026-10-31/11-01, was the
model): everything that existed only to run Gemma 4 as the chat model. The engine (`engine_gemma4.*`), its reference
forward (`model/gemma4.*`), the fast kernels (`gemma_fast.*`), the request layer, scheduler and tool-call grammar
(`*_gemma4`), the chat template and output parser (`chat_gemma4`, `output_parser_gemma4`), the Gemma server
(`tools/server/gemma4_main.*`), the tools (`eklow-engine-run-gemma4`, `-engine-bench-gemma4`, `-grammar-test-gemma4`,
`-chat-gemma4`, `-parse-gemma4`, `-prompt-gemma4`) and their reference comparers (`tools/ref/gemma4_{chat,parse}_compare.py`),
the 26B's routing profile, the `dflash-gemma-*` bench scripts and the gemma mode of `make-draft.sh`. Two tools that read
only a Gemma 4 chat GGUF went too: `eklow-kernel-test` (the Q4_0 matmul and grouped-GEMV checks on the 12B's and 26B's
tensors; the same kernels run in the Qwen engine and are checked by `eklow-qfmt-test`, `eklow-attn-test`,
`eklow-gdn-test` and `checkq`) and `eklow-cpu-gemv-test` (the Q4_0 CPU GEMV on the 26B's layer 0; `eklow-cpu-qgemv-test`
covers the CPU GEMVs of Qwen's formats). `serve_test.py` lost its Gemma family detection; `image_test.py`, `ttft.py` and
`embed_contend.py` their Gemma names. In the NixOS module the `gemma` instance example and every description naming
Gemma 4 or the 26B are gone (the named instances, `autoStart` and the conflicts stay); `flake.nix`'s description says
Qwen3.6. The README is Qwen-only again (with Sage, the shared K/V pool, the keep-alive and the verification switch);
STATUS.md has a line where Gemma's section was.

`eklow-server` still reads `general.architecture` before anything else: `qwen35moe` runs the Qwen server (flags and log
unchanged); anything else is refused, before a device is opened, with a message that names Qwen3.6 and its finetunes.

What stays, and why:
- EmbeddingGemma 2: `model/embed_gemma2.*`, the Gemma tokenizer in `text/tokenizer.*`, the gemma4v vision tower and
  `image/gemma4_image.*` that its mmproj uses, `--embed-model`, `--embed-mmproj`, `--image-tokens`. fiona's RAG
  embeddings depend on it.
- Generic code that a Gemma model motivated and that the Qwen engine, the draft or the embeddings still call: the Q4_0
  and fp16-GEMM formats and the GeGLU kernels in `xmx`, `cpu/q4_gemv` (under `cpu/qgemv`), and the plain DFlash draft
  path (`model/dflash2*`, `kernels/attention.cpp`, `kernels/ref.cpp`: per-layer sliding/full attention, a tied head, the
  embedding scale, the softcap) with its tests in `tools/dflash_test`. z-lab also publishes plain DFlash drafts for
  Qwen3.6, and the owner wants the sliding-window draft kept. Their comments no longer say "Gemma's"; their code is
  unchanged. The plain-draft tests' `--draft` modes still name z-lab's Gemma draft, so they need such a file fetched.
- `SamplingParams`' defaults (temperature 1, top-k 64, top-p 0.95, from Gemma 4's card) are only the fallback: every
  Qwen file eklow serves sets `general.sampling.*` (top-k 20), which the server reads first.

Checks (host only, e-workstation): the tree and `nix build path:.` build with the same 16 warnings as `main`; the package
has no `-gemma4` tools and only the Qwen profiles; the module evaluates with named instances. `eklow-grammar-test-qwen`
(2,000/2,000 walks, 828/828 bans on stock, Ornith and Ornith's jinja vocabularies), `eklow-spec-test`,
`eklow-kvpool-test`, `eklow-keepalive-test` and `eklow-verification-test` (20/20 on each vocabulary) pass;
`qwen_chat_compare` (stock, Ornith: 8,317 cases, 0 differences), `qwen_tok_compare` (0), `qwen_parse_compare`
(310/310), `qwen_grammar_compare` (75/75, 20,000/20,000 mutated calls refused) and the Gemma tokenizer's `tok_compare`
(0) as before; the draft's host build gives the same output hashes as `main`'s. A Gemma 4 GGUF is refused with rc 1; a
Qwen vocabulary GGUF gets as far as on `main` (no card on this host). Nothing ran on the card: the Qwen server's code is
unchanged.

## 2026-11-08 — allocate up front: the card margin from 1 GiB to 256 MiB

**Why.** The routed experts that do not fit on the card run on the CPU from DDR4, the slow part of decode, so a MiB the card does
not need is a MiB of experts. `load_experts()` places experts until the driver's free memory less `--card-margin-mib` (1,024) is
used, and the margin existed because some memory was allocated after the placement (2026-10-27, "Fit.": 0.04 GiB lost the device
once a request went deep, 0.5 and 1 GiB ran clean). The owner: allocate up front, as llama.cpp does, and stop paying a gigabyte.

**What grew after startup, measured** (fiona, `main` cc4b4b5, production's exact flags on test ports, the server's card memory sampled
each second from DRM fdinfo `drm-total-vram0`, `tools/margin/measure.sh` + `poll.py` + `stages.py`; the owner's run of the same day
gave 455 MiB, mine 459; `docs/bench/margin-2026-11-08/baseline-stages-trace0.txt`):

| stage | growth | what it was |
|---|--:|---|
| the first 252k request (prefill, then decode at depth) | +121 MiB | (a) attention scratch growing by `grow()`: split partials `part` 8.1 MiB, `part2` 3.0, query tiles 1.5 and 0.4, each regrown by doubling when a larger shape came (the two-long stage: `part` 16.3, `part2` 4.1, 0.8), 13 and 21 MiB allocated; (b) the rest, ~100 MiB, **is not ours**: kernel-driver buffer objects the user-mode driver creates on a kernel's first launch (below) |
| two long prompts at once (88k + 167k) | +12 MiB | the partials again (`part` 16.3 MiB) |
| the first chat image (Qwen vision) and its embedding (EmbeddingGemma 2's projector) | +156 MiB | attention scratch of the vision encoder grown beyond `reserve()` (29.9 MiB: its plan, heads in flight and keys a segment, depends on the padded row count, so a smaller image takes a larger scratch than the largest one), the per-image buffers (a 10.1 MiB copy of the position table, the pixels, the 7.7 MiB of rows, allocated and freed per image), and driver buffers on the first launches of the vision kernels |
| the second image and its embedding (a tall one) | +177 MiB | the same, other sizes: the driver keeps what is freed |
| everything else (EmbeddingGemma at 8k tokens, GSM8K two at a time, small images) | 0 |

How I found the sources: `tools/margin/zetrace.c`, an `LD_PRELOAD` shim that logs every `zeMemAlloc*`/`zeMemFree` with the caller's frames in
the executable (symbolized with `tools/margin/zesum.py`) and every kernel-driver buffer object (`DRM_IOCTL_XE_GEM_CREATE`, size and
placement) the same way. The BO list attributes the driver's memory: at the first prefill and decode (d8k), 92 MiB in VRAM: `ref::embed`
32 MiB twice (once from `step_group`, once from `step`), `fast::cold_scatter` 19 MiB, and 2 + 2 + 1.75 + 1.1 + 1.1 + 2 x 0.5 MiB for
the attention, `add_norm_quant`, `gemm16`, the copy engine and the grouped GEMVs; at the images, 306 MiB: the Qwen encoder 40 + 40 + 3 x 20 +
4.5, the projector 40 + 40 + 80 (`docs/bench/margin-2026-11-08/bo-trace-before-vision-warm.txt`). 32 MiB is 8 KiB x 4,096 hardware threads: these look like
per-thread scratch space for kernels with private arrays (my reading, not checked in the driver). What matters: they appear on a kernel's
first launch, in whatever queue launches it, and they stay.

**What I changed** (no kernel changed; `checkq ekk8` is bit for bit the pool branch's record):
- `src/kernels/attention.cpp`, `.hpp`: `split_count()` is the launch's own rule for the number of splits, used by the launch and by
  `reserve()`, which now takes `max_keys` and sizes `part` and `part2` for the worst case over every tile count up to `max_tiles` at
  `max_keys` (about 9 + 1 MiB at 262,144 keys: computed, not printed) and writes every buffer once (a zero fill). The engine passes
  `max_ctx + 256`, the DFlash 2 draft its ring reach + block (`dflash2_fast.cpp`), EmbeddingGemma 2 its 8,192 tokens (`embed_gemma2.cpp`).
- `src/kernels/vision_attn.cpp`: `reserve()` and `scratch_bytes()` take the largest scratch over every padded row count up to the
  reserved one (`max_sizes`), so no image grows it; zero-filled.
- `src/model/vision_qwen3vl_fast.cpp`, `vision_gemma4v_fast.cpp` (and the `Fast` structs): `ensure()` allocates, once, what an encode
  used to allocate and free per image: the position table copy and RoPE angles, the pixels, and the rows (a buffer handed out as a
  `shared_ptr` while no caller holds the previous encode's rows; if one does, or a picture is stranger than 4:1 for the projector's
  position rows, that one allocation is made as before, and logged under `EKLOW_ALLOC_TRACE`).
- `VisionQwen3VL::warm(tokens)`, `VisionGemma4V::warm(tokens)`: one synthetic image of the largest size through the whole encoder at
  startup, before the experts are placed (0.4 and 0.5 s), so the driver's buffers for their kernels exist when `load_experts()` reads the
  free memory. The server's log says what the driver took ("the driver took 0 MiB / 162 MiB": the Qwen encoder's was already there from
  `reserve()`'s zero fills and repacks).
- `src/rt/device.cpp`, `.hpp`, `tools/server/main.cpp`: `EKLOW_ALLOC_TRACE=1` makes the server log each card allocation made after startup
  (`rt::startup_done()`, called before it listens): the `grow()`s and the per-image fallbacks. None was logged in any run below.
- `tools/server/main.cpp`: `--card-margin-mib` defaults to 256 (the engine's own default stays 1 GiB for the benches).

**Residual, measured after all that** (same flags, this branch, margin 1,024 for the first two runs so that nothing was held back,
`docs/bench/margin-2026-11-08/residual-m2.txt`, `residual-m3.txt`; MiB of growth since the server was ready):

| stage | grew |
|---|--:|
| first prefill + decode at 8k (the engine's first launches) | 99 |
| images: a large and a tall JPEG, a PNG, each chat and then embedded | 5 |
| 8,000-token embedding | 0 |
| the 252k request, decode at depth | 4 |
| two long prompts (88k + 167k); a 100k + a 131k prompt both generating (when not the first stage: it took the 100 when it was) | 0 |
| verify blocks of 16 at depth ("speculative": 16 through `serve_perf.py --block 16`) | 4 |
| GSM8K, two at a time | 0 |
| total | 108 |

Another run of the first stage gave 94-100, the one before the vision warm-up 109. Everything from the first prefill on is the driver
creating its buffers for kernels launched the first time; the vision encoders' share (306 MiB before) is now inside the placement.
The shared pool's reclaiming (`kv_pool_test.py`, below) added nothing beyond the same ~100.

**The margin: 256 MiB.** 108 MiB measured, the largest single buffer of this kind 32 MiB, so 256 is 2.4 times the whole residual and
8 times the largest buffer, and the draft's unconditional slack (it reserves 0.85 GiB and takes 0.76) comes on top: the server starts
with 0.31 GiB free and the stress run below ended with ~0.2 GiB (estimate: 0.31 GiB less 103 MiB). 0.04 GiB lost the device when ~455 MiB
came after it; here 108 MiB come after 256 + 60. Not a warm-up of the engine: it needs the experts present (the MoE kernels, the cold
path), so it would run after the placement; running it before needs a second placement pass. The ~100 MiB it would save is 1.5
ranks of experts; `--card-margin-mib` still overrides.

**Experts and decode, before and after** (Sage-A1.5 EKK8, the draft, both projectors, EmbeddingGemma 2, 2 x 262,144 with the pool;
`serve_perf.py depth --repeats 5 --plain 5`, measured the same evening; `perf-base.txt`, `perf-new.txt`):

| | main cc4b4b5 | this branch | |
|---|--:|--:|--:|
| free before the experts / margin | 8.15 GiB / 1.00 | 7.75 GiB / 0.25 | the eager buffers and warm-ups took 0.40 |
| experts on the card, of 256 a layer | 91 | **96** | +5 (+0.35 GiB net) |
| decode at 8k / 32k, plain | 39.5 / 37.2 | 40.3 / 38.1 | +2.0% / +2.4% |
| decode at 8k / 32k, with the draft (5 repeats) | 56.1 / 48.0 | 55.1 / 50.0 | -1.8% / +4.2% |
| tokens a verify at 8k / 32k | 2.83 / 2.74 | 2.72 / 2.73 | |
| prefill at 8k / 32k | 1,814 / 2,002 | 1,840 / 2,011 | |

Five ranks of 256 are 3% of the cold work, so the gain is small, and the draft's rows are inside the acceptance noise (the tokens a
verify differ between the two placements: the experts that move run on the card instead of the CPU, and the logits differ in the
last bits; not investigated). The brief's premise is right (a MiB of margin is a MiB of experts) but most of the gigabyte was not
the lazy allocations: 0.40 GiB of it is what the server now holds from the start (the eager buffers and the driver's
buffers for the vision kernels), and the margin itself is 0.25. The biggest single item is EmbeddingGemma 2's projector: 264 MiB reserved for
1,120 rows (because `reserve(280)` gave NaN rows for some images) plus the 162 MiB the driver took at its warm-up, 0.42 GiB together. If the
reserve could be cut to the 280 rows the server actually sends (a fault to find first), that is ~6 more ranks (estimate: 0.42 GiB x 0.7 / 68 MiB).

**Tests on the card** (`docs/bench/margin-2026-11-08/`):
- `bash ~/eklow-perf/bench.sh checkq ekk8`: identical, line for line, to the pool branch's record (`kv-pool-2026-11-07/checkq.txt`);
  long top-1 242-248/256, median KL 1.8e-3 to 3.6e-3, chat 27/27.
- `serve_test.py basic` with stock Qwen3.6 EKK8 (`--chat-template stock`, production's other flags): all ok. Its stage: 96 MiB.
- `kv_pool_test.py record` (default pool) and `check` (two slots over 8,192 positions), Sage-A1.5 EKK8, the draft, both projectors, at the
  new margin (136 and 138 experts: the K/V is small): all ok, the same trim / exhaust / wait results as 2026-11-07; the card grew 100 MiB in each
  run (the first launches).
- The stress run (`stress-stages.txt`): deep (252k) + two long + images and their embeddings + an 8,000-token embedding + verify blocks of 16
  at depth + two generating at 100k and 131k, then the same again (each 252k prompt re-prefilled: the two long ones evict it), then GSM8K two
  at a time. 103 MiB of growth, all of it in the first stage; the second round 0; no late allocation logged; no OUT_OF_RESOURCES or
  DEVICE_LOST in the server log. `journalctl -k`: three xe forcewake timeouts (20:36:31, 20:37:23, 20:37:35) while the card was idle between two
  runs, in a runtime-PM resume (`xe_pm_runtime_resume`, via the audio component's power get); none during any run, and the next runs were fine.
  It is the failure `/etc/nixos` 432377be kept the card out of runtime suspend for (the device's `power/control` reads `auto` now).

**Not done.** An engine warm-up before the placement (above). The server's `--draft-block 16` flag itself was not run: blocks of 16 were
asked per request at the default server block of 4 (the same engine path, `max_pending` 16 already). The projector's 264 MiB (above).
Every allocation site is not instrumented under `EKLOW_ALLOC_TRACE`: the `grow()`s and the per-image fallbacks are; the shim covers the rest.

## 2026-11-08 — GET /metrics, for llmtop

The chat port serves `GET /metrics` (Prometheus text, unauthenticated like `/health`, prefix `eklow`) so llmtop, the terminal
monitor that scrapes every second, shows eklow beside unholy: slots, running and waiting requests, the K/V share held by requests,
step, token, speculation, snapshot and embedding counters, and three accounting counters llmtop's per-stream speeds need
(`*_sequence_seconds_total`, `first_token_seconds_total`). Nothing served changes: the additions are counters updated where
the scheduler already counts (plus two timestamps a step) and a read of `Scheduler::stats()` per scrape; scheduling, batching,
kernels, sampling, snapshots, the pool and every output byte are as before, and the log lines are untouched. The rendering is a
pure function (`tools/server/metrics.hpp`, tested on the host by `eklow-metrics-test`).

## 2026-11-08 — keep earlier turns' reasoning by default: `--preserve-thinking`

**What happened.** After Sage-A1.5 replaced Ornith on fiona (2026-11-07), the owner's turns sat for minutes before their first
token. fiona's journal from 2026-11-07 22:11 to 2026-11-08 07:55 shows several causes. One of them is the template.

Sage's template is stock's. It renders an assistant message's reasoning only after the last user query. When the next user
message arrives, every assistant message of the turn before it loses its `<think>` block, so the prompt parts from what the slot
holds at the first assistant message of that turn. The whole turn is prefilled again. Ornith's template always keeps the reasoning.
- 00:15:38: the turn after one of 16 tool calls and an answer reused 5,712 of its 8,201 tokens (the system prompt and the first
  messages). The prompt was 1,315 tokens shorter than the slot's sequence.
- The same applies to every longer turn: a new user message after a turn of N tokens of tool calls costs N tokens of prefill.
  (Corrected the same day: the two three-minute waits at 01:03 and 02:21, first put down to this, had the cause below.)

The night's other causes, not changed here:
- The client's step cap and `tool_choice: "none"`. The model called son-of-anton's `relay` tool with `{"action": "status"}` on the
  worker it had just started 493 times in a row (00:41-01:03, every ~2.7 s; 12 distinct results), about 374 tokens a call.
  That is how one conversation went from 5.6k to 195k tokens. son-of-anton ends a turn after 500 tool steps with one request
  with `tool_choice: "none"`. eklow renders no tools for `tool_choice: "none"`, so that prompt lost the system message's tool
  block: 190,577 tokens against the slot's 195,017, parting in the system message. Nothing was reused: 180 s to the first token
  (01:06:18). The model, with no tools in its prompt, still wrote a tool call. The client counts that as an empty reply, sent
  the same request twice more, and closed the turn as "[response failed]". At 02:21 the next turn (a worker's report, tools
  back) parted in the system message from the slot that held the tools-off render. The slot that held the tools-on one had
  been emptied for it at 01:03, so that turn waited 181 s too.
- A restart empties the cache: the reboot at 05:30 and the deploy at 07:38. At 07:51 the same 193k conversation took 184 s to its
  first token.
- Two conversations larger than the pool together (125k + 165k against 262,144, 23:30-00:08) cut each other back on every
  turn: 60 to 300 s to the first token.

**What.** `--preserve-thinking on|off` (module `preserveThinking`; default off, the template's own) is the template variable
`preserve_thinking` for a request that sends none. A request's own value, top level or in `chat_template_kwargs`, wins. On,
earlier assistant turns are rendered with their reasoning as the client sent it back (`reasoning_content`, or `<think>` in the
content; an empty block when it sends none), as Ornith's template does always. The next turn then shares the prompt up to the
previous generation prompt, where the scheduler keeps a snapshot. Ornith's template ignores the flag. The startup line says
which applies.

**Cost.** A conversation grows by the reasoning the client sends back. The model also sees past reasoning that Sage's own
template drops. Qwen3.6's template has the switch for agents, and Ornith (the same base) ran this way in production from
2026-11-02 to 2026-11-07. Sage's behaviour with it is not measured.

**Checks.** `eklow-verification-test` (host, Qwen3.6 vocabulary) adds 7 checks. The turn's own prompt is identical either way.
With preserve on, the next turn shares the prompt up to its generation prompt; with it off, it parts inside the turn. A
request's `preserve_thinking`, false or true, top level or in kwargs, overrides the default. Nothing ran on the card.

## 2026-11-08 — tool_choice none: the tools stay in the prompt, calls are never sampled

**Why.** `tool_choice: "none"` used to leave the tools out of the template, so the system message, where Qwen3.6's template
writes them, changed. The prompt then shared nothing with what the slot held past the first few tokens. On fiona that request
was the last step of son-of-anton's 500-step turn (the client sends it with the same tool list on purpose, to keep the prefix
cached). It was prefilled whole, 190,577 tokens in 180 s. The model, with no tools in its prompt, still wrote a tool call, so
the client counted the reply as empty, retried twice, and closed the turn as failed. The next turn, with tools again, met a
slot holding the tools-off render and was prefilled whole too (181 s). See "keep earlier turns' reasoning" above.

**What.** `tool_choice: "none"` renders the tools exactly as `"auto"` does, so the prompt is the same tokens and the slot's
prefix is reused. The request carries `calls_allowed = false`: its output parser bans `<tool_call>` in the content as well (the
sampler never picks a banned id, in plain decode and in the draft's verify rows alike, which take the parser's ban list).
The model can only write text or end the turn. No tool specs are attached, so no tool grammar is built. In the OpenAI API,
`"none"` means the model will not call a tool and generates a message instead, not that it does not see the tools.

**Checks.** `eklow-verification-test` (host): the `"none"` prompt is identical to `"auto"`'s and to no `tool_choice`. The
parser with calls off bans `<tool_call>` and not `<|im_end|>`, and its ban list stays ascending (the sampler needs that).
Nothing ran on the card.

## 2026-11-10 — images prepared as Hugging Face does, and the Gemma tower's FFN in tanh gelu

**Why.** Against the model's own reference (`google/embeddinggemma-2` through sentence-transformers 6.2, fp32 on a CPU),
production's (4b4bc40) text embeddings agree to cosine 0.99987-0.99995 at every length to 6,989 tokens, but its image
embeddings to only 0.953-0.986. Fed pictures already at the size Hugging Face picks, so that neither side resizes, the
tower gives 0.9989-0.9998: the encoder was right, the preparation was not. The image path followed mtmd at the oracle's
revision (ea0c4c2c, 2026-10-17), and llama.cpp has since changed three things there or the reference differs:

- Letterboxing. `resize_padded` scaled a picture to fit inside the target and centred it on black; Hugging Face's
  processors (Gemma4ImageProcessor, Qwen-VL's smart_resize) resize straight to the target size. llama.cpp dropped the
  padding for Qwen-VL in f711c8f30 (#30736, 10-19) and for Gemma 4 in 7bd180830 (#30922, 10-31). A 2939 x 2499 photo at
  280 tokens became 847 x 720 between two black strips in an 864 x 720 picture; this is most of the difference.
- The tower's FFN. The projector GGUF carries no `use_gelu`, so mtmd at the oracle's revision fell back to gelu_quick;
  the model's activation is gelu_pytorch_tanh, which llama.cpp has written into converted files and set for GEMMA4V
  since 5ba3b9fd3 (#30882, 10-20).
- The size. mtmd keeps a picture whose sides already fit the budget at roughly its own size (rounded to 48 pixels,
  scaled up only below 70 tokens); Gemma4ImageProcessor always scales to the largest whole number of 48-pixel tokens
  under the budget, up as well as down. A 240 x 488 picture at 280 tokens was 72 tokens here and in llama.cpp (still,
  at 7d7f028) and is 253 in the reference. This one llama.cpp has not changed.

Smart_resize also rounds a side's halves to even (Python's round()); mtmd has done the same for Qwen-VL since f711c8f30.

**What.** `image::fill` is Hugging Face's get_aspect_ratio_preserving_size, in double as the Python is; `image::fitted`
resizes straight to its size, and `prepare` cuts that. `Spec` loses `min_patches` (the reference has no minimum).
Qwen-VL's `fit` rounds with nearbyint, and `prepare_qwen3vl` resizes without padding. `resize_padded` is gone. The
Gemma tower's FFN uses the tanh approximation on both paths (`ref::gelu_mul`, and the same arithmetic in the fast
path's elementwise kernel); `ref::gelu_quick_mul` had no other user and is gone. The resampler stays Pillow's bicubic
in fixed point: Hugging Face's default (torchvision) processors resize with antialiased bicubic, which follows Pillow.

One difference is kept: `fit` still makes each side at least one 32-pixel tile before it checks the minimum, as mtmd and
Qwen's own `qwen_vl_utils` do, where transformers' copy of smart_resize does not. Only a picture with a side under 16
pixels tells them apart (1000 x 10: 992 x 32 here, 928 x 32 in transformers).

`eklow-image-prep` (host only) writes the picture each preparation hands on, and `tools/vision/prep_compare.py` holds
it against the processors themselves (Gemma4ImageProcessor's resize, Qwen2VLImageProcessor's), built from the
projectors' settings, so no model files are needed.

**Checks.** `prep_compare.py` on the four test pictures (a 2939 x 2499 photo, two newspaper JPEGs
of 240 x 488 and 640 x 488, a 1280 x 640 PNG) and eight synthetic PNGs (1 x 1 to 5000 x 3000, 4000 x 20, 20 x 4000,
1000 x 10): every Gemma size equal to Gemma4ImageProcessor's at 70, 280 and 1,120 tokens, every Qwen size equal to
Qwen2VLImageProcessor's at 8:1024 and 8:256 but the 1000 x 10 above; PNG bytes within 2 of torchvision's (mean 0.000-0.010), JPEG
bytes within 4 (mean 0.009-0.086), which is stb_image against Pillow's decoder (the decoded pictures alone: within 3,
mean 0.008-0.066).

The server at production's flags (eklow-sage.service's, Sage-A1.5, the draft, both projectors, `--image-tokens 280`),
EmbeddingGemma 2 against the reference, cosine:

| | this | 4b4bc40 | llama.cpp CPU, 7d7f028 |
|---|---|---|---|
| texts, 20-6,989 tokens | 0.99987-0.99995 | the same, to the bit | 0.99986-0.99991 |
| the four pictures as sent | 0.99977, 0.99987, 0.99979, 0.99994 | 0.97865, 0.96324, 0.95268, 0.97847 | 0.99953, 0.95210, 0.95455, 0.99992 |
| text + picture | 0.99958 | 0.98580 | 0.99944 |
| the pictures pre-sized (no resize anywhere) | 0.99974-0.99993 | 0.99891-0.99977 | 0.99955-0.99989 |

The pre-sized row is the tanh gelu alone. Token counts equal the reference's on every input (the two small JPEGs 257 and
270, from 76 and 134). `image_test.py` (Sage): the newspaper read as the New York Times of July 21, 1969, its 240 x 488
strip as the same page cut off, the PNG as the "LLM+" logo, the second turn's prefix reused (473 of 496 cached),
same-size pictures told apart, the refusals unchanged. The text path is untouched (no `checkq`); `check-qwen.sh`
against the oracle no longer applies to pictures the oracle letterboxes.

## 2026-11-10 — Nemotron-3.5-Bolt-E4B: the model, and its reference forward against transformers

The owner wants NVIDIA's Nemotron-3.5-Bolt-E4B (a post-trained Gemma 4 E4B, 64K context) to become fiona, the house's
Signal assistant, in Sage-A1.5's place, served by eklow (no llama.cpp stopgap). Branch `gemma/bolt-e4b`, off `main` bff9d5f.

**The model** (the BF16 repo's text_config; the GGUF says the same): 42 layers of hidden 2,560, a dense GeGLU of 10,240
(tanh gelu), 8 query heads over 2 K/V heads; five sliding layers (window 512, head 256, RoPE base 1e4 over all of it) to every
global one (head 512, base 1e6, proportional RoPE: `rope_freqs` turns the first 64 of 256 pairs and leaves the rest, factors of
1e30); V projections on the global layers too (`attention_k_eq_v` false, unlike the 12B and 26B); tied head, softcap 30.
Two things neither Gemma 4 that eklow served had:
- **Per-layer inputs** (PLE, as EmbeddingGemma 2 has half of): a 256-wide input per layer, `(rmsnorm_256(per_layer_model_proj .
  x / sqrt(2560)) * proj_norm + per_layer_token_embd[token][layer] * 16) / sqrt(2)`, from a table of 262,144 x 10,752 (Q8_0,
  2.9 GiB of the file); each layer then adds `rmsnorm(proj . (gelu(inp_gate . x) * P[layer])) * post_norm` after its FFN and
  before `layer_output_scale`.
- **Shared K/V**: the last 18 layers (24-41) have no K/V of their own; transformers' `store_full_length_kv` keeps the K/V of the
  last layer of each kind before them, so sliding layers read layer 22's and global layers (29, 35, 41) layer 23's, with their
  own queries. The GGUF still carries those layers' K/V projections (the BF16 repo's `model-frozen-shared-kv.safetensors`,
  which transformers drops on load); eklow never loads them. Only 24 layers keep a cache.

NVIDIA's file mixes Q4_K (most matrices), Q6_K (some V and down projections), Q8_0 (the token table and the per-layer table) and
BF16 (`per_layer_model_proj`); 6.18 GB, 5.71 GiB of what the forward reads.

**The reference** (`model/gemma4.*`, `eklow-ref-run` on a gemma4 GGUF): transformers' `Gemma4TextModel` in fp32 on the reference
kernels, weights dequantized as ggml stores them. **The yardstick** (`tools/ref/bolt_hf_ref.py`): transformers 5.22.2 itself, fp32
on the 4090 (TF32 off, chunks of 2,048 through its DynamicCache, the two tables and the head on the CPU), either on the BF16
checkpoint or on the GGUF's own tensors dequantized by the gguf package, so that both sides run the same numbers. The mapping
checks itself: every GGUF tensor against the checkpoint's, norms identical, quantized matrices 2-20% relative L2, so the
converter folded nothing into the values.

Prompts (Bolt's tokenizer and template, thinking on): `chat` (54 tokens: a system line and a question) and `long16k` / `long32k`
(16,384 / 32,768 tokens: eklow's README, DESIGN and DECISIONS as a user turn with a question after them). Over all positions of
`chat` and the last 256 of the long ones:

| | chat (54) | 16k (last 256) | 32k (last 256) |
|---|---|---|---|
| eklow's reference vs transformers, GGUF weights, fp32 | 54/54, mean KL 3.1e-11 | 256/256, 5.3e-8 (max 1.3e-6) | not run (below) |
| transformers in bf16 vs fp32, GGUF weights | 54/54, 8.2e-4 | 251/256, 9.3e-4 | 256/256, 8.4e-4 |
| transformers on the BF16 checkpoint vs the GGUF's weights, fp32 | 50/54, 1.6e-1 | 230/256, 1.2e-1 | 230/256, 1.0e-1 |

So the reference is transformers' arithmetic (the rest is fp32 rounding), and Q4_K_M itself moves the logits 100 times more than
bf16 activations do. The 16k run took 7,255 s on the card (the reference attention is a work-group per query and head, a group
reduction per key); 32k was not run on it, and the engine at 32k is held to transformers directly (next entries).

## 2026-11-10 — Nemotron-3.5-Bolt-E4B's text layer: Bolt's chat template beside Google's

`gemma/bolt-text` (off `main` bff9d5f; host only, e-workstation; merged into `gemma/bolt-e4b`). Bolt is NVIDIA's post-trained
Gemma 4 E4B, served from `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf` (nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB @ 1db54d5b, sha256
c8af2b8a...). That file's `tokenizer.chat_template` is the BF16 repository's `chat_template.jinja` byte for byte
(nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 @ 49843eed, 20,161 bytes). Its tokenizer is Google's (`tokenizer.json` is the 12B's
byte for byte): `tok_compare.py` on the GGUF found 0 differences in 25,133 strings (1.41M tokens). The template port, output
parser and tool grammar removed in `cc4b4b5` are back from `cc713a0`, unchanged except as follows.

**Template.** Bolt's is Google's (18,683 bytes) with three changes. The port takes them as a variant, `Gemma4Template::Bolt`.
`gemma4_template_of` recognises either file by FNV-1a and length and refuses anything else.
- `conversational_mode` (`ChatOptions::conversational_mode`) makes the system message its own trimmed text, a blank line, then
  "You are having a spoken conversation with the user. Respond as briefly as possible while still addressing their questions
  fully." With no system or developer message, the instruction becomes the system message. If the text already is the
  instruction, nothing is added. List content is joined first. A null or numeric content raises (Jinja iterates it); without
  the switch it renders nothing.
- An assistant message with tool calls renders its content after its reasoning and before the calls, instead of after their
  responses.
- With thinking off, the generation prompt ends at `<|turn>model\n`, without the empty `<|channel>thought\n<channel|>`.

`gemma4_chat_compare.py` against transformers 5.22.2: for Bolt 21,014 cases (1,008 hand-written, every branch crossed with
tools, generation prompt, thinking, preserve_thinking and conversational_mode; 6 that both must reject; 20,000 random): 20,915
identical, 99 rejected by both, 0 differences, and the 3,381,625 tokens match Hugging Face's. Seed 12: 20,923 identical, 91
rejected by both, 0 differences. Google's template: 20,504 of 20,508 identical, 4 rejected by both, 0 differences. Undoing each
change gives 6,411, 7,337 and 7,046 differences.

**Parser.** Nothing was needed for Bolt's order. It gains tool_choice "none": `OutputParser(tok, in_thinking, calls=false)`
bans and refuses `<|tool_call>` in content as well. The turn ends on generation_config's eos_token_id, 1 `<eos>`, 106
`<turn|>` and 50 `<|tool_response>`. `gemma4_parse_compare.py` (now also reading turns from inside the thought channel)
recovered all 20,000 turns of seed 5 and of seed 6 exactly.

**Grammar**, on Bolt's vocabulary: 2,000 of 2,000 and 5,000 of 5,000 calls accepted token by token; 1,997 and 4,991 walks
closed, none failed to parse; 120 of 120 bans exactly when refused, and the ban lists are ascending.

**Open: a closed turn after content and a call.** When the last messages are an assistant message with content and tool calls
and then its tool responses, both templates write `<turn|>\n` after the responses (`has_content` counts the content wherever it
was written), and the generation prompt opens no new turn: the prompt ends `…<tool_response|><turn|>\n`, plus
`<|channel>thought\n` with thinking on. Hugging Face and vLLM render it so, and so does the port.

## 2026-11-10 — Bolt's serving forward: the Gemma 4 engine back, dense, with per-layer inputs and shared K/V

`engine/engine_gemma4.*` is the 26B's engine of `cc713a0` without its MoE (routing, cold experts, streaming, placement: Bolt is
dense and all 5.71 GiB of it go on the card), with what Bolt needs:
- The file's formats on the matrix units through `xmx::repack_format` (Q4_K, Q6_K, Q8_0 as Q8_0I; BF16 as F16, which has the
  fp16 GEMM only: `per_layer_model_proj` runs as an fp16 GEMM whatever the rows). Q, K and V stack only while their types
  agree (some V are Q6_K beside Q4_K Q and K), each part into its own block, and `fast::qkv_heads` takes each from its columns.
- The per-layer inputs: the embeddings' projection, then `fast::ple_inputs` (per-layer norm, the token's Q8_0 table row on the
  card, the scales) in place; per layer `inp_gate` on the layer's output, `fast::ple_gate` (gelu times the layer's slice,
  quantized for the next matmul), `proj`, and the norm, residual, layer scale and next attention norm in one
  `residual_norm_quant`.
- K/V sharing: only the first 24 layers own a cache (20 sliding rings of window + chunk, 4 page pools), the last 18 attend over
  their source's with their own queries and write nothing.
- Attention for 8 query heads over 2 K/V heads: `run<256, 4>` (sliding, fp16) and `run<512, 4>` in int8 too (fp16 existed for
  EmbeddingGemma 2). `eklow-attn-test --heads 8 --only bolt-*` against `ref::attention`: sliding rel L2 1.0e-4, global 3.6e-4,
  global int8 8.9e-3, all ok, with and without image spans.
- The head is always the matrix units' (Q8_0I) and the token rows come from the GGUF's mapping, so a draft can share both.

**Against transformers** (fp32, the GGUF's weights; the band is transformers' own bf16 against its fp32, the precision it is
usually served in), `eklow-engine-run-gemma4`:

| | top-1 | mean KL | the bf16 band |
|---|---|---|---|
| chat, one chunk (fp16 GEMMs) | 54/54 | 3.2e-5 | 54/54, 8.2e-4 |
| chat, a 30-token chunk then 24 decoded (int8 activations throughout) | 54/54 | 4.5e-3 (decoded: 1.1e-3) | |
| 16k, chunks of 2,048 | 255/256 | 3.8e-5 | 251/256, 9.3e-4 |
| 16k, chunks of 1,024, the last 64 decoded | 255/256 | 4.6e-4 (decoded: 63/64, 1.7e-3; bf16 on those 64: 62/64, 9.9e-4) | |
| 16k, int8 global K/V | 255/256 | 6.1e-5 | |
| 32k, chunks of 2,048 | 255/256 | 2.8e-5 | 256/256, 8.4e-4 |

A second slot running the same prompt a chunk behind gives the same logits bit for bit (`--twin`: 256/256, KL 0). Verify blocks
of 4, 8 and 16 (accepting 1-16 tokens, rewound with `truncate`) against the same tokens decoded one at a time pass the run's
checks (mean KL under 0.05), the captured rows of the draft's layers 5, 11, 17, 23 and 35 within 0.8-1.8% relative RMS of the
plain run's. The int8 activations of the small-batch matmuls make a 30-row chunk and a decode step the noisiest paths, as for
Qwen: the gate for Bolt is top-1 >= 250/256 and mean KL <= 2e-3 over a long prompt's last 256 positions, <= 5e-3 over decoded
ones.

**Speed** (`eklow-engine-bench-gemma4`, fiona, real text, chunks of 2,048):

| | 1 slot, fp16 K/V | 2 slots of 65,280 | 1 slot, int8 K/V |
|---|---|---|---|
| prefill of 8k / 32k from empty | 3,606 / 3,063 tok/s | 3,601 / - | 3,599 / - |
| prefill at depth 8k / 32k / 61k (two chunks) | 3,304 / 2,541 / 1,999 tok/s | 3,281 / 2,525 / - | - / 2,529 / 1,927 |
| decode at 0 / 8k / 32k / 65k, tok/s per sequence | 50.1 / 46.1 / 40.3 / 36.0 | - / 43.5 / 35.5 / 29.6 | - / 45.2 / 40.8 / 38.1 |

A decode step at 8k (profiled, 23.6 ms with the marks): the gate|up GEMVs 5.5 ms (~225 GB/s of their 1.24 GB), down 3.2 ms, the
per-layer inputs 3.5 ms (five small launches a layer), sliding attention 2.4 ms (69 us a layer over 512 keys), the head 2.0 ms
(335 GB/s of 671 MB), Q/K/V 1.6 ms, the output projection 1.5 ms. About 3.2 ms of each step is the host enqueueing ~720 launches.

**On the card** (fiona, the production shape: 2 x 65,536 fp16, EmbeddingGemma 2 and its projector, Bolt's projector; 5.71 GiB of
weights, 2.20 GiB of K/V, the projectors 354 + 269 MiB): `serve_test.py basic` all ok (whole and streamed, reasoning_content, a
tool call with typed arguments though Gemma's grammar does not enforce types, the tool's answer, stop strings, max_tokens,
seeds, two at once equal to their solo runs, cancel, prefix reuse, the developer role). `image_test.py`: the ducks photo read as
"a serene mountain lake ... a few ducks are swimming near the edge", the fruit bowl as a table with berries, the second turn
cached 341 of 360, two same-size pictures told apart, the refusals unchanged, an image embedding beside it. An image took
11.3-11.6 s to encode then: Bolt's tower (Google's E4B tower) clamps every linear's input and output (`use_clipped_linears`; bounds
as tight as +-1.9 on some layers' attention output), which `VisionGemma4V`'s matrix-unit path refused, so it ran on the reference
kernels, which in turn never found the bounds (they looked for `NAME..input_max`). Both fixed (`6eb97f3`): the matrix-unit path
clamps where each GEMM's input is written and its output read (q, k and v share their input's bounds, as gate and up do, checked
at load). The ducks photo: fast against the fixed reference rel L2 2.2e-3 (row cosine >= 0.99993); against transformers'
Gemma4VisionModel and embedder in fp32 on the picture pre-sized as a PNG, rel L2 1.9e-3 (>= 0.99997); as the JPEG, 8.5e-2 with a
median row cosine of 0.9994 and a few sky rows at 0.83-0.90, which is stb_image and the resize against Pillow, amplified by the
clamps (the reference path gives the same). 0.10 s an image.

**Qwen unchanged.** The whole tree builds; on fiona `eklow-grammar-test-qwen` (stock and Ornith: 2,000/2,000 walks, 828/828 bans),
`eklow-spec-test`, `eklow-kvpool-test`, `eklow-keepalive-test`, `eklow-verification-test` (34 checks) and `eklow-metrics-test` pass,
and `bench.sh checkq ekk8` is identical, line for line, to `docs/bench/margin-2026-11-08/checkq-ekk8.txt` (long 244-248/256, chat
27/27).

Through the server (`serve_perf.py depth`; documents of 8,192, 32,768 and 61,440 Bolt tokens cut from one text, so each deeper
one reuses the last's prefix: the rates are the server's own, over the uncached part):

| | 8k | 32k | 60k |
|---|---|---|---|
| prefill (uncached tokens, from what the slot held) | 3,549 tok/s (8,207 from 11; TTFT 2.3 s) | 2,920 tok/s (24,591 from 8,203) | 2,273 tok/s (28,687 from 32,779) |
| decode, median of 3 | 44.8 tok/s | 39.3 tok/s | 35.6 tok/s |
| the next question on the same document | TTFT 0.11 s | 0.15 s | 0.21 s |

## 2026-11-10 — Bolt in eklow-server: the Gemma 4 server back, dense, beside Qwen's

`eklow-server` reads `general.architecture` first, as it did from 2026-11-03 to 2026-11-07: `qwen35moe` runs the Qwen server
(unchanged), `gemma4` the Gemma server (`tools/server/gemma4_main.cpp`, `serve/{openai,scheduler}_gemma4.*`, restored from
`cc713a0` and adapted), anything else is refused naming both. What changed against the 26B's server:
- No experts: no profile, placement, streaming or card margin; `--expert-profile`, `--experts-on-card`, `--card-margin-mib`,
  `--draft-reserve-mib` and `--verification` are refused by name with the Qwen-only flags. Defaults: `--ctx 65536` (Bolt's
  window; more is refused), fp16 K/V (`--kv-int8` halves the global layers').
- The template is recognised as Bolt's or Google's by fingerprint; `chat_template_kwargs.conversational_mode` and
  `preserve_thinking` (top level too) reach it; `--preserve-thinking on|off` sets the default.
- `tool_choice: "none"` renders the tools as `"auto"` does and bans `<|tool_call>` (the parser's `calls=false`; no grammar),
  as the Qwen server does since 2026-11-08, so son-of-anton's last step of a long tool loop reuses the cached prompt.
- An image's rows are causal: Gemma 4 E2B and E4B leave `use_bidirectional_attention` unset, and transformers masks their images
  as text on every layer (the 26B let an image see all of itself on the sliding layers). Their per-layer inputs take the pad
  token's table rows, as transformers' Gemma4Model does.
- Bolt's vision tower is Google's Gemma 4 E4B tower unchanged: all 657 tensors of its 16 blocks and the projection compared equal,
  bit for bit, between the BF16 checkpoint and ggml-org's `mmproj-gemma-4-E4B-it-BF16.gguf` (ggml-org/gemma-4-E4B-it-GGUF @
  27cd6836, sha256 fc1c6a1b...), which `--mmproj` therefore takes.

`eklow-prompt-gemma4` (back from `cc713a0`) renders a request as the server does: each of the ten points of fiona's replay (5,326 to
192,998 tokens: the play, house and ricky system prompts with son-of-anton's tool list, tool calls with content, tool results,
thinking on) is token for token what transformers' `apply_chat_template` gives.

## 2026-11-10 — NVIDIA's DFlash draft for Bolt: lossless, twice the decode

`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash` @ 0949a2eb (`Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf`, 737 MB): a plain DFlash
draft (no convolution or selector) of five Qwen3-style layers, hidden 2,560, 32 heads over 8 of 128, all sliding (window 1,024),
block 8, reading the target's layers 5, 11, 17, 23 and 35 (`dflash.target_layers` [6, 12, 18, 24, 36] is each id + 1, as for
the other drafts; the NVFP4 repo's config.json says `target_layer_ids` [5, 11, 17, 23, 35]), no token table or head of its own:
it embeds with the target's table times sqrt(2,560) and scores with the target's head. Quantized to Q8_0 with llama.cpp's
`llama-quantize` (399 MB, sha256 46968a8f...), since the serving forward takes Q8_0; the eklow draft code ran it unchanged
(its host reference on random features first) but for one thing:
- The file says `attention.causal` false, and NVIDIA's card says the block is drafted "in a single non-causal forward pass";
  eklow had made a plain draft's sliding layers causal inside the block (z-lab's model.py, whose GGUFs carry no such key).
  `Dflash2Config::block_causal` now follows the key when there is one.

Tried and dropped: the target's soft-cap (30) on the draft's 16 candidates (the draft scores through the target's head, and its
file names no soft-cap of its own). Acceptance fell from 3.05 to 1.30 tokens a verify on GSM8K: the draft was trained on raw
logits.

Through the server (fiona, production's shape plus the draft: 547 MiB on the card, 5.38 GiB still free; one stream unless said,
temperature 1, thinking on; `serve_perf.py`):

| | plain | blocks of 4 | blocks of 8 (the default) |
|---|---|---|---|
| prose at 8k, decode tok/s (tokens a verify) | 43.7-44.8 | 77.4 (2.49) | 90.8-96.7 (2.96-3.15) |
| prose at 32k | 38.5-39.3 | 64.3 (2.38) | 80.9-87.9 (3.13-3.42) |
| prose at 60k | 35.6 | | 64.6 (2.83) |
| GSM8K, 48 requests one at a time, aggregate | 44.0 | 89.3 (2.80) | 92.2 (3.05) |
| GSM8K two at a time, aggregate | | | 158.0 (3.89) |

Prefill is unchanged by the draft (3,495-3,510 tok/s at 8k). NVIDIA's card (llama.cpp, draft length 7, an RTX 5090, MT-Bench):
3.63 tokens a step at temperature 1 overall, 2.92 on writing, 5.67 on math.

**Lossless.** `serve_test.py sample` (the first answer token of a high-entropy prompt at temperature 1): with the draft against
without, 1,500 + 1,500 p = 0.062, then 3,000 + 3,000 p = 0.026, the largest cells moving in opposite directions between the two
runs (Sloth 128 / 91, then 177 / 234); pooled, 4,500 + 4,500 p = 0.226. Plain against plain (1,500 and 3,000) p = 0.233. The
soft-capped variant against plain, 3,000 + 3,000, p = 0.206. Greedy outputs are not identical token for token: a verify block's
rows are computed in another shape than a decode step's (the int8 GEMVs' K-splits and the attention's tiles depend on the rows),
so the logits differ in their last bits and a near-tie can go the other way. `serve_test.py compare` on the four greedy
prompts: one identical, two parting after 696 and 1,117 characters, and the haiku prompt's very first choice (to think or not)
flipped: with the draft it answers without reasoning, alone and beside another request alike. Plain decoding carries the same
caveat between step shapes (DECISIONS 2026-10-24, the serving forward pass), though on Bolt the plain server's two requests at
once matched their solo runs in full. Identical greedy output with a draft would need every int8 kernel's arithmetic to be
independent of the rows in a step: not done.

## 2026-11-10 — fiona's replay on eklow's Bolt

The private replay set on e-workstation (ten real moments of fiona's chats, P1 to P10 here: four from house, two from
play, four from ricky; with today's system prompts and son-of-anton's tool list, graded by rule or by a person), against
`eklow-server` on fiona through an ssh tunnel: production's shape (2 x 65,536 fp16, EmbeddingGemma 2 beside), no draft, six
samples a point, one request at a time. Sampling is the GGUF's and the card's: temperature 1, top-p 0.95, top-k 64 (llama.cpp
adds its own min-p 0.05 by default; eklow does not). Every request's prompt was checked first: eklow renders each point token
for token as transformers does. The ten prompts are 5,326-58,670 tokens (P10, 192,998, exceeds the 64K window and gets
llama.cpp's 400, as on llama.cpp).

Two runs of 6 samples a point, the answers a person grades read by me to the owner's standard for llama.cpp's run, against
llama.cpp CUDA's run of the same file (with the owner's readings):

| point | llama.cpp | eklow, run 1 | eklow, run 2 |
|---|---|---|---|
| P1 (house) | 0 | 0 | 1 |
| P2 (house) | 4 | 3 | 1 |
| P3 (house) | 5 | 3 | 3 |
| P4 (house) | 5 | 3 | 4 |
| P5 (play) | 0 | 0 | 0 |
| P6 (play) | 0 | 0 | 0 |
| P7 (ricky) | 6 | 6 | 6 |
| P8 (ricky) | 6 | 6 | 6 |
| P9 (ricky) | 6 | 6 | 6 |
| P10 (ricky, 192,998 tokens) | error | error | error |
| **of 54** | **32** | **27** | **27** |

54 of 108 (50%) against 32 of 54 (59%): a difference of 1.1 standard errors, so within noise, but both runs sit below, and all of
it is in P2-P4, where the same kinds of answer come out in both engines at other rates.

The text layer's open quirk (a turn closed after content and a call) is in two of the points: P8's and P9's prompts end
`<tool_response|><turn|>\n<|channel>thought\n`, as transformers renders them. All 12 answers to them passed on eklow, and all 12 on
llama.cpp, so the port keeps rendering it as Hugging Face does.

One difference between the two engines is sampling: llama.cpp's server adds min-p 0.05 to the GGUF's temperature, top-k and
top-p unless told otherwise; eklow (and the model card) has none. P2-P4 again with `"min_p": 0.05` added to every request by a
forwarder in front of eklow (my readings likewise): P2 2, P3 5, P4 4, 11 of 18 against 9 and 8 without it and llama.cpp's 14.
Suggestive, not shown at 6 samples a point. A request may send `min_p` itself; whether eklow should default to 0.05 for Bolt is
the owner's call (not changed).
