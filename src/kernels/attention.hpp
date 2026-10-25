// Attention over the KV cache, for a step's batch of tokens from several sequences, and the
// kernels that write K and V into the cache.
//
// Per query token the batch carries its sequence slot, its position, and where its sequence's
// chunk in this step starts (position and batch index) and ends (last position). Sliding layers keep
// a ring of the last `window` positions per slot; keys older than the chunk come from the ring, keys
// in the chunk from the step's own K/V (the ring cannot hold both a chunk and the window before
// it). Global layers keep paged K/V, written before attention, read through each slot's page table.
//
// Version 1 runs one sub-group of 16 lanes per (query, head, key split), each lane holding
// head_dim/16 dimensions; long key ranges are split across sub-groups and merged
// (flash-decoding). Matrix-unit flash attention for long prefill is M4 work.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace eklow::attn {

struct Meta {
  const int32_t* slot;
  const int32_t* pos;
  const int32_t* chunk_pos;   // position of the first token of this token's chunk
  const int32_t* chunk_base;  // batch index of that token
  const int32_t* chunk_last;  // position of the chunk's last token
  int64_t n_tok;
  int32_t max_pos;            // over the batch
};

struct Ring {
  sycl::half *k, *v;     // this layer's rings: [slot][window][n_head_kv][head_dim]
  int64_t slot_stride;   // elements between slots
  int64_t window;
};

struct Pages {
  sycl::half *k, *v;          // this layer's pools: [page][page_size][head_dim] (one K/V head)
  const int32_t* table;       // [slot][max_pages]
  int64_t max_pages, page_size;
};

// Scratch for the split partial results, reused across calls.
struct Scratch {
  float* p = nullptr;
  size_t floats = 0;
};

// Q [n_tok][n_head][head_dim] (roped), K/V of the step [n_tok][n_head_kv][head_dim] fp32.
void sliding(sycl::queue& q, const float* Q, const float* K, const float* V, const Meta& m, const Ring& ring,
             int64_t n_head, int64_t n_head_kv, int64_t head_dim, float scale, float* out, Scratch& scratch);
void global(sycl::queue& q, const float* Q, const Meta& m, const Pages& pages, int64_t n_head, int64_t head_dim,
            float scale, float* out, Scratch& scratch);

// Writes the step's K/V into the ring: only each chunk's last `window` positions, so no two tokens
// of a chunk land on the same ring entry.
void write_ring(sycl::queue& q, const float* K, const float* V, const Meta& m, const Ring& ring, int64_t n_head_kv,
                int64_t head_dim);
void write_pages(sycl::queue& q, const float* K, const float* V, const Meta& m, const Pages& pages, int64_t head_dim);

}  // namespace eklow::attn
