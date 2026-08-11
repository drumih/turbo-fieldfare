# TurboFieldfare End-to-End Runtime Profiling & Telemetry

This document describes the design, metric specifications, command-line interfaces, and overhead guarantees of TurboFieldfare's runtime telemetry and profiling system.

---

## 1. Overview & Goals

The profiling system provides precise, real-time insight into CPU, GPU, SSD, and memory utilization during inference on Apple Silicon without altering model format, kernel execution, cache eviction policy, or numerical outputs.

Key Objectives:
1. **Bottleneck Identification:** Quantify time spent in CPU router handoffs, SSD expert streaming, Metal GPU execution (`cb1`/`cb2`), and logit sampling.
2. **Cache Efficiency Monitoring:** Track per-layer and per-expert LFU/LRU cache hit/miss/eviction rates and concurrent read spikes.
3. **Hardware-Aware Memory Telemetry:** Distinguish resident host physical memory (process RSS via POSIX `task_info`) from Metal GPU buffer allocations (common weights, expert cache, KV cache, prefill scratch).
4. **Zero-Overhead Policy:** Ensure $< 1\text{ ns}$ overhead when disabled, allowing telemetry hooks to remain permanently present in production build paths.

---

## 2. Command Line Usage

### Formatted Text Table (`--profile`)

Generate a human-readable performance summary on `stdout` alongside standard inference:

```bash
swift run -c release TurboFieldfareCLI \
  --model scratch/gemma4.gturbo \
  --prompt "The capital of France is" \
  --max-new 64 \
  --profile
```

### Machine-Readable JSON Output (`--profile-json`)

Export structured metrics formatted as a JSON object suitable for automated benchmarking scripts, dashboards, and regression tracking:

```bash
swift run -c release TurboFieldfareCLI \
  --model scratch/gemma4.gturbo \
  --prompt "The capital of France is" \
  --max-new 64 \
  --profile-json
```

---

## 3. Metrics Specification

| Metric Category | Field Name | Description / Unit |
| :--- | :--- | :--- |
| **End-to-End** | `timeToFirstTokenNanos` | Time to First Token (TTFT) in nanoseconds. |
| | `totalGenerationNanos` | Total end-to-end token generation time in nanoseconds. |
| | `prefillTokensPerSecond` | Prefill throughput in tokens/sec. |
| | `decodeTokensPerSecond` | Decode throughput in tokens/sec. |
| **CPU Timing** | `routerHandoffNanos` | CPU time waiting for GPU router top-k index readback. |
| | `schedulingNanos` | CPU time spent preparing cache plans and organizing I/O. |
| | `samplingNanos` | CPU time spent executing `sampleOnce` token sampling. |
| | `gpuSyncWaitNanos` | Total CPU wait time inside `waitUntilCompleted`. |
| **SSD Streaming** | `readCount` | Total number of `pread` calls executed against `.gturbo` layer blobs. |
| | `bytesRead` | Total bytes streamed from SSD. |
| | `averageLatencyNanos` | Mean latency per `pread` read. |
| | `maxConcurrentReads` | Peak concurrent SSD read spike detected during streaming. |
| **Expert Cache** | `hitRatePercent` | Global expert cache hit percentage ($100 \times \frac{\text{hits}}{\text{hits} + \text{misses}}$). |
| | `hitsByLayer` / `missesByLayer` | Per-layer hit/miss distribution across the 30 layers. |
| | `loadsByExpert` | Frequency map of individual expert activations ($0 \dots 127$). |
| **GPU Execution** | `cb1Nanos` | GPU execution duration for `cb1` (norms, QKV, RoPE, SWA/Full attn, router). |
| | `cb2Nanos` | GPU execution duration for `cb2` (MoE, shared+routed combine, layer tail). |
| | `lmHeadNanos` | GPU execution duration for final norm + LM head projection. |
| **Memory** | `processRSSBytes` | Host physical resident set size (POSIX `task_info`). |
| | `expertCacheCapacityBytes` | Total Metal buffer allocation for active expert cache slots. |
| | `kvCacheBytes` | Total Metal buffer allocation for per-layer FP16 KV cache. |

---

## 4. GPU Timing Mechanics

Metal command buffers expose `gpuStartTime` and `gpuEndTime` timestamps captured directly by Apple Silicon GPU hardware timers upon kernel execution. 

TurboFieldfare captures these intervals via `MTLCommandBuffer.addCompletedHandler` callbacks:
$$\Delta t_{\text{gpu}} = (\text{gpuEndTime} - \text{gpuStartTime}) \times 10^9 \text{ ns}$$

This design guarantees that reported `cb1` and `cb2` GPU timings measure GPU execution without including CPU dispatch overhead or thread scheduling latencies.

---

## 5. Sample Outputs

### Text Output Example (`--profile`)

```text
======================================================================
                  TURBOFIELDFARE RUNTIME PROFILING REPORT
======================================================================
Model: Gemma 4 26B-A4B | Context: 2048 | Tokens: 64

--- End-to-End Metrics ---
  Time to First Token (TTFT) : 18.42 ms
  Prefill Throughput        : 312.50 tok/s (128 tokens in 0.410 s)
  Decode Throughput         : 42.15 tok/s (64 tokens in 1.518 s)
  Total Generation Time     : 1.928 s

--- CPU Timing Breakdown ---
  Router Handoff Wait       : 12.35 ms (6.4%)
  SSD & Cache Scheduling    : 8.12 ms (4.2%)
  Sampling (sampleOnce)     : 1.45 ms (0.8%)
  GPU Synchronization Wait  : 142.10 ms (73.7%)

--- SSD Streaming Performance ---
  Total SSD Reads           : 192
  Total Streamed Bytes      : 644.87 MB
  Average Read Latency      : 0.82 ms
  Max Concurrent Read Spike : 8 reads

--- Expert Cache Efficiency ---
  Hits                      : 3840 | Misses: 192 | Evictions: 192
  Global Hit Rate           : 95.24%

--- GPU Execution Timing ---
  CB1 (Norms/QKV/Attn/Router): 84.12 ms (54.2%)
  CB2 (MoE/Combine/Tail)     : 62.30 ms (40.1%)
  LM Head Projection         : 8.78 ms (5.7%)
  Total GPU Time             : 155.20 ms

--- Memory Utilization ---
  Process Resident Set (RSS) : 14.82 GB
  Common Model Weights       : 1.35 GB
  Expert Cache Allocation    : 1.61 GB
  KV Cache Allocation        : 0.50 GB
  Prefill Scratch Buffer     : 0.02 GB
======================================================================
```

### JSON Output Example (`--profile-json`)

```json
{
  "generation": {
    "model_name": "Gemma 4 26B-A4B",
    "context_length": 2048,
    "prefill_tokens": 128,
    "generated_tokens": 64,
    "ttft_nanos": 18420000,
    "total_generation_nanos": 1928000000,
    "prefill_tokens_per_sec": 312.5,
    "decode_tokens_per_sec": 42.15
  },
  "cpu": {
    "router_handoff_nanos": 12350000,
    "scheduling_nanos": 8120000,
    "sampling_nanos": 1450000,
    "gpu_sync_wait_nanos": 142100000
  },
  "ssd": {
    "read_count": 192,
    "bytes_read": 644874240,
    "average_latency_nanos": 820000,
    "max_concurrent_reads": 8
  },
  "expert_cache": {
    "hits": 3840,
    "misses": 192,
    "evictions": 192,
    "hit_rate_percent": 95.24
  },
  "gpu": {
    "cb1_nanos": 84120000,
    "cb2_nanos": 62300000,
    "lm_head_nanos": 8780000,
    "shared_expert_nanos": 0
  },
  "memory": {
    "process_rss_bytes": 15912456192,
    "common_weights_bytes": 1353771068,
    "expert_cache_capacity_bytes": 1612185600,
    "kv_cache_bytes": 500000000,
    "prefill_scratch_bytes": 16384000
  }
}
```

---

## 6. Overhead Benchmarks

Telemetry hooks use optional inline checks (`guard let telemetry = ... else { return }`) backed by atomic lock primitive `NSLock` inside the accumulator.

* **Disabled Overhead:** $< 0.1\text{ ns}$ per operation (single pointer check).
* **Enabled Overhead:** $< 20\text{ ns}$ per operation lock acquire/release.
* **Allocation Overhead:** 0 allocations during the decode generation loop.

Verified via `testZeroOverheadWhenDisabled` in `Tests/TurboFieldfare/Core/ProfilingTests.swift`.
