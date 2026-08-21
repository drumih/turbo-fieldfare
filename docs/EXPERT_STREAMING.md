# Expert Streaming & Access Tracing Architecture

This document specifies the expert-streaming pipeline, slot-cache mechanisms, expert access tracing subsystem, offline analysis tools, and empirical evidence prerequisites for speculative prefetching in **TurboFieldfare**.

---

## 1. Expert-Streaming Architecture

In **Gemma 4 26B-A4B**, MoE layers contain **128 total experts**, of which a subset (Top-K) are dynamically selected per token by the router network. Because total expert parameters (~26B) exceed typical unified memory bandwidth budgets when fully resident, TurboFieldfare streams routed experts on-demand from disk.

### Component Pipeline

```
┌─────────────────┐       ┌────────────────────┐       ┌──────────────────────┐
│  Router Network │ ----> │ planExpertsCached  │ ----> │ executeExpertCache   │
│  (GPU Top-K)    │       │ (Slot Assignment)  │       │ (I/O & Metal Dispatch│
└─────────────────┘       └────────────────────┘       └──────────────────────┘
                                    │                             │
                                    ▼                             ▼
                          ┌────────────────────┐       ┌──────────────────────┐
                          │  Slot Cache (LFU)  │       │  pread(2) SSD Reads  │
                          │  16 slots / layer  │       │  3.2 MB per expert   │
                          └────────────────────┘       └──────────────────────┘
```

1. **`PreadExpertStreamer`**: Each transformer layer owns an independent `pread`-based streamer with a fixed per-layer slot cache (default 16 slots, supporting 8, 16, 24, 32).
2. **`pread(2)` I/O**: Missing experts are loaded directly from disk via non-blocking, offset-aligned `pread` system calls into zero-copy, page-aligned `MTLBuffer` slots.
3. **Eviction Policy**: Default replacement is **LFU** (Least Frequently Used) with **LRU** fallback.

---

## 2. Expert Access Tracing Subsystem

The expert tracing system provides fine-grained, thread-safe access observation for every routed expert access during prefill and decode phases without mutating runtime state, numerical precision, or production cache policies.

### Event Schema (`ExpertAccessEvent`)

Every routed access records:
- `tokenStep`: Generation/token step index (0-indexed).
- `layer`: Transformer layer index (0..29).
- `expert`: Expert ID (0..127).
- `routingRank`: Top-K routing rank (0 for top-1, 1 for top-2, etc.).
- `routingScore`: Floating-point probability / router weight.
- `hit`: Boolean indicator of slot-cache hit status.
- `ssdRead`: Boolean indicating whether disk I/O occurred.
- `readSize`: Bytes transferred from disk (e.g. 3,358,720 bytes).
- `readLatencyNanos`: Time taken for `pread` in nanoseconds.
- `cacheInsertion`: Boolean indicating if expert was loaded into a slot.
- `evictedExpert`: Optional ID of expert evicted from slot, if any.

---

## 3. CLI Tools & Usage

### Live Tracing during Text Generation

```bash
# Print human-readable trace analysis & cache simulation table after run
swift run -c release TurboFieldfareCLI \
  --model scratch/gemma4.gturbo \
  --prompt "The capital of France is" \
  --max-new 64 \
  --expert-trace

# Export machine-readable JSON trace to file
swift run -c release TurboFieldfareCLI \
  --model scratch/gemma4.gturbo \
  --prompt "The capital of France is" \
  --max-new 64 \
  --expert-trace-json scratch/trace.json
```

### Offline Analysis & Cache Simulation

```bash
# Analyze saved trace file offline
swift run -c release TurboFieldfareCLI \
  --analyze-expert-trace scratch/trace.json
```

Output includes:
- **Top Experts:** Most frequently accessed experts across all layers.
- **Locality Metrics:** Reuse distance & inter-arrival token distance statistics.
- **Cross-Token Reuse:** Ratio of expert overlap between consecutive tokens.
- **Cache Simulator:** Simulated hit rates for `LFU`, `LRU`, and `FIFO` policies across 4, 8, 16, and 32 slot capacities.

---

## 4. Empirical Evidence Needed Before Prefetching

Before implementing speculative expert prefetching (asynchronous I/O loading of anticipated future experts), the following empirical thresholds must be established from trace analysis:

1. **Inter-Arrival Token Distance Predictability:**
   - The median inter-arrival token distance for top experts must be small ($< 4$ tokens) with low variance. If expert access patterns are uniform random across 128 experts, prefetch accuracy will drop below hit threshold.

2. **Cross-Token Overlap Rate:**
   - Cross-token reuse ratio $\frac{|E_t \cap E_{t-1}|}{|E_t|}$ must exceed **40%**. High temporal locality is mandatory to justify prefetch buffer allocation.

3. **Layer Handoff Window vs. SSD Latency:**
   - The time window between Layer 0 router calculation and Layer $L$ execution must exceed the SSD `pread` latency (~0.1 - 0.3 ms). If GPU layer execution completes faster than SSD read latency, prefetching cannot hide I/O wait times.

4. **Bandwidth Saturation & Slot Contention:**
   - Offline simulation across 8, 16, and 32 slots must confirm that prefetch slot allocations do not increase eviction churn or evict active hits for upcoming layers.
