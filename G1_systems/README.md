# Part G1 · Training & inference systems, from first principles

Fifteen labs that take you from "what is a GPU actually doing" to planning 4D parallelism and low-communication
training. Every lab follows the same loop: **first-principles explanation → predict → implement it yourself →
check against PyTorch or a single-process reference → measure on this machine → interview questions with model
answers.** Each lab is a 90–120 min study session; the `**Target:**` in its header is study time, not a timed drill.

**The reference machine is the textbook** (4× RTX 3090 over PCIe, where every lab was verified). The labs measure it rather than quoting spec sheets, and compare against the cited
NVIDIA GA102 whitepaper (v2, Appendix A Table 9):

| Quantity (one RTX 3090, sm_86) | Spec (cited) | Measured in the labs |
| --- | --- | --- |
| FP16/BF16 tensor-core matmul, FP32 accumulate | 71 TFLOPS | ≈ 72 TFLOPS (G1.1) |
| FP16 matmul with FP16 accumulate (Triton) | 142 TFLOPS | ≈ 126 TFLOPS (G1.5) |
| DRAM (GDDR6X) bandwidth | 936 GB/s | ≈ 845 GB/s copy (G1.1) |
| Dependent-load latency L1 / L2 / DRAM | – | ≈ 25 / 130 / 300 ns (G1.1) |
| Eager op launch vs CUDA-graph replay | – | ≈ 7.6 µs vs ≈ 1 µs (G1.1) |
| GPT-2-small-shape training MFU, bf16 | – | ≈ 66 % (G1.3) |
| GPU↔GPU all-reduce bus bandwidth (PCIe, **no P2P, no NVLink**: NCCL goes through host shared memory) | – | ≈ 3.5 GB/s, ≈ 50 µs per ring step (G1.10), vs ≈ 370 GB/s on 8×H100 NVSwitch (cited) |
| FP8 | not supported on Ampere | conceptual only (G1.2) |

Numbers vary a few % run to run, and the `solutions/` notebooks hold the latest executed values. The slow interconnect
makes this box a good place to learn why communication dominates distributed training.

## Learning path

As a weekend thread, one lab per weekend starting in week 2 (see the [top-level README](../README.md#is-the-plan-reasonable-review-and-changes-oct-22-2026)):

| # | Lab | You will be able to… | Interview weight |
| --- | --- | --- | --- |
| 1 | [G1.1 GPU mental model & roofline](G1.01_gpu_roofline.ipynb) | time GPU code correctly; compute arithmetic intensity; say whether an op is memory-, compute- or overhead-bound and prove it | everyone |
| 2 | [G1.2 Memory math & mixed precision](G1.02_memory_math_mixed_precision.ipynb) | read float formats from bits; explain loss scaling and bf16 vs fp16; predict training memory (16Ψ + activations) and checkpointing savings; size a 7B run | everyone |
| 3 | [G1.3 FLOPs & MFU](G1.03_flops_mfu.ipynb) | derive 6N(+attention) FLOPs/token, match `FlopCounterMode` exactly, compute MFU, estimate GPU-hours for 7B on N tokens | everyone |
| 4 | [G1.4 Online softmax & FlashAttention](G1.04_online_softmax_flashattention.ipynb) | derive and implement online softmax, tiled attention forward with logsumexp, and the recompute backward; state the IO complexity | everyone, often asked |
| 5 | [G1.10 Collectives](G1.10_collectives.ipynb) | state the semantics of every collective; implement ring all-reduce; apply the α–β cost model and algbw vs busbw | Prime Intellect, big labs |
| 6 | [G1.11 DDP from scratch](G1.11_ddp_from_scratch.ipynb) | write naive → overlapped → bucketed DDP and prove it equals full-batch training; know what the Reducer does | everyone ("write a 2-process DDP script") |
| 7 | [G1.12 ZeRO & FSDP](G1.12_zero_fsdp.ipynb) | explain ZeRO stages from a memory table; implement a ZeRO-1 sharded AdamW; use FSDP2 `fully_shard` and measure it | Prime Intellect, big labs |
| 8 | [G1.6 Profiling & torch.compile](G1.06_torch_compile_profiling.ipynb) | read a profiler trace; explain Dynamo/AOTAutograd/Inductor; predict a fusion speedup from bytes; know when CUDA graphs help | Modal, everyone |
| 9 | [G1.5 Triton kernels](G1.05_triton_kernels.ipynb) | write vector-add, fused softmax, matmul and a FlashAttention-2 forward in Triton | Modal, kernel teams |
| 10 | [G1.13 Tensor & sequence parallel](G1.13_tensor_sequence_parallel.ipynb) | implement Megatron f/g ops, column→row MLP, head-parallel attention, sequence parallelism; say why TP stays inside a node | big labs |
| 11 | [G1.14 Pipeline parallel](G1.14_pipeline_parallel.ipynb) | simulate GPipe/1F1B and the bubble (p−1)/(m+p−1); run a real send/recv pipeline without deadlocks | big labs |
| 12 | [G1.15 Context & expert parallel](G1.15_context_expert_parallel.ipynb) | implement ring attention with logsumexp merging; Ulysses; MoE expert parallelism with capacity and all-to-all; Switch aux loss | big labs, MoE teams |
| 13 | [G1.16 Parallelism planning](G1.16_parallelism_planning.ipynb) | build a memory + step-time calculator and answer "70B on 64 H100s: pick a layout" style questions | design rounds |
| 14 | [G1.20 Inference serving](G1.20_inference_serving.ipynb) | prefill vs decode intensity; KV-cache sizing (MHA/GQA/MLA); measure the memory-bound decode regime; continuous batching, paged KV, prefix caching, chunked prefill | Modal, inference teams, agents (F4) |
| 15 | [G1.21 DiLoCo & low-communication training](G1.21_diloco_low_comm.ipynb) | implement DiLoCo, compare it with DDP at equal compute, compress pseudo-gradients; Streaming DiLoCo, DeMo, async RL | **Prime Intellect** |

**If you only have a few weekends before a specific loop:**
- Modal-style infra or inference: 1 → 4 → 6 → 9 → 14.
- Prime Intellect-style distributed training: 1 → 2 → 5 → 6 → 7 → 15 → 13.
- Big-lab research-scientist systems questions: 1 → 2 → 3 → 4 → 6 → 13.

## How the distributed labs run

Multi-process code can't run inside a Jupyter kernel. In G1.10–G1.16 and G1.21, each program is a
`%%writefile runs_<ID>/<name>.py` cell, with the fill-in markers inside it. A launch cell runs
`python -m torch.distributed.run --standalone --nproc_per_node=NPROC`, where `NPROC = min(4, #GPUs)`, or 2 Gloo
CPU processes if there are fewer than 2 GPUs. Rank 0 writes a JSON file that the next cell checks. Edit the
`%%writefile` cell, run it, then run the launch cell. The `runs_G1.*/` folders are scratch output (gitignored).
The checks hold for any world size ≥ 2.

## Core references (each lab lists its full, verified sources)

- N. Tazi et al., *The Ultra-Scale Playbook: Training LLMs on GPU Clusters*, Hugging Face, 2025 — https://huggingface.co/spaces/nanotron/ultrascale-playbook
- J. Austin et al., *How to Scale Your Model*, Google DeepMind, 2025 — https://jax-ml.github.io/scaling-book/
- Modal, *GPU Glossary* — https://modal.com/gpu-glossary
- H. He, *Making Deep Learning Go Brrrr From First Principles* — https://horace.io/brrr_intro.html
- Stanford CS336 (Spring 2026) Assignment 2, *Systems* — https://github.com/stanford-cs336/assignment2-systems (MIT). The lab structure for profiling, FlashAttention-2, DDP, sharded optimizer and FSDP follows its handout; test ideas adapted.
- HF picotron, minimal 4D-parallel training — https://github.com/huggingface/picotron (Apache-2.0). Reference for the TP/PP/CP/DP code.
- Triton tutorials — https://triton-lang.org/main/getting-started/tutorials/index.html (MIT)
- NVIDIA Ampere GA102 whitepaper v2 — https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
- Q. Anthony, S. Biderman, H. Schoelkopf, *Transformer Math 101*, EleutherAI, 2023 — https://blog.eleuther.ai/transformer-math/
- kipply, *Transformer Inference Arithmetic*, 2022 — https://kipp.ly/transformer-inference-arithmetic/
- Papers: FlashAttention 1/2/3 (arXiv:2205.14135, 2307.08691, 2407.08608) · ZeRO (arXiv:1910.02054) · PyTorch FSDP (arXiv:2304.11277) · Megatron-LM (arXiv:1909.08053, 2104.04473) · Korthikanti et al. activation recomputation (arXiv:2205.05198) · GPipe (arXiv:1811.06965) · Ring Attention (arXiv:2310.01889) · vLLM PagedAttention (arXiv:2309.06180) · DiLoCo (arXiv:2311.08105) · OpenDiLoCo (arXiv:2407.07852) · INTELLECT-1 (arXiv:2412.01152)
