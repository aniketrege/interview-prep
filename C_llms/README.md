# Part C · Transformers & LLMs from scratch: where to study each topic

Part C is a **map, not new material**. The backbone is Raschka's *Build a Large Language Model (From Scratch)*
(local copy `../../LLMs-from-scratch`, commit `f9313ea`, Apache-2.0) and, for C6, *Build a Reasoning Model
(From Scratch)* ([rasbt/reasoning-from-scratch@a788466](https://github.com/rasbt/reasoning-from-scratch/tree/a788466),
Apache-2.0). It is a separate repository. To read it offline, clone it next to the other repos: `git clone https://github.com/rasbt/reasoning-from-scratch ../../reasoning-from-scratch` (run from this folder).

**How to use it for interviews:** read the section → close it → re-implement in a blank notebook using the
matching **drill** (blank-editor version with checks) → run Raschka's own tests where they exist
(`ch04/01_main-chapter-code/tests.py`, `ch05/01_main-chapter-code/tests.py`, `ch04/03_kv-cache/tests.py`). Links marked
"gap" are topics the books don't cover; they point to the primary source instead.

`LFS` = `../../LLMs-from-scratch` · `RFS` = reasoning-from-scratch on GitHub · `rlhf` = `../../rlhf-book`

## C1 · Tokenization (plan week 4)

| Topic | Study | Drill |
| --- | --- | --- |
| BPE train / encode / decode | [LFS ch02/05 bpe-from-scratch-simple.ipynb](../../LLMs-from-scratch/ch02/05_bpe-from-scratch/bpe-from-scratch-simple.ipynb) §1-3, then the GPT-2-compatible [bpe-from-scratch.ipynb](../../LLMs-from-scratch/ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb) | [A3.14](../A_coding/A3.14_bpe.ipynb) |
| Byte-level BPE, tiktoken, implementations compared | [LFS ch02.ipynb](../../LLMs-from-scratch/ch02/01_main-chapter-code/ch02.ipynb) §2.5 · [ch02/02_bonus_bytepair-encoder](../../LLMs-from-scratch/ch02/02_bonus_bytepair-encoder) | |
| Special tokens, extending a vocabulary | LFS ch02 §2.4 · [ch05/09 extend-tiktoken.ipynb](../../LLMs-from-scratch/ch05/09_extending-tokenizers/extend-tiktoken.ipynb) | |
| Chat / prompt templates | [LFS ch07.ipynb](../../LLMs-from-scratch/ch07/01_main-chapter-code/ch07.ipynb) §7.2 (Alpaca format); Qwen3 tokenizer + chat template in [RFS appendix C §C.8](https://github.com/rasbt/reasoning-from-scratch/blob/a788466/chC/01_main-chapter-code/chC_main.ipynb) and as used in [RFS ch08 §8.4](https://github.com/rasbt/reasoning-from-scratch/blob/a788466/ch08/01_main-chapter-code/ch08_main.ipynb) | |
| Why tokenization breaks arithmetic / spelling | gap → A. Karpathy, *Let's build the GPT Tokenizer* (2024) https://www.youtube.com/watch?v=zduSFxRajkE and [minbpe](https://github.com/karpathy/minbpe) | |

## C2 · The transformer (plan week 4)

| Topic | Study | Drill |
| --- | --- | --- |
| Token + learned positional embeddings | LFS ch02 §2.7-2.8 · [ch02/03 embeddings-vs-matmul](../../LLMs-from-scratch/ch02/03_bonus_embedding-vs-matmul/embeddings-and-linear-layers.ipynb) | |
| Self-attention → causal → multi-head | [LFS ch03.ipynb](../../LLMs-from-scratch/ch03/01_main-chapter-code/ch03.ipynb) §3.3-3.6 · 9 MHA variants incl. SDPA/FlexAttention: [ch03/02 mha-implementations.ipynb](../../LLMs-from-scratch/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) · buffers/masks: [ch03/03](../../LLMs-from-scratch/ch03/03_understanding-buffers/understanding-buffers.ipynb) | [A3.10](../A_coding/A3.10_attention.ipynb) |
| MQA / GQA, MLA, sliding-window, KV sharing | [LFS ch04/04_gqa](../../LLMs-from-scratch/ch04/04_gqa) · [ch04/05_mla](../../LLMs-from-scratch/ch04/05_mla) · [ch04/06_swa](../../LLMs-from-scratch/ch04/06_swa) · [ch04/10_kv-sharing](../../LLMs-from-scratch/ch04/10_kv-sharing) (each has a memory estimator script) | A3.10 (GQA) |
| LayerNorm, GELU FFN, residuals, pre-norm block, full GPT | [LFS ch04.ipynb](../../LLMs-from-scratch/ch04/01_main-chapter-code/ch04.ipynb) §4.2-4.6 · summary [gpt.py](../../LLMs-from-scratch/ch04/01_main-chapter-code/gpt.py). Pre- vs post-norm: Xiong et al. 2020, arXiv:2002.04745 | |
| Weight tying, parameter counting | LFS ch04 §4.6 (GPT-2 124M with/without tying) | |
| RMSNorm, SiLU/SwiGLU, RoPE (implement it) | [LFS ch05/07 converting-gpt-to-llama2.ipynb](../../LLMs-from-scratch/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb) §1.1-1.5. Papers: RMSNorm arXiv:1910.07467 · GLU variants arXiv:2002.05202 · RoFormer arXiv:2104.09864 | |
| RoPE scaling / context extension | [LFS converting-llama2-to-llama3.ipynb](../../LLMs-from-scratch/ch05/07_gpt_to_llama/converting-llama2-to-llama3.ipynb) §1.2 (Llama 3.1 frequency scaling). gap → YaRN arXiv:2309.00071; ALiBi arXiv:2108.12409 | |
| Modern architectures end to end | [LFS ch05/11 Qwen3](../../LLMs-from-scratch/ch05/11_qwen3) · [12 Gemma 3](../../LLMs-from-scratch/ch05/12_gemma3) · [13 OLMo 3](../../LLMs-from-scratch/ch05/13_olmo3) · [16 Qwen3.5](../../LLMs-from-scratch/ch05/16_qwen3.5) · [17 Gemma 4](../../LLMs-from-scratch/ch05/17_gemma4) | |
| FLOP / param counting (6ND) | [LFS ch04/02 flops-analysis.ipynb](../../LLMs-from-scratch/ch04/02_performance-analysis/flops-analysis.ipynb) | [G1.3](../G1_systems/G1.03_flops_mfu.ipynb) |
| Mixture-of-Experts: router, top-k gating | [LFS ch04/07_moe](../../LLMs-from-scratch/ch04/07_moe) · [Qwen3-MoE](../../LLMs-from-scratch/ch05/11_qwen3/standalone-qwen3-moe.ipynb). Load-balancing loss and capacity: gap in LFS → Switch Transformer arXiv:2101.03961 | [G1.15](../G1_systems/G1.15_context_expert_parallel.ipynb) |
| Linear attention / SSMs | [LFS ch04/08_deltanet](../../LLMs-from-scratch/ch04/08_deltanet) (Gated DeltaNet). gap → Mamba arXiv:2312.00752 | |
| Encoder vs decoder, BERT MLM | gap (LFS ch06 only fine-tunes BERT via HF in [ch06/03](../../LLMs-from-scratch/ch06/03_bonus_imdb-classification)) → BERT arXiv:1810.04805 | |

## C3 · Pretraining (plan week 5)

| Topic | Study | Drill |
| --- | --- | --- |
| Next-token CE, perplexity, train/val loss | [LFS ch05.ipynb](../../LLMs-from-scratch/ch05/01_main-chapter-code/ch05.ipynb) §5.1-5.2 · [gpt_train.py](../../LLMs-from-scratch/ch05/01_main-chapter-code/gpt_train.py) | [P2.7](../P_pytorch/P2.07_losses_by_hand.ipynb) (CE) |
| Sliding-window data loader | LFS ch02 §2.6 · [ch02/04 dataloader-intuition](../../LLMs-from-scratch/ch02/04_bonus_dataloader-intuition/dataloader-intuition.ipynb) | |
| Warmup, cosine decay, gradient clipping | [LFS appendix-D.ipynb](../../LLMs-from-scratch/appendix-D/01_main-chapter-code/appendix-D.ipynb) · [ch05/04 schedulers](../../LLMs-from-scratch/ch05/04_learning_rate_schedulers) | [P2.13](../P_pytorch/P2.13_training_utilities.ipynb) |
| Larger-scale pretraining, hparam search, Muon | [LFS ch05/03 Gutenberg](../../LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg) · [ch05/05 hparam](../../LLMs-from-scratch/ch05/05_bonus_hparam_tuning) · [ch05/18_muon](../../LLMs-from-scratch/ch05/18_muon) | |
| Training throughput (12.5k → 142k tok/s on one A100) | [LFS ch05/10_llm-training-speed](../../LLMs-from-scratch/ch05/10_llm-training-speed) (diff `00_orig.py` vs `01_opt_single_gpu.py`) | [G1.6](../G1_systems/G1.06_torch_compile_profiling.ipynb) |
| Dedup, packing + document masking, data mixtures | partial: [LFS ch07/02 find-near-duplicates.py](../../LLMs-from-scratch/ch07/02_dataset-utilities/find-near-duplicates.py). gap → Lee et al. 2021 dedup arXiv:2107.06499; Krell et al. 2021 packing arXiv:2107.02027 | |
| Scaling laws | gap → Kaplan et al. arXiv:2001.08361; Chinchilla arXiv:2203.15556 | G1.3 |
| Evaluation: MMLU, verifiers, LLM-as-judge, pass@k | [RFS chF](https://github.com/rasbt/reasoning-from-scratch/tree/a788466/chF) (MMLU, leaderboards, judge) · [RFS ch03](https://github.com/rasbt/reasoning-from-scratch/blob/a788466/ch03/01_main-chapter-code/ch03_main.ipynb) (MATH-500 verifier). pass@k unbiased estimator: gap → Chen et al. 2021 arXiv:2107.03374 §2.1 | |

## C4 · Inference (plan week 5)

| Topic | Study | Drill |
| --- | --- | --- |
| KV cache | [LFS ch04/03_kv-cache](../../LLMs-from-scratch/ch04/03_kv-cache) (`gpt_with_kv_cache.py`, then `_optimized`) · [RFS ch02 §2.8-2.9](https://github.com/rasbt/reasoning-from-scratch/blob/a788466/ch02/01_main-chapter-code/ch02_main.ipynb) (KV cache, torch.compile) | [A3.13](../A_coding/A3.13_kv_cache_decoding.ipynb) |
| Temperature, top-k | LFS ch05 §5.3 | [A3.11](../A_coding/A3.11_sampling.ipynb) |
| Top-p, self-consistency | [RFS ch04 §4.4-4.6](https://github.com/rasbt/reasoning-from-scratch/blob/a788466/ch04/01_main-chapter-code/ch04_main.ipynb) | A3.11 |
| Min-p, repetition penalty, beam search | gap in both books | A3.11, [A3.12](../A_coding/A3.12_beam_search.ipynb) |
| Speculative decoding (acceptance rule) | gap → Leviathan et al. arXiv:2211.17192; Chen et al. arXiv:2302.01318 | [A3.18](../A_coding/A3.18_speculative_decoding.ipynb) |
| Batching, padding masks, throughput | [RFS appendix E](https://github.com/rasbt/reasoning-from-scratch/blob/a788466/chE/chE_main.ipynb) | [G1.20](../G1_systems/G1.20_inference_serving.ipynb) |
| Paged attention, continuous batching, prefix caching | gap → vLLM arXiv:2309.06180; SGLang arXiv:2312.07104 | G1.20 |
| Quantization (int8/int4, GPTQ/AWQ) | gap → LLM.int8() arXiv:2208.07339; GPTQ arXiv:2210.17323; AWQ arXiv:2306.00978 | G1.20 (one paragraph) |

## C5 · Supervised fine-tuning (plan week 6)

| Topic | Study | Drill |
| --- | --- | --- |
| Instruction data, collate with `ignore_index=-100` padding | [LFS ch07.ipynb](../../LLMs-from-scratch/ch07/01_main-chapter-code/ch07.ipynb) §7.2-7.6 · [gpt_instruction_finetuning.py](../../LLMs-from-scratch/ch07/01_main-chapter-code/gpt_instruction_finetuning.py) | |
| Loss masking on prompt tokens | LFS ch07 [exercise-solutions.ipynb](../../LLMs-from-scratch/ch07/01_main-chapter-code/exercise-solutions.ipynb), *Exercise 7.2: Instruction and input masking* | [A3.17](../A_coding/A3.17_rm_dpo_grpo.ipynb) (response-masked log-probs) |
| LoRA (implement) | [LFS appendix-E.ipynb](../../LLMs-from-scratch/appendix-E/01_main-chapter-code/appendix-E.ipynb) · ch07 Exercise 7.4. Papers: LoRA arXiv:2106.09685; QLoRA arXiv:2305.14314 | |
| Classification fine-tuning (which layers to train) | [LFS ch06.ipynb](../../LLMs-from-scratch/ch06/01_main-chapter-code/ch06.ipynb) · [ch06/02 additional experiments](../../LLMs-from-scratch/ch06/02_bonus_additional-experiments) | |
| Evaluating instruction responses (LLM judge) | [LFS ch07/03_model-evaluation](../../LLMs-from-scratch/ch07/03_model-evaluation) · LFS ch07 §7.8 | |
| DPO from scratch | [LFS ch07/04 dpo-from-scratch.ipynb](../../LLMs-from-scratch/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb) | A3.17 |
| SFT at real scale (OLMo-2-1B on No Robots) | [rlhf code/instruction_tuning](../../rlhf-book/code/instruction_tuning) · [rlhf book ch. 4](../../rlhf-book/book/chapters/04-instruction-tuning.md) | |
| Forgetting, data quality vs quantity | gap → LIMA arXiv:2305.11206 | |

## C6 · Reasoning models (plan weeks 8, 14) [Target]

| Topic | Study | Drill |
| --- | --- | --- |
| Verifiers, answer extraction, MATH-500 eval | [RFS ch03](https://github.com/rasbt/reasoning-from-scratch/blob/a788466/ch03/01_main-chapter-code/ch03_main.ipynb) §3.1-3.9 | |
| CoT, temperature, top-p, self-consistency | [RFS ch04](https://github.com/rasbt/reasoning-from-scratch/blob/a788466/ch04/01_main-chapter-code/ch04_main.ipynb). Self-consistency arXiv:2203.11171 | |
| Log-prob scoring, self-refinement, best-of-N | [RFS ch05](https://github.com/rasbt/reasoning-from-scratch/blob/a788466/ch05/01_main-chapter-code/ch05_main.ipynb) | |
| RLVR with GRPO from scratch | [RFS ch06](https://github.com/rasbt/reasoning-from-scratch/blob/a788466/ch06/01_main-chapter-code/ch06_main.ipynb) §6.5-6.11 (rollouts → rewards → advantages → seq log-probs → loss) | A3.17 |
| GRPO fixes: clipped ratios, KL term, format reward, entropy/advantage tracking | [RFS ch07](https://github.com/rasbt/reasoning-from-scratch/blob/a788466/ch07/01_main-chapter-code/ch07_main.ipynb) §7.2-7.7 · [rlhf code/policy_gradients](../../rlhf-book/code/policy_gradients) (GRPO/RLOO/PPO variants) | A3.17 |
| Distilling reasoning traces | [RFS ch08](https://github.com/rasbt/reasoning-from-scratch/blob/a788466/ch08/01_main-chapter-code/ch08_main.ipynb) · [rlhf code/distillation](../../rlhf-book/code/distillation) | |
| Theory: RLVR, reasoning, over-optimization | [rlhf book ch. 7 reasoning](../../rlhf-book/book/chapters/07-reasoning.md) · [ch. 14 over-optimization](../../rlhf-book/book/chapters/14-over-optimization.md) | |
| DeepSeek-R1, PRMs vs ORMs, test-time scaling, budget forcing | gap → DeepSeek-R1 arXiv:2501.12948; Lightman et al. arXiv:2305.20050; Snell et al. arXiv:2408.03314; s1 arXiv:2501.19393 | |

## Suggested order (fits plan weeks 4-6)

1. Week 4: LFS ch02 → ch03 → ch04, reading one chapter per evening. Do A3.14 (BPE) and A3.10 (attention) cold the next morning.
2. Week 4 Saturday: LFS ch05 §5.1-5.2 training run (GPT-2-small-style on the short story) plus the llama2 conversion notebook (RoPE/RMSNorm/SwiGLU).
3. Week 5: LFS ch04/03 KV cache → A3.13; LFS ch05 §5.3 + RFS ch04 → A3.11; A3.12, A3.18.
4. Week 6: LFS ch07 (+ Exercise 7.2 masking) and appendix E (LoRA), then ch07/04 DPO → A3.17.
5. Week 8: RFS ch06-07 for GRPO; read `rlhf-book/code/policy_gradients/loss.py` side by side.
