# Glossary

Every acronym used across the 31 chapter summaries, sourced from the book's own Glossary of Acronyms (reproduced in [Chapter 0](../chapters/00-preface-and-introduction.md)) plus additional acronyms defined in the chapter text. Grouped alphabetically by letter — use Ctrl-F / your browser's find within a section if you know the term.

**141 entries.**

---

## A

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **A2A** | Agent-to-Agent | Communication protocol | [24-agent-to-agent-communication](../chapters/24-agent-to-agent-communication.md) |
| **ACI** | Agent-Computer Interface | LLM-optimized shell commands (search_file, open, edit) that cut action-space complexity versus raw bash, introduced by SWE-agent | [21-agentic-environments-and-benchmarks](../chapters/21-agentic-environments-and-benchmarks.md) |
| **AdamW** | Adam with decoupled Weight decay | Optimizer variant | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |
| **ADK** | Agent Development Kit | Google's agent framework with native session/event handling and A2A support | [29-quick-reference](../chapters/29-quick-reference.md) |
| **AG2** | AutoGen | Conversation-driven multi-agent framework using GroupChat coordination | [29-quick-reference](../chapters/29-quick-reference.md) |
| **AOSE** | Agent-Oriented Software Engineering | Academic discipline studying autonomous agent architectures | [00-preface-and-introduction](../chapters/00-preface-and-introduction.md) |
| **ARQ** | Attentive Reasoning Queries | Prompting technique decomposing a query into focused sub-queries targeting narrow context slices | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |

## B

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **BDI** | Belief-Desire-Intention | Agent architecture | [25-multi-agent-systems](../chapters/25-multi-agent-systems.md) |
| **BERT** | Bidirectional Encoder Representations from Transformers | Encoder-only transformer model | [16-retrieval-augmented-generation](../chapters/16-retrieval-augmented-generation.md) |
| **BLEU** | Bilingual Evaluation Understudy | N-gram overlap metric | [14-llm-evaluation](../chapters/14-llm-evaluation.md) |
| **BM25** | Best Matching 25 | Sparse keyword-overlap ranking function used for lexical retrieval | [17-agentic-memory-systems](../chapters/17-agentic-memory-systems.md) |
| **BoN** | Best-of-N | Sampling N candidates and selecting the best by a reward model or scoring rule | [13-rl-for-large-reasoning-models](../chapters/13-rl-for-large-reasoning-models.md) |
| **BPE** | Byte-Pair Encoding | Subword tokenization algorithm | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |
| **BT** | Bradley-Terry | Preference model | [09-reward-model-training](../chapters/09-reward-model-training.md) |

## C

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **CAI** | Constitutional AI | Model critiques and revises its own outputs against a written constitution to generate preference data without human exposure to harmful content | [14-llm-evaluation](../chapters/14-llm-evaluation.md) |
| **CISPO** | Clipped IS Policy Optimization | GRPO variant combining batch-level reward normalization with token-level clipped importance sampling, used in ScaleRL | [07-grpo-group-relative-policy-optimization](../chapters/07-grpo-group-relative-policy-optimization.md) |
| **CNP** | Contract Net Protocol | Classic multi-agent coordination mechanism with announcement, bidding, award, and execution/reporting phases | [24-agent-to-agent-communication](../chapters/24-agent-to-agent-communication.md) |
| **CoALA** | Cognitive Architectures for Language Agents | Reference design language decomposing an agent into modular memory, a structured action space, and a sense-plan-act decision cycle | [17-agentic-memory-systems](../chapters/17-agentic-memory-systems.md) |
| **CoT** | Chain of Thought | Stepwise reasoning trace | [13-rl-for-large-reasoning-models](../chapters/13-rl-for-large-reasoning-models.md) |
| **CRAG** | Corrective RAG | RAG variant that adds a feedback loop correcting poor retrievals before generation | [16-retrieval-augmented-generation](../chapters/16-retrieval-augmented-generation.md) |
| **CTDE** | Centralized Training, Decentralized Execution | Multi-agent RL training paradigm | [25-multi-agent-systems](../chapters/25-multi-agent-systems.md) |
| **CUDA** | Compute Unified Device Architecture | NVIDIA GPU programming platform | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |

## D

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **DAG** | Directed Acyclic Graph | Dependency graph structure | [20-agent-design-patterns](../chapters/20-agent-design-patterns.md) |
| **DAPO** | Dynamic Adaptive Policy Optimization | GRPO variant | [07-grpo-group-relative-policy-optimization](../chapters/07-grpo-group-relative-policy-optimization.md) |
| **DDP** | Distributed Data Parallel | Data-parallel training strategy | [11-system-architecture-at-scale](../chapters/11-system-architecture-at-scale.md) |
| **DFSDT** | Depth-First Search-based Decision Tree | Search strategy ToolLLM uses to generate tool-use solution paths across large API sets | [18-agent-harness-context-and-orchestration](../chapters/18-agent-harness-context-and-orchestration.md) |
| **DiLoCo** | Distributed Low-Communication | Cross-datacenter training method | [11-system-architecture-at-scale](../chapters/11-system-architecture-at-scale.md) |
| **DP** | Data Parallelism | Parallelism strategy replicating the model across devices | [11-system-architecture-at-scale](../chapters/11-system-architecture-at-scale.md) |
| **DPO** | Direct Preference Optimization | RL-free preference-alignment algorithm | [06-dpo-direct-preference-optimization](../chapters/06-dpo-direct-preference-optimization.md) |
| **DPR** | Dense Passage Retrieval | Dense neural retrieval method using learned embeddings for semantic matching | [16-retrieval-augmented-generation](../chapters/16-retrieval-augmented-generation.md) |
| **DQN** | Deep Q-Network | Value-based deep RL algorithm | [03-introduction-to-reinforcement-learning](../chapters/03-introduction-to-reinforcement-learning.md) |
| **DRAM** | Dynamic Random-Access Memory | Off-chip GPU memory technology | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |

## E

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **EBNF** | Extended Backus-Naur Form | Grammar notation used to constrain structured LLM generation (e.g., via XGrammar) | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **ECE** | Expected Calibration Error | Metric measuring the gap between an LLM judge's confidence and its actual accuracy across confidence bins | [14-llm-evaluation](../chapters/14-llm-evaluation.md) |
| **ELO** | Elo rating system | Pairwise comparison rating scheme named after Arpad Elo | [14-llm-evaluation](../chapters/14-llm-evaluation.md) |
| **EM** | Exact Match | Binary indicator of exact string match after normalization, standard for extractive QA | [14-llm-evaluation](../chapters/14-llm-evaluation.md) |
| **EOS** | End of Sequence | Token marking sequence termination | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **EWC** | Elastic Weight Consolidation | Regularizer penalizing changes to high-Fisher-information parameters to reduce catastrophic forgetting | [10-sft-best-practices](../chapters/10-sft-best-practices.md) |

## F

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **FA** | Flash Attention | IO-aware fused attention kernel | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |
| **FAISS** | Facebook AI Similarity Search | Library providing efficient approximate nearest-neighbor search over embeddings | [16-retrieval-augmented-generation](../chapters/16-retrieval-augmented-generation.md) |
| **FFN** | Feed-Forward Network | Per-token MLP sublayer in a transformer block | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **FLOP** | Floating-Point Operation | Unit of compute | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **FSDP** | Fully Sharded Data Parallel | Parameter/gradient/optimizer-state sharded training strategy | [11-system-architecture-at-scale](../chapters/11-system-architecture-at-scale.md) |
| **FSM** | Finite State Machine | State-machine backend (used by Outlines) for constrained/structured generation | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |

## G

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **GAE** | Generalized Advantage Estimation | Bias-variance-controlled advantage estimator | [03-introduction-to-reinforcement-learning](../chapters/03-introduction-to-reinforcement-learning.md) |
| **GAIA** | General AI Assistants | Agent benchmark | [21-agentic-environments-and-benchmarks](../chapters/21-agentic-environments-and-benchmarks.md) |
| **GDPO** | Group Reward-Decoupled Policy Optimization | Normalizes each reward independently before aggregating, preventing a high-variance reward from dominating the advantage in multi-objective GRPO | [07-grpo-group-relative-policy-optimization](../chapters/07-grpo-group-relative-policy-optimization.md) |
| **GEMM** | General Matrix Multiplication | Core matrix-multiply compute primitive | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **GoT** | Graph-of-Thoughts | Reasoning method that merges branches of a reasoning tree into a graph structure | [13-rl-for-large-reasoning-models](../chapters/13-rl-for-large-reasoning-models.md) |
| **GPU** | Graphics Processing Unit | Massively parallel processor that is the primary compute substrate for LLM training and inference | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **GQA** | Grouped Query Attention | Attention variant sharing key/value heads across query groups | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |
| **GRPO** | Group Relative Policy Optimization | Value-free RL method using group-relative advantages | [07-grpo-group-relative-policy-optimization](../chapters/07-grpo-group-relative-policy-optimization.md) |

## H

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **HBM** | High Bandwidth Memory | GPU memory technology | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **HCI** | Human-Computer Interaction | Field agentic UIs draw on, alongside explainable AI and software engineering | [27-agentic-ui-frameworks](../chapters/27-agentic-ui-frameworks.md) |
| **HDR** | High Data Rate | Prior-generation InfiniBand signaling rate (200 Gb/s), superseded by NDR | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **HF** | Hugging Face | Company/ecosystem providing model hubs and libraries (Transformers, TRL) referenced throughout | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **HITL** | Human-in-the-Loop | Interaction design keeping meaningful human oversight over agent actions without bottlenecking automation | [27-agentic-ui-frameworks](../chapters/27-agentic-ui-frameworks.md) |

## I

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **ICL** | In-Context Learning | Learning tasks at inference time purely from examples in the prompt, without gradient updates | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |
| **IPO** | Identity Preference Optimization | DPO variant with a bounded loss | [08-preference-optimization-variants](../chapters/08-preference-optimization-variants.md) |

## K

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **KD** | Knowledge Distillation | Training a student model on a teacher's soft targets via a weighted cross-entropy/KL loss | [29-quick-reference](../chapters/29-quick-reference.md) |
| **KL** | Kullback-Leibler | Divergence between two distributions | [05-ppo-proximal-policy-optimization](../chapters/05-ppo-proximal-policy-optimization.md) |
| **KTO** | Kahneman-Tversky Optimization | Preference optimization from unpaired binary feedback | [08-preference-optimization-variants](../chapters/08-preference-optimization-variants.md) |
| **KV** | Key-Value | Cache of attention keys/values for decoding | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |

## L

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **LATS** | Language Agent Tree Search | Tree-search agent planning method | [12-llm-agentic-training](../chapters/12-llm-agentic-training.md) |
| **LCS** | Longest Common Subsequence | Sequence-matching measure underlying the ROUGE-L metric | [14-llm-evaluation](../chapters/14-llm-evaluation.md) |
| **LLM** | Large Language Model | Transformer-based language model | [20-agent-design-patterns](../chapters/20-agent-design-patterns.md) |
| **LMQL** | Language Model Query Language | SQL-like prompting query language with constraints for structured generation | [26-agent-development-frameworks](../chapters/26-agent-development-frameworks.md) |
| **LoRA** | Low-Rank Adaptation | Parameter-efficient fine-tuning method | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |
| **LR** | Learning Rate | Optimizer step-size hyperparameter | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |
| **LSP** | Language Server Protocol | IDE tooling protocol that MCP is explicitly modeled after, turning an N×M integration problem into N+M | [22-model-context-protocol](../chapters/22-model-context-protocol.md) |

## M

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **MAS** | Multi-Agent Systems | Systems that split work across multiple specialized, communicating agents | [25-multi-agent-systems](../chapters/25-multi-agent-systems.md) |
| **MCP** | Model Context Protocol | Standard protocol for connecting LLMs to tools and data | [22-model-context-protocol](../chapters/22-model-context-protocol.md) |
| **MCTS** | Monte Carlo Tree Search | Simulation-based tree search algorithm | [13-rl-for-large-reasoning-models](../chapters/13-rl-for-large-reasoning-models.md) |
| **MDP** | Markov Decision Process | Formalism for sequential decision problems | [03-introduction-to-reinforcement-learning](../chapters/03-introduction-to-reinforcement-learning.md) |
| **MFU** | Model FLOPS Utilization | Fraction of peak hardware FLOPS achieved | [11-system-architecture-at-scale](../chapters/11-system-architecture-at-scale.md) |
| **MHA** | Multi-Head Attention | Standard multi-head self-attention mechanism | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |
| **MIG** | Multi-Instance GPU | NVIDIA feature (from Ampere) that partitions a single GPU into isolated instances | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **MLA** | Multi-head Latent Attention | Compressed KV-cache attention variant | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |
| **MLP** | Multi-Layer Perceptron | Fully connected feed-forward network | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |
| **MMLU** | Massive Multitask Language Understanding | Raw-capability benchmark used to detect the "alignment tax" of RLHF/RL training | [10-sft-best-practices](../chapters/10-sft-best-practices.md) |
| **MoE** | Mixture of Experts | Sparsely activated expert-routing architecture | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |

## N

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **NDR** | Next Data Rate | Current-generation InfiniBand signaling rate (400 Gb/s) | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **NIC** | Network Interface Card | Per-GPU network adapter; rail-optimized wiring routes each GPU's NIC to a different top-of-rack switch | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **NLE** | NetHack Learning Environment | Gym-compatible procedurally generated roguelike benchmark demanding long-term planning | [21-agentic-environments-and-benchmarks](../chapters/21-agentic-environments-and-benchmarks.md) |
| **NLL** | Negative Log-Likelihood | Loss based on the negative log probability of the target | [06-dpo-direct-preference-optimization](../chapters/06-dpo-direct-preference-optimization.md) |
| **NOOA** | NVIDIA Object-Oriented Agents | Agent framework | [26-agent-development-frameworks](../chapters/26-agent-development-frameworks.md) |

## O

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **OODA** | Observe-Orient-Decide-Act | Military-strategy decision loop mirrored by the read-act-reflect-write cycle of agentic memory systems | [17-agentic-memory-systems](../chapters/17-agentic-memory-systems.md) |
| **OPSD** | On-Policy Self-Distillation | On-policy distillation training method | [13-rl-for-large-reasoning-models](../chapters/13-rl-for-large-reasoning-models.md) |
| **ORM** | Outcome Reward Model | Reward model scoring only the final answer | [09-reward-model-training](../chapters/09-reward-model-training.md) |
| **ORPO** | Odds Ratio Preference Optimization | Reference-free preference optimization combining SFT and preference loss | [08-preference-optimization-variants](../chapters/08-preference-optimization-variants.md) |

## P

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **P2P** | Peer-to-Peer | Decentralized multi-agent coordination topology, high resilience and scalability but harder to coordinate | [25-multi-agent-systems](../chapters/25-multi-agent-systems.md) |
| **PBRS** | Potential-Based Reward Shaping | Reward-shaping technique that provably preserves the optimal policy using a potential function | [03-introduction-to-reinforcement-learning](../chapters/03-introduction-to-reinforcement-learning.md) |
| **PBT** | Population-Based Training | Maintains a diverse population of policies/hyperparameters, replacing underperformers with mutated copies of top performers | [25-multi-agent-systems](../chapters/25-multi-agent-systems.md) |
| **PDA** | Pushdown Automaton | Automaton model used as the compiled index for constrained/structured generation | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **PEFT** | Parameter-Efficient Fine-Tuning | Family of methods that fine-tune a small parameter subset | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |
| **PL** | Plackett-Luce | Preference model extending Bradley-Terry to full rankings over multiple responses | [09-reward-model-training](../chapters/09-reward-model-training.md) |
| **POMDP** | Partially Observable Markov Decision Process | Formalism for sequential decisions where the agent only ever sees partial state, used to model agentic tasks | [12-llm-agentic-training](../chapters/12-llm-agentic-training.md) |
| **PP** | Pipeline Parallelism | Parallelism strategy splitting layers across devices | [11-system-architecture-at-scale](../chapters/11-system-architecture-at-scale.md) |
| **PPO** | Proximal Policy Optimization | Clipped-objective actor-critic RL algorithm | [05-ppo-proximal-policy-optimization](../chapters/05-ppo-proximal-policy-optimization.md) |
| **PRM** | Process Reward Model | Reward model scoring intermediate reasoning steps | [09-reward-model-training](../chapters/09-reward-model-training.md) |

## Q

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **QLoRA** | Quantized Low-Rank Adaptation | LoRA fine-tuning on a quantized base model | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |

## R

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **RAFT** | Retrieval-Augmented Fine-Tuning | Trains a model to answer using a mix of relevant and distractor documents so it learns to identify and use only the relevant context | [16-retrieval-augmented-generation](../chapters/16-retrieval-augmented-generation.md) |
| **RAG** | Retrieval-Augmented Generation | Generation grounded in retrieved documents | [16-retrieval-augmented-generation](../chapters/16-retrieval-augmented-generation.md) |
| **RAGAS** | RAG Assessment | Evaluation harness providing RAG-specific metrics | [26-agent-development-frameworks](../chapters/26-agent-development-frameworks.md) |
| **RDMA** | Remote Direct Memory Access | Reads/writes remote GPU memory without involving the remote CPU, used over InfiniBand | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **ReAct** | Reasoning + Acting | Interleaved reasoning-and-acting agent pattern | [20-agent-design-patterns](../chapters/20-agent-design-patterns.md) |
| **REPL** | Read-Eval-Print Loop | Environment pattern giving an LLM harness a variable holding context it can inspect and query programmatically | [18-agent-harness-context-and-orchestration](../chapters/18-agent-harness-context-and-orchestration.md) |
| **RFT** | Rejection Sampling Fine-Tuning | Generate many responses, select the best, SFT on the selected responses, and repeat | [08-preference-optimization-variants](../chapters/08-preference-optimization-variants.md) |
| **RL** | Reinforcement Learning | Learning from trial-and-error reward feedback | [10-sft-best-practices](../chapters/10-sft-best-practices.md) |
| **RLAIF** | Reinforcement Learning from AI Feedback | Model-generated preference feedback used in place of human labels, e.g. via Constitutional AI | [10-sft-best-practices](../chapters/10-sft-best-practices.md) |
| **RLHF** | Reinforcement Learning from Human Feedback | RL alignment using human preference signals | [11-system-architecture-at-scale](../chapters/11-system-architecture-at-scale.md) |
| **RLM** | Recursive Language Model | Harness pattern replacing one monolithic call with recursive sub-calls over partitioned context | [18-agent-harness-context-and-orchestration](../chapters/18-agent-harness-context-and-orchestration.md) |
| **RLVR** | Reinforcement Learning with Verifiable Rewards | RL using automatically checkable rewards | [04-rl-foundations-for-language-models](../chapters/04-rl-foundations-for-language-models.md) |
| **RM** | Reward Model | Model that scores outputs for RL training | [09-reward-model-training](../chapters/09-reward-model-training.md) |
| **RoPE** | Rotary Position Embedding | Rotation-based relative position encoding | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation | Summarization overlap metric | [14-llm-evaluation](../chapters/14-llm-evaluation.md) |
| **RRF** | Reciprocal Rank Fusion | Rank-based method for fusing retrieval result lists | [16-retrieval-augmented-generation](../chapters/16-retrieval-augmented-generation.md) |
| **RSC** | React Server Components | Vercel AI SDK mechanism streaming rendered UI components inline in a chat as generative UI | [27-agentic-ui-frameworks](../chapters/27-agentic-ui-frameworks.md) |

## S

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **SDK** | Software Development Kit | Software toolkit for building on a platform | [26-agent-development-frameworks](../chapters/26-agent-development-frameworks.md) |
| **SFT** | Supervised Fine-Tuning | Fine-tuning on labeled input-output pairs | [10-sft-best-practices](../chapters/10-sft-best-practices.md) |
| **SGD** | Stochastic Gradient Descent | Iterative gradient-based optimization algorithm | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |
| **SK** | Semantic Kernel | Microsoft agent development framework | [26-agent-development-frameworks](../chapters/26-agent-development-frameworks.md) |
| **SM** | Streaming Multiprocessor | GPU core compute unit | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **SPLADE** | SParse Lexical AnD Expansion | Learned sparse retrieval model | [16-retrieval-augmented-generation](../chapters/16-retrieval-augmented-generation.md) |
| **SRA** | Step-level Reasoning Accuracy | Fraction of correct intermediate reasoning steps, verified by a PRM or human annotation | [14-llm-evaluation](../chapters/14-llm-evaluation.md) |
| **SRAM** | Static Random-Access Memory | Fast on-chip GPU memory | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **SSE** | Server-Sent Events | One-way streaming HTTP protocol | [24-agent-to-agent-communication](../chapters/24-agent-to-agent-communication.md) |
| **SWE-bench** | Software Engineering Benchmark | Real-world code-repair benchmark | [21-agentic-environments-and-benchmarks](../chapters/21-agentic-environments-and-benchmarks.md) |

## T

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **TD** | Temporal Difference | Learning from bootstrapped value estimates | [03-introduction-to-reinforcement-learning](../chapters/03-introduction-to-reinforcement-learning.md) |
| **ToT** | Tree-of-Thoughts | Reasoning method adding structured search (branch, evaluate, backtrack) over a reasoning tree | [13-rl-for-large-reasoning-models](../chapters/13-rl-for-large-reasoning-models.md) |
| **TP** | Tensor Parallelism | Parallelism strategy splitting individual tensors across devices | [11-system-architecture-at-scale](../chapters/11-system-architecture-at-scale.md) |
| **TRL** | Transformer Reinforcement Learning | Hugging Face RL fine-tuning library | [07-grpo-group-relative-policy-optimization](../chapters/07-grpo-group-relative-policy-optimization.md) |
| **TSR** | Task Success Rate | Binary end-to-end agent evaluation metric: did the trajectory achieve the goal | [14-llm-evaluation](../chapters/14-llm-evaluation.md) |
| **TUA** | Tool-Use Accuracy | Fraction of tool calls that use the right tool, valid arguments, and appropriate timing | [14-llm-evaluation](../chapters/14-llm-evaluation.md) |

## U

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **UCB** | Upper Confidence Bound | Exploration strategy balancing value and uncertainty | [13-rl-for-large-reasoning-models](../chapters/13-rl-for-large-reasoning-models.md) |

## V

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **vLLM** | Virtual LLM | High-throughput inference engine | [02-systems-foundations-for-llms](../chapters/02-systems-foundations-for-llms.md) |
| **VRAM** | Video RAM | On-GPU memory whose capacity bounds trainable model/batch size | [06-dpo-direct-preference-optimization](../chapters/06-dpo-direct-preference-optimization.md) |
| **VS** | Verbalized Sampling | Training-free prompting technique that has the model verbalize a probability distribution over candidates, then samples from it | [07-grpo-group-relative-policy-optimization](../chapters/07-grpo-group-relative-policy-optimization.md) |

## W

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **W8A8** | 8-bit Weights, 8-bit Activations | Quantization scheme (e.g. SmoothQuant) enabling INT8 GEMM | [01-llm-architecture-and-optimization](../chapters/01-llm-architecture-and-optimization.md) |

## X

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **XAI** | Explainable AI | Field agentic UIs draw on to make agent behavior interpretable to users | [27-agentic-ui-frameworks](../chapters/27-agentic-ui-frameworks.md) |

## Z

| Term | Expansion | Meaning | Chapter |
|---|---|---|---|
| **ZB-H1** | Zero-Bubble (schedule H1) | Pipeline-parallelism schedule splitting the backward pass into separate B and W phases to approach zero pipeline bubble | [11-system-architecture-at-scale](../chapters/11-system-architecture-at-scale.md) |

---

*Reference material for the [chapter summaries](../README.md) of [The Hitchhiker's Guide to Agentic AI](https://arxiv.org/abs/2606.24937) by Haggai Roitman. Licensed CC BY-SA 4.0.*
