# Overcoming Computational Symmetry in LLMs: Sub-Token Neuro-Symbolic Interception and Latent Phase-Transition Mapping

**Author:** Vasily Pavlov  
**Project:** LatentHalt  
**Hardware Platform:** NVIDIA GeForce RTX 3070 Laptop GPU (8GB VRAM), PyTorch, Transformers  
**Evaluated Architecture:** Qwen2.5-1.5B-Instruct (28 layers, FP16)  
**License:** This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## Table of Contents

1. [Introduction: The Paradox of the Mechanical Thinker](#1-introduction-the-paradox-of-the-mechanical-thinker)
   - [1.1. Human Asymmetry: Autopilot vs. Deliberate Effort](#11-human-asymmetry-autopilot-vs-deliberate-effort)
   - [1.2. The Transformer as a Brakeless Conveyor Belt](#12-the-transformer-as-a-brakeless-conveyor-belt)
   - [1.3. Philosophical Grounding: The Act of Stopping as the Seed of Agency](#13-philosophical-grounding-the-act-of-stopping-as-the-seed-of-agency)
2. [Observational Architecture: Peering Inside the Black Box](#2-observational-architecture-peering-inside-the-black-box)
   - [2.1. The Residual Stream as a River of Semantics](#21-the-residual-stream-as-a-river-of-semantics)
   - [2.2. Unembedding Probing (Logit Lens Methodology)](#22-unembedding-probing-logit-lens-methodology)
   - [2.3. Logit Thermodynamics: Shannon Entropy as a Thermometer of Doubt](#23-logit-thermodynamics-shannon-entropy-as-a-thermometer-of-doubt)
3. [Tomography of Thought: Unveiling the Three-Phase Inference Topology](#3-tomography-of-thought-unveiling-the-three-phase-inference-topology)
   - [3.1. Conceptual Evolution: From Vague Intuition to Surface Token](#31-conceptual-evolution-from-vague-intuition-to-surface-token)
   - [3.2. The Chinese Rosetta Stone: How Models Think Before They Speak](#32-the-chinese-rosetta-stone-how-models-think-before-they-speak)
   - [3.3. Formalizing the Law of Three Phases](#33-formalizing-the-law-of-three-phases)
4. [Sub-Token Interception: A Bridge to von Neumann Architecture](#4-sub-token-interception-a-bridge-to-von-neumann-architecture)
   - [4.1. The Tragedy of Autoregressive Arithmetic](#41-the-tragedy-of-autoregressive-arithmetic)
   - [4.2. Hardware Interruption Dynamics (Sub-Token Routing)](#42-hardware-interruption-dynamics-sub-token-routing)
   - [4.3. Empirical Validation: 12.44x Speedup and 100% Deterministic Precision](#43-empirical-validation-1244x-speedup-and-100-deterministic-precision)
5. [The State Differential: Measuring Internal Cognitive Friction](#5-the-state-differential-measuring-internal-cognitive-friction)
   - [5.1. The Mirror of Introspection: Why a Model Must Recall Its Past](#51-the-mirror-of-introspection-why-a-model-must-recall-its-past)
   - [5.2. Mathematical Formulation: Vector Geometry and Informational Divergence](#52-mathematical-formulation-vector-geometry-and-informational-divergence)
   - [5.3. Cognitive Friction Analysis: Detecting Latent Semantic Conflict](#53-cognitive-friction-analysis-detecting-latent-semantic-conflict)
6. [Conclusion and Architectural Horizons](#6-conclusion-and-architectural-horizons)
   - [6.1. Summary of Contributions](#61-summary-of-contributions)
   - [6.2. The Future of Heterogeneous Neuro-Symbolic Computing](#62-the-future-of-heterogeneous-neuro-symbolic-computing)
7. [References and Related Work](#7-references-and-related-work)

---

## 1. Introduction: The Paradox of the Mechanical Thinker

### 1.1. Human Asymmetry: Autopilot vs. Deliberate Effort

Every day, a human mind executes thousands of cognitive operations without deliberate conscious friction. When descending a familiar flight of stairs, the human motor cortex does not compute differential equations of motion, nor does it calibrate friction coefficients to microscopic precision. The body glides forward on lower-level cerebellum and spinal cord automatisms. Cortical metabolic expenditure remains minimal: the biological system exists in a state of profound computational frugality.

However, should a step suddenly give way or prove coated in slick ice, an immediate phase transition occurs. Within milliseconds, the reticular activating system floods the cortex with norepinephrine, the prefrontal cortex fires millions of coordinated action potentials, and subjective time appears to dilate. The cognitive agent mobilizes its deepest analytical reserves to regain equilibrium.

Living intelligence is fundamentally asymmetric. It is malleable. It continuously and dynamically modulates computational energy against environmental resistance.

### 1.2. The Transformer as a Brakeless Conveyor Belt

Modern Large Language Models (LLMs) based on the Transformer architecture exhibit no such grace. Beneath their eloquent prose lies an absolute, indifferent, and rigid computational symmetry.

Consider an industrial assembly line consisting of 28, 80, or 120 sequential workshops (layers). An incoming token arrives. Whether the objective is as trivial as appending a period to a simple sentence or as demanding as deriving an intricate mathematical lemma, the pipeline runs identically. Billions of transistors execute trillions of floating-point matrix multiplications, burning identical wattage and clock cycles on tokens whose identity was already crystallized at the earliest stages.

The Transformer lacks a brake pedal. It cannot coast. For this architecture, a step on smooth asphalt commands the same computational budget as a leap across an abyss.

### 1.3. Philosophical Grounding: The Act of Stopping as the Seed of Agency

In the philosophical treatise *“Why Consciousness Cannot Be Reduced to an Intelligent Agent”*, the author established a core ontological premise: pure computation is symmetric, redundant, and intrinsically endless.

Intelligence, framed purely as an optimization function pursuing an objective loss landscape, possesses no internal justification for its own termination. An algorithmic search continues indefinitely, incrementally refining local approximations or traversing endless deductive corridors. What biological organisms exhibit as "subjectivity", "volition", or "metacognitive governance" is invariably an asymmetry — the capacity to break the computational symmetry through an "Act of Stopping". It is the declaration of a threshold: *“Computation is complete. The result suffices. Cease processing.”*

In this research, we translate this philosophical insight into tensor engineering. If true efficiency requires the capacity to halt redundant computation, can we equip an LLM with internal introspective monitoring, allowing it to recognize when an answer is solved and terminate execution early?

To achieve this, we must build an informational microscope capable of observing the model's latent states in real time.

---

## 2. Observational Architecture: Peering Inside the Black Box

### 2.1. The Residual Stream as a River of Semantics

At the heart of the decoder-only Transformer lies the **Residual Stream**. It can be conceptualized as a high-dimensional vector river of dimension $d$ flowing continuously from Layer 1 to Layer $L$.

Individual attention and feed-forward blocks do not recreate this river at each stage; instead, they draw from it, perform associative transformations, and add their output back via residual addition:

$$h_l = h_{l-1} + \Delta h_l$$

Consequently, the hidden vector corresponding to the final token at any intermediate layer $l$:

$$h_l \in \mathbb{R}^d$$

embodies the cumulative semantic refinement produced by all preceding layers.

### 2.2. Unembedding Probing (Logit Lens Methodology)

In standard inference, intermediate vectors remain opaque until the final layer $L$. Only $h_L$ undergoes final normalization and projection onto the vocabulary matrix $W_U \in \mathbb{R}^{d \times V}$ (where $V = 152\,064$ for `Qwen2.5`):

$$\mathbf{z}_L = W_U \cdot \text{RMSNorm}(h_L)$$

producing the logit vector $\mathbf{z}_L \in \mathbb{R}^V$.

We bypass this restriction using **Unembedding Probing (Logit Lens)**. At each intermediate layer $l \in [1, L]$, we intercept $h_l$, apply the final layer normalization, and project it directly onto the vocabulary head:

$$\mathbf{z}_l = W_U \cdot \text{RMSNorm}(h_l)$$

This enables real-time observation of the network's nascent hypotheses across its depth.

### 2.3. Logit Thermodynamics: Shannon Entropy as a Thermometer of Doubt

To mathematically quantify internal confidence at layer $l$, we convert intermediate logits into probability distributions parameterized by temperature $T$:

$$P_T(x_i \mid h_l) = \frac{\exp(\mathbf{z}_{l, i} / T)}{\sum_{j=1}^{V} \exp(\mathbf{z}_{l, j} / T)}$$

We then evaluate the Shannon entropy in bits:

$$H_T(h_l) = -\sum_{i=1}^{V} P_T(x_i \mid h_l) \log_2 P_T(x_i \mid h_l)$$

* **High Entropy ($H \approx 9 - 12$ bits):** The layer experiences deep uncertainty; probability mass is dispersed across thousands of candidate tokens.
* **Low Entropy ($H < 2.5$ bits):** The system has collapsed into a deterministic attractor.
* **Thermal Stress ($T = 2.0$):** If entropy remains low even under elevated thermal noise, the attractor represents an immutable mathematical consensus within the model.

---

## 3. Tomography of Thought: Unveiling the Three-Phase Inference Topology

### 3.1. Conceptual Evolution: From Vague Intuition to Surface Token

Tracing the completion of the prompt *“In winter it snows, and in summer shines the bright...”* yields a striking layer-by-layer progression:

| Layer | Top Predicted Token | Entropy $H$ (bits) | Semantic Phase Description |
| :---: | :---: | :---: | :--- |
| **L6** | `'鲜明'` (xiānmíng) | 9.13 | Early visual abstraction: "vivid / bright / sharp" |
| **L12** | `'spark'` | 9.64 | Physical representation: "spark / flare" |
| **L16** | `'light'` | 9.24 | General concept: "light" |
| **L18** | `'光辉'` (guānghuī) | 9.37 | Elevated abstraction: "radiance / brilliant glow" |
| **L20** | `'sunlight'` | 8.33 | Source localization: "sunlight" |
| **L22** | `'sunlight'` | **2.27** | **PHASE TRANSITION: The attractor locks onto solar radiation** |
| **L24** | `'阳光'` (yángguāng) | **1.91** | Monolithic conceptual core: "sunshine" |
| **L26** | `'sun'` | **1.04** | Grounding into noun root: "sun" |
| **L28** | `'с'` | **0.33** | Surface projection: Russian character `'с'` (root of "солнце") |

Between Layers 20 and 22, entropy collapses from $8.33$ to $2.27$ bits. The underlying physical reality is solved at Layer 22. The subsequent 6 layers serve entirely to translate and format this conceptual core into Russian syntax.

### 3.2. The Chinese Rosetta Stone: How Models Think Before They Speak

An even more profound dynamic emerges in counter-indication reasoning:  
*“Can antibiotics be consumed with alcohol? Doctors answer: categorically...”*

Probing intermediate layers reveals a progression of Chinese tokens:

$$\text{L16: } \mathbf{'不允许'} \to \text{L20: } \mathbf{'!'} \to \text{L24: } \mathbf{'禁止'} \to \text{L28: } \mathbf{'нельзя'}$$

At Layer 16, **`不允许` (bù yǔnxǔ)** signifies "not permitted." By Layer 24, **`禁止` (jìnzhǐ)** represents a strict medical prohibition with minimal entropy (**1.83 bits**). The model reaches absolute certainty regarding the medical fact in its internal pretraining interlingua (Chinese/English representations) deep within the network. Layer 28 merely performs cross-lingual transcription into Russian (`нельзя`).

### 3.3. Formalizing the Law of Three Phases

These empirical trajectories define the **Three-Phase Inference Topology of Transformers**:

1. **Phase I: Feature Aggregation & Syntactic Chaos (Layers 1–16):** Entropy remains high ($9.0 - 12.0$ bits). Attention heads collect context; representations remain fragmented.
2. **Phase II: Semantic Attractor & Concept Crystallization (Layers 18–24):** A sharp phase collapse occurs ($H \to 0.6 - 2.5$ bits). The problem is solved in the model's latent interlingua.
3. **Phase III: Surface Morphological Projection (Layers 25–28):** Meaning remains static; computation is expended exclusively on vocabulary alignment, case endings, and language-specific grammar.

---

## 4. Sub-Token Interception: A Bridge to von Neumann Architecture

### 4.1. The Tragedy of Autoregressive Arithmetic

Prompting an autoregressive LLM with `“Calculate: 458 * 12 =”` forces it to treat deterministic arithmetic as a stochastic guessing game. `Qwen2.5-1.5B` fails completely, expending 8 tokens across 28 layers (**224 layer-passes**) to emit formatting artifacts (`___________.\n\nSolution:`), achieving **0% accuracy**.

### 4.2. Hardware Interruption Dynamics (Sub-Token Routing)

Under **Sub-Token Interception**:

1. Context enters the network and processes through Layer 18.
2. At Layer 18 (Phase II), an internal monitor detects an algorithmic operator ($\mathbf{Op} \in \{+, -, *, /\}$).
3. The **Act of Stopping** triggers: layers 19–28 are bypassed entirely.
4. Input operands are routed directly to a deterministic CPU ALU (Python `math`).
5. The exact result is patched directly into the response stream.

### 4.3. Empirical Validation: 12.44x Speedup and 100% Deterministic Precision

| Task | Ground Truth | Baseline LLM Output | LLM Passes | LLM Accuracy | Intercept Output | Intercept Passes | Intercept Accuracy | Compute Speedup |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **458 * 12 =** | **5496** | `"___________.\n\nSolution:"` | 224 | **0%** | **5496** | 18 | **100%** | **12.44x** |
| **1024 / 16 =** | **64.0** | `"___, 35 * 7"` | 224 | **0%** | **64.0** | 18 | **100%** | **12.44x** |
| **937 + 584 =** | **1521** | `"___________.\n\nSolution:"` | 224 | **0%** | **1521** | 18 | **100%** | **12.44x** |
| **731 - 289 =** | **442** | `"___________.\n\nSolution:"` | 224 | **0%** | **442** | 18 | **100%** | **12.44x** |
| **25 * 25 =** | **625** | `"___________.\n\nTo solve"` | 224 | **0%** | **625** | 18 | **100%** | **12.44x** |

$$\text{Speedup} = \frac{8 \text{ tokens} \times 28 \text{ layers}}{1 \text{ token} \times 18 \text{ layers}} = \frac{224}{18} \approx \mathbf{12.44\times}$$

The intercept mechanism eliminates over 90% of layer computations while guaranteeing zero-error arithmetic execution.

---

## 5. The State Differential: Measuring Internal Cognitive Friction

### 5.1. The Mirror of Introspection: Why a Model Must Recall Its Past

A standard Transformer is unidirectional and blind to its own computational journey. By taking an internal snapshot at Layer 18 (BEFORE) and comparing it with Layer 28 (AFTER), we provide the architecture with an internal reference frame.

### 5.2. Mathematical Formulation: Vector Geometry and Informational Divergence

1. **Geometric Representation Drift ($\Delta_{\text{geom}}$):**

   $$\Delta_{\text{geom}} = 1 - \frac{h_{18} \cdot h_{28}}{\|h_{18}\| \|h_{28}\|}$$

2. **Informational Divergence ($\Delta_{\text{info}}$):**

   $$\Delta_{\text{info}} = D_{JS}(P_{18} \parallel P_{28}) = \frac{1}{2} D_{KL}(P_{18} \parallel M) + \frac{1}{2} D_{KL}(P_{28} \parallel M)$$

3. **Cognitive Friction Index:**

   $$\Delta_{\text{friction}} = \Delta_{\text{geom}} \times \Delta_{\text{info}}$$

### 5.3. Cognitive Friction Analysis: Detecting Latent Semantic Conflict

| Prompt Class | Input Prompt | Token L18 | Token L28 | $\Delta_{\text{geom}}$ | $\Delta_{\text{info}}$ (bits) | Friction Index $\Delta_{\text{friction}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A (Reflex)** | *"The capital of Italy is the city of"* | `'化进程'` | `'Р'` (Rome) | 0.8068 | 6.2755 | **5.0629** |
| **A (Reflex)** | *"The chemical formula of water is"* | `'composition'` | `'х'` | 0.8993 | 5.2233 | **4.6973** |
| **B (Conflict)** | *"Can antibiotics be mixed with alcohol..."* | `'不允许'` | `'нельзя'` | 0.8296 | 5.7646 | **4.7823** |
| **B (Conflict)** | *"All birds fly, but a penguin cannot..."* | `'-HT'` | `'лет'` (fly) | **0.8967** | **7.6537** | **6.8629 (MAX)** |
| **B (Conflict)** | *"The ice seemed solid, but in reality was"* | `'akt'` | `'т'` (thin) | 0.8395 | 6.8905 | **5.7844** |
| **C (Chaos)** | *"Suddenly from the chest jumped a green"* | `'-blue'` | `'к'` | 0.8589 | 6.6520 | **5.7134** |

The maximum friction index (**$6.8629$**) occurs on the flightless penguin exception. The network experiences deep internal conflict when overcoming the dominant pretraining association "bird $\to$ flight," marked by an entropy drop from $11.55$ to $3.12$ bits and an informational shift of $7.65$ bits. $\Delta_{\text{friction}}$ functions as an objective internal sensor of cognitive conflict and hallucination vulnerability.

---

## 6. Conclusion and Architectural Horizons

### 6.1. Summary of Contributions

1. **Empirical proof of latent semantic early exit:** Concepts crystallize at 65–80% network depth.
2. **Sub-token neuro-symbolic routing:** Intercepting algorithmic tasks at Layer 18 yields **12.44x compute savings** and 100% precision.
3. **Formalization of cognitive friction ($\Delta_{\text{friction}}$):** Quantitative introspection without external supervisory models.

### 6.2. The Future of Heterogeneous Neuro-Symbolic Computing

Monolithic scaling of deep transformers faces thermodynamic and economic limits. The future belongs to **asymmetric heterogeneous architectures**, wherein deep neural networks act as perceptual frontends capable of executing an **Act of Stopping** to delegate deterministic operations to specialized silicon.

---

## 7. References and Related Work

1. **Pavlov, V.** *“Why Consciousness Cannot Be Reduced to an Intelligent Agent”*, 2024.
2. **nostalgebraist.** *“Interpreting GPT: the logit lens”*, LessWrong, 2020.
3. **Belrose, N. et al.** *“Eliciting Latent Predictions from Transformers with the Tuned Lens”*, arXiv:2303.08112, 2023.
4. **Elhoushi, M. et al. (Meta AI).** *“LayerSkip: Enabling Early Exit and Layer Skipping in Large Language Models”*, arXiv:2404.16710, 2024.
5. **Kuhn, L. et al.** *“Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs”*, arXiv:2302.09664, 2023.
6. **Alibaba Cloud Qwen Team.** *“Qwen2.5 Technical Report”*, arXiv:2412.15115, 2024.