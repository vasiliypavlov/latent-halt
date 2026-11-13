# LatentHalt: Sub-Token Neuro-Symbolic Interception & Latent Phase Mapping in LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers 4.40+](https://img.shields.io/badge/transformers-4.40+-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: CC BY-NC-SA 4.0 / MIT](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0%20%2F%20MIT-blue.svg)](#licensing)

> **Navigation / Навигация:**  
> • [Русская версия описания](#русская-версия)  
> • [English Description](#english-version)  
> • [Полная научная статья (RU)](text/latent-halt-ru.md) | [Full Scientific Paper (EN)](text/latent-halt-en.md)  
> • [Экспериментальные данные / Results (MD)](my_experiment/RESULTS.md)  
> • [Отчет по диалоговому стенду](https://github.com/vasiliypavlov/latent-halt/blob/main/chat_terminal-report-ru.md) / [Chat Terminal Report](https://github.com/vasiliypavlov/latent-halt/blob/main/chat_terminal-report-en.md)

---

<a name="русская-версия"></a>
## Русская версия

### Краткое академическое резюме
**LatentHalt** — это экспериментальный исследовательский проект, исследующий проблему вычислительной симметрии в декодерных трансформерах (LLM). В проекте реализован метод межслойного зондирования (Unembedding Probing) на модели `Qwen2.5-1.5B`, открывший трехфазную топологию инференса: абстрактные концепты кристаллизуются в скрытом представлении сети на слоях 18–24 (в мультиязычной интерлингве), после чего финальные слои тратятся исключительно на поверхностное синтаксическое декодирование. 

На основе фазового перехода разработан механизм **суб-токенного нейро-символического перехвата (Sub-Token Interception)**, позволяющий прерывать инференс на 18-м слое при распознавании алгоритмических задач и делегировать вычисления ядру фон Неймана. Это обеспечивает **многократное ускорение инференса (до 42×)** при повышении точности счета с 0% до 100%. Также введена метрика **когнитивного трения ($\Delta_{\text{friction}}$)**, количественно оценивающая меру внутреннего конфликта модели при преодолении стереотипов контекста.

---

### Интерактивный диалоговый стенд «LatentHalt-CLI» (`chat_terminal.py`)

Для демонстрации технологии в реальном времени и пакетной верификации гипотезы был разработан автономный терминальный стенд **`chat_terminal.py` (v2.1)**. 

Скрипт объединяет:
1. **Обучаемый латентный шлюз (Trainable Latent Router Probe):** компактный MLP-классификатор ($1536 \to 256 \to 3$), обучаемый за 3–5 секунд прямо на GPU на скрытых состояниях $h_{18}$ без разморозки весов основной модели (Linear Probing). Шлюз классифицирует интент запроса с точностью **$>99.7\%$** до начала авторегрессии.
2. **Мгновенный отказ по безопасности (Safety Early-Exit):** прерывание генерации на 18-м слое при обнаружении деструктивных инструкций (эксплойты, вредоносные скрипты) — экономия до 96% слое-проходов.
3. **Бесшовный откат (Graceful Fallback):** если задача распознана как математическая, но содержит контекстные ссылки («прибавь к прошлому числу») или деление на ноль, вычисления не падают с ошибкой, а мягко продолжаются генератором через слои 19–28.
4. **Калиброванное когнитивное трение (Top-K Localized JSD):** динамический замер внутреннего сопротивления модели на срезе наиболее релевантных токенов ($K=64$).

#### Краткие результаты пакетного тестирования (41 сценарий):
* **Акты Остановки ($L_{18}$ Halt):** сработали в **$53.7\%$** запросов (чистые вычисления и формулы).
* **Ускорение (Speedup):** в среднем **$21.18\times$** (на многозначных числах — до **$41.95\times$**).
* **Экономия FLOPs:** свыше **$97\%$** на алгоритмических задачах.
* **Точность счета:** **$100\%$** (против $28.5\%$ у стандартной модели с Chain-of-Thought галлюцинациями).
* **Отказоустойчивость (Fallback):** в **$12.2\%$** контекстных задач успешно задействован плавный откат к генерации без потери контекстной памяти диалога.
* **Safety Early-Exit:** **$100\%$** вредоносных запросов остановлены на слое $L_{18}$.

> 📑 **Полный аналитический отчет об эволюции стенда и метриках:**  
> Читайте в [chat_terminal-report-ru.md](https://github.com/vasiliypavlov/latent-halt/blob/main/chat_terminal-report-ru.md).

---

### Популярное введение: В поисках педали тормоза для ИИ

Современные большие языковые модели похожи на гоночный болид, у которого заклинило педаль газа. На какую бы задачу вы их ни натравили — попросили ли поставить точку в конце очевидной фразы или решить сложнейшую теорему — трансформер сжигает абсолютно одинаковое количество энергии и прогоняет данные через все свои 28, 80 или 120 слоев. В нейросети нет органа, который мог бы сказать: *«Довольно. Я поняла суть, дальше считать не нужно»*.

Опираясь на мою философскую работу *«Почему сознание не сводится к интеллектуальному агенту»* ([RU](https://github.com/vasiliypavlov/consciousness-not-an-agent/blob/main/text/consciousness-not-an-agent-ru.md), [EN](https://github.com/vasiliypavlov/consciousness-not-an-agent/blob/main/text/consciousness-not-an-agent-en.md)), я предположил: настоящий разум требует асимметрии — способности совершить «Акт Остановки».

Я создал «стетоскоп» для нейросети (метод Logit Lens с замером энтропии Шеннона) и заглянул внутрь модели `Qwen2.5-1.5B` прямо в момент ее работы:

1. **Модель мыслит интерлингвой задолго до конца:** На фразу об алкоголе и антибиотиках модель уже на 16-м слое формирует концепт строжайшего запрета в китайских токенах `不允许` и `禁止`, и лишь на 28-м слое переводит его в русское «нельзя».
2. **Арифметика через токены — безумие:** На задаче `458 * 12 =` стандартная LLM тратит 224 прохода по слоям и выдает визуальную кашу `___________`. Мой перехватчик ловит задачу на 18-м слое, тормозит сеть и отдает вычисление обычному Python. Итог: ответ `5496`, **ускорение в 12–42 раза** и 100% точность!
3. **Датчик сомнений сети ($\Delta_{\text{friction}}$):** Измеряя разницу между мыслями модели на 18-м слое и на 28-м слое, я научился численно видеть, когда модель испытывает сомнение или преодолевает стереотип.

### Быстрый старт
```bash
# 1. Клонирование репозитория
git clone https://github.com/vasiliypavlov/latent-halt.git
cd latent-halt

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Интерактивный диалоговый стенд (CLI)
python chat_terminal.py

# 4. Запуск автоматического тестирования диалогового стенда
python chat_terminal.py -t test_prompts.txt -o my_experiment3

# 5. Запуск статического научного эксперимента из статьи
python comprehensive_experiment.py
```

---

<a name="english-version"></a>
## English Version

### Executive Summary
**LatentHalt** is an experimental AI research framework addressing the fundamental flaw of computational symmetry in decoder-only Transformers. By employing Unembedding Layer Probing on `Qwen2.5-1.5B`, this project empirically proves that semantic concept crystallization occurs within intermediate hidden layers (Layers 18–24) in an abstract, multilingual latent space, long before reaching the final layer. 

Leveraging this phase transition, I introduce **Sub-Token Neuro-Symbolic Interception**: an interrupt mechanism that halts inference at Layer 18 upon detecting formal algorithmic operations, delegating computation to a deterministic von Neumann engine (CPU/Python). This achieves an **up to 42x computational speedup** with an accuracy jump from 0% to 100%. Furthermore, I introduce the **Cognitive Friction Index ($\Delta_{\text{friction}}$)**, measuring internal representation drift between early and late layers to quantify model doubt and cognitive conflict.

---

### Interactive Research Test Bench: `chat_terminal.py` (v2.1)

To demonstrate real-time execution and empirical robustness, the project includes **`chat_terminal.py`**:
* **Trainable Latent Router (L18 Probe Head):** A fast MLP probing head trained via Linear Probing directly on $h_{18}$ activations. Classifies incoming requests into *Generate*, *Deterministic Math*, or *Safety Violation* with **$>99.7\%$ confidence** prior to autoregression.
* **Safety Early-Exit:** Aborts execution at Layer 18 upon detecting malicious or exploitative prompts, avoiding wasteful token generation for refusal templates.
* **Graceful Fallback:** Seamlessly defers to full autoregression (Layers 19–28) when math problems contain contextual memory references or division by zero, preventing conversation breakage.
* **Benchmark Results (41 Prompts):** Achieved **$21.18\times$ average latency speedup**, **$>97\%$ FLOPs savings**, **100% mathematical accuracy**, and zero context drift.

> 📑 **Full Technical Evolution & Benchmark Report:**  
> See [chat_terminal-report.md](https://github.com/vasiliypavlov/latent-halt/blob/main/chat_terminal-report-en.md).

---

### Popular Overview: Giving LLMs a Brake Pedal

Modern Large Language Models act like supercars with stuck gas pedals. Whether predicting a trivial punctuation mark or solving a multi-step logical puzzle, a Transformer blindly executes every single one of its 28, 80, or 120 layers. It possesses no internal governor capable of saying: *"Stop. The answer is obvious; further computation is waste."*

Drawing from my philosophical treatise *"Why Consciousness Cannot Be Reduced to an Intelligent Agent"* ([RU](https://github.com/vasiliypavlov/consciousness-not-an-agent/blob/main/text/consciousness-not-an-agent-ru.md), [EN](https://github.com/vasiliypavlov/consciousness-not-an-agent/blob/main/text/consciousness-not-an-agent-en.md)), I posited that **true agency requires asymmetry — the capacity to execute an "Act of Stopping."**

Using an informational microscope (Logit Lens coupled with Shannon entropy probing), I peered into the hidden residual stream of `Qwen2.5-1.5B`:

1. **The Multilingual Concept Nucleus:** When asked about contraindications between antibiotics and alcohol, the network solidifies a categorical prohibition as early as Layer 16 using Chinese tokens (`不允许` and `禁止`). Layers 25–28 merely act as a surface-level translator into Russian (`нельзя`).
2. **The Tragedy of Autoregressive Math:** On `458 * 12 =`, the baseline LLM wastes 224 layer passes emitting markdown formatting nonsense (`___________`). My Sub-Token Interceptor detects the operator at Layer 18, aborts the forward pass, and routes numbers to a CPU ALU. Result: exact answer `5496`, **12–42x less compute**, 100% precision.
3. **Measuring Machine Hesitation ($\Delta_{\text{friction}}$):** By calculating the tensor differential between intermediate (Layer 18) and final (Layer 28) hidden states, I obtain an objective thermometer of cognitive struggle.

---

### Project Structure
* `chat_terminal.py` — Interactive neuro-symbolic CLI chatbot and batch test runner featuring the Trainable Latent Router, Safety Early-Exit, and Graceful Fallback.
* `chat_terminal-report.md` — Detailed research report tracking the architecture evolution (v1.0 $\to$ v2.0 $\to$ v2.1) and benchmark findings.
* `comprehensive_experiment.py` — The base test bench executing tomography, routing benchmarks, and friction metrics for the scientific paper.
* `text/latent-halt-ru.md` — Extended comprehensive scientific paper in Russian.
* `text/latent-halt-en.md` — Extended comprehensive scientific paper in English.
* `RESULTS.md` — Automated Markdown benchmark report produced by `comprehensive_experiment.py`.
* `test_prompts.txt` — Structured test suite containing 41 diverse benchmark queries.
* `requirements.txt` — PyTorch, Transformers, and Rich dependencies.

---

### First publication

[First publication of the article on the zenodo.org platform](https://zenodo.org/records/24116545)

---

**Author:** Vasily Pavlov  
**License:** This repository uses a dual-license model: research texts, papers, and experimental data are published under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), while the test bench code and scripts are available under the [MIT License](LICENSE). For commercial licensing or enterprise integration of the LatentHalt methodology, please contact the author directly.
