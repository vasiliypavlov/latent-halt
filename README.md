# LatentHalt: Sub-Token Neuro-Symbolic Interception & Latent Phase Mapping in LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers 4.40+](https://img.shields.io/badge/transformers-4.40+-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: CC BY-NC-SA 4.0 / MIT](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0%20%2F%20MIT-blue.svg)](#licensing)

> **Navigation / Навигация:**  
> • [Русская версия описания](#русская-версия)  
> • [English Description](#english-version)  
> • [Полная научная статья (RU)](text/latent-halt-ru.md) | [Full Scientific Paper (EN)](text/latent-halt-en.md)  
> • [Экспериментальные данные / Results (MD)](RESULTS.md)

---

<a name="русская-версия"></a>
## Русская версия

### Краткое академическое резюме
**LatentHalt** — это экспериментальный исследовательский проект, исследующий проблему вычислительной симметрии в декодерных трансформерах (LLM). В проекте реализован метод межслойного зондирования (Unembedding Probing) на модели `Qwen2.5-1.5B`, открывший трехфазную топологию инференса: абстрактные концепты кристаллизуются в скрытом представлении сети на слоях 18–24 (в мультиязычной интерлингве), после чего финальные слои тратятся исключительно на поверхностное синтаксическое декодирование. На основе фазового перехода разработан механизм **суб-токенного нейро-символического перехвата (Sub-Token Interception)**, позволяющий прерывать инференс на 18-м слое при распознавании алгоритмических задач и делегировать вычисления ядру фон Неймана. Это обеспечивает **12.44-кратное сокращение вычислительных затрат** при повышении точности с 0% до 100%. Также введена метрика **когнитивного трения ($\Delta_{\text{friction}}$)**, количественно оценивающая меру внутреннего конфликта модели при преодолении стереотипов контекста.

---

### Популярное введение: В поисках педали тормоза для ИИ

Современные большие языковые модели похожи на гоночный болид, у которого заклинило педаль газа. На какую бы задачу вы их ни натравили — попросили ли поставить точку в конце очевидной фразы или решить сложнейшую теорему — трансформер сжигает абсолютно одинаковое количество энергии и прогоняет данные через все свои 28, 80 или 120 слоев. В нейросети нет органа, который мог бы сказать: *«Довольно. Я поняла суть, дальше считать не нужно»*.

Опираясь на мою философскую работу *«Почему сознание не сводится к интеллектуальному агенту»* ([RU](https://github.com/vasiliypavlov/consciousness-not-an-agent/blob/main/text/consciousness-not-an-agent-ru.md), [EN](https://github.com/vasiliypavlov/consciousness-not-an-agent/blob/main/text/consciousness-not-an-agent-en.md)), я предположил: настоящий разум требует асимметрии — способности совершить «Акт Остановки».

Я создал «стетоскоп» для нейросети (метод Logit Lens с замером энтропии Шеннона) и заглянул внутрь модели `Qwen2.5-1.5B` прямо в момент ее работы. То, что я увидел, поражает:

1. **Модель мыслит интерлингвой задолго до конца:** На фразу об алкоголе и антибиотиках модель уже на 16-м слое формирует концепт строжайшего запрета в китайских токенах `不允许` и `禁止`, и лишь на 28-м слое переводит его в русское «нельзя».
2. **Арифметика через токены — безумие:** На задаче `458 * 12 =` стандартная LLM тратит 224 прохода по слоям и выдает визуальную кашу `___________`. Мой перехватчик ловит задачу на 18-м слое, тормозит сеть и отдает вычисление обычному Python. Итог: ответ `5496`, **ускорение в 12.44 раза** и 100% точность!
3. **Датчик сомнений сети ($\Delta_{\text{friction}}$):** Измеряя разницу между мыслями модели на 18-м слое и на 28-м слое, я научился численно видеть, когда модель испытывает сомнение или преодолевает стереотип (например, в задаче о нелетающем пингвине трение подскочило до рекордных 6.86 бит).

### Быстрый старт
```bash
# Клонирование репозитория
git clone https://github.com/vasiliypavlov/latent-halt.git
cd latent-halt

# Установка зависимостей
pip install -r requirements.txt

# Запуск полного исследовательского цикла и генерация отчета RESULTS.md
python comprehensive_experiment.py
```

---

<a name="english-version"></a>
## English Version

### Executive Summary
**LatentHalt** is an experimental AI research framework addressing the fundamental flaw of computational symmetry in decoder-only Transformers. By employing Unembedding Layer Probing on `Qwen2.5-1.5B`, this project empirically proves that semantic concept crystallization occurs within intermediate hidden layers (Layers 18–24) in an abstract, multilingual latent space, long before reaching the final layer. Leveraging this phase transition, I introduce **Sub-Token Neuro-Symbolic Interception**: an interrupt mechanism that halts inference at Layer 18 upon detecting formal algorithmic operations, delegating computation to a deterministic von Neumann engine (CPU/Python). This achieves a **12.44x computational speedup** with an accuracy jump from 0% to 100%. Furthermore, I introduce the **Cognitive Friction Index ($\Delta_{\text{friction}}$)**, measuring internal representation drift between early and late layers to quantify model doubt and cognitive conflict.

---

### Popular Overview: Giving LLMs a Brake Pedal

Modern Large Language Models act like supercars with stuck gas pedals. Whether predicting a trivial punctuation mark or solving a multi-step logical puzzle, a Transformer blindly executes every single one of its 28, 80, or 120 layers. It possesses no internal governor capable of saying: *"Stop. The answer is obvious; further computation is waste."*

Drawing from my philosophical treatise *"Why Consciousness Cannot Be Reduced to an Intelligent Agent"* ([RU](https://github.com/vasiliypavlov/consciousness-not-an-agent/blob/main/text/consciousness-not-an-agent-ru.md), [EN](https://github.com/vasiliypavlov/consciousness-not-an-agent/blob/main/text/consciousness-not-an-agent-en.md)), I posited that **true agency requires asymmetry — the capacity to execute an "Act of Stopping."**

Using an informational microscope (Logit Lens coupled with Shannon entropy probing), I peered into the hidden residual stream of `Qwen2.5-1.5B`:

1. **The Multilingual Concept Nucleus:** When asked about contraindications between antibiotics and alcohol, the network solidifies a categorical prohibition as early as Layer 16 using Chinese tokens (`不允许` and `禁止`). Layers 25–28 merely act as a surface-level translator into Russian (`нельзя`).
2. **The Tragedy of Autoregressive Math:** On `458 * 12 =`, the baseline LLM wastes 224 layer passes emitting markdown formatting nonsense (`___________`). My Sub-Token Interceptor detects the operator at Layer 18, aborts the forward pass, and routes numbers to a CPU ALU. Result: exact answer `5496`, **12.44x less compute**, 100% precision.
3. **Measuring Machine Hesitation ($\Delta_{\text{friction}}$):** By calculating the tensor differential between intermediate (Layer 18) and final (Layer 28) hidden states, I obtain an objective thermometer of cognitive struggle (peaking at 6.86 on counter-intuitive prompts like the flightless penguin problem).

### Project Structure
* `comprehensive_experiment.py` — The unified test bench executing tomography, routing benchmarks, and friction metrics.
* `text/latent-halt-ru.md` — Extended comprehensive scientific paper in Russian.
* `text/latent-halt-en.md` — Extended comprehensive scientific paper in English.
* `RESULTS.md` — Automated Markdown benchmark report produced by the test suite.
* `requirements.txt` — PyTorch and HuggingFace environment dependencies.

---
**Author:** Vasily Pavlov  
**License:** This repository uses a dual-license model: research texts, papers, and experimental data are published under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), while the test bench code and scripts are available under the [MIT License](LICENSE). For commercial licensing or enterprise integration of the LatentHalt methodology, please contact the author directly.
