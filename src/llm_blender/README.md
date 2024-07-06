# LLM-Blender API Reference

## Table of Contents

- [LLM-Blender API Reference](#llm-blender-api-reference)
  - [Table of Contents](#table-of-contents)
  - [llm\_blender.llm\_blender\_ranker](#llm_blenderllm_blender_ranker)
    - [LLMBlenderRanker](#llmblenderranker)
      - [\_\_init\_\_](#__init__)
      - [warm\_up](#warm_up)
      - [run](#run)
  - [llm\_blender.llm\_blender\_evaluator](#llm_blenderllm_blender_evaluator)
    - [LLMBlenderEvaluator Objects](#llmblenderevaluator-objects)
      - [\_\_init\_\_](#__init__-1)
      - [prepare\_inputs](#prepare_inputs)
      - [compute\_mean\_scores](#compute_mean_scores)
      - [compute\_bleurt](#compute_bleurt)
      - [compute\_bartscore](#compute_bartscore)
      - [compute\_bertscore](#compute_bertscore)
      - [compute\_metrics](#compute_metrics)

<a id="llm_blender.llm_blender_ranker"></a>

## llm\_blender.llm\_blender\_ranker

<a id="llm_blender.llm_blender_ranker.LLMBlenderRanker"></a>

### LLMBlenderRanker

```python
@component
class LLMBlenderRanker()
```

Implements a LLM output ranking method with a pairwise reward model using the LLM Blender framework.

Usage Example:
```python
llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")
answers = [
    GeneratedAnswer(data="Paris is the capital of France.", query="What makes Paris unique?", documents=[]),
    GeneratedAnswer(
        data="The Eiffel Tower is an iconic landmark in Paris.", query="What makes Paris unique?", documents=[]
    ),
    GeneratedAnswer(data="Berlin is a beautiful city.", query="What makes Paris unique?", documents=[]),
]
output = llm_ranker.run(answers=answers)
ranked_answers = output["answers"]
print(ranked_answers)

# [
#     GeneratedAnswer(
#         data="The Eiffel Tower is an iconic landmark in Paris.",
#         query="What makes Paris unique?",
#         documents=[],
#         meta={},
#     ),
#     GeneratedAnswer(
#         data="Paris is the capital of France.", query="What makes Paris unique?", documents=[], meta={}
#     ),
#     GeneratedAnswer(data="Berlin is a beautiful city.", query="What makes Paris unique?", documents=[], meta={}),
# ]
```

<a id="llm_blender.llm_blender_ranker.LLMBlenderRanker.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model: str = "llm-blender/PairRM",
             device: str = "cpu",
             model_kwargs: Optional[Dict[str, Any]] = None)
```

Initialize a LLMBlenderRanker.

**Arguments**:

- `model`: Local path or name of the model in Hugging Face's model hub, such as ``'llm-blender/PairRM'``.
- `device`: The device on which the model is loaded. If `None`, the default device is automatically selected.
- `model_kwargs`: Keyword arguments to be passed to the LLM Blender model.

<a id="llm_blender.llm_blender_ranker.LLMBlenderRanker.warm_up"></a>

#### warm\_up

```python
def warm_up()
```

Warm up the pair ranking model used for scoring the answers.

<a id="llm_blender.llm_blender_ranker.LLMBlenderRanker.run"></a>

#### run

```python
@component.output_types(documents=List[GeneratedAnswer])
def run(answers: Variadic[List[GeneratedAnswer]])
```

Rank the output answers using the LLM Blender model.

**Arguments**:

- `answers`: A list of answers to be ranked.

**Returns**:

A list of ranked answers.


<a id="llm_blender.llm_blender_evaluator"></a>

## llm\_blender.llm\_blender\_evaluator

<a id="llm_blender.llm_blender_evaluator.LLMBlenderEvaluator"></a>

### LLMBlenderEvaluator Objects

```python
class LLMBlenderEvaluator()
```

Implements an evaluator for assessing the performance of predictions against labels using BLEURT, BARTScore, and
BERTScore.

<a id="llm_blender.llm_blender_evaluator.LLMBlenderEvaluator.__init__"></a>

#### \_\_init\_\_

```python
def __init__(preds, labels)
```

Evaluates the performance of predictions against labels using BLEURT, BARTScore, and BERTScore.

**Arguments**:

- `preds`: A list of predicted outputs.
- `labels`: A list of reference or target outputs.

<a id="llm_blender.llm_blender_evaluator.LLMBlenderEvaluator.prepare_inputs"></a>

#### prepare\_inputs

```python
def prepare_inputs()
```

Ensures that predictions and labels are formatted correctly before computing scores.

<a id="llm_blender.llm_blender_evaluator.LLMBlenderEvaluator.compute_mean_scores"></a>

#### compute\_mean\_scores

```python
def compute_mean_scores(scores) -> float
```

Computes the mean of a list of scores.

**Arguments**:

- `scores`: A list of scores.

**Returns**:

The mean score.

<a id="llm_blender.llm_blender_evaluator.LLMBlenderEvaluator.compute_bleurt"></a>

#### compute\_bleurt

```python
def compute_bleurt() -> float
```

Computes the BLEURT score for the provided predictions and labels.

**Returns**:

The BLEURT score.

<a id="llm_blender.llm_blender_evaluator.LLMBlenderEvaluator.compute_bartscore"></a>

#### compute\_bartscore

```python
def compute_bartscore() -> float
```

Computes the BARTScore for the provided predictions and labels.

**Returns**:

The BARTScore.

<a id="llm_blender.llm_blender_evaluator.LLMBlenderEvaluator.compute_bertscore"></a>

#### compute\_bertscore

```python
def compute_bertscore() -> float
```

Computes the BERTScore for the provided predictions and labels.

**Returns**:

The BERTScore.

<a id="llm_blender.llm_blender_evaluator.LLMBlenderEvaluator.compute_metrics"></a>

#### compute\_metrics

```python
def compute_metrics() -> Dict[str, float]
```

Computes BLEURT, BARTScore, and BERTScore for the provided predictions and labels.

**Returns**:

A dictionary containing the computed metrics.

