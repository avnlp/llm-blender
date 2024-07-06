from datasets import load_dataset
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator

from llm_blender import LLMBlenderEvaluator


def generate_result(
    generator: LlamaCppGenerator,
    prompt: str = "",
    instruction: str = "",
) -> str:

    # Format prompt to be compatible with starling-lm-7b-alpha
    formatted_prompt = f"""GPT4 Correct User: {instruction}\n{prompt}<|end_of_turn|>GPT4 Correct Assistant:"""

    # Generate text
    result = generator.run(
        formatted_prompt,
        generation_kwargs={"max_tokens": 128, "temperature": 0.2},
    )
    generated_answer = result["replies"][0]
    return generated_answer


model = "models/starling-lm-7b-alpha.Q4_K_M.gguf"
generator = LlamaCppGenerator(
    model=model,
    n_ctx=256,
)
generator.warm_up()

dataset = load_dataset("llm-blender/mix-instruct", split="validation")
dataset = dataset.to_pandas()
dataset.loc[:, "result"] = dataset.apply(
    lambda row: str(generate_result(generator=generator, prompt=row["input"], instruction=row["instruction"])), axis=1
)
dataset.to_csv("output_starling.csv", index=False)


evaluator = LLMBlenderEvaluator(preds=dataset["result"], labels=dataset["output"])
metrics = evaluator.compute_metrics()

print("BLEURT Score", metrics["bleurt"])
print("BARTSCORE Score", metrics["bartscore"])
print("BERTSCORE Score", metrics["bertscore"])
