from datasets import load_dataset
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator

from llm_blender import LLMBlenderEvaluator


def generate_result(
    generator: LlamaCppGenerator,
    prompt: str = "",
) -> str:

    instruction = (
        """ Provide a comprehensive summary of the given text. """
        """The summary should cover all the key points and main ideas presented in the original text, """
        """while also condensing the information into a concise and easy-to-understand format."""
    )

    # Format prompt to be compatible with meta-llama-3-8b-instruct
    formatted_prompt = (
        """<|begin_of_text|><|start_header_id|>user<|end_header_id|> """
        f"""{instruction} {prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    )

    # Generate text
    result = generator.run(
        formatted_prompt,
        generation_kwargs={"max_tokens": 500, "temperature": 0.1},
    )
    generated_answer = result["replies"][0]
    return generated_answer


model = "meta-llama-3-8b-instruct.Q4_K_M.gguf"
generator = LlamaCppGenerator(
    model=model,
    n_ctx=256,
)
generator.warm_up()

dataset = load_dataset("billsum", split="test")
dataset = dataset.to_pandas()
dataset.loc[:, "result"] = dataset.apply(
    lambda row: str(generate_result(generator=generator, prompt=row["text"])), axis=1
)
dataset.to_csv("output_llama.csv", index=False)


evaluator = LLMBlenderEvaluator(preds=dataset["result"], labels=dataset["output"])
metrics = evaluator.compute_metrics()

print("BLEURT Score", metrics["bleurt"])
print("BARTSCORE Score", metrics["bartscore"])
print("BERTSCORE Score", metrics["bertscore"])
