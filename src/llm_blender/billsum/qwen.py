from datasets import load_dataset
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator

from llm_blender import LLMBlenderEvaluator


def construct_prompt(prompt=""):
    prompt_with_instruction = (
        """ Provide a comprehensive summary of the given text. """
        """The summary should cover all the key points and main ideas presented in the original text, """
        f"""while also condensing the information into a concise and easy-to-understand format.:\n{prompt}"""
    )
    # Format prompt to be compatible with qwen1.5-7b
    formatted_prompt = f"""<|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        {prompt_with_instruction}<|im_end|>
        <|im_start|>assistant"""

    return formatted_prompt


def generate_result(
    generator: LlamaCppGenerator,
    prompt: str = "",
) -> str:

    # Format prompt to be compatible with qwen1.5-7b
    formatted_prompt = construct_prompt(prompt)

    # Generate text
    result = generator.run(
        formatted_prompt,
        generation_kwargs={"max_tokens": 500, "temperature": 0.1},
    )
    generated_answer = result["replies"][0]
    return generated_answer


model = "models/qwen1_5-7b-chat-q4_k_m.gguf"
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
dataset.to_csv("output_openchat.csv", index=False)


evaluator = LLMBlenderEvaluator(preds=dataset["result"], labels=dataset["output"])
metrics = evaluator.compute_metrics()

print("BLEURT Score", metrics["bleurt"])
print("BARTSCORE Score", metrics["bartscore"])
print("BERTSCORE Score", metrics["bertscore"])
