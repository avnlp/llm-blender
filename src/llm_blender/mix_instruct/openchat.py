"""Evaluation of OpenChat-3.5 on the Mix-Instruct dataset

This script implements a pipeline to evaluate the OpenChat-3.5 model on the Mix-Instruct dataset
on the BLEURT, BARTScore, and BERTScore metrics.

The pipeline performs the following steps:
1. Loads the OpenChat-3.5 model using the LlamaCppGenerator from Haystack.
2. Generates responses for prompts and instructions from the Mix-Instruct dataset.
3. Evaluates the generated responses against reference outputs using the BLEURT, BARTScore, and BERTScore metrics.

This evaluation provides a baseline for the model's performance on the Mix-Instruct dataset.
"""

from datasets import load_dataset
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator

from llm_blender import LLMBlenderEvaluator


def generate_result(
    generator: LlamaCppGenerator,
    prompt: str = "",
    instruction: str = "",
) -> str:
    """
    Generate a response using the LlamaCppGenerator.

    The prompt and instruction are formatted to be compatible with the model.

    Args:
        generator (LlamaCppGenerator): The initialized LlamaCppGenerator object.
        prompt (str): The main text input for the model.
        instruction (str): Additional instructions for the model.

    Returns:
        str: The generated response from the model.
    """

    # Format prompt to be compatible with openchat-3.5-0106
    # This specific format is required for the model to distinguish between user input and expected output
    formatted_prompt = f"""GPT4 Correct User: {instruction}\n{prompt}<|end_of_turn|>GPT4 Correct Assistant:"""

    # Generate text using the LlamaCppGenerator
    result = generator.run(
        formatted_prompt,
        generation_kwargs={"max_tokens": 128, "temperature": 0.2},
    )

    # Extract the generated text from the result
    generated_answer = result["replies"][0]
    return generated_answer


# Define the path to the model weights
model = "models/openchat-3.5-0106.Q4_K_M.gguf"

# Initialize the LlamaCppGenerator with the specified model and context window size
generator = LlamaCppGenerator(
    model=model,
    n_ctx=256,
)

# Warm up the generator (loading the model into memory)
generator.warm_up()

# Load the dataset from the HuggingFace
dataset = load_dataset("llm-blender/mix-instruct", split="validation")

# Convert the dataset to a pandas DataFrame for easier manipulation
dataset = dataset.to_pandas()

# Generate results for each row in the dataset
# Apply the generate_result function to each row, using the 'input' and 'instruction' columns
# Store the results in the 'result' column
dataset.loc[:, "result"] = dataset.apply(
    lambda row: str(generate_result(generator=generator, prompt=row["input"], instruction=row["instruction"])), axis=1
)

# Save the generated texts to a CSV file
dataset.to_csv("output_openchat.csv", index=False)

# Initialize the LLMBlenderEvaluator with the generated results and the reference outputs
evaluator = LLMBlenderEvaluator(preds=dataset["result"], labels=dataset["output"])

# Compute various metrics to evaluate the generated results against the reference outputs
metrics = evaluator.compute_metrics()

# Print the computed metrics
print("BLEURT Score", metrics["bleurt"])
print("BARTSCORE Score", metrics["bartscore"])
print("BERTSCORE Score", metrics["bertscore"])
