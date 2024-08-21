"""Evaluation of Qwen on the BillSum dataset

This script implements a pipeline to evaluate the Qwen model on the BillSum dataset
on the BLEURT, BARTScore, and BERTScore metrics.

The pipeline performs the following steps:
1. Loads the Qwen model using the LlamaCppGenerator from Haystack.
2. Generates responses for prompts and instructions from the BillSum dataset.
3. Evaluates the generated responses against reference outputs using the BLEURT, BARTScore, and BERTScore metrics.

This evaluation provides a baseline for the model's performance on the BillSum dataset.
"""

from datasets import load_dataset
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator

from llm_blender import LLMBlenderEvaluator


def construct_prompt(prompt: str = ""):
    """
    Construct a prompt with instructions for summarization.

    Args:
        prompt (str): The main text input for the model.

    Returns:
        str: The constructed for the model.
    """
    # Additional instructions for the model for Summarization
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
    """
    Generate a response using the LlamaCppGenerator.

    The prompt and instruction are formatted to be compatible with the model.

    Args:
        generator (LlamaCppGenerator): The initialized LlamaCppGenerator object.
        prompt (str): The main text input for the model.

    Returns:
        str: The generated response from the model.
    """
    # Format prompt to be compatible with qwen1.5-7b
    # This specific format is required for the model to distinguish between user input and expected output
    formatted_prompt = construct_prompt(prompt)

    # Generate text using the LlamaCppGenerator
    result = generator.run(
        formatted_prompt,
        generation_kwargs={"max_tokens": 500, "temperature": 0.1},
    )

    # Extract the generated text from the result
    generated_answer = result["replies"][0]
    return generated_answer


# Define the path to the model weights
model = "models/qwen1_5-7b-chat-q4_k_m.gguf"

# Initialize the LlamaCppGenerator with the specified model and context window size
generator = LlamaCppGenerator(
    model=model,
    n_ctx=256,
)

# Warm up the generator (loading the model into memory)
generator.warm_up()

# Load the dataset from the HuggingFace
dataset = load_dataset("billsum", split="test")

# Convert the dataset to a pandas DataFrame for easier manipulation
dataset = dataset.to_pandas()

# Generate results for each row in the dataset
# Apply the generate_result function to each row, using the 'text' column
# Store the results in the 'result' column
dataset.loc[:, "result"] = dataset.apply(
    lambda row: str(generate_result(generator=generator, prompt=row["text"])), axis=1
)

# Save the generated texts to a CSV file
dataset.to_csv("output_qwen.csv", index=False)

# Initialize the LLMBlenderEvaluator with the generated results and the reference outputs
evaluator = LLMBlenderEvaluator(preds=dataset["result"], labels=dataset["output"])

# Compute various metrics to evaluate the generated results against the reference outputs
metrics = evaluator.compute_metrics()

# Print the computed metrics
print("BLEURT Score", metrics["bleurt"])
print("BARTSCORE Score", metrics["bartscore"])
print("BERTSCORE Score", metrics["bertscore"])
