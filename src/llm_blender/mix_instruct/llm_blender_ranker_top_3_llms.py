"""
Evaluation of ensemble of best performing LLMs on the Mix-Instruct dataset using LLM Blender

This script implements a pipeline to ensemble multiple language models on the Mix-Instruct dataset. The pipeline is
 evaluated on the BLEURT, BARTScore, and BERTScore metrics.

The pipeline performs the following steps:
1. Loads 3 top performing LLMs: LLaMA, Phi and  Mistral.
2. Builds prompts for each model using specific templates.
3. Generates responses for prompts from the Mix-Instruct dataset using each model.
4. Ranks the generated responses from all the models using the LLM Blender Ranker.
5. Evaluates the top-ranked response against reference outputs using multiple metrics.

The evaluation showcases the effectiveness of the ensembling approach using LLM-Blender with diverse LLMs.
"""

from datasets import load_dataset
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator

from llm_blender import LLMBlenderEvaluator, LLMBlenderRanker

# Load the Mix-Instruct dataset
dataset = load_dataset("llm-blender/mix-instruct", split="validation")

# Define prompt templates for each model
llama_prompt_template = (
    """<|begin_of_text|><|start_header_id|>user<|end_header_id|> Provide a comprehensive summary of the given """
    """text. The summary should cover all the key points and main ideas presented in the original text, while """
    """also condensing the information into a concise and easy-to-understand format. {{ prompt }}<|eot_id|>"""
    """<|start_header_id|>assistant<|end_header_id|>"""
)

phi_prompt_template = (
    """<|user|>\nProvide a comprehensive summary of the given text. The summary should cover all """
    """the key points and main ideas presented in the original text, while also condensing the information into a """
    """concise and easy-to-understand format. {prompt} <|end|>\n<|assistant|>"""
)

mistral_prompt_template = (
    """<s>[INST] Provide a comprehensive summary of the given text. The summary should cover """
    """all the key points and main ideas presented in the original text, while also condensing the information into """
    """a concise and easy-to-understand format.: {{ prompt }} [/INST] """
)

# Initialize PromptBuilder for each model
llama_prompt_builder = PromptBuilder(template=llama_prompt_template)
phi_prompt_builder = PromptBuilder(template=phi_prompt_template)
mistral_prompt_builder = PromptBuilder(template=mistral_prompt_template)

# Define model and generation parameters for all models
model_params = {"n_ctx": 256, "generation_kwargs": {"max_tokens": 128, "temperature": 0.2}}

# Initialize LlamaCppGenerator for each model
llama_model = LlamaCppGenerator(model="models/meta-llama-3-8b-instruct.Q4_K_M.gguf", **model_params)
phi_model = LlamaCppGenerator(model="models/phi-3-mini-4k-instruct.Q4_K_M.gguf", **model_params)
mistral_model = LlamaCppGenerator(model="models/mistral-7b-Q4_K_M.gguf", **model_params)

# Initialize LLMBlenderRanker to ensemble multiple models
llm_blender_ranker = LLMBlenderRanker(model="llm-blender/PairRM", device="cpu")

# Create the main pipeline
blender_pipeline = Pipeline()

# Add components to the pipeline
blender_pipeline.add_component(instance=llama_prompt_builder, name="llama_prompt_builder")
blender_pipeline.add_component(instance=llama_model, name="llama_model")

blender_pipeline.add_component(instance=phi_prompt_builder, name="phi_prompt_builder")
blender_pipeline.add_component(instance=phi_model, name="phi_model")

blender_pipeline.add_component(instance=mistral_prompt_builder, name="mistral_prompt_builder")
blender_pipeline.add_component(instance=mistral_model, name="mistral_model")

blender_pipeline.add_component(instance=llm_blender_ranker, name="llm_blender_ranker")

# Connect components in the pipeline
# Connect the prompt builders to the respective model
blender_pipeline.connect("llama_prompt_builder", "llama_model")
blender_pipeline.connect("phi_prompt_builder", "phi_model")
blender_pipeline.connect("mistral_prompt_builder", "mistral_model")

# Connect all the models to the LLMBlenderRanker for ensembling
blender_pipeline.connect("llama_model", "llm_blender_ranker")
blender_pipeline.connect("phi_model", "llm_blender_ranker")
blender_pipeline.connect("mistral_model", "llm_blender_ranker")

# Process the dataset and generate answers
generated_answers_labels = []
for row in dataset:
    instruction = row["instruction"]
    prompt = row["input"]
    label = row["output"]

    # Run the pipeline for each input
    output = blender_pipeline.run(
        {
            {"llama_prompt_builder": {"instruction": instruction, "prompt": prompt}},
            {"phi_prompt_builder": {"instruction": instruction, "prompt": prompt}},
            {"mistral_prompt_builder": {"instruction": instruction, "prompt": prompt}},
        }
    )
    generated_answers_labels.append((output["answers"], label))

# Prepare data for evaluation
preds = []
labels = []
for ranked_answers, label in generated_answers_labels:
    # Use top ranked output as the answer
    preds.append(ranked_answers[0].data)
    labels.append(label)

# Initialize the LLMBlenderEvaluator with the generated results and the reference outputs
evaluator = LLMBlenderEvaluator(preds=preds, labels=labels)

# Compute various metrics to evaluate the generated results against the reference outputs
metrics = evaluator.compute_metrics()

# Print the evaluation metrics
print("BLEURT Score", metrics["bleurt"])
print("BARTSCORE Score", metrics["bartscore"])
print("BERTSCORE Score", metrics["bertscore"])
