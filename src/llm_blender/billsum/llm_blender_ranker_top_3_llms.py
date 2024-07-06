from datasets import load_dataset
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator

from llm_blender import LLMBlenderEvaluator, LLMBlenderRanker

dataset = load_dataset("billsum", split="test")

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

llama_prompt_builder = PromptBuilder(template=llama_prompt_template)
phi_prompt_builder = PromptBuilder(template=phi_prompt_template)
mistral_prompt_builder = PromptBuilder(template=mistral_prompt_template)

model_params = {"n_ctx": 256, "generation_kwargs": {"max_tokens": 500, "temperature": 0.1}}

llama_model = LlamaCppGenerator(model="models/meta-llama-3-8b-instruct.Q4_K_M.gguf", **model_params)
phi_model = LlamaCppGenerator(model="models/phi-3-mini-4k-instruct.Q4_K_M.gguf", **model_params)
mistral_model = LlamaCppGenerator(model="models/mistral-7b-Q4_K_M.gguf", **model_params)

llm_blender_ranker = LLMBlenderRanker(model="llm-blender/PairRM", device="cpu")

blender_pipeline = Pipeline()

blender_pipeline.add_component(instance=llama_prompt_builder, name="llama_prompt_builder")
blender_pipeline.add_component(instance=llama_model, name="llama_model")

blender_pipeline.add_component(instance=phi_prompt_builder, name="phi_prompt_builder")
blender_pipeline.add_component(instance=phi_model, name="phi_model")

blender_pipeline.add_component(instance=mistral_prompt_builder, name="mistral_prompt_builder")
blender_pipeline.add_component(instance=mistral_model, name="mistral_model")

blender_pipeline.add_component(instance=llm_blender_ranker, name="llm_blender_ranker")

blender_pipeline.connect("llama_prompt_builder", "llama_model")
blender_pipeline.connect("phi_prompt_builder", "phi_model")
blender_pipeline.connect("mistral_prompt_builder", "mistral_model")

blender_pipeline.connect("llama_model", "llm_blender_ranker")
blender_pipeline.connect("phi_model", "llm_blender_ranker")
blender_pipeline.connect("mistral_model", "llm_blender_ranker")

generated_answers_labels = []
for row in dataset:
    prompt = row["input"]
    label = row["output"]
    output = blender_pipeline.run(
        {
            {
                {"llama_prompt_builder": {"prompt": prompt}},
                {"phi_prompt_builder": {"prompt": prompt}},
                {"mistral_prompt_builder": {"prompt": prompt}},
            }
        }
    )
    generated_answers_labels.append((output["answers"], label))

preds = []
labels = []
for ranked_answers, label in generated_answers_labels:
    # Use top ranked output as the answer
    preds.append(ranked_answers[0].data)
    labels.append(label)

evaluator = LLMBlenderEvaluator(preds=preds, labels=labels)
metrics = evaluator.compute_metrics()

print("BLEURT Score", metrics["bleurt"])
print("BARTSCORE Score", metrics["bartscore"])
print("BERTSCORE Score", metrics["bertscore"])
