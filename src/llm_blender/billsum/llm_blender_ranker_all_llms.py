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

openchat_prompt_template = (
    """GPT4 Correct User: Provide a comprehensive summary of the given text. The summary """
    """should cover all the key points and main ideas presented in the original text, while also condensing the """
    """information into a concise and easy-to-understand format.: \n {{ prompt }}GPT4 Correct Assistant:"""
)

openhermes_prompt_template = (
    """<|im_start|>system\nProvide a comprehensive summary of the given text. The summary should cover all the key """
    """points and main ideas presented in the original text, while also condensing the information into a concise and"""
    """easy-to-understand format.:<|im_end|><|im_start|>user\n{{ prompt }}<|im_end|>\n<|im_start|>assistant"""
)

solar_prompt_template = (
    """### User:  Provide a comprehensive summary of the given text. The summary should cover """
    """all the key points and main ideas presented in the original text, while also condensing the information """
    """into a concise and easy-to-understand format.:\n{{ prompt }} ### Assistant:"""
)

qwen_prompt_template = (
    """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>Provide a comprehensive summary of """
    """the given text. The summary should cover all the key points and main ideas presented in the original text, """
    """while also condensing the information into a concise and easy-to-understand format.:: {{ prompt }}<|im_end|>"""
    """<|im_start|>assistant"""
)

mistral_prompt_template = (
    """<s>[INST] Provide a comprehensive summary of the given text. The summary should cover """
    """all the key points and main ideas presented in the original text, while also condensing the information into """
    """a concise and easy-to-understand format.: {{ prompt }} [/INST] """
)

llama_prompt_builder = PromptBuilder(template=llama_prompt_template)
phi_prompt_builder = PromptBuilder(template=phi_prompt_template)
openchat_prompt_builder = PromptBuilder(template=openchat_prompt_template)
openhermes_prompt_builder = PromptBuilder(template=openhermes_prompt_template)
solar_prompt_builder = PromptBuilder(template=solar_prompt_template)
qwen_prompt_builder = PromptBuilder(template=qwen_prompt_template)
mistral_prompt_builder = PromptBuilder(template=mistral_prompt_template)

model_params = {"n_ctx": 256, "generation_kwargs": {"max_tokens": 500, "temperature": 0.1}}

llama_model = LlamaCppGenerator(model="models/meta-llama-3-8b-instruct.Q4_K_M.gguf", **model_params)
phi_model = LlamaCppGenerator(model="models/phi-3-mini-4k-instruct.Q4_K_M.gguf", **model_params)
openchat_model = LlamaCppGenerator(model="models/openchat-3.5-0106.Q4_K_M.gguf", **model_params)
openhermes_model = LlamaCppGenerator(model="models/openhermes-2.5-mistral-7b.Q4_K_M.gguf", **model_params)
solar_model = LlamaCppGenerator(model="models/solar-7b-Q4_K_M.gguf", **model_params)
qwen_model = LlamaCppGenerator(model="models/qwen1_5-7b-chat-Q4_K_M.gguf", **model_params)
mistral_model = LlamaCppGenerator(model="models/mistral-7b-Q4_K_M.gguf", **model_params)

llm_blender_ranker = LLMBlenderRanker(model="llm-blender/PairRM", device="cpu")


blender_pipeline = Pipeline()

blender_pipeline.add_component(instance=llama_prompt_builder, name="llama_prompt_builder")
blender_pipeline.add_component(instance=llama_model, name="llama_model")

blender_pipeline.add_component(instance=phi_prompt_builder, name="phi_prompt_builder")
blender_pipeline.add_component(instance=phi_model, name="phi_model")

blender_pipeline.add_component(instance=openchat_prompt_builder, name="openchat_prompt_builder")
blender_pipeline.add_component(instance=openchat_model, name="openchat_model")

blender_pipeline.add_component(instance=openhermes_prompt_builder, name="openhermes_prompt_builder")
blender_pipeline.add_component(instance=openhermes_model, name="openhermes_model")

blender_pipeline.add_component(instance=solar_prompt_builder, name="solar_prompt_builder")
blender_pipeline.add_component(instance=solar_model, name="solar_model")

blender_pipeline.add_component(instance=qwen_prompt_builder, name="qwen_prompt_builder")
blender_pipeline.add_component(instance=qwen_model, name="qwen_model")

blender_pipeline.add_component(instance=mistral_prompt_builder, name="mistral_prompt_builder")
blender_pipeline.add_component(instance=mistral_model, name="mistral_model")

blender_pipeline.add_component(instance=llm_blender_ranker, name="llm_blender_ranker")

blender_pipeline.connect("llama_prompt_builder", "llama_model")
blender_pipeline.connect("phi_prompt_builder", "phi_model")
blender_pipeline.connect("openchat_prompt_builder", "openchat_model")
blender_pipeline.connect("openhermes_prompt_builder", "openhermes_model")
blender_pipeline.connect("solar_prompt_builder", "solar_model")
blender_pipeline.connect("qwen_prompt_builder", "qwen_model")
blender_pipeline.connect("mistral_prompt_builder", "mistral_model")

blender_pipeline.connect("llama_model", "llm_blender_ranker")
blender_pipeline.connect("phi_model", "llm_blender_ranker")
blender_pipeline.connect("openchat_model", "llm_blender_ranker")
blender_pipeline.connect("openhermes_model", "llm_blender_ranker")
blender_pipeline.connect("solar_model", "llm_blender_ranker")
blender_pipeline.connect("qwen_model", "llm_blender_ranker")
blender_pipeline.connect("mistral_model", "llm_blender_ranker")

generated_answers_labels = []
for row in dataset:
    prompt = row["input"]
    label = row["output"]
    output = blender_pipeline.run(
        {
            {"llama_prompt_builder": {"prompt": prompt}},
            {"phi_prompt_builder": {"prompt": prompt}},
            {"openchat_prompt_builder": {"prompt": prompt}},
            {"openhermes_prompt_builder": {"prompt": prompt}},
            {"solar_prompt_builder": {"prompt": prompt}},
            {"qwen_prompt_builder": {"prompt": prompt}},
            {"mistral_prompt_builder": {"prompt": prompt}},
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
