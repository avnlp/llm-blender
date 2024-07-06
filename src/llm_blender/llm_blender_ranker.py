import logging
from collections import defaultdict
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

from haystack import ComponentError, GeneratedAnswer, component
from haystack.core.component.types import Variadic

from llm_blender.llm_blender_utils import Blender

logger = logging.getLogger(__name__)


@component
class LLMBlenderRanker:
    """
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
    """

    def __init__(
        self,
        model: str = "llm-blender/PairRM",
        device: str = "cpu",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a LLMBlenderRanker.

        :param model:
            Local path or name of the model in Hugging Face's model hub, such as ``'llm-blender/PairRM'``.
        :param device:
            The device on which the model is loaded. If `None`, the default device is automatically selected.
        :param model_kwargs:
            Keyword arguments to be passed to the LLM Blender model.
        """

        self.model_name_or_path = model
        self.device = device
        self.model = None
        self.model_kwargs = model_kwargs or {}

    def warm_up(self):
        """
        Warm up the pair ranking model used for scoring the answers.
        """
        if self.model is None:
            blender = Blender()
            blender.loadranker(self.model_name_or_path, device=self.device, **self.model_kwargs)
            self.model = blender

    def _generate_inputs_candidates(
        self,
        answers_list: List[List[GeneratedAnswer]],
    ) -> Tuple[List[str], List[List[str]], List[List[Dict[str, Any]]]]:
        """
        Generate candidates for each query by combining all answers where the query (input) is the same.

        If the length of the candidate list is less than the length of the smallest candidate list among all queries,
        the candidate list is trimmed to match the length of the smallest candidate list.

        :param answers_list:
            A list of lists of answers.
        :return:
            A list of inputs, a list of lists of candidates, and a list of lists of metadata.
        """
        inputs_candidates_meta = defaultdict(list)
        for answers in answers_list:
            for generated_answer in answers:
                inputs_candidates_meta[generated_answer.query].append((generated_answer.data, generated_answer.meta))

        # Find the smallest length among all candidate lists for each query
        lengths = {query: len(candidates_list) for query, candidates_list in inputs_candidates_meta.items()}
        min_length = min(lengths.values())

        # Trim each candidate list to match the smallest length
        for query, candidates_list in inputs_candidates_meta.items():
            inputs_candidates_meta[query] = list(candidates_list[:min_length])

        inputs = list(inputs_candidates_meta.keys())
        candidates_meta = list(inputs_candidates_meta.values())
        candidates = [[data for data, _ in lst] for lst in candidates_meta]
        meta = [[meta for _, meta in lst] for lst in candidates_meta]

        return inputs, candidates, meta

    def _generate_answers_ranked_candidates(
        self,
        inputs: List[str],
        candidates: List[List[str]],
        ranks_list: List[List[int]],
        meta: List[List[Dict[str, str]]],
    ) -> List[GeneratedAnswer]:
        """
        Generate the ranked candidates for each input using the ranks from the Pair Ranker model.

        :param inputs:
            A list of inputs.
        :param candidates:
            A list of lists of candidates.
        :param ranks_list:
            A list of lists of ranks.
        :param meta:
            A list of lists of metadata.
        :return:
             A list of Generated Answers.
        """
        # Create a dictionary to store the ranked candidates for each input
        ranked_candidates = {}

        # Iterate through the inputs and ranks
        for i in range(len(inputs)):
            input_str = inputs[i]
            ranks = ranks_list[i]
            candidates_for_input = candidates[i]
            meta_for_input = meta[i]

            # Store the candidates, their ranks, and their metadata in a dictionary
            ranked_candidates[input_str] = list(zip(candidates_for_input, ranks, meta_for_input))

        # Sort the dictionary based on the ranks and extract the sorted candidates
        sorted_candidates = {key: sorted(values, key=lambda item: item[1]) for key, values in ranked_candidates.items()}

        # Convert the sorted candidates to a list of Generated Answers for each input
        ranked_generated_answers = [
            [
                GeneratedAnswer(query=input_str, data=candidate, documents=[], meta=meta)
                for candidate, _, meta in sorted_candidates[input_str]
            ]
            for input_str in inputs
        ]

        ranked_generated_answers = list(chain.from_iterable(ranked_generated_answers))

        return ranked_generated_answers

    @component.output_types(documents=List[GeneratedAnswer])
    def run(self, answers: Variadic[List[GeneratedAnswer]]):
        """
        Rank the output answers using the LLM Blender model.

        :param answers:
            A list of answers to be ranked.
        :return:
            A list of ranked answers.
        """

        if not answers:
            return {"answers": []}

        if self.model is None:
            msg = "The component LLMBlenderRanker wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            raise ComponentError(msg)

        inputs, candidates, meta = self._generate_inputs_candidates(answers)
        ranks = self.model.rank(inputs, candidates)
        ranked_answers = self._generate_answers_ranked_candidates(inputs, candidates, ranks, meta)

        return {"answers": ranked_answers}
