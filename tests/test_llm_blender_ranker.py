import pytest
from haystack import ComponentError
from haystack.dataclasses import GeneratedAnswer

from llm_blender import LLMBlenderRanker


class TestLLMBlenderRanker:
    def test_init(self):
        """
        Test that the LLMBlenderRanker is initialized correctly with default parameters.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")

        assert llm_ranker.model_name_or_path == "llm-blender/PairRM"
        assert llm_ranker.device == "cpu"
        assert llm_ranker.model_kwargs == {}

    def test_init_custom_parameters(self):
        """
        Test that the LLMBlenderRanker is initialized correctly with custom parameters.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM", device="cuda", model_kwargs={"cache_dir": "/models"})

        assert llm_ranker.model_name_or_path == "llm-blender/PairRM"
        assert llm_ranker.device == "cuda"
        assert llm_ranker.model_kwargs == {"cache_dir": "/models"}

    def test_run_without_warm_up(self):
        """
        Test that ranker loads the PairRanker model correctly during warm up.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")
        answers = [
            GeneratedAnswer(data="answer 1", query="query 1", documents=[]),
            GeneratedAnswer(data="answer 2", query="query 2", documents=[]),
        ]

        assert llm_ranker.model is None
        with pytest.raises(ComponentError, match="The component LLMBlenderRanker wasn't warmed up."):
            llm_ranker.run(answers=[answers])

        llm_ranker.warm_up()
        assert llm_ranker.model is not None

    def test_generation_of_inputs_and_candidates(self):
        """
        Test that the LLMBlenderRanker generates the correct inputs and candidates for a list of answers.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")
        answers = [
            GeneratedAnswer(data="answer 1", query="query 1", documents=[]),
            GeneratedAnswer(data="answer 2", query="query 2", documents=[]),
        ]
        inputs, candidates, meta = llm_ranker._generate_inputs_candidates([answers])

        assert inputs == ["query 1", "query 2"]
        assert candidates == [["answer 1"], ["answer 2"]]
        assert meta == [[{}], [{}]]

    def test_generation_of_inputs_and_candidates_for_same_input(self):
        """
        Test that the LLMBlenderRanker generates the correct inputs and candidates for a list of answers with the same
        input.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")
        answers_1 = [
            GeneratedAnswer(data="answer 1", query="query 1", documents=[]),
            GeneratedAnswer(data="answer 2", query="query 2", documents=[]),
        ]
        answers_2 = [
            GeneratedAnswer(data="answer 3", query="query 1", documents=[]),
            GeneratedAnswer(data="answer 4", query="query 2", documents=[]),
        ]

        inputs, candidates, meta = llm_ranker._generate_inputs_candidates([answers_1, answers_2])

        assert inputs == ["query 1", "query 2"]
        assert candidates == [["answer 1", "answer 3"], ["answer 2", "answer 4"]]
        assert meta == [[{}, {}], [{}, {}]]

    def test_ranking_candidates(self):
        """
        Test that the LLMBlenderRanker ranks the candidates correctly for a list of inputs and candidates.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")
        inputs = ["query 1", "query 2"]
        candidates = [["answer 1", "answer 2"], ["answer 3", "answer 4"]]
        ranks = [[1, 0], [0, 1]]
        meta = [[{"answer": "answer 1"}, {"answer": "answer 2"}], [{"answer": "answer 3"}, {"answer": "answer 4"}]]
        ranked_answers = llm_ranker._generate_answers_ranked_candidates(inputs, candidates, ranks, meta)

        assert ranked_answers == [
            GeneratedAnswer(data="answer 2", query="query 1", documents=[], meta={"answer": "answer 2"}),
            GeneratedAnswer(data="answer 1", query="query 1", documents=[], meta={"answer": "answer 1"}),
            GeneratedAnswer(data="answer 3", query="query 2", documents=[], meta={"answer": "answer 3"}),
            GeneratedAnswer(data="answer 4", query="query 2", documents=[], meta={"answer": "answer 4"}),
        ]

    def test_run(self):
        """
        Test that the LLMBlenderRanker ranks the answers correctly.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")
        llm_ranker.warm_up()

        answers_1 = [
            GeneratedAnswer(data="answer 1", query="query 1", documents=[]),
            GeneratedAnswer(data="answer 2", query="query 2", documents=[]),
        ]
        answers_2 = [
            GeneratedAnswer(data="answer 3", query="query 1", documents=[]),
            GeneratedAnswer(data="answer 4", query="query 2", documents=[]),
        ]

        output = llm_ranker.run(answers=[answers_1, answers_2])
        ranked_answers = output["answers"]

        assert ranked_answers == [
            GeneratedAnswer(data="answer 3", query="query 1", documents=[], meta={}),
            GeneratedAnswer(data="answer 1", query="query 1", documents=[], meta={}),
            GeneratedAnswer(data="answer 4", query="query 2", documents=[], meta={}),
            GeneratedAnswer(data="answer 2", query="query 2", documents=[], meta={}),
        ]

    def test_run_empty_answers(self):
        """
        Test that the LLMBlenderRanker handles an empty list of answers correctly.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")
        llm_ranker.warm_up()

        output = llm_ranker.run(answers=[])
        ranked_answers = output["answers"]

        assert ranked_answers == []
