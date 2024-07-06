from itertools import chain
from typing import Dict

from llm_blender.llm_blender_utils.common.evaluation import eval_bartscore, eval_bertscore, eval_bleurt


class LLMBlenderEvaluator:
    """
    Implements an evaluator for assessing the performance of predictions against labels using BLEURT, BARTScore, and
    BERTScore.
    """

    def __init__(self, preds, labels):
        """
        Evaluates the performance of predictions against labels using BLEURT, BARTScore, and BERTScore.

        :param preds: A list of predicted outputs.
        :param labels: A list of reference or target outputs.
        """
        if not isinstance(preds, list) or not isinstance(labels, list):
            err_msg = "Both preds and labels must be lists."
            raise ValueError(err_msg)
        if len(preds) != len(labels):
            err_msg = f"The length of preds and labels must be the same. Got {len(preds)} and {len(labels)}."
            raise ValueError(err_msg)
        self.preds = preds
        self.labels = labels
        self.bleurt = None
        self.bartscore = None
        self.bertscore = None

    def prepare_inputs(self):
        """
        Ensures that predictions and labels are formatted correctly before computing scores.
        """
        if not isinstance(self.preds[0], list):
            self.preds = [[pred] for pred in self.preds]
        if not isinstance(self.labels[0], list):
            self.labels = [[label] for label in self.labels]

    def compute_mean_scores(self, scores) -> float:
        """
        Computes the mean of a list of scores.

        :param scores: A list of scores.
        :return: The mean score.
        """
        return sum(scores) / len(scores)

    def compute_bleurt(self) -> float:
        """
        Computes the BLEURT score for the provided predictions and labels.

        :return: The BLEURT score.
        """
        self.prepare_inputs()
        bleurt_scores = eval_bleurt(self.preds, self.labels)
        bleurt_scores = list(chain.from_iterable(bleurt_scores))
        self.bleurt = self.compute_mean_scores(bleurt_scores)
        return self.bleurt

    def compute_bartscore(self) -> float:
        """
        Computes the BARTScore for the provided predictions and labels.

        :return: The BARTScore.
        """
        self.prepare_inputs()
        bartscore_scores = eval_bartscore(self.preds, self.labels)
        bartscore_scores = list(chain.from_iterable(bartscore_scores))
        self.bartscore = self.compute_mean_scores(bartscore_scores)
        return self.bartscore

    def compute_bertscore(self) -> float:
        """
        Computes the BERTScore for the provided predictions and labels.

        :return: The BERTScore.
        """
        self.prepare_inputs()
        bertscore_scores = eval_bertscore(self.preds, self.labels)
        bertscore_scores = list(chain.from_iterable(bertscore_scores))
        self.bertscore = self.compute_mean_scores(bertscore_scores)
        return self.bertscore

    def compute_metrics(self) -> Dict[str, float]:
        """
        Computes BLEURT, BARTScore, and BERTScore for the provided predictions and labels.

        :return: A dictionary containing the computed metrics.
        """
        self.prepare_inputs()
        bleurt = self.compute_bleurt()
        bartscore = self.compute_bartscore()
        bertscore = self.compute_bertscore()
        return {"bleurt": bleurt, "bartscore": bartscore, "bertscore": bertscore}
