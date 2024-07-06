"""
    Eval results will be continuously saved to ../../data/prepared/{dataset_name}/{set_name}/dataset.jsonl
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import psutil
import tabulate
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from common.evaluation import SUPPORTED_METRICS, overall_eval
from common.utils import (
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
    seed_everything,
    str2bool,
    tabulate_data_stats,
)


def save_prepared(
    dataset,
    set_name,
    data_dir,
):
    ds_path = Path(data_dir) / dataset / f"{set_name}_data.json"
    save_prepared_path = Path(data_dir) / dataset / f"{set_name}_data_prepared.json"
    assert ds_path.exists(), f"{ds_path} does not exist"
    ds_data = load_json(ds_path)
    # load candidates
    candidates_dir = Path(data_dir) / dataset / "candidates" / set_name
    decoding_method_dirs = [x for x in candidates_dir.iterdir() if x.is_dir()]
    for decoding_method_dir in decoding_method_dirs:
        decoding_method = decoding_method_dir.name
        # load candidates with eval scores
        candidate_eval_files = [
            x for x in decoding_method_dir.iterdir() if x.is_file() and x.suffixes[-2:] == [".eval", ".jsonl"]
        ]
        for candidate_eval_file in candidate_eval_files:
            model_name = Path(candidate_eval_file.stem).stem  # remove .eval.jsonl
            eval_candidates = load_jsonl(candidate_eval_file)
            eval_candidates = {x["id"]: x["candidates"] for x in eval_candidates}
            assert set(eval_candidates.keys()) == {
                x["id"] for x in ds_data
            }, f"candidate ids do not match for {dataset} {set_name} {decoding_method} {model_name}. That is, candidates are not generated for all examples"
            for example in ds_data:
                example_id = example["id"]
                if "candidates" not in example:
                    example["candidates"] = []
                for eval_candidate in eval_candidates[example_id]:
                    example["candidates"].append(
                        {
                            "decoding_method": decoding_method,
                            "model": model_name,
                            "text": eval_candidate["text"],
                            "scores": eval_candidate["scores"],
                        }
                    )
    print(f"Total no. of {set_name} examples in the aggregated dataset: {len(ds_data)}")
    save_json(ds_data, save_prepared_path)
    print(f"Saved aggregated {set_name} data to {save_prepared_path}")

    # sources = set([x["id"].split('/')[0] for x in ds_data])
    # for source in sources:
    #     tabulate_data_stats(ds_data, [source])
    tabulate_data_stats(ds_data)


def main(args):
    # seed
    seed_everything(args.seed)

    # prepare metrics
    if "rouge" in args.metrics:
        args.metrics.extend(["rouge1", "rouge2", "rougeL", "rougeLsum"])
        args.metrics.remove("rouge")
    metrics = args.metrics
    assert set(metrics).issubset(
        set(SUPPORTED_METRICS)
    ), f"Unsupported metrics: {set(SUPPORTED_METRICS) - set(metrics)}"

    for dataset in args.datasets:

        for set_name in args.sets:
            print(f"Evaluating dataset: {dataset} \t set: {set_name}")
            # get all the decoding method
            candidates_dir = Path(args.data_dir) / dataset / "candidates" / set_name
            decoding_methods = [f.name for f in candidates_dir.iterdir() if f.is_dir()]
            if len(decoding_methods) == 0:
                print(f"No candidates generated for {dataset}-{set_name}")
                continue
            for decoding_method in decoding_methods:
                print(f"Decoding method: {decoding_method}")
                candidate_files = [
                    f
                    for f in (candidates_dir / decoding_method).iterdir()
                    if f.is_file() and ".eval" not in f.suffixes and f.suffix == ".jsonl"
                ]
                if len(candidate_files) == 0:
                    print(f"No candidates generated for {dataset}-{set_name}-{decoding_method}")
                    continue
                for candidate_file in candidate_files:
                    print(f"Model name: {candidate_file.stem}")
                    # load candidates
                    candidate_eval_file = candidate_file.with_suffix(".eval.jsonl")
                    if not candidate_eval_file.exists() or args.overwrite:
                        print(f"Create a new eval file: {candidate_eval_file}")
                        candidates = load_jsonl(candidate_file)
                        eval_candidates = candidates
                    else:
                        print(f"Load existing eval file: {candidate_eval_file}")
                        eval_candidates = load_jsonl(candidate_eval_file)
                        # check completeness
                        candidates = load_jsonl(candidate_file)
                        eval_ids = {x["id"] for x in eval_candidates}
                        for cand in candidates:
                            if cand["id"] not in eval_ids:
                                eval_candidates.append(cand)
                        candidates_id_map = {x["id"]: x for x in candidates}
                        for eval_cand in eval_candidates:
                            eval_cand["candidates"][0]["text"] = candidates_id_map[eval_cand["id"]]["candidates"][0][
                                "text"
                            ]
                    # get the unevaluated candidates
                    un_eval_idxs = []
                    evaled_metrics = set(eval_candidates[0]["candidates"][0]["scores"].keys())
                    for i, item in enumerate(eval_candidates):
                        is_eval = True
                        for cand in item["candidates"]:
                            evaled_metrics = evaled_metrics.intersection(set(cand["scores"].keys()))
                            if not all(metric in cand["scores"] for metric in metrics):
                                is_eval = False
                                break
                        if not is_eval:
                            un_eval_idxs.append(i)
                    to_eval_metrics = set(metrics).difference(evaled_metrics)
                    print(f"Evaluated metrics: {evaled_metrics}")
                    print(f"To evaluate metrics: {to_eval_metrics}")
                    if len(un_eval_idxs) != 0:
                        print(
                            f"Existing eval file is incomplete. Evaluating {len(un_eval_idxs)}/{len(eval_candidates)} candidates"
                        )
                        un_eval_candidates = [eval_candidates[i] for i in un_eval_idxs]
                        DS = load_json(Path(args.data_dir) / dataset / f"{set_name}_data.json")
                        DS = {x["id"]: x for x in DS}
                        un_eval_targets = [DS[x["id"]]["output"] for x in un_eval_candidates]
                        pure_un_eval_candidates = [
                            [x["text"] for x in item["candidates"]] for item in un_eval_candidates
                        ]
                        # evaluate
                        scores = overall_eval(
                            pure_un_eval_candidates, un_eval_targets, to_eval_metrics, args.num_workers
                        )
                        assert set(scores.keys()) == set(to_eval_metrics)
                        # assign scores
                        for i, un_eval_candidate in enumerate(un_eval_candidates):
                            for metric in scores.keys():
                                metric_scores = scores[metric]
                                for j, cand in enumerate(un_eval_candidate["candidates"]):
                                    cand["scores"][metric] = metric_scores[i][j]
                        # save
                        save_jsonl(eval_candidates, candidate_eval_file)
                        print(f"Evaluation results saved to {candidate_eval_file}")
                    else:
                        save_jsonl(eval_candidates, candidate_eval_file)
                        print("All candidates have already been evaluated, skip")

                    # Report the evaluation results
                    for metric in metrics:
                        scores = [[x["scores"][metric] for x in item["candidates"]] for item in eval_candidates]
                        scores = np.array(scores)
                        print(f"Metric: {metric}")
                        print(f"Average Min Score: {scores.min(axis=1).mean():.3f}")
                        print(f"Average Max Score: {scores.max(axis=1).mean():.3f}")
                        print(f"Average Mean Score: {scores.mean(axis=1).mean():.3f}")
                        print(f"Average Default Top-1 Score: {scores[:, 0].mean():.3f}")
                        print(f"Average Default Bottom-1 Score: {scores[:, -1].mean():.3f}")
        print(f"Done for dataset: {dataset}")

        if args.save_prepared:
            for set_name in args.sets:
                save_prepared(dataset, set_name, args.data_dir)

    print(f"Done for all datasets: {args.datasets}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../data")
    parser.add_argument("--dataset", type=str, default="cnndm")
    parser.add_argument("--set", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument(
        "--save_prepared",
        type=str2bool,
        default=True,
        help="aggregate the candidates and save them to a single file for each dataset and set",
    )
    # metrics
    parser.add_argument(
        "--metrics",
        type=str,
        default="rouge,bleu",
        help="metrics to compute, support rouge, bleu, bleurt, cider, spice, bleu4, bertscore, gptscore",
    )
    args = parser.parse_args()
    args.metrics = args.metrics.split(",")
    args.datasets = args.dataset.split(",")
    args.sets = args.set.split(",")
    print(args)
    main(args)
