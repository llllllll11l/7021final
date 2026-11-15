import argparse
import bisect
import math
import random
import time

from common import distribution, fingerprint, load_prepared, read_json, read_jsonl, write_json


def length_buckets(rows):
    lengths = sorted(row["length"] for row in rows)
    boundaries = sorted({lengths[round((len(lengths) - 1) * fraction / 4)] for fraction in (1, 2, 3)})
    buckets = [[] for _ in range(len(boundaries) + 1)]
    for row in rows:
        buckets[bisect.bisect_left(boundaries, row["length"])].append(row)
    buckets = [bucket for bucket in buckets if bucket]
    for index in range(len(buckets) - 1, 0, -1):
        if len(buckets[index]) < 8:
            buckets[index - 1].extend(buckets.pop(index))
    if len(buckets) > 1 and len(buckets[0]) < 8:
        buckets[1].extend(buckets.pop(0))
    return buckets


def pack(pool, quota, rng):
    shuffled = list(pool)
    rng.shuffle(shuffled)
    selected, used = [], 0
    for row in shuffled:
        if used + row["length"] <= quota:
            selected.append(row)
            used += row["length"]
    return selected


def relative_gap(first, second):
    return abs(first - second) / max(first, second, 1)


def select_pair(buckets, middle, quotas, seed):
    """仅依据长度重抽样随机组，使两组 token、回答 token 和样本数接近。"""
    rng = random.Random(seed)
    selected_random, selected_moderate = [], []
    for bucket, eligible, quota in zip(buckets, middle, quotas):
        moderate = pack(eligible, quota, rng)
        target = (sum(row["length"] for row in moderate), sum(row["answer_length"] for row in moderate), len(moderate))
        best, best_distance = None, float("inf")
        for _ in range(64):
            candidate = pack(bucket, quota, rng)
            actual = (sum(row["length"] for row in candidate), sum(row["answer_length"] for row in candidate), len(candidate))
            distance = sum(relative_gap(first, second) for first, second in zip(target, actual))
            if distance < best_distance:
                best, best_distance = candidate, distance
        selected_random.extend(best)
        selected_moderate.extend(moderate)
    return selected_random, selected_moderate


def select(config):
    root, metadata = load_prepared(config)
    rows = read_jsonl(root / "candidates.jsonl")
    score_metadata = read_json(root / "scores" / "metadata.json")
    if score_metadata["data_fingerprint"] != metadata["fingerprint"]:
        raise ValueError("损失缓存与当前数据不一致。")
    scores = read_jsonl(root / "scores" / "losses.jsonl")
    losses = {row["id"]: row["loss"] for row in scores}
    if len(scores) != len(losses) or set(losses) != {row["id"] for row in rows}:
        raise ValueError("损失缓存必须完整覆盖候选池，且无重复 ID。")
    if not all(math.isfinite(value) for value in losses.values()):
        raise ValueError("损失缓存含非有限值。")
    output = root / "selections"
    if output.exists():
        raise FileExistsError(f"{output} 已存在。")
    started = time.perf_counter()
    buckets = length_buckets(rows)
    middle = []
    for bucket in buckets:
        ranked = sorted(bucket, key=lambda row: (losses[row["id"]], row["id"]))
        middle.append(ranked[math.floor(len(ranked) * 0.25):math.ceil(len(ranked) * 0.75)])
    totals = [sum(row["length"] for row in bucket) for bucket in buckets]
    weights = [total / sum(totals) for total in totals]
    capacities = [sum(row["length"] for row in pool) for pool in middle]
    budget = min(config["token_budget"], math.floor(min(capacity / weight for capacity, weight in zip(capacities, weights))))
    quotas = [math.floor(budget * weight) for weight in weights]
    if any(quota < min(row["length"] for row in pool) for quota, pool in zip(quotas, middle)):
        raise ValueError("预算太小，无法在所有长度桶选择完整样本。")
    results = {}
    bucket_ids = {row["id"]: index for index, bucket in enumerate(buckets) for row in bucket}
    for seed in config["seeds"]:
        random_rows, moderate_rows = select_pair(buckets, middle, quotas, seed)
        summaries = [distribution(selected) for selected in (random_rows, moderate_rows)]
        gaps = {key: relative_gap(summaries[0][key], summaries[1][key]) for key in ("T", "T_y", "count")}
        if gaps["T"] > 0.01 or gaps["T_y"] > 0.05 or gaps["count"] > 0.05:
            raise ValueError(f"种子 {seed} 预算/长度匹配不足：{gaps}。检查长度分布或统一调整预算。")
        for method, selected, summary in zip(("random", "moderate"), (random_rows, moderate_rows), summaries):
            ids = [row["id"] for row in selected]
            summary["buckets"] = [distribution([row for row in selected if bucket_ids[row["id"]] == index])
                                  for index in range(len(buckets))]
            effective_batch = config["train"]["batch_size"] * config["train"]["gradient_accumulation_steps"]
            results[f"{method}_{seed}"] = {
                "method": method, "seed": seed, "ids": ids, "ids_fingerprint": fingerprint(ids),
                "data_fingerprint": metadata["fingerprint"], "requested_budget": config["token_budget"],
                "common_budget": budget, "bucket_quotas": quotas, "summary": summary,
                "pair_relative_gaps": gaps, "expected_update_steps": math.ceil(len(ids) / effective_batch),
            }
    output.mkdir()
    for name, result in results.items():
        write_json(output / f"{name}.json", result)
        print(name, {key: result["summary"][key] for key in ("count", "T", "T_y")})
    write_json(output / "metadata.json", {
        "data_fingerprint": metadata["fingerprint"], "common_budget": budget, "bucket_quotas": quotas,
        "middle_capacities": capacities, "seconds": time.perf_counter() - started,
        "buckets": [{"min_length": min(row["length"] for row in bucket),
                     "max_length": max(row["length"] for row in bucket), "count": len(bucket)} for bucket in buckets],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成长度分桶、token 配额匹配的 B/C 清单")
    parser.add_argument("--config", default="config.json")
    select(read_json(parser.parse_args().config))
