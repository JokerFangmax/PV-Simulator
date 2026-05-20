#!/usr/bin/env python3
import argparse
import json
import os
import tarfile
from collections import Counter
from pathlib import Path


def resolve_webdataset_root(data_root: str) -> Path:
    root = Path(data_root)
    candidates = [root, root / "webdataset"]
    for candidate in candidates:
        if (
            (candidate / "dataset_manifest.json").is_file()
            and (candidate / "manifest.jsonl").is_file()
            and (candidate / "shards").is_dir()
        ):
            return candidate
    raise FileNotFoundError(f"Could not find webdataset root under {data_root}")


def iter_manifest_records(webdataset_root: Path):
    manifest_path = webdataset_root / "manifest.jsonl"
    with manifest_path.open("r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def count_instances_from_record(record: dict, shard_handles: dict, webdataset_root: Path) -> int:
    shard_path = webdataset_root / "shards" / record["shard_path"]
    shard_key = str(shard_path)
    outer_tf = shard_handles.get(shard_key)
    if outer_tf is None:
        outer_tf = tarfile.open(shard_path, "r:")
        shard_handles[shard_key] = outer_tf

    payload_name = f'{record["sample_id"]}.payload.tar'
    payload_member = outer_tf.getmember(payload_name)
    with outer_tf.extractfile(payload_member) as payload_fobj:
        with tarfile.open(fileobj=payload_fobj, mode="r:") as inner_tf:
            with inner_tf.extractfile("metadata.json") as meta_f:
                meta = json.load(meta_f)
    return len(meta["instances"])


def write_subset_root(
    source_manifest: dict,
    records: list[dict],
    subset_root: Path,
    source_webdataset_root: Path,
):
    subset_root.mkdir(parents=True, exist_ok=True)
    manifest_path = subset_root / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    subset_manifest = dict(source_manifest)
    subset_manifest["num_samples"] = len(records)
    shard_paths = sorted({record["shard_path"] for record in records})
    shard_index_map = {entry["path"]: entry for entry in source_manifest["shards"]}
    subset_manifest["num_shards"] = len(shard_paths)
    subset_manifest["shards"] = [shard_index_map[path] for path in shard_paths]
    subset_manifest["subset_from"] = str(source_webdataset_root)
    subset_manifest["sample_manifest"] = "manifest.jsonl"
    subset_manifest["output_root"] = str(subset_root)
    with (subset_root / "dataset_manifest.json").open("w") as f:
        json.dump(subset_manifest, f, indent=2)

    shards_link = subset_root / "shards"
    if shards_link.exists() or shards_link.is_symlink():
        shards_link.unlink()
    shards_link.symlink_to(source_webdataset_root / "shards", target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description="Build MOVI shard subsets filtered by object count.")
    parser.add_argument("--data_root", required=True, help="MOVI dataset root or webdataset root.")
    parser.add_argument(
        "--object_counts",
        type=int,
        nargs="+",
        default=[1, 2, 5],
        help="Object counts to filter into subsets.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Where to create subset roots. Defaults to <webdataset_root>/subsets_by_n_objects",
    )
    parser.add_argument(
        "--limit_per_count",
        type=int,
        default=0,
        help="Optional cap per object count. 0 means keep all matching samples.",
    )
    args = parser.parse_args()

    webdataset_root = resolve_webdataset_root(args.data_root)
    source_manifest_path = webdataset_root / "dataset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())

    output_root = Path(args.output_root) if args.output_root else webdataset_root / "subsets_by_n_objects"
    output_root.mkdir(parents=True, exist_ok=True)

    requested = set(args.object_counts)
    selected = {count: [] for count in args.object_counts}
    distribution = Counter()
    shard_handles = {}

    try:
        for idx, record in enumerate(iter_manifest_records(webdataset_root), start=1):
            n_objects = count_instances_from_record(record, shard_handles, webdataset_root)
            distribution[n_objects] += 1
            if n_objects in requested:
                bucket = selected[n_objects]
                if args.limit_per_count <= 0 or len(bucket) < args.limit_per_count:
                    bucket.append(record)
            if idx % 5000 == 0:
                print(f"Scanned {idx} samples...")
    finally:
        for tf in shard_handles.values():
            tf.close()

    summary = {
        "webdataset_root": str(webdataset_root),
        "distribution": dict(sorted(distribution.items())),
        "subsets": {},
    }

    for count in args.object_counts:
        subset_name = f"nobj_{count}"
        subset_root = output_root / subset_name
        records = selected[count]
        write_subset_root(source_manifest, records, subset_root, webdataset_root)
        sample_ids_path = subset_root / "sample_ids.txt"
        with sample_ids_path.open("w") as f:
            for record in records:
                f.write(f'{record["sample_id"]}\n')
        summary["subsets"][str(count)] = {
            "subset_root": str(subset_root),
            "num_samples": len(records),
            "sample_ids_path": str(sample_ids_path),
        }
        print(f"Built subset n_objects={count}: {len(records)} samples -> {subset_root}")

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
