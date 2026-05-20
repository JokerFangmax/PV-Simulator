#!/usr/bin/env python3
import argparse
import json
import tarfile
from collections import Counter, defaultdict
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


def group_records_by_shard(webdataset_root: Path):
    grouped = defaultdict(list)
    for record in iter_manifest_records(webdataset_root):
        grouped[record["shard_path"]].append(record)
    return grouped


def write_subset_root(source_manifest: dict, records: list[dict], subset_root: Path, source_webdataset_root: Path):
    subset_root.mkdir(parents=True, exist_ok=True)
    with (subset_root / "manifest.jsonl").open("w") as f:
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

    with (subset_root / "sample_ids.txt").open("w") as f:
        for record in records:
            f.write(f'{record["sample_id"]}\n')


def main():
    parser = argparse.ArgumentParser(description="Build MOVI shard subsets filtered by object count.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--object_counts", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--limit_per_count", type=int, default=0)
    args = parser.parse_args()

    webdataset_root = resolve_webdataset_root(args.data_root)
    source_manifest = json.loads((webdataset_root / "dataset_manifest.json").read_text())
    output_root = Path(args.output_root) if args.output_root else webdataset_root / "subsets_by_n_objects"
    output_root.mkdir(parents=True, exist_ok=True)

    requested = set(args.object_counts)
    selected = {count: [] for count in args.object_counts}
    distribution = Counter()
    records_by_shard = group_records_by_shard(webdataset_root)
    total_scanned = 0

    for shard_idx, shard_name in enumerate(sorted(records_by_shard.keys()), start=1):
        shard_records = records_by_shard[shard_name]
        record_by_payload = {f'{record["sample_id"]}.payload.tar': record for record in shard_records}
        shard_path = webdataset_root / "shards" / shard_name
        with tarfile.open(shard_path, "r:") as outer_tf:
            for member in outer_tf:
                if not member.isfile():
                    continue
                record = record_by_payload.get(member.name)
                if record is None:
                    continue
                with outer_tf.extractfile(member) as payload_fobj:
                    with tarfile.open(fileobj=payload_fobj, mode="r:") as inner_tf:
                        with inner_tf.extractfile("metadata.json") as meta_f:
                            meta = json.load(meta_f)
                n_objects = len(meta["instances"])
                distribution[n_objects] += 1
                if n_objects in requested:
                    bucket = selected[n_objects]
                    if args.limit_per_count <= 0 or len(bucket) < args.limit_per_count:
                        bucket.append(record)
                total_scanned += 1
        print(f"Scanned shard {shard_idx}/{len(records_by_shard)} ({total_scanned} samples)...", flush=True)

    summary = {
        "webdataset_root": str(webdataset_root),
        "distribution": dict(sorted(distribution.items())),
        "subsets": {},
    }

    for count in args.object_counts:
        subset_root = output_root / f"nobj_{count}"
        records = selected[count]
        write_subset_root(source_manifest, records, subset_root, webdataset_root)
        summary["subsets"][str(count)] = {
            "subset_root": str(subset_root),
            "num_samples": len(records),
            "sample_ids_path": str(subset_root / "sample_ids.txt"),
        }
        print(f"Built subset n_objects={count}: {len(records)} samples -> {subset_root}", flush=True)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
