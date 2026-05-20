#!/usr/bin/env python3
import argparse
import json
import tarfile
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
    parser = argparse.ArgumentParser(description="Build small MOVI shard subsets by object count.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--object_counts", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--limit_per_count", type=int, default=500)
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()

    webdataset_root = resolve_webdataset_root(args.data_root)
    source_manifest = json.loads((webdataset_root / "dataset_manifest.json").read_text())
    output_root = Path(args.output_root) if args.output_root else webdataset_root / "subsets_by_n_objects_small"
    output_root.mkdir(parents=True, exist_ok=True)

    targets = {count: [] for count in args.object_counts}
    manifest_records = list(iter_manifest_records(webdataset_root))
    current_shard = None
    outer_tf = None

    try:
        for idx, record in enumerate(manifest_records, start=1):
            if all(len(records) >= args.limit_per_count for records in targets.values()):
                break

            shard_name = record["shard_path"]
            if shard_name != current_shard:
                if outer_tf is not None:
                    outer_tf.close()
                current_shard = shard_name
                outer_tf = tarfile.open(webdataset_root / "shards" / shard_name, "r:")

            payload_name = f'{record["sample_id"]}.payload.tar'
            payload_member = outer_tf.getmember(payload_name)
            with outer_tf.extractfile(payload_member) as payload_fobj:
                with tarfile.open(fileobj=payload_fobj, mode="r:") as inner_tf:
                    with inner_tf.extractfile("metadata.json") as meta_f:
                        meta = json.load(meta_f)

            n_objects = len(meta["instances"])
            bucket = targets.get(n_objects)
            if bucket is not None and len(bucket) < args.limit_per_count:
                bucket.append(record)

            if idx % 1000 == 0:
                status = ", ".join(f"{k}:{len(v)}" for k, v in sorted(targets.items()))
                print(f"Scanned {idx} samples -> {status}", flush=True)
    finally:
        if outer_tf is not None:
            outer_tf.close()

    summary = {
        "webdataset_root": str(webdataset_root),
        "limit_per_count": args.limit_per_count,
        "subsets": {},
    }

    for count in args.object_counts:
        subset_root = output_root / f"nobj_{count}"
        records = targets[count]
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
