from __future__ import annotations
import argparse, csv, io, re, subprocess
from pathlib import Path


def dataset_ref(url: str) -> str:
    match = re.search(r"/datasets/([^/]+/[^/?#]+)", url)
    if not match:
        raise ValueError(f"Cannot parse Kaggle dataset URL: {url}")
    return match.group(1)


def run(command: list[str], *, capture: bool = False) -> str:
    print("+", " ".join(command))
    result = subprocess.run(command, check=True, text=True, capture_output=capture)
    return result.stdout if capture else ""


def list_dataset_files(ref: str, page_size: int = 200) -> list[str]:
    output = run(["kaggle", "datasets", "files", ref, "--csv", "--page-size", str(page_size)], capture=True)
    rows = list(csv.DictReader(io.StringIO(output)))
    names = []
    for row in rows:
        name = row.get("name") or row.get("Name") or row.get("fileName")
        if name:
            names.append(name)
    return names


def main() -> None:
    p = argparse.ArgumentParser(description="Download public CABT episode datasets listed in manifest.csv")
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--output", default="data/episodes")
    p.add_argument("--latest", type=int, default=1)
    p.add_argument("--file", action="append", default=[], help="Exact dataset filename; repeat as needed")
    p.add_argument("--max-files", type=int, default=0, help="Download up to N listed JSON/archive files (maximum first page: 200)")
    p.add_argument("--pattern", default=r"\.(json|zip|tar|tar\.gz|tgz)$", help="Regex used with --max-files")
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--whole-dataset", action="store_true", help="Download the full selected daily datasets")
    args = p.parse_args()
    if args.latest <= 0:
        raise ValueError("--latest must be positive")
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))[-args.latest:]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    for row in rows:
        ref = dataset_ref(row["daily_dataset_url"])
        target = out / row["date"]; target.mkdir(parents=True, exist_ok=True)
        size_gb = int(row["total_bytes"]) / 1e9
        if args.list_only or args.max_files:
            files = list_dataset_files(ref)
            matches = [name for name in files if re.search(args.pattern, name, re.I)]
            print(f"{ref}: listed {len(files)} files on the first page; {len(matches)} match {args.pattern!r}")
            for name in matches[: max(args.max_files, 20) if args.list_only else args.max_files]:
                print(name)
            if args.list_only:
                continue
            for filename in matches[:args.max_files]:
                run(["kaggle", "datasets", "download", ref, "-f", filename, "-p", str(target), "--unzip"])
        elif args.whole_dataset:
            print(f"Downloading approximately {size_gb:.1f} GB from {ref}")
            run(["kaggle", "datasets", "download", ref, "-p", str(target), "--unzip"])
        elif args.file:
            for filename in args.file:
                run(["kaggle", "datasets", "download", ref, "-f", filename, "-p", str(target), "--unzip"])
        else:
            print(f"Selected {ref}. The manifest reports approximately {size_gb:.1f} GB.")
            print("Use --list-only, --max-files N, exact --file names, --whole-dataset, or attach the dataset as a Kaggle Notebook input.")

if __name__ == "__main__":
    main()
