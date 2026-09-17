import argparse
import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from census_sources import CENSUS_SOURCES

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def download(url: str, destination: Path):
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; KenyaCensusSemanticSearch/1.0; "
                "+https://www.knbs.or.ke/)"
            )
        },
    )

    with urlopen(request, timeout=90) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def download_source(source, force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / source["filename"]

    if destination.exists() and destination.stat().st_size > 50_000 and not force:
        print(f"[skip] Volume {source['volume']}: {destination.name}")
        return True

    tmp = destination.with_suffix(".pdf.part")
    if tmp.exists():
        tmp.unlink()

    urls = [source["repository_url"], source["fallback_url"]]

    for number, url in enumerate(urls, start=1):
        try:
            print(
                f"[download] Volume {source['volume']} — {source['title']} "
                f"(source {number}/{len(urls)})"
            )
            download(url, tmp)

            if tmp.stat().st_size < 50_000:
                raise RuntimeError("Downloaded file is unexpectedly small.")

            tmp.replace(destination)
            mb = destination.stat().st_size / (1024 * 1024)
            print(f"[saved] {destination.name} ({mb:.2f} MB)")
            return True
        except (HTTPError, URLError, TimeoutError, RuntimeError, OSError) as exc:
            print(f"[warning] Could not download from {url}")
            print(f"          {exc}")
            if tmp.exists():
                tmp.unlink()

    print(
        f"[failed] Volume {source['volume']}. You can manually download it from "
        "the KNBS census reports page and save it as "
        f"{destination.name}"
    )
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download the four main 2019 KNBS census volumes."
    )
    parser.add_argument(
        "--volume",
        choices=["I", "II", "III", "IV"],
        help="Download only one census volume.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files that already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the configured census volumes without downloading.",
    )
    args = parser.parse_args()

    selected = [
        source
        for source in CENSUS_SOURCES
        if args.volume is None or source["volume"] == args.volume
    ]

    if args.list:
        for source in selected:
            print(
                f"Volume {source['volume']}: {source['title']} "
                f"-> data/{source['filename']}"
            )
        return

    successes = sum(download_source(source, force=args.force) for source in selected)

    print()
    print(f"Downloaded/available: {successes}/{len(selected)} volume(s).")

    if successes != len(selected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
