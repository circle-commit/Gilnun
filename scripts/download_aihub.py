"""Download AI Hub dataset files and unpack them, resuming where a previous run stopped.

AI Hub's aihubshell does the same, but on macOS it can corrupt large files: it joins
split parts in `sort -V` order, and macOS's sort puts part10 before part2. This script
joins parts by their number, unzips each file, deletes the archives as it goes, and
skips files it has already finished, so it can simply be run again after an
interruption.

By default it downloads the bounding-box files of 인도보행 영상 (about 300 GB) into
datasets/15.인도보행영상/바운딩박스, where scripts/convert_cvat_to_yolo.py reads them.
Run from the repository root. It asks for the AI Hub API key (or reads AIHUB_APIKEY):
    python3 scripts/download_aihub.py --list    # files and sizes only
    python3 scripts/download_aihub.py
    python3 scripts/download_aihub.py --status  # progress, from another terminal
Other datasets: pick files by the folders in their path, and unpack each file into its
own folder under --out so files with the same name in different folders do not collide:
    python3 scripts/download_aihub.py --dataset-key 159 --folder '' --path BBOX --path 라벨링 \
        --keep-tree --out datasets/aihub
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


API_URL = "https://api.aihub.or.kr"
DOWNLOAD_VERSION = "0.6"  # The download API version that aihubshell 0.6 uses.
DEFAULT_DATASET_KEY = 189  # 인도보행 영상
DEFAULT_FOLDER = "바운딩박스"
DEFAULT_OUT = Path("datasets/15.인도보행영상/바운딩박스")
DEFAULT_WORK = Path("datasets/aihub_download")

GIB = 1024**3
CHUNK = 16 * 1024**2
MAX_ATTEMPTS = 10
SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": GIB, "TB": 1024**4}
TREE_LINE = re.compile(r"[├└]─(?P<name>.+?)(?: \| (?P<size>[\d.]+) (?P<unit>[KMGT]?B) \| (?P<key>\d+))?\s*$")
PART_NAME = re.compile(r"^(?P<prefix>.+)\.part(?P<index>\d+)$")


@dataclass(frozen=True)
class RemoteFile:
    key: str
    name: str
    folder: str
    size: int  # Approximate, from the rounded size in the file tree.
    path: str = ""  # Folders from the dataset's root, joined with "/".


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and unpack AI Hub dataset files.")
    parser.add_argument("--dataset-key", type=int, default=DEFAULT_DATASET_KEY, help="AI Hub dataset key.")
    parser.add_argument(
        "--folder",
        default=DEFAULT_FOLDER,
        help="Only files in this folder of the dataset's file tree. Pass '' for every file.",
    )
    parser.add_argument("--files", default=None, help="Comma-separated file keys to download (default: all in --folder).")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Only files whose folder path contains this text. Repeat to require several.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Where the zip contents are extracted.")
    parser.add_argument(
        "--keep-tree",
        action="store_true",
        help="Unpack each file into <out>/<its folder path>/<file name> instead of straight into --out.",
    )
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK, help="Download and progress directory.")
    parser.add_argument("--list", action="store_true", help="Show the files and exit.")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show the progress of the last download run (works while it runs) and exit.",
    )
    return parser.parse_args()


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def parse_tree(tree: str) -> list[RemoteFile]:
    """Files in a dataset's file tree, which lists them as `name | size | file key`."""

    files = []
    # Open folders as (column of their branch mark, name); deeper entries start further right.
    folders: list[tuple[int, str]] = []
    for line in tree.splitlines():
        match = TREE_LINE.search(line)
        if not match:
            continue
        column = match.start()
        while folders and folders[-1][0] >= column:
            folders.pop()
        if match["key"]:
            size = int(float(match["size"]) * SIZE_UNITS[match["unit"]])
            folder = folders[-1][1] if folders else ""
            path = "/".join(name for _, name in folders)
            files.append(RemoteFile(match["key"], nfc(match["name"]), folder, size, path))
        else:
            folders.append((column, nfc(match["name"])))
    return files


def list_files(dataset_key: int) -> list[RemoteFile]:
    url = f"{API_URL}/info/{dataset_key}.do"
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            body = error.read()  # AI Hub sends the file tree with HTTP 502.
        except urllib.error.URLError as error:
            print(f"Could not reach AI Hub ({error.reason}).", file=sys.stderr)
            body = b""
        files = parse_tree(body.decode("utf-8", errors="replace"))
        if files:
            return files
        if attempt < 3:
            print("No file list from AI Hub; retrying in 20 seconds.", file=sys.stderr)
            time.sleep(20)
    raise SystemExit(f"Could not get the file list of dataset {dataset_key} from AI Hub. Try again later.")


def keep_awake() -> None:
    """Keep a Mac from idle-sleeping while this process runs (does nothing elsewhere)."""

    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])


def run_curl(url: str, api_key: str, output: Path, start: int) -> tuple[int, str]:
    """Fetch url from byte `start` into output. Returns curl's exit code and the HTTP status."""

    command = [
        "curl",
        "--location",
        "--output",
        str(output),
        # The key goes in on stdin so it never shows in the process list.
        "--header",
        "@-",
        "--write-out",
        "%{http_code}",
        # Treat a stalled connection (under 1 KB/s for 2 minutes) as an error and retry.
        "--speed-limit",
        "1024",
        "--speed-time",
        "120",
    ]
    if start:
        command += ["--range", f"{start}-"]
    result = subprocess.run(command + [url], input=f"apikey:{api_key}\n", stdout=subprocess.PIPE, text=True)
    return result.returncode, result.stdout.strip()[-3:]


def error_message(path: Path) -> str:
    """The text of a short error response; "" for an empty response or an HTML error page,
    which usually means a temporary server problem worth retrying."""

    if not path.exists() or path.stat().st_size > 64 * 1024:
        return ""
    text = path.read_bytes().decode("utf-8", errors="replace").strip()
    return "" if text.startswith("<") else text


def download(remote: RemoteFile, dataset_key: int, api_key: str, tar_path: Path) -> None:
    """Download the file's tar from AI Hub, resuming a partial download.

    Each request writes to its own file, so an error response never ends up in the tar.
    AI Hub answers errors such as a wrong API key with HTTP 502 and a short message, so
    the message, not the status, tells whether retrying can help.
    """

    url = f"{API_URL}/down/{DOWNLOAD_VERSION}/{dataset_key}.do?fileSn={remote.key}"
    response_path = tar_path.with_name(tar_path.name + ".response")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = tar_path.stat().st_size if tar_path.exists() else 0
        exit_code, status = run_curl(url, api_key, response_path, start)

        if status == "206":
            with open(tar_path, "ab") as out, open(response_path, "rb") as rest:
                shutil.copyfileobj(rest, out, CHUNK)
            response_path.unlink()
        elif status == "200":
            # A fresh download, or a server that ignored the range and sent everything again.
            response_path.replace(tar_path)
        else:
            message = error_message(response_path)
            response_path.unlink(missing_ok=True)
            if message:
                raise SystemExit(
                    f"AI Hub: {message}\n"
                    "Check the API key and that the dataset application is approved on AI Hub."
                )

        if exit_code == 0 and status in {"200", "206"}:
            if tarfile.is_tarfile(tar_path):
                return
            message = error_message(tar_path)
            tar_path.unlink()
            raise SystemExit(f"AI Hub did not send a tar file: {message or 'unknown response'}")
        if attempt < MAX_ATTEMPTS:
            print(f"Download interrupted (HTTP {status}, curl exit {exit_code}); retrying in 30 seconds.", file=sys.stderr)
            time.sleep(30)
    raise SystemExit(f"Download kept failing after {MAX_ATTEMPTS} attempts. Run the script again to resume.")


class PartsError(Exception):
    """The parts in a downloaded tar do not add up to whole files."""


def check_part_order(prefix: str, parts: list[tuple[int, tarfile.TarInfo]]) -> None:
    """Parts are numbered either 0, 1, 2, ... or, as AI Hub does, by their byte offset in the file."""

    indices = [index for index, _ in parts]
    if indices == list(range(indices[0], indices[0] + len(indices))):
        return
    offset = 0
    for index, member in parts:
        if index != offset:
            raise PartsError(f"{prefix}: expected a part at byte {offset}, found part{index}")
        offset += member.size


def merge_parts(tar_path: Path, dest: Path) -> list[Path]:
    """Write each file in the tar to dest, joining split parts (name.partN) in order.

    AI Hub's tar can hold every part twice; copies of a part must have the same size, and
    the last one is used, as tar extraction would. The zip's CRC checks catch a bad copy.
    """

    outputs = []
    with tarfile.open(tar_path) as tar:
        groups: dict[str, dict[int, tarfile.TarInfo]] = {}
        copies = 0
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # Only the base name is used, so paths inside the tar cannot point outside dest.
            name = PurePosixPath(member.name).name
            match = PART_NAME.match(name)
            prefix, index = (match["prefix"], int(match["index"])) if match else (name, 0)
            parts = groups.setdefault(prefix, {})
            if index in parts:
                copies += 1
                if parts[index].size != member.size:
                    raise PartsError(f"{prefix}: two copies of part{index} differ in size")
            parts[index] = member

        for prefix, parts in groups.items():
            ordered = sorted(parts.items())
            check_part_order(prefix, ordered)
            size = sum(member.size for _, member in ordered)
            note = f", {copies} duplicate parts skipped" if copies else ""
            print(f"{prefix}: {len(ordered)} parts, {size / GIB:.1f} GB{note}")
            output = dest / prefix
            with open(output, "wb") as out:
                for _, member in ordered:
                    shutil.copyfileobj(tar.extractfile(member), out, CHUNK)
            outputs.append(output)
    return outputs


def member_name(info: zipfile.ZipInfo) -> str:
    """Zips made on Korean Windows store names in CP949 without the UTF-8 flag."""

    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("cp949")
        except UnicodeError:
            pass
    return nfc(name.replace("\\", "/"))


def extract_zip(zip_path: Path, out_dir: Path) -> int:
    """Unzip into out_dir, skipping files already there with the same size. Returns the file count."""

    root = out_dir.resolve()
    count = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = member_name(info)
            if info.is_dir() or name.startswith("__MACOSX/"):
                continue
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise SystemExit(f"Refusing to extract {name!r} outside {root}")
            count += 1
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Reading to the end checks the CRC, so a corrupt zip raises BadZipFile here.
            with archive.open(info) as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination, CHUNK)
    return count


def fetch(remote: RemoteFile, args: argparse.Namespace, api_key: str) -> None:
    """Download, join and unzip one file. Markers in its work directory let a rerun skip finished steps."""

    state = args.work / remote.key
    state.mkdir(parents=True, exist_ok=True)
    tar_path = state / "download.tar"
    downloaded = state / "downloaded"
    merged = state / "merged"

    if not merged.exists():
        if not downloaded.exists():
            partial = tar_path.stat().st_size if tar_path.exists() else 0
            # The tar (which can hold every part twice) and the joined zip exist side by side.
            needed = 3 * remote.size + 5 * GIB - partial
            free = shutil.disk_usage(state).free
            if free < needed:
                raise SystemExit(f"Not enough disk space: {free / GIB:.0f} GiB free, about {needed / GIB:.0f} GiB needed.")
            download(remote, args.dataset_key, api_key, tar_path)
            downloaded.touch()
        print("Joining parts...")
        try:
            merge_parts(tar_path, state)
        except PartsError as error:
            # Keep the download: a fixed script can join it without downloading it again.
            raise SystemExit(
                f"Could not join the parts of {remote.name} ({error}). The download is kept in {state}; "
                "delete that folder to download it again."
            )
        except (tarfile.TarError, EOFError) as error:
            shutil.rmtree(state)
            raise SystemExit(f"The download of {remote.name} is damaged ({error}); run the script again.")
        merged.touch()
        tar_path.unlink()

    target = args.out / remote.path / Path(remote.name).stem if args.keep_tree else args.out
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(state.iterdir()):
        if item.suffix.lower() == ".zip":
            print(f"Unzipping {item.name}...")
            try:
                count = extract_zip(item, target)
            except zipfile.BadZipFile as error:
                shutil.rmtree(state)
                raise SystemExit(f"{item.name} is corrupt ({error}); run the script again to download it again.")
            item.unlink()
            print(f"{count} files in {target}")
        elif item.name not in {downloaded.name, merged.name}:
            shutil.move(str(item), str(target / item.name))
    shutil.rmtree(state)


def size_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:  # Replaced or deleted by a running download.
        return 0


def file_progress(remote: RemoteFile, work: Path) -> tuple[str, int]:
    """The step a file is at, from the markers fetch() leaves, and how many of its bytes have arrived."""

    if (work / "done" / remote.key).exists():
        return "done", remote.size
    state = work / remote.key
    if (state / "merged").exists():
        return "unzipping", remote.size
    if (state / "downloaded").exists():
        return "joining", remote.size
    received = sum(size_of(path) for path in state.glob("download.tar*")) if state.is_dir() else 0
    return ("partial" if received else "pending"), received


def print_table(files: list[RemoteFile], work: Path) -> None:
    print(f"{'key':<8}{'size':>8}  {'status':<16}name")
    for remote in files:
        step, received = file_progress(remote, work)
        if step == "partial":
            step = f"{min(99, 100 * received // remote.size)}% ({received / GIB:.1f} GB)"
        print(f"{remote.key:<8}{remote.size / GIB:>5.0f} GB  {step:<16}{remote.name}")


def received_bytes(files: list[RemoteFile], work: Path) -> int:
    return sum(file_progress(remote, work)[1] for remote in files)


def finished_rate(files: list[RemoteFile], work: Path) -> float | None:
    """Bytes per second over the finished files, including joining and unzipping."""

    size = seconds = 0
    for remote in files:
        try:
            record = json.loads((work / "done" / remote.key).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            continue
        size += record["size"]
        seconds += record["seconds"]
    return size / seconds if seconds else None


def summary(files: list[RemoteFile], work: Path, speed: float | None = None) -> str:
    done = sum((work / "done" / remote.key).exists() for remote in files)
    total = sum(remote.size for remote in files)
    # Listed sizes are rounded, so a file can grow past its own; count at most that much.
    received = sum(min(file_progress(remote, work)[1], remote.size) for remote in files)
    line = f"{done} of {len(files)} files done, {received / GIB:.0f} of {total / GIB:.0f} GB ({100 * received / total:.0f}%)"
    rate = finished_rate(files, work) or speed
    if rate and received < total:
        line += f", about {(total - received) / rate / 3600:.1f} h left"
    return line


def downloader_running() -> bool:
    """Whether another process is running a download (not --list or --status)."""

    try:
        processes = subprocess.run(["ps", "ax", "-o", "pid=,command="], capture_output=True, text=True).stdout
    except OSError:
        return False
    for line in processes.splitlines():
        pid, *command = line.split()
        # Python running this script, not an editor or a shell that merely mentions it.
        if (
            command
            and "python" in Path(command[0]).name.lower()
            and any(arg.endswith("download_aihub.py") for arg in command[1:])
            and not {"--status", "--list"} & set(command)
            and int(pid) != os.getpid()
        ):
            return True
    return False


def show_status(files: list[RemoteFile], work: Path) -> None:
    running = downloader_running()
    all_done = all((work / "done" / remote.key).exists() for remote in files)
    if running:
        print("The download is running.")
    elif not all_done:
        print("The download is not running. Run python3 scripts/download_aihub.py to continue.")
    print_table(files, work)

    speed = None
    if running:
        before = received_bytes(files, work)
        time.sleep(3)
        speed = max(0, received_bytes(files, work) - before) / 3 or None
    print(summary(files, work, speed))
    if speed:
        print(f"Download speed now: {speed / 1024**2:.1f} MB/s")
    print(f"Free disk space: {shutil.disk_usage(work if work.exists() else Path.cwd()).free / GIB:.0f} GB")


def selected_files(args: argparse.Namespace) -> list[RemoteFile]:
    files = list_files(args.dataset_key)
    if args.folder:
        files = [remote for remote in files if remote.folder == nfc(args.folder)]
    for part in args.path:
        files = [remote for remote in files if nfc(part) in remote.path]
    if args.files:
        wanted = {key.strip() for key in args.files.split(",")}
        files = [remote for remote in files if remote.key in wanted]
    if not files:
        raise SystemExit(f"No matching files in dataset {args.dataset_key}.")
    return files


def main() -> None:
    args = parse_args()
    # The files of the last download run, so --status works without asking AI Hub.
    saved_files = args.work / "files.json"
    if args.status:
        if saved_files.exists():
            files = [RemoteFile(**item) for item in json.loads(saved_files.read_text(encoding="utf-8"))]
        else:
            files = selected_files(args)
        show_status(files, args.work)
        return

    files = selected_files(args)
    done_dir = args.work / "done"
    pending = [remote for remote in files if not (done_dir / remote.key).exists()]
    print_table(files, args.work)
    print(f"{len(pending)} of {len(files)} files to download, about {sum(r.size for r in pending) / GIB:.0f} GB")
    if args.list or not pending:
        return

    if os.environ.get("AIHUB_APIKEY"):
        print("Using the API key in AIHUB_APIKEY.")
        api_key = os.environ["AIHUB_APIKEY"].strip()
    else:
        api_key = getpass.getpass("AI Hub API key (input hidden): ").strip()
    if not api_key:
        raise SystemExit("An API key is required.")

    keep_awake()
    args.out.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)
    saved_files.write_text(json.dumps([asdict(remote) for remote in files], ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.monotonic()
    try:
        for index, remote in enumerate(pending, start=1):
            print(f"\n[{index}/{len(pending)}] {remote.name} (about {remote.size / GIB:.0f} GB)")
            file_started = time.monotonic()
            fetch(remote, args, api_key)
            # The time per file lets --status estimate how long the rest will take.
            record = {"name": remote.name, "size": remote.size, "seconds": round(time.monotonic() - file_started)}
            (done_dir / remote.key).write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            print(summary(files, args.work))
    except KeyboardInterrupt:
        raise SystemExit("\nStopped. Run the same command again to resume.")
    print(f"\nDone in {(time.monotonic() - started) / 3600:.1f} h. Files are in {args.out}")


if __name__ == "__main__":
    main()
