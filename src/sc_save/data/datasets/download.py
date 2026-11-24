"""Download, HTTP Range ZIP64 extraction, and GenePT embedding utilities."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib

from tqdm import tqdm

from sc_save.data.datasets.registry import (
    DatasetSpec,
    GENEPT_PICKLE_NAME,
    GENEPT_ZENODO_BACKUP_URL,
    GENEPT_ZENODO_URL,
    _get_raw_paths,
)

DEFAULT_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": "curl/8.5.0",
    "Accept": "*/*",
}
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ZIP_MAGIC = b"PK\x03\x04"


def _resolve_download_file():
    """Allow monkeypatching ``sc_save.data.datasets._download_file`` or ``sc_save_public.datasets._download_file``."""
    pkg = sys.modules.get("sc_save.data.datasets")
    if pkg is not None and hasattr(pkg, "_download_file"):
        return pkg._download_file
    return _download_file


def _resolve_download_gdrive_file():
    pkg = sys.modules.get("sc_save.data.datasets")
    if pkg is not None and hasattr(pkg, "_download_gdrive_file"):
        return pkg._download_gdrive_file
    return _download_gdrive_file


def _build_request(
    url: str,
    headers: dict[str, str] | None = None,
    method: str = "GET",
) -> urllib.request.Request:
    req_headers = dict(DEFAULT_HTTP_HEADERS)
    if headers:
        req_headers.update(headers)
    return urllib.request.Request(url, headers=req_headers, method=method)


def _http_range_get(
    url: str,
    byte_range: str,
    timeout: int = 120,
    max_retries: int = 4,
) -> tuple[bytes, str | None, int]:
    """Execute a single HTTP Range GET request returning (body, Content-Range, status)."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = _build_request(url, headers={"Range": f"bytes={byte_range}"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                content_range = resp.headers.get("Content-Range")
                data = resp.read()
                if status not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {status} for {url}")
                return data, content_range, status
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_exc = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401, 403, 404):
                raise
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Range request failed for {url} ({byte_range}): {last_exc}")


def _parse_total_size_from_content_range(content_range: str | None) -> int | None:
    if not content_range or "/" not in content_range:
        return None
    total_str = content_range.split("/")[-1].strip()
    if total_str == "*":
        return None
    return int(total_str)


@dataclass(frozen=True)
class _ZipEntryMeta:
    name: str
    compress_method: int
    crc32: int
    comp_size: int
    uncomp_size: int
    local_header_offset: int


def _parse_zip64_extra(
    extra: bytes,
    need_uncomp: bool,
    need_comp: bool,
    need_offset: bool,
) -> tuple[int | None, int | None, int | None]:
    pos = 0
    uncomp_val = comp_val = offset_val = None
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        payload = extra[pos : pos + data_size]
        pos += data_size
        if header_id != 0x0001:
            continue
        idx = 0
        if need_uncomp and idx + 8 <= len(payload):
            (uncomp_val,) = struct.unpack_from("<Q", payload, idx)
            idx += 8
        if need_comp and idx + 8 <= len(payload):
            (comp_val,) = struct.unpack_from("<Q", payload, idx)
            idx += 8
        if need_offset and idx + 8 <= len(payload):
            (offset_val,) = struct.unpack_from("<Q", payload, idx)
        break
    return uncomp_val, comp_val, offset_val


def _read_remote_zip_directory(
    url: str,
    timeout: int = 120,
) -> dict[str, _ZipEntryMeta]:
    """Fetch only the EOCD + Central Directory of a remote ZIP/ZIP64 archive via HTTP Range."""
    tail_size = 65560
    tail_bytes, content_range, status = _http_range_get(
        url, f"-{tail_size}", timeout=timeout
    )
    eocd_pos = tail_bytes.rfind(b"PK\x05\x06")
    if eocd_pos == -1 or eocd_pos + 22 > len(tail_bytes):
        raise RuntimeError(f"Could not find ZIP EOCD signature in tail of {url}")

    total_archive_size = _parse_total_size_from_content_range(content_range)
    if total_archive_size is None:
        if status == 200:
            total_archive_size = len(tail_bytes)
        else:
            raise RuntimeError(
                f"Server did not provide Content-Range total size for {url}: {content_range}"
            )
    tail_start = max(0, total_archive_size - len(tail_bytes))

    _, _, _, _, _, cd_size, cd_offset, _ = struct.unpack_from(
        "<4sHHHHIIH", tail_bytes, eocd_pos
    )
    if cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        loc_pos = tail_bytes.rfind(b"PK\x06\x07", 0, eocd_pos)
        if loc_pos == -1 or loc_pos + 20 > len(tail_bytes):
            raise RuntimeError(f"ZIP64 marker set, but ZIP64 EOCD locator missing for {url}")
        _, _, zip64_eocd_offset, _ = struct.unpack_from("<4sIQH", tail_bytes, loc_pos)
        if zip64_eocd_offset >= tail_start:
            rel = zip64_eocd_offset - tail_start
            z64 = tail_bytes[rel : rel + 56]
        else:
            z64, _, _ = _http_range_get(
                url,
                f"{zip64_eocd_offset}-{zip64_eocd_offset + 55}",
                timeout=timeout,
            )
        if not z64.startswith(b"PK\x06\x06") or len(z64) < 56:
            raise RuntimeError(f"Invalid ZIP64 EOCD header in {url}")
        _, _, _, _, _, _, _, _, cd_size, cd_offset = struct.unpack_from(
            "<4sQHHIIQQQQ", z64, 0
        )

    if cd_offset >= tail_start and cd_offset + cd_size <= tail_start + len(tail_bytes):
        rel = cd_offset - tail_start
        cd_bytes = tail_bytes[rel : rel + cd_size]
    else:
        cd_bytes, _, _ = _http_range_get(
            url, f"{cd_offset}-{cd_offset + cd_size - 1}", timeout=timeout
        )

    entries: dict[str, _ZipEntryMeta] = {}
    pos = 0
    while pos + 46 <= len(cd_bytes):
        if cd_bytes[pos : pos + 4] != b"PK\x01\x02":
            break
        (
            _,
            _,
            _,
            _,
            method,
            _,
            _,
            crc32,
            comp_size,
            uncomp_size,
            fname_len,
            extra_len,
            comment_len,
            _,
            _,
            _,
            local_offset,
        ) = struct.unpack_from("<4sHHHHHHIIIHHHHHII", cd_bytes, pos)
        pos += 46
        name = cd_bytes[pos : pos + fname_len].decode("utf-8", errors="replace")
        pos += fname_len
        extra = cd_bytes[pos : pos + extra_len]
        pos += extra_len + comment_len

        need_u = uncomp_size == 0xFFFFFFFF
        need_c = comp_size == 0xFFFFFFFF
        need_o = local_offset == 0xFFFFFFFF
        if need_u or need_c or need_o:
            u64, c64, o64 = _parse_zip64_extra(extra, need_u, need_c, need_o)
            if u64 is not None:
                uncomp_size = u64
            if c64 is not None:
                comp_size = c64
            if o64 is not None:
                local_offset = o64

        entries[name] = _ZipEntryMeta(
            name=name,
            compress_method=method,
            crc32=crc32,
            comp_size=comp_size,
            uncomp_size=uncomp_size,
            local_header_offset=local_offset,
        )
    return entries


def _stream_remote_zip_entry(
    url: str,
    meta: _ZipEntryMeta,
    dest: Path,
    timeout: int = 120,
    chunk_size: int = 4 * 1024 * 1024,
) -> None:
    """Stream one compressed entry from a remote ZIP/ZIP64 archive to dest atomically."""
    lh_bytes, _, _ = _http_range_get(
        url,
        f"{meta.local_header_offset}-{meta.local_header_offset + 29}",
        timeout=timeout,
    )
    if len(lh_bytes) < 30 or lh_bytes[:4] != ZIP_MAGIC:
        raise RuntimeError(f"Invalid local ZIP header for {meta.name} in {url}")
    _, _, _, _, _, _, _, _, _, lf_fname_len, lf_extra_len = struct.unpack_from(
        "<4sHHHHHIIIHH", lh_bytes, 0
    )
    data_start = meta.local_header_offset + 30 + lf_fname_len + lf_extra_len
    if meta.comp_size == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"")
        return
    data_end = data_start + meta.comp_size - 1

    if meta.compress_method not in (0, 8):
        raise RuntimeError(
            f"Unsupported ZIP compression method {meta.compress_method} for {meta.name}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    decomp = zlib.decompressobj(-zlib.MAX_WBITS) if meta.compress_method == 8 else None
    crc = 0
    written = 0

    req = _build_request(url, headers={"Range": f"bytes={data_start}-{data_end}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(
            tmp_path, "wb"
        ) as out_f, tqdm(
            total=meta.comp_size,
            unit="B",
            unit_scale=True,
            desc=f"Extracting {dest.name}",
            leave=True,
        ) as pbar:
            status = getattr(resp, "status", 200)
            if status not in (200, 206):
                raise RuntimeError(f"Unexpected HTTP status {status} streaming {meta.name}")
            if status == 200 and data_start > 0:
                remaining_skip = data_start
                while remaining_skip > 0:
                    discard = resp.read(min(chunk_size, remaining_skip))
                    if not discard:
                        raise RuntimeError(f"Unexpected EOF skipping to {data_start}")
                    remaining_skip -= len(discard)

            remaining = meta.comp_size
            while remaining > 0:
                chunk = resp.read(min(chunk_size, remaining))
                if not chunk:
                    raise RuntimeError(
                        f"Truncated stream for {meta.name}: {remaining} bytes left"
                    )
                remaining -= len(chunk)
                pbar.update(len(chunk))
                out_chunk = decomp.decompress(chunk) if decomp is not None else chunk
                if out_chunk:
                    out_f.write(out_chunk)
                    crc = zlib.crc32(out_chunk, crc)
                    written += len(out_chunk)

            if decomp is not None:
                tail = decomp.flush()
                if tail:
                    out_f.write(tail)
                    crc = zlib.crc32(tail, crc)
                    written += len(tail)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    if written != meta.uncomp_size:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Size mismatch for {meta.name}: expected {meta.uncomp_size}, got {written}"
        )
    if (crc & 0xFFFFFFFF) != (meta.crc32 & 0xFFFFFFFF):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"CRC32 mismatch for {meta.name}: expected {meta.crc32:#x}, got {crc:#x}"
        )
    os.replace(tmp_path, dest)


def extract_zip_entries_via_http_range(
    url: str,
    member_to_dest: dict[str, Path],
    backup_urls: Sequence[str] = (),
    force: bool = False,
    timeout: int = 120,
    max_workers: int = 1,
) -> dict[str, Path]:
    """Download only requested entries from a remote ZIP/ZIP64 archive via HTTP Range."""
    pending = {
        member: dest
        for member, dest in member_to_dest.items()
        if force or not (dest.exists() and dest.stat().st_size > 0)
    }
    if not pending:
        return dict(member_to_dest)

    last_err: Exception | None = None
    for candidate_url in (url, *backup_urls):
        try:
            entries = _read_remote_zip_directory(candidate_url, timeout=timeout)
            missing = [m for m in pending if m not in entries]
            if missing:
                raise KeyError(
                    f"Archive entries not found in {candidate_url}: {missing}"
                )
            n_workers = max(1, min(int(max_workers), len(pending)))
            if n_workers == 1 or len(pending) == 1:
                for member, dest in pending.items():
                    _stream_remote_zip_entry(
                        candidate_url, entries[member], dest, timeout=timeout
                    )
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = [
                        pool.submit(
                            _stream_remote_zip_entry,
                            candidate_url,
                            entries[member],
                            dest,
                            timeout=timeout,
                        )
                        for member, dest in pending.items()
                    ]
                    for fut in as_completed(futures):
                        fut.result()
            return dict(member_to_dest)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(
        f"Failed to extract ZIP entries {list(pending.keys())} from {url}: {last_err}"
    )


def _download_file(
    url: str,
    dest: Path,
    *,
    backup_urls: Sequence[str] = (),
    expected_magic: bytes | None = None,
    force: bool = False,
    timeout: int = 120,
    max_retries: int = 4,
    chunk_size: int = 4 * 1024 * 1024,
) -> Path:
    """Download a file atomically with HTTP Range resume support and header checks."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not force and dest.exists() and dest.stat().st_size > 0:
        if expected_magic is None:
            return dest
        with open(dest, "rb") as f:
            if f.read(len(expected_magic)) == expected_magic:
                return dest

    part_path = dest.with_suffix(dest.suffix + ".part")
    if force and part_path.exists():
        part_path.unlink()

    last_err: Exception | None = None
    for candidate_url in (url, *backup_urls):
        for attempt in range(max_retries):
            try:
                existing_bytes = part_path.stat().st_size if part_path.exists() else 0
                headers = (
                    {"Range": f"bytes={existing_bytes}-"} if existing_bytes > 0 else {}
                )
                req = _build_request(candidate_url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    status = getattr(resp, "status", 200)
                    resuming = status == 206 and existing_bytes > 0
                    mode = "ab" if resuming else "wb"
                    content_len = resp.headers.get("Content-Length")
                    total_bytes = (
                        existing_bytes + int(content_len)
                        if resuming and content_len is not None
                        else (int(content_len) if content_len is not None else None)
                    )
                    initial = existing_bytes if resuming else 0
                    with open(part_path, mode) as out_f, tqdm(
                        total=total_bytes,
                        initial=initial,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {dest.name}",
                        leave=True,
                    ) as pbar:
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            out_f.write(chunk)
                            pbar.update(len(chunk))

                if expected_magic is not None:
                    with open(part_path, "rb") as f:
                        head = f.read(len(expected_magic))
                    if head != expected_magic:
                        part_path.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"Downloaded {dest.name} from {candidate_url} failed magic check "
                            f"(expected {expected_magic!r}, got {head!r})."
                        )
                os.replace(part_path, dest)
                return dest
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                OSError,
                RuntimeError,
            ) as exc:
                last_err = exc
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to download {dest.name} from {url}: {last_err}")


def _download_gdrive_file(
    file_id: str,
    dest: Path,
    *,
    expected_magic: bytes | None = HDF5_MAGIC,
    force: bool = False,
) -> Path:
    """Download a large Google Drive file via gdown and verify its magic bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not force and dest.exists() and dest.stat().st_size > 0:
        if expected_magic is None:
            return dest
        with open(dest, "rb") as f:
            if f.read(len(expected_magic)) == expected_magic:
                return dest

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download Google Drive datasets (PBMC). "
            "Install it with `pip install gdown` or pass an already downloaded "
            f"raw file at {dest}."
        ) from exc

    part_path = dest.with_suffix(dest.suffix + ".part")
    if force and part_path.exists():
        part_path.unlink()

    result = gdown.download(
        id=file_id,
        output=str(part_path),
        quiet=False,
        resume=not force,
    )
    if result is None or not part_path.exists() or part_path.stat().st_size == 0:
        raise RuntimeError(
            f"gdown failed to download Google Drive ID '{file_id}' to {dest}."
        )
    if expected_magic is not None:
        with open(part_path, "rb") as f:
            head = f.read(len(expected_magic))
        if head != expected_magic:
            part_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded Google Drive file '{file_id}' is not a valid HDF5/.h5ad "
                f"(got header {head!r}); Drive quota or link sharing may be blocking it."
            )
    os.replace(part_path, dest)
    return dest


def _select_genept_zip_member(zf: zipfile.ZipFile, zip_dest: Path) -> str:
    """Locate the GenePT v2 model-3 text embedding pickle inside the downloaded ZIP."""
    candidates = [
        info.filename
        for info in zf.infolist()
        if not info.is_dir()
        and not info.filename.startswith("__MACOSX/")
        and "/__MACOSX/" not in info.filename
        and PurePosixPath(info.filename).name.rstrip(".") == GENEPT_PICKLE_NAME
    ]
    if not candidates:
        raise KeyError(
            f"Could not find {GENEPT_PICKLE_NAME} inside {zip_dest}; "
            f"entries preview: {[info.filename for info in zf.infolist()[:20]]}"
        )
    exact = [
        member
        for member in candidates
        if PurePosixPath(member).name == GENEPT_PICKLE_NAME
    ]
    return exact[0] if exact else candidates[0]


def ensure_genept_embeddings(
    data_root: str | Path,
    gene_embedding_path: str | Path | None = None,
    force: bool = False,
    skip_download: bool = False,
) -> Path:
    """Ensure GenePT_gene_protein_embedding_model_3_text.pickle exists and return its path."""
    if gene_embedding_path is not None:
        explicit = Path(gene_embedding_path)
        if not explicit.exists():
            raise FileNotFoundError(
                f"Specified gene_embedding_path does not exist: {explicit}"
            )
        return explicit

    root = Path(data_root)
    pickle_dest = root / "genept" / GENEPT_PICKLE_NAME
    if not force and pickle_dest.exists() and pickle_dest.stat().st_size > 0:
        return pickle_dest

    if skip_download:
        raise FileNotFoundError(
            f"Missing GenePT embeddings with skip_download=True: {pickle_dest}"
        )

    zip_dest = root / "raw" / "genept" / "GenePT_emebdding_v2.zip"
    _resolve_download_file()(
        GENEPT_ZENODO_URL,
        zip_dest,
        backup_urls=(GENEPT_ZENODO_BACKUP_URL,),
        expected_magic=ZIP_MAGIC,
        force=force,
    )

    pickle_dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_pickle = pickle_dest.with_suffix(".pickle.part")
    try:
        with zipfile.ZipFile(zip_dest, "r") as zf:
            selected_member = _select_genept_zip_member(zf, zip_dest)
            with zf.open(selected_member) as src_f, open(tmp_pickle, "wb") as dst_f:
                while True:
                    buf = src_f.read(4 * 1024 * 1024)
                    if not buf:
                        break
                    dst_f.write(buf)
        os.replace(tmp_pickle, pickle_dest)
    except Exception:
        tmp_pickle.unlink(missing_ok=True)
        raise
    return pickle_dest


def _ensure_raw_files(
    spec: DatasetSpec,
    data_root: Path,
    *,
    skip_download: bool = False,
    force: bool = False,
) -> list[Path]:
    raw_paths = _get_raw_paths(spec, data_root)
    if skip_download:
        missing = [p for p in raw_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing raw files with --skip-download for '{spec.name}': {missing}"
            )
        return raw_paths

    dl_file = _resolve_download_file()
    dl_gdrive = _resolve_download_gdrive_file()
    if spec.download_kind == "cfgen_zip64":
        mapping = dict(zip(spec.archive_members, raw_paths))
        extract_zip_entries_via_http_range(
            spec.source_urls[0],
            mapping,
            backup_urls=spec.backup_urls,
            force=force,
        )
    elif spec.download_kind == "direct":
        dl_file(
            spec.source_urls[0],
            raw_paths[0],
            backup_urls=spec.backup_urls,
            expected_magic=HDF5_MAGIC,
            force=force,
        )
    elif spec.download_kind == "multi_direct":
        for url, dest in zip(spec.source_urls, raw_paths):
            dl_file(
                url,
                dest,
                expected_magic=HDF5_MAGIC,
                force=force,
            )
    elif spec.download_kind == "gdrive":
        assert spec.gdrive_id is not None
        dl_gdrive(
            spec.gdrive_id,
            raw_paths[0],
            expected_magic=HDF5_MAGIC,
            force=force,
        )
    else:
        raise ValueError(f"Unsupported download_kind: {spec.download_kind}")
    return raw_paths


def _prepare_unique_raw_files(
    pending_specs: Sequence[DatasetSpec],
    data_root: Path,
    *,
    skip_download: bool,
    force: bool,
    workers: int,
) -> None:
    """Ensure all unique raw inputs required by pending_specs exist without duplicate downloads."""
    if not pending_specs:
        return

    seen_dests: set[Path] = set()
    ordered_dests: list[Path] = []
    cfgen_groups: dict[tuple[str, tuple[str, ...]], dict[str, Path]] = {}
    download_jobs: list = []

    dl_file = _resolve_download_file()
    dl_gdrive = _resolve_download_gdrive_file()

    for spec in pending_specs:
        raw_paths = _get_raw_paths(spec, data_root)
        if spec.download_kind == "cfgen_zip64":
            group_key = (spec.source_urls[0], spec.backup_urls)
            member_map = cfgen_groups.setdefault(group_key, {})
            for member, dest in zip(spec.archive_members, raw_paths):
                resolved = dest.resolve()
                if resolved not in seen_dests:
                    seen_dests.add(resolved)
                    ordered_dests.append(dest)
                    member_map[member] = dest
        elif spec.download_kind == "direct":
            dest = raw_paths[0]
            resolved = dest.resolve()
            if resolved not in seen_dests:
                seen_dests.add(resolved)
                ordered_dests.append(dest)
                url = spec.source_urls[0]
                backups = spec.backup_urls
                download_jobs.append(
                    lambda u=url, b=backups, d=dest, fn=dl_file: fn(
                        u,
                        d,
                        backup_urls=b,
                        expected_magic=HDF5_MAGIC,
                        force=force,
                    )
                )
        elif spec.download_kind == "multi_direct":
            for url, dest in zip(spec.source_urls, raw_paths):
                resolved = dest.resolve()
                if resolved not in seen_dests:
                    seen_dests.add(resolved)
                    ordered_dests.append(dest)
                    download_jobs.append(
                        lambda u=url, d=dest, fn=dl_file: fn(
                            u,
                            d,
                            expected_magic=HDF5_MAGIC,
                            force=force,
                        )
                    )
        elif spec.download_kind == "gdrive":
            dest = raw_paths[0]
            resolved = dest.resolve()
            if resolved not in seen_dests:
                seen_dests.add(resolved)
                ordered_dests.append(dest)
                gid = spec.gdrive_id
                assert gid is not None
                download_jobs.append(
                    lambda g=gid, d=dest, fn=dl_gdrive: fn(
                        g,
                        d,
                        expected_magic=HDF5_MAGIC,
                        force=force,
                    )
                )
        else:
            raise ValueError(f"Unsupported download_kind: {spec.download_kind}")

    if skip_download:
        missing = [p for p in ordered_dests if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing raw files with --skip-download: {missing}"
            )
        return

    for (url, backups), member_map in cfgen_groups.items():
        if member_map:
            download_jobs.append(
                lambda u=url, b=backups, m=member_map: extract_zip_entries_via_http_range(
                    u,
                    m,
                    backup_urls=b,
                    force=force,
                    max_workers=workers,
                )
            )

    if not download_jobs:
        return
    if workers <= 1 or len(download_jobs) == 1:
        for job in download_jobs:
            job()
        return

    with ThreadPoolExecutor(max_workers=min(workers, len(download_jobs))) as pool:
        futures = [pool.submit(job) for job in download_jobs]
        for fut in as_completed(futures):
            fut.result()


__all__ = [
    "DEFAULT_HTTP_HEADERS",
    "HDF5_MAGIC",
    "ZIP_MAGIC",
    "_ZipEntryMeta",
    "_build_request",
    "_download_file",
    "_download_gdrive_file",
    "_ensure_raw_files",
    "_http_range_get",
    "_parse_total_size_from_content_range",
    "_parse_zip64_extra",
    "_prepare_unique_raw_files",
    "_read_remote_zip_directory",
    "_select_genept_zip_member",
    "_stream_remote_zip_entry",
    "ensure_genept_embeddings",
    "extract_zip_entries_via_http_range",
]
