# /mortal/preprocess.py
import argparse, json, gzip, re, sys, os, time
from pathlib import Path
from typing import Iterable, Optional, List, Union
import numpy as np

# numpy pickle-safe (optional)
try:
    from torch.serialization import add_safe_globals
    import numpy as _np
    add_safe_globals([_np.core.multiarray.scalar])
except Exception:
    pass

# ---- PlayerState 바인딩 ----
def _import_player_state():
    tried = []
    for modpath in ("libriichi.mjai", "libriichi.state", "libriichi"):
        try:
            mod = __import__(modpath, fromlist=["PlayerState"])
            if hasattr(mod, "PlayerState"):
                return getattr(mod, "PlayerState")
        except Exception as e:
            tried.append((modpath, repr(e)))
    print("[FATAL] cannot import PlayerState from libriichi.*", file=sys.stderr)
    for m, err in tried:
        print(f"  tried {m} -> {err}", file=sys.stderr)
    sys.exit(2)


PlayerState = _import_player_state()

# ---- feature / label 모듈 ----
try:
    from feature import FeatureExtractor
except Exception as e:
    print(f"[FATAL] cannot import feature.FeatureExtractor: {e}", file=sys.stderr)
    sys.exit(5)

try:
    from label import get_label
except Exception as e:
    print(f"[FATAL] cannot import label.get_label: {e}", file=sys.stderr)
    sys.exit(6)

PathLike = Union[str, Path]


def _is_mjson_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".mjson"


def _is_gzip(path: Path) -> bool:
    if path.suffix.lower() == ".mjson":
        return True
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False


def _open_and_fix_start_game(path: PathLike) -> Iterable[str]:
    p = Path(path)
    opener = gzip.open if _is_gzip(p) else open
    with opener(p, "rt", encoding="utf-8-sig", newline="") as f:
        for line in f:
            if '"type":"start_game"' in line:
                try:
                    obj = json.loads(line)
                    obj["names"] = ["0", "1", "2", "3"]
                    line = json.dumps(obj, ensure_ascii=False)
                except Exception:
                    line = re.sub(
                        r'"names"\s*:\s*\[[^\]]*\]', '"names":["0","1","2","3"]', line
                    )
            if not line.endswith("\n"):
                line += "\n"
            yield line


def _parse_players(arg: Optional[str]) -> List[int]:
    if not arg:
        return [0, 1, 2, 3]
    ret: List[int] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v not in (0, 1, 2, 3):
            raise ValueError("--players 는 0~3만 허용")
        ret.append(v)
    return sorted(set(ret))


def _pack_mask_bits(mask) -> int:
    """ndarray(bool)[ACTION_SPACE] -> int 비트플래그 (0~45)."""
    bits = 0
    for i, m in enumerate(mask):
        try:
            if bool(m):
                bits |= (1 << i)
        except Exception:
            pass
    return bits


def _load_all_lines_and_json(in_path: Path):
    """로그 전체를 메모리로 올리고, 각 라인에 대한 json 파싱까지 해둔다."""
    lines = list(_open_and_fix_start_game(in_path))
    jsons: List[Optional[dict]] = []
    for ln in lines:
        s = ln.strip()
        try:
            js = json.loads(s) if s else None
        except Exception:
            js = None
        jsons.append(js)
    return lines, jsons


def _ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


class TimerAgg:
    __slots__ = ("n", "t_update", "t_encode", "t_feat", "t_write", "t_total")

    def __init__(self) -> None:
        self.n = 0
        self.t_update = 0.0
        self.t_encode = 0.0
        self.t_feat = 0.0
        self.t_write = 0.0
        self.t_total = 0.0

    def add(self, upd, enc, fea, wri, tot):
        self.n += 1
        self.t_update += upd
        self.t_encode += enc
        self.t_feat += fea
        self.t_write += wri
        self.t_total += tot

    def pct(self, x):
        return (x / self.t_total * 100.0) if self.t_total > 0 else 0.0

    def summary_lines(self, prefix=""):
        return [
            f"{prefix}count={self.n}",
            f"{prefix}total={self.t_total:.4f}s",
            f"{prefix}update={self.t_update:.4f}s ({self.pct(self.t_update):.1f}%)",
            f"{prefix}encode={self.t_encode:.4f}s ({self.pct(self.t_encode):.1f}%)",
            f"{prefix}feature={self.t_feat:.4f}s ({self.pct(self.t_feat):.1f}%)",
            f"{prefix}write={self.t_write:.4f}s ({self.pct(self.t_write):.1f}%)",
            (
                f"{prefix}avg/row={self.t_total/self.n:.6f}s"
                if self.n
                else f"{prefix}avg/row=nan"
            ),
        ]


def _process_player(
    in_path: Path,
    out_dir: Path,
    rel_parent: Path,
    player_id: int,
    version: int,
    *,
    profile: bool,
    profile_each: bool,
    profile_interval: int,
    profile_sink=None,
) -> Path:
    """하나의 .mjson + 하나의 플레이어에 대해 CSV 1개 생성."""
    out_dir2 = out_dir / rel_parent
    _ensure_parent_dir(out_dir2 / "dummy")
    out_path = out_dir2 / f"{in_path.stem}.p{player_id}.feat.csv"

    lines, jsons = _load_all_lines_and_json(in_path)

    st = PlayerState(player_id)
    update_func = getattr(st, "update_json", None) or getattr(st, "update", None)
    if update_func is None:
        print("[FATAL] PlayerState has neither update_json nor update", file=sys.stderr)
        sys.exit(3)
    if not hasattr(st, "encode_obs") and not hasattr(st, "encode_mask_only"):
        print("[FATAL] PlayerState.encode_obs/encode_mask_only 가 필요합니다.", file=sys.stderr)
        sys.exit(4)

    extractor = FeatureExtractor()
    agg = TimerAgg()

    with open(out_path, "w", encoding="utf-8") as w:
        for i, line in enumerate(lines):
            start_total = time.perf_counter()
            upd = enc = fea = wri = 0.0

            sline = line.rstrip("\n")
            if not sline:
                t0 = time.perf_counter()
                # 빈 라인도 1 row 로 유지
                w.write('0,""\n')
                wri += time.perf_counter() - t0
                tot = time.perf_counter() - start_total
                if profile:
                    agg.add(upd, enc, fea, wri, tot)
                    if profile_each:
                        print(
                            f"[DEBUG p{player_id} #{i+1}] "
                            f"update={upd:.6f} encode={enc:.6f} "
                            f"feat={fea:.6f} write={wri:.6f} total={tot:.6f}"
                        )
                continue

            # 1) 상태 업데이트
            t0 = time.perf_counter()
            try:
                update_func(sline)
            except Exception as e:
                print(
                    f"[warn] p{player_id} update failed at #{i+1} ({in_path.name}): {e}",
                    file=sys.stderr,
                )
            upd = time.perf_counter() - t0

            # 2) mask만 인코딩 → 46차원 bool
            mask_bits = 0
            t0 = time.perf_counter()
            try:
                if hasattr(st, "encode_mask_only"):
                    mask = st.encode_mask_only(version, False)
                else:
                    # fallback: encode_obs 후 mask만 사용
                    _, mask = st.encode_obs(version, False)
                mask_bits = _pack_mask_bits(mask)
            except Exception as e:
                print(
                    f"[warn] p{player_id} encode_mask_only failed at #{i+1} ({in_path.name}): {e}",
                    file=sys.stderr,
                )
                mask_bits = 0
            enc = time.perf_counter() - t0

            # 3) label (다음 row의 json 기반)
            nxt = jsons[i + 1] if (i + 1) < len(jsons) else None
            label_str = get_label(nxt, player_id) or ""

            # 4) feature: mask_bits==0 이라도 row는 유지 (feature만 생략 가능)
            t0 = time.perf_counter()
            flat = None
            if mask_bits != 0:
                try:
                    feat = extractor.build(st)  # (C, 34) 기대
                    feat = np.asarray(feat, dtype=float)
                    if feat.ndim != 2 or feat.shape[1] != 34:
                        raise ValueError(
                            f"feature shape must be (C,34), got {feat.shape}"
                        )
                    flat = feat.reshape(-1).tolist()
                except Exception as e:
                    print(
                        f"[warn] p{player_id} feature build failed at #{i+1} ({in_path.name}): {e}",
                        file=sys.stderr,
                    )
                    flat = None
            fea = time.perf_counter() - t0

            # 5) 파일 쓰기
            t0 = time.perf_counter()
            # 항상 1row 출력 (mask_bits, [features...], "label")
            w.write(str(mask_bits))
            if flat is not None:
                for v in flat:
                    w.write("," + ("%.6g" % float(v)))
            w.write(f',"{label_str}"\n')
            wri = time.perf_counter() - t0

            tot = time.perf_counter() - start_total
            if profile:
                agg.add(upd, enc, fea, wri, tot)
                if profile_each:
                    print(
                        f"[DEBUG p{player_id} #{i+1}] "
                        f"update={upd:.6f} encode={enc:.6f} "
                        f"feat={fea:.6f} write={wri:.6f} total={tot:.6f}"
                    )
                if profile_interval > 0 and (i + 1) % profile_interval == 0:
                    # 중간 요약
                    for ln in agg.summary_lines(prefix=f"[MID p{player_id} #{i+1}] "):
                        print(ln)
                        if profile_sink is not None:
                            profile_sink.write(ln + "\n")

    # 플레이어 단위 요약 출력/저장
    if profile:
        lines = agg.summary_lines(prefix=f"[SUMMARY p{player_id}] ")
        for ln in lines:
            print(ln)
        if profile_sink is not None:
            for ln in lines:
                profile_sink.write(ln + "\n")
    return out_path


def _process_one_input_file(
    in_path: Path,
    out_root: Path,
    base_root: Path,
    players: List[int],
    version: int,
    profargs,
) -> List[str]:
    """하나의 .mjson 파일에 대해 p0~p3까지 처리."""
    rel_parent = in_path.parent.relative_to(base_root)
    outs = []
    for pid in players:
        p = _process_player(
            in_path,
            out_root,
            rel_parent,
            pid,
            version,
            profile=profargs["profile"],
            profile_each=profargs["profile_each"],
            profile_interval=profargs["profile_interval"],
            profile_sink=profargs["sink"],
        )
        outs.append(str(p))
    return outs


def _worker_entry(args):
    """multiprocessing용 wrapper."""
    (
        in_path,
        out_root,
        base_root,
        players,
        version,
        profargs,
    ) = args
    in_path = Path(in_path)
    out_root = Path(out_root)
    base_root = Path(base_root)
    return _process_one_input_file(
        in_path, out_root, base_root, players, version, profargs
    )


def _collect_input_files(inp: Path) -> List[Path]:
    if inp.is_file():
        return [inp] if _is_mjson_file(inp) else []
    if inp.is_dir():
        return sorted([p for p in inp.rglob("*.mjson") if p.is_file()])
    return []


def run(
    inp: str,
    out_dir: Optional[str],
    players: List[int],
    version: int,
    workers: int,
    profargs,
) -> dict:
    in_path = Path(inp)
    files = _collect_input_files(in_path)
    if not files:
        raise FileNotFoundError(f".mjson 입력을 찾지 못했습니다: {in_path}")

    out_root = Path(out_dir) if out_dir else (
        in_path if in_path.is_dir() else in_path.parent
    )
    out_root.mkdir(parents=True, exist_ok=True)

    base_root = in_path if in_path.is_dir() else in_path.parent
    all_outputs: List[str] = []

    print(f"[preprocess] mode={'DIR' if in_path.is_dir() else 'FILE'}  root={in_path}")
    print(f"[preprocess] out_root={out_root}")
    print(f"[preprocess] players={players}  version={version}")
    print(f"[preprocess] workers={workers}")
    sys.stdout.flush()

    # 진행상태 출력만 먼저 해 둠 (병렬 여부와 무관)
    for idx, f in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {f}")
    sys.stdout.flush()

    # workers==1 → 순차 처리, >1 → 파일 단위 멀티프로세스
    if workers <= 1:
        for f in files:
            outs = _process_one_input_file(
                f, out_root, base_root, players, version, profargs
            )
            all_outputs.extend(outs)
    else:
        from multiprocessing import Pool

        # profile_out 은 파일 핸들이라 프로세스에 그대로 넘기기 애매하니, worker 에서는 sink=None
        worker_profargs = {
            "profile": bool(profargs["profile"]),
            "profile_each": bool(profargs["profile_each"]),
            "profile_interval": int(profargs["profile_interval"]),
            "sink": None,
        }
        tasks = [
            (str(f), str(out_root), str(base_root), players, version, worker_profargs)
            for f in files
        ]
        with Pool(processes=workers) as pool:
            for outs in pool.imap_unordered(_worker_entry, tasks):
                all_outputs.extend(outs)

    return {
        "mode": "dir" if in_path.is_dir() else "file",
        "input": str(in_path),
        "num_inputs": len(files),
        "outputs": all_outputs,
        "version": version,
        "players": players,
        "workers": workers,
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "MJAI 로그(.mjson) → (mask_bits, C×34 feature, label) CSV 생성 "
            "(파일/폴더 지원, 멀티프로세스 + 프로파일링 지원)"
        )
    )
    ap.add_argument("--in", dest="inp", required=True, help="입력 파일 또는 폴더(.mjson)")
    ap.add_argument("--out", dest="out_dir", default=None, help="출력 루트 디렉터리")
    ap.add_argument("--players", dest="players", default=None, help="예: 0,1,2,3")
    ap.add_argument("--version", dest="version", type=int, default=4)

    # --- multiprocessing options ---
    ap.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=1,
        help="동시에 처리할 프로세스 수(파일 단위 병렬, 기본=1=단일 프로세스)",
    )

    # --- profiling options ---
    ap.add_argument("--profile", action="store_true", help="구간별 시간 누적 요약 출력")
    ap.add_argument(
        "--profile-each",
        action="store_true",
        help="각 라인별 상세 로그 출력(매우 많음)",
    )
    ap.add_argument(
        "--profile-interval",
        type=int,
        default=0,
        help="N 라인마다 중간 요약 출력",
    )
    ap.add_argument(
        "--profile-out",
        type=str,
        default=None,
        help="요약을 파일로도 저장(단일 프로세스 혹은 parent만 기록)",
    )

    args = ap.parse_args()

    sink = None
    if args.profile_out:
        sink = open(args.profile_out, "w", encoding="utf-8")

    profargs = {
        "profile": bool(args.profile),
        "profile_each": bool(args.profile_each),
        "profile_interval": int(args.profile_interval),
        "sink": sink,
    }

    try:
        res = run(
            args.inp,
            args.out_dir,
            _parse_players(args.players),
            int(args.version),
            int(args.workers),
            profargs,
        )
        print(json.dumps(res, ensure_ascii=False))
    finally:
        if sink:
            try:
                sink.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
