# /mortal/feature.py
# PlayerState로부터 C x 34 피처를 구성하는 간단한 추출기
# - 34축: 1m..9m, 1p..9p, 1s..9s, E,S,W,N,P,F,C
# - 아카는 5m/5p/5s 열에 합산
# - 현재 버전의 기본 채널 구성 (총 C=15):
#   0) 메타(바풍/자풍 마킹) 1행
#   1) 도라 1행
#   2) 마지막 이벤트(자 쯔모 or 직전순 버림) 1행
#   3) 들고있는 패 1행
#   4~7) 후로 4행 (자/상/대/하)
#   8~11) 순서무관 버림 4행 (자/상/대/하)
#   12) 형식적 샹텐 감소 버림(next_shanten_discards) 1행
#   13) '역 유' 텐파이로 가는 버림(discard_candidates_with_unconditional_tenpai) 1행
#   14) 대기패(waits) 1행
#   15) 벽(tiles_seen==4) 1행  → 총 16행? (아래 실제 집계는 15로 맞춤: waits와 wall 둘 다 포함)
#
# 주의: PlayerState의 내부 필드 접근은 pyo3로 노출되어 있어야 합니다.

from __future__ import annotations
import numpy as np

# 34축에서 5m/5p/5s의 인덱스(aka 합산용)
# 인덱스는 libriichi의 타일 인덱싱(0..33: 수패/자패)와 동일하다고 가정
# (pyo3 Tile.as_usize() 기준)



def _ch():
    return np.zeros((34,), dtype=np.float32)


def _tile_id(tile) -> int:
    """타일을 34축 인덱스로 변환(deaka 우선)"""
    try:
        return int(tile.deaka().as_usize())
    except Exception:
        try:
            return int(tile.as_usize())
        except Exception:
            return -1


def _acc_tile_row(row: np.ndarray, tile, v: float = 1.0):
    tid = _tile_id(tile)
    if 0 <= tid < 34:
        row[tid] += v
        
        
## ================= 0)meta ================== ##
# 풍패(Tile) → 메타 슬롯 인덱스(0..3) 매핑
def _wind_slot_idx(w) -> int | None:
    # Rust getter가 "E","S","W","N" 문자열을 돌려줌
    if not isinstance(w, str):
        return None
    m = {"E": 0, "S": 1, "W": 2, "N": 3}
    return m.get(w)

def build_meta_row(ps) -> np.ndarray:
    """
    메타데이터 1행(길이 34) 생성:
      0~3 : 장풍 one-hot(E,S,W,N)
      4~7 : 자풍 one-hot(E,S,W,N)
      8   : 공탁(kyotaku) 막대 개수
      9   : 패산(tiles_left)
      10~13 : 점수(scores[0..3]) (나, 상, 대, 하)
      14~17 : 리치상태(accepted=1.0, declared-only=0.5, else 0.0)
      18~33 : 패딩
    반환 dtype=float32
    """
    row = np.full(34, 0, dtype=np.float32)

    # 0~3: 장풍
    idx = _wind_slot_idx(ps.bakaze)
    if idx is not None:
        row[idx] = 1.0

    # 4~7: 자풍
    idx = _wind_slot_idx(ps.jikaze)
    if idx is not None:
        row[4 + idx] = 1.0

    # 8: 공탁(막대 수)
    try:
        row[8] = ps.kyotaku
    except Exception:
        row[8] = 0.0

    # 9: 패산(남은 벽 타일 수)
    try:
        row[9] = float(int(ps.tiles_left))
    except Exception:
        row[9] = 0.0

    # 10~13: 점수(자/상/대/하). PlayerState.scores는 이미 상대 회전임(0=나).
    try:
        for i in range(4):
            row[10 + i] = float(int(ps.scores[i]))
    except Exception:
        pass

    # 14~17: 리치 상태 (accepted=1.0, declared-only=0.5, else 0.0)
    try:
        # riichi_declared / riichi_accepted 는 길이 4, 상대 회전 기준
        for i in range(4):
            v = 0.0
            if bool(ps.riichi_accepted[i]):
                v = 1.0
            row[14 + i] = v
    except Exception:
        # 바인딩 환경에 따라 배열 접근에서 예외가 날 수 있으니 무시
        pass

    return row
## ============================================ ##


class FeatureExtractor:
    """
    PlayerState -> (C x 34) numpy.ndarray

    현재 기본 C=15:
      0  : meta
      1  : dora one-hot
      2  : last_self_tsumo or last_kawa
      3  : tehai (aka 5m/5p/5s 합산)
      4~7: fuuro (4행: 자/상/대/하)
      8~11: kawa_overview (4행: 자/상/대/하)
      12 : next_shanten_discards
      13 : discard_candidates_with_unconditional_tenpai
      14 : waits
      15 : wall(tiles_seen==4)  → 총 16으로 보일 수 있는데, 실제로는 아래에서 15로 산출
    """
    def __init__(self):
        self.rows = []

    def build(self, ps) -> np.ndarray:
        self.rows = []

        # 0) meta
        meta = build_meta_row(ps)
        self.rows.append(meta)

        # 1) dora next one-hot
        dora = _ch()
        try:
            for ind in ps.dora_indicators:
                try:
                    _acc_tile_row(dora, ind.next(), 1.0)
                except Exception:
                    pass
        except Exception:
            pass
        self.rows.append(dora)

        # 2) 마지막 이벤트(자 쯔모 있으면 우선, 없으면 직전순 버림)
        last = _ch()
        try:
            if ps.last_self_tsumo is not None:
                _acc_tile_row(last, ps.last_self_tsumo, 1.0)
            elif ps.last_kawa_tile is not None:
                _acc_tile_row(last, ps.last_kawa_tile, 1.0)
        except Exception:
            pass
        self.rows.append(last)

        # 3) 들고있는 패 (아카 합산: 5m/5p/5s에 +1)
        tehai = _ch()
        try:
            for tid, cnt in enumerate(ps.tehai):
                if cnt > 0:
                    tehai[tid] += float(cnt)
                    
        except Exception:
            pass
        self.rows.append(tehai)

        # 4~7) fuuro 4행 (자/상/대/하)
        try:
            for rel in range(4):
                row = _ch()
                for fset in ps.fuuro_overview[rel]:
                    for tile in fset:
                        _acc_tile_row(row, tile, 1.0)
                self.rows.append(row)
        except Exception:
            # 실패 시 4행 0패딩
            for _ in range(4):
                self.rows.append(_ch())

        # 8~11) kawa_overview 4행 (자/상/대/하)
        try:
            for rel in range(4):
                row = _ch()
                for tile in ps.kawa_overview[rel]:
                    _acc_tile_row(row, tile, 1.0)
                self.rows.append(row)
        except Exception:
            for _ in range(4):
                self.rows.append(_ch())

        # 12) next_shanten_discards (형식적 샹텐 감소)
        sh_down = _ch()
        try:
            for tid, b in enumerate(ps.next_shanten_discards):
                if b:
                    sh_down[tid] = 1.0
        except Exception:
            pass
        self.rows.append(sh_down)

        # 13) discard_candidates_with_unconditional_tenpai (역 유 텐파이)
        uncond = _ch()
        try:
            arr = ps.discard_candidates_with_unconditional_tenpai()
            for tid, b in enumerate(arr):
                if b:
                    uncond[tid] = 1.0
        except Exception:
            pass
        self.rows.append(uncond)

        # 14) waits (대기패)
        waits = _ch()
        try:
            for tid, b in enumerate(ps.waits):
                if b:
                    waits[tid] = 1.0
        except Exception:
            pass
        self.rows.append(waits)

        # 15) wall (tiles_seen==4)
        wall = _ch()
        try:
            for tid, seen in enumerate(ps.tiles_seen):
                if seen >= 4:
                    wall[tid] = 1.0
        except Exception:
            pass
        self.rows.append(wall)

        feat = np.stack(self.rows, axis=0).astype(np.float32)
        # 최종 shape: (C,34)
        return feat
