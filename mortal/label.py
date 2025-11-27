# /mortal/label.py
from __future__ import annotations
from typing import Any, Optional, Union, Tuple
import json, re

def _as_json(obj: Union[str, dict, None]) -> Optional[dict]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None

# ---- tile utils ----
_TILE_RE = re.compile(r"^(?P<num>[1-9])(?P<suit>[mps])(?P<aka>r)?$")

def _parse_tile_id(tile_str: str) -> Optional[Tuple[int, str]]:
    """예: '5mr' -> (5, 'm')"""
    if not isinstance(tile_str, str):
        return None
    m = _TILE_RE.match(tile_str)
    if not m:
        return None
    num = int(m.group("num"))
    suit = m.group("suit")
    return num, suit

def _classify_chi(pai: str, consumed: list[str]) -> Optional[str]:
    """
    chi를 low/mid/high로 분류.
    세 타일을 정렬하여 a,a+1,a+2이면
      pai==a   -> chi_low
      pai==a+1 -> chi_mid
      pai==a+2 -> chi_high
    """
    pc = _parse_tile_id(pai)
    if not pc:
        return None
    pn, ps = pc

    if not isinstance(consumed, list) or len(consumed) != 2:
        return None
    c0 = _parse_tile_id(consumed[0])
    c1 = _parse_tile_id(consumed[1])
    if not c0 or not c1:
        return None

    (n0, s0), (n1, s1) = c0, c1
    if not (ps == s0 == s1):
        return None  # 수트 달라 chi 아님

    trio = sorted([pn, n0, n1])
    a, b, c = trio
    if not (a + 1 == b and b + 1 == c):
        return None  # 연속 아님

    if pn == a:
        return "chi_low"
    elif pn == b:
        return "chi_mid"
    elif pn == c:
        return "chi_high"
    return None

# ---- public API ----
def get_label(next_row: Union[str, dict, None], player_id: int) -> Optional[str]:
    """
    다음 이벤트로 라벨을 만든다.
      - type == "dahai" : pai 그대로 반환 ("7m")
      - type == "chi"   : "pai|chi_low" 와 같이 pai와 chi형태 동시 반환
    """
    ev = _as_json(next_row)
    if not ev:
        return None

    typ = ev.get("type")
    actor = ev.get("actor")
    if actor is not None and actor != player_id:
        return None

    if typ == "dahai":
        pai = ev.get("pai")
        return pai if isinstance(pai, str) else None

    elif typ == "chi":
        pai = ev.get("pai")
        consumed = ev.get("consumed")
        if isinstance(pai, str) and isinstance(consumed, list):
            chi_kind = _classify_chi(pai, consumed)
            if chi_kind:
                return f"{pai}|{chi_kind}"
            return pai  # fallback: chi형태 못구분 시 pai만 반환
        
    if typ == "tsumo":
        return ""
        
    else:
        pai = ev.get("pai")
        return pai if isinstance(pai, str) else None

    return None
