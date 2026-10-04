import re


def mmss(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^\w]+", "-", s, flags=re.UNICODE).strip("-")[:n] or "track"
