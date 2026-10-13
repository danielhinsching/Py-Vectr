"""Geometria de contornos: leitura/escrita do atributo `d` do SVG e "endireitar bordas".

Um subcaminho (sempre fechado) é representado como (start, segs), onde cada seg é
    ('L', p)            linha até p
    ('C', c1, c2, p)    bézier cúbica até p
com pontos como tuplas (x, y).
"""

from __future__ import annotations

import math
import re

import numpy as np

_TOK = re.compile(r"[MmLlHhVvCcQqZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[SsTtAa]")


def _fmt(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# --------------------------------------------------------------------------- #
# Conversões
# --------------------------------------------------------------------------- #

def parse_d(d: str) -> list | None:
    """Lê um `d` com comandos M/L/H/V/C/Q/Z (absolutos ou relativos). None se não suportado."""
    toks = _TOK.findall(d)
    subs: list = []
    segs = None
    cur = start = (0.0, 0.0)
    cmd = None
    i = 0

    def num():
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    try:
        while i < len(toks):
            if toks[i].isalpha():
                cmd = toks[i]
                i += 1
                if cmd in "Zz":
                    cur = start
                    continue
            if cmd is None or cmd in "SsTtAa":
                return None
            rel = cmd.islower()
            ox, oy = cur if rel else (0.0, 0.0)
            c = cmd.upper()
            if c == "M":
                cur = start = (num() + ox, num() + oy)
                segs = []
                subs.append((start, segs))
                cmd = "l" if rel else "L"
            elif c == "L":
                cur = (num() + ox, num() + oy)
                segs.append(("L", cur))
            elif c == "H":
                cur = (num() + ox, cur[1])
                segs.append(("L", cur))
            elif c == "V":
                cur = (cur[0], num() + oy)
                segs.append(("L", cur))
            elif c == "C":
                c1 = (num() + ox, num() + oy)
                c2 = (num() + ox, num() + oy)
                cur = (num() + ox, num() + oy)
                segs.append(("C", c1, c2, cur))
            elif c == "Q":
                q = (num() + ox, num() + oy)
                p0, p = cur, (num() + ox, num() + oy)
                c1 = (p0[0] + 2 / 3 * (q[0] - p0[0]), p0[1] + 2 / 3 * (q[1] - p0[1]))
                c2 = (p[0] + 2 / 3 * (q[0] - p[0]), p[1] + 2 / 3 * (q[1] - p[1]))
                cur = p
                segs.append(("C", c1, c2, cur))
    except (IndexError, ValueError, TypeError, AttributeError):
        return None
    return subs


def from_potrace(curve) -> tuple:
    sp = curve.start_point
    segs = []
    for seg in curve:
        if seg.is_corner:
            segs.append(("L", (seg.c.x, seg.c.y)))
            segs.append(("L", (seg.end_point.x, seg.end_point.y)))
        else:
            segs.append(("C", (seg.c1.x, seg.c1.y), (seg.c2.x, seg.c2.y),
                         (seg.end_point.x, seg.end_point.y)))
    return (sp.x, sp.y), segs


def to_d(subs: list, s: float = 1.0) -> str:
    out = []
    for start, segs in subs:
        out.append(f"M{_fmt(start[0] * s)} {_fmt(start[1] * s)}")
        for seg in segs:
            if seg[0] == "L":
                out.append(f"L{_fmt(seg[1][0] * s)} {_fmt(seg[1][1] * s)}")
            else:
                _, c1, c2, p = seg
                out.append(f"C{_fmt(c1[0] * s)} {_fmt(c1[1] * s)} {_fmt(c2[0] * s)} {_fmt(c2[1] * s)} "
                           f"{_fmt(p[0] * s)} {_fmt(p[1] * s)}")
        out.append("Z")
    return "".join(out)


def _bez(p0, c1, c2, p, n):
    t = np.linspace(0, 1, n + 1)[1:, None]
    mt = 1 - t
    return (mt ** 3 * np.array(p0) + 3 * mt ** 2 * t * np.array(c1)
            + 3 * mt * t ** 2 * np.array(c2) + t ** 3 * np.array(p))


def flatten(sub: tuple, s: float = 1.0, max_len: float = 0.5) -> list[tuple[float, float]]:
    """Polilinha fechada (sem repetir o ponto inicial); linhas viram 1 vértice, curvas são
    subdivididas em trechos de no máx. `max_len` (unidades já escaladas por `s`)."""
    start, segs = sub
    pts = [start]
    cur = start
    for seg in segs:
        if seg[0] == "L":
            pts.append(seg[1])
        else:
            _, c1, c2, p = seg
            est = (math.dist(cur, c1) + math.dist(c1, c2) + math.dist(c2, p)) * s
            n = max(2, min(200, int(est / max_len) + 1))
            pts.extend(map(tuple, _bez(cur, c1, c2, p, n)))
        cur = pts[-1]
    if len(pts) > 1 and math.dist(pts[0], pts[-1]) < 1e-9:
        pts.pop()
    return [(x * s, y * s) for x, y in pts]


def _sample(sub: tuple, h: float) -> np.ndarray:
    """Amostra o contorno com espaçamento uniforme h."""
    start, segs = sub
    dense = [np.array([start], float)]
    cur = start
    for seg in segs:
        if seg[0] == "L":
            p = seg[1]
            n = max(1, int(math.dist(cur, p) / (h * 0.5)) + 1)
            t = np.linspace(0, 1, n + 1)[1:, None]
            dense.append(np.array(cur) + t * (np.array(p) - np.array(cur)))
        else:
            _, c1, c2, p = seg
            est = math.dist(cur, c1) + math.dist(c1, c2) + math.dist(c2, p)
            dense.append(_bez(cur, c1, c2, p, max(2, int(est / (h * 0.5)) + 1)))
            cur = p
            continue
        cur = p
    D = np.concatenate(dense)
    if np.linalg.norm(D[-1] - D[0]) > 1e-9:
        D = np.vstack([D, D[:1]])
    seglen = np.linalg.norm(np.diff(D, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seglen)])
    total = cum[-1]
    if total <= 0:
        return D[:1]
    n = max(3, int(total / h))
    s = np.linspace(0, total, n, endpoint=False)
    return np.stack([np.interp(s, cum, D[:, 0]), np.interp(s, cum, D[:, 1])], 1)


# --------------------------------------------------------------------------- #
# Endireitar bordas
# --------------------------------------------------------------------------- #

def _dp(Q: np.ndarray, eps: float) -> list:
    """Douglas-Peucker (iterativo): mantém os pontos necessários para desvio <= eps."""
    if len(Q) < 3:
        return list(Q)
    keep = np.zeros(len(Q), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(Q) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        v = Q[b] - Q[a]
        nv = np.linalg.norm(v)
        seg = Q[a + 1:b] - Q[a]
        d = np.abs(seg[:, 0] * v[1] - seg[:, 1] * v[0]) / nv if nv > 1e-12 else np.linalg.norm(seg, axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return list(Q[keep])


def straighten(sub: tuple, tol: float, unit: float = 1.0, snap_deg: float = 0.0) -> tuple | None:
    """Troca trechos quase retos do contorno por retas perfeitas, com cantos vivos na
    interseção das retas vizinhas. Curvas de verdade (arcos) são preservadas.

    tol      desvio máximo aceito para considerar um trecho reto (unidades do caminho)
    unit     tamanho de 1 pixel da imagem original nessas unidades (escala do ruído)
    snap_deg retas a menos de X graus da horizontal/vertical são alinhadas exatamente
    Retorna None se nada mudou.
    """
    h = max(0.25 * unit, min(tol / 3.0, 2.0 * unit))
    P = _sample(sub, h)
    n = len(P)
    lmin = max(4.0 * tol, 8.0 * unit)
    minpts = max(6, int(math.ceil(lmin / h)))
    if n < 2 * minpts:
        return None
    curv_tol = 0.45 * tol + 0.15 * unit

    # começa no canto mais agudo, para não partir uma reta ao meio
    k = max(2, minpts // 2)
    v1 = P - np.roll(P, k, 0)
    v2 = np.roll(P, -k, 0) - P
    ang = np.abs(np.arctan2(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0], (v1 * v2).sum(1)))
    P = np.roll(P, -int(np.argmax(ang)), 0)
    P2 = np.concatenate([P, P])

    def fit(i: int, j: int):
        Q = P2[i:j]
        c = Q.mean(0)
        X = Q - c
        w, V = np.linalg.eigh(X.T @ X)
        u, nr = V[:, 1], V[:, 0]
        d = X @ nr
        if np.abs(d).max() > tol:
            return None
        # arco x reta: o desvio de um arco em relação à reta tem forma de parábola
        t = X @ u
        tn = t / (np.abs(t).max() + 1e-9)
        m = tn * tn
        a11, a12, a22 = (m * m).sum(), m.sum(), float(len(m))
        det = a11 * a22 - a12 * a12
        if det > 1e-12:
            a = (a22 * (m * d).sum() - a12 * d.sum()) / det
            if abs(a) > curv_tol:
                return None
        return c, u

    runs = []
    i = 0
    step_fail = max(1, minpts // 4)
    while i < n:
        limit = n - i
        if limit < minpts or fit(i, i + minpts) is None:
            i += step_fail
            continue
        lo, hi = minpts, minpts * 2
        while hi <= limit and fit(i, i + hi) is not None:
            lo, hi = hi, hi * 2
        hi = min(hi, limit + 1)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fit(i, i + mid) is not None:
                lo = mid
            else:
                hi = mid
        runs.append([i, i + lo])
        i += lo
    if not runs:
        return None

    gap_small = max(3, int(4.0 * tol / h) + 3)

    def line_of(a: int, b: int) -> dict:
        """Reta final do trecho (sem as pontas, que podem invadir o canto)."""
        trim = min((b - a) // 6, max(1, int(tol / h)))
        Q = P2[a + trim:b - trim] if b - a - 2 * trim >= 3 else P2[a:b]
        for _ in range(3):  # robusto: ignora pontas que invadem cantos/arcos
            c = Q.mean(0)
            _, V = np.linalg.eigh((Q - c).T @ (Q - c))
            u = V[:, 1]
            dev = np.abs((Q - c) @ V[:, 0])
            keep = dev <= max(0.6 * tol, 1.0 * unit)
            if keep.all() or keep.sum() < max(3, len(Q) // 2):
                break
            Q = Q[keep]
        if np.dot(u, P2[b - 1] - P2[a]) < 0:
            u = -u
        return {"c": c, "u": u, "a": a, "b": b}

    def turn(l1: dict, l2: dict) -> float:
        u, v = l1["u"], l2["u"]
        return math.atan2(u[0] * v[1] - u[1] * v[0], float(np.dot(u, v)))

    def gap_between(i: int, lst: list) -> int:
        nxt = lst[(i + 1) % len(lst)]
        return nxt["a"] + (n if i == len(lst) - 1 else 0) - lst[i]["b"]

    lines = [line_of(a, b) for a, b in runs]

    # 1) junta retas vizinhas quase colineares (uma reta partida em duas)
    merged = True
    while merged and len(lines) > 1:
        merged = False
        for i in range(len(lines) - 1):
            if gap_between(i, lines) <= gap_small and abs(turn(lines[i], lines[i + 1])) < math.radians(4):
                cand = line_of(lines[i]["a"], lines[i + 1]["b"])
                Q = P2[cand["a"]:cand["b"]] - cand["c"]
                if (np.abs(Q @ np.array([-cand["u"][1], cand["u"][0]])) <= tol).mean() >= 0.95:
                    lines[i:i + 2] = [cand]
                    merged = True
                    break

    # 2) correntes de retas curtas girando sempre para o mesmo lado = arco facetado
    def circle_resid(a: int, b: int) -> float:
        Q = P2[a:b]
        x, y = Q[:, 0], Q[:, 1]
        M = np.stack([x, y, np.ones_like(x)], 1)
        sol, *_ = np.linalg.lstsq(M, x * x + y * y, rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-12))
        return float(np.abs(np.hypot(x - cx, y - cy) - r).max())

    Ln = len(lines)
    if Ln >= 2:
        link = []
        for i in range(Ln):
            if Ln == 2 and i == 1:
                link.append(0)
                break
            th = turn(lines[i], lines[(i + 1) % Ln])
            ok = gap_between(i, lines) <= gap_small and abs(th) < math.radians(45)
            link.append(int(np.sign(th)) if ok else 0)
        drop = set()

        def find_arcs(i0: int, i1: int) -> None:
            """Maior sub-corrente de retas (i0..i1, ligadas) que é um arco; recursivo."""
            if i1 - i0 < 1:
                return
            for size in range(i1 - i0 + 1, 1, -1):
                for p0 in range(i0, i1 - size + 2):
                    q0 = p0 + size - 1
                    if circle_resid(lines[p0]["a"], lines[q0]["b"]) <= tol + 0.5 * unit:
                        drop.update(range(p0, q0 + 1))
                        find_arcs(i0, p0 - 1)
                        find_arcs(q0 + 1, i1)
                        return

        ring = Ln > 2 and all(l != 0 and l == link[0] for l in link)
        if ring and circle_resid(lines[0]["a"], lines[-1]["b"]) <= tol + 0.5 * unit:
            return None  # contorno inteiro é uma curva suave (ex.: círculo)
        i = 0
        while i < Ln - 1:
            if link[i] == 0:
                i += 1
                continue
            j = i
            while j + 1 < Ln - 1 and link[j + 1] == link[i]:
                j += 1
            find_arcs(i, j + 1)  # corrente = retas i..j+1
            i = j + 1
        if drop:
            lines = [ln for k, ln in enumerate(lines) if k not in drop]
    if not lines:
        return None

    for ln in lines:
        c, u = ln["c"], ln["u"]
        if snap_deg > 0:
            angd = math.degrees(math.atan2(u[1], u[0]))
            target = round(angd / 90.0) * 90.0
            if abs(angd - target) <= snap_deg:
                r = math.radians(target)
                u = ln["u"] = np.array([math.cos(r), math.sin(r)])
        ln["A"] = c + u * np.dot(P2[ln["a"]] - c, u)
        ln["B"] = c + u * np.dot(P2[ln["b"] - 1] - c, u)

    # junções: canto vivo por interseção ou trecho curvo entre retas
    L = len(lines)
    for idx, ln in enumerate(lines):
        nx = lines[(idx + 1) % L]
        gap = (nx["a"] + (n if idx == L - 1 else 0)) - ln["b"]
        joined = False
        if gap <= gap_small:
            cr = ln["u"][0] * nx["u"][1] - ln["u"][1] * nx["u"][0]
            if abs(cr) > math.sin(math.radians(8)):
                dpt = nx["c"] - ln["c"]
                t = (dpt[0] * nx["u"][1] - dpt[1] * nx["u"][0]) / cr
                X = ln["c"] + t * ln["u"]
                ref = P2[ln["b"] % (2 * n)] if ln["b"] < 2 * n else P2[-1]
                if np.linalg.norm(X - ref) <= 4.0 * tol + gap * h + 2.0 * unit:
                    ln["B"] = X
                    nx["A"] = X
                    joined = True
            else:  # quase colineares: une no meio do vão
                X = (ln["B"] + nx["A"]) / 2
                ln["B"] = X
                nx["A"] = X
                joined = True
        ln["joined_next"] = joined

    # nós: (ponto, aresta até o próximo 'L'/'C', direção da reta no nó ou None)
    nodes: list = []
    for idx, ln in enumerate(lines):
        # A de uma reta unida à anterior já foi adicionado (é o B/canto da anterior)
        if idx == 0 or not lines[idx - 1]["joined_next"]:
            nodes.append([ln["A"], "L", ln["u"]])
        is_last = idx == L - 1
        if not (is_last and ln["joined_next"]):  # canto final == A da 1ª reta
            nodes.append([ln["B"], "L" if ln["joined_next"] else "C", ln["u"]])
        if not ln["joined_next"]:
            a = ln["b"]
            b = lines[(idx + 1) % L]["a"] + (n if is_last else 0)
            idxs = np.arange(a, b + 1) % (2 * n)
            for q in _dp(P2[idxs], 0.5 * tol)[1:-1]:
                nodes.append([q, "C", None])

    clean: list = []
    for nd in nodes:
        if clean and np.linalg.norm(clean[-1][0] - nd[0]) < 1e-6:
            clean[-1][1] = nd[1]
            continue
        clean.append(nd)
    if len(clean) > 1 and np.linalg.norm(clean[0][0] - clean[-1][0]) < 1e-6:
        clean.pop()
    N = len(clean)
    if N < 3:
        return None
    pts = [c[0] for c in clean]

    def unit_vec(v: np.ndarray) -> np.ndarray:
        nv = np.linalg.norm(v)
        return v / nv if nv > 1e-12 else np.zeros(2)

    def tangent(i: int, chord: np.ndarray) -> np.ndarray:
        u = clean[i][2]
        if u is not None:
            # junção reta/curva: continua na direção da reta, a menos que seja um canto
            if np.dot(u, unit_vec(chord)) >= math.cos(math.radians(25)):
                return u
            return unit_vec(chord)
        return unit_vec(pts[(i + 1) % N] - pts[i - 1])

    segs = []
    for i in range(N):
        p0, p1 = pts[i], pts[(i + 1) % N]
        if clean[i][1] == "L":
            segs.append(("L", (float(p1[0]), float(p1[1]))))
        else:
            dist = np.linalg.norm(p1 - p0)
            c1 = p0 + tangent(i, p1 - p0) * dist / 3.0
            c2 = p1 - tangent((i + 1) % N, p1 - p0) * dist / 3.0
            segs.append(("C", (float(c1[0]), float(c1[1])), (float(c2[0]), float(c2[1])),
                         (float(p1[0]), float(p1[1]))))
    return (float(pts[0][0]), float(pts[0][1])), segs


def polygonize(sub: tuple, tol: float, unit: float = 1.0, snap_deg: float = 0.0) -> tuple | None:
    """Modo "só retas": cada borda vira UMA reta entre cantos, mesmo que o original seja
    ondulado. Ondulações menores que `tol` somem; a reta passa pelo meio da onda
    (ajuste por mínimos quadrados) e os cantos ficam na interseção das retas vizinhas."""
    h = max(0.5 * unit, tol / 12.0)
    P = _sample(sub, h)
    n = len(P)
    if n < 8:
        return None
    # Douglas-Peucker fechado: parte do ponto mais distante do primeiro
    far = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
    P = np.roll(P, -far, 0)
    far = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
    half1 = _dp_idx(P[: far + 1], tol)
    half2 = _dp_idx(np.vstack([P[far:], P[:1]]), tol)
    verts = sorted(set(half1) | {far + i for i in half2 if far + i < n})
    if len(verts) < 3:
        return None

    def fit(a: int, b: int):
        seg = P[a:b + 1] if b > a else np.vstack([P[a:], P[:b + 1]])
        m = len(seg)
        cut = m // 8
        Q = seg[cut:m - cut] if m - 2 * cut >= 3 else seg
        c = Q.mean(0)
        _, V = np.linalg.eigh((Q - c).T @ (Q - c))
        u = V[:, 1]
        if np.dot(u, seg[-1] - seg[0]) < 0:
            u = -u
        if snap_deg > 0:
            angd = math.degrees(math.atan2(u[1], u[0]))
            target = round(angd / 90.0) * 90.0
            if abs(angd - target) <= snap_deg:
                r = math.radians(target)
                u = np.array([math.cos(r), math.sin(r)])
        return c, u

    # remove vértices entre bordas quase colineares (ondas que o DP ainda deixou)
    changed = True
    while changed and len(verts) > 3:
        changed = False
        k = len(verts)
        lines = [fit(verts[i], verts[(i + 1) % k]) for i in range(k)]
        for i in range(k):
            u0, u1 = lines[i - 1][1], lines[i][1]
            if abs(math.atan2(u0[0] * u1[1] - u0[1] * u1[0], float(np.dot(u0, u1)))) < math.radians(12):
                del verts[i]
                changed = True
                break

    k = len(verts)
    lines = [fit(verts[i], verts[(i + 1) % k]) for i in range(k)]
    corners = []
    for i in range(k):
        (c0, u0), (c1, u1) = lines[i - 1], lines[i]
        cr = u0[0] * u1[1] - u0[1] * u1[0]
        v = P[verts[i]]
        if abs(cr) > math.sin(math.radians(10)):
            d = c1 - c0
            X = c0 + (d[0] * u1[1] - d[1] * u1[0]) / cr * u0
            if np.linalg.norm(X - v) <= 4.0 * tol + 3.0 * unit:
                v = X
        corners.append(v)
    start = (float(corners[0][0]), float(corners[0][1]))
    segs = [("L", (float(q[0]), float(q[1]))) for q in corners[1:]] + [("L", start)]
    return start, segs


def _dp_idx(Q: np.ndarray, eps: float) -> list[int]:
    """Índices mantidos pelo Douglas-Peucker."""
    if len(Q) < 3:
        return list(range(len(Q)))
    keep = np.zeros(len(Q), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(Q) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        v = Q[b] - Q[a]
        nv = np.linalg.norm(v)
        seg = Q[a + 1:b] - Q[a]
        d = np.abs(seg[:, 0] * v[1] - seg[:, 1] * v[0]) / nv if nv > 1e-12 else np.linalg.norm(seg, axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return list(np.nonzero(keep)[0])


def straighten_all(subs: list, tol: float, unit: float = 1.0, snap_deg: float = 0.0,
                   polygon: bool = False) -> list:
    fn = polygonize if polygon else straighten
    out = []
    for sp in subs:
        try:
            r = fn(sp, tol, unit, snap_deg)
        except Exception:
            r = None
        out.append(r if r is not None else sp)
    return out


def straighten_svg_paths(svg: str, tol: float, unit: float = 1.0, snap_deg: float = 0.0,
                         polygon: bool = False) -> str:
    """Aplica o endireitamento em todo atributo d="..." de um SVG."""
    def repl(m):
        subs = parse_d(m.group(1))
        if not subs:
            return m.group(0)
        return 'd="' + to_d(straighten_all(subs, tol, unit, snap_deg, polygon)) + '"'
    return re.sub(r'(?<=\s)d="([^"]*)"', repl, svg)
