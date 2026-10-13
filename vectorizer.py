"""Motor de vetorização.

Dois modos:
  * standard -> SVG colorido (vtracer) ou preto e branco (potrace), para uso geral/design.
  * laser    -> SVG em milímetros (e DXF) com contornos fechados, pronto para
                cortadora/gravadora a laser (LightBurn, RDWorks, LaserGRBL, Glowforge...).
"""

from __future__ import annotations

import io
import math
import re

import cv2
import numpy as np
import potrace
import vtracer
from PIL import Image, ImageOps

import geometry as G

try:  # suporte opcional a HEIC/HEIF/AVIF (fotos de iPhone etc.)
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass


MAX_SIDE = 3000          # imagens maiores são reduzidas (velocidade)
TARGET_SMALL = 2000      # imagens pequenas são ampliadas até ~este tamanho
MAX_UPSCALE = 4.0
LASER_HAIRLINE_MM = 0.0254  # 0.001" – "hairline" padrão de corte (Epilog, Trotec, LightBurn)


class VectorizeError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Carregamento / pré-processamento
# --------------------------------------------------------------------------- #

def load_image(data: bytes) -> tuple[np.ndarray, float | None]:
    """Abre qualquer formato suportado pelo Pillow e devolve RGBA uint8 + DPI (se houver)."""
    try:
        im = Image.open(io.BytesIO(data))
        im.seek(0)  # GIF/TIFF animados: primeiro quadro
        im = ImageOps.exif_transpose(im)
    except Exception as exc:
        raise VectorizeError(
            "Não foi possível abrir a imagem. Formatos aceitos: JPG, PNG, WEBP, BMP, GIF, TIFF, HEIC, AVIF..."
        ) from exc

    dpi = None
    if isinstance(im.info.get("dpi"), tuple):
        try:
            d = float(im.info["dpi"][0])
            if 30 <= d <= 2400:
                dpi = d
        except Exception:
            pass

    im = im.convert("RGBA")
    return np.array(im), dpi


def _working_scale(w: int, h: int, upscale: bool) -> float:
    side = max(w, h)
    if side > MAX_SIDE:
        return MAX_SIDE / side
    if upscale and side < TARGET_SMALL:
        return min(MAX_UPSCALE, TARGET_SMALL / side)
    return 1.0


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-3:
        return img
    h, w = img.shape[:2]
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (nw, nh), interpolation=interp)


def _border_color(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray | None:
    """Cor mais comum na borda da imagem (provável fundo)."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    a = np.concatenate([alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]])
    border = border[a > 127]
    if len(border) == 0:
        return None
    q = (border // 8).astype(np.int32)
    keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    top = vals[np.argmax(counts)]
    return border[keys == top].mean(axis=0)


def _remove_background(rgb: np.ndarray, alpha: np.ndarray, tol: float = 38.0) -> np.ndarray:
    """Torna transparente a região de fundo conectada à borda."""
    bg = _border_color(rgb, alpha)
    if bg is None:
        return alpha
    dist = np.linalg.norm(rgb.astype(np.float32) - bg.astype(np.float32), axis=2)
    mask = ((dist < tol) & (alpha > 127)).astype(np.uint8)
    n, labels = cv2.connectedComponents(mask, connectivity=4)
    if n <= 1:
        return alpha
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    edge = edge[edge != 0]
    out = alpha.copy()
    out[np.isin(labels, edge)] = 0
    return out


def _gradient(rgb: np.ndarray) -> np.ndarray:
    grad = np.zeros(rgb.shape[:2], np.float32)
    for c in range(3):
        ch = rgb[..., c].astype(np.float32)
        grad = np.maximum(grad, np.abs(cv2.Sobel(ch, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(ch, cv2.CV_32F, 0, 1)))
    return grad


def _learn_palette(px: np.ndarray, colors: int, sep: float = 38.0, min_frac: float = 0.00012) -> np.ndarray:
    """Escolhe até `colors` cores por busca de modas: a cor mais frequente, depois a mais
    frequente entre os pixels distantes (> sep) das já escolhidas, e assim por diante.
    Não depende da proporção entre as cores, então detalhes pequenos (texto) entram."""
    px = px.astype(np.float32)
    n = len(px)
    min_n = max(20, n * min_frac)
    q = (px // 8).astype(np.int32)
    keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    remaining = np.ones(n, bool)
    pal: list[np.ndarray] = []
    while len(pal) < colors and remaining.sum() >= min_n:
        k, c = np.unique(keys[remaining], return_counts=True)
        seed = px[remaining & (keys == k[np.argmax(c)])].mean(0)
        near = remaining & (np.linalg.norm(px - seed, axis=1) < 20)
        if near.sum() < min_n:
            break
        center = px[near].mean(0)
        pal.append(center)
        remaining &= np.linalg.norm(px - center, axis=1) > sep
    if not pal:
        pal = [px.mean(0)]
    P = np.array(pal, np.float32)
    sample = px if n <= 50_000 else px[np.random.default_rng(3).choice(n, 50_000, replace=False)]
    for _ in range(3):  # refinamento tipo k-means
        lab = np.argmin(((sample[:, None, :] - P[None]) ** 2).sum(-1), axis=1)
        for i in range(len(P)):
            sel = lab == i
            if sel.sum() >= 5:
                P[i] = sample[sel].mean(0)
    return P


def _quantize(rgb: np.ndarray, colors: int, mask: np.ndarray | None = None,
              radius: float = 3) -> tuple[np.ndarray, int]:
    """Reduz `rgb` para no máx. `colors` cores sem halos.

    1. A paleta vem só de pixels "lisos" (baixo gradiente): cores de transição de
       antialiasing/JPG nunca são lisas, então não viram cores próprias.
    2. Pixels lisos recebem a cor mais próxima; pixels de borda são decididos por
       watershed a partir das regiões vizinhas — a fronteira cai no ponto de maior
       contraste, exatamente onde está a borda real.
    """
    h, w = rgb.shape[:2]
    grad = _gradient(rgb)
    flat = grad < 48
    if mask is not None:
        flat &= mask
    pts = rgb[flat]
    if len(pts) < 500:
        pts = rgb.reshape(-1, 3) if mask is None else rgb[mask]
    if len(pts) > 250_000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 250_000, replace=False)]
    P = _learn_palette(pts, colors)

    flat_pal = [int(round(v)) for k in P for v in k]
    flat_pal += flat_pal[:3] * (256 - len(P))  # preenche com cor válida (nunca preto "fantasma")
    pimg = Image.new("P", (1, 1))
    pimg.putpalette(flat_pal)
    idx = np.array(Image.fromarray(rgb, "RGB").quantize(palette=pimg, dither=Image.Dither.NONE)).astype(np.int32)
    idx = np.minimum(idx, len(P) - 1)

    # pixels de borda: só podem receber cores de regiões lisas vizinhas (escolhe a mais
    # próxima entre elas). Uma mistura amarelo/azul vira amarelo ou azul, nunca uma 3ª cor,
    # e a fronteira fica na isolinha de meio-termo — suave, sem serrilhado.
    lab = idx.copy()
    edge = ~flat if mask is None else (~flat & mask)
    if edge.any() and flat.any():
        r = max(2, int(radius))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        ey, ex = np.nonzero(edge)
        epx = rgb[ey, ex].astype(np.float32)
        best = np.full(len(ey), np.inf, np.float32)
        choice = idx[ey, ex].copy()
        for k in range(len(P)):
            m = ((idx == k) & flat).astype(np.uint8)
            if not m.any():
                continue
            cover = cv2.dilate(m, ker)[ey, ex].astype(bool)
            if not cover.any():
                continue
            d = ((epx[cover] - P[k]) ** 2).sum(-1)
            sel = np.nonzero(cover)[0]
            upd = d < best[sel]
            best[sel[upd]] = d[upd]
            choice[sel[upd]] = k
        lab[ey, ex] = choice
    return P.round().clip(0, 255).astype(np.uint8)[lab], len(P)


def _mode_filter(rgb: np.ndarray, iterations: int = 1, ksize: int = 3) -> np.ndarray:
    """Remove pixels isolados após quantização sem criar cores novas (filtro de moda 3x3)."""
    h, w = rgb.shape[:2]
    keys = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
    uniq, inv = np.unique(keys, return_inverse=True)
    if len(uniq) > 64:
        return rgb
    idx = inv.reshape(h, w)
    for _ in range(iterations):
        best = np.full((h, w), -1.0, np.float32)
        new = idx.copy()
        for k in range(len(uniq)):
            v = cv2.boxFilter((idx == k).astype(np.float32), -1, (ksize, ksize), normalize=False,
                              borderType=cv2.BORDER_REPLICATE)
            v += (idx == k) * 0.5  # empate: mantém a cor original
            better = v > best
            best[better] = v[better]
            new[better] = k
        idx = new
    k = uniq[idx]
    return np.stack([(k >> 16) & 255, (k >> 8) & 255, k & 255], axis=-1).astype(np.uint8)


def _png_bytes(rgba: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise VectorizeError("Falha ao preparar imagem.")
    return buf.tobytes()


def _fmt(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# --------------------------------------------------------------------------- #
# Potrace (traço de alta precisão em preto e branco)
# --------------------------------------------------------------------------- #

def _trace_bitmap(fg: np.ndarray, turdsize: int, alphamax: float, opttolerance: float):
    # potracer: Bitmap() inverte os dados, então passamos o "fundo" como True.
    bmp = potrace.Bitmap(~fg)
    return bmp.trace(
        turdsize=int(turdsize),
        turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY,
        alphamax=float(alphamax),
        opticurve=opttolerance > 0,
        opttolerance=float(opttolerance) if opttolerance > 0 else 0.2,
    )


def _light_on_transparent(gray: np.ndarray, alpha: np.ndarray) -> bool:
    """PNG com fundo transparente e arte clara (ex.: logo branco): a arte deve ser traçada."""
    opaque = alpha >= 128
    if (~opaque).mean() < 0.01 or not opaque.any():
        return False
    return float(gray[opaque].mean()) > 140


def _straighten_opts(p: dict, scale: float) -> tuple[float, float, float, bool] | None:
    """(tolerância, unidade, graus p/ alinhar, modo polígono) em pixels de trabalho,
    ou None se desligado."""
    tol = float(p.get("straighten") or 0)
    if tol <= 0:
        return None
    return tol * scale, scale, 3.0 if p.get("snap") else 0.0, p.get("edge_mode") == "polygon"


# --------------------------------------------------------------------------- #
# Modo 1: SVG padrão
# --------------------------------------------------------------------------- #

STANDARD_PRESETS = {
    # colors=0 -> sem redução de cores
    "logo":        dict(colors=8,  speckle=3,  corner=60,  denoise=True,  mode="spline", upscale=True),
    "ilustracao":  dict(colors=24, speckle=3,  corner=60,  denoise=True,  mode="spline", upscale=True),
    "foto":        dict(colors=0,  speckle=6,  corner=120, denoise=False, mode="spline", upscale=False),
    "pb":          dict(colors=2,  speckle=2,  corner=60,  denoise=True,  mode="spline", upscale=True),
}


def vectorize_standard(data: bytes, p: dict) -> dict:
    rgba, _ = load_image(data)
    oh, ow = rgba.shape[:2]
    preset = STANDARD_PRESETS.get(p.get("preset", "logo"), STANDARD_PRESETS["logo"])

    colors = int(p.get("colors", preset["colors"]))
    speckle = int(p.get("speckle", preset["speckle"]))
    corner = int(p.get("corner", preset["corner"]))
    denoise = bool(p.get("denoise", preset["denoise"]))
    mode = p.get("curve_mode", preset["mode"])
    remove_bg = bool(p.get("remove_bg", False))
    upscale = bool(p.get("upscale", preset["upscale"]))

    scale = _working_scale(ow, oh, upscale)
    work = _resize(rgba, scale)
    if scale > 1.2:
        # arredonda os "degraus" dos pixels originais antes de traçar
        work = cv2.GaussianBlur(work, (0, 0), 0.55 * scale)
    rgb, alpha = work[..., :3].copy(), work[..., 3].copy()
    alpha = np.where(alpha > 127, 255, 0).astype(np.uint8)

    if denoise:
        rgb = cv2.bilateralFilter(rgb, 9, 40, 9)

    if p.get("preset") == "pb":
        return _standard_bw(rgb, alpha, ow, oh, scale, p, remove_bg)

    if colors > 0:
        colors = max(2, min(64, colors))
        rgb, colors = _quantize(rgb, colors, alpha > 0, radius=2 + 1.5 * max(1.0, scale))
        if denoise:
            # remove pixels soltos sem arredondar cantos nem fechar furos pequenos
            rgb = _mode_filter(rgb, 1, 3)
        if remove_bg:  # depois da quantização: o halo já foi absorvido pela cor do fundo
            alpha = _remove_background(rgb, alpha, tol=8)
        color_precision, layer_diff = 8, 1
    else:
        if remove_bg:
            alpha = _remove_background(rgb, alpha)
        color_precision = int(p.get("color_precision", 8))
        layer_diff = int(p.get("layer_difference", 24))

    rgba_out = np.dstack([rgb, alpha])
    if not (alpha > 0).any():
        raise VectorizeError("A imagem ficou totalmente transparente (tente desativar 'Remover fundo').")

    svg = vtracer.convert_raw_image_to_svg(
        _png_bytes(rgba_out),
        img_format="png",
        colormode="color",
        hierarchical="stacked",
        mode="spline" if mode == "spline" else "polygon",
        filter_speckle=max(1, round(speckle * scale)) if speckle else 0,
        color_precision=color_precision,
        layer_difference=layer_diff,
        corner_threshold=corner,
        length_threshold=4.0,
        max_iterations=10,
        splice_threshold=45,
        path_precision=2,
    )
    ph, pw = rgb.shape[:2]
    svg = re.sub(r"<\?xml[^>]*>\s*", "", svg)
    svg = re.sub(r"<!--.*?-->\s*", "", svg, flags=re.S)
    svg = re.sub(
        r"<svg[^>]*>",
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{ow}" height="{oh}" '
        f'viewBox="0 0 {pw} {ph}">',
        svg,
        count=1,
    )
    st = _straighten_opts(p, scale)
    if st:
        svg = G.straighten_svg_paths(svg, *st)
    svg = '<?xml version="1.0" encoding="UTF-8"?>\n' + svg
    return {
        "svg": svg,
        "stats": {
            "paths": svg.count("<path"),
            "width": ow,
            "height": oh,
            "colors": colors or "todas",
            "bytes": len(svg.encode()),
        },
    }


def _standard_bw(rgb, alpha, ow, oh, scale, p, remove_bg):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    light = _light_on_transparent(gray, alpha)
    gray[alpha == 0] = 0 if light else 255
    fg, t = _threshold(gray, p.get("threshold"))
    if light:
        fg = gray > t
    if p.get("invert"):
        fg = ~fg & (alpha > 0)
    turd = max(2, int(int(p.get("speckle", 4)) * scale * scale))
    alphamax = 1.0 if p.get("curve_mode", "spline") == "spline" else 0.0
    paths = _trace_bitmap(fg, turd, alphamax, 0.2)
    ph, pw = gray.shape
    subs = [G.from_potrace(c) for c in paths]
    st = _straighten_opts(p, scale)
    if st:
        subs = G.straighten_all(subs, *st)
    d = G.to_d(subs)
    bg = "" if remove_bg else f'<rect width="{pw}" height="{ph}" fill="#FFFFFF"/>\n'
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{ow}" height="{oh}" '
        f'viewBox="0 0 {pw} {ph}">\n{bg}'
        f'<path d="{d}" fill="#000000" fill-rule="evenodd"/>\n</svg>\n'
    )
    return {
        "svg": svg,
        "stats": {"paths": len(subs), "width": ow, "height": oh, "colors": 2,
                  "bytes": len(svg.encode()), "threshold": t},
    }


# --------------------------------------------------------------------------- #
# Modo 2: Laser
# --------------------------------------------------------------------------- #

def _threshold(gray: np.ndarray, manual) -> tuple[np.ndarray, int]:
    if manual is None or manual == "" or manual == "auto":
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = int(t)
    else:
        t = int(manual)
    return gray < t, t


def vectorize_laser(data: bytes, p: dict) -> dict:
    rgba, dpi = load_image(data)
    oh, ow = rgba.shape[:2]

    # transparente = fundo. Arte escura: compõe sobre branco; arte clara: sobre preto
    light = _light_on_transparent(cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_RGB2GRAY),
                                  rgba[..., 3])
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    bgv = 0.0 if light else 255.0
    rgb = (rgba[..., :3].astype(np.float32) * a + bgv * (1 - a)).astype(np.uint8)

    scale = _working_scale(ow, oh, True)
    scale = min(scale, 2500 / max(ow, oh))  # potrace em Python: limitar p/ velocidade
    rgb = _resize(rgb, scale)

    blur = float(p.get("smooth_input", 1))
    if blur > 0:
        rgb = cv2.GaussianBlur(rgb, (0, 0), blur * max(1.0, scale) * 0.6)

    if p.get("separation") == "background":
        # tudo que difere da cor do fundo (cor predominante na borda) é traçado
        bg = _border_color(rgb, np.full(rgb.shape[:2], 255, np.uint8))
        dist = np.linalg.norm(rgb.astype(np.float32) - bg.astype(np.float32), axis=2)
        t = int(p.get("tolerance", 60))
        fg = dist > t
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        fg, t = _threshold(gray, p.get("threshold"))
        if light:
            fg = gray > t
    if p.get("invert"):
        fg = ~fg

    if p.get("silhouette"):
        cnts, _ = cv2.findContours(fg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sil = np.zeros(fg.shape, np.uint8)
        cv2.drawContours(sil, cnts, -1, 1, thickness=cv2.FILLED)
        fg = sil.astype(bool)

    if not fg.any():
        raise VectorizeError("Nada para vetorizar. Ajuste o 'Limiar' ou use 'Inverter'.")

    speckle = float(p.get("speckle", 4))
    turd = max(0, int(speckle * scale * scale))
    alphamax = float(p.get("smoothness", 1.0))
    opttol = float(p.get("optimize", 0.2))
    paths = _trace_bitmap(fg, turd, alphamax, opttol)
    subs = [G.from_potrace(c) for c in paths]
    st = _straighten_opts(p, scale)
    if st:
        subs = G.straighten_all(subs, *st)

    # dimensões físicas
    width_mm = p.get("width_mm")
    try:
        width_mm = float(width_mm)
    except (TypeError, ValueError):
        width_mm = None
    if not width_mm or width_mm <= 0:
        width_mm = round(ow / dpi * 25.4, 2) if dpi else 100.0
    ph, pw = fg.shape
    mm_per_px = width_mm / pw
    height_mm = ph * mm_per_px

    style = p.get("style", "cut")
    color = p.get("color", "#FF0000")
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color or ""):
        color = "#FF0000"

    W, H = _fmt(width_mm), _fmt(height_mm)
    head = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">\n'
    )
    if style == "engrave":
        d = G.to_d(subs, mm_per_px)
        body = f'<path d="{d}" fill="#000000" fill-rule="evenodd" stroke="none"/>\n'
    else:
        body = f'<g fill="none" stroke="{color}" stroke-width="{LASER_HAIRLINE_MM}">\n' + "".join(
            f'<path d="{G.to_d([sp], mm_per_px)}"/>\n' for sp in subs
        ) + "</g>\n"
    svg = head + body + "</svg>\n"

    polys = [G.flatten(sp, mm_per_px, max_len=0.2) for sp in subs]
    dxf = _to_dxf(polys, height_mm)
    cut_len = sum(
        sum(math.dist(pl[i], pl[(i + 1) % len(pl)]) for i in range(len(pl)))
        for pl in polys if len(pl) > 1
    )
    return {
        "svg": svg,
        "dxf": dxf,
        "stats": {
            "paths": len(subs),
            "width_mm": round(width_mm, 2),
            "height_mm": round(height_mm, 2),
            "threshold": t,
            "cut_length_mm": round(cut_len, 1),
            "bytes": len(svg.encode()),
            "dpi_detected": dpi,
        },
    }


def _to_dxf(polys: list, height_mm: float) -> str:
    import ezdxf

    doc = ezdxf.new("R2010", setup=False)
    doc.units = ezdxf.units.MM
    doc.header["$INSUNITS"] = 4
    doc.header["$MEASUREMENT"] = 1
    doc.layers.add("CORTE", color=1)
    msp = doc.modelspace()
    for pl in polys:
        pts = [(x, height_mm - y) for x, y in pl]  # DXF: eixo Y para cima
        clean = [pts[0]]
        for q in pts[1:]:
            if math.dist(q, clean[-1]) > 1e-4:
                clean.append(q)
        if len(clean) >= 3:
            msp.add_lwpolyline(clean, close=True, dxfattribs={"layer": "CORTE"})
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue()
