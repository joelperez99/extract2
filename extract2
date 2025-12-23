import re
import io
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
import numpy as np

import cv2
import pytesseract
from pytesseract import Output


# -----------------------------
# Helpers: limpieza y regex
# -----------------------------
PRICE_RE = re.compile(r"\$?\s*(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})|\d+(?:\.\d{2}))")
GRAM_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(kg|kgs|kilogramos|g|gr|gramos|ml|l)\b",
    re.IGNORECASE,
)
# Caso especial: número suelto (20..2000) al final (tu regla)
GRAM_END_NUM_RE = re.compile(r"\b(\d{2,4})\b")

PRESENT_CAN_KW = re.compile(r"\b(lata|bote|tarro)\b", re.IGNORECASE)
PRESENT_BAG_KW = re.compile(r"\b(bolsa|pouch|sobre)\b", re.IGNORECASE)

def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def parse_price(text: str) -> Optional[float]:
    if not text:
        return None
    # buscamos el primer precio "con sentido"
    m = PRICE_RE.search(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1).replace(" ", ""))
    except:
        return None

def parse_gramaje(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower()
    m = GRAM_RE.search(t.replace("gr.", "gr").replace("kgs", "kg"))
    if m:
        num = m.group(1).replace(",", ".")
        unit = m.group(2).lower()
        # normaliza unidades
        unit_map = {"kgs": "kg", "kilogramos": "kg", "gr": "g", "gramos": "g"}
        unit = unit_map.get(unit, unit)
        return f"{num} {unit}"
    return None

def infer_gramaje_fallback(name: str) -> Optional[str]:
    """Regla: si NO hay g/kg visible, pero hay número al final 20..2000 => gramos"""
    if not name:
        return None
    tokens = re.findall(r"\b\d{2,4}\b", name)
    if not tokens:
        return None
    # toma el último número
    n = int(tokens[-1])
    if 20 <= n <= 2000:
        return f"{n} g"
    return None

def infer_presentacion(text: str) -> str:
    if not text:
        return ""
    if PRESENT_CAN_KW.search(text):
        return "Lata"
    if PRESENT_BAG_KW.search(text):
        return "Bolsa"
    return ""


# -----------------------------
# OCR por tiles
# -----------------------------
@dataclass
class OcrWord:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: int

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def preprocess_for_ocr(img_cv: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # mejora contraste
    gray = cv2.equalizeHist(gray)
    # binariza suave
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return thr

def ocr_words(img_cv: np.ndarray, lang: str = "spa") -> List[OcrWord]:
    # pytesseract necesita que Tesseract esté instalado en el sistema
    data = pytesseract.image_to_data(img_cv, lang=lang, output_type=Output.DICT)
    out = []
    n = len(data["text"])
    for i in range(n):
        txt = norm_spaces(data["text"][i])
        if not txt:
            continue
        try:
            conf = int(float(data["conf"][i]))
        except:
            conf = -1
        if conf < 40:  # filtra ruido
            continue
        out.append(
            OcrWord(
                text=txt,
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i]),
                conf=conf,
            )
        )
    return out

def tile_image(img: Image.Image, tile_h: int, overlap: int) -> List[Tuple[int, int, Image.Image]]:
    """Devuelve lista de (y0, y1, tile_pil)"""
    W, H = img.size
    tiles = []
    y = 0
    while y < H:
        y0 = y
        y1 = min(H, y + tile_h)
        tile = img.crop((0, y0, W, y1))
        tiles.append((y0, y1, tile))
        if y1 == H:
            break
        y = y1 - overlap
    return tiles

def run_ocr_tiled(img_pil: Image.Image, tile_h: int, overlap: int, lang: str) -> List[OcrWord]:
    words_all: List[OcrWord] = []
    tiles = tile_image(img_pil, tile_h=tile_h, overlap=overlap)
    for (y0, y1, tile) in tiles:
        cv_img = pil_to_cv(tile)
        cv_img = preprocess_for_ocr(cv_img)
        words = ocr_words(cv_img, lang=lang)
        # Ajusta coordenadas a la imagen completa
        for w in words:
            words_all.append(OcrWord(w.text, w.x, w.y + y0, w.w, w.h, w.conf))
    return words_all


# -----------------------------
# Agrupar palabras -> líneas -> productos
# -----------------------------
@dataclass
class Line:
    y: int
    x0: int
    x1: int
    text: str

def words_to_lines(words: List[OcrWord], y_tol: int = 10) -> List[Line]:
    # agrupa por cercanía en Y
    words = sorted(words, key=lambda w: (w.y, w.x))
    lines: List[List[OcrWord]] = []
    for w in words:
        placed = False
        for ln in lines:
            if abs(ln[0].y - w.y) <= y_tol:
                ln.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    out: List[Line] = []
    for ln in lines:
        ln_sorted = sorted(ln, key=lambda w: w.x)
        text = " ".join([w.text for w in ln_sorted])
        x0 = min(w.x for w in ln_sorted)
        x1 = max(w.x + w.w for w in ln_sorted)
        y = int(np.median([w.y for w in ln_sorted]))
        out.append(Line(y=y, x0=x0, x1=x1, text=norm_spaces(text)))
    out.sort(key=lambda L: (L.y, L.x0))
    return out

def find_price_lines(lines: List[Line]) -> List[int]:
    idxs = []
    for i, ln in enumerate(lines):
        if "$" in ln.text or PRICE_RE.search(ln.text):
            price = parse_price(ln.text)
            if price is not None:
                idxs.append(i)
    return idxs

def extract_product_around_price(lines: List[Line], price_idx: int, window_above: int = 6) -> Dict:
    """Toma líneas arriba del precio y arma un producto."""
    price_line = lines[price_idx].text
    price = parse_price(price_line)

    # candidato: líneas arriba (mismo bloque visual)
    start = max(0, price_idx - window_above)
    context = [lines[j].text for j in range(start, price_idx + 1)]
    context_text = "\n".join(context)

    # heurística marca/nombre:
    # - muchas páginas: marca es la primera línea "corta" arriba del nombre
    # - nombre: la(s) línea(s) antes del gramaje/precio
    above = [lines[j].text for j in range(start, price_idx)]
    above = [t for t in above if t and not t.startswith("$")]

    # limpia cosas típicas del UI
    drop_kw = re.compile(r"\b(agregar|pocas piezas|reseñas|envío|oferta|rebaja)\b", re.IGNORECASE)
    above = [t for t in above if not drop_kw.search(t)]

    marca = ""
    nombre = ""
    gramaje = ""

    if above:
        # marca = primera línea corta (<= 3 palabras) si existe
        for t in above:
            if 1 <= len(t.split()) <= 3:
                marca = t
                break
        # nombre = concat de las líneas siguientes a marca, limitando longitud
        if marca and marca in above:
            mi = above.index(marca)
            name_lines = above[mi + 1 : mi + 4]
        else:
            name_lines = above[-3:]
        nombre = norm_spaces(" ".join(name_lines))

        # gramaje desde todo el contexto (porque a veces está en nombre)
        gramaje = parse_gramaje(context_text) or parse_gramaje(nombre) or infer_gramaje_fallback(nombre) or ""

    presentacion = infer_presentacion(context_text) or infer_presentacion(nombre)

    return {
        "Marca": marca,
        "Nombre": nombre,
        "Gramaje": gramaje,
        "Precio MXN": price,
        "Presentación": presentacion,
        "Contexto": context_text,  # útil para depurar (opcional)
    }

def dedupe_products(rows: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for r in rows:
        key = (
            (r.get("Marca") or "").lower(),
            (r.get("Nombre") or "").lower(),
            str(r.get("Precio MXN") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# -----------------------------
# UI Streamlit
# -----------------------------
st.set_page_config(page_title="OCR productos (captura larga)", layout="wide")
st.title("📸➡️📊 Extraer productos desde una captura larga (Walmart / listados)")

st.markdown(
    """
Sube una **captura larga** (scroll completo) y el sistema intentará extraer:
**marca, nombre, gramaje, precio y si es bolsa o lata**, y generar un **Excel**.
"""
)

with st.sidebar:
    st.header("Ajustes OCR")
    lang = st.selectbox("Idioma OCR", ["spa", "eng", "spa+eng"], index=0)
    tile_h = st.slider("Alto de tile (px)", 900, 2400, 1400, 100)
    overlap = st.slider("Overlap (px)", 100, 600, 220, 20)
    y_tol = st.slider("Tolerancia para agrupar líneas (px)", 6, 20, 10, 1)
    window_above = st.slider("Líneas arriba del precio (contexto)", 3, 12, 7, 1)
    show_debug = st.checkbox("Mostrar depuración (OCR/Contextos)", value=False)

uploaded = st.file_uploader("Sube la imagen (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption=f"Imagen cargada ({img.size[0]}x{img.size[1]})", use_container_width=True)

    if st.button("🔎 Extraer productos"):
        with st.spinner("Procesando OCR por tiles..."):
            words = run_ocr_tiled(img, tile_h=tile_h, overlap=overlap, lang=lang)

        if not words:
            st.error(
                "No se detectó texto. Esto casi siempre significa que **Tesseract no está instalado** "
                "o que el OCR no está configurado correctamente en tu entorno."
            )
            st.stop()

        with st.spinner("Agrupando texto en líneas y detectando precios..."):
            lines = words_to_lines(words, y_tol=y_tol)
            price_idxs = find_price_lines(lines)

        st.write(f"✅ Palabras detectadas: {len(words)} | Líneas: {len(lines)} | Líneas con precio: {len(price_idxs)}")

        rows = []
        for pi in price_idxs:
            rows.append(extract_product_around_price(lines, pi, window_above=window_above))

        rows = dedupe_products(rows)

        df = pd.DataFrame(rows)
        # Quita columna Contexto si no estás en debug
        if not show_debug and "Contexto" in df.columns:
            df = df.drop(columns=["Contexto"])

        # limpia filas sin nombre/precio
        df = df[df["Precio MXN"].notna()]
        df["Marca"] = df["Marca"].fillna("").astype(str)
        df["Nombre"] = df["Nombre"].fillna("").astype(str)
        df["Gramaje"] = df["Gramaje"].fillna("").astype(str)
        df["Presentación"] = df["Presentación"].fillna("").astype(str)

        st.subheader("Resultados")
        st.dataframe(df, use_container_width=True, height=520)

        # Export excel
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Productos")
        out.seek(0)

        st.download_button(
            "⬇️ Descargar Excel (.xlsx)",
            data=out,
            file_name="productos_extraidos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if show_debug:
            st.subheader("Debug: primeras 80 líneas OCR")
            st.code("\n".join([f"{i:04d}: {lines[i].text}" for i in range(min(80, len(lines)))]))
