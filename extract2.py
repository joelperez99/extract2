import re
import io
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import cv2
import pytesseract
from pytesseract import Output


# =========================
# Regex y parsers estrictos
# =========================
PRICE_STRICT_RE = re.compile(r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})")  # exige $ y .00
PRICE_NUM_RE = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2}))")

GRAM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kg|g|gr)\b", re.IGNORECASE)
END_NUM_RE = re.compile(r"\b(\d{2,4})\b")

PRESENT_CAN_KW = re.compile(r"\b(lata|bote|tarro)\b", re.IGNORECASE)
PRESENT_BAG_KW = re.compile(r"\b(bolsa|pouch|sobre)\b", re.IGNORECASE)

UI_BLACKLIST = re.compile(
    r"\b(ordenar|más relevantes|relevantes|cuenta|pickup|envío|llega mañana|agregar|opciones|piezas|promoción|meses sin intereses)\b",
    re.IGNORECASE
)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def parse_price_strict(text: str) -> Optional[float]:
    if not text:
        return None
    # solo si hay $xxx.xx
    m = PRICE_NUM_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except:
        return None

def parse_gramaje(text: str) -> str:
    if not text:
        return ""
    t = text.lower().replace("gr.", "gr")
    m = GRAM_RE.search(t)
    if m:
        num = m.group(1).replace(",", ".")
        unit = m.group(2).lower()
        unit = "g" if unit in ["gr"] else unit
        return f"{num} {unit}"
    return ""

def infer_gramaje_fallback(name: str) -> str:
    if not name:
        return ""
    nums = END_NUM_RE.findall(name)
    if not nums:
        return ""
    n = int(nums[-1])
    if 20 <= n <= 2000:
        return f"{n} g"
    return ""

def infer_presentacion(text: str) -> str:
    if not text:
        return ""
    if PRESENT_CAN_KW.search(text):
        return "Lata"
    if PRESENT_BAG_KW.search(text):
        return "Bolsa"
    return ""


# =========================
# CV: detectar “cards”
# =========================
def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def preprocess_for_cards(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # resalta bordes de tarjetas / cajas
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    # dilata para cerrar contornos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dil = cv2.dilate(edges, kernel, iterations=2)
    return dil

def find_card_boxes(bgr: np.ndarray,
                    min_w: int,
                    min_h: int,
                    max_w: int,
                    max_h: int) -> List[Tuple[int,int,int,int]]:
    mask = preprocess_for_cards(bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = bgr.shape[:2]
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        # filtra tamaños que no parecen card
        if w < min_w or h < min_h:
            continue
        if w > max_w or h > max_h:
            continue
        # filtra contenedores gigantes (toda página)
        if w > 0.95 * W:
            continue
        boxes.append((x,y,w,h))

    # Ordena y quita duplicados por IoU simple
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    def iou(a, b) -> float:
        ax,ay,aw,ah = a
        bx,by,bw,bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax+aw, bx+bw)
        y2 = min(ay+ah, by+bh)
        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = aw*ah
        area_b = bw*bh
        return inter / float(area_a + area_b - inter + 1e-9)

    dedup = []
    for b in boxes:
        if all(iou(b, d) < 0.65 for d in dedup):
            dedup.append(b)

    return dedup


# =========================
# OCR dentro de cada card
# =========================
def ocr_lines_in_roi(roi_bgr: np.ndarray, lang: str) -> List[str]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)

    data = pytesseract.image_to_data(thr, lang=lang, output_type=Output.DICT)

    words = []
    n = len(data["text"])
    for i in range(n):
        t = norm(data["text"][i])
        if not t:
            continue
        try:
            conf = int(float(data["conf"][i]))
        except:
            conf = -1
        if conf < 40:
            continue
        words.append((t, data["left"][i], data["top"][i], data["width"][i], data["height"][i]))

    # agrupa en líneas por y
    words = sorted(words, key=lambda w: (w[2], w[1]))
    lines = []
    y_tol = 10
    for w in words:
        placed = False
        for line in lines:
            if abs(line["y"] - w[2]) <= y_tol:
                line["words"].append(w)
                placed = True
                break
        if not placed:
            lines.append({"y": w[2], "words": [w]})

    out_lines = []
    for line in lines:
        ws = sorted(line["words"], key=lambda w: w[1])
        out_lines.append(norm(" ".join([x[0] for x in ws])))

    # limpia líneas UI
    out_lines = [ln for ln in out_lines if ln and not UI_BLACKLIST.search(ln)]
    return out_lines

def extract_from_card(lines: List[str]) -> Optional[Dict]:
    if not lines:
        return None

    text_all = "\n".join(lines)

    # precio: debe existir con $
    if not PRICE_STRICT_RE.search(text_all):
        return None
    price = parse_price_strict(text_all)
    if price is None:
        return None

    # intenta marca / nombre:
    # - marca: primera línea corta (1-3 palabras) que NO sea precio
    # - nombre: siguientes 1-3 líneas (hasta antes del precio)
    marca = ""
    nombre = ""

    # elimina líneas que sean puro precio
    no_price = [ln for ln in lines if not PRICE_STRICT_RE.search(ln)]

    for ln in no_price[:6]:
        if 1 <= len(ln.split()) <= 3 and not any(ch.isdigit() for ch in ln):
            marca = ln
            break

    # nombre: toma las líneas “más largas” cerca del inicio
    candidates = [ln for ln in no_price if len(ln) >= 10]
    if candidates:
        # evita agarrar “ordenar...”
        candidates = [c for c in candidates if not UI_BLACKLIST.search(c)]
        nombre = candidates[0] if candidates else ""
    else:
        nombre = no_price[0] if no_price else ""

    gramaje = parse_gramaje(text_all) or parse_gramaje(nombre) or infer_gramaje_fallback(nombre)
    presentacion = infer_presentacion(text_all) or infer_presentacion(nombre)

    return {
        "Marca": marca,
        "Nombre": nombre,
        "Gramaje": gramaje,
        "Precio MXN": price,
        "Presentación": presentacion
    }

def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["k"] = (df["Marca"].fillna("").str.lower().str.strip() + " | " +
               df["Nombre"].fillna("").str.lower().str.strip() + " | " +
               df["Precio MXN"].astype(str))
    df = df.drop_duplicates("k").drop(columns=["k"])
    return df


# =========================
# UI Streamlit
# =========================
st.set_page_config(page_title="Walmart OCR por tarjetas", layout="wide")
st.title("🧠📸 Extraer productos desde captura larga (Walmart) — versión robusta")

with st.sidebar:
    st.header("Ajustes detección de cards")
    lang = st.selectbox("OCR idioma", ["spa", "spa+eng", "eng"], index=0)

    min_w = st.slider("Card min ancho (px)", 200, 600, 280, 10)
    min_h = st.slider("Card min alto (px)", 180, 900, 260, 10)

    max_w = st.slider("Card max ancho (px)", 350, 1200, 650, 10)
    max_h = st.slider("Card max alto (px)", 400, 1600, 900, 10)

    pad = st.slider("Padding alrededor del card (px)", 0, 30, 8, 1)
    show_debug = st.checkbox("Debug: mostrar boxes", value=False)

uploaded = st.file_uploader("Sube la imagen larga (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption=f"Imagen ({img.size[0]}x{img.size[1]})", use_container_width=True)

    if st.button("🔎 Extraer productos"):
        bgr = pil_to_bgr(img)
        H, W = bgr.shape[:2]

        boxes = find_card_boxes(
            bgr=bgr,
            min_w=min_w, min_h=min_h,
            max_w=min(max_w, W),
            max_h=min(max_h, H)
        )

        st.write(f"📦 Cards detectadas: {len(boxes)}")

        rows = []
        debug_img = bgr.copy()

        for (x,y,w,h) in boxes:
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(W, x + w + pad)
            y1 = min(H, y + h + pad)

            roi = bgr[y0:y1, x0:x1]

            lines = ocr_lines_in_roi(roi, lang=lang)
            item = extract_from_card(lines)
            if item:
                rows.append(item)

            if show_debug:
                cv2.rectangle(debug_img, (x0,y0), (x1,y1), (0,255,0), 2)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = dedupe(df)
            df = df.sort_values(["Marca","Nombre","Precio MXN"], ascending=True)

        if show_debug:
            st.subheader("Debug boxes")
            st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.subheader("Resultados")
        st.dataframe(df, use_container_width=True, height=520)

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Productos")
        out.seek(0)

        st.download_button(
            "⬇️ Descargar Excel (.xlsx)",
            data=out,
            file_name="productos_walmart_ocr.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
