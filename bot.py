import os
import io
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import re
import json
from collections import defaultdict
import aiohttp
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import cv2
import numpy as np
import discord
from PIL import Image, ImageDraw, ImageFont

# =========================================================
# VARIABLES DE ENTORNO DESDE RAILWAY
# =========================================================

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
CLIENT_ID = int(os.environ.get("CLIENT_ID", "0"))
FORUM_CHANNEL_ID = int(os.environ.get("FORUM_CHANNEL_ID", "0"))
LOG_CHANNEL_ID = int(os.environ.get("LOG_CHANNEL_ID", "0"))
INVALID_LOG_CHANNEL_ID = int(os.environ.get("INVALID_LOG_CHANNEL_ID", "0"))
UPSTASH_REDIS_REST_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
UPSTASH_REDIS_REST_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")

# =========================================================
# RUTAS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ONE_STAR_DIR = BASE_DIR / "one_star_detect"
TWO_STAR_DIR = BASE_DIR / "two_star_detect"
INVALID_DIR = BASE_DIR / "invalid_detect"
HD_DIR = BASE_DIR / "cards_hd"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("gp_detector")

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

REFERENCE_W = 240
REFERENCE_H = 227
MIN_CONFIDENCE_RATIO = 1.08

# NUEVAS CAJAS DE PRUEBA MÁS GRANDES Y CENTRADAS
# Ajustables después viendo el overlay
SLOT_BOXES_REF = [
    (0, 12 , 76,100),    # slot 1
    (81, 12, 159, 100),  # slot 2
    (163, 12, 242, 100),  # slot 3
    (38, 127, 115, 215),  # slot 4
    (123, 127, 200, 215), # slot 5
]
#SLOT_BOXES_REF = [
 #   (0, 5 , 78, 113),    # slot 1
    #(80, 5, 160, 113),  # slot 2
   # (162, 5, 240, 113),  # slot 3
   # (36, 119, 116, 227),  # slot 4
   # (121, 119, 201, 227), # slot 5
#]

CANVAS_W = 2200
CANVAS_H = 2000

CARD_W = 640
CARD_H = 890

DRAW_SLOTS = [
    (120, 50),
    (780, 50),
    (1440, 50),
    (450, 970),
    (1110, 970),
]

TRIGGER_PATTERNS = [
    re.compile(r"god\s*pack", re.IGNORECASE),
    re.compile(r"\[\d/5\]\[\d+P\]\[[^\]]+\]", re.IGNORECASE),
]

VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
PROCESSED_MESSAGES = set()
INVALID_LOG_COOLDOWN = 10
INVALID_LOG_LAST_SENT = {}
DIRECT_GP_WIDTH = 1270
DIRECT_GP_HEIGHT = 300
MAX_SCORE_ACCEPT = 2200
MAX_SCORE_ACCEPT_WITH_GAP = 3600
MIN_SCORE_GAP = 140
MIN_CONFIDENCE_RATIO = 1.10
SAVE_DEBUG_SLOTS = False
# Si está True, NO crea imagen HD.
# Usa la imagen original del webhook para crear el post.
MAINTENANCE_USE_ORIGINAL_IMAGE = True
# =========================================================
# CONFIG GISTS POR GRUPO
# =========================================================
MIN_TWO_STAR_BY_GROUP = {
    "Trainer": 1,
    "Gym_Leader": 1,
    "Elite_Four": 2,
}
CHANNEL_GROUP_MAP = {
   # 1486277594629275770: "Elite_Four",
   # 1487362022864588902: "Trainer",
   # 1491238471556403281: "Gym_Leader",
    1497179630027800728: "Elite_Four",
    1497179450960379915: "Trainer",
    1534456693126332568: "Gym_Leader",
}

GROUP_CONFIG = {
    "Trainer": {
        "FORUM_CHANNEL_ID": 1497177430828384396,
    },
    "Gym_Leader": {
        "FORUM_CHANNEL_ID": 1501347022773686483,
    },
    "Elite_Four": {
        "FORUM_CHANNEL_ID": 1497179653276827720,
    },
}

group_locks = defaultdict(asyncio.Lock)
# =========================================================
# CLASE TEMPLATE
# =========================================================

class TemplateCard:
    def __init__(self, name: str, rarity: str, detect_path: Path, hd_path: Path):
        self.name = name
        self.rarity = rarity
        self.detect_path = detect_path
        self.hd_path = hd_path

        self.detect_bgr = self._load_detect_image(detect_path)
        self.detect_gray = cv2.cvtColor(self.detect_bgr, cv2.COLOR_BGR2GRAY)
        self.detect_hist = self._compute_hist(self.detect_bgr)

        self.hd_rgba = Image.open(hd_path).convert("RGBA")
        self.hd_resized = self.hd_rgba.resize((CARD_W, CARD_H), Image.LANCZOS)

    @staticmethod
    def _load_detect_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img

        try:
            pil_img = Image.open(path).convert("RGB")
            rgb = np.array(pil_img)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
        except Exception as e:
            raise ValueError(f"No se pudo cargar template detect: {path} | detalle: {e}")

    @staticmethod
    def _compute_hist(img_bgr: np.ndarray) -> np.ndarray:
        hist = cv2.calcHist(
            [img_bgr],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 256, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        return hist

# =========================================================
# CARGA DE TEMPLATES
# =========================================================

def load_templates() -> List[TemplateCard]:
    templates: List[TemplateCard] = []

    detect_groups = [
        ("1★", ONE_STAR_DIR),
        ("2★", TWO_STAR_DIR),
        ("INVALID", INVALID_DIR),
    ]

    logger.info("BASE_DIR: %s", BASE_DIR)
    logger.info("HD_DIR existe: %s -> %s", HD_DIR.exists(), HD_DIR)

    if not HD_DIR.exists():
        raise RuntimeError(f"No existe la carpeta cards_hd: {HD_DIR}")

    valid_suffixes = {".png", ".jpg", ".jpeg", ".webp"}

    hd_files = sorted(
        [
            p for p in HD_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in valid_suffixes
        ],
        key=lambda p: p.name.lower()
    )

    for rarity, detect_dir in detect_groups:
        logger.info("Leyendo templates %s desde %s", rarity, detect_dir)

        if not detect_dir.exists():
            logger.warning("No existe carpeta de detección %s: %s", rarity, detect_dir)
            continue

        detect_files = sorted(
            [
                p for p in detect_dir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_suffixes
            ],
            key=lambda p: p.name.lower()
        )

        if not detect_files:
            logger.warning("Carpeta vacía para %s: %s", rarity, detect_dir)
            continue

        for detect_file in detect_files:
            name = detect_file.stem

            if detect_file.stat().st_size == 0:
                logger.warning("Archivo detect vacío, se omite: %s", detect_file.name)
                continue

            possible_hd_files = [
                p for p in hd_files
                if p.stem.lower() == name.lower()
            ]

            if not possible_hd_files:
                logger.warning("No existe versión HD para %s", name)
                continue

            hd_file = possible_hd_files[0]

            if hd_file.stat().st_size == 0:
                logger.warning("Archivo HD vacío, se omite: %s", hd_file.name)
                continue

            try:
                templates.append(TemplateCard(name, rarity, detect_file, hd_file))
                logger.info("Template cargado: %s | rareza=%s", name, rarity)
            except Exception as e:
                logger.warning("Error cargando template %s (%s): %s", name, rarity, e)

    logger.info("Total templates válidos: %s", len(templates))

    if not templates:
        raise RuntimeError("No se cargó ningún template válido en las carpetas de detección")

    return templates


try:
    TEMPLATES = load_templates()
except Exception as e:
    logger.exception("No se pudieron cargar templates: %s", e)
    TEMPLATES = []

# =========================================================
# HELPERS DE IMAGEN
# =========================================================

def pil_to_cv(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def scale_box(box: Tuple[int, int, int, int], src_w: int, src_h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    sx = src_w / REFERENCE_W
    sy = src_h / REFERENCE_H
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def crop_slot(img_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = img_bgr.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    return img_bgr[y1:y2, x1:x2].copy()


def compute_hist(img_bgr: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist(
        [img_bgr],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist
    
def preprocess_slot(slot_bgr: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(slot_bgr, (3, 3), 0)
    yuv = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def compare_images(slot_bgr: np.ndarray, template: TemplateCard) -> float:
    resized = cv2.resize(
        slot_bgr,
        (template.detect_bgr.shape[1], template.detect_bgr.shape[0]),
        interpolation=cv2.INTER_AREA
    )
    resized = preprocess_slot(resized)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)
    tpl_gray = cv2.equalizeHist(template.detect_gray)

    diff = gray.astype(np.float32) - tpl_gray.astype(np.float32)
    mse = float(np.mean(diff ** 2))

    hist_slot = compute_hist(resized)
    hist_corr = cv2.compareHist(
        hist_slot.astype(np.float32),
        template.detect_hist.astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    hist_penalty = (1.0 - max(-1.0, min(1.0, hist_corr))) * 1000.0

    l2_distance = cv2.norm(gray, tpl_gray, cv2.NORM_L2) / gray.size

    total_score = (mse * 0.50) + (hist_penalty * 0.20) + (l2_distance * 1000.0 * 0.30)
    return total_score


def detect_card(slot_bgr: np.ndarray, templates: List[TemplateCard]) -> Tuple[Optional[TemplateCard], List[Tuple[str, float]]]:
    ranking = []

    for t in templates:
        try:
            score = compare_images(slot_bgr, t)
            ranking.append((t, score))
        except Exception:
            continue

    ranking.sort(key=lambda x: x[1])

    if not ranking:
        return None, []

    best_t, best_score = ranking[0]

    top_debug = [
        (f"{x[0].name} [{x[0].rarity}]", round(x[1], 3))
        for x in ranking[:8]
    ]

    if len(ranking) > 1:
        second_t, second_score = ranking[1]
        gap = second_score - best_score
        ratio = second_score / max(best_score, 1e-6)
    else:
        gap = 999999.0
        ratio = 999999.0

    # Regla 1: match muy bueno directo
    if best_score <= MAX_SCORE_ACCEPT and gap >= 80:
        return best_t, top_debug

    # Regla 2: match aceptable, debe ganar claramente
    if (
        best_score <= MAX_SCORE_ACCEPT_WITH_GAP
        and gap >= MIN_SCORE_GAP
        and ratio >= MIN_CONFIDENCE_RATIO
    ):
        return best_t, top_debug

    # Regla 3: caso de score bueno pero gap bajo.
    # Solo aceptar si el score es razonable y NO parece carta nueva/desconocida.
    if best_score <= 2800 and gap >= 90 and ratio >= 1.06:
        return best_t, top_debug

    return None, top_debug


def extract_slots(source_img: Image.Image) -> List[np.ndarray]:
    img_bgr = pil_to_cv(source_img)
    h, w = img_bgr.shape[:2]

    slots = []
    for ref_box in SLOT_BOXES_REF:
        scaled_box = scale_box(ref_box, w, h)
        slot = crop_slot(img_bgr, scaled_box)
        slots.append(slot)

    return slots


def build_hd_canvas(detected_cards: List[Optional[TemplateCard]]) -> Image.Image:
    #canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (20, 20, 20, 255))
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))

    # ===== GRID CONFIG =====
    cols = 3
    rows = 2
    gap_x = 40
    gap_y = 60

    # calcular tamaño total del grid
    total_w = cols * CARD_W + (cols - 1) * gap_x
    total_h = rows * CARD_H + (rows - 1) * gap_y

    # 👇 CENTRADO AUTOMÁTICO
    start_x = (CANVAS_W - total_w) // 2
    start_y = (CANVAS_H - total_h) // 2

    positions = [
        (0, 0), (1, 0), (2, 0),
        (0.5, 1), (1.5, 1)  # fila de abajo centrada
    ]

    for i, card in enumerate(detected_cards):
        if card is None:
            continue

        col, row = positions[i]

        x = int(start_x + col * (CARD_W + gap_x))
        y = int(start_y + row * (CARD_H + gap_y))

        canvas.alpha_composite(card.hd_resized, (x, y))

    return canvas


def create_debug_contact_sheet(
    source_img: Image.Image,
    slots: List[np.ndarray],
    detected_cards: List[Optional[TemplateCard]]
) -> Image.Image:
    thumb_w = 180
    thumb_h = 240
    margin = 20

    width = margin + (thumb_w + margin) * 5
    height = 420

    sheet = Image.new("RGB", (width, height), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)

    draw.text((20, 20), "Debug deteccion GP", fill=(255, 255, 255))

    for i, slot in enumerate(slots):
        thumb = cv_to_pil(slot).resize((thumb_w, thumb_h), Image.LANCZOS)
        x = margin + i * (thumb_w + margin)
        y = 100

        sheet.paste(thumb, (x, y))
        draw.text((x, 60), f"Slot {i + 1}", fill=(255, 255, 255))

        label = f"{detected_cards[i].name} [{detected_cards[i].rarity}]" if detected_cards[i] else "No detectada"
        draw.text((x, 350), label[:24], fill=(180, 220, 255))

    return sheet


def create_box_overlay(source_img: Image.Image) -> Image.Image:
    overlay = source_img.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)

    w, h = overlay.size

    for i, ref_box in enumerate(SLOT_BOXES_REF):
        x1, y1, x2, y2 = scale_box(ref_box, w, h)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=4)
        draw.text((x1 + 5, y1 + 5), f"S{i+1}", fill=(255, 255, 0))

    return overlay


def attachment_looks_like_gp_grid(att: discord.Attachment) -> bool:
    filename = att.filename.lower()
    content_type = (att.content_type or "").lower()

    if content_type.startswith("image/"):
        return True

    if filename.endswith(tuple(ext.lower() for ext in VALID_IMAGE_EXTENSIONS)):
        return True

    return False


async def download_pil_image(attachment: discord.Attachment) -> Image.Image:
    data = await attachment.read()
    return Image.open(io.BytesIO(data)).convert("RGBA")

def is_direct_gp_passthrough_image(img: Image.Image) -> bool:
    # Deja pasar si es exactamente 1270x300 O si es exactamente 1270x1270
    return img.size == (1270, 300) or img.size == (1270, 1270)


def build_pack_label_from_meta(meta: dict) -> str:
    pos = meta.get("pack_position")
    if isinstance(pos, int) and 1 <= pos <= 5:
        return f"[{pos}/5]"
    return "[?/?]"


def process_direct_gp_passthrough(
    message_id: int,
    heartbeat_text: str,
    original_image_path: Path
) -> dict:
    meta = parse_heartbeat_metadata(heartbeat_text)
    pack_label = build_pack_label_from_meta(meta)

    return {
        "two_star_count": 0,
        "found_count": 5,
        "overlay_path": None,
        "debug_path": None,
        "reply_text": "Direct GP image detected. Using original attachment without HD detection.",
        "debug_lines": [
            f"Direct passthrough enabled: first attachment is exactly {DIRECT_GP_WIDTH}x{DIRECT_GP_HEIGHT}.",
            "HD detection was skipped."
        ],
        "files": [],
        "pack_label": pack_label,
        "heartbeat_meta": meta,
        "final_image_path": original_image_path,
        "has_invalid": False,
        "direct_passthrough": True,
    }

async def get_best_gp_image_attachment(message: discord.Message) -> Optional[Tuple[discord.Attachment, Image.Image]]:
    image_attachments = [
        att for att in message.attachments
        if attachment_looks_like_gp_grid(att)
    ]

    if not image_attachments:
        return None

    first_att = image_attachments[0]

    try:
        img = await download_pil_image(first_att)
        logger.info("Selected first image attachment for GP detection: %s", first_att.filename)
        return first_att, img
    except Exception as e:
        logger.warning("Failed to load first image attachment %s: %s", first_att.filename, e)
        return None

async def download_attachment_to_file(att: discord.Attachment, save_path: Path) -> Optional[Path]:
    try:
        data = await att.read()
        save_path.write_bytes(data)
        return save_path
    except Exception as e:
        logger.warning("No se pudo guardar attachment %s: %s", att.filename, e)
        return None


# =========================================================
# HELPERS UPSTASH REDIS
# =========================================================

def redis_headers() -> dict:
    return {
        "Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}",
    }


def users_key(group: str) -> str:
    return f"users:{group}"


def online_key(group: str) -> str:
    return f"online:{group}"


def vip_key(group: str) -> str:
    return f"vip:{group}"


def gp_users_key() -> str:
    return "gp_users"


def live_stats_key(group: str) -> str:
    return f"gp_live_stats:{group}"


def vote_state_key(group: str) -> str:
    return f"gp_votes:{group}"


def safe_json_loads(value, default):
    try:
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        return json.loads(value)
    except Exception:
        return default


async def redis_command(*parts):
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        raise RuntimeError("Missing UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN")

    encoded = "/".join(quote(str(p), safe="") for p in parts)
    url = f"{UPSTASH_REDIS_REST_URL}/{encoded}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=redis_headers()) as resp:
            text = await resp.text()

            if resp.status not in (200, 201):
                raise RuntimeError(f"Redis command failed {resp.status}: {text}")

            data = json.loads(text)
            return data.get("result")


async def redis_get(key: str, default=None):
    result = await redis_command("get", key)
    if result is None:
        return default
    return result


async def redis_set_json(key: str, data) -> None:
    await redis_command("set", key, json.dumps(data))


async def redis_get_json(key: str, default):
    raw = await redis_get(key)
    return safe_json_loads(raw, default)


async def redis_hgetall_json(key: str) -> dict:
    result = await redis_command("hgetall", key)

    if not result:
        return {}

    # Upstash puede devolver dict o lista alternada dependiendo del comando/API.
    if isinstance(result, dict):
        items = result.items()
    else:
        items = zip(result[0::2], result[1::2])

    out = {}

    for field, value in items:
        out[str(field)] = safe_json_loads(value, {})

    return out


async def redis_hset_json(key: str, field: str, value) -> None:
    await redis_command("hset", key, field, json.dumps(value))


async def redis_smembers_ids(key: str) -> List[str]:
    result = await redis_command("smembers", key)

    if not result:
        return []

    return [
        str(x).strip()
        for x in result
        if re.fullmatch(r"\d{16}", str(x).strip())
    ]


async def redis_sadd_id(key: str, value: str) -> bool:
    value = str(value or "").strip()

    if not re.fullmatch(r"\d{16}", value):
        return False

    await redis_command("sadd", key, value)
    return True

async def collect_message_attachments(message: discord.Message) -> List[discord.File]:
    files: List[discord.File] = []

    for idx, att in enumerate(message.attachments, start=1):
        saved_path = OUTPUT_DIR / f"log_original_{message.id}_{idx}_{att.filename}"
        saved = await download_attachment_to_file(att, saved_path)
        if saved:
            files.append(discord.File(str(saved), filename=saved.name))

    return files

async def find_registered_discord_user(
    discord_id: str,
    preferred_group: Optional[str] = None
) -> Optional[dict]:
    """
    Devuelve info si el ID de Discord está en los registros
    (users:<grupo> o miembros de rival_duos). Si no, devuelve None.
    """
    discord_id = str(discord_id or "").strip()

    if not discord_id.isdigit():
        return None

    # primero el grupo del canal, luego los demás
    groups = []
    if preferred_group in GROUP_CONFIG:
        groups.append(preferred_group)
    groups += [g for g in GROUP_CONFIG if g not in groups]

    for group in groups:
        try:
            users = await load_group_users(group)
        except Exception as e:
            logger.warning("No se pudo leer users de %s: %s", group, e)
            continue

        if discord_id in users:
            data = users[discord_id] or {}
            return {
                "discord_id": discord_id,
                "group": group,
                "name": data.get("name") or data.get("heartbeatName") or discord_id,
            }

    # Rival Duos
    try:
        duos = await redis_hgetall_json(rival_duos_key())

        for duo in duos.values():
            members = (duo or {}).get("members") or {}

            if discord_id in members:
                member = members[discord_id] or {}
                return {
                    "discord_id": discord_id,
                    "group": "Rival_Duo",
                    "name": member.get("name") or member.get("heartbeatName") or discord_id,
                }
    except Exception as e:
        logger.warning("No se pudo leer rival_duos: %s", e)

    return None

async def forward_invalid_message(message: discord.Message, reason: str):
    if not INVALID_LOG_CHANNEL_ID:
        return

    original_text = message.content or "(sin texto)"

    # ---------------------------------------------------------
    # Detectar el ID de Discord de la primera línea y validarlo
    # contra los registros. Si no está registrado, no se etiqueta.
    # ---------------------------------------------------------
    tagged_user = None
    owner_id = extract_owner_discord_id_from_first_line(original_text)

    if owner_id:
        try:
            tagged_user = await find_registered_discord_user(
                owner_id,
                get_group_from_channel(message.channel.id)
            )
        except Exception as e:
            logger.exception("Error buscando usuario a etiquetar: %s", e)

    # ---------------------------------------------------------
    # Cooldown: por canal, pero si hay usuario a etiquetar se
    # separa por usuario para no perder avisos de personas distintas
    # ---------------------------------------------------------
    cooldown_key = (
        f"{message.channel.id}:{tagged_user['discord_id']}"
        if tagged_user
        else str(message.channel.id)
    )

    now = time.monotonic()
    last_sent = INVALID_LOG_LAST_SENT.get(cooldown_key, 0)

    if now - last_sent < INVALID_LOG_COOLDOWN:
        try:
            await message.delete()
        except Exception:
            pass

        return

    INVALID_LOG_LAST_SENT[cooldown_key] = now

    try:
        log_channel = client.get_channel(INVALID_LOG_CHANNEL_ID)

        if log_channel is None:
            log_channel = await client.fetch_channel(INVALID_LOG_CHANNEL_ID)

        if log_channel is None:
            return

        mention_line = ""
        allowed = discord.AllowedMentions.none()

        if tagged_user:
            mention_line = f"<@{tagged_user['discord_id']}>\n"
            allowed = discord.AllowedMentions(
                users=[discord.Object(id=int(tagged_user["discord_id"]))]
            )

        content = (
            f"{mention_line}"
            f"**Rejected message from channel <#{message.channel.id}>**\n"
            f"**Author:** {message.author} (`{message.author.id}`)\n"
            f"**Reason:** {reason}\n\n"
            f"**Original message:**\n"
            f"```{original_text[:1300]}```"
        )

        await log_channel.send(
            content=content,
            allowed_mentions=allowed
        )

        try:
            await message.delete()
        except Exception:
            pass

        logger.info(
            "Mensaje inválido registrado y eliminado: "
            "message_id=%s | razón=%s | tagged=%s",
            message.id,
            reason,
            tagged_user["discord_id"] if tagged_user else None
        )

    except Exception as e:
        logger.exception(
            "No se pudo registrar/eliminar mensaje inválido %s: %s",
            message.id,
            e
        )

def is_target_message(message: discord.Message) -> bool:
    if message.webhook_id is None:
        return False

    if message.channel.id not in CHANNEL_GROUP_MAP:
        return False

    content = message.content or ""

    if any(pattern.search(content) for pattern in TRIGGER_PATTERNS):
        return True

    if len(message.attachments) > 0:
        return True

    return False

def parse_heartbeat_metadata(content: str) -> dict:
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    result = {
        "obtainer_user": extract_first_line_username_hint(content),
        "owner_discord_id": extract_owner_discord_id_from_first_line(content),
        "bot_name": None,
        "game_id": None,
        "packs_count": None,
        "pack_position": None,
        "pack_name": None,
        "filename": None,
        "raw_pack_line": None,
    }

    for line in lines:
        m = re.match(r"^(.+?)\s*\((\d+)\)$", line)
        if m:
            result["bot_name"] = m.group(1).strip()
            result["game_id"] = m.group(2).strip()
            continue

        m = re.search(r"(\[(\d)/5\]\[(\d+)P\]\[([^\]]*)\])", line, re.IGNORECASE)
        if m:
            result["raw_pack_line"] = m.group(1)
            result["pack_position"] = int(m.group(2))
            result["packs_count"] = int(m.group(3))
            result["pack_name"] = m.group(4).strip()
            continue

        m = re.match(r"^File name:\s*(.+)$", line, re.IGNORECASE)
        if m:
            result["filename"] = m.group(1).strip()

    return result

def get_group_from_channel(channel_id: int) -> Optional[str]:
    return CHANNEL_GROUP_MAP.get(channel_id)


def extract_owner_discord_id_from_first_line(content: str) -> Optional[str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return None

    first = lines[0]

    # 1. Formato con corchetes, con o sin @ o !:
    #    <@123...>  <@!123...>  <123...>
    m = re.search(r"<@?!?(\d{17,20})>", first)
    if m:
        return m.group(1)

    # 2. ID suelto al inicio de la línea, sin corchetes
    m = re.match(r"^@?(\d{17,20})\b", first)
    if m:
        return m.group(1)

    return None


def extract_first_line_username_hint(content: str) -> Optional[str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return None

    first = lines[0]

    # @wR98 Gold star for you!
    m = re.match(r"^@([^\s]+)", first)
    if m:
        return m.group(1).strip()

    return None


def extract_friend_id(content: str) -> Optional[str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    for line in lines:
        if "God Pack found" in line:
            break

        if line.startswith("<@"):
            continue

        m = re.search(r"\((\d{16})\)", line)
        if m:
            return m.group(1)

    return None
    
async def load_group_users(group: str) -> dict:
    if group not in GROUP_CONFIG:
        return {}

    return await redis_hgetall_json(users_key(group))

def normalize_name_for_match(value: str) -> str:
    value = str(value or "").lower().strip()
    value = value.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    value = re.sub(r"[*_`~|>]", "", value)
    value = re.sub(r"^@+", "", value)
    value = re.sub(r"[:：]+$", "", value)
    value = re.sub(r"[^\w]", "", value)
    return value


def get_user_name_candidates(user_data: dict) -> list:
    names = [
        user_data.get("name"),
        user_data.get("heartbeatName"),
        user_data.get("displayName"),
        user_data.get("display_name"),
        user_data.get("username"),
    ]

    aliases = user_data.get("aliases")
    if isinstance(aliases, list):
        names.extend(aliases)

    return [str(x).strip() for x in names if str(x or "").strip()]


def names_match(input_name: str, user_data: dict) -> bool:
    clean_input = normalize_name_for_match(input_name)

    if not clean_input:
        return False

    for candidate in get_user_name_candidates(user_data):
        clean_candidate = normalize_name_for_match(candidate)

        if not clean_candidate:
            continue

        if clean_candidate == clean_input:
            return True

        if clean_candidate in clean_input or clean_input in clean_candidate:
            return True

    return False

async def resolve_gp_owner(client, content: str, group: str):
    """
    Detecta quién obtuvo el GP a partir del mensaje del webhook.
    Prioridad:
    1. Mención <@id>
    2. Username tipo @nombre
    3. Buscar en Gist (por nombre)
    4. Buscar en Discord API
    5. Fallback a ID o texto
    """

    owner_discord_id = None
    username_hint = None

    # =========================
    # 1. DETECTAR MENCIÓN <@id> o <@!id>
    # =========================
    mention_match = re.search(r"<@!?(\d+)>", content)
    if mention_match:
        owner_discord_id = mention_match.group(1)

    # =========================
    # 2. DETECTAR @username
    # =========================
    username_match = re.search(r"@([a-zA-Z0-9_\.]+)", content)
    if username_match:
        username_hint = username_match.group(1)

    # =========================
    # 3. BUSCAR EN GIST DEL GRUPO
    # =========================
    users = await load_group_users(group)  # ya tienes esta función

    if owner_discord_id and owner_discord_id in users:
        user_data = users[owner_discord_id]
        return {
            "discord_id": owner_discord_id,
            "display_name": user_data.get("name") or user_data.get("heartbeatName") or owner_discord_id,
            "mention": f"<@{owner_discord_id}>",
        }

    # =========================
    # 4. BUSCAR POR NOMBRE EN GIST
    # =========================

    if username_hint:
        for uid, data in users.items():
            if names_match(username_hint, data):
                return {
                    "discord_id": uid,
                    "display_name": data.get("name") or data.get("heartbeatName") or username_hint,
                    "mention": f"<@{uid}>",
                }
    
    # =========================
    # 5. BUSCAR EN DISCORD API
    # =========================
    if owner_discord_id:
        try:
            user = await client.fetch_user(int(owner_discord_id))
            return {
                "discord_id": owner_discord_id,
                "display_name": user.name,
                "mention": f"<@{owner_discord_id}>",
            }
        except Exception:
            pass

    # =========================
    # 6. FALLBACK FINAL
    # =========================
    return {
        "discord_id": owner_discord_id,
        "display_name": username_hint or owner_discord_id or "unknown",
        "mention": f"<@{owner_discord_id}>" if owner_discord_id else "@unknown",
    }

# =========================================================
# RIVAL DUO HELPERS
# =========================================================

def rival_duos_key() -> str:
    return "rival_duos"


def rival_duo_by_gameid_key() -> str:
    return "rival_duo_by_gameid"


async def redis_hget_json(key: str, field: str, default=None):
    try:
        raw = await redis_command("hget", key, field)
        return safe_json_loads(raw, default)
    except Exception as e:
        logger.warning("redis_hget_json error key=%s field=%s: %s", key, field, e)
        return default


async def get_rival_duo_by_id(duo_id: str):
    if not duo_id:
        return None

    return await redis_hget_json(rival_duos_key(), str(duo_id), None)


async def resolve_rival_duo_owner_by_game_id(game_id: str):
    try:
        # CAMBIA ESTA LÍNEA para leer el Hash de Redis directamente
        rival_duos = await redis_hgetall_json(rival_duos_key())

        for duo_id, duo in rival_duos.items():
            online_users = duo.get("onlineUsers") or {}
            members = duo.get("members") or {}

            for member_discord_id, member_data in members.items():
                member_game_id = str(member_data.get("gameId") or "").strip()

                if member_game_id != game_id:
                    continue

                if not online_users.get(str(member_discord_id), False):
                    continue

                display_name = (
                    member_data.get("name")
                    or member_data.get("heartbeatName")
                    or member_discord_id
                )

                return {
                    "discord_id": member_discord_id,
                    "display_name": display_name,
                    "mention": f"<@{member_discord_id}>",
                    "duo_id": duo_id,
                    "duo_name": " & ".join(
                        [
                            (m.get("name") or m.get("heartbeatName") or uid)
                            for uid, m in members.items()
                        ]
                    ),
                    "game_id": game_id,
                }

        return None

    except Exception as e:
        print(f"❌ Error resolving rival duo owner: {e}")
        return None

async def get_rival_duo_mentions_from_online_ids(online_ids):
    """
    Devuelve menciones de Rival Duo usando SOLO el ID activo que está en online:Elite_Four.
    No menciona a los dos, solo al dueño del ID activo.
    """
    mentions = []

    for game_id in online_ids:
        owner = await resolve_rival_duo_owner_by_game_id(game_id)

        if not owner:
            continue

        mention = owner.get("mention")

        if mention and mention not in mentions:
            mentions.append(mention)

    return mentions
    
async def add_vip_id(friend_id: str, group: str) -> bool:
    if not friend_id:
        return False

    if group not in GROUP_CONFIG:
        return False

    friend_id = str(friend_id).strip()

    if not re.fullmatch(r"\d{16}", friend_id):
        logger.warning("VIP inválido, no se guarda: %s", friend_id)
        return False

    try:
        # =====================================================
        # 1. AGREGAR AL SET VIP NORMAL (NO ROMPE TU SISTEMA)
        # =====================================================
        await redis_sadd_id(vip_key(group), friend_id)

        # =====================================================
        # 2. GUARDAR TIMESTAMP APARTE
        # =====================================================
        now_ts = int(datetime.now(timezone.utc).timestamp())

        await redis_command(
            "hset",
            f"{vip_key(group)}:timestamps",
            friend_id,
            now_ts
        )

        logger.info(
            "VIP agregado %s -> %s",
            group,
            friend_id
        )

        return True

    except Exception as e:
        logger.exception("add_vip_id Redis error: %s", e)
        return False

async def cleanup_expired_vips(group: str):
    """
    Elimina SOLO IDs VIP que tengan más de 48h.
    """

    try:
        ts_key = f"{vip_key(group)}:timestamps"

        vip_data = await redis_command("hgetall", ts_key)

        if not vip_data:
            return

        now_ts = int(datetime.now(timezone.utc).timestamp())
        expire_seconds = 48 * 60 * 60

        items = zip(vip_data[0::2], vip_data[1::2])

        for friend_id, saved_ts in items:
            try:
                saved_ts = int(saved_ts)

                if (now_ts - saved_ts) >= expire_seconds:

                    # borrar del set VIP original
                    await redis_command(
                        "srem",
                        vip_key(group),
                        friend_id
                    )

                    # borrar timestamp
                    await redis_command(
                        "hdel",
                        ts_key,
                        friend_id
                    )

                    logger.info(
                        "VIP expirado eliminado %s -> %s",
                        group,
                        friend_id
                    )

            except Exception as e:
                logger.warning(
                    "Error limpiando VIP %s: %s",
                    friend_id,
                    e
                )

    except Exception as e:
        logger.exception("cleanup_expired_vips error: %s", e)

async def load_users_gp() -> dict:
    return await redis_hgetall_json(gp_users_key())


async def save_users_gp(data: dict) -> None:
    for discord_id, value in data.items():
        await redis_hset_json(gp_users_key(), str(discord_id), value)


async def register_user_gp(owner_info: dict) -> None:
    discord_id = owner_info.get("discord_id")
    if not discord_id:
        return

    data = await load_users_gp()

    if discord_id not in data:
        data[discord_id] = {
            "name": owner_info.get("display_name", "Unknown"),
            "gp": 0
        }

    data[discord_id]["gp"] += 1
    data[discord_id]["name"] = owner_info.get("display_name", "Unknown")

    await redis_hset_json(gp_users_key(), discord_id, data[discord_id])


async def get_online_mentions(group: str) -> List[str]:
    if group not in GROUP_CONFIG:
        return []

    try:
        online_ids = await redis_smembers_ids(online_key(group))
        users = await load_group_users(group)

        mentions = []

        for discord_id, user_info in users.items():
            main_id = str(user_info.get("main_id", "")).strip()
            sec_id = str(user_info.get("sec_id", "")).strip()

            if main_id in online_ids or (sec_id and sec_id in online_ids):
                mentions.append(f"<@{discord_id}>")

        if group in ("Elite_Four", "Gym_Leader"):
            duo_mentions = await get_rival_duo_mentions_from_online_ids(online_ids)

            for mention in duo_mentions:
                if mention not in mentions:
                    mentions.append(mention)

        return mentions

    except Exception as e:
        logger.exception("get_online_mentions Redis error: %s", e)
        return []

def get_utc6_date_string() -> str:
    now_utc = datetime.now(timezone.utc)
    utc6 = now_utc - timedelta(hours=6)
    return utc6.date().isoformat()

async def load_live_stats(group: str) -> dict:
    if group not in GROUP_CONFIG:
        return {}

    return await redis_get_json(
        live_stats_key(group),
        {
            "totalGP": 0,
            "totalAlive": 0,
            "currentDay": None,
            "daily": {"gp": 0, "alive": 0},
            "history": [],
            "processedMessages": [],
        }
    )


async def save_live_stats(group: str, stats: dict) -> None:
    if group not in GROUP_CONFIG:
        return

    await redis_set_json(live_stats_key(group), stats)

async def check_daily_reset(group: str, stats: dict) -> dict:
    today = get_utc6_date_string()

    if not stats.get("currentDay"):
        stats["currentDay"] = today
        return stats

    if stats["currentDay"] != today:
        stats["history"].insert(0, {
            "date": stats["currentDay"],
            "gp": stats["daily"]["gp"],
            "alive": stats["daily"]["alive"],
        })
        stats["history"] = stats["history"][:5]
        stats["currentDay"] = today
        stats["daily"] = {"gp": 0, "alive": 0}

    return stats


async def update_stats_safe(group: str, callback):
    async with group_locks[group]:
        stats = await load_live_stats(group)
        stats = await check_daily_reset(group, stats)
        stats = await callback(stats)
        await save_live_stats(group, stats)
        return stats

async def increment_gp_callback(stats: dict) -> dict:
    stats["totalGP"] += 1
    stats["daily"]["gp"] += 1
    return stats

def build_pack_rarity_label(detected_cards: List[Optional["TemplateCard"]]) -> str:
    one_star = 0
    two_star = 0
    invalid = 0

    for card in detected_cards:
        if card is None:
            continue
        if card.rarity == "1★":
            one_star += 1
        elif card.rarity == "2★":
            two_star += 1
        else:
            invalid += 1

    if invalid > 0:
        return f"[INVALID:{invalid}/5]"

    return f"[{two_star}/5]"

def get_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def build_final_poster(
    hd_canvas: Image.Image,
    pack_label: str,
    packs_count: Optional[int],
    bot_name: Optional[str],
) -> Image.Image:
    footer_h = 110
    final_img = Image.new("RGBA", (hd_canvas.width, hd_canvas.height + footer_h), (20, 20, 20, 255))
    final_img.alpha_composite(hd_canvas, (0, 0))

    draw = ImageDraw.Draw(final_img)
    font = get_font(72)

    packs_text = f"[{packs_count}P]" if packs_count is not None else "[?P]"
    bot_text = bot_name or "UnknownBot"
    footer_text = f"{pack_label}   {packs_text}   {bot_text}"

    bbox = draw.textbbox((0, 0), footer_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (final_img.width - text_w) // 2
    y = hd_canvas.height + (footer_h - text_h) // 2

    draw.text((x, y), footer_text, fill=(235, 235, 235), font=font)
    return final_img

def build_forum_post_text(meta: dict, pack_label: str, online_mentions: List[str]) -> str:
    obtainer = meta.get("owner_display_name") or "unknown"
    bot_name = meta.get("bot_name") or "UnknownBot"
    game_id = meta.get("game_id") or "UnknownID"
    packs_count = meta.get("packs_count")
    packs_text = f"[{packs_count}P]" if packs_count is not None else "[?P]"
    filename = meta.get("filename") or "unknown_file.xml"

    active_text = " ".join(online_mentions) if online_mentions else "No active users"

    return (
        "```"
        f"GP found by {obtainer}\n"
        f"{pack_label}{packs_text}[{meta.get('pack_name') or 'PulsingAura'}]\n"
        f"{bot_name} ({game_id})\n"
        f"{filename}\n"
        f"{active_text}"
        "```"
    )


def build_post_title(meta: dict, pack_label: str) -> str:
    packs_count = meta.get("packs_count")
    packs_text = f"[{packs_count}P]" if packs_count is not None else "[?P]"
    bot_name = meta.get("bot_name") or "UnknownBot"
    game_id = meta.get("game_id") or "UnknownID"

    return f"{pack_label} {packs_text} {bot_name} [{game_id}]"

def build_forum_info_panel(meta: dict, pack_label: str, online_mentions: List[str]) -> str:
    obtainer = meta.get("owner_display_name") or "unknown"
    bot_name = meta.get("bot_name") or "UnknownBot"
    game_id = meta.get("game_id") or "UnknownID"
    packs_count = meta.get("packs_count")
    packs_text = f"[{packs_count}P]" if packs_count is not None else "[?P]"
    filename = meta.get("filename") or "unknown_file.xml"

    active_text = " ".join(online_mentions) if online_mentions else "No active users"

    return (
        "```"
        f"GP found by {obtainer}\n"
        f"{pack_label}{packs_text}[{meta.get('pack_name') or 'PulsingAura'}]\n"
        f"{bot_name} ({game_id})\n"
        f"{filename}"
        "```\n"
        f"{active_text}"
        
    )


async def create_forum_post_with_image(
    client: discord.Client,
    group: str,
    title: str,
    image_path: Path,
    content: str = "‎",
) -> Optional[dict]:

    try:
        forum_id = GROUP_CONFIG[group]["FORUM_CHANNEL_ID"]

        channel = client.get_channel(forum_id)
        if channel is None:
            channel = await client.fetch_channel(forum_id)

        if not isinstance(channel, discord.ForumChannel):
            return None

        file = discord.File(str(image_path), filename=image_path.name)

        created = await channel.create_thread(
            name=title,
            content=content,
            file=file,
        )

        thread = created.thread if hasattr(created, "thread") else created

        return {
            "thread": thread,
            "jump_url": thread.jump_url
        }

    except Exception as e:
        logger.exception("No se pudo crear el post del foro: %s", e)
        return None
        
class ForumLinkView(discord.ui.View):
    def __init__(self, post_url: str, meta: dict, pack_label: str, status: str = "none"):
        super().__init__(timeout=None)

        packs = meta.get("packs_count", "?")
        bot = meta.get("bot_name", "Bot")
        game_id = meta.get("game_id", "ID")
        owner = meta.get("owner_display_name") or "unknown"

        if status == "alive":
            prefix = "✅"
        elif status == "dead":
            prefix = "❌"
        else:
            prefix = ""

        label = f"{prefix} {owner} | {pack_label} [{packs}P] {bot} [{game_id}]".strip()

        self.add_item(
            discord.ui.Button(
                label=label[:80],
                style=discord.ButtonStyle.link,
                url=post_url
            )
        )


def build_log_summary(meta: dict, pack_label: str, debug_lines: List[str]) -> str:
    obtainer = meta.get("owner_mention") or "@desconocido"
    bot_name = meta.get("bot_name") or "UnknownBot"
    game_id = meta.get("game_id") or "UnknownID"
    packs_count = meta.get("packs_count")
    packs_text = f"[{packs_count}P]" if packs_count is not None else "[?P]"
    filename = meta.get("filename") or "unknown_file.xml"

    slot_lines = [line for line in debug_lines if line.startswith("Slot ")]
    slot_lines = slot_lines[:5]

    return (
        f"**Resumen GP**\n"
        f"```"
        f"{obtainer}\n"
        f"{bot_name} ({game_id})\n"
        f"{pack_label}{packs_text}[{meta.get('pack_name') or 'PulsingAura'}]\n"
        f"{filename}\n\n"
        + "\n".join(slot_lines) +
        f"```"
    )


async def delete_message_later(message: discord.Message, delay_seconds: int = 172800):
    try:
        await asyncio.sleep(delay_seconds)
        await message.delete()
    except Exception:
        pass
        
def process_gp_image(source_img: Image.Image, message_id: int, heartbeat_text: str) -> dict:
    logger.info("Procesando imagen para message_id=%s", message_id)

    debug_source = OUTPUT_DIR / f"debug_source_{message_id}.png"
    source_img.save(debug_source)

    box_overlay = create_box_overlay(source_img)
    overlay_path = OUTPUT_DIR / f"box_overlay_{message_id}.png"
    box_overlay.save(overlay_path)

    slots = extract_slots(source_img)

    if SAVE_DEBUG_SLOTS:
        for i, slot in enumerate(slots):
            slot_path = OUTPUT_DIR / f"debug_slot_{message_id}_{i+1}.png"
            cv_to_pil(slot).save(slot_path)
            logger.debug("Guardado slot %s: %s", i + 1, slot_path)

    detected_cards: List[Optional[TemplateCard]] = []
    debug_lines: List[str] = []

    for idx, slot in enumerate(slots):
        card, ranking = detect_card(slot, TEMPLATES)
        logger.info("Slot %s ranking: %s", idx + 1, ranking[:5])

        detected_cards.append(card)

        if card:
            debug_lines.append(f"Slot {idx + 1}: {card.name} | rareza={card.rarity}")
        else:
            debug_lines.append(f"Slot {idx + 1}: no detectada")

        if ranking:
            top_text = [f"{name} ({score})" for name, score in ranking[:3]]
            debug_lines.append(f"Top {idx + 1}: {top_text}")

    found_count = sum(1 for c in detected_cards if c is not None)
    logger.info("Detectadas: %s/5", found_count)

    meta = parse_heartbeat_metadata(heartbeat_text)
    pack_label = build_pack_rarity_label(detected_cards)
    has_invalid = any(card is not None and card.rarity == "INVALID" for card in detected_cards)
    two_star_count = sum(
        1 for card in detected_cards
        if card is not None and card.rarity == "2★"
    )

    debug_sheet = create_debug_contact_sheet(source_img, slots, detected_cards)
    out_debug = OUTPUT_DIR / f"gp_debug_{message_id}.png"
    debug_sheet.save(out_debug)

    if found_count == 0:
        return {
            "two_star_count": two_star_count,
            "found_count": found_count,
            "overlay_path": overlay_path,
            "debug_path": out_debug,
            "reply_text": "No se detectó ninguna carta",
            "debug_lines": debug_lines,
            "files": [
                discord.File(str(overlay_path), filename="box_overlay.png"),
                discord.File(str(out_debug), filename="gp_debug.png"),
            ],
            "pack_label": pack_label,
            "heartbeat_meta": meta,
            "final_image_path": None,
            "has_invalid": False,
        }
    if has_invalid:
        return {
            "two_star_count": two_star_count,
            "found_count": found_count,
            "overlay_path": overlay_path,
            "debug_path": out_debug,
            "reply_text": "GP inválido detectado. No se generó imagen HD.",
            "debug_lines": debug_lines,
            "files": [
                discord.File(str(overlay_path), filename="box_overlay.png"),
                discord.File(str(out_debug), filename="gp_debug.png"),
            ],
            "pack_label": pack_label,
            "heartbeat_meta": meta,
            "final_image_path": None,
            "has_invalid": True,
        }

    hd_canvas = build_hd_canvas(detected_cards)

    out_hd = OUTPUT_DIR / f"gp_hd_{message_id}.png"
    hd_canvas.save(out_hd)

    reply_text = "Reconstrucción HD del GP\n\n"
    reply_text += f"{pack_label} "
    reply_text += f"[{meta.get('packs_count', '?')}P] "
    reply_text += f"{meta.get('bot_name', 'UnknownBot')}\n"
    reply_text += "```" + "\n".join(debug_lines[:20])[:1800] + "```"

    return {
        "two_star_count": two_star_count,
        "found_count": found_count,
        "overlay_path": overlay_path,
        "debug_path": out_debug,
        "reply_text": reply_text,
        "debug_lines": debug_lines,
        "files": [
            discord.File(str(out_hd), filename="gp_hd.png"),
            discord.File(str(overlay_path), filename="box_overlay.png"),
            discord.File(str(out_debug), filename="gp_debug.png"),
        ],
        "pack_label": pack_label,
        "heartbeat_meta": meta,
        "final_image_path": out_hd,
        "has_invalid": has_invalid,
    }

# =========================================================
# VOTOS GP
# =========================================================

async def load_vote_state(group: str) -> dict:
    if group not in GROUP_CONFIG:
        return {}

    data = await redis_hgetall_json(vote_state_key(group))


    cutoff = time.time() - (72 * 60 * 60)

    data = {
        key: value
        for key, value in data.items()
        if value.get("created_at", time.time()) >= cutoff
    }

    return data


async def save_vote_state(group: str, data: dict) -> None:
    if group not in GROUP_CONFIG:
        return

    # Guardar solamente los estados individualmente.
    for vote_key, state in data.items():
        await redis_hset_json(vote_state_key(group), vote_key, state)

def clean_old_vote_states(data):
    now = time.time()
    cutoff = now - (72 * 60 * 60)

    return {
        key: value
        for key, value in data.items()
        if value.get("created_at", now) >= cutoff
    }
    
async def update_gp_thread_status(thread_id: int, status: str):
    try:
        thread = client.get_channel(thread_id)
        if thread is None:
            thread = await client.fetch_channel(thread_id)

        if thread is None:
            return

        current_name = thread.name

        # quitar iconos previos para no duplicar
        clean_name = current_name
        clean_name = clean_name.removeprefix("✅ ").removeprefix("❌ ").strip()

        if status == "alive":
            new_name = f"✅ {clean_name}"
        elif status == "dead":
            new_name = f"❌ {clean_name}"
        else:
            return

        await thread.edit(name=new_name[:100])

    except Exception as e:
        logger.warning("No se pudo renombrar thread %s con status %s: %s", thread_id, status, e)

async def update_main_link_button(state: dict, status: str, meta: dict, pack_label: str):
    try:
        channel_id = state.get("link_channel_id")
        message_id = state.get("link_message_id")
        post_url = state.get("post_url")

        if not channel_id or not message_id or not post_url:
            return

        channel = client.get_channel(int(channel_id))
        if channel is None:
            channel = await client.fetch_channel(int(channel_id))

        msg = await channel.fetch_message(int(message_id))

        new_view = ForumLinkView(
            post_url,
            meta,
            pack_label,
            status=status
        )

        await msg.edit(view=new_view)

    except Exception as e:
        logger.warning("No se pudo actualizar botón link: %s", e)
        
class GPVoteView(discord.ui.View):
    def __init__(self, vote_key: str, group: str, fallback_state: dict = None):
        super().__init__(timeout=None)
        self.vote_key = vote_key
        self.group = group
        self.fallback_state = fallback_state or {}

    @discord.ui.button(label="🟢 Alive (0)", style=discord.ButtonStyle.success, custom_id="gp_alive")
    async def alive_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_vote(interaction, "alive")

    @discord.ui.button(label="🔴 Dead (0)", style=discord.ButtonStyle.danger, custom_id="gp_dead")
    async def dead_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_vote(interaction, "dead")

    async def handle_vote(self, interaction: discord.Interaction, vote_type: str):
        data = await load_vote_state(self.group)
        state = data.get(self.vote_key)

        if not state:
            await interaction.response.send_message("No vote state found.", ephemeral=True)
            return

        user_id = str(interaction.user.id)

        if user_id in state["alive_users"] or user_id in state["dead_users"]:
            await interaction.response.send_message("You already voted.", ephemeral=True)
            return

        if vote_type == "alive":
            state["alive_users"].append(user_id)
        else:
            state["dead_users"].append(user_id)

        alive_count = len(state["alive_users"])
        dead_count = len(state["dead_users"])

        status = state.get("status", "none")

        if alive_count >= 1:
            status = "alive"
        elif dead_count >= 4:
            status = "dead"

        state["status"] = status

        if status == "alive" and not state.get("counted_alive", False):
            await update_stats_safe(self.group, self._increment_alive)
            state["counted_alive"] = True

        data[self.vote_key] = state
        await save_vote_state(self.group, data)

        for child in self.children:
            if child.custom_id == "gp_alive":
                child.label = f"🟢 Alive ({alive_count})"
            elif child.custom_id == "gp_dead":
                child.label = f"🔴 Dead ({dead_count})"

        if status in ("alive", "dead"):
            await update_gp_thread_status(int(self.vote_key), status)

            await update_main_link_button(
                state,
                status,
                state.get("meta", {}),
                state.get("pack_label", "")
            )

            for child in self.children:
                child.disabled = True

        voter_name = interaction.user.display_name

        if vote_type == "alive":
            vote_text = f"{voter_name} voted Alive."
        else:
            vote_text = f"{voter_name} voted Dead."

        await interaction.response.send_message(
            vote_text,
            ephemeral=False
        )

        try:
            await interaction.message.edit(view=self)
        except Exception as e:
            logger.warning("Failed to update vote buttons after vote: %s", e)

    async def _increment_alive(self, stats: dict) -> dict:
        stats["totalAlive"] += 1
        stats["daily"]["alive"] += 1
        return stats
# =========================================================
# BOT DISCORD
# =========================================================

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

async def restore_persistent_views():
    logger.info("🔄 Restaurando botones GP...")

    restored = 0

    for group in GROUP_CONFIG.keys():
        try:
            vote_data = await load_vote_state(group)

            for vote_key, state in vote_data.items():
                try:
                    view = GPVoteView(
                        vote_key=str(vote_key),
                        group=group
                    )

                    # restaurar labels actuales
                    alive_count = len(state.get("alive_users", []))
                    dead_count = len(state.get("dead_users", []))

                    for child in view.children:
                        if child.custom_id == "gp_alive":
                            child.label = f"🟢 Alive ({alive_count})"

                        elif child.custom_id == "gp_dead":
                            child.label = f"🔴 Dead ({dead_count})"

                        # si ya terminó la votación, deshabilitar
                        if state.get("status") in ("alive", "dead"):
                            child.disabled = True

                    client.add_view(view)

                    restored += 1

                except Exception as e:
                    logger.exception(
                        "❌ Error restaurando view %s (%s): %s",
                        vote_key,
                        group,
                        e
                    )

        except Exception as e:
            logger.exception(
                "❌ Error cargando vote state %s: %s",
                group,
                e
            )

    logger.info("✅ Views restauradas: %s", restored)

@client.event
async def on_ready():
    logger.info("Bot conectado como %s", client.user)

    try:
        await restore_persistent_views()
    except Exception as e:
        logger.exception("❌ restore_persistent_views failed: %s", e)

@client.event
async def on_message(message: discord.Message):
    try:
        logger.info(
            "on_message author=%s author_id=%s bot=%s webhook_id=%s channel_id=%s content=%r attachments=%s",
            message.author,
            message.author.id,
            message.author.bot,
            message.webhook_id,
            message.channel.id,
            message.content,
            [a.filename for a in message.attachments]
        )

        if message.author.id == client.user.id:
            logger.info("Ignorado: mensaje del propio bot")
            return

        if message.id in PROCESSED_MESSAGES:
            logger.info("Ignorado: mensaje ya procesado %s", message.id)
            return

        if not TEMPLATES:
            logger.info("Ignorado: no hay templates cargados")
            return


        if not is_target_message(message):
            if (
                message.webhook_id is not None
                and message.channel.id in CHANNEL_GROUP_MAP
            ):
                await forward_invalid_message(
                    message,
                    "No coincide con el formato esperado de God Pack."
                )

            logger.info("Ignorado: no coincide con filtro webhook/canal/trigger")
            return

        logger.info("Mensaje webhook objetivo detectado, procesando...")

        gp_result = await get_best_gp_image_attachment(message)
        if gp_result is None:
            logger.info("No se encontró imagen válida en attachments")

            await forward_invalid_message(
                message,
                "El mensaje coincide con el formato de GP, pero no contiene una imagen válida."
            )

            return


        gp_attachment, source_img = gp_result
        original_gp_image_path = OUTPUT_DIR / f"original_gp_{message.id}_{gp_attachment.filename}"
        await download_attachment_to_file(gp_attachment, original_gp_image_path)
        logger.info("gp_attachment seleccionado: %s", gp_attachment.filename)
        logger.info("source_img size: %s", source_img.size)

        PROCESSED_MESSAGES.add(message.id)

        if len(PROCESSED_MESSAGES) > 1000:
            PROCESSED_MESSAGES.clear()

        # result = await asyncio.to_thread(process_gp_image, source_img, message.id, message.content)
        # =====================================================
        # PROCESAMIENTO DE IMAGEN
        # =====================================================

        if MAINTENANCE_USE_ORIGINAL_IMAGE:
            # =================================================
            # MODO MANTENIMIENTO
            # NO HACER DETECCIÓN HD
            # =================================================
            logger.info(
                "🔧 MODO MANTENIMIENTO ACTIVO - "
                "se omite completamente la detección HD para message_id=%s",
                message.id
            )

            maintenance_meta = parse_heartbeat_metadata(message.content)
            maintenance_pack_label = build_pack_label_from_meta(maintenance_meta)

            result = {
                "two_star_count": 0,
                "found_count": 0,
                "overlay_path": None,
                "debug_path": None,
                "reply_text": "Modo mantenimiento: detección HD omitida.",
                "debug_lines": [
                    "🔧 MODO MANTENIMIENTO",
                    "Detección HD omitida.",
                    "Se utilizará la imagen original del webhook."
                ],
                "files": [],
                "pack_label": maintenance_pack_label,
                "heartbeat_meta": maintenance_meta,
                "final_image_path": original_gp_image_path,
                "has_invalid": False,
                "direct_passthrough": False,
                "maintenance_mode": True,
            }

        else:
            # =================================================
            # MODO NORMAL
            # =================================================
            if is_direct_gp_passthrough_image(source_img):
                logger.info(
                    "Direct passthrough image detected for message_id=%s with size=%s. Skipping HD detection.",
                    message.id,
                    source_img.size
                )

                result = process_direct_gp_passthrough(
                    message.id,
                    message.content,
                    original_gp_image_path
                )

            else:
                result = await asyncio.to_thread(
                    process_gp_image,
                    source_img,
                    message.id,
                    message.content
                )

        group = get_group_from_channel(message.channel.id)

        if not group:
            logger.warning("Canal sin grupo configurado: %s",message.channel.id)
            return

# limpiar VIPs expirados
        await cleanup_expired_vips(group)



        owner_info = await resolve_gp_owner(client, message.content, group)
        friend_id = result["heartbeat_meta"].get("game_id") or extract_friend_id(message.content)
        logger.info("Extracted VIP friend_id=%s from message_id=%s", friend_id, message.id)


        if group in ("Elite_Four", "Gym_Leader") and friend_id:
            rival_owner = await resolve_rival_duo_owner_by_game_id(friend_id)

            if rival_owner:
                logger.info(
                    "Rival Duo GP owner override: friend_id=%s discord_id=%s duo=%s",
                    friend_id,
                    rival_owner.get("discord_id"),
                    rival_owner.get("duo_name"),
                )

                owner_info = {
                    "discord_id": rival_owner.get("discord_id"),
                    "display_name": rival_owner.get("display_name"),
                    "mention": rival_owner.get("mention"),
                }

        # Enriquecer meta para el post
        result["heartbeat_meta"]["owner_discord_id"] = owner_info.get("discord_id")
        result["heartbeat_meta"]["owner_display_name"] = owner_info.get("display_name")
        result["heartbeat_meta"]["owner_mention"] = owner_info.get("mention")

        

        min_two_star = MIN_TWO_STAR_BY_GROUP.get(group, 0)



        is_valid_gp = (
            MAINTENANCE_USE_ORIGINAL_IMAGE
            or result.get("direct_passthrough", False)
            or (
                not result.get("has_invalid", False)
                and result.get("found_count", 0) == 5
                and result.get("two_star_count", 0) >= min_two_star
            )
        )

        logger.info(
            "GP VALIDATION | group=%s | is_valid_gp=%s | maintenance=%s | "
            "direct=%s | found=%s | two_star=%s | invalid=%s | friend_id=%s",
            group,
            is_valid_gp,
            MAINTENANCE_USE_ORIGINAL_IMAGE,
            result.get("direct_passthrough", False),
            result.get("found_count", 0),
            result.get("two_star_count", 0),
            result.get("has_invalid", False),
            friend_id,
        )
        if is_valid_gp:
            logger.info("Valid GP confirmed. Trying to add VIP ID: %s | group=%s", friend_id, group)
            if friend_id:
                try:
                    await add_vip_id(friend_id, group)
                except Exception as e:
                    logger.exception("Failed to add VIP, continuing GP flow: %s", e)

            try:
                await register_user_gp(owner_info)
            except Exception as e:
                logger.exception("Failed to save user GP, continuing GP flow: %s", e)

            try:
                await update_stats_safe(group, increment_gp_callback)
            except Exception as e:
                logger.exception("Failed to update live stats, continuing GP flow: %s", e)
        

        post_data = None
        post_url = None
        post_thread = None

        should_create_post = is_valid_gp or MAINTENANCE_USE_ORIGINAL_IMAGE

        post_image_path = None
        if should_create_post:
            if MAINTENANCE_USE_ORIGINAL_IMAGE:
                post_image_path = original_gp_image_path
            else:
                post_image_path = result["final_image_path"]

        if should_create_post and post_image_path is not None:
            post_title = build_post_title(result["heartbeat_meta"], result["pack_label"])
            online_mentions = await get_online_mentions(group)

          #  post_body = build_forum_post_text(
           #     result["heartbeat_meta"],
            #    result["pack_label"],
             #   online_mentions
            #)     

            post_data = await create_forum_post_with_image(
                client,
                group,
                post_title,
                post_image_path,
                content="‎"
            )



        if post_data:
            post_thread = post_data["thread"]
            post_url = post_data["jump_url"]

            if post_thread is not None:
                online_mentions = []

                try:
                    online_mentions = await get_online_mentions(group)
                except Exception as e:
                    logger.exception("Failed to load online mentions: %s", e)

                info_panel = build_forum_info_panel(
                    result["heartbeat_meta"],
                    result["pack_label"],
                    online_mentions
                )

                vote_key = str(post_thread.id)
                vote_data_saved = False

                try:
                    vote_data = await load_vote_state(group)
                    logger.info("Vote state loaded: %d entries for %s", len(vote_data), group)

                    if vote_key not in vote_data:
                        vote_data[vote_key] = {
                            "created_at": time.time(),
                            "group": group,
                            "owner_discord_id": owner_info.get("discord_id"),
                            "friend_id": friend_id,
                            "status": "none",
                            "alive_users": [],
                            "dead_users": [],
                            "counted_alive": False,
                            "link_message_id": None,
                            "link_channel_id": None,
                            "post_url": post_url,
                            "meta": result["heartbeat_meta"],
                            "pack_label": result["pack_label"],
                        }

                    cutoff = time.time() - (72 * 60 * 60)

                    vote_data = {
                        key: value
                        for key, value in vote_data.items()
                        if value.get("created_at", 0) >= cutoff
                    }

                    await save_vote_state(group, vote_data)
                    vote_data_saved = True

                except Exception as e:
                    logger.exception("Failed to save vote state: %s", e)

                if vote_data_saved:
                    vote_view = GPVoteView(vote_key=vote_key, group=group)

                    try:
                        await post_thread.send(
                            content=info_panel,
                            view=vote_view,
                            allowed_mentions=discord.AllowedMentions(users=True)
                        )
                    except Exception as e:
                        logger.exception("Failed to send forum info panel with buttons: %s", e)
                else:
                    try:
                        await post_thread.send(
                            content=info_panel + "\n\nVoting disabled (state not saved)."
                        )
                    except Exception as e:
                        logger.exception("Failed to send forum info panel without buttons: %s", e)

###########
        view = ForumLinkView(
            post_url,
            result["heartbeat_meta"],
            result["pack_label"]
        ) if post_url else None

        # =========================
        # 1. RESPUESTA LIMPIA EN CANAL ORIGINAL
        # =========================
        sent_main = None
        original_files = []



        if is_valid_gp or MAINTENANCE_USE_ORIGINAL_IMAGE:
            if MAINTENANCE_USE_ORIGINAL_IMAGE:
                original_files.append(
                    discord.File(str(original_gp_image_path), filename="gp_original.png")
                )
            elif result.get("direct_passthrough", False):
                original_files.append(
                    discord.File(str(result["final_image_path"]), filename="gp_original.png")
                )
            elif result["final_image_path"] is not None:
                original_files.append(
                    discord.File(str(result["final_image_path"]), filename="gp_hd.png")
                )
        

        if original_files or view is not None:
            sent_main = await message.channel.send(
                files=original_files,
                view=view
            )
        else:
            logger.info("No se responde en canal principal (GP inválido o incompleto).")


        if post_data and sent_main is not None:
            try:
                post_thread = post_data["thread"]
                vote_key = str(post_thread.id)

                vote_data = await load_vote_state(group)

                if vote_key in vote_data:
                    vote_data[vote_key]["link_message_id"] = sent_main.id
                    vote_data[vote_key]["link_channel_id"] = sent_main.channel.id

                    await save_vote_state(group, vote_data)

            except Exception as e:
                logger.exception("Failed to save main link data, continuing GP flow: %s", e)

        # =========================
        # 2. ENVÍO COMPLETO A CANAL DE REGISTRO
        # =========================
        if LOG_CHANNEL_ID:
            log_channel = client.get_channel(LOG_CHANNEL_ID)
            if log_channel is None:
                try:
                    log_channel = await client.fetch_channel(LOG_CHANNEL_ID)
                except Exception as e:
                    logger.exception("No se pudo obtener el canal log: %s", e)
                    log_channel = None

            if log_channel is not None:
                log_summary = build_log_summary(
                    result["heartbeat_meta"],
                    result["pack_label"],
                    result.get("debug_lines", [])
                )

                original_message_files = await collect_message_attachments(message)
                original_text = message.content or "(sin texto)"

                sent_original = await log_channel.send(
                    content=f"**Mensaje original:**\n```{original_text[:1800]}```",
                    files=original_message_files if original_message_files else None
                )

                log_files = []

                if result.get("direct_passthrough", False):
                    passthrough_note = (
                        f"{log_summary}\n"
                        f"**Direct passthrough:** first attachment matched "
                        f"exact size {DIRECT_GP_WIDTH}x{DIRECT_GP_HEIGHT}, so HD detection was skipped."
                    )

                    sent_log = await log_channel.send(
                        content=passthrough_note
                    )
                else:
                    if result.get("overlay_path"):
                        log_files.append(
                            discord.File(str(result["overlay_path"]), filename="box_overlay.png")
                        )

                    if result.get("debug_path"):
                        log_files.append(
                            discord.File(str(result["debug_path"]), filename="gp_debug.png")
                        )

                    sent_log = await log_channel.send(
                        content=log_summary,
                        files=log_files if log_files else None
                    )
                

                asyncio.create_task(delete_message_later(sent_original, 172800))
                asyncio.create_task(delete_message_later(sent_log, 172800))

        # =========================
        # 3. BORRAR MENSAJE ORIGINAL
        # =========================
        try:
            await message.delete()
        except Exception as e:
            logger.warning("No se pudo borrar el mensaje original: %s", e)
    except Exception as e:
        logger.exception("🔥🔥🔥 ERROR REAL EN on_message: %s", e)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("Falta la variable DISCORD_TOKEN en Railway")

    client.run(DISCORD_TOKEN)
