"""Форензик-карты в JPEG-домене: считаются на РОДНОМ разрешении, до ресайза.

Зачем этот модуль вообще есть. Признаки, по которым правку видно в блоке 8x8
(соотношение полос DCT, поведение хромы, остаток по решётке квантования), живут
на сетке JPEG. Ресайз кадра эту сетку уничтожает: блочный пробник теряет 0.13
AUC на `coco` и 0.06 на `vision` при переходе 1024 -> 768 (замер в
`docs/forensic_cues.md`, воспроизводится `scripts/forensic_cues.py probe`).
Сеть смотрит на кадр уже после ресайза, то есть на признак, которого там больше
нет — сколько бы фильтров она ни выучила.

Решение простое: посчитать карты до ресайза и ресайзить КАРТЫ, а не картинку.
Карта живёт на сетке 1/8, поэтому её уменьшение почти ничего не стоит и ничего
не ломает — это уже агрегат, а не сигнал на решётке.

Чем это отличается от `pipeline/aic_pipeline/streams.py`. Там SRM/Bayar-остаток
считается сетью из уже отресайзенного входа, и это (а) шум яркости, который по
замерам пуст на `plain_l9` и `coco`, и (б) после ресайза. Здесь другой набор
признаков и другой этап.

Контракт:

    maps = forensic_maps(rgb_uint8, qtable)     # (C, H//8, W//8) float32
    crop = maps[:, y0 // 8:(y0 + side) // 8, x0 // 8:(x0 + side) // 8]

Карта всегда считается по ПОЛНОМУ кадру, а кроп берётся уже из неё. Так
получаются сразу две нужные вещи: сетка 8x8 совпадает с настоящей (кроп в
пикселях обязан быть кратен 8, см. `align8`), а «относительные» каналы
нормируются медианой всего кадра, а не случайного куска.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: (i, j) -> позиция в зигзаге; нужно, чтобы развернуть таблицу квантования,
#: которую PIL отдаёт плоским списком в зигзаг-порядке
ZIGZAG = np.array([
    [0, 1, 5, 6, 14, 15, 27, 28], [2, 4, 7, 13, 16, 26, 29, 42],
    [3, 8, 12, 17, 25, 30, 41, 43], [9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54], [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61], [35, 36, 48, 49, 57, 58, 62, 63],
])

_u, _v = np.mgrid[0:8, 0:8]
_rad = _u + _v
#: полосы по сумме индексов частот: низкие / средние / высокие / верхние AC
BAND_LOW = (_rad >= 1) & (_rad <= 2)
BAND_MID = (_rad >= 3) & (_rad <= 5)
BAND_HIGH = (_rad >= 6) & (_rad <= 9)
BAND_TOP = _rad >= 10

#: Порядок каналов карты. Менять только вместе с `CHANNEL_RANGE` и с чекпоинтами:
#: ветка модели строится под это число каналов.
CHANNELS: tuple[str, ...] = (
    "y31",       # log2 E_high(Y) - log2 E_low(Y): нехватка/избыток верха при той же основе
    "y31_rel",   # то же минус медиана кадра — «относительно этой сцены»
    "y42",       # log2 E_top(Y) - log2 E_mid(Y): самая верхняя полоса
    "y42_rel",
    "c31",       # то же по хроме: единственный живой пиксельный признак на plain_l9
    "c31_rel",
    "cy3",       # хрома минус яркость в верхней полосе: сильнейший одиночный на coco
    "cy3_rel",
    "latt",      # |c/Q - round(c/Q)|: насколько блок сидит на решётке квантования
    "nz_rel",    # число ненулевых квантованных коэффициентов, минус медиана кадра
    "mid_rel",   # log2 E_mid(Y) минус медиана: опорный канал «сколько тут фактуры»
    "qt",        # log2 Q[0,0] — сама таблица квантования, как в CAT-Net
)

#: На что делить каждый канал, чтобы он пришёл в сеть примерно в [-1, 1].
#: Числа сняты по 400 кадрам валидации (p1/p99), а не выведены: логарифмические
#: отношения живут в районе +-10, `latt` — в [0, 0.3], `nz` — в [0, 64].
CHANNEL_RANGE: tuple[float, ...] = (
    8.0, 4.0, 8.0, 4.0, 8.0, 4.0, 8.0, 4.0, 0.25, 16.0, 6.0, 4.0,
)

N_CHANNELS = len(CHANNELS)
STRIDE = 8


def align8(value: int) -> int:
    """Ближайшее снизу число, кратное 8. Кроп обязан быть кратен 8 по всем
    четырём координатам — иначе сетка блоков внутри кропа съезжает и карта
    перестаёт соответствовать картинке."""
    return (int(value) // 8) * 8


def luma_qtable(source: str | Path | bytes | bytearray | memoryview) -> np.ndarray | None:
    """Таблица квантования яркости 8x8 из JPEG. None, если это не JPEG.

    Принимает и путь, и байты: при аугментации перекодированием кадр живёт в
    буфере, а таблица у него уже своя — та, что была у файла, к нему больше не
    относится, и подсунуть её значило бы мерить шум в канале `latt`.

    Читается только заголовок: PIL разбирает DQT в `open`, до декодирования.
    """
    import io

    from PIL import Image

    handle = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray, memoryview)) else source
    try:
        with Image.open(handle) as im:
            tables = getattr(im, "quantization", None)
        if not tables:
            return None
        return np.asarray(tables[0], dtype=np.float32)[ZIGZAG]
    except Exception:
        return None


def _blocks(x: np.ndarray) -> np.ndarray:
    """(H, W) -> (H//8, W//8, 8, 8)."""
    h, w = x.shape[0] // 8, x.shape[1] // 8
    return x[:h * 8, :w * 8].reshape(h, 8, w, 8).transpose(0, 2, 1, 3)


def _dct8(planes: np.ndarray) -> np.ndarray:
    """DCT-II по последним двум осям блока 8x8, ortho-нормировка.

    Через матричное умножение, а не через `scipy.fft.dctn`: на массиве
    (h, w, 8, 8) два einsum с матрицей 8x8 в три-четыре раза быстрее, а в
    даталоадере это единственное место, где считается что-то тяжелее чтения JPEG.
    """
    return np.einsum("ik,...kl,jl->...ij", _DCT_M, planes, _DCT_M, optimize=True)


def _dct_matrix() -> np.ndarray:
    k = np.arange(8)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / 16.0)
    m *= np.sqrt(2.0 / 8.0)
    m[0] /= np.sqrt(2.0)
    return m.astype(np.float32)


_DCT_M = _dct_matrix()


def forensic_maps(
    rgb: np.ndarray,
    qtable: np.ndarray | None = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """RGB (H, W, 3) uint8 -> карта признаков (C, H//8, W//8) float32.

    `qtable` — таблица квантования яркости исходного JPEG (`luma_qtable`).
    Если её нет (кадр не JPEG, синтетика, уже пережатый буфер), берётся единичная:
    канал `latt` тогда вырождается, канал `qt` уходит в 0, остальные не меняются.
    Именно поэтому таблица подаётся отдельным аргументом, а не читается здесь из
    файла: карту часто считают по массиву, которого на диске нет.

    `normalize=False` отдаёт сырые величины — так их удобно сравнивать с
    `scripts/forensic_cues.py`; сеть же всегда получает нормированные.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"нужен RGB (H, W, 3), пришло {rgb.shape}")
    if rgb.shape[0] < 8 or rgb.shape[1] < 8:
        raise ValueError(f"кадр меньше блока 8x8: {rgb.shape}")

    import cv2

    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    dy = _dct8(_blocks(ycc[..., 0]) - 128.0)
    dr = _dct8(_blocks(ycc[..., 1]) - 128.0)
    db = _dct8(_blocks(ycc[..., 2]) - 128.0)

    # свёртка с маской полосы через einsum, а не через d[:, :, band]: fancy-index
    # копирует блок целиком и на кадре 1024x1024 стоит дороже самой DCT
    power_y, power_c = dy * dy, dr * dr + db * db

    def energy(power: np.ndarray, band: np.ndarray) -> np.ndarray:
        return np.einsum("hwij,ij->hw", power, band.astype(np.float32), optimize=True)

    eps = 1e-3
    y_low = np.log2(energy(power_y, BAND_LOW) + eps)
    y_mid = np.log2(energy(power_y, BAND_MID) + eps)
    y_high = np.log2(energy(power_y, BAND_HIGH) + eps)
    y_top = np.log2(energy(power_y, BAND_TOP) + eps)
    c_low = np.log2(energy(power_c, BAND_LOW) + eps)
    c_high = np.log2(energy(power_c, BAND_HIGH) + eps)

    q = np.ones((8, 8), np.float32) if qtable is None else np.asarray(qtable, np.float32)
    # умножение на обратную таблицу вместо деления: это самый большой массив в
    # функции, и деление по нему заметно в общем времени даталоадера
    quantized = dy * (1.0 / q)[None, None, :, :]
    latt = np.abs(quantized - np.round(quantized)).mean((2, 3))
    nz = (np.abs(quantized) > 0.5).sum((2, 3)).astype(np.float32)

    y31, y42 = y_high - y_low, y_top - y_mid
    c31, cy3 = c_high - c_low, c_high - y_high
    rel = lambda x: x - np.median(x)  # noqa: E731 — «относительно этой сцены»

    out = np.stack([
        y31, rel(y31), y42, rel(y42), c31, rel(c31), cy3, rel(cy3),
        latt, rel(nz), rel(y_mid),
        np.full_like(y31, np.log2(float(q[0, 0]))),
    ]).astype(np.float32)

    if normalize:
        out /= np.asarray(CHANNEL_RANGE, np.float32)[:, None, None]
        np.clip(out, -4.0, 4.0, out=out)
    return out


def crop_maps(maps: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    """Вырезать из карты кусок, соответствующий кропу картинки в пикселях.

    Все четыре числа обязаны быть кратны 8 (`align8`): при некратном кропе сетка
    блоков внутри него уже не совпадает с настоящей, и признак становится другим.
    """
    for name, value in (("top", top), ("left", left), ("height", height), ("width", width)):
        if value % 8:
            raise ValueError(f"{name}={value} не кратно 8 — кроп разъедет с сеткой JPEG")
    return maps[:, top // 8:(top + height) // 8, left // 8:(left + width) // 8]


def resize_maps(maps: np.ndarray, size: int) -> np.ndarray:
    """Карту — под модельную сетку `size`, то есть в (C, size//8, size//8).

    Билинейно и без всякого стеснения: карта уже агрегат по блоку, интерполяция
    её не портит. Это и есть весь смысл упражнения — ресайзить карту, а не кадр.
    """
    import cv2

    target = max(1, size // STRIDE)
    if maps.shape[1] == target and maps.shape[2] == target:
        return maps
    moved = np.ascontiguousarray(maps.transpose(1, 2, 0))
    resized = cv2.resize(moved, (target, target), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:  # один канал cv2 схлопывает
        resized = resized[..., None]
    return np.ascontiguousarray(resized.transpose(2, 0, 1))


def maps_after_resize(rgb: np.ndarray, size: int, qtable: np.ndarray | None = None) -> np.ndarray:
    """Контрольная ветка: сначала ресайз кадра, потом те же признаки.

    Нужна не для пользы, а для честности сравнения: без неё выигрыш «карты до
    ресайза» неотличим от выигрыша «в сети стало на 12 каналов и одну ветку
    больше». Плечо с этой функцией отличается от основного ровно одним —
    порядком «ресайз / посчитать».
    """
    import cv2

    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    # таблица исходного JPEG к отресайзенному кадру уже не относится: решётки
    # квантования там нет, и подсовывать её значило бы мерить шум
    return forensic_maps(small, None if qtable is None else np.ones((8, 8), np.float32))
