"""Чтение и запись изображений, устойчивые к не-ASCII путям.

`cv2.imread` и `cv2.imwrite` на Windows ходят в файловую систему через ANSI-API,
поэтому на пути с кириллицей (а домашняя папка пользователя вполне может быть
`C:\\Users\\Юрий`) они молча возвращают None и молча ничего не записывают —
без исключения, без сообщения. Отлаживать такое потом крайне неприятно.

Обход стандартный: файл читается/пишется средствами numpy, а кодек OpenCV
получает уже готовый буфер в памяти.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Аналог cv2.imread. Возвращает None, если файла нет или он не декодируется."""
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, FileNotFoundError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str | Path, image: np.ndarray, params: Sequence[int] | None = None) -> bool:
    """Аналог cv2.imwrite. Формат определяется расширением пути."""
    path = Path(path)
    ok, buffer = cv2.imencode(path.suffix, image, list(params or []))
    if not ok:
        return False
    buffer.tofile(str(path))
    return True
