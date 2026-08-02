"""Сабмит: механика регламента и раннер поверх вашей функции предсказания.

Формат по условию задачи:

    submission.zip
      submission.csv          колонки img_path, prediction_path
      predictions/*.png       одноканальные PNG, значения 0/255,
                              размер совпадает с входным изображением

Отдельная проверка существует потому, что сабмит легко испортить незаметно:
маска не того размера, PNG с тремя каналами, значения 0/1 вместо 0/255,
пропущенная строка. Всё это выясняется уже после загрузки, когда попытка
потрачена.

О моделях и чекпоинтах модуль не знает ничего: `predict_fn` ваша.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .data import imread, imwrite
from .metric import FP_AREA_THRESHOLD

REQUIRED_COLUMNS = ["img_path", "prediction_path"]

def _flip(dims: list[int]) -> Callable:
    """Отражение по осям. torch импортируется внутри: модуль читают и без него."""
    def op(tensor):
        import torch

        return torch.flip(tensor, dims=dims)

    return op


#: виды TTA: (прямое преобразование входа, обратное для предсказания). Все
#: обратимы сами собой, поэтому пара всюду одинаковая
TTA_OPS: dict[str, tuple[Callable, Callable]] = {
    "none": (lambda t: t, lambda t: t),
    "hflip": (_flip([-1]), _flip([-1])),
    "vflip": (_flip([-2]), _flip([-2])),
    "hvflip": (_flip([-2, -1]), _flip([-2, -1])),
}


# ---------------------------------------------------------------------------
# таблица тестовой выборки
# ---------------------------------------------------------------------------

def load_test_table(
    test_csv: str | Path,
    template: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Возвращает (таблицу с img_path и prediction_path, корень тестовых путей).

    Пути в `test.csv` заданы относительно папки, где лежит сам CSV. Шаблон
    организаторов, если он есть, переиспользуется как есть: правило именования
    уже задано ими, угадывать его не нужно.
    """
    test_csv = Path(test_csv)
    if not test_csv.exists():
        raise FileNotFoundError(f"нет {test_csv}")
    root = test_csv.parent

    table = pd.read_csv(test_csv)
    if "img_path" not in table.columns:
        raise ValueError(f"{test_csv}: нет колонки img_path (есть {list(table.columns)})")
    table = table[["img_path"]].copy()

    template = Path(template) if template else None
    if template and template.exists():
        tpl = pd.read_csv(template)
        if set(REQUIRED_COLUMNS) <= set(tpl.columns):
            mapping = dict(zip(tpl["img_path"], tpl["prediction_path"]))
            missing = [p for p in table["img_path"] if p not in mapping]
            if missing:
                raise ValueError(
                    f"шаблон {template} не покрывает {len(missing)} строк test.csv "
                    f"(например {missing[:3]})"
                )
            table["prediction_path"] = [mapping[p] for p in table["img_path"]]
            return table, root

    table["prediction_path"] = [
        f"predictions/{Path(p).stem}_pred.png" for p in table["img_path"]
    ]
    return table, root


# ---------------------------------------------------------------------------
# предсказание
# ---------------------------------------------------------------------------

def to_original(prob, orig_h: int, orig_w: int, val_mode: str = "resize") -> np.ndarray:
    """(1, S, S) на модельной сетке -> (H, W) в исходном разрешении.

    Возврат к исходному размеру делается здесь, а не в даталоадере, чтобы
    метрика считалась ровно на том, что уйдёт в сабмит.
    """
    import torch.nn.functional as F

    size = prob.shape[-1]
    if val_mode == "pad":
        # правая и нижняя части сетки — паддинг, а не картинка
        scale = size / float(max(orig_h, orig_w))
        h = max(1, int(round(orig_h * scale)))
        w = max(1, int(round(orig_w * scale)))
        prob = prob[..., :h, :w]
    resized = F.interpolate(
        prob.unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0).squeeze(0).float().cpu().numpy()


def postprocess(
    prob: np.ndarray,
    cls_prob: float = 1.0,
    *,
    mask_threshold: float = 0.5,
    cls_threshold: float = 0.0,
    min_area: float = 0.0,
) -> np.ndarray:
    """Вероятности -> uint8 маска 0/255 с применением всех правил постобработки."""
    mask = prob >= mask_threshold
    if cls_prob < cls_threshold:
        mask[:] = False
    elif min_area > 0 and mask.mean() < min_area:
        mask[:] = False
    return mask.astype(np.uint8) * 255


def _load_batch(paths: Sequence[Path], size: int, val_mode: str):
    """Читает картинки и приводит к модельной сетке. Возвращает батч и размеры."""
    import torch

    images, shapes = [], []
    for path in paths:
        image = imread(path)
        if image is None:
            raise FileNotFoundError(f"не читается: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shapes.append(image.shape[:2])
        if val_mode == "pad":
            scale = size / float(max(image.shape[:2]))
            h = max(1, int(round(image.shape[0] * scale)))
            w = max(1, int(round(image.shape[1] * scale)))
            resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((size, size, 3), dtype=resized.dtype)
            canvas[:h, :w] = resized
            image = canvas
        else:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        images.append(image)

    batch = np.stack(images).astype(np.float32) / 255.0
    return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous(), shapes


def predict_folder(
    paths: Sequence[str | Path],
    predict_fn: Callable,
    *,
    size: int,
    val_mode: str = "resize",
    tta: Sequence[str] = ("none",),
    batch_size: int = 8,
    device=None,
    progress: Callable[[str], None] | None = None,
) -> Iterator[tuple[str, np.ndarray, float]]:
    """Обход теста над вашей функцией. Отдаёт `(stem, вероятности, cls_prob)`.

    `predict_fn(batch) -> (B, 1, H, W)` вероятностей либо пара
    `(вероятности, (B,) вероятность «кадр изменён»)`. Ни про модель, ни про
    чекпоинты библиотека не знает.

    Итератор, а не список: полный тест не обязан помещаться в память, а между
    предсказанием и записью кто-то захочет вставить своё — ансамбль, свою
    постобработку, свои пороги.

    Нормализация входа НЕ делается: чем нормировать, знает только ваша модель.
    На вход `predict_fn` приходит RGB в [0, 1].
    """
    import torch

    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("список изображений пуст")
    unknown = [name for name in tta if name not in TTA_OPS]
    if unknown:
        raise ValueError(f"неизвестные виды TTA {unknown}; есть: {sorted(TTA_OPS)}")

    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        batch, shapes = _load_batch(chunk, int(size), val_mode)
        if device is not None:
            batch = batch.to(device)

        prob_sum, cls_sum = None, None
        with torch.no_grad():
            for op_name in tta:
                forward_op, inverse_op = TTA_OPS[op_name]
                output = predict_fn(forward_op(batch))
                probs, cls = output if isinstance(output, tuple) else (output, None)
                probs = inverse_op(probs).float()
                if cls is None:
                    cls = torch.ones(probs.shape[0], device=probs.device)
                prob_sum = probs if prob_sum is None else prob_sum + probs
                cls = cls.reshape(-1).float()
                cls_sum = cls if cls_sum is None else cls_sum + cls
        probs, cls_probs = prob_sum / len(tta), cls_sum / len(tta)

        for i, path in enumerate(chunk):
            orig_h, orig_w = shapes[i]
            yield (
                path.stem,
                to_original(probs[i], int(orig_h), int(orig_w), val_mode),
                float(cls_probs[i]),
            )
        if progress is not None:
            progress(f"  {min(start + batch_size, len(paths))}/{len(paths)}")


# ---------------------------------------------------------------------------
# запись
# ---------------------------------------------------------------------------

def write(
    out_dir: str | Path,
    items: Iterable[tuple[str, np.ndarray, float]],
    *,
    table: pd.DataFrame | None = None,
    mask_threshold: float = 0.5,
    cls_threshold: float = 0.0,
    min_area: float = 0.0,
) -> dict:
    """Записать маски и `submission.csv`. `items` — то, что отдал раннер.

    `table` — результат `load_test_table`: из неё берутся пути назначения. Без
    неё маски кладутся в `predictions/<stem>_pred.png`, а csv не пишется — так
    можно собрать половину сабмита и дописать вторую другим прогоном.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets: dict[str, str] = {}
    if table is not None:
        targets = {Path(img).stem: rel
                   for img, rel in zip(table["img_path"], table["prediction_path"])}

    areas: list[float] = []
    for stem, prob, cls_prob in items:
        mask = postprocess(
            prob, cls_prob,
            mask_threshold=mask_threshold, cls_threshold=cls_threshold, min_area=min_area,
        )
        target = out_dir / targets.get(stem, f"predictions/{stem}_pred.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        imwrite(target, mask)
        areas.append(float(np.count_nonzero(mask)) / mask.size)

    if table is not None:
        table.to_csv(out_dir / "submission.csv", index=False)

    values = np.asarray(areas) if areas else np.zeros(0)
    non_empty = values > 0
    return {
        "n_images": len(areas),
        "out_dir": str(out_dir),
        "empty_masks": int((~non_empty).sum()),
        "flagged_share": round(float((values >= FP_AREA_THRESHOLD).mean()), 4) if areas else 0.0,
        "area_median_nonempty": round(float(np.median(values[non_empty])), 4)
        if non_empty.any() else 0.0,
    }


def pack_zip(out_dir: str | Path, zip_path: str | Path | None = None) -> Path:
    """Кладёт submission.csv и predictions/ в корень архива."""
    out_dir = Path(out_dir)
    zip_path = Path(zip_path) if zip_path else out_dir.with_suffix(".zip")
    csv_path = out_dir / "submission.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"нет {csv_path}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(csv_path, "submission.csv")
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path != csv_path:
                zf.write(path, path.relative_to(out_dir).as_posix())
    return zip_path


# ---------------------------------------------------------------------------
# проверка
# ---------------------------------------------------------------------------

class _Source:
    """Единый доступ к содержимому — папке или zip-архиву."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.zf = zipfile.ZipFile(self.path) if self.path.suffix == ".zip" else None
        if self.zf is not None:
            self.names = set(self.zf.namelist())

    def read_bytes(self, rel: str) -> bytes | None:
        rel = str(rel).replace("\\", "/")
        if self.zf is not None:
            return self.zf.read(rel) if rel in self.names else None
        target = self.path / rel
        return target.read_bytes() if target.is_file() else None

    def close(self) -> None:
        if self.zf is not None:
            self.zf.close()


def validate(
    path: str | Path,
    test_csv: str | Path,
    *,
    max_problems: int = 20,
    check_sizes: bool = True,
    partial: bool = False,
) -> dict:
    """Проверяет сабмит до загрузки. Возвращает отчёт со списком проблем.

    `partial=True` пропускает сверку состава с test.csv (нужно для частичных
    сборок), но формат каждой перечисленной маски проверяется как обычно.
    """
    source = _Source(path)
    problems: list[str] = []
    areas: list[float] = []
    n_checked = 0
    try:
        raw = source.read_bytes("submission.csv")
        if raw is None:
            return {"ok": False, "problems": ["в сабмите нет submission.csv в корне"]}
        table = pd.read_csv(io.BytesIO(raw))

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in table.columns]
        if missing_cols:
            return {"ok": False, "problems": [f"в submission.csv нет колонок {missing_cols}"]}

        expected, root = load_test_table(test_csv, template=None)
        expected_set = set(expected["img_path"])
        got_set = set(table["img_path"])
        if partial and got_set <= expected_set:
            pass
        elif got_set != expected_set:
            if expected_set - got_set:
                problems.append(
                    f"нет предсказаний для {len(expected_set - got_set)} изображений из test.csv"
                )
            if got_set - expected_set:
                problems.append(
                    f"{len(got_set - expected_set)} строк ссылаются на изображения не из test.csv"
                )
        if table["img_path"].duplicated().any():
            problems.append(f"дубликаты img_path: {int(table['img_path'].duplicated().sum())}")

        for img_rel, pred_rel in zip(table["img_path"], table["prediction_path"]):
            data = source.read_bytes(pred_rel)
            if data is None:
                if len(problems) < max_problems:
                    problems.append(f"файл маски не найден: {pred_rel}")
                continue

            mask = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if mask is None:
                problems.append(f"не читается как изображение: {pred_rel}")
                continue
            if not str(pred_rel).lower().endswith(".png"):
                problems.append(f"не PNG: {pred_rel}")
            if mask.ndim != 2:
                problems.append(f"маска не одноканальная ({mask.ndim} измерения): {pred_rel}")
                mask = mask[..., 0]

            values = set(np.unique(mask).tolist())
            if not values <= {0, 255}:
                problems.append(
                    f"значения не только 0/255 ({sorted(values)[:5]}...): {pred_rel}"
                )

            if check_sizes:
                image_path = root / img_rel
                if image_path.exists():
                    with Image.open(image_path) as im:
                        width, height = im.size
                    if mask.shape[:2] != (height, width):
                        problems.append(
                            f"размер маски {mask.shape[:2]} != изображения "
                            f"{(height, width)}: {pred_rel}"
                        )

            areas.append(float(np.count_nonzero(mask)) / mask.size)
            n_checked += 1
            if len(problems) >= max_problems:
                problems.append("... дальнейшие проблемы не перечисляю")
                break
    finally:
        source.close()

    values_arr = np.asarray(areas) if areas else np.zeros(0)
    non_empty = values_arr > 0
    return {
        "ok": not problems,
        "problems": problems,
        "n_rows": int(len(table)),
        "n_checked": n_checked,
        "empty_share": round(float((~non_empty).mean()), 4) if len(values_arr) else None,
        "flagged_share": round(float((values_arr >= FP_AREA_THRESHOLD).mean()), 4)
        if len(values_arr) else None,
        "area_median_nonempty": round(float(np.median(values_arr[non_empty])), 4)
        if non_empty.any() else 0.0,
    }
