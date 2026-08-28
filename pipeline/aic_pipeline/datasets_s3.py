"""Данные серии `s3` (ноутбук 16): датасет, геометрия, загрузчики.

**Почему это модуль, а не ячейка ноутбука.** Воркеры `DataLoader` на Windows
стартуют через `spawn`: дочерний процесс не наследует память родителя, а
распикливает датасет. Пикль хранит класс ссылкой — `<модуль>.SegDataset`, — и
ребёнок ищет его у себя. Если класс объявлен в ячейке, ссылка указывает на
`__main__`, а `__main__` у спавненного процесса — заглушка `multiprocessing`:
у ядра Jupyter нет `__file__`, переимпортировать «главный модуль» не из чего.
Класс не находится, воркер падает, и остаётся `num_workers=0`.

Отсюда правило: **всё, что едет в воркер, живёт в импортируемом файле.** Модель,
лосс и цикл обучения остаются в ноутбуке — они работают в главном процессе и
пикклиться не должны.

На Linux с `fork` это не нужно (ребёнок наследует адресное пространство), но
хуже не делает: ценой одного импорта ноутбук ведёт себя одинаково на обеих
системах. С Python 3.14, где дефолтом на POSIX стал `forkserver`, Linux начнёт
вести себя как Windows — и тогда модуль понадобится и там.

Что здесь сознательно НЕ делается: геометрия не отдаётся `albumentations`.
Карту признаков надо провести ровно через те же кроп, флипы и ресайз, что и
кадр, но на своей сетке 1/8, а `Compose` умеет только полное разрешение — это
32 МБ на кадр при двенадцати каналах. Плюс `RandomResizedCrop` режет по
произвольным координатам, а кроп обязан быть кратен 8, иначе сетка блоков
внутри него разъезжается с настоящей.
"""

from __future__ import annotations

import math
import types

# `aic` первым: он выставляет KMP_DUPLICATE_LIB_OK (две копии OpenMP в conda-среде
# роняют процесс при совместном импорте torch и numpy) и NO_ALBUMENTATIONS_UPDATE
import aic  # noqa: F401
from aic import data as aic_data
from aic.forensic import align8, crop_maps, forensic_maps, luma_qtable, resize_maps

import albumentations as A  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from albumentations.pytorch import ToTensorV2  # noqa: E402
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # noqa: E402

cv2.setNumThreads(0)  # иначе воркеры дерутся за ядра с главным процессом

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_pixel(train: bool) -> A.Compose:
    """Только пиксельные операции: геометрия сделана до этого руками.

    Чего здесь нет и почему. `ImageCompression` уехал в `SegDataset`, на родное
    разрешение — иначе он пережимал бы кадр уже после расчёта карты, и у плеча
    `native` карта перестала бы соответствовать картинке. `Downscale` убран по
    той же причине (это тоже ресемплинг). То, что осталось, — яркость и цвет:
    каналы карты это логарифмические ОТНОШЕНИЯ полос, и общий множитель из них
    сокращается.
    """
    pixel = [] if not train else [
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.3),
        A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=12, val_shift_limit=8, p=0.2),
    ]
    return A.Compose(pixel + [A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])


def jpeg_roundtrip(rgb: np.ndarray, quality: int):
    """Пережать кадр и вернуть его вместе с НОВОЙ таблицей квантования.

    Таблица исходного файла к пережатому кадру уже не относится: коэффициенты
    теперь сидят на другой решётке, и канал `latt` со старой таблицей мерил бы
    шум. Читаем её из того же буфера, из которого декодируем.
    """
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return rgb, None
    blob = buf.tobytes()
    decoded = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), luma_qtable(blob)


def sample_crop(h: int, w: int, scale, rng):
    """Кроп со случайным масштабом, все координаты кратны 8.

    Смещаем весь квадрат, а не обрезаем по рамке маски: обрезка по bbox ставила
    бы правку всегда в середину кадра, и модель выучила бы позицию, которой в
    тесте нет.

    Порядок ограничений важен: **пол в 64 px накладывается ДО потолка по кадру,
    а не после.** В обратном порядке на кадре короче 64 px сторона кропа выходит
    больше самого кадра, и розыгрыш смещения падает с `high <= 0`. Таких кадров
    в датасете два из 103 тысяч (51x72 и 72x56), но сэмплер берёт с возвращением
    24 000 раз за эпоху — то есть примерно раз в пару эпох он в них попадает.
    """
    limit = align8(min(h, w))          # кроп не больше кадра и кратен 8
    if limit < 8:
        raise ValueError(f"кадр {h}x{w} меньше блока 8x8 — кропать нечего")
    side = int(round(math.sqrt(rng.uniform(*scale)) * min(h, w)))
    side = min(align8(max(64, side)), limit)
    top = align8(int(rng.integers(0, h - side + 1)))
    left = align8(int(rng.integers(0, w - side + 1)))
    return top, left, side


def dihedral(x: np.ndarray, k: int, flip_h: bool, flip_v: bool, spatial=(0, 1)):
    """Поворот на k*90° и отражения по осям `spatial`.

    Одна и та же тройка применяется к кадру, маске и карте. Карте это не вредит:
    полосы DCT заданы по сумме индексов частот, а она симметрична относительно
    транспонирования; знак коэффициента при отражении меняется, энергия — нет.
    Исключение — канал `latt`: таблица квантования при повороте на 90°
    транспонируется не сама в себя, и остаток по решётке слегка «плывёт». Эффект
    мелкий, а плечи видят одну и ту же аугментацию, так что сравнение не портит.
    """
    if k:
        x = np.rot90(x, k, axes=spatial)
    if flip_h:
        x = np.flip(x, axis=spatial[1])
    if flip_v:
        x = np.flip(x, axis=spatial[0])
    return np.ascontiguousarray(x)


class SegDataset(Dataset):
    """Отдаёт image, mask, label и — если плечо этого просит — fmap.

    `fmap_mode`:
      none    поля нет, батч как у якоря;
      post    карта считается из уже отресайзенного кропа (контроль ёмкости);
      native  карта считается по полному кадру до ресайза, из неё вырезается
              кроп и уже он уменьшается до сетки size/8.

    `fmap_keep` — индексы каналов `aic.forensic.CHANNELS`, которые доходят до
    сети. По умолчанию из набора выброшен `qt`: это `log2 Q[0,0]`, константа по
    кадру, и она одна отделяет позитив от негатива с AUC 0.727 — признак кадра,
    а не области.

    `label` считается ПОСЛЕ кропа: кроп вполне может пройти мимо правки, и тогда
    гейт обязан увидеть «чисто», иначе он учится на заведомо неверной метке.
    """

    #: потолок числа компонент GT в одном кадре. Лосс раскладывает их в тензор
    #: фиксированной ширины, чтобы не спрашивать у GPU максимум и не платить
    #: синхронизацией на каждом кадре батча.
    MAX_COMPONENTS = 256

    def __init__(self, ws, df: pd.DataFrame, size: int, *, fmap_mode: str = "none",
                 fmap_keep=None, train: bool = False, crop_scale=(0.35, 1.0),
                 jpeg_aug_p: float = 0.0, jpeg_aug_quality=(60, 100), seed: int = 0,
                 need_comp: bool = False, need_dist: bool = False,
                 full_frame: bool = False):
        if fmap_mode not in {"none", "post", "native"}:
            raise ValueError(f"неизвестный режим карты: {fmap_mode}; есть none|post|native")
        self.ws, self.df = ws, df.reset_index(drop=True)
        self.size, self.fmap_mode, self.train = size, fmap_mode, train
        self.fmap_keep = None if fmap_keep is None else list(fmap_keep)
        self.crop_scale, self.seed = crop_scale, seed
        self.jpeg_aug_p, self.jpeg_aug_quality = jpeg_aug_p, jpeg_aug_quality
        self.pixel = build_pixel(train)
        self.need_comp, self.need_dist = bool(need_comp), bool(need_dist)
        self.full_frame = bool(full_frame)
        self.is_negative = self.df["is_negative"].to_numpy(dtype=bool)

    def __len__(self) -> int:
        return len(self.df)

    def _select(self, maps: np.ndarray) -> np.ndarray:
        return maps if self.fmap_keep is None else maps[self.fmap_keep]

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        path = self.ws.resolve(row["chng_path"])
        bgr = aic_data.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"не читается кадр: {row['chng_path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        mask = aic_data.imread(self.ws.resolve(row["gt_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"не читается маска: {row['gt_path']}")
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (mask >= 128).astype(np.float32)

        qtable = luma_qtable(path)
        # своё зерно на воркер и индекс: иначе воркеры повторяют друг друга
        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2 ** 31))

        # 1. перекодирование — на родном разрешении и ДО расчёта карты
        if self.train and self.jpeg_aug_p and rng.random() < self.jpeg_aug_p:
            rgb, qtable = jpeg_roundtrip(rgb, rng.integers(*self.jpeg_aug_quality))

        # 2. карта по полному кадру: медианы «относительных» каналов должны
        #    считаться по всей сцене, а не по случайному куску
        maps = forensic_maps(rgb, qtable) if self.fmap_mode == "native" else None

        # 3. кроп, кратный 8
        #
        # `full_frame` выключает случайный кроп и отдаёт обучению ТУ ЖЕ рамку,
        # что видит инференс. Ради этого режим и написан: замер показывает две
        # рассогласованности между обучением и валидацией, и обе тут.
        #
        # Масштаб. `sample_crop` берёт в среднем sqrt(E[U(0.35,1)]) = 0.813
        # короткой стороны и растягивает до `size`, а валидация ужимает в те же
        # `size` ВЕСЬ кадр. То есть объекты в обучении крупнее в 1.23 раза.
        # Отсюда и то, что модель, обученная на 640, показывает лучший AIC на
        # входе 768 (множитель 1.20) без всякого дообучения.
        #
        # Пропорции. Кроп КВАДРАТНЫЙ, а валидация сплющивает прямоугольник в
        # квадрат: для кадра 1024x768 это сжатие по горизонтали в 0.75, которого
        # модель не видела ни разу.
        if self.train and not self.full_frame:
            top, left, side = sample_crop(h, w, self.crop_scale, rng)
            box = (top, left, side, side)
        else:
            box = (0, 0, align8(h), align8(w))   # хвост до 7 пикселей отрезаем,
            if min(box[2], box[3]) < 8:          # иначе кадр и карта описывают разное
                raise ValueError(f"кадр {h}x{w} меньше блока 8x8: {row['chng_path']}")
        top, left, ch, cw = box
        rgb = rgb[top:top + ch, left:left + cw]
        mask = mask[top:top + ch, left:left + cw]
        if maps is not None:
            maps = crop_maps(maps, top, left, ch, cw)

        # 4. дигедральная группа — одна и та же для кадра, маски и карты
        #
        # В режиме `full_frame` повороты на 90 градусов ОТКЛЮЧЕНЫ. Они меняют
        # ориентацию кадра, а значит и сторону, по которой ресайз его сплющит:
        # у повёрнутого пейзажа сжатие уходит по вертикали. Плечо проверяет
        # ровно совпадение геометрии с инференсом, и оставить повороты значило
        # бы протащить обратно то самое рассогласование. Отражения безопасны —
        # они пропорции не трогают.
        if self.train:
            k = 0 if self.full_frame else int(rng.integers(4))
            flip_h, flip_v = rng.random() < 0.5, rng.random() < 0.2
            rgb = dihedral(rgb, k, flip_h, flip_v, (0, 1))
            mask = dihedral(mask, k, flip_h, flip_v, (0, 1))
            if maps is not None:
                maps = dihedral(maps, k, flip_h, flip_v, (1, 2))

        # 5. ресайз: кадр — в модельную сетку, карта — в её восьмую долю
        image = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask_t = torch.from_numpy((mask >= 0.5).astype(np.float32)).unsqueeze(0)

        out = {
            "image": self.pixel(image=image)["image"],
            "mask": mask_t,
            "label": torch.tensor([float(mask_t.max() > 0)], dtype=torch.float32),
        }
        # Компоненты и расстояние считаются УЖЕ на модельной сетке, после ресайза.
        # Так их не нужно тащить через кроп и дигедральную группу, и они точно
        # соответствуют той сетке, на которой живёт лосс. Цена — пара
        # миллисекунд на кадр в воркере, и только когда плечо этого просит.
        if self.need_comp or self.need_dist:
            binary = (mask >= 0.5).astype(np.uint8)
            if self.need_comp:
                count, labels = cv2.connectedComponents(binary, connectivity=8)
                if count > self.MAX_COMPONENTS:
                    # Лишние куски сливаем в фон, а не обрезаем метки: иначе
                    # scatter_add сложил бы разные компоненты в одну корзину.
                    labels[labels >= self.MAX_COMPONENTS] = 0
                out["comp"] = torch.from_numpy(labels.astype(np.int16)).unsqueeze(0)
            if self.need_dist:
                if binary.any():
                    dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
                    dist = dist / float(np.hypot(*binary.shape))
                else:
                    dist = np.zeros(binary.shape, dtype=np.float32)
                out["dist"] = torch.from_numpy(dist.astype(np.float32)).unsqueeze(0)

        if self.fmap_mode == "native":
            out["fmap"] = torch.from_numpy(self._select(resize_maps(maps, self.size)))
        elif self.fmap_mode == "post":
            # то же самое, что `maps_after_resize`, но без второго ресайза: кадр
            # уже приведён к модельной сетке. Таблица не передаётся — решётки
            # квантования тут больше нет
            out["fmap"] = torch.from_numpy(self._select(forensic_maps(image, None)))
        return out


def split_frames(folds: pd.DataFrame, cfg: dict):
    """Train/val по фолду. Валидация прореживает ТОЛЬКО позитивы.

    Негативов в фолде ~640, и именно они дают FPR_neg — половину метрики.
    Прореженные вчетверо, они дали бы шаг дискретизации 0.006 по FPR, то есть
    больше ожидаемого эффекта серии.
    """
    train_df = folds[folds["fold"] != cfg["fold"]].reset_index(drop=True)
    val_df = folds[folds["fold"] == cfg["fold"]].reset_index(drop=True)
    frac = cfg["val_frac"]
    if frac and frac < 1.0:
        if cfg["val_keep_negatives"]:
            negatives = val_df[val_df["is_negative"]]
            positives = val_df[~val_df["is_negative"]].sample(frac=frac,
                                                              random_state=cfg["seed"])
            val_df = pd.concat([positives, negatives])
        else:
            val_df = val_df.sample(frac=frac, random_state=cfg["seed"])
        val_df = val_df.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
    return train_df, val_df


def make_data(ws, folds: pd.DataFrame, cfg: dict, *, fmap_keep=None,
              workers: int | None = None, pin_memory: bool = False):
    """Датасеты, сэмплер и загрузчики под одно плечо.

    Плечи различаются `fmap` и `seed`, поэтому данные собираются заново на
    каждое — но из одного и того же разбиения по фолдам.
    """
    train_df, val_df = split_frames(folds, cfg)
    leak = set(train_df["group_id"]) & set(val_df["group_id"])
    if leak:
        raise AssertionError(f"утечка group_id между train и val: {len(leak)}")

    # Дополнительные цели считаются, только если их просит лосс плеча: и
    # компоненты, и distance transform стоят времени воркера, а даталоадер тут
    # и без того близок к тому, чтобы стать узким местом.
    common = dict(size=cfg["size"], fmap_mode=cfg["fmap"], fmap_keep=fmap_keep,
                  seed=cfg["seed"])
    # Дополнительные цели нужны ТОЛЬКО обучению: `validate` их не читает, а
    # стоят они времени воркера, и даталоадер тут и без того близок к тому,
    # чтобы стать узким местом.
    train_ds = SegDataset(ws, train_df, train=True, crop_scale=cfg["crop_scale"],
                          jpeg_aug_p=cfg["jpeg_aug_p"],
                          jpeg_aug_quality=cfg["jpeg_aug_quality"],
                          need_comp=float(cfg.get("w_comp", 0.0)) > 0,
                          need_dist=float(cfg.get("w_far", 0.0)) > 0,
                          full_frame=bool(cfg.get("full_frame", False)), **common)
    val_ds = SegDataset(ws, val_df, train=False, **common)

    # Негативов в данных 3%, а они дают половину метрики: без перевзвешивания
    # модель почти не видит чистых кадров и щедро галлюцинирует на них маски.
    neg = train_ds.is_negative
    weights = np.where(neg, cfg["negative_fraction"] / neg.sum(),
                       (1 - cfg["negative_fraction"]) / (~neg).sum())
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=cfg["epoch_size"], replacement=True)

    n = cfg["num_workers"] if workers is None else int(workers)
    kwargs = dict(num_workers=n, pin_memory=pin_memory, persistent_workers=n > 0)
    return types.SimpleNamespace(
        train_df=train_df, val_df=val_df, train_ds=train_ds, val_ds=val_ds, workers=n,
        train_loader=DataLoader(train_ds, batch_size=cfg["bs"], sampler=sampler,
                                drop_last=True, **kwargs),
        val_loader=DataLoader(val_ds, batch_size=cfg["bs"] * 2, shuffle=False, **kwargs),
    )
