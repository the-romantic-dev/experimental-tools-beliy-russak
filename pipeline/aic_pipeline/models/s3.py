"""Модель серии s3 отдельным модулем: собирать её приходится и вне ноутбука.

Ноутбук 16 держит архитектуру в ячейках, и для обучения это нормально. Но
сабмит и разбор ошибок — отдельные процессы, которым нужен ТОТ ЖЕ класс: веса
грузятся `strict=True`, и расхождение в именах полей обязано ронять загрузку, а
не давать тихо неверные предсказания. Плюс воркеры `DataLoader` на Windows
стартуют через spawn и распикливают датасет по ссылке `<модуль>.Класс` — из
ячейки, которая живёт в `__main__`, он не поднимется. Та же причина, по которой
рядом лежит `datasets_s3.py`.

**Имена полей здесь обязаны совпадать с ноутбуком.** Если правите архитектуру
там — правьте и тут, иначе первым сломается сабмит, причём на загрузке весов.

Главное, ради чего модуль вообще написан: на инференсе форензик-карта считается
по РОДНОМУ кадру, до ресайза. `aic.submit.predict_folder` этого не умеет — он
отдаёт только уже уменьшенный батч, и ветка увидела бы другое распределение
входа, чем в обучении. Молча, без единой ошибки, с потерей ровно того признака,
ради которого серия и затевалась.
"""

from __future__ import annotations

import math
from pathlib import Path

import aic  # раньше cv2/numpy/torch: он выставляет KMP_DUPLICATE_LIB_OK

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from aic.forensic import CHANNELS, align8, crop_maps, forensic_maps, luma_qtable, resize_maps

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


# ---------------------------------------------------------------------------
# блоки: копия ноутбука, имена полей менять нельзя
# ---------------------------------------------------------------------------

def make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
    raise ValueError(f"неизвестная нормализация: {kind}")


class DWSep(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))


class ForensicBranch(nn.Module):
    def __init__(self, in_ch: int, ch8=64, ch16=96, ch32=128):
        super().__init__()
        self.norm0 = nn.BatchNorm2d(in_ch)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch8, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch8), nn.GELU(),
        )
        self.ref8 = DWSep(ch8, ch8)
        self.to16 = DWSep(ch8, ch16, stride=2)
        self.to32 = DWSep(ch16, ch32, stride=2)
        self.out_channels = {8: ch8, 16: ch16, 32: ch32}

    def forward(self, fmap):
        s8 = self.ref8(self.stem(self.norm0(fmap)))
        s16 = self.to16(s8)
        return {8: s8, 16: s16, 32: self.to32(s16)}


class GatedFuse(nn.Module):
    def __init__(self, enc_ch: int, aux_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(enc_ch + aux_ch, enc_ch, 1, bias=False),
            nn.BatchNorm2d(enc_ch), nn.GELU(),
        )
        self.gamma = nn.Parameter(torch.zeros(1, enc_ch, 1, 1))

    def forward(self, enc, aux):
        if aux.shape[-2:] != enc.shape[-2:]:
            aux = F.interpolate(aux, size=enc.shape[-2:], mode="bilinear", align_corners=False)
        return enc + self.gamma * self.proj(torch.cat([enc, aux], 1))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = make_norm(norm, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = make_norm(norm, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm1(self.conv1(x)))
        return self.act(self.norm2(self.conv2(x)))


class GateHead(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, deepest):
        return self.fc(self.drop(torch.flatten(self.pool(deepest), 1)))


class Segmenter(nn.Module):
    """U-Net-подобный сегментатор с веткой по форензик-карте.

    `fmap_used` приходит числом, а не глобалью: ноутбук берёт его из ячейки
    конфига, а здесь единственный источник правды — `config.yaml` прогона.
    """

    FUSE_AT = (8, 16, 32)

    def __init__(self, encoder="convnext_tiny", decoder_channels=(128, 64, 32, 16, 16),
                 norm="batch", pretrained=False, use_fmap=False, fmap_ch=(64, 96, 128),
                 fmap_used: int = 11, aux_weight=0.0, verbose=False):
        super().__init__()
        self.encoder = timm.create_model(encoder, features_only=True,
                                         pretrained=pretrained, in_chans=3)
        self.reductions = list(self.encoder.feature_info.reduction())
        channels = list(self.encoder.feature_info.channels())

        self.use_fmap = use_fmap
        self.fmap_used = int(fmap_used)
        if use_fmap:
            missing = [r for r in self.FUSE_AT if r not in self.reductions]
            if missing:
                raise ValueError(f"у {encoder} нет карт на страйдах {missing}: {self.reductions}")
            self.branch = ForensicBranch(self.fmap_used, *fmap_ch)
            self.fuse = nn.ModuleDict({
                str(r): GatedFuse(channels[self.reductions.index(r)],
                                  self.branch.out_channels[r])
                for r in self.FUSE_AT
            })

        skip_at = dict(zip(self.reductions[:-1], channels[:-1]))
        steps = int(math.log2(self.reductions[-1]))
        self.skip_reductions, blocks = [], []
        in_ch, reduction = channels[-1], self.reductions[-1]
        for i in range(steps):
            reduction //= 2
            skip_ch = skip_at.get(reduction, 0)
            self.skip_reductions.append(reduction if skip_ch else None)
            blocks.append(DecoderBlock(in_ch, skip_ch, decoder_channels[i], norm))
            in_ch = decoder_channels[i]
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(in_ch, 1, 3, padding=1)
        self.gate = GateHead(channels[-1])
        self.aux_weight = aux_weight
        self.aux4 = nn.Conv2d(decoder_channels[2], 1, 1) if aux_weight > 0 else None

        if verbose:
            n = sum(p.numel() for p in self.parameters()) / 1e6
            print(f"  {encoder}: страйды {self.reductions}, каналы {channels}, {n:.1f}M")

    def forward(self, x, fmap=None):
        size = x.shape[-2:]
        feats = list(self.encoder(x))

        if self.use_fmap:
            if fmap is None:
                fmap = x.new_zeros(x.shape[0], self.fmap_used, size[0] // 8, size[1] // 8)
            branch = self.branch(fmap.to(x.dtype))
            for r in self.FUSE_AT:
                index = self.reductions.index(r)
                feats[index] = self.fuse[str(r)](feats[index], branch[r])

        by_red = dict(zip(self.reductions, feats))
        out = feats[-1]
        for block, r in zip(self.blocks, self.skip_reductions):
            out = block(out, by_red.get(r) if r else None)

        logits = self.head(out)
        if logits.shape[-2:] != size:
            logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return {"logits": logits, "cls_logits": self.gate(feats[-1])}


# ---------------------------------------------------------------------------
# сборка из прогона
# ---------------------------------------------------------------------------

WEIGHT_KEYS = ("ema", "model")


def fmap_keep_indices(drop) -> list[int]:
    """Индексы каналов `aic.forensic.CHANNELS`, доходящих до сети."""
    drop = set(drop or ())
    return [i for i, name in enumerate(CHANNELS) if name not in drop]


def build_model(cfg: dict, *, pretrained: bool = False, verbose: bool = False) -> Segmenter:
    return Segmenter(
        encoder=cfg["encoder"],
        decoder_channels=tuple(cfg["decoder_channels"]),
        norm=cfg.get("norm", "batch"),
        pretrained=pretrained,
        use_fmap=cfg.get("fmap", "none") != "none",
        fmap_ch=tuple(cfg.get("fmap_ch", (64, 96, 128))),
        fmap_used=len(fmap_keep_indices(cfg.get("fmap_drop"))),
        aux_weight=cfg.get("aux_weight", 0.0),
        verbose=verbose,
    )


def load_run_model(run, checkpoint: str = "best", use_ema: bool = True, device=None):
    """`(model, cfg, ключ_весов)`. Веса строго, как в error_analysis.

    Голова `aux4` живёт только в обучении, но её веса в чекпоинте есть, поэтому
    модель собирается с тем же `aux_weight` — иначе `strict=True` справедливо
    ругнётся на лишний ключ.
    """
    name = checkpoint if str(checkpoint).endswith(".pt") else f"{checkpoint}.pt"
    state = run.load_state(name, map_location="cpu")
    cfg = run.snapshot or state.get("cfg") or {}
    if "encoder" not in cfg:
        raise ValueError(f"{run.dir.name}: в config.yaml нет ключа encoder — "
                         "это не прогон серии s3")

    model = build_model(cfg)
    keys = WEIGHT_KEYS if use_ema else ("model",)
    key = next((k for k in keys if state.get(k)), None)
    if key is None:
        raise ValueError(f"{run.dir.name}: нет весов под ключами {keys} (есть {list(state)})")
    model.load_state_dict(state[key], strict=True)
    if device is not None:
        model.to(device, memory_format=torch.channels_last)
    return model.eval(), cfg, key


# ---------------------------------------------------------------------------
# данные инференса
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """Кадр, форензик-карта по РОДНОМУ разрешению и, если есть, GT.

    Повторяет ветку `train=False` из `s3_data.SegDataset` шаг в шаг:
    таблица квантования читается из файла, карта считается по полному кадру,
    кадр обрезается до кратности 8, и только потом обе стороны ресайзятся.
    Любое отклонение здесь — это тихое расхождение распределений между
    обучением и сабмитом.
    """

    def __init__(self, records, size: int, fmap_mode: str = "native", fmap_keep=None):
        #: (stem, абсолютный путь к кадру, абсолютный путь к GT или None)
        self.records = list(records)
        self.size = int(size)
        self.fmap_mode = str(fmap_mode)
        self.fmap_keep = None if fmap_keep is None else list(fmap_keep)

    def __len__(self) -> int:
        return len(self.records)

    def _select(self, maps: np.ndarray) -> np.ndarray:
        return maps if self.fmap_keep is None else maps[self.fmap_keep]

    def __getitem__(self, i: int) -> dict:
        stem, img_path, gt_path = self.records[i]
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return {"ok": False, "index": i, "stem": stem,
                    "image": torch.zeros(3, self.size, self.size),
                    "fmap": torch.zeros(len(self.fmap_keep or CHANNELS),
                                        self.size // 8, self.size // 8),
                    "orig_h": 1, "orig_w": 1, "crop_h": 1, "crop_w": 1,
                    "gt": np.zeros((1, 1), dtype=bool)}

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qtable = luma_qtable(str(img_path))
        maps = forensic_maps(rgb, qtable) if self.fmap_mode == "native" else None

        ch, cw = align8(h), align8(w)
        if min(ch, cw) < 8:
            raise ValueError(f"кадр {h}x{w} меньше блока 8x8: {img_path}")
        rgb = rgb[:ch, :cw]
        if maps is not None:
            maps = crop_maps(maps, 0, 0, ch, cw)

        image = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        image = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        gt = np.zeros((h, w), dtype=bool)
        if gt_path is not None:
            raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                if raw.shape[:2] != (h, w):
                    raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
                gt = raw >= 128

        out = {
            "ok": True, "index": i, "stem": stem,
            "image": torch.from_numpy(image).permute(2, 0, 1).contiguous(),
            "orig_h": h, "orig_w": w, "crop_h": ch, "crop_w": cw, "gt": gt,
        }
        if self.fmap_mode == "native":
            out["fmap"] = torch.from_numpy(self._select(resize_maps(maps, self.size)))
        elif self.fmap_mode == "post":
            out["fmap"] = torch.from_numpy(self._select(forensic_maps(image, None)))
        return out


def collate(batch: list[dict]) -> dict:
    """Тензоры складываются, GT остаётся списком: у каждого кадра свой размер."""
    out = {"image": torch.stack([item["image"] for item in batch])}
    if "fmap" in batch[0]:
        out["fmap"] = torch.stack([item["fmap"] for item in batch])
    for key in ("ok", "index", "stem", "orig_h", "orig_w", "crop_h", "crop_w", "gt"):
        out[key] = [item[key] for item in batch]
    return out


def to_original(prob: np.ndarray, orig_h: int, orig_w: int,
                crop_h: int, crop_w: int) -> np.ndarray:
    """Карта с модельной сетки обратно в исходный кадр.

    Модель видела кадр, обрезанный до кратности 8, поэтому карта растягивается
    именно в `(crop_h, crop_w)`. Хвост шириной до семи пикселей в сетку не
    попадал вовсе и достраивается ПОВТОРЕНИЕМ КРАЯ, а не нулями: ноль там —
    гарантированно неверная полоса по краю кадра, и на масках, доходящих до
    границы, она стоила бы Dice ни за что.
    """
    resized = cv2.resize(prob.astype(np.float32), (crop_w, crop_h),
                         interpolation=cv2.INTER_LINEAR)
    if (crop_h, crop_w) == (orig_h, orig_w):
        return resized
    return np.pad(resized, ((0, orig_h - crop_h), (0, orig_w - crop_w)), mode="edge")


@torch.no_grad()
def iter_predictions(model, loader, device, *, amp: str = "fp16"):
    """`(index, stem, prob в исходном разрешении, cls_prob, gt)` по кадру."""
    use_amp = amp != "off" and device.type == "cuda"
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        fmap = batch["fmap"].to(device, non_blocking=True) if "fmap" in batch else None
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
            out = model(images, fmap)
        probs = torch.sigmoid(out["logits"].float()).cpu().numpy()
        cls = torch.sigmoid(out["cls_logits"].float()).reshape(-1).cpu().numpy()

        for i in range(len(batch["stem"])):
            if not batch["ok"][i]:
                yield batch["index"][i], batch["stem"][i], None, 0.0, batch["gt"][i]
                continue
            yield (
                batch["index"][i],
                batch["stem"][i],
                to_original(probs[i, 0], batch["orig_h"][i], batch["orig_w"][i],
                            batch["crop_h"][i], batch["crop_w"][i]),
                float(cls[i]),
                batch["gt"][i],
            )


def run_records(ws, frames) -> list[tuple[str, Path, Path | None]]:
    """Строки таблицы кадров -> записи для `InferenceDataset`."""
    records = []
    for row in frames.itertuples():
        gt = getattr(row, "gt_path", None)
        records.append((
            str(row.stem),
            ws.resolve(str(row.chng_path)),
            ws.resolve(str(gt)) if isinstance(gt, str) else None,
        ))
    return records
