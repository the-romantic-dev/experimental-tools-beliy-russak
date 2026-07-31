"""Низкоуровневый (forensic) поток признаков — второй вход модели.

Зачем. Диагностика OOF baseline показала: кадры, которые модель промахивает
полностью (8.7% позитивов, Dice<0.05), имеют oracle-Dice 0.157 и recall внутри
GT 0.16 при пороге 0.05. То есть в вероятностной карте сигнала нет вообще —
это не проблема порога, а проблема признаков. Семантический ImageNet-энкодер
видит «правдоподобный объект» и не видит следов вставки.

Что здесь есть:

* `SRMConv`   — три фиксированных высокочастотных фильтра из SRM (Fridrich).
  Ядра нулевой суммы: гасят содержимое сцены, оставляют шумовой остаток.
* `BayarConv` — обучаемый аналог с тем же ограничением (центр -1, остальное
  суммируется в 1), из Bayar & Stamm. Учится под конкретный тип артефактов.
* `ResidualExtractor` — RGB -> K каналов остатка (+BatchNorm, иначе масштаб
  остатка на два порядка меньше входа и стем его просто не заметит).
* `NoiseBranch` / `GatedDualEncoder` — вторая ветка и гейт-фьюз с пирамидой
  основного энкодера.

Две ступени интеграции, обе управляются конфигом (`model.fuse`):

    fuse: input   RGB и остаток склеиваются по каналам, энкодер один
                  (in_channels = 3 + K). Нулевая цена по памяти.
    fuse: gate    остаток идёт в отдельную лёгкую ветку, её фичи вливаются
                  в пирамиду основного энкодера через обучаемый гейт:
                      f <- f + sigmoid(gate([f, a])) * proj(a)
                  Гейт инициализирован закрытым (bias -2), поэтому на старте
                  модель эквивалентна RGB-baseline и не ломается на первых шагах.

Остаток считается из ДЕнормализованного RGB: фильтры линейные, и приводить
вход к [0,1] дешевле, чем потом гадать, что делает с ними ImageNet-нормировка.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import IMAGENET_MEAN, IMAGENET_STD

# Три ядра SRM в той форме, в какой их используют работы по image forensics
# (RGB-N, ManTra-Net): каждое нулевой суммы, поэтому постоянная составляющая
# и любой сдвиг яркости из отклика уходят.
_SRM_KERNELS = (
    (
        ((0, 0, 0, 0, 0),
         (0, -1, 2, -1, 0),
         (0, 2, -4, 2, 0),
         (0, -1, 2, -1, 0),
         (0, 0, 0, 0, 0)),
        4.0,
    ),
    (
        ((-1, 2, -2, 2, -1),
         (2, -6, 8, -6, 2),
         (-2, 8, -12, 8, -2),
         (2, -6, 8, -6, 2),
         (-1, 2, -2, 2, -1)),
        12.0,
    ),
    (
        ((0, 0, 0, 0, 0),
         (0, 0, 0, 0, 0),
         (0, 1, -2, 1, 0),
         (0, 0, 0, 0, 0),
         (0, 0, 0, 0, 0)),
        2.0,
    ),
)


class SRMConv(nn.Module):
    """Три фиксированных фильтра, каждый применяется ко всем трём каналам RGB."""

    def __init__(self) -> None:
        super().__init__()
        weight = torch.zeros(3, 3, 5, 5)
        for i, (kernel, norm) in enumerate(_SRM_KERNELS):
            k = torch.tensor(kernel, dtype=torch.float32) / norm
            weight[i] = k.unsqueeze(0).expand(3, 5, 5)
        self.register_buffer("weight", weight, persistent=False)

    @property
    def out_channels(self) -> int:
        return 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight.to(x.dtype), padding=2)


class BayarConv(nn.Module):
    """Обучаемый высокочастотный фильтр со связью «центр -1, остальное в сумме 1».

    Ограничение применяется на каждом forward, а не регуляризацией: так фильтр
    физически не может выродиться в обычную свёртку и начать выучивать контент.
    """

    def __init__(self, out_channels: int = 3, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        centre = kernel_size // 2
        mask = torch.ones(1, 1, kernel_size, kernel_size)
        mask[..., centre, centre] = 0.0
        self.register_buffer("mask", mask, persistent=False)

        # Старт из почти равномерного ядра с суммой ~1, а не из N(0, sigma):
        # нормировка ниже делит на сумму весов, и у случайной инициализации
        # эта сумма запросто оказывается около нуля — ядро тогда улетает
        # на несколько порядков и заваливает обучение на первом же шаге.
        n_off_centre = kernel_size * kernel_size - 1
        weight = torch.full((out_channels, 3, kernel_size, kernel_size), 1.0 / n_off_centre)
        weight = weight + torch.randn_like(weight) * (0.1 / n_off_centre)
        self.weight = nn.Parameter(weight * mask)

    @property
    def out_channels(self) -> int:
        return self.weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centre = self.kernel_size // 2
        weight = self.weight * self.mask
        # знак знаменателя сохраняем: сумма должна стать ровно +1, а не -1
        total = weight.sum(dim=(2, 3), keepdim=True)
        floor = torch.full_like(total, 1e-3)
        total = torch.where(total.abs() < 1e-3, torch.copysign(floor, total), total)
        weight = weight / total

        weight = weight.clone()
        weight[..., centre, centre] = -1.0
        return F.conv2d(x, weight.to(x.dtype), padding=centre)


class ResidualExtractor(nn.Module):
    """Нормализованный RGB -> K каналов низкоуровневого остатка.

    `kinds` — какие фильтры включить: srm, bayar или оба.
    BatchNorm на выходе обязателен: сырой отклик SRM живёт в районе 1e-2, и без
    приведения масштаба стем предобученного энкодера его игнорирует.
    """

    def __init__(self, kinds: tuple[str, ...] = ("srm", "bayar")) -> None:
        super().__init__()
        if not kinds:
            raise ValueError("нужен хотя бы один вид фильтра: srm | bayar")

        filters: list[nn.Module] = []
        for kind in kinds:
            if kind == "srm":
                filters.append(SRMConv())
            elif kind == "bayar":
                filters.append(BayarConv())
            else:
                raise ValueError(f"неизвестный фильтр остатка: {kind}; есть srm|bayar")
        self.filters = nn.ModuleList(filters)

        self.out_channels = sum(f.out_channels for f in filters)
        self.norm = nn.BatchNorm2d(self.out_channels)

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x * self.std.to(x.dtype) + self.mean.to(x.dtype)
        residual = torch.cat([f(rgb) for f in self.filters], dim=1)
        return self.norm(residual.float()).to(x.dtype)


class InputFusion(nn.Module):
    """RGB + остаток по каналам. Модель под ним строится с in_channels = 3 + K."""

    def __init__(self, kinds: tuple[str, ...] = ("srm", "bayar")) -> None:
        super().__init__()
        self.extractor = ResidualExtractor(kinds)
        self.out_channels = 3 + self.extractor.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.extractor(x)], dim=1)


def _conv_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class NoiseBranch(nn.Module):
    """Лёгкий энкодер по остатку: пирамида фич на тех же шагах, что у основного.

    Специально маленький. Задача ветки — не выучить сцену заново, а донести до
    декодера «здесь статистика шума рвётся», и на 8 ГБ второй полноразмерный
    энкодер рядом с 768 px просто не поместится.
    """

    def __init__(
        self,
        strides: tuple[int, ...],
        kinds: tuple[str, ...] = ("srm", "bayar"),
        width: int = 32,
    ) -> None:
        super().__init__()
        self.extractor = ResidualExtractor(kinds)
        self.strides = strides

        blocks: list[nn.Module] = []
        channels: list[int] = []
        in_ch, current = self.extractor.out_channels, 1
        for i, stride in enumerate(strides):
            factor = stride // current
            if factor < 1:
                raise ValueError(f"шаги пирамиды должны расти: {strides}")
            out_ch = min(width * 2 ** i, 256)
            block: list[nn.Module] = []
            while factor > 1:  # доводим до нужного шага серией stride-2 блоков
                block.append(_conv_block(in_ch, out_ch, stride=2))
                in_ch, factor = out_ch, factor // 2
            block.append(_conv_block(in_ch, out_ch, stride=1))
            blocks.append(nn.Sequential(*block))
            in_ch, current = out_ch, stride
            channels.append(out_ch)

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features, out = self.extractor(x), []
        for block in self.blocks:
            features = block(features)
            out.append(features)
        return out


class GatedDualEncoder(nn.Module):
    """Основной энкодер + ветка по остатку, слитые обучаемым гейтом.

    Контракт наружу тот же, что у энкодера smp (`out_channels` + список фич),
    поэтому декодер и головы остаются нетронутыми — подменяется только
    `model.encoder`, и трюк работает с любой архитектурой из smp.
    """

    def __init__(self, encoder: nn.Module, branch: NoiseBranch, fuse_from: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        self.aux_stream = branch
        self.out_channels = list(encoder.out_channels)
        self.output_stride = getattr(encoder, "output_stride", 32)

        # индексы стадий основного энкодера, которые реально фьюзим:
        # у части энкодеров ранние стадии пустые (0 каналов) или это сам вход
        self.fuse_index = [
            i for i, ch in enumerate(self.out_channels) if ch > 0 and i >= fuse_from
        ]
        projections, gates = [], []
        for stage, index in enumerate(self.fuse_index):
            main_ch, aux_ch = self.out_channels[index], branch.out_channels[stage]
            # проекция занулена, гейт почти закрыт => на шаге 0 вклад ветки РОВНО
            # нулевой и модель побитово равна RGB-baseline. Градиент по проекции
            # при этом ненулевой (он равен gate * aux), так что ветка учится
            projection = nn.Conv2d(aux_ch, main_ch, 1)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            projections.append(projection)

            gate = nn.Conv2d(main_ch * 2, main_ch, 1)
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -2.0)
            gates.append(gate)
        self.projections = nn.ModuleList(projections)
        self.gates = nn.ModuleList(gates)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = list(self.encoder(x))
        aux_features = self.aux_stream(x)

        for stage, index in enumerate(self.fuse_index):
            main, aux = features[index], aux_features[stage]
            if aux.shape[-2:] != main.shape[-2:]:
                aux = F.interpolate(aux, size=main.shape[-2:], mode="bilinear", align_corners=False)
            aux = self.projections[stage](aux)
            gate = torch.sigmoid(self.gates[stage](torch.cat([main, aux], dim=1)))
            features[index] = main + gate * aux
        return features


@torch.no_grad()
def encoder_strides(encoder: nn.Module, probe: int = 128) -> list[int]:
    """Шаги пирамиды энкодера — снимаются пробным прогоном, а не угадываются.

    У разных семейств (ConvNeXt, MiT, ResNet) число стадий и наличие пустых
    уровней различаются, и захардкоженный список [1,2,4,8,16,32] врёт.
    """
    was_training = encoder.training
    encoder.eval()
    in_channels = encoder.out_channels[0] if encoder.out_channels else 3
    features = encoder(torch.zeros(1, in_channels, probe, probe))
    encoder.train(was_training)
    return [probe // f.shape[-1] for f in features]
