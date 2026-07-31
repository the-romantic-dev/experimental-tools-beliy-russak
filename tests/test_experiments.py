"""Механика экспериментов e1-e3: разрешение/паддинг, профили по площади, потоки.

Всё на синтетике, без датасета и без GPU. Каждый тест проверяет ровно то
свойство, на котором держится вывод соответствующего прогона: если сравнение
e0 и e1 портит неверный учёт паддинга, а e3 стартует не из baseline, то и
числа в `board` сравнивать бессмысленно.
"""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experimental_tools_beliy_russak.datasets import build_balanced_sampler, valid_region
from experimental_tools_beliy_russak.engine import SCRATCH_MARKERS, _histograms, build_optimizer
from experimental_tools_beliy_russak.inference import _to_original
from experimental_tools_beliy_russak.losses import build_loss, soft_dice_loss
from experimental_tools_beliy_russak.models import build_model
from experimental_tools_beliy_russak.streams import BayarConv, InputFusion, ResidualExtractor, SRMConv, encoder_strides

# --- e1: паддинг и полезная область -----------------------------------------


def test_srm_kernels_are_zero_sum():
    """Ядра нулевой суммы гасят контент сцены: постоянная яркость в отклик
    не проходит, остаётся только высокочастотный остаток."""
    weight = SRMConv().weight
    assert weight.shape == (3, 3, 5, 5)
    assert weight.sum(dim=(2, 3)).abs().max() < 1e-6


def test_srm_ignores_flat_image():
    srm = SRMConv()
    flat = torch.full((1, 3, 16, 16), 0.7)
    assert srm(flat)[..., 2:-2, 2:-2].abs().max() < 1e-5


def _bayar_effective_kernel(bayar: BayarConv) -> torch.Tensor:
    """Ядро в том виде, в каком оно реально уходит в свёртку."""
    centre = bayar.kernel_size // 2
    weight = (bayar.weight * bayar.mask).detach()
    total = weight.sum(dim=(2, 3), keepdim=True)
    floor = torch.full_like(total, 1e-3)
    total = torch.where(total.abs() < 1e-3, torch.copysign(floor, total), total)
    weight = weight / total
    weight[..., centre, centre] = -1.0
    return weight


def test_bayar_constraint_holds():
    """Центр -1, сумма остальных 1 — иначе фильтр вырождается в обычную
    свёртку и начинает выучивать содержимое вместо следов правки."""
    bayar = BayarConv(out_channels=3)
    weight = _bayar_effective_kernel(bayar)

    centre = bayar.kernel_size // 2
    assert float(weight[..., centre, centre].abs().max()) == pytest.approx(1.0)
    off_centre = (weight * bayar.mask).sum(dim=(2, 3))
    assert float(off_centre.min()) == pytest.approx(1.0, abs=1e-4)
    assert float(off_centre.max()) == pytest.approx(1.0, abs=1e-4)
    # всё ядро в сумме нулевое => постоянная яркость в отклик не проходит
    assert float(weight.sum(dim=(2, 3)).abs().max()) < 1e-4


def test_bayar_survives_degenerate_weights():
    """Регресс: нормировка делит на сумму весов, и у суммы около нуля ядро
    улетало на порядки. Инициализация даёт сумму ~1, но обучение может увести
    её куда угодно — деление обязано оставаться ограниченным."""
    bayar = BayarConv(out_channels=3)
    with torch.no_grad():
        bayar.weight.zero_()  # худший случай: сумма ровно 0

    out = bayar(torch.randn(1, 3, 16, 16))
    assert torch.isfinite(out).all()
    assert float(_bayar_effective_kernel(bayar).abs().max()) <= 1.0 + 1e-6


def test_residual_extractor_channel_counts():
    assert ResidualExtractor(("srm",)).out_channels == 3
    assert ResidualExtractor(("srm", "bayar")).out_channels == 6
    assert InputFusion(("srm", "bayar")).out_channels == 9

    out = ResidualExtractor(("srm", "bayar"))(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 6, 32, 32)


def test_residual_extractor_rejects_unknown_filter():
    with pytest.raises(ValueError, match="неизвестный фильтр"):
        ResidualExtractor(("wavelet",))


# --- e3: потоки и гейт ------------------------------------------------------


@pytest.fixture(scope="module")
def gated_model():
    return build_model({
        "backend": "smp", "arch": "unet", "encoder": "tu-resnet18",
        "encoder_weights": None, "cls_head": "aux",
        "stream": "srm_bayar", "fuse": "gate", "stream_width": 16,
    }).eval()


def test_gate_starts_closed_and_is_identical_to_rgb_baseline(gated_model):
    """Проекция занулена, гейт закрыт => на шаге 0 двухпоточная модель РОВНО
    равна однопоточной. Иначе e3 стартовал бы из другой точки и разница с e0
    мерила бы не поток признаков, а неудачную инициализацию."""
    dual = gated_model.core.encoder
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        fused = dual(x)
        plain = dual.encoder(x)

    for a, b in zip(fused, plain):
        assert torch.equal(a, b)


def test_gate_opens_under_gradient(gated_model):
    """Занулённая проекция не должна означать мёртвую ветку: градиент по ней
    равен gate * aux и не нулевой."""
    model = build_model({
        "backend": "smp", "arch": "unet", "encoder": "tu-resnet18",
        "encoder_weights": None, "cls_head": "aux",
        "stream": "srm_bayar", "fuse": "gate", "stream_width": 16,
    })
    out = model(torch.randn(2, 3, 64, 64))
    out["logits"].square().mean().backward()

    projection = model.core.encoder.projections[0].weight
    assert projection.grad is not None
    assert projection.grad.abs().sum() > 0


def test_dual_encoder_keeps_decoder_contract(gated_model):
    """Каналы пирамиды не меняются — декодер и головы остаются нетронутыми."""
    dual = gated_model.core.encoder
    assert list(dual.out_channels) == list(dual.encoder.out_channels)
    assert dual.output_stride == getattr(dual.encoder, "output_stride", 32)


def test_input_fusion_widens_the_stem():
    model = build_model({
        "backend": "smp", "arch": "unet", "encoder": "tu-resnet18",
        "encoder_weights": None, "cls_head": "aux",
        "stream": "srm_bayar", "fuse": "input",
    })
    assert model.core.encoder.out_channels[0] == 9
    out = model(torch.randn(2, 3, 64, 64))
    assert out["logits"].shape == (2, 1, 64, 64)


def test_unknown_stream_and_fuse_are_rejected():
    base = {"backend": "smp", "arch": "unet", "encoder": "tu-resnet18",
            "encoder_weights": None, "cls_head": "aux"}
    with pytest.raises(ValueError, match="неизвестный поток"):
        build_model({**base, "stream": "wavelet"})
    with pytest.raises(ValueError, match="неизвестный способ фьюза"):
        build_model({**base, "stream": "srm", "fuse": "concat_later"})


def test_encoder_strides_are_probed_not_guessed():
    model = build_model({
        "backend": "smp", "arch": "unet", "encoder": "tu-convnext_tiny",
        "encoder_weights": None, "cls_head": "aux",
    })
    assert encoder_strides(model.core.encoder) == [1, 2, 4, 8, 16, 32]


def test_scratch_branch_gets_decoder_lr(gated_model):
    """Ветка лежит внутри encoder по имени, но учится с нуля: под encoder_lr
    она обучалась бы втрое медленнее декодера просто из-за пути параметра."""
    optimizer = build_optimizer(gated_model, {"lr": 3e-4, "encoder_lr": 1e-4})
    # групп с одинаковым lr две (decay и nodecay) — их надо объединять, а не
    # затирать одну другой
    by_lr: dict[float, set[int]] = {}
    for group in optimizer.param_groups:
        by_lr.setdefault(group["lr"], set()).update(id(p) for p in group["params"])

    scratch = [
        p for name, p in gated_model.named_parameters()
        if any(marker in name for marker in SCRATCH_MARKERS)
    ]
    assert scratch, "в двухпоточной модели должны быть параметры, обучаемые с нуля"
    assert all(id(p) in by_lr[3e-4] for p in scratch)

    # предобученный энкодер при этом остаётся на пониженном encoder_lr
    pretrained = dict(gated_model.named_parameters())["core.encoder.encoder.model.conv1.weight"]
    assert id(pretrained) in by_lr[1e-4]


# --- e1: полезная область и метрика ----------------------------------------


def test_valid_region_matches_inference_geometry():
    """valid_region и inference._to_original описывают одну и ту же геометрию;
    разъедься они — площадь в метрике перестанет соответствовать сабмиту."""
    orig_h, orig_w, size = 683, 1024, 256
    valid_h, valid_w = valid_region(size, size, orig_h, orig_w, pad_mode=True)

    probe = torch.zeros(1, size, size)
    probe[:, :valid_h, :valid_w] = 1.0
    restored = _to_original(probe, orig_h, orig_w, "pad")
    assert restored.mean() == pytest.approx(1.0, abs=1e-3)
    assert (valid_h, valid_w) == (171, 256)


def test_valid_region_is_full_grid_in_resize_mode():
    assert valid_region(256, 256, 683, 1024, pad_mode=False) == (256, 256)


def test_histograms_exclude_padding_from_area():
    """Правило FPR смотрит на долю площади кадра. Если считать её от холста
    вместе с паддингом, 1% срабатывает позже, чем на настоящем сабмите."""
    probs = torch.zeros(1, 1, 100, 100)
    probs[..., :20, :50] = 1.0          # 1000 пикселей
    masks = torch.zeros(1, 1, 100, 100)
    valid_h = torch.tensor([50])         # полезная область 50x50 = 2500
    valid_w = torch.tensor([50])

    _, _, _, n_pixels = _histograms(probs, masks, 64, valid_h, valid_w)
    assert int(n_pixels[0]) == 2500

    hist_all, _, _, _ = _histograms(probs, masks, 64, valid_h, valid_w)
    predicted = int(hist_all[0, 1:].sum())   # всё, что выше нулевого бина
    assert predicted == 20 * 50              # предсказание целиком внутри области

    # то же предсказание, но сдвинутое в паддинг, площади давать не должно
    shifted = torch.zeros(1, 1, 100, 100)
    shifted[..., 60:80, :50] = 1.0
    hist_shifted, _, _, _ = _histograms(shifted, masks, 64, valid_h, valid_w)
    assert int(hist_shifted[0, 1:].sum()) == 0


def test_histograms_without_valid_region_use_whole_grid():
    probs = torch.zeros(2, 1, 10, 10)
    _, _, _, n_pixels = _histograms(probs, torch.zeros(2, 1, 10, 10), 32)
    assert list(n_pixels) == [100, 100]


# --- e2: профили лосса по площади ------------------------------------------


def _batch(area: float, size: int = 8) -> dict:
    mask = torch.zeros(1, 1, size, size)
    n = int(round(area * size * size))
    mask.view(-1)[:n] = 1.0
    return {"mask": mask, "label": torch.tensor([[float(n > 0)]]),
            "area": torch.tensor([[area]])}


def test_loss_without_area_profile_is_bit_identical_to_plain_sum():
    """Режим по умолчанию не должен ничего менять: иначе опорный прогон нельзя
    сопоставлять ни с одним прошлым."""
    criterion = build_loss({"seg": {"bce": 1.0, "dice": 1.0}, "cls_weight": 0.0})
    logits = torch.randn(4, 1, 8, 8)
    targets = (torch.rand(4, 1, 8, 8) > 0.5).float()

    total, _ = criterion({"logits": logits}, {"mask": targets})
    expected = F.binary_cross_entropy_with_logits(logits, targets) + soft_dice_loss(logits, targets)
    assert float(total) == pytest.approx(float(expected), abs=1e-6)


def test_small_masks_switch_to_the_small_profile():
    cfg = {"seg": {"bce": 1.0}, "cls_weight": 0.0,
           "area": {"threshold": 0.06, "small_seg": {"tversky": 1.0}, "small_weight": 1.0}}
    criterion = build_loss(cfg)
    logits = torch.randn(1, 1, 8, 8)

    small, _ = criterion({"logits": logits}, _batch(0.02))
    large, _ = criterion({"logits": logits}, _batch(0.30))

    # у мелкого кадра лосс считается по Tversky, у крупного — по BCE;
    # совпасть они могут только случайно
    assert float(small) != pytest.approx(float(large), abs=1e-4)


def test_negatives_stay_on_the_default_profile():
    """Площадь 0 — это чистый кадр, а не «мелкая маска». Перекос в сторону
    полноты на негативах разгонял бы ровно тот FPR_neg, который мы бережём."""
    cfg = {"seg": {"bce": 1.0}, "cls_weight": 0.0,
           "area": {"threshold": 0.06, "small_seg": {"tversky": 5.0}, "small_weight": 10.0}}
    criterion = build_loss(cfg)
    logits = torch.randn(1, 1, 8, 8)

    with_profile, _ = criterion({"logits": logits}, _batch(0.0))
    plain = build_loss({"seg": {"bce": 1.0}, "cls_weight": 0.0})
    without_profile, _ = plain({"logits": logits}, _batch(0.0))
    assert float(with_profile) == pytest.approx(float(without_profile), abs=1e-6)


def test_small_weight_does_not_inflate_loss_scale():
    """Веса нормируются на среднее по батчу: small_weight меняет БАЛАНС между
    корзинами, но не масштаб лосса, иначе он подменял бы собой lr."""
    cfg = {"seg": {"bce": 1.0}, "cls_weight": 0.0,
           "area": {"threshold": 0.06, "small_seg": {"bce": 1.0}, "small_weight": 7.0}}
    criterion = build_loss(cfg)
    logits = torch.randn(2, 1, 8, 8)
    batch = {"mask": torch.zeros(2, 1, 8, 8), "label": torch.zeros(2, 1),
             "area": torch.tensor([[0.01], [0.02]])}   # оба кадра мелкие

    total, stats = criterion({"logits": logits}, batch)
    assert stats["small_frac"] == pytest.approx(1.0)
    assert float(total) == pytest.approx(float(stats["bce"]), abs=1e-6)


def test_loss_rejects_unknown_component_in_small_profile():
    """Опечатка в профиле по площади должна вылезти при сборке, а не на первом шаге."""
    with pytest.raises(ValueError) as error:
        build_loss({"seg": {"bce": 1.0}, "area": {"small_seg": {"lovasz": 1.0}}})

    message = str(error.value)
    assert "lovasz" in message
    assert "bce, dice, focal, tversky" in message   # что вообще можно писать
    assert "register_loss" in message               # и как добавить своё


# --- e2: сэмплер по площади -------------------------------------------------


class AreaDataset:
    """10 негативов, 20 мелких позитивов, 70 крупных."""

    is_negative = np.array([True] * 10 + [False] * 90)
    area = np.array([0.0] * 10 + [0.02] * 20 + [0.30] * 70, dtype=np.float32)

    def __len__(self):
        return 100


def test_area_sampler_hits_requested_shares():
    dataset = AreaDataset()
    sampler = build_balanced_sampler(
        dataset, negative_fraction=0.25, num_samples=40000,
        small_area_fraction=0.4, small_area_threshold=0.06,
    )
    drawn = np.array(list(sampler))
    is_neg = dataset.is_negative[drawn]
    is_small = (~is_neg) & (dataset.area[drawn] < 0.06)

    assert is_neg.mean() == pytest.approx(0.25, abs=0.02)
    assert is_small.mean() == pytest.approx(0.75 * 0.4, abs=0.02)


# --- подвыборка валидации ---------------------------------------------------


@pytest.fixture
def val_frame():
    import pandas as pd

    return pd.DataFrame({"is_negative": [True] * 639 + [False] * 20000})


def test_val_subset_keeps_every_negative(val_frame):
    """FPR_neg считается по негативам, их всего ~3%. Прореживать надо позитивы:
    при обычном val_frac=0.25 негативов осталось бы полторы сотни и одна ложная
    тревога двигала бы метрику сильнее, чем эффект проверяемой гипотезы."""
    from experimental_tools_beliy_russak.train import _subset

    out = _subset(val_frame, 0.25, None, seed=0, keep_negatives=True)
    assert int(out["is_negative"].sum()) == 639
    assert int((~out["is_negative"]).sum()) == pytest.approx(5000, rel=0.02)


def test_val_subset_respects_a_hard_limit(val_frame):
    """Регресс: при val_limit меньше числа негативов выборка вырождалась
    в одни негативы, и Dice_pos обнулялся при полностью живой модели."""
    from experimental_tools_beliy_russak.train import _subset

    out = _subset(val_frame, None, 120, seed=0, keep_negatives=True)
    assert len(out) == 120
    assert 0 < int(out["is_negative"].sum()) < 120
    assert int((~out["is_negative"]).sum()) > 0


def test_val_subset_without_flag_is_plain_sampling(val_frame):
    from experimental_tools_beliy_russak.train import _subset

    out = _subset(val_frame, 0.25, None, seed=0, keep_negatives=False)
    assert len(out) == pytest.approx(len(val_frame) * 0.25, rel=0.01)


def test_area_sampler_falls_back_when_a_bucket_is_empty():
    """Все позитивы крупные — сэмплер обязан остаться рабочим, а не обнулить
    веса и уронить WeightedRandomSampler."""
    class NoSmall(AreaDataset):
        area = np.array([0.0] * 10 + [0.30] * 90, dtype=np.float32)

    sampler = build_balanced_sampler(
        NoSmall(), negative_fraction=0.25, num_samples=2000, small_area_fraction=0.4,
    )
    drawn = np.array(list(sampler))
    assert NoSmall.is_negative[drawn].mean() == pytest.approx(0.25, abs=0.05)
