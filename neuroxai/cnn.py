"""EfficientNet-B0: 2.5D yamalarda ince ayar, 1280-boyutlu embedding (GAP) ve Grad-CAM (proje formu 4.6.1-4.6.2, 4.8).

Her vizit 3 koronal dilimden oluşur; her dilim 3 kanallı (orijinal, Δ, kenar) bir görüntü olarak ağa girer.
Vizit embedding'i = üç dilimin GAP embedding'lerinin ortalaması; vizit olasılığı = üç dilim logit ortalaması.
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))


def girdi_olcekle(yamalar: np.ndarray) -> np.ndarray:
    """(..., 3 kanal, H, W): orijinal z ~[-4,6]/3, Δ (σ biriminde ±3)/3, kenar [0,1] -> [-1,1]."""
    x = np.array(yamalar, dtype=np.float32, copy=True)
    x[..., 0, :, :] = np.clip(x[..., 0, :, :], -4, 6) / 3
    x[..., 1, :, :] = x[..., 1, :, :] / 3
    x[..., 2, :, :] = x[..., 2, :, :] * 2 - 1
    return x


def model_olustur(n_sinif: int, onegitimli: bool = True) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if onegitimli else None)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, n_sinif))
    return model


class DilimVeri(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, vizitler: np.ndarray, artirma: bool):
        self.X, self.y, self.artirma = X, y, artirma
        self.ogeler = [(v, s) for v in vizitler for s in range(X.shape[1])]

    def __len__(self):
        return len(self.ogeler)

    def __getitem__(self, i):
        v, s = self.ogeler[i]
        x = torch.from_numpy(self.X[v, s])
        if self.artirma:
            if torch.rand(1) < 0.5:  # sol-sağ yansıma (koronal dilimde x ekseni)
                x = torch.flip(x, dims=[2])
            dy, dx = torch.randint(-6, 7, (2,)).tolist()
            x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
            x = x * (1 + 0.1 * (torch.rand(3, 1, 1) - 0.5))
        return x, int(self.y[v])


@torch.no_grad()
def vizit_ciktilari(model: nn.Module, X: np.ndarray, vizitler: np.ndarray, toplu: int = 96):
    """Vizit başına (embedding 1280, logit) — 3 dilimin ortalaması."""
    model.eval()
    n_dilim = X.shape[1]
    gommeler, logitler = [], []
    for b in range(0, len(vizitler), max(1, toplu // n_dilim)):
        parca = vizitler[b:b + max(1, toplu // n_dilim)]
        x = torch.from_numpy(X[parca].reshape(-1, *X.shape[2:]))
        oz = torch.flatten(model.avgpool(model.features(x)), 1)
        lg = model.classifier(oz)
        gommeler.append(oz.view(len(parca), n_dilim, -1).mean(1).numpy())
        logitler.append(lg.view(len(parca), n_dilim, -1).mean(1).numpy())
    return np.concatenate(gommeler), np.concatenate(logitler)


def egit(X: np.ndarray, y: np.ndarray, egitim: np.ndarray, dogrulama: np.ndarray, n_sinif: int,
         epok: int = 12, toplu: int = 64, ogrenme: float = 3e-4, sabir: int = 3, tohum: int = 2026,
         gunluk=print) -> nn.Module:
    """Sınıf ağırlıklı çapraz entropi + AdamW; doğrulama macro-F1'e göre en iyi epok tutulur."""
    torch.manual_seed(tohum)
    np.random.seed(tohum)
    model = model_olustur(n_sinif)
    sayim = np.bincount(y[egitim], minlength=n_sinif).astype(np.float32)
    agirlik = torch.tensor(sayim.sum() / (n_sinif * np.maximum(sayim, 1)))
    kayip = nn.CrossEntropyLoss(weight=agirlik)
    opt = torch.optim.AdamW(model.parameters(), lr=ogrenme, weight_decay=1e-4)
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epok)
    yukleyici = DataLoader(DilimVeri(X, y, egitim, artirma=True), batch_size=toplu, shuffle=True, num_workers=0)
    en_iyi, en_iyi_f1, bekle = None, -1.0, 0
    for e in range(epok):
        model.train()
        toplam = 0.0
        for xb, yb in yukleyici:
            opt.zero_grad()
            l = kayip(model(xb), yb)
            l.backward()
            opt.step()
            toplam += l.item() * len(yb)
        plan.step()
        _, lg = vizit_ciktilari(model, X, dogrulama)
        f1 = f1_score(y[dogrulama], lg.argmax(1), average="macro")
        gunluk(f"    epok {e + 1}/{epok}: kayıp {toplam / len(yukleyici.dataset):.3f}  doğrulama macro-F1 {f1:.3f}")
        if f1 > en_iyi_f1:
            en_iyi, en_iyi_f1, bekle = copy.deepcopy(model.state_dict()), f1, 0
        else:
            bekle += 1
            if bekle >= sabir:
                break
    model.load_state_dict(en_iyi)
    model.eval()
    return model


def gradcam(model: nn.Module, x: np.ndarray, sinif: int) -> np.ndarray:
    """x: (3, H, W) ölçeklenmiş tek dilim. Son evrişim bloğu çıktısına göre Grad-CAM, [0,1], (H, W)."""
    model.eval()
    aktivasyon = {}
    katman = model.features[-1]
    h1 = katman.register_forward_hook(lambda m, i, o: aktivasyon.__setitem__("a", o))
    h2 = katman.register_full_backward_hook(lambda m, gi, go: aktivasyon.__setitem__("g", go[0]))
    try:
        girdi = torch.from_numpy(np.ascontiguousarray(x))[None].requires_grad_(True)
        model.zero_grad()
        model(girdi)[0, sinif].backward()
        a, g = aktivasyon["a"][0], aktivasyon["g"][0]
        cam = torch.relu((g.mean(dim=(1, 2), keepdim=True) * a).sum(0))
        cam = torch.nn.functional.interpolate(cam[None, None], size=x.shape[1:], mode="bilinear",
                                              align_corners=False)[0, 0]
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().numpy()
    finally:
        h1.remove()
        h2.remove()
