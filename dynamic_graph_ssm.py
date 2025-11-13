# Dynamic-Graph SSM — single-cell script version
# Debugged version of the provided Colab cell for easier execution outside of notebooks.

import os
import math
import time
import random
import gc
import contextlib
from dataclasses import dataclass
from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import imageio
except ImportError:  # pragma: no cover - dependency optional for debugging
    imageio = None

try:
    from einops import rearrange, reduce
except ImportError as exc:  # pragma: no cover - clarify missing dependency
    raise RuntimeError("einops is required for dynamic_graph_ssm") from exc


CONFIG = {
    "demo": "video_pred",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 1337,
    "V": {
        "frames": 12,
        "H": 64,
        "W": 64,
        "num_objects": (2, 3),
        "train_samples": 512,
        "val_samples": 128,
        "batch": 8,
    },
    "TOK": {"patch": 8, "dim": 64, "frozen": True},
    "SSM": {"d_model": 128, "depth": 4, "drop": 0.0},
    "OPT": {"lr": 2e-3, "wd": 0.0},
    "TRAIN": {"steps": 400, "val_every": 100, "amp": True},
    "GDEMO": {
        "n_shots": 3,
        "n_queries": 64,
        "colors": ["red", "green", "blue", "yellow", "magenta", "cyan"],
        "shapes": ["circle", "square"],
        "retention_thresh": 0.2,
        "steps": 300,
        "batch": 32,
    },
    "SAVE_DIR": os.path.join(os.getcwd(), "dgssm_artifacts"),
}

os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)


def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(-1)


def to_gif(frames: torch.Tensor, path: str, fps: int = 12) -> str:
    if imageio is None:
        raise RuntimeError("imageio is required to export GIFs")
    frames = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8).tolist()
    imageio.mimsave(path, frames, duration=1.0 / max(fps, 1))
    return path


class MovingShapes(Dataset):
    def __init__(self, n, T=12, H=64, W=64, nobj_range=(2, 3)):
        self.n = n
        self.T = T
        self.H = H
        self.W = W
        self.nobj_range = nobj_range

    def __len__(self):
        return self.n

    def _draw_square(self, canvas, cx, cy, s, col):
        x0 = max(0, cx - s)
        x1 = min(self.W, cx + s)
        y0 = max(0, cy - s)
        y1 = min(self.H, cy + s)
        canvas[:, y0:y1, x0:x1] = torch.maximum(
            canvas[:, y0:y1, x0:x1], col[:, None, None]
        )

    def _draw_circle(self, canvas, cx, cy, r, col):
        yy, xx = torch.meshgrid(
            torch.arange(self.H), torch.arange(self.W), indexing="ij"
        )
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        canvas[:, mask] = torch.maximum(canvas[:, mask], col[:, None])

    def __getitem__(self, idx):
        T, H, W = self.T, self.H, self.W
        nobj = random.randint(*self.nobj_range)
        objs = []
        for _ in range(nobj):
            cx = random.randint(8, W - 8)
            cy = random.randint(8, H - 8)
            vx = random.choice([-1, 1]) * random.randint(1, 2)
            vy = random.choice([-1, 1]) * random.randint(1, 2)
            s = random.randint(4, 8)
            r = random.randint(4, 7)
            col = torch.rand(3)
            shp = random.choice(["square", "circle"])
            objs.append([cx, cy, vx, vy, s, r, col, shp])
        frames = []
        for _ in range(T):
            canvas = torch.zeros(3, H, W)
            for i in range(nobj):
                cx, cy, vx, vy, s, r, col, shp = objs[i]
                drawer = self._draw_square if shp == "square" else self._draw_circle
                drawer(canvas, cx, cy, s if shp == "square" else r, col)
                cx += vx
                cy += vy
                if cx < 8 or cx > W - 8:
                    vx *= -1
                    cx += 2 * vx
                if cy < 8 or cy > H - 8:
                    vy *= -1
                    cy += 2 * vy
                objs[i] = [cx, cy, vx, vy, s, r, col, shp]
            frames.append(canvas)
        return torch.stack(frames, 0).clamp(0, 1)


class PatchTokenizer(nn.Module):
    def __init__(self, in_ch=3, dim=64, patch=8, frozen=True):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.proj.bias)
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        y = self.proj(x)
        return rearrange(y, "(b t) d h w -> b t d h w", b=B, t=T)


class PatchDecoder(nn.Module):
    def __init__(self, out_ch=3, dim=64, patch=8):
        super().__init__()
        self.deproj = nn.ConvTranspose2d(dim, out_ch, kernel_size=patch, stride=patch)
        nn.init.kaiming_normal_(self.deproj.weight, nonlinearity="relu")
        nn.init.zeros_(self.deproj.bias)

    def forward(self, tokens):
        B, T, D, h, w = tokens.shape
        x = rearrange(tokens, "b t d h w -> (b t) d h w")
        imgs = self.deproj(x)
        return rearrange(imgs, "(b t) c h w -> b t c h w", b=B, t=T)


class TinySSM1D(nn.Module):
    def __init__(self, d_model, drop=0.0):
        super().__init__()
        self.inp = nn.Linear(d_model, 3 * d_model, bias=False)
        self.A = nn.Parameter(torch.randn(d_model) * -0.1)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, L, D = x.shape
        u, i, o = self.inp(x).chunk(3, dim=-1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)
        s = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outs = []
        a = torch.exp(self.A).clamp(max=0.999)
        for t in range(L):
            s = a * s + i[:, t] * u[:, t]
            outs.append(o[:, t] * s)
        y = torch.stack(outs, 1)
        return self.out(self.drop(y))


def scan_2d(x, block: TinySSM1D):
    B, D, H, W = x.shape
    lr = rearrange(x, "b d h w -> (b h) w d")
    lr = block(lr)
    lr = rearrange(lr, "(b h) w d -> b d h w", b=B, h=H)

    rl = rearrange(torch.flip(x, [3]), "b d h w -> (b h) w d")
    rl = block(rl)
    rl = rearrange(rl, "(b h) w d -> b d h w", b=B, h=H)
    rl = torch.flip(rl, [3])

    ud = rearrange(x, "b d h w -> (b w) h d")
    ud = block(ud)
    ud = rearrange(ud, "(b w) h d -> b d h w", b=B, w=W)

    du = rearrange(torch.flip(x, [2]), "b d h w -> (b w) h d")
    du = block(du)
    du = rearrange(du, "(b w) h d -> b d h w", b=B, w=W)
    du = torch.flip(du, [2])
    return (lr + rl + ud + du) / 4


class SpatioTemporalSSM(nn.Module):
    def __init__(self, d_model=128, depth=4, drop=0.0):
        super().__init__()
        self.s_blocks = nn.ModuleList([TinySSM1D(d_model, drop) for _ in range(depth)])
        self.t_blocks = nn.ModuleList([TinySSM1D(d_model, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tok):
        B, T, D, h, w = tok.shape
        x = tok
        for s_block, t_block in zip(self.s_blocks, self.t_blocks):
            xs = rearrange(x, "b t d h w -> (b t) d h w")
            xs = scan_2d(xs, s_block)
            xs = rearrange(xs, "(b t) d h w -> b t d h w", b=B, t=T)
            xt = rearrange(xs, "b t d h w -> (b h w) t d")
            xt = t_block(xt)
            xt = rearrange(xt, "(b h w) t d -> b t d h w", b=B, h=h, w=w)
            x = x + xt
        x = self.norm(rearrange(x, "b t d h w -> b t h w d"))
        return rearrange(x, "b t h w d -> b t d h w")


class TinyVideoModel(nn.Module):
    def __init__(self, tok_dim=64, d_model=128, depth=4, drop=0.0):
        super().__init__()
        self.in_proj = nn.Conv2d(tok_dim, d_model, 1)
        self.backbone = SpatioTemporalSSM(d_model, depth, drop)
        self.out_proj = nn.Conv2d(d_model, tok_dim, 1)

    def forward(self, toks):
        B, T, D, h, w = toks.shape
        x = rearrange(toks, "b t d h w -> (b t) d h w")
        x = self.in_proj(x)
        x = rearrange(x, "(b t) d h w -> b t d h w", b=B, t=T)
        x = self.backbone(x)
        x = rearrange(x, "b t d h w -> (b t) d h w")
        x = self.out_proj(x)
        return rearrange(x, "(b t) d h w -> b t d h w", b=B, t=T)


@dataclass
class GraphNode:
    id: int
    typ: str
    feat: torch.Tensor


@dataclass
class GraphEdge:
    src: int
    dst: int
    typ: str
    w: float


class DynamicGraphMemory(nn.Module):
    def __init__(self, d=64, retention_thresh=0.2):
        super().__init__()
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []
        self.next_id = 0
        self.retention_thresh = retention_thresh
        self.scorer = nn.Sequential(
            nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1)
        )

    def reset(self):
        self.nodes = []
        self.edges = []
        self.next_id = 0

    def add_node(self, typ, feat):
        nid = self.next_id
        self.next_id += 1
        self.nodes.append(GraphNode(nid, typ, feat.detach().clone()))
        return nid

    def add_edge(self, src, dst, typ, w=1.0):
        self.edges.append(GraphEdge(src, dst, typ, w))

    def prune(self):
        if not self.edges:
            return
        keep = []
        with torch.no_grad():
            for e in self.edges:
                fs = self.nodes[e.src].feat
                fd = self.nodes[e.dst].feat
                z = torch.cat([fs, fd, fs * fd], -1)
                s = torch.sigmoid(self.scorer(z)).item()
                if s >= self.retention_thresh:
                    keep.append(e)
        self.edges = keep

    def retrieve(self, query_vec, from_typ="IMG", to_typ="COLOR"):
        results = []
        for e in self.edges:
            nsrc = self.nodes[e.src]
            ndst = self.nodes[e.dst]
            if nsrc.typ == from_typ and ndst.typ == to_typ:
                results.append((ndst, cosine_sim(query_vec, ndst.feat).item()))
        if not results:
            return None, 0.0
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0]


class SymbolBank(nn.Module):
    def __init__(self, colors, shapes, d=64):
        super().__init__()
        self.colors = colors
        self.shapes = shapes
        self.color_tbl = nn.Parameter(torch.randn(len(colors), d))
        self.shape_tbl = nn.Parameter(torch.randn(len(shapes), d))

    def color(self, name):
        return self.color_tbl[self.colors.index(name)]

    def shape(self, name):
        return self.shape_tbl[self.shapes.index(name)]


def synth_image_token(color_vec, shape_kind, H=32, W=32, patch=8, tok_dim=64):
    tok = PatchTokenizer(3, tok_dim, patch, frozen=True).to(CONFIG["device"])
    img = torch.zeros(3, H, W)
    col = (torch.sigmoid(color_vec[:3]) * 0.8 + 0.2).clamp(0, 1)
    cx, cy = random.randint(10, W - 10), random.randint(10, H - 10)
    if shape_kind == "square":
        s = random.randint(4, 7)
        img[:, cy - s : cy + s, cx - s : cx + s] = col[:, None, None]
    else:
        yy, xx = torch.meshgrid(
            torch.arange(H), torch.arange(W), indexing="ij"
        )
        r = random.randint(4, 7)
        img[:, ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r] = col[:, None]
    tokens = tok(img.unsqueeze(0).to(CONFIG["device"]))
    pooled = reduce(tokens, "b d h w -> b d", "mean")[0]
    return img, tokens[0], pooled


def _create_autocast_and_scaler(device: str, amp_enabled: bool):
    if device != "cuda" or not amp_enabled:
        class _NullScaler:
            def scale(self, loss):
                return loss

            def step(self, opt):
                opt.step()

            def update(self):
                return None

        return contextlib.nullcontext(), _NullScaler()

    autocast = torch.cuda.amp.autocast(dtype=torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    return autocast, scaler


def train_video_pred_and_visualize():
    V = CONFIG["V"]
    TOK = CONFIG["TOK"]
    SSM = CONFIG["SSM"]
    OPT = CONFIG["OPT"]
    TR = CONFIG["TRAIN"]
    train_ds = MovingShapes(
        V["train_samples"], V["frames"], V["H"], V["W"], V["num_objects"]
    )
    val_ds = MovingShapes(
        V["val_samples"], V["frames"], V["H"], V["W"], V["num_objects"]
    )
    TL = DataLoader(
        train_ds,
        batch_size=V["batch"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    VL = DataLoader(
        val_ds, batch_size=V["batch"], shuffle=False, num_workers=0
    )

    tok = PatchTokenizer(3, TOK["dim"], TOK["patch"], TOK["frozen"]).to(CONFIG["device"])
    dec = PatchDecoder(3, TOK["dim"], TOK["patch"]).to(CONFIG["device"])
    net = TinyVideoModel(TOK["dim"], SSM["d_model"], SSM["depth"], SSM["drop"]).to(
        CONFIG["device"]
    )

    opt = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad] + list(dec.parameters()),
        lr=OPT["lr"],
        weight_decay=OPT["wd"],
    )

    amp_enabled = CONFIG["device"] == "cuda" and TR["amp"]
    autocast_ctx, scaler = _create_autocast_and_scaler(CONFIG["device"], amp_enabled)

    step = 0
    t0 = time.time()
    net.train()

    for epoch in range(999999):
        for vids in TL:
            vids = vids.to(CONFIG["device"])
            with autocast_ctx:
                toks = tok(vids)
                pred = net(toks[:, :-1])
                loss = F.mse_loss(pred, toks[:, 1:])
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            if (step + 1) % 25 == 0:
                print(f"[video_pred] step {step + 1:04d} | token-MSE {loss.item():.4f}")
            if (step + 1) % TR["val_every"] == 0:
                net.eval()
                vs = 0
                vn = 0
                with torch.no_grad():
                    for v in VL:
                        v = v.to(CONFIG["device"])
                        with autocast_ctx:
                            tt = tok(v)
                            pr = net(tt[:, :-1])
                            tar = tt[:, 1:]
                            vs += F.mse_loss(pr, tar, reduction="sum").item()
                            vn += pr.numel()
                print(f"  -> val token-MSE: {vs / vn:.6f}")
                net.train()

            step += 1
            if step >= TR["steps"]:
                torch.save(net.state_dict(), f"{CONFIG['SAVE_DIR']}/videomodel.pt")
                net.eval()
                with torch.no_grad():
                    v = next(iter(VL)).to(CONFIG["device"])
                    with autocast_ctx:
                        tt = tok(v)
                        pr_tok = net(tt[:, :-1]).detach()
                    rec = dec(pr_tok).clamp(0, 1)
                    gt = v[:, 1:]
                if imageio is not None:
                    pred_path = to_gif(
                        rec[0], f"{CONFIG['SAVE_DIR']}/pred.gif", fps=12
                    )
                    gt_path = to_gif(gt[0], f"{CONFIG['SAVE_DIR']}/gt.gif", fps=12)
                    print(
                        "Done (video_pred). Saved:",
                        CONFIG["SAVE_DIR"],
                        "| GIFs:",
                        pred_path,
                        gt_path,
                        "| Wall time",
                        f"{time.time() - t0:.1f}s",
                    )
                return


def train_fewshot_graph():
    G = CONFIG["GDEMO"]
    TOK = CONFIG["TOK"]
    sym = SymbolBank(G["colors"], G["shapes"], TOK["dim"]).to(CONFIG["device"])
    gmem = DynamicGraphMemory(TOK["dim"], G["retention_thresh"]).to(CONFIG["device"])
    opt = torch.optim.AdamW(
        list(sym.parameters()) + list(gmem.scorer.parameters()),
        lr=2e-3,
        weight_decay=0.0,
    )
    amp_enabled = CONFIG["device"] == "cuda"
    autocast_ctx, scaler = _create_autocast_and_scaler(CONFIG["device"], amp_enabled)

    def make_context():
        gmem.reset()
        for _ in range(G["n_shots"]):
            cn = random.choice(G["colors"])
            sn = random.choice(G["shapes"])
            cv = sym.color(cn)
            sv = sym.shape(sn)
            _, _, pooled = synth_image_token(
                cv, sn, 32, 32, TOK["patch"], TOK["dim"]
            )
            nid_img = gmem.add_node("IMG", pooled.detach())
            nid_col = gmem.add_node("COLOR", cv.detach())
            nid_shp = gmem.add_node("SHAPE", sv.detach())
            gmem.add_edge(nid_img, nid_col, "HAS_COLOR")
            gmem.add_edge(nid_img, nid_shp, "HAS_SHAPE")
        gmem.prune()

    t0 = time.time()
    for step in range(1, G["steps"] + 1):
        opt.zero_grad(set_to_none=True)
        make_context()
        total_loss = 0.0
        acc = 0
        with autocast_ctx:
            for _ in range(G["n_queries"]):
                cn = random.choice(G["colors"])
                sn = random.choice(G["shapes"])
                cv = sym.color(cn)
                sv = sym.shape(sn)
                _, _, pooled = synth_image_token(
                    cv, sn, 32, 32, TOK["patch"], TOK["dim"]
                )
                _, _ = gmem.retrieve(pooled, "IMG", "COLOR")
                logits = torch.stack(
                    [cosine_sim(pooled, sym.color(name)) for name in G["colors"]],
                    dim=0,
                )
                target = torch.tensor(G["colors"].index(cn), device=CONFIG["device"])
                loss = F.cross_entropy(logits[None, :], target[None])
                total_loss += loss
                acc += int(logits.argmax().item() == target.item())
        scaler.scale(total_loss / G["n_queries"]).backward()
        scaler.step(opt)
        scaler.update()
        if step % 25 == 0:
            print(
                f"[fewshot_graph] step {step:04d} | CE {float(total_loss) / G['n_queries']:.4f} "
                f"| acc {acc / G['n_queries']:.3f} | |E|={len(gmem.edges)}"
            )
    print(f"Done (fewshot_graph). Wall time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    if CONFIG["demo"] == "video_pred":
        train_video_pred_and_visualize()
    else:
        train_fewshot_graph()
    print("\nArtifacts ->", CONFIG["SAVE_DIR"])
