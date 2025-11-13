# coding: utf-8
"""Physics-aware multimodal agent demo."""
import os
import sys
import json
import math
import random
import time
import shutil
import pathlib
import subprocess
import textwrap
import urllib.parse
from pathlib import Path

# ===== 0) Environment: install deps (kept minimal; pin versions only when necessary) =====
pkgs = [
  'torch', 'torchvision', 'torchaudio',
  'einops', 'tqdm', 'numpy', 'pillow', 'requests', 'ffmpeg-python', 'opencv-python',
  'spikingjelly',        # SNNs
  'tonic',               # event data utils
  'v2e',                 # video->events conversion
  'torchdiffeq',         # ODE solvers for Liquid
  'mamba-ssm',           # SSM (Mamba); will GRU-fallback if unavailable
  'transformers', 'accelerate', 'safetensors', # (optional) for generators
  'diffusers'            # (optional) physics-aware generator adapter
]

# Quiet install; ignore failures for optional libs
for p in pkgs:
    try:
        __import__(p.split('==')[0].replace('-', '_'))
    except Exception:
        print(f"Installing {p}…")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', p])

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
import numpy as np
from tqdm import tqdm
import ffmpeg
import cv2
import requests

# Optional vendor SDKs (tool use)
try:
    import openai  # openai>=1.x
except Exception:
    openai = None
try:
    import anthropic
except Exception:
    anthropic = None
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ===== 1) Config =====
class CONFIG:
    ROOT = Path('/content') if Path('/content').exists() else Path.cwd()
    DATA = ROOT / 'pa_agent_data'
    VIDEOS = DATA / 'videos'
    FRAMES = DATA / 'frames'
    EVENTS = DATA / 'events'
    CKPT = DATA / 'checkpoints'
    LOGS = DATA / 'logs'

    # Small demo list (Internet Archive / CC). Replace/add your own.
    VIDEO_URLS = [
        # Big Buck Bunny (CC-BY) — short MP4 mirror on archive
        'https://archive.org/download/BigBuckBunny_328/BigBuckBunny_512kb.mp4',
        # Tears of Steel (CC-BY)
        'https://archive.org/download/TearsOfSteel_720p/TearsOfSteel_720p.mp4',
    ]

    # Clip extraction
    NUM_VIDEOS = 2
    CLIP_SECONDS = 6
    TARGET_FPS = 15
    RESOLUTION = (224, 224)  # H, W
    EVENT_DT = 1/240.0       # synthetic event step for v2e (~240Hz)

    # Training
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH = 2
    EPOCHS = 1
    STEPS = 40               # small demo
    LR = 2e-4

    # Model dims
    C_RGB = 3
    T = TARGET_FPS * CLIP_SECONDS
    D_LATENT = 256
    D_FUSED = 384
    D_REASON = 384

CONFIG = CONFIG()
for d in [CONFIG.DATA, CONFIG.VIDEOS, CONFIG.FRAMES, CONFIG.EVENTS, CONFIG.CKPT, CONFIG.LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# ===== 2) Utilities: download, frame extraction, events via v2e =====

def download(url: str, out: Path):
    if out.exists():
        return out
    print(f"Downloading: {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(out, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1<<20):
            if chunk:
                f.write(chunk)
    return out


def extract_clip(mp4_path: Path, out_dir: Path, seconds=6, fps=15, size=(224,224)):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Take a random 6s window
    try:
        probe = ffmpeg.probe(str(mp4_path))
        dur = float(next(s for s in probe['format'].get('duration', 60.0).split()) )
    except Exception:
        dur = 60.0
    start = max(0.0, random.uniform(0, max(0.0, dur - seconds - 1)))
    out_pat = str(out_dir / 'f_%05d.jpg')
    (
      ffmpeg
      .input(str(mp4_path), ss=start, t=seconds)
      .filter('fps', fps=fps)
      .filter('scale', size[1], size[0])
      .output(out_pat, start_number=0, vframes=seconds*fps, loglevel='error')
      .overwrite_output()
      .run()
    )
    frames = sorted(out_dir.glob('f_*.jpg'))
    return frames


def frames_to_tensor(frames):
    # (T, C, H, W), float32 [0,1]
    imgs = []
    for p in frames:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    arr = np.stack(imgs, axis=0)
    arr = torch.from_numpy(arr).float()/255.0
    arr = rearrange(arr, 't h w c -> t c h w')
    return arr

# Synthetic events using v2e's API (simple call via CLI)

def video_to_events_with_v2e(video_path: Path, out_npz: Path, dt=CONFIG.EVENT_DT):
    if out_npz.exists():
        return out_npz
    # v2e CLI: v2e --input input.mp4 --output output.npz --dvs240
    # We'll use a simplified invocation with defaults suitable for demos.
    cmd = [sys.executable, '-m', 'v2e', '--input', str(video_path), '--output', str(out_npz), '--dvs240']
    print('Running v2e:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return out_npz


def load_event_voxels(npz_path: Path, T=CONFIG.T, H=CONFIG.RESOLUTION[0], W=CONFIG.RESOLUTION[1]):
    # Expect keys: t, x, y, p (polarity). We'll voxelize into (T, 2, H, W)
    data = np.load(npz_path)
    t = data['t']  # seconds
    x = data['x'].astype(np.int64)
    y = data['y'].astype(np.int64)
    p = data['p'].astype(np.int64)  # 0/1
    tmin, tmax = float(t.min()), float(t.max() if len(t)>0 else 1.0)
    bins = np.linspace(tmin, tmax + 1e-6, T+1)
    vox = np.zeros((T, 2, H, W), dtype=np.float32)
    if len(t):
        ti = np.clip(np.digitize(t, bins) - 1, 0, T-1)
        pol = p
        for i in range(len(t)):
            vox[ti[i], pol[i], y[i], x[i]] += 1.0
        # normalize per-bin
        vmax = vox.max()
        if vmax>0: vox /= vmax
    return torch.from_numpy(vox)

# ===== 3) Dataset over web videos =====
class WebVideoEventDataset(torch.utils.data.Dataset):
    def __init__(self, urls, root: Path, clip_s=CONFIG.CLIP_SECONDS, fps=CONFIG.TARGET_FPS, resize=CONFIG.RESOLUTION):
        self.items = []
        urls = urls[:CONFIG.NUM_VIDEOS]
        for u in urls:
            fn = Path(urllib.parse.urlparse(u).path).name or f"v_{abs(hash(u))}.mp4"
            vpath = root / 'videos' / fn
            download(u, vpath)
            # frames
            fdir = root / 'frames' / fn.replace('.mp4','')
            frames = extract_clip(vpath, fdir, seconds=clip_s, fps=fps, size=resize)
            # small temporary MP4 from extracted frames (for v2e)
            tmp_mp4 = fdir / 'clip.mp4'
            (
                ffmpeg
                .input(str(fdir / 'f_%05d.jpg'), pattern_type='sequence', framerate=fps)
                .output(str(tmp_mp4), vcodec='libx264', pix_fmt='yuv420p', loglevel='error')
                .overwrite_output().run()
            )
            # events
            npz = root / 'events' / (fn.replace('.mp4','') + '.npz')
            video_to_events_with_v2e(tmp_mp4, npz)
            self.items.append({'rgb_dir': fdir, 'rgb_mp4': tmp_mp4, 'event_npz': npz})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        frames = sorted(Path(it['rgb_dir']).glob('f_*.jpg'))
        rgb = frames_to_tensor(frames)[:CONFIG.T]       # (T,3,H,W)
        events = load_event_voxels(it['event_npz'], T=rgb.shape[0], H=rgb.shape[2], W=rgb.shape[3])  # (T,2,H,W)
        return rgb, events

# ===== 4) Models =====
# 4.1 Event SNN encoder (very small)
from spikingjelly.activation_based import neuron, functional as SF

class EventSNN(nn.Module):
    def __init__(self, in_ch=2, d_out=128):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, 16, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.spike1 = neuron.LIFNode(tau=2.0, detach_reset=True)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=(2,2,2), padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.spike2 = neuron.LIFNode(tau=2.5, detach_reset=True)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.spike3 = neuron.LIFNode(tau=3.0, detach_reset=True)
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d((None,1,1)), nn.Conv3d(64, d_out, 1))

    def forward(self, ev):
        # ev: (B, T, 2, H, W) -> (B, C=2, T, H, W)
        ev = rearrange(ev, 'b t c h w -> b c t h w')
        x = self.conv1(ev); x = self.bn1(x); x = self.spike1(x)
        x = self.conv2(x); x = self.bn2(x); x = self.spike2(x)
        x = self.conv3(x); x = self.bn3(x); x = self.spike3(x)
        x = self.head(x)  # (B, d_out, T', 1, 1)
        x = rearrange(x, 'b d t 1 1 -> b t d')
        return x

# 4.2 RGB video encoder with Mamba (fallback GRU if mamba unavailable)
try:
    from mamba_ssm import Mamba
    class MambaBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.m = Mamba(d_model=d, d_state=16)
        def forward(self, x):  # x: (B,T,D)
            return self.m(x)
    USE_MAMBA = True
except Exception:
    class MambaBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.gru = nn.GRU(d, d, batch_first=True)
        def forward(self, x):
            y,_ = self.gru(x); return y
    USE_MAMBA = False

class RGBEncoder(nn.Module):
    def __init__(self, d_out=128, patch=16):
        super().__init__()
        self.patch = patch
        self.conv = nn.Conv3d(3, 64, kernel_size=(3,7,7), stride=(1,4,4), padding=(1,3,3))
        self.bn = nn.BatchNorm3d(64)
        self.proj = nn.Linear(64*((CONFIG.RESOLUTION[0]//4)//patch)*((CONFIG.RESOLUTION[1]//4)//patch), d_out)
        self.mamba = MambaBlock(d_out)
    def forward(self, rgb):  # (B,T,3,H,W)
        x = rearrange(rgb, 'b t c h w -> b c t h w')
        x = self.conv(x); x = self.bn(x); x = F.gelu(x)
        # simple spatiotemporal downsample into tokens along time
        x = rearrange(x, 'b c t h w -> b t c h w')
        # average pool to patch grid then flatten
        ph, pw = self.patch, self.patch
        h, w = x.shape[-2], x.shape[-1]
        x = F.adaptive_avg_pool3d(rearrange(x,'b t c h w->b c t h w'), (x.shape[2], h//ph, w//pw))
        x = rearrange(x, 'b c t hh ww -> b t (c hh ww)')
        x = self.proj(x)
        x = self.mamba(x)
        return x

# 4.3 Liquid ODE world model (compact continuous-time RNN)
from torchdiffeq import odeint

class LiquidCell(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, d) * 0.05)
        self.u = nn.Parameter(torch.randn(d, d) * 0.05)
        self.b = nn.Parameter(torch.zeros(d))
        self.alpha = nn.Parameter(torch.ones(d)*0.5)  # time constant
    def forward(self, t, y, inp):
        # dy/dt = -alpha*y + tanh(Wy + Ux + b)
        return -self.alpha * y + torch.tanh(y @ self.w.T + inp @ self.u.T + self.b)

class LiquidWorldModel(nn.Module):
    def __init__(self, d_in, d_lat=CONFIG.D_LATENT):
        super().__init__()
        self.enc = nn.Linear(d_in, d_lat)
        self.cell = LiquidCell(d_lat)
        self.dec = nn.Linear(d_lat, d_lat)
    def forward(self, seq):  # seq: (B,T,D)
        B,T,D = seq.shape
        h0 = torch.zeros(B, self.dec.in_features, device=seq.device)
        ts = torch.linspace(0, 1, T, device=seq.device)
        xs = self.enc(seq)
        def f(t, y):
            # nearest time index
            i = min(int(t.item()*(T-1)+0.5), T-1)
            return self.cell(t, y, xs[:, i, :])
        ys = []
        y = h0
        for i in range(T):
            t0, t1 = ts[i-1] if i>0 else ts[0], ts[i]
            y = odeint(lambda tt, yy: f(tt, yy), y, torch.stack([t0, t1]))[-1]
            ys.append(y)
        ys = torch.stack(ys, dim=1)
        return self.dec(ys)  # (B,T,D_lat)

# 4.4 Fusion + Reasoner (Mamba/GRU tiny head)
class FusionReasoner(nn.Module):
    def __init__(self, d_ev=128, d_rgb=128, d_fused=CONFIG.D_FUSED, d_reason=CONFIG.D_REASON):
        super().__init__()
        self.lin_ev = nn.Linear(d_ev, d_fused//2)
        self.lin_rgb = nn.Linear(d_rgb, d_fused//2)
        self.ln = nn.LayerNorm(d_fused)
        self.reason = MambaBlock(d_reason)
        self.proj = nn.Linear(d_fused, d_reason)
    def forward(self, ev_tok, rgb_tok):  # (B,T,De), (B,T,Dr)
        x = torch.cat([self.lin_ev(ev_tok), self.lin_rgb(rgb_tok)], dim=-1)
        x = self.ln(x)
        x = self.proj(x)
        x = self.reason(x)
        return x  # (B,T,D_reason)

# 4.5 JEPA head (predict masked future latents)
class JEPAHead(nn.Module):
    def __init__(self, d_in, d_lat=CONFIG.D_LATENT):
        super().__init__()
        self.pred = nn.Sequential(nn.Linear(d_in, d_lat), nn.GELU(), nn.Linear(d_lat, d_lat))
    def forward(self, ctx):
        return self.pred(ctx)

# Full model wrapper
class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.snn = EventSNN(in_ch=2, d_out=128)
        self.rgb = RGBEncoder(d_out=128)
        self.fuser = FusionReasoner(d_ev=128, d_rgb=128)
        self.world = LiquidWorldModel(d_in=CONFIG.D_REASON, d_lat=CONFIG.D_LATENT)
        self.jepa = JEPAHead(d_in=CONFIG.D_REASON, d_lat=CONFIG.D_LATENT)
    def forward_ctx_tgt(self, rgb, ev, t_split=None):
        B,T,C,H,W = rgb.shape
        if t_split is None: t_split = T//2
        ev_lat = self.snn(ev)          # (B,T',De)
        rgb_lat = self.rgb(rgb)        # (B,T',Dr)
        Tm = min(ev_lat.shape[1], rgb_lat.shape[1])
        ev_lat, rgb_lat = ev_lat[:, :Tm], rgb_lat[:, :Tm]
        fused = self.fuser(ev_lat, rgb_lat)   # (B,Tm,D)
        ctx, tgt = fused[:, :t_split], fused[:, t_split:]
        # JEPA prediction: predict latent of tgt from ctx via world model rollouts
        world_roll = self.world(ctx)          # (B,t_split,D_lat)
        pred = self.jepa(ctx)                 # map ctx tokens to latent space
        # align pred to tgt length (simple last-state repeat)
        if pred.shape[1] < tgt.shape[1]:
            pred = torch.cat([pred, pred[:, -1:].repeat(1, tgt.shape[1]-pred.shape[1], 1)], dim=1)
        else:
            pred = pred[:, :tgt.shape[1]]
        return pred, world_roll, tgt

# ===== 5) Training loop (tiny demo) =====

def cosine_loss(a, b, eps=1e-8):
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    return (1 - (a*b).sum(dim=-1)).mean()

@torch.no_grad()
def physics_plausibility_stub(rgb):
    # Very lightweight heuristic: penalize large inter-frame intensity jumps
    diffs = (rgb[:,1:] - rgb[:,:-1]).abs().mean(dim=[2,3,4])  # (B,T-1)
    score = torch.clamp(1.0 - diffs.mean()/0.25, 0.0, 1.0)
    return score.item()

# Data
print("\nPreparing data…")
DS = WebVideoEventDataset(CONFIG.VIDEO_URLS, CONFIG.DATA)
loader = torch.utils.data.DataLoader(DS, batch_size=CONFIG.BATCH, shuffle=True, drop_last=True)

# Model
agent = Agent().to(CONFIG.DEVICE)
opt = torch.optim.AdamW(agent.parameters(), lr=CONFIG.LR)

print(f"Device: {CONFIG.DEVICE} | Mamba: {USE_MAMBA}")
print("Starting tiny training demo…")
agent.train()
step = 0
for epoch in range(CONFIG.EPOCHS):
    for rgb, ev in loader:
        step += 1
        rgb = rgb.to(CONFIG.DEVICE)   # (B,T,3,H,W)
        ev = ev.to(CONFIG.DEVICE)     # (B,T,2,H,W)
        pred, world_lat, tgt = agent.forward_ctx_tgt(rgb, ev)
        loss_pred = cosine_loss(pred, tgt)
        loss_reg = 1e-3 * (world_lat**2).mean()
        loss = loss_pred + loss_reg
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 5 == 0:
            print(f"step {step:04d}  loss={loss.item():.4f}  pred={loss_pred.item():.4f}  reg={loss_reg.item():.6f}  phys≈{physics_plausibility_stub(rgb):.3f}")
        if step >= CONFIG.STEPS:
            break
    if step >= CONFIG.STEPS:
        break

print("Training demo complete.")

# ===== 6) Optional generator adapter (diffusers) =====
try:
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    HAVE_DIFFUSERS = True
except Exception:
    HAVE_DIFFUSERS = False

class GeneratorAdapter(nn.Module):
    def __init__(self, d_lat=CONFIG.D_LATENT):
        super().__init__()
        self.lin = nn.Linear(d_lat, 64)
    def forward(self, lat_seq):
        # Placeholder: returns a dummy tensor; integrate Stable Video Diffusion if desired
        return torch.tanh(self.lin(lat_seq))

gen = GeneratorAdapter().to(CONFIG.DEVICE)

# ===== 7) Tool schemas & vendor hooks (planner/verifier) =====
TOOL_SCHEMAS = [
  {
    "name":"imagine","description":"Roll out dynamics for a scene.",
    "parameters":{"type":"object","properties":{
      "scene_graph":{"type":"object"},"actions":{"type":"array","items":{"type":"object"}},"horizon_s":{"type":"number"}},
      "required":["scene_graph","horizon_s"]}
  },
  {
    "name":"physics_score","description":"Score rollout for intuitive-physics violations.",
    "parameters":{"type":"object","properties":{"rollout":{"type":"object"}},"required":["rollout"]}
  },
  {
    "name":"render","description":"Turn rollout into a short video sample (latent diffusion adapter).",
    "parameters":{"type":"object","properties":{"rollout":{"type":"object"},"caption":{"type":"string"}},"required":["rollout"]}
  }
]

# Local tool implementations (LLM will call these)

def tool_imagine(scene_graph: dict, actions: list, horizon_s: float):
    # For demo: use last batch's fused tokens to roll forward
    with torch.no_grad():
        B = 1
        ctx = torch.randn(B, max(1,int(horizon_s*CONFIG.TARGET_FPS/2)), CONFIG.D_REASON, device=CONFIG.DEVICE)
        roll = agent.world(ctx).cpu().numpy().tolist()
    return {"latent_rollout": roll}

def tool_physics_score(rollout: dict):
    # Simple L2 velocity smoothness as plausibility proxy
    lat = torch.tensor(rollout.get('latent_rollout', [[[]]]), dtype=torch.float32)
    v = lat[:,1:] - lat[:,:-1]
    score = float(torch.exp(- (v**2).mean()).item())
    return {"score": score}

def tool_render(rollout: dict, caption: str = ""):
    lat = torch.tensor(rollout.get('latent_rollout', [[[]]]), dtype=torch.float32).to(CONFIG.DEVICE)
    with torch.no_grad(): vid_lat = gen(lat).cpu().numpy().tolist()
    return {"video_latents": vid_lat, "caption": caption}

# Vendor adapters (only used if keys provided). They share TOOL_SCHEMAS.
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

class LLMPlanner:
    def __init__(self):
        self.enabled = any([OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY])
        if genai and GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
        self.sys_prompt = (
            "You are a physics-aware planner. Use the available tools (imagine, physics_score, render)\n"
            "to propose and test counterfactuals. Prefer pairwise comparisons and provide JSON plans."
        )
    def plan_once(self, query:str):
        # For simplicity: call local tools directly in this demo.
        # In production, register TOOL_SCHEMAS with your chosen vendor and forward tool calls.
        rollout = tool_imagine({}, [], 2.0)
        score = tool_physics_score(rollout)
        return {"rollout": rollout, "score": score}

planner = LLMPlanner()
print("LLM planner enabled:", planner.enabled)
print("Done. This notebook produced a trained (toy) world model and live tools you can wire to ChatGPT/Claude/Gemini.")
