# -*- coding: utf-8 -*-
"""USM v1.0: Unified Science Machine

This file combines the hard binding CTM benchmark (v0.8b) with a minimal
GridWorld + hypergraph + active inference agent. Everything is contained in a
single, Colab-friendly Python file.
"""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------
# Config + device
# ---------------------------------------------------------------------

@dataclass
class USMConfig:
    world_type: str = "hard_binding"  # "hard_binding", "gridworld", "polymarket_stub", "all"
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "usm_logs"
    run_name: str = "usm_v0_9"
    # Hard-binding specific
    hb_train_samples: int = 8000
    hb_eval_samples: int = 4000
    hb_epochs: int = 100
    hb_batch_size: int = 256
    hb_lr: float = 1e-3
    hb_pred_loss_weight: float = 0.05
    hb_bind_loss_weight: float = 0.5
    # GridWorld specific
    gw_size: int = 5
    gw_n_episodes: int = 200
    gw_max_steps: int = 40
    gw_lr: float = 1e-3
    gw_gamma: float = 0.99
    gw_latent_dim: int = 64
    # Polymarket stub
    polymarket_csv_path: str = "polymarket_data.csv"
    polymarket_use_synthetic: bool = True


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: USMConfig) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# set default device (updated in main)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# HARD BINDING DATASET (augmented with candidate object mask)
# ---------------------------------------------------------------------

class HardBindingDataset(Dataset):
    """
    Scene with 4 objects, each has:
      - color ∈ {red, green, blue, yellow}
      - shape ∈ {ball, cube, cone, star}
      - (x, y) ∈ {0,0.2,0.4,0.6,0.8}²

    Query:
      "What shape is <target_color> <relation> of <ref_color>?"

    Relations:
      - left_of, right_of, above, below

    Label:
      - shape index of one candidate object satisfying the relation.
      - Some scenes are ambiguous (multiple candidates) → marked separately.

    Extra (for binding probes, no gradient):
      - cand_mask: [4] bool, which object indices satisfy the relation.

    Encoding:
      - 4 objects × (color one-hot 4 + shape one-hot 4 + x + y) = 4 × 10 = 40
      - query = target_color (4) + relation (4) + ref_color (4) = 12
      - total input dim = 52
    """
    COLORS = ["red", "green", "blue", "yellow"]
    SHAPES = ["ball", "cube", "cone", "star"]
    RELS = ["left_of", "right_of", "above", "below"]

    def __init__(self, n_samples: int = 8000, seed: int = 0, permute_objects: bool = False):
        self.n_samples = n_samples
        self.rng = random.Random(seed)
        self.permute_objects = permute_objects
        # (x, y, ambiguous, cand_mask, objects, query)
        self.samples: List[Tuple[torch.Tensor, int, bool, torch.Tensor, list, tuple]] = []
        self._generate()

    def _sample_coord(self) -> float:
        # 5×5 grid → coordinates in {0,0.2,0.4,0.6,0.8}
        return self.rng.choice([0.0, 0.2, 0.4, 0.6, 0.8])

    def _relation_holds(self, obj, ref, rel: str) -> bool:
        x, y = obj["x"], obj["y"]
        rx, ry = ref["x"], ref["y"]
        if rel == "left_of":
            return x < rx and abs(y - ry) <= 0.4
        if rel == "right_of":
            return x > rx and abs(y - ry) <= 0.4
        if rel == "above":
            return y > ry and abs(x - rx) <= 0.4
        if rel == "below":
            return y < ry and abs(x - rx) <= 0.4
        return False

    def _generate_one(self):
        colors = self.COLORS
        shapes = self.SHAPES
        rels = self.RELS

        # 4 random objects
        objects = []
        for _ in range(4):
            obj = {
                "color": self.rng.choice(colors),
                "shape": self.rng.choice(shapes),
                "x": self._sample_coord(),
                "y": self._sample_coord(),
            }
            objects.append(obj)

        # Query + label
        attempts = 0
        while True:
            attempts += 1
            ref_idx = self.rng.randrange(4)
            ref = objects[ref_idx]
            ref_color = ref["color"]

            target_color = self.rng.choice(colors)
            rel = self.rng.choice(rels)

            # Candidate targets: objects of target_color in relation to any ref_color object
            candidates = []
            for i, obj in enumerate(objects):
                if obj["color"] != target_color:
                    continue
                holds_any = False
                for j, ref2 in enumerate(objects):
                    if ref2["color"] == ref_color and i != j:
                        if self._relation_holds(obj, ref2, rel):
                            holds_any = True
                            break
                if holds_any:
                    candidates.append(i)

            if candidates:
                break
            if attempts > 20:
                # Fallback: no candidate, treat as non-ambiguous random label
                candidates = [self.rng.randrange(4)]
                break

        ambiguous = len(candidates) > 1
        chosen_idx = self.rng.choice(candidates)
        answer_shape = objects[chosen_idx]["shape"]
        y = shapes.index(answer_shape)

        # Candidate mask [4] for binding probes
        cand_mask = torch.zeros(4, dtype=torch.bool)
        for idx in candidates:
            cand_mask[idx] = True

        # Encode features
        feat = []
        for obj in objects:
            color_oh = [1.0 if obj["color"] == c else 0.0 for c in colors]
            shape_oh = [1.0 if obj["shape"] == s else 0.0 for s in shapes]
            feat.extend(color_oh + shape_oh + [obj["x"], obj["y"]])

        tgt_oh = [1.0 if target_color == c else 0.0 for c in colors]
        rel_oh = [1.0 if rel == r else 0.0 for r in rels]
        ref_oh = [1.0 if ref_color == c else 0.0 for c in colors]
        feat.extend(tgt_oh + rel_oh + ref_oh)

        x = torch.tensor(feat, dtype=torch.float32)
        query = (target_color, rel, ref_color)
        return x, y, ambiguous, cand_mask, objects, query

    def _generate(self):
        self.samples = [self._generate_one() for _ in range(self.n_samples)]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x, y, amb, cand_mask, objects, query = self.samples[idx]
        x = x.clone()
        cand_mask = cand_mask.clone()

        if self.permute_objects:
            obj_feats = x[:40].view(4, 10)
            perm = torch.randperm(4)
            obj_feats = obj_feats[perm]
            x[:40] = obj_feats.view(-1)
            cand_mask = cand_mask[perm]

        return x, y, amb, cand_mask

    def sample_humans(self, k: int = 3):
        """Return a few human-readable examples for logging."""
        out = []
        for i in range(min(k, len(self.samples))):
            x, y, amb, cand_mask, objects, query = self.samples[i]
            out.append((objects, query, self.SHAPES[y], amb))
        return out


def ambig_fraction(ds: HardBindingDataset):
    ambigs = sum(1 for _, _, amb, _, _, _ in ds.samples if amb)
    return ambigs / len(ds.samples), ambigs


# ---------------------------------------------------------------------
# MODELS: MLP BASELINE
# ---------------------------------------------------------------------

class MLPBaseline(nn.Module):
    def __init__(self, input_dim=52, hidden=64, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# CTM BLOCK (with LTC dynamics + adaptive p + latent prediction loss)
# ---------------------------------------------------------------------

class CTMBlock(nn.Module):
    def __init__(self, dim, n_slots, n_ticks=3, dt=1.0,
                 p_min=2.0, p_max=4.0):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.n_ticks = n_ticks
        self.dt = dt
        self.p_min = p_min
        self.p_max = p_max

        self.in_proj = nn.Linear(dim, dim)
        self.pre_ln = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # LTC per-dimension timescales
        self.log_tau = nn.Parameter(torch.zeros(dim))

        # Latent prediction network: predict z_{t+1} from z_t
        self.pred_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, z, return_attn: bool = False):
        """
        z: [B, T, d]

        Returns:
          z_new: [B, T, d]
          pred_loss: scalar
          eff_p_mean: scalar (detached, for logging)
          last_attn: [B, T, T] or None (final tick attention)
        """
        B, T, d = z.shape
        tau = torch.exp(self.log_tau).view(1, 1, d)  # [1,1,d]

        pred_losses = []
        eff_ps = []
        last_attn = None

        for _ in range(self.n_ticks):
            h = self.pre_ln(self.in_proj(z))  # [B,T,d]

            Q = self.q_proj(h)  # [B,T,d]
            K = self.k_proj(h)
            V = self.v_proj(h)

            # Base attention
            sim = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)  # [B,T,T]
            attn = torch.softmax(sim, dim=-1)  # [B,T,T]

            # Attention peakiness → sync confidence
            row_max = attn.max(dim=-1).values  # [B,T]
            uniform = 1.0 / T
            conf = (row_max.mean(dim=-1) - uniform) / (1.0 - uniform + 1e-6)  # [B]
            conf = conf.clamp(0.0, 1.0)

            eff_p = self.p_min + (self.p_max - self.p_min) * conf  # [B]
            eff_ps.append(eff_p.detach())

            # Adaptive p: sharpen or flatten the logits
            eff_p_exp = eff_p.view(B, 1, 1)
            sim_sharp = sim * eff_p_exp
            attn_sharp = torch.softmax(sim_sharp, dim=-1)  # [B,T,T]
            last_attn = attn_sharp  # keep last tick's sharp attention

            context = torch.matmul(attn_sharp, V)  # [B,T,d]
            target = self.out_proj(context)        # [B,T,d]

            # Latent prediction: z_pred ≈ z_next
            z_pred = self.pred_net(z)  # [B,T,d]

            # LTC update: dz/dt = (target - z) / τ
            dz = (target - z) / (tau + 1e-6)
            z_next = z + self.dt * dz

            pred_loss = F.mse_loss(z_pred, z_next.detach())
            pred_losses.append(pred_loss)

            z = z_next

        pred_loss_mean = torch.stack(pred_losses).mean()
        eff_p_mean = torch.cat(eff_ps).mean()
        if return_attn:
            return z, pred_loss_mean, eff_p_mean.detach(), last_attn
        else:
            return z, pred_loss_mean, eff_p_mean.detach(), None


# ---------------------------------------------------------------------
# HIERARCHICAL CTM MODEL: 2 LAYERS (FAST + SLOW)
# ---------------------------------------------------------------------

class CTMHierModel(nn.Module):
    """
    2-layer CTM-ish model for hard binding:

    - Encodes 4 objects into slots.
    - Encodes query into a separate slot.
    - Total slots: 5 (4 objects + 1 query).
    - Layer 1: fast CTM (few ticks).
    - Layer 2: slow CTM (more ticks).
    - Reads out from query slot after slow layer.
    """

    def __init__(self, input_dim=52, slot_dim=32, n_slots=4, n_classes=4,
                 n_ticks_fast=2, n_ticks_slow=3):
        super().__init__()
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.n_slots = n_slots
        self.n_classes = n_classes

        # Each object chunk: color(4) + shape(4) + x + y = 10 dims
        self.obj_dim = 10
        assert 4 * self.obj_dim + 12 == input_dim, "input_dim mismatch"

        # Object encoder: 10 -> slot_dim
        self.slot_enc = nn.Sequential(
            nn.Linear(self.obj_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

        # Query encoder: 12 -> slot_dim
        self.query_enc = nn.Sequential(
            nn.Linear(12, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

        # Hierarchical CTM blocks
        self.ctm_fast = CTMBlock(slot_dim, n_slots + 1, n_ticks=n_ticks_fast)
        self.ctm_slow = CTMBlock(slot_dim, n_slots + 1, n_ticks=n_ticks_slow)

        # Per-slot shape head (object-wise logits)
        self.slot_shape_head = nn.Linear(slot_dim, n_classes)

    def encode_slots(self, x):
        """
        x: [B, 52]
        Returns:
          z0: [B, 5, slot_dim]
          slots 0..3: objects
          slot 4: query
        """
        B = x.shape[0]
        obj_feats = x[:, :4 * self.obj_dim].view(B, 4, self.obj_dim)  # [B,4,10]
        query_feats = x[:, 4 * self.obj_dim:]                          # [B,12]

        obj_emb = self.slot_enc(obj_feats)                             # [B,4,D]
        query_emb = self.query_enc(query_feats).unsqueeze(1)           # [B,1,D]

        z0 = torch.cat([obj_emb, query_emb], dim=1)                    # [B,5,D]
        return z0

    def forward(self, x, return_attn: bool = False):
        """
        x: [B, 52]
        Returns:
          logits: [B, n_classes]
          pred_loss: scalar
          eff_p_mean: scalar
          attn_q2s: [B, n_slots] or None (query→object attention)
        """
        z = self.encode_slots(x)
        z, pred_loss_fast, eff_p_fast, _ = self.ctm_fast(z, return_attn=False)
        # Always request attention from slow CTM to drive pointer-based readout
        z, pred_loss_slow, eff_p_slow, attn_last = self.ctm_slow(z, return_attn=True)

        pred_loss = pred_loss_fast + pred_loss_slow
        eff_p_mean = 0.5 * (eff_p_fast + eff_p_slow)
        slots = z[:, :self.n_slots, :]  # [B, n_slots, D]

        # Query→slot attention (from last slot row to the object slots)
        if attn_last is not None:
            q_idx = attn_last.shape[1] - 1
            obj_attn = attn_last[:, q_idx, :self.n_slots]  # [B, n_slots]
            attn_q2s = obj_attn / (obj_attn.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            attn_q2s = None

        # Per-slot logits
        slot_logits = self.slot_shape_head(slots)  # [B, n_slots, n_classes]

        # Pointer-style aggregation (log-space mix)
        log_p_slots = F.log_softmax(slot_logits, dim=-1)
        if attn_q2s is None:
            # Fallback uniform attention if missing (should not happen in normal use)
            attn_q2s = torch.full((slots.size(0), self.n_slots), 1.0 / self.n_slots, device=slots.device)
        log_attn = torch.log(attn_q2s.clamp_min(1e-8)).unsqueeze(-1)
        joint_log = log_p_slots + log_attn
        final_logits = torch.logsumexp(joint_log, dim=1)

        if not return_attn:
            attn_q2s = None

        return final_logits, pred_loss, eff_p_mean, attn_q2s


# ---------------------------------------------------------------------
# CLASSIFICATION EVALUATION
# ---------------------------------------------------------------------

def evaluate_model(model, loader, device):
    model.eval()
    correct = total = 0
    correct_simple = total_simple = 0
    correct_amb = total_amb = 0

    with torch.no_grad():
        for x, y, amb, cand_mask in loader:
            x = x.to(device)
            y = y.to(device)

            if isinstance(model, CTMHierModel):
                logits, _, _, _ = model(x)
            else:
                logits = model(x)

            preds = logits.argmax(dim=-1)

            correct += (preds == y).sum().item()
            total += y.numel()

            amb = amb.to(torch.bool)
            simple_mask = ~amb
            amb_mask = amb

            simple_idx = simple_mask.nonzero(as_tuple=False).squeeze(-1)
            amb_idx = amb_mask.nonzero(as_tuple=False).squeeze(-1)

            if simple_idx.numel() > 0:
                correct_simple += (preds[simple_idx] == y[simple_idx]).sum().item()
                total_simple += simple_idx.numel()

            if amb_idx.numel() > 0:
                correct_amb += (preds[amb_idx] == y[amb_idx]).sum().item()
                total_amb += amb_idx.numel()

    overall = correct / total if total else 0.0
    simple = correct_simple / total_simple if total_simple else 0.0
    amb = correct_amb / total_amb if total_amb else 0.0
    return overall, simple, amb


# ---------------------------------------------------------------------
# BINDING EVALUATION (query-slot attention → object slots)
# ---------------------------------------------------------------------

def evaluate_binding(ctm: CTMHierModel, loader, device, n_obj_slots: int = 4):
    """
    Probes whether the query slot's attention actually binds to the correct objects.

    For each sample:
      - Take attention from query slot row to the first n_obj_slots.
      - Binding-argmax is index of max attention among object slots.
      - If that index is in cand_mask (one of the valid candidates), count as success.

    Returns:
      (overall_acc, simple_acc, amb_acc, overall_mass_on_correct)
    """
    ctm.eval()
    correct = total = 0
    correct_simple = total_simple = 0
    correct_amb = total_amb = 0
    mass_sum = 0.0
    mass_count = 0

    with torch.no_grad():
        for x, y, amb, cand_mask in loader:
            x = x.to(device)
            amb = amb.to(torch.bool)
            cand_mask = cand_mask.to(device)  # [B,4]

            logits, _, _, attn_q2s = ctm(x, return_attn=True)
            if attn_q2s is None:
                continue

            obj_attn = attn_q2s[:, :n_obj_slots]  # [B,4]
            B = obj_attn.shape[0]

            # argmax binding
            max_idx = obj_attn.argmax(dim=-1)  # [B]
            batch_indices = torch.arange(B, device=device)
            correct_flags = cand_mask[batch_indices, max_idx]  # bool

            correct += correct_flags.sum().item()
            total += B

            # attention mass on ANY correct candidate(s)
            mass_on_correct = (obj_attn * cand_mask.float()).sum(dim=-1)  # [B]
            mass_sum += mass_on_correct.sum().item()
            mass_count += B

            # split simple vs amb
            simple_mask = ~amb
            amb_mask = amb

            if simple_mask.any():
                sm_idx = simple_mask.nonzero(as_tuple=False).squeeze(-1)
                correct_simple += correct_flags[sm_idx].sum().item()
                total_simple += sm_idx.numel()

            if amb_mask.any():
                am_idx = amb_mask.nonzero(as_tuple=False).squeeze(-1)
                correct_amb += correct_flags[am_idx].sum().item()
                total_amb += am_idx.numel()

    overall = correct / total if total else 0.0
    simple = correct_simple / total_simple if total_simple else 0.0
    amb = correct_amb / total_amb if total_amb else 0.0
    mean_mass = mass_sum / mass_count if mass_count else 0.0
    return overall, simple, amb, mean_mass


def compute_binding_loss(attn_q2s: torch.Tensor,
                         cand_mask: torch.Tensor,
                         amb: torch.Tensor,
                         eps: float = 1e-8) -> torch.Tensor:
    """
    Supervised binding loss for query→object attention.

    - Simple scenes: cross-entropy to the single correct slot.
    - Ambiguous scenes: soft target over all valid candidates.
    """
    if attn_q2s is None or not torch.is_tensor(attn_q2s):
        # If attention is unavailable (should be rare), skip binding supervision.
        # Use cand_mask device if available to keep the computation graph consistent.
        dev = cand_mask.device if torch.is_tensor(cand_mask) else None
        return torch.tensor(0.0, device=dev)

    cand_mask = cand_mask.to(attn_q2s.device)
    amb = amb.to(attn_q2s.device).bool()

    simple_mask = ~amb
    loss_terms = []

    if simple_mask.any():
        idx = simple_mask.nonzero(as_tuple=False).squeeze(-1)
        attn_simple = attn_q2s[idx]
        cand_simple = cand_mask[idx].float()
        target_idx = cand_simple.argmax(dim=-1)
        log_attn = (attn_simple + eps).log()
        ce_simple = F.nll_loss(log_attn, target_idx, reduction="mean")
        loss_terms.append(ce_simple)

    if amb.any():
        idx = amb.nonzero(as_tuple=False).squeeze(-1)
        attn_amb = attn_q2s[idx]
        cand_amb = cand_mask[idx].float()
        cand_sum = cand_amb.sum(dim=-1, keepdim=True)
        valid = cand_sum.squeeze(-1) > 0
        if valid.any():
            attn_amb = attn_amb[valid]
            cand_amb = cand_amb[valid]
            cand_soft = cand_amb / (cand_sum[valid] + eps)
            log_attn = (attn_amb + eps).log()
            ce_amb = -(cand_soft * log_attn).sum(dim=-1).mean()
            loss_terms.append(ce_amb)

    if not loss_terms:
        return torch.tensor(0.0, device=attn_q2s.device)

    return torch.stack(loss_terms).mean()


# ---------------------------------------------------------------------
# HARD BINDING EXPERIMENT WRAPPER
# ---------------------------------------------------------------------

def run_hard_binding_experiment(cfg: USMConfig):
    global device
    device = resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    print("=" * 60)
    print("USM v1.0: HARD BINDING + 2-LAYER CTM + ADAPTIVE p + LATENT PRED + BINDING PROBES")
    print("=" * 60)
    print("Device:", device)

    # Datasets
    train_ds = HardBindingDataset(
        n_samples=cfg.hb_train_samples, seed=cfg.seed, permute_objects=True
    )
    eval_ds = HardBindingDataset(
        n_samples=cfg.hb_eval_samples, seed=cfg.seed + 1, permute_objects=False
    )

    amb_train, amb_train_n = ambig_fraction(train_ds)
    amb_eval, amb_eval_n = ambig_fraction(eval_ds)

    print("\nHard Binding Task:")
    print(f"  Objects: 4")
    print(f"  Colors can REPEAT: True")
    print(f"  Input dim: 52")
    print(f"  Output dim: 4 (shapes)")
    print(f"  Ambiguous fraction (train): {amb_train*100:.1f}% ({amb_train_n} cases)")
    print(f"  Ambiguous fraction (eval):  {amb_eval*100:.1f}% ({amb_eval_n} cases)")

    # Show a few examples
    print("\n--- Examples ---\n")
    for objects, query, answer, amb in train_ds.sample_humans(3):
        qs = f"What shape is {query[0]} {query[1]} of {query[2]}?"
        print(f"Objects: {[(o['color'], o['shape'], o['x'], o['y']) for o in objects]}")
        print(f"Query: {qs}")
        print(f"Answer: {answer}")
        print(f"Has ambiguity: {amb}\n")

    train_loader = DataLoader(train_ds, batch_size=cfg.hb_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.hb_batch_size, shuffle=False)

    # -----------------------------------------------------------------
    # MODELS
    # -----------------------------------------------------------------
    mlp = MLPBaseline(input_dim=52, hidden=64, n_classes=4).to(device)
    ctm = CTMHierModel(input_dim=52, slot_dim=32, n_slots=4, n_classes=4,
                       n_ticks_fast=2, n_ticks_slow=3).to(device)

    n_params_ctm = sum(p.numel() for p in ctm.parameters())
    n_params_mlp = sum(p.numel() for p in mlp.parameters())
    print("----------------------------------------")
    print("MODELS")
    print("----------------------------------------")
    print(f"CTM: {n_params_ctm:,} params")
    print(f"MLP: {n_params_mlp:,} params")
    print(f"Ratio: {n_params_ctm / max(1,n_params_mlp):.2f}x")

    # -----------------------------------------------------------------
    # TRAIN MLP BASELINE
    # -----------------------------------------------------------------
    print("\n----------------------------------------")
    print(f"TRAINING MLP ({cfg.hb_epochs} epochs)")
    print("----------------------------------------")
    ce = nn.CrossEntropyLoss()
    opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=cfg.hb_lr)

    for epoch in range(1, cfg.hb_epochs + 1):
        mlp.train()
        for x, y, amb, cand_mask in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt_mlp.zero_grad()
            logits = mlp(x)
            loss = ce(logits, y)
            loss.backward()
            opt_mlp.step()

        if epoch % 20 == 0 or epoch == cfg.hb_epochs:
            acc, acc_simple, acc_amb = evaluate_model(mlp, eval_loader, device)
            print(f"Epoch {epoch:3d} | MLP | "
                  f"Eval: {acc*100:4.1f}% | "
                  f"Simple: {acc_simple*100:4.1f}% | "
                  f"Ambig: {acc_amb*100:4.1f}%")

    # -----------------------------------------------------------------
    # TRAIN CTM (with adaptive p + latent prediction loss)
    # -----------------------------------------------------------------
    print("\n----------------------------------------")
    print(f"TRAINING CTM ({cfg.hb_epochs} epochs) + Adaptive p + Latent Pred Loss")
    print("----------------------------------------")

    opt_ctm = torch.optim.AdamW(ctm.parameters(), lr=cfg.hb_lr)
    pred_loss_weight = cfg.hb_pred_loss_weight
    bind_loss_weight = cfg.hb_bind_loss_weight

    avg_eff_p_final = 0.0
    pred_loss_final = 0.0

    for epoch in range(1, cfg.hb_epochs + 1):
        ctm.train()
        running_eff_p = 0.0
        running_batches = 0
        last_pred_loss_val = 0.0
        last_bind_loss_val = 0.0
        attn_debug_stats: Optional[Tuple[float, float, float, bool]] = None

        for x, y, amb, cand_mask in train_loader:
            x = x.to(device)
            y = y.to(device)
            amb = amb.to(device)
            cand_mask = cand_mask.to(device)

            want_attn = attn_debug_stats is None and (epoch % 20 == 0)

            opt_ctm.zero_grad()
            logits, pred_loss, eff_p_mean, attn_q2s = ctm(x, return_attn=True)
            loss_cls = ce(logits, y)
            bind_loss = compute_binding_loss(attn_q2s, cand_mask, amb)
            loss = loss_cls + pred_loss_weight * pred_loss + bind_loss_weight * bind_loss
            loss.backward()
            opt_ctm.step()

            running_eff_p += float(eff_p_mean)
            running_batches += 1
            last_pred_loss_val = float(pred_loss.detach().item())
            last_bind_loss_val = float(bind_loss.detach().item())

            if want_attn and attn_q2s is not None:
                with torch.no_grad():
                    attn_sum = attn_q2s.sum(dim=-1)
                    attn_debug_stats = (
                        float(attn_q2s.min()),
                        float(attn_q2s.max()),
                        float(attn_sum.mean()),
                        bool(attn_q2s.requires_grad),
                    )

        if epoch % 20 == 0 or epoch == cfg.hb_epochs:
            acc, acc_simple, acc_amb = evaluate_model(ctm, eval_loader, device)
            avg_eff_p = running_eff_p / max(1, running_batches)
            if attn_debug_stats is not None:
                attn_min, attn_max, attn_sum_mean, attn_req_grad = attn_debug_stats
                print(f"  [AttnDbg] min={attn_min:.4f} max={attn_max:.4f} "
                      f"mean_sum={attn_sum_mean:.4f} requires_grad={attn_req_grad}")
            print(f"Epoch {epoch:3d} | CTM | "
                  f"Eval: {acc*100:4.1f}% | "
                  f"Simple: {acc_simple*100:4.1f}% | "
                  f"Ambig: {acc_amb*100:4.1f}% | "
                  f"PredLoss: {last_pred_loss_val:.4f} | "
                  f"BindLoss: {last_bind_loss_val:.4f} | "
                  f"eff_p: {avg_eff_p:.2f}")
            avg_eff_p_final = avg_eff_p
            pred_loss_final = last_pred_loss_val

    # -----------------------------------------------------------------
    # FINAL CLASSIFICATION COMPARISON
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DETAILED EVALUATION (CLASSIFICATION)")
    print("=" * 60)

    mlp_overall, mlp_simple, mlp_amb = evaluate_model(mlp, eval_loader, device)
    ctm_overall, ctm_simple, ctm_amb = evaluate_model(ctm, eval_loader, device)

    print("\nMLP Results:")
    print(f"  Overall: {mlp_overall*100:4.1f}%")
    print(f"  Simple (no ambiguity): {mlp_simple*100:4.1f}%")
    print(f"  AMBIGUOUS (hard): {mlp_amb*100:4.1f}%")

    print("\nCTM Results:")
    print(f"  Overall: {ctm_overall*100:4.1f}%")
    print(f"  Simple (no ambiguity): {ctm_simple*100:4.1f}%")
    print(f"  AMBIGUOUS (hard): {ctm_amb*100:4.1f}%")

    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPARISON")
    print("=" * 60)
    print("\nOVERALL:")
    print(f"  MLP: {mlp_overall*100:4.1f}%")
    print(f"  CTM: {ctm_overall*100:4.1f}%")
    print(f"  Diff: {(ctm_overall-mlp_overall)*100:4.1f}%")

    print("\nAMBIGUOUS CASES (TRUE BINDING TEST, via answers only):")
    print(f"  MLP: {mlp_amb*100:4.1f}%")
    print(f"  CTM: {ctm_amb*100:4.1f}%")
    print(f"  Diff: {(ctm_amb-mlp_amb)*100:4.1f}%")

    # -----------------------------------------------------------------
    # BINDING PROBES
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BINDING PROBES (QUERY ATTENTION → OBJECT SLOTS)")
    print("=" * 60)

    bind_overall, bind_simple, bind_amb, mass_mean = evaluate_binding(ctm, eval_loader, device, n_obj_slots=4)
    print(f"\nBinding argmax accuracy (query attention to correct object):")
    print(f"  Overall: {bind_overall*100:4.1f}%")
    print(f"  Simple:  {bind_simple*100:4.1f}%")
    print(f"  Ambig:   {bind_amb*100:4.1f}%")
    print(f"\nMean attention mass on correct candidate(s): {mass_mean*100:4.1f}%")

    print("\n" + "=" * 60)
    print("v1.0 HARD BINDING COMPLETE")
    print("=" * 60)

    results = {
        "world": "hard_binding",
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "ambig_fraction_train": amb_train,
        "ambig_fraction_eval": amb_eval,
        "mlp": {
            "eval_acc": mlp_overall,
            "simple_acc": mlp_simple,
            "ambig_acc": mlp_amb,
            "n_params": n_params_mlp,
        },
        "ctm": {
            "eval_acc": ctm_overall,
            "simple_acc": ctm_simple,
            "ambig_acc": ctm_amb,
            "n_params": n_params_ctm,
            "avg_eff_p": avg_eff_p_final,
            "pred_loss": pred_loss_final,
            "bind_loss": last_bind_loss_val,
        },
        "binding_probes": {
            "argmax_acc_overall": bind_overall,
            "argmax_acc_simple": bind_simple,
            "argmax_acc_ambig": bind_amb,
            "mean_correct_mass": mass_mean,
        },
    }
    return results


# ---------------------------------------------------------------------
# SIMPLE GRIDWORLD
# ---------------------------------------------------------------------

class GridWorld:
    """
    Simple 2D grid world with a single agent and a fixed goal.
    - Grid size: cfg.gw_size x cfg.gw_size
    - State: one-hot agent position + one-hot goal position.
    - Actions: 0=up, 1=down, 2=left, 3=right.
    - Reward:
        +1.0 on reaching goal,
        -0.01 per step otherwise.
    - Episode ends when goal reached or max_steps exceeded.
    """

    def __init__(self, size: int = 5):
        self.size = size
        self.n_actions = 4
        self.obs_dim = size * size * 2
        self.reset()

    def reset(self) -> torch.Tensor:
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        return self._get_obs()

    def _get_obs(self) -> torch.Tensor:
        agent_oh = torch.zeros(self.size * self.size)
        goal_oh = torch.zeros(self.size * self.size)
        agent_idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        goal_idx = self.goal_pos[0] * self.size + self.goal_pos[1]
        agent_oh[agent_idx] = 1.0
        goal_oh[goal_idx] = 1.0
        return torch.cat([agent_oh, goal_oh], dim=0)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        # Move agent
        if action == 0:   # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # down
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # right
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)

        reward = -0.01
        done = False
        if self.agent_pos == self.goal_pos:
            reward = 1.0
            done = True

        return self._get_obs(), reward, done


# ---------------------------------------------------------------------
# ETERNAL HYPERGRAPH (minimal)
# ---------------------------------------------------------------------

class NodeType(Enum):
    EXPERIENCE = "experience"
    GOAL = "goal"


@dataclass
class HyperNode:
    id: str
    node_type: NodeType
    embedding: torch.Tensor  # [D] on CPU
    data: Dict[str, Any]
    reward: float = 0.0
    created_at: int = 0


class EternalHypergraph:
    """
    Tiny experience-based world model:
    - Stores (z_t, a, z_{t+1}, r) as EXPERIENCE nodes.
    - Supports simple similarity-based retrieval over embeddings.
    - Tracks a global step counter.
    """

    def __init__(self, embedding_dim: int, max_nodes: int = 5000):
        self.embedding_dim = embedding_dim
        self.max_nodes = max_nodes
        self.nodes: Dict[str, HyperNode] = {}
        self.experience_ids: List[str] = []
        self.step = 0

    def add_experience(self, z: torch.Tensor, action: int,
                       z_next: torch.Tensor, reward: float) -> str:
        self.step += 1
        node_id = f"exp_{len(self.nodes)}_{self.step}"
        emb = z.detach().cpu().clone()
        node = HyperNode(
            id=node_id,
            node_type=NodeType.EXPERIENCE,
            embedding=emb,
            data={
                "z_next": z_next.detach().cpu().clone(),
                "action": int(action),
                "reward": float(reward),
            },
            reward=float(reward),
            created_at=self.step,
        )
        self.nodes[node_id] = node
        self.experience_ids.append(node_id)
        if len(self.experience_ids) > self.max_nodes:
            old_id = self.experience_ids.pop(0)
            self.nodes.pop(old_id, None)
        return node_id

    def retrieve_similar(self, z: torch.Tensor, k: int = 16) -> List[HyperNode]:
        if not self.experience_ids:
            return []
        ids = list(self.experience_ids)
        embs = torch.stack([self.nodes[i].embedding for i in ids], dim=0).to(z.device)
        z_norm = F.normalize(z.unsqueeze(0), dim=-1)
        embs_norm = F.normalize(embs, dim=-1)
        sims = torch.matmul(embs_norm, z_norm.transpose(0, 1)).squeeze(-1)  # [N]
        topk = torch.topk(sims, k=min(k, sims.numel()))
        return [self.nodes[ids[idx]] for idx in topk.indices.tolist()]

    def get_action_values(self, z: torch.Tensor, n_actions: int) -> torch.Tensor:
        if not self.experience_ids:
            return torch.zeros(n_actions, device=z.device)
        ids = list(self.experience_ids)
        embs = torch.stack([self.nodes[i].embedding for i in ids], dim=0).to(z.device)
        z_norm = F.normalize(z.unsqueeze(0), dim=-1)
        embs_norm = F.normalize(embs, dim=-1)
        sims = torch.matmul(embs_norm, z_norm.transpose(0, 1)).squeeze(-1)  # [N]
        q_vals = torch.zeros(n_actions, device=z.device)
        counts = torch.zeros(n_actions, device=z.device) + 1e-6
        for sim, node in zip(sims, [self.nodes[i] for i in ids]):
            a = node.data.get("action", 0)
            if 0 <= a < n_actions:
                q_vals[a] += sim * node.data.get("reward", 0.0)
                counts[a] += sim.abs()
        return q_vals / counts


# ---------------------------------------------------------------------
# ACTIVE INFERENCE AGENT (minimal)
# ---------------------------------------------------------------------

class ActiveInferenceAgent(nn.Module):
    """
    Lightweight Active Inference agent:
    - Latent belief z ∈ R^D comes from a CTM encoder.
    - Has a small neural transition model f(z, a).
    - Uses EternalHypergraph as a memory-based world model.
    - Selects actions by minimizing a simple free energy objective.
    """

    def __init__(self, latent_dim: int, n_actions: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions = n_actions

        self.transition = nn.Sequential(
            nn.Linear(latent_dim + n_actions, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Learnable weights for combining neural vs memory predictions
        self.w_reward = nn.Parameter(torch.tensor(1.0))
        self.w_memory = nn.Parameter(torch.tensor(0.5))

    def forward_transition(self, z: torch.Tensor, action: int) -> torch.Tensor:
        a_onehot = torch.zeros(self.n_actions, device=z.device)
        a_onehot[action] = 1.0
        inp = torch.cat([z, a_onehot], dim=-1)
        return self.transition(inp)

    def select_action(self, z: torch.Tensor,
                      hypergraph: EternalHypergraph,
                      temperature: float = 1.0,
                      deterministic: bool = False) -> Tuple[int, Dict[str, Any]]:
        memory_q = hypergraph.get_action_values(z.detach(), self.n_actions)
        scores = []
        for a in range(self.n_actions):
            # simple transition usage; reward estimate is zero here
            score = self.w_reward * 0.0 + self.w_memory * memory_q[a]
            scores.append(score)
        scores_t = torch.stack(scores)
        if deterministic:
            action = int(scores_t.argmax().item())
            probs = torch.zeros_like(scores_t)
            probs[action] = 1.0
        else:
            probs = torch.softmax(scores_t / temperature, dim=-1)
            action = int(torch.multinomial(probs, 1).item())
        return action, {"scores": scores_t.detach().cpu(), "probs": probs.detach().cpu(), "q_memory": memory_q.detach().cpu()}


# ---------------------------------------------------------------------
# USM AGENT FOR GRIDWORLD
# ---------------------------------------------------------------------

class USMGridWorldAgent(nn.Module):
    """
    USM agent for GridWorld:
    - Encodes observations to a latent z using a small MLP + CTMBlock.
    - Uses ActiveInferenceAgent to choose actions given z and EternalHypergraph memories.
    - Includes a small Q head for TD learning.
    """

    def __init__(self, obs_dim: int, latent_dim: int, n_actions: int):
        super().__init__()
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.ctm = CTMBlock(dim=latent_dim, n_slots=1, n_ticks=1)
        self.active = ActiveInferenceAgent(latent_dim, n_actions)
        self.q_head = nn.Linear(latent_dim, n_actions)

    def encode_latent(self, obs: torch.Tensor) -> torch.Tensor:
        z0 = self.obs_enc(obs.unsqueeze(0))  # [1,D]
        z_seq = z0.unsqueeze(1)  # [1,1,D]
        z_out, _, _, _ = self.ctm(z_seq, return_attn=False)
        return z_out.squeeze(1).squeeze(0)

    def act(self, z: torch.Tensor, hypergraph: EternalHypergraph,
            temperature: float = 1.0, deterministic: bool = False) -> Tuple[int, Dict[str, Any]]:
        return self.active.select_action(z, hypergraph, temperature, deterministic)


# ---------------------------------------------------------------------
# GRIDWORLD EXPERIMENT
# ---------------------------------------------------------------------

def run_gridworld_experiment(cfg: USMConfig):
    global device
    device = resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    print("=" * 60)
    print("USM v1.0: GRIDWORLD + HYPERGRAPH + ACTIVE INFERENCE")
    print("=" * 60)
    print("Device:", device)

    env = GridWorld(size=cfg.gw_size)
    agent = USMGridWorldAgent(env.obs_dim, cfg.gw_latent_dim, env.n_actions).to(device)
    hypergraph = EternalHypergraph(embedding_dim=cfg.gw_latent_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.gw_lr)
    gamma = cfg.gw_gamma

    def tensorize(obs: torch.Tensor) -> torch.Tensor:
        return obs.to(device).float()

    reward_history = []
    success_history = []

    for ep in range(1, cfg.gw_n_episodes + 1):
        obs = env.reset()
        total_r = 0.0
        success = False
        for step in range(cfg.gw_max_steps):
            obs_t = tensorize(obs)
            z = agent.encode_latent(obs_t)
            action, info = agent.act(z, hypergraph, temperature=1.0)
            next_obs, reward, done = env.step(action)
            total_r += reward
            if done and reward > 0:
                success = True

            next_obs_t = tensorize(next_obs)
            z_next = agent.encode_latent(next_obs_t)

            # Memory insertion
            hypergraph.add_experience(z, action, z_next, reward)

            # Simple TD(0) on Q head
            q_pred = agent.q_head(z)
            q_val = q_pred[action]
            with torch.no_grad():
                q_next = agent.q_head(z_next)
                target = reward + (0.0 if done else gamma * q_next.max().item())
            loss_td = F.mse_loss(q_val, torch.tensor(target, device=device))

            optimizer.zero_grad()
            loss_td.backward()
            optimizer.step()

            obs = next_obs
            if done:
                break

        reward_history.append(total_r)
        success_history.append(1.0 if success else 0.0)

        if ep % 20 == 0 or ep == cfg.gw_n_episodes:
            avg_r = sum(reward_history[-20:]) / min(20, len(reward_history))
            avg_succ = sum(success_history[-20:]) / min(20, len(success_history))
            print(f"Episode {ep:3d} | avg_reward(last20)={avg_r:.3f} | success_rate(last20)={avg_succ*100:4.1f}% | mem_size={len(hypergraph.experience_ids)}")

    print("\n" + "=" * 60)
    print("GRIDWORLD TRAINING COMPLETE")
    print("=" * 60)

    last_k = min(100, len(reward_history))
    success_rate_last_k = sum(success_history[-last_k:]) / max(1, last_k)
    avg_return_last_k = sum(reward_history[-last_k:]) / max(1, last_k)

    results = {
        "world": "gridworld",
        "n_episodes": cfg.gw_n_episodes,
        "max_steps": cfg.gw_max_steps,
        "success_rate_last_k": success_rate_last_k,
        "avg_return_last_k": avg_return_last_k,
        "hypergraph": {
            "num_experiences": len(hypergraph.experience_ids),
        },
    }
    return results


# ---------------------------------------------------------------------
# POLYMARKET STUB
# ---------------------------------------------------------------------

def run_polymarket_stub(cfg: USMConfig) -> dict:
    print("\n" + "=" * 60)
    print("USM: POLYMARKET STUB")
    print("=" * 60)
    print("This is a placeholder for a real PolymarketWorld.")
    print(f"CSV path: {cfg.polymarket_csv_path}")
    print(f"use_synthetic: {cfg.polymarket_use_synthetic}")
    print("For now, this just returns a dummy metrics dict.\n")

    metrics = {
        "world": "polymarket_stub",
        "csv_path": cfg.polymarket_csv_path,
        "use_synthetic": cfg.polymarket_use_synthetic,
        "status": "stub",
    }
    return metrics


# ---------------------------------------------------------------------
# UNIFIED USM CELL RUNNER
# ---------------------------------------------------------------------

def run_usm_cell(cfg: USMConfig) -> dict:
    """
    Unified USM cell runner.

    Depending on cfg.world_type, runs:
      - 'hard_binding': hard binding CTM vs MLP experiment
      - 'gridworld': gridworld USM experiment
      - 'polymarket_stub': polymarket stub
      - 'all': runs all three and aggregates metrics

    Returns:
        A dict with per-world metrics.
    """
    set_global_seed(cfg.seed)

    all_results: Dict[str, Any] = {}

    if cfg.world_type in ("hard_binding", "all"):
        hb_results = run_hard_binding_experiment(cfg)
        all_results["hard_binding"] = hb_results

    if cfg.world_type in ("gridworld", "all"):
        gw_results = run_gridworld_experiment(cfg)
        all_results["gridworld"] = gw_results

    if cfg.world_type in ("polymarket_stub", "all"):
        pm_results = run_polymarket_stub(cfg)
        all_results["polymarket_stub"] = pm_results

    return all_results


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_type", type=str, default="hard_binding",
                        choices=["hard_binding", "gridworld", "polymarket_stub", "all"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = USMConfig(world_type=args.world_type, seed=args.seed)

    print(f"Device: {device}")
    print("=" * 60)
    print("USM v0.9: Unified USM Cell (Hard Binding + GridWorld + Polymarket Stub)")
    print("=" * 60)

    results = run_usm_cell(cfg)

    print("\n" + "=" * 60)
    print("USM SUMMARY")
    print("=" * 60)
    for world_name, metrics in results.items():
        print(f"\n[{world_name}]")
        for k, v in metrics.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("USM v0.9 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
