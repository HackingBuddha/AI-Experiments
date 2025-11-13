import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReGraMMConfig:
    img_size: int = 32
    img_channels: int = 3
    num_slots: int = 6
    slot_dim: int = 48
    slot_iters: int = 3
    num_ts_entities: int = 6
    ts_input_dim: int = 4
    ts_embed_dim: int = 48
    node_dim: int = 48
    edge_hidden_dim: int = 64
    pr_state_dim: int = 48
    controller_hidden_dim: int = 96
    controller_num_actions: int = 8
    num_op_types: int = 4
    max_nodes: int = 32
    support_size: int = 4
    task_embed_dim: int = 96
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SlotAttention(nn.Module):
    def __init__(self, num_slots: int, dim: int, iters: int = 3,
                 eps: float = 1e-8, hidden_dim: Optional[int] = None):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        if hidden_dim is None:
            hidden_dim = dim

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))

        self.project_q = nn.Linear(dim, dim, bias=False)
        self.project_k = nn.Linear(dim, dim, bias=False)
        self.project_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_mlp = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x = self.norm_input(x)

        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        k = self.project_k(x)
        v = self.project_v(x)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm)

            attn_logits = torch.einsum("bsd,bnd->bsn", q, k) * self.scale
            attn = attn_logits.softmax(dim=-1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bnd,bsn->bsd", v, attn)

            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D),
            )
            slots = slots.view(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_pre_mlp(slots))

        return slots


class VisionSlotEncoder(nn.Module):
    def __init__(self, cfg: ReGraMMConfig):
        super().__init__()
        C = cfg.img_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, cfg.slot_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )
        self.slot_attention = SlotAttention(
            num_slots=cfg.num_slots,
            dim=cfg.slot_dim,
            iters=cfg.slot_iters,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        feat = self.cnn(x)
        B, D, H, W = feat.shape
        tokens = feat.view(B, D, H * W).transpose(1, 2)
        slots = self.slot_attention(tokens)
        return slots


class TimeSeriesStepEncoder(nn.Module):
    def __init__(self, cfg: ReGraMMConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.ts_input_dim, cfg.ts_embed_dim)
        self.to_node = nn.Linear(cfg.ts_embed_dim, cfg.node_dim)

    def forward(self, ts_step: torch.Tensor) -> torch.Tensor:
        x = self.proj(ts_step)
        x = F.relu(x, inplace=True)
        nodes = self.to_node(x)
        return nodes


class LiquidSlotCell(nn.Module):
    def __init__(self, node_dim: int, state_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.state_dim = state_dim

        self.in_proj = nn.Linear(node_dim, state_dim)
        self.f_mlp = nn.Sequential(
            nn.Linear(state_dim + state_dim, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, state_dim),
        )
        self.time_mlp = nn.Linear(state_dim + state_dim, state_dim)
        self.out_proj = nn.Linear(state_dim, node_dim)
        self.log_dt = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        node_feats: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = node_feats.shape
        _, _, S = state.shape
        assert S == self.state_dim

        x_state = self.in_proj(node_feats)
        sx = torch.cat([state, x_state], dim=-1)

        ds = self.f_mlp(sx)
        raw_tau = self.time_mlp(sx)
        tau = F.softplus(raw_tau) + 1e-3
        dt = torch.exp(self.log_dt)
        s_new = state + (dt * ds / tau)
        delta_node = self.out_proj(s_new)
        new_feats = node_feats + delta_node

        return new_feats, s_new


class GraphDynamics(nn.Module):
    def __init__(self, cfg: ReGraMMConfig):
        super().__init__()
        D = cfg.node_dim
        H = cfg.edge_hidden_dim

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * D, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, D),
            nn.ReLU(inplace=True),
        )

        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * D, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, D),
        )

        self.norm_nodes = nn.LayerNorm(D)

    def forward(
        self,
        nodes: torch.Tensor,
        node_mask: torch.Tensor
    ) -> torch.Tensor:
        B, N, D = nodes.shape
        mask = node_mask.unsqueeze(-1)
        x = self.norm_nodes(nodes) * mask

        x_i = x.unsqueeze(2).expand(B, N, N, D)
        x_j = x.unsqueeze(1).expand(B, N, N, D)

        msg_input = torch.cat([x_i, x_j], dim=-1)
        msgs = self.msg_mlp(msg_input)

        send_mask = mask.unsqueeze(1)
        msgs = msgs * send_mask

        agg = msgs.sum(dim=2)

        upd_input = torch.cat([nodes, agg], dim=-1)
        new_nodes = nodes + self.upd_mlp(upd_input)

        new_nodes = new_nodes * mask
        return new_nodes


class WorldGraphState:
    def __init__(
        self,
        node_feats: torch.Tensor,
        slot_state: torch.Tensor,
        node_mask: torch.Tensor,
    ):
        self.node_feats = node_feats
        self.slot_state = slot_state
        self.node_mask = node_mask


class WorldGraph(nn.Module):
    def __init__(self, cfg: ReGraMMConfig):
        super().__init__()
        self.cfg = cfg
        self.total_nodes = cfg.num_slots + cfg.num_ts_entities

        self.liquid_cell = LiquidSlotCell(cfg.node_dim, cfg.pr_state_dim)
        self.dynamics = GraphDynamics(cfg)

    def init_state(self, batch_size: int, device: Optional[str] = None) -> WorldGraphState:
        if device is None:
            device = self.cfg.device
        node_feats = torch.zeros(batch_size, self.total_nodes, self.cfg.node_dim, device=device)
        slot_state = torch.zeros(batch_size, self.total_nodes, self.cfg.pr_state_dim, device=device)
        node_mask = torch.ones(batch_size, self.total_nodes, device=device)
        return WorldGraphState(node_feats=node_feats, slot_state=slot_state, node_mask=node_mask)

    def forward_step(
        self,
        vision_slots: torch.Tensor,
        ts_nodes: torch.Tensor,
        prev_state: WorldGraphState,
    ) -> WorldGraphState:
        B, S, D = vision_slots.shape
        B2, N_ts, D2 = ts_nodes.shape
        assert B == B2 and D == D2
        assert S == self.cfg.num_slots and N_ts == self.cfg.num_ts_entities

        nodes = torch.cat([vision_slots, ts_nodes], dim=1)
        assert nodes.shape[1] == self.total_nodes

        node_mask = prev_state.node_mask
        slot_state = prev_state.slot_state

        node_feats_liq, new_slot_state = self.liquid_cell(nodes, slot_state)
        node_feats_dyn = self.dynamics(node_feats_liq, node_mask=node_mask)
        node_feats_dyn = node_feats_dyn * node_mask.unsqueeze(-1)

        return WorldGraphState(node_feats=node_feats_dyn,
                               slot_state=new_slot_state,
                               node_mask=node_mask)

    def apply_soft_mask(
        self,
        state: WorldGraphState,
        soft_mask: torch.Tensor,
        combine: str = "mul",
    ) -> WorldGraphState:
        assert soft_mask.shape == state.node_mask.shape

        soft_mask_clamped = soft_mask.clamp(0.0, 1.0)

        if combine == "mul":
            node_mask = state.node_mask * soft_mask_clamped
        elif combine == "override":
            node_mask = soft_mask_clamped
        else:
            raise ValueError(f"Unknown combine mode: {combine}")

        node_feats = state.node_feats * node_mask.unsqueeze(-1)
        slot_state = state.slot_state * node_mask.unsqueeze(-1)

        return WorldGraphState(node_feats=node_feats,
                               slot_state=slot_state,
                               node_mask=node_mask)


class GraphICLController(nn.Module):
    def __init__(self, cfg: ReGraMMConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.node_dim
        H = cfg.controller_hidden_dim
        self.n_total = cfg.num_slots + cfg.num_ts_entities

        self.graph_readout = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, H),
            nn.ReLU(inplace=True),
        )

        self.mlp = nn.Sequential(
            nn.Linear(H + cfg.task_embed_dim, H),
            nn.ReLU(inplace=True),
        )

        self.label_head = nn.Linear(H, 1)
        self.node_keep_head = nn.Linear(H, self.n_total)

    def forward(
        self,
        world_state: WorldGraphState,
        task_embed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        nodes = world_state.node_feats
        mask = world_state.node_mask

        mask_exp = mask.unsqueeze(-1)
        denom = mask_exp.sum(dim=1).clamp(min=1.0)
        g = (nodes * mask_exp).sum(dim=1) / denom

        h_g = self.graph_readout(g)

        h = torch.cat([h_g, task_embed], dim=-1)
        h = self.mlp(h)

        label_pred = self.label_head(h)
        node_keep_logits = self.node_keep_head(h)

        return {
            "label_pred": label_pred,
            "node_keep_logits": node_keep_logits,
        }


class TaskEncoder(nn.Module):
    def __init__(self, cfg: ReGraMMConfig):
        super().__init__()
        D = cfg.node_dim
        H = cfg.task_embed_dim

        self.graph_readout = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, H),
            nn.ReLU(inplace=True),
        )

        self.label_proj = nn.Linear(1, H)
        self.combine = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, H),
        )

    def encode_single(self, state: WorldGraphState, label: torch.Tensor) -> torch.Tensor:
        nodes = state.node_feats
        mask = state.node_mask

        mask_exp = mask.unsqueeze(-1)
        denom = mask_exp.sum(dim=1).clamp(min=1.0)
        g = (nodes * mask_exp).sum(dim=1) / denom

        h_g = self.graph_readout(g)
        h_y = self.label_proj(label)

        h = torch.cat([h_g, h_y], dim=-1)
        h = self.combine(h)
        return h

    def forward(self, states, labels: torch.Tensor) -> torch.Tensor:
        B, K, _ = labels.shape

        embeds = []
        for k in range(K):
            state_k = states[k]
            label_k = labels[:, k, :]
            h_k = self.encode_single(state_k, label_k)
            embeds.append(h_k)

        stacked = torch.stack(embeds, dim=1)
        task_embed = stacked.mean(dim=1)
        return task_embed


class ReGraMMICL(nn.Module):
    def __init__(self, cfg: ReGraMMConfig):
        super().__init__()
        self.cfg = cfg

        self.vision_encoder = VisionSlotEncoder(cfg)
        self.ts_encoder = TimeSeriesStepEncoder(cfg)
        self.world_graph = WorldGraph(cfg)
        self.controller = GraphICLController(cfg)
        self.task_encoder = TaskEncoder(cfg)

    def init_world_state(self, batch_size: int, device: Optional[str] = None) -> WorldGraphState:
        return self.world_graph.init_state(batch_size, device)

    def encode_sequence_to_state(
        self,
        img_seq: torch.Tensor,
        ts_seq: torch.Tensor,
    ) -> WorldGraphState:
        B, T, C, H, W = img_seq.shape
        B2, T2, N_ts, F = ts_seq.shape
        assert B == B2 and T == T2
        assert N_ts == self.cfg.num_ts_entities and F == self.cfg.ts_input_dim

        device = img_seq.device
        state = self.world_graph.init_state(B, device)

        for t in range(T):
            imgs_t = img_seq[:, t]
            ts_t = ts_seq[:, t]

            vision_slots = self.vision_encoder(imgs_t)
            ts_nodes = self.ts_encoder(ts_t)
            state = self.world_graph.forward_step(vision_slots, ts_nodes, state)

        return state

    def forward_icltask(
        self,
        img_support: torch.Tensor,
        ts_support: torch.Tensor,
        y_support: torch.Tensor,
        img_query: torch.Tensor,
        ts_query: torch.Tensor,
        combine_mode: str = "mul",
    ) -> Dict[str, torch.Tensor]:
        B, K, T, C, H, W = img_support.shape

        support_states = []
        for k in range(K):
            img_s = img_support[:, k, :, :, :, :]
            ts_s = ts_support[:, k, :, :, :]
            state_s = self.encode_sequence_to_state(img_s, ts_s)
            support_states.append(state_s)

        task_embed = self.task_encoder(support_states, y_support)
        query_state = self.encode_sequence_to_state(img_query, ts_query)
        ctrl_out = self.controller(query_state, task_embed=task_embed)

        node_keep = torch.sigmoid(ctrl_out["node_keep_logits"])
        query_state_masked = self.world_graph.apply_soft_mask(
            query_state,
            node_keep,
            combine=combine_mode,
        )

        ctrl_out_masked = self.controller(query_state_masked, task_embed=task_embed)

        return {
            "y_pred": ctrl_out_masked["label_pred"],
            "node_keep_logits": ctrl_out_masked["node_keep_logits"],
            "query_state": query_state_masked,
        }


def sample_icltask_batch(cfg: ReGraMMConfig, batch_size: int, seq_len: int, device):
    B = batch_size
    K = cfg.support_size
    N_ts = cfg.num_ts_entities
    F = cfg.ts_input_dim
    C = cfg.img_channels
    H = W = cfg.img_size
    T = seq_len

    rel_mask = torch.zeros(B, N_ts, device=device)
    for b in range(B):
        num_rel = torch.randint(1, N_ts + 1, (1,), device=device).item()
        idx = torch.randperm(N_ts, device=device)[:num_rel]
        rel_mask[b, idx] = 1.0

    def gen_seq(B_mult):
        ts_seq = torch.zeros(B_mult, T, N_ts, F, device=device)
        img_seq = torch.zeros(B_mult, T, C, H, W, device=device)

        latent = torch.randn(B_mult, N_ts, 1, device=device) * 0.1
        for t in range(T):
            latent = latent + 0.1 * torch.randn_like(latent)
            ts_features = latent.expand(-1, -1, F) + 0.05 * torch.randn(B_mult, N_ts, F, device=device)
            ts_seq[:, t] = ts_features

        img_seq = torch.randn(B_mult, T, C, H, W, device=device) * 0.1
        brightness = ts_seq.mean(dim=(2, 3))
        brightness = brightness.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        img_seq = img_seq + brightness

        return img_seq, ts_seq

    img_support = torch.zeros(B, K, T, C, H, W, device=device)
    ts_support = torch.zeros(B, K, T, N_ts, F, device=device)
    y_support = torch.zeros(B, K, 1, device=device)

    for k in range(K):
        img_s, ts_s = gen_seq(B)
        img_support[:, k] = img_s
        ts_support[:, k] = ts_s

        ts_last = ts_s[:, -1]
        rel_sum = (ts_last * rel_mask.unsqueeze(-1)).sum(dim=1)
        rel_count = rel_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        rel_mean = rel_sum / rel_count
        y = rel_mean.mean(dim=1, keepdim=True)
        y_support[:, k, :] = y

    img_query, ts_query = gen_seq(B)
    ts_last_q = ts_query[:, -1]
    rel_sum_q = (ts_last_q * rel_mask.unsqueeze(-1)).sum(dim=1)
    rel_count_q = rel_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    rel_mean_q = rel_sum_q / rel_count_q
    y_query = rel_mean_q.mean(dim=1, keepdim=True)

    return (
        img_support, ts_support, y_support,
        img_query, ts_query, y_query,
        rel_mask,
    )


if __name__ == "__main__":
    cfg = ReGraMMConfig()
    device = cfg.device

    model = ReGraMMICL(cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    num_steps = 2
    batch_size = 2
    seq_len = 4

    print("Starting ICL training (Liquid ReGraMM + graph soft-ops)...")
    model.train()

    for step in range(num_steps):
        (
            img_support, ts_support, y_support,
            img_query, ts_query, y_query,
            rel_mask,
        ) = sample_icltask_batch(cfg, batch_size, seq_len, device)

        out = model.forward_icltask(
            img_support, ts_support, y_support,
            img_query, ts_query,
            combine_mode="mul",
        )

        y_pred = out["y_pred"]
        node_keep_logits = out["node_keep_logits"]

        loss_query = F.mse_loss(y_pred, y_query)

        N_total = cfg.num_slots + cfg.num_ts_entities
        ts_mask_logits = node_keep_logits[:, cfg.num_slots:]

        loss_mask = F.binary_cross_entropy_with_logits(ts_mask_logits, rel_mask)

        loss = loss_query + 0.1 * loss_mask

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 1 == 0:
            with torch.no_grad():
                mse_val = loss_query.item()
                mask_bce = loss_mask.item()
            print(f"step {step+1:04d} | query-MSE {mse_val:.5f} | mask-BCE {mask_bce:.5f}")

    print("Training finished.")
