"""BIND model: DINOv3 backbone + N-view discretized world-space volume head.

Self-contained extraction of ``DinoVolumeSceneV4`` (the volumetric action head
used in the paper). Per future waypoint it predicts a probability over a
discretized N-view world voxel grid (argmax = the end-effector target), plus
gripper and rotation-bin logits read from the chosen voxel.

DINOv3 weights are NOT bundled (Meta license). Download the ViT-S/16+ checkpoint
from https://github.com/facebookresearch/dinov3 and point DINOV3_WEIGHTS at it:

    export DINOV3_WEIGHTS=/path/to/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
    # optional: a local clone of the dinov3 repo (else pulled from torch hub)
    export DINOV3_REPO=/path/to/dinov3
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_dino_backbone():
    """Load the DINOv3 ViT-S/16+ backbone from env config (see module docstring)."""
    weights = os.environ.get("DINOV3_WEIGHTS")
    if not weights:
        raise RuntimeError(
            "Set DINOV3_WEIGHTS to a downloaded DINOv3 ViT-S/16+ checkpoint "
            "(https://github.com/facebookresearch/dinov3)."
        )
    model_name = os.environ.get("DINOV3_MODEL", "dinov3_vits16plus")
    repo = os.environ.get("DINOV3_REPO")
    if repo:
        return torch.hub.load(repo, model_name, source="local", weights=weights)
    return torch.hub.load("facebookresearch/dinov3", model_name, weights=weights)


def sin_pe(n: int, d: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding. (n, d)."""
    pe = torch.zeros(n, d)
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32)
                    * -(math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class AdaLNZeroBlock(nn.Module):
    """LayerNorm → scale/shift from cond → MLP → gate from cond. Residual."""

    def __init__(self, d_model: int, d_cond: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.modulation = nn.Linear(d_cond, 3 * d_model)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift, gate = self.modulation(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + gate * self.mlp(h)


class FiveLayerCNN(nn.Module):
    """Modest 5-layer 3×3 CNN, same in/out channels, GELU between layers.

    Last conv ungated (no activation) so the output is unbounded — matches
    the production ``F_scene`` distribution that the factored scoring
    expects. Plain (non-residual) since 5 layers is shallow enough that
    vanishing gradients aren't a concern.
    """

    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel, padding=pad), nn.GELU(),
            nn.Conv2d(channels, channels, kernel, padding=pad), nn.GELU(),
            nn.Conv2d(channels, channels, kernel, padding=pad), nn.GELU(),
            nn.Conv2d(channels, channels, kernel, padding=pad), nn.GELU(),
            nn.Conv2d(channels, channels, kernel, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DinoHiresBase(nn.Module):
    """Shared DINOv3 backbone + 1×1 conv to ``feat_dim`` + bilinear up to
    ``pred_size`` + 5-layer CNN refinement. Subclasses add the head-specific
    final stage (factorized scoring vs per-voxel expansion + scoring).
    """

    def __init__(
        self,
        n_window: int,
        n_height_bins: int,
        pred_size: int,
        feat_dim: int,
        d_model: int,
        d_cond: int,
        d_sin_z: int,
        d_sin_t: int,
        n_blocks: int,
        n_gripper_bins: int,
        n_rot_clusters: int,
        dino_repo: str = None,
        dino_weights: str = None,
    ):
        super().__init__()
        self.n_window = n_window
        self.n_z = n_height_bins
        self.pred_size = pred_size
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.d_sin_z = d_sin_z
        self.d_sin_t = d_sin_t
        self.n_gripper_bins = n_gripper_bins
        self.n_rot_clusters = n_rot_clusters

        self.backbone = _load_dino_backbone()
        embed_dim = self.backbone.embed_dim
        self.embed_dim = embed_dim

        # Per-pixel refinement stage: 1×1 conv to feat_dim → bilinear up
        # to pred_size → 5-layer CNN. (Bilinear up + CNN replaces the
        # production ``ResidualMLP2`` and bumps pred_size 128→256.)
        self.feat_head = nn.Conv2d(embed_dim, feat_dim, 1)
        self.refine_cnn = FiveLayerCNN(feat_dim, kernel=3)

        # Sin/cos PEs (unchanged from production).
        self.register_buffer("z_sin", sin_pe(n_height_bins, d_sin_z))
        self.register_buffer("t_sin", sin_pe(n_window, d_sin_t))

        # Input projection: (eef_feat ⊕ cls) → d_model.
        self.input_proj = nn.Linear(feat_dim + embed_dim, d_model)
        self.t_cond_proj = nn.Linear(d_sin_t, d_cond)
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(d_model, d_cond, mlp_ratio=4)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Subclasses override d_q logic + add scoring stage.
        self.grip_head = nn.Linear(d_model, n_gripper_bins)
        self.rot_head = nn.Linear(d_model, n_rot_clusters)

    def patch_features(self, rgb: torch.Tensor):
        """rgb (B, 3, S, S) → (patch_grid (B, embed, P_dino, P_dino), cls (B, embed))."""
        out = self.backbone.forward_features(rgb)
        cls = out["x_norm_clstoken"]
        toks = out["x_norm_patchtokens"]
        P = int(round(math.sqrt(toks.shape[1])))
        return toks.transpose(1, 2).reshape(rgb.size(0), -1, P, P), cls

    def _pixel_features(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """rgb → (F_scene (B, feat_dim, P, P), patch (raw DINOv3 grid), cls)."""
        patch, cls = self.patch_features(rgb)
        F0 = self.feat_head(patch)
        F1 = F.interpolate(F0, size=(self.pred_size, self.pred_size),
                              mode="bilinear", align_corners=False)
        F_scene = self.refine_cnn(F1)              # (B, feat_dim, P, P)
        return F_scene, patch, cls


class _ViTBlock(nn.Module):
    """Pre-norm transformer block: norm → multi-head self-attention → norm → MLP.

    The attention call uses ``F.scaled_dot_product_attention`` (flash on
    H100 / A100 when available). No relative position bias — we lean on
    the DINO patch features already carrying spatial structure.
    """

    def __init__(self, d: int, n_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.proj = nn.Linear(d, d, bias=True)
        self.norm2 = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim) \
                            .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # (B, H, L, D/H)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossViewAttnStack(nn.Module):
    """DA3-style alternating within-view / cross-view self-attention over
    the native DINO patch grid (~31×31 per view).

    Each block is a plain ViT block. Layer i is cross-view if
    ``i % 2 == 0`` else within-view — same "token reordering" idea
    Depth-Anything-3 (arXiv:2511.10647) uses to share information across
    views without introducing a separate cross-attention module.
    """

    def __init__(self, n_layers: int, embed_dim: int, n_heads: int = 6,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.n_layers = n_layers
        self.blocks = nn.ModuleList(
            [_ViTBlock(embed_dim, n_heads, mlp_ratio) for _ in range(n_layers)])
        # is_cross[i] toggles which axis attention spans. Start with cross
        # so view-context lands early, then alternate.
        self.is_cross = [i % 2 == 0 for i in range(n_layers)]

    def forward(self, patches: list[torch.Tensor]) -> list[torch.Tensor]:
        """Input/output: list of (B, D, P, P) patch grids, one per view."""
        N = len(patches)
        if N == 0:
            return patches
        B, D, P, _ = patches[0].shape
        # Stack into (B, N, P*P, D).
        x = torch.stack(
            [p.flatten(2).transpose(1, 2) for p in patches], dim=1)
        for i, blk in enumerate(self.blocks):
            if self.is_cross[i]:
                x = blk(x.reshape(B, N * P * P, D)).reshape(B, N, P * P, D)
            else:
                x = blk(x.reshape(B * N, P * P, D)).reshape(B, N, P * P, D)
        # Unstack back to per-view (B, D, P, P).
        return [x[:, k].transpose(1, 2).reshape(B, D, P, P) for k in range(N)]


def _v4_per_view_bin_centers(z_lo: float, z_hi_b: torch.Tensor, Z: int) -> torch.Tensor:
    """Per-batch bin centres for a view whose z-range upper bound varies
    per sample (e.g. wrist cam moves with the EE). Returns ``(B, Z)`` in m.
    """
    device, dtype = z_hi_b.device, z_hi_b.dtype
    idx = (torch.arange(Z, device=device, dtype=dtype) + 0.5) / Z      # (Z,)
    z_lo_t = torch.full_like(z_hi_b, float(z_lo))                      # (B,)
    span = (z_hi_b - z_lo_t).clamp(min=1e-3)
    return z_lo_t.unsqueeze(1) + idx.unsqueeze(0) * span.unsqueeze(1)  # (B, Z)


def _v4_fourier_pe_meters(z_lo: float, z_hi_global: float,
                            bin_centers_b: torch.Tensor, dim: int) -> torch.Tensor:
    """Sin/cos PE of bin centres in metres, normalised by the GLOBAL
    (z_lo, z_hi_global) span — NOT per-sample — so PE at z=0.15m is the
    same regardless of which view or batch entry sampled it. Wrist
    concentrates its bins in low z; they hit a subset of the global PE
    range with the same basis as scene at matching physical heights.

    Input ``bin_centers_b`` is (B, Z) in metres. Returns (B, Z, dim).
    """
    assert dim % 2 == 0
    K = dim // 2
    device, dtype = bin_centers_b.device, bin_centers_b.dtype
    span = max(float(z_hi_global - z_lo), 1e-6)
    z_norm = (bin_centers_b - z_lo) / span                              # (B, Z)
    freqs = (2.0 ** torch.arange(K, device=device, dtype=dtype))        # (K,)
    args = z_norm.unsqueeze(-1) * freqs * math.pi                       # (B, Z, K)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)        # (B, Z, dim)


class DinoVolumeSceneV4(_DinoHiresBase):
    """N-view world-space voxel grid with shared DINO backbone + per-voxel
    feature fusion across views."""

    def __init__(
        self,
        *,
        views: list[str],                       # e.g. ["scene"], ["wrist"], ["scene", "wrist"]
        n_window: int = 32,
        n_height_bins: int = 64,
        pred_size: int = 128,
        feat_dim: int = 32,
        height_enc_dim: int = 16,
        voxel_mlp_hidden: int = 64,
        d_model: int = 256,
        mlp_layers: int = 3,
        mlp_hidden_mult: float = 2.0,
        n_gripper_bins: int = 32,
        n_rot_clusters: int = 64,
        z_lo: float = 0.0,
        z_hi: float = 0.5,
        img_size: int = 504,
        cls_fusion: str = "concat",             # "concat" (all-view CLS into grip/rot head) or "primary"
        cross_view_layers: int = 0,             # 0 disables; recommended 4-5 for true cross-view fusion
        cross_view_heads: int = 6,
        per_voxel_grip_rot: bool = False,       # local grip/rot via voxel feature at GT (train) / argmax (infer) voxel
        temporal_head: str = "flat_mlp",        # "flat_mlp" (default 2026-06-10) or "temporal_cnn"
        temporal_cnn_layers: int = 3,           # only used when temporal_head=="temporal_cnn"
        temporal_cnn_kernel: int = 3,
        temporal_mlp_hidden: int = 512,         # only used when temporal_head=="flat_mlp"
        past_n: int = 0,                        # # of past (grip, z) frames concatenated into the global input
        past_enc_dim: int = 64,                 # encoder output dim for the past trajectory
        cam_margin_m: float = 0.01,             # v4: voxel grid stays this far below the cam (per-view z_hi cap)
        lang_dim: int = 0,                      # >0 enables FiLM language conditioning (CLIP text dim, e.g. 512)
        lang_hidden: int = 256,                 # hidden width of the FiLM generator MLP
        **_unused,
    ):
        super().__init__(
            n_window=n_window, n_height_bins=n_height_bins,
            pred_size=pred_size, feat_dim=feat_dim,
            d_model=d_model, d_cond=128, d_sin_z=48, d_sin_t=48, n_blocks=0,
            n_gripper_bins=n_gripper_bins, n_rot_clusters=n_rot_clusters,
        )
        self.views = list(views)
        self.n_views = len(views)
        self.height_enc_dim = height_enc_dim
        self.z_lo = float(z_lo); self.z_hi = float(z_hi)
        self.img_size = int(img_size)
        self.cls_fusion = cls_fusion
        # Past-trajectory encoder. Receives past_n frames of (grip, z) →
        # past_n * 2 raw scalars → encoder MLP → past_enc_dim vector that
        # gets concatenated into the global query input. This is the
        # proprio signal Cameron added 2026-06-10 so the model knows
        # "current gripper is closed / EE is at height X" before predicting
        # the next chunk.
        self.past_n = int(past_n)
        self.past_enc_dim = int(past_enc_dim)
        if self.past_n > 0:
            self.past_encoder = nn.Sequential(
                nn.Linear(self.past_n * 2, self.past_enc_dim), nn.GELU(),
                nn.Linear(self.past_enc_dim, self.past_enc_dim),
            )
        else:
            self.past_encoder = None

        # --- Optional FiLM language conditioning (Cameron 2026-07-22) ---
        # A frozen CLIP-text embedding (lang_dim) FiLM-modulates each DINO patch
        # token at low res, BEFORE the bilinear-up + refine CNN, so the modulation
        # flows into the shared per-view feature map that every voxel samples ->
        # the volume scoring becomes language-aware ("pick the soup" boosts the
        # soup patches). Identity-initialised (zero-init last layers) so it starts
        # as a no-op and language capability grows on top of the working vision path.
        self.lang_dim = int(lang_dim)
        lang_global_dim = 0
        if self.lang_dim > 0:
            self.lang_film = nn.Sequential(
                nn.Linear(self.lang_dim, lang_hidden), nn.GELU(),
                nn.Linear(lang_hidden, 2 * self.embed_dim),
            )
            nn.init.zeros_(self.lang_film[-1].weight); nn.init.zeros_(self.lang_film[-1].bias)
            # extra refine capacity (residual, zero-init -> identity at start)
            self.refine_extra = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.GELU(),
                nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            )
            nn.init.zeros_(self.refine_extra[-1].weight); nn.init.zeros_(self.refine_extra[-1].bias)
            # language into the global query vector (feeds grip/rot heads + q)
            lang_global_dim = 64
            self.lang_global = nn.Sequential(nn.Linear(self.lang_dim, lang_global_dim), nn.GELU())
        else:
            self.lang_film = None; self.refine_extra = None; self.lang_global = None

        # Override input_proj for both multi-CLS fusion AND optional past
        # conditioning. Input is always feat_dim + (cls fan-in) + (past fan-in) + (lang fan-in).
        cls_fan = self.n_views * self.embed_dim if cls_fusion == "concat" else self.embed_dim
        past_fan = self.past_enc_dim if self.past_n > 0 else 0
        in_dim = feat_dim + cls_fan + past_fan + lang_global_dim
        if cls_fusion not in ("concat", "primary"):
            raise ValueError(f"cls_fusion must be 'concat' or 'primary'; got {cls_fusion!r}")
        if in_dim != self.input_proj.in_features:
            self.input_proj = nn.Linear(in_dim, d_model)

        # DA3-style cross-view attention over native DINO patch grid. Only
        # built when (a) explicitly requested AND (b) there are 2+ views;
        # a single-view model has nothing to cross-attend with.
        self.cross_view_layers = int(cross_view_layers)
        if self.cross_view_layers > 0 and self.n_views > 1:
            self.cross_view_attn = CrossViewAttnStack(
                n_layers=self.cross_view_layers,
                embed_dim=self.embed_dim,
                n_heads=cross_view_heads,
            )
        else:
            self.cross_view_attn = None

        # Per-voxel grip/rot heads. At train time we teacher-force by
        # gathering ``voxel_feat`` at the GT-nearest voxel per timestep;
        # at inference we gather at the argmax voxel. Either way we get
        # a (B, T, F) sequence which a small 1D CNN smooths along T
        # before two linear heads emit grip / rot bin logits.
        self.per_voxel_grip_rot = bool(per_voxel_grip_rot)
        self.temporal_head_kind = temporal_head
        if self.per_voxel_grip_rot:
            if temporal_head == "temporal_cnn":
                assert temporal_cnn_kernel % 2 == 1, \
                    "use an odd kernel so padding preserves T"
                pad = temporal_cnn_kernel // 2
                cnn: list[nn.Module] = []
                for _ in range(int(temporal_cnn_layers)):
                    cnn.append(nn.Conv1d(feat_dim, feat_dim, temporal_cnn_kernel,
                                           padding=pad))
                    cnn.append(nn.GELU())
                self.temporal_cnn = nn.Sequential(*cnn)
                self.pv_grip_head = nn.Linear(feat_dim, n_gripper_bins)
                self.pv_rot_head = nn.Linear(feat_dim, n_rot_clusters)
                self.temporal_mlp = None
            elif temporal_head == "flat_mlp":
                # 4-layer MLP. Input = T·F flattened, output = T·(n_grip+n_rot)
                # flattened then split. Gives full receptive field over T AND
                # absolute positional information for free (input channel
                # index encodes (t, f) pair) — no sin/cos PE needed.
                in_dim = n_window * feat_dim
                out_dim = n_window * (n_gripper_bins + n_rot_clusters)
                h = int(temporal_mlp_hidden)
                self.temporal_mlp = nn.Sequential(
                    nn.Linear(in_dim, h), nn.GELU(),
                    nn.Linear(h, h),      nn.GELU(),
                    nn.Linear(h, h),      nn.GELU(),
                    nn.Linear(h, out_dim),
                )
                self.temporal_cnn = None
                self.pv_grip_head = None
                self.pv_rot_head = None
            else:
                raise ValueError(
                    f"temporal_head must be 'flat_mlp' or 'temporal_cnn'; "
                    f"got {temporal_head!r}")
        else:
            self.temporal_cnn = None
            self.temporal_mlp = None
            self.pv_grip_head = None
            self.pv_rot_head = None

        # v4: bin_centers_z buffer kept for inference fallback / decode
        # XYZ lookup; the actual per-view per-batch bin centres are
        # computed inside ``forward`` from the cam pose. v4 replaces the
        # learned height embedding with a sin/cos PE.
        Z = n_height_bins
        centers = torch.tensor(
            [self.z_lo + (i + 0.5) * (self.z_hi - self.z_lo) / Z
             for i in range(Z)], dtype=torch.float32)
        self.register_buffer("bin_centers_z", centers)
        self.cam_margin_m = float(cam_margin_m)

        # Per-voxel MLP: concat all view features + height emb.
        d_in = self.n_views * feat_dim + height_enc_dim
        h = voxel_mlp_hidden
        self.voxel_mlp = nn.Sequential(
            nn.Linear(d_in, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, feat_dim),
        )

        # Flat global MLP (same as volumetric).
        T = n_window
        hidden = int(d_model * mlp_hidden_mult)
        layers: list[nn.Module] = [nn.Linear(d_model, hidden), nn.GELU()]
        for _ in range(mlp_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
        self.global_mlp = nn.Sequential(*layers)
        self.q_head = nn.Linear(hidden, T * feat_dim)
        self.grip_head = nn.Linear(hidden, T * n_gripper_bins)
        self.rot_head = nn.Linear(hidden, T * n_rot_clusters)

    # ────────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────
    # Backbone hook (subclass swap point)
    # ────────────────────────────────────────────────────────────────────

    def _per_view_patch_features(self, rgb: list[torch.Tensor]):
        """Returns (per-view patch grids, per-view global tokens).

        Default DINOv3 path: loops over views, each through the (shared)
        backbone independently — no cross-view interaction at the
        backbone level (use ``CrossViewAttnStack`` for that).

        Override in subclasses to swap the backbone (v4_da3 uses DA3's
        multi-view ViT with built-in cross-attention; v4_vlm uses
        InternVL's vision tower). Subclass must return:
          - ``view_patches``: list of N tensors of shape (B, embed_dim, ph, pw)
          - ``view_global``:  list of N tensors of shape (B, embed_dim)
        with ``embed_dim`` matching ``self.embed_dim`` (so feat_head and
        input_proj line up). ph/pw can differ from DINOv3's 31×31; the
        downstream 5-layer refine CNN bilinear-ups to ``pred_size``.
        """
        view_patches: list[torch.Tensor] = []
        view_cls: list[torch.Tensor] = []
        for k in range(self.n_views):
            patch, cls = self.patch_features(rgb[k])
            view_patches.append(patch)
            view_cls.append(cls)
        return view_patches, view_cls

    def _unproject_grid(self, K_in: torch.Tensor, T_w2c: torch.Tensor,
                          bin_centers_b: torch.Tensor) -> torch.Tensor:
        """For ONE camera, return (B, Z, P, P, 3) world voxel positions.

        ``bin_centers_b`` is (B, Z) per-batch z bin centres in metres —
        lets the wrist view shrink its z range to its own cam-Z each
        frame.
        """
        B = K_in.shape[0]
        P, Z, S = self.pred_size, self.n_z, self.img_size
        device = K_in.device; dtype = K_in.dtype
        cell = S / P
        u_idx = torch.arange(P, device=device, dtype=dtype)
        v_idx = torch.arange(P, device=device, dtype=dtype)
        u_pix = (u_idx + 0.5) * cell
        v_pix = (v_idx + 0.5) * cell
        vv, uu = torch.meshgrid(v_pix, u_pix, indexing="ij")
        ones = torch.ones_like(uu)
        pix_h = torch.stack([uu, vv, ones], dim=-1)              # (P, P, 3)
        K_inv = torch.inverse(K_in)
        ray_cam = torch.einsum("bij,hwj->bhwi", K_inv, pix_h)
        T_c2w = torch.inverse(T_w2c)
        R = T_c2w[:, :3, :3]; cam_o = T_c2w[:, :3, 3]
        ray_world = torch.einsum("bij,bhwj->bhwi", R, ray_cam)
        rz = ray_world[..., 2]
        cam_oz = cam_o[..., 2].unsqueeze(-1).unsqueeze(-1)
        rz_safe = torch.where(rz.abs() < 1e-6,
                              torch.full_like(rz, 1e-6), rz)
        z_centers = bin_centers_b.to(rz.dtype)                   # (B, Z)
        lam = ((z_centers.view(B, 1, 1, Z) - cam_oz.unsqueeze(-1))
               / rz_safe.unsqueeze(-1))
        world = (cam_o.view(B, 1, 1, 1, 3)
                 + lam.unsqueeze(-1) * ray_world.unsqueeze(3))
        return world.permute(0, 3, 1, 2, 4).contiguous()         # (B, Z, P, P, 3)

    @staticmethod
    def _sample_view_feat(F_map: torch.Tensor, xyz_world: torch.Tensor,
                          K_in: torch.Tensor, T_w2c: torch.Tensor,
                          img_size: int) -> torch.Tensor:
        """Project ``xyz_world`` (B, V, 3) into a camera and bilinear-sample
        features from ``F_map`` (B, F, P, P). Returns (B, V, F).

        Out-of-frustum voxels (negative depth OR projected pixel outside
        the image) get zero features via padding_mode='zeros'.
        """
        B, V, _ = xyz_world.shape
        ones = torch.ones(B, V, 1, device=xyz_world.device, dtype=xyz_world.dtype)
        xyz_h = torch.cat([xyz_world, ones], dim=-1)             # (B, V, 4)
        cam = torch.einsum("bij,bvj->bvi", T_w2c, xyz_h)[..., :3]  # (B, V, 3)
        # Mask voxels with depth ≤ 0 (behind the camera) BEFORE the projective
        # divide — feature stays zero for those via grid_sample's border range.
        z = cam[..., 2:3].clamp(min=1e-3)                         # (B, V, 1)
        uv = torch.einsum("bij,bvj->bvi", K_in, cam) / z          # (B, V, 3)
        # Convert pixel coords → grid_sample normalized [-1, 1].
        S = float(img_size)
        norm_u = (uv[..., 0] / S) * 2.0 - 1.0
        norm_v = (uv[..., 1] / S) * 2.0 - 1.0
        # Zero-out voxels that are behind the camera by pushing their
        # sample coord well outside [-1, 1] so padding_mode='zeros' kicks in.
        behind = cam[..., 2] <= 0
        norm_u = torch.where(behind, torch.full_like(norm_u, 5.0), norm_u)
        norm_v = torch.where(behind, torch.full_like(norm_v, 5.0), norm_v)
        grid = torch.stack([norm_u, norm_v], dim=-1).unsqueeze(1)  # (B, 1, V, 2)
        sampled = F.grid_sample(F_map, grid, mode="bilinear",
                                 padding_mode="zeros",
                                 align_corners=False)               # (B, F, 1, V)
        return sampled.squeeze(2).transpose(1, 2)                   # (B, V, F)

    # ────────────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────────────

    def forward(self,
                rgb: list[torch.Tensor],
                start_pix: torch.Tensor,
                K_in: list[torch.Tensor],
                T_w2c: list[torch.Tensor],
                target_xyz: torch.Tensor | None = None,
                border_mask_px: int = 0,
                past_grip: torch.Tensor | None = None,
                past_z: torch.Tensor | None = None,
                lang_emb: torch.Tensor | None = None) -> dict:
        """Lists are length N (one entry per view, in ``self.views`` order).

        ``rgb[k]``     : (B, 3, S, S) — view k's image at img_size×img_size
        ``K_in[k]``    : (B, 3, 3)    — view k's intrinsics, scaled to img_size
        ``T_w2c[k]``   : (B, 4, 4)    — view k's world-to-cam pose
        ``start_pix``  : (B, 2)       — EE pixel in PRIMARY view (views[0])

        Returns:
          - ``volume_logits`` (B, T, N*Z, P, P) — flat-decode-compatible
          - ``voxel_positions`` (B, N*Z, P, P, 3) — for loss (nearest-3D) + decode
          - ``grip_logits``, ``rot_logits``, ``F_refined`` (primary view),
            ``patch_features`` (primary view)
        """
        assert len(rgb) == self.n_views and len(K_in) == self.n_views
        assert len(T_w2c) == self.n_views
        B, _, S, _ = rgb[0].shape
        T, Z, P, F_d = self.n_window, self.n_z, self.pred_size, self.feat_dim
        N = self.n_views

        # 1. Per-view backbone → raw patch grids + global tokens.
        # Factored into a method so subclasses can swap the backbone
        # (v4_da3, v4_vlm) without duplicating the full forward pass.
        view_patches, view_cls = self._per_view_patch_features(rgb)

        # 1-lang. FiLM the DINO patch grid with the CLIP-text embedding (low res,
        # before upsampling). gamma/beta are per-channel; identity at init.
        if self.lang_film is not None and lang_emb is not None:
            gb = self.lang_film(lang_emb)                             # (B, 2*embed_dim)
            gamma, beta = gb.chunk(2, dim=-1)                         # each (B, embed_dim)
            gamma = gamma.view(B, -1, 1, 1); beta = beta.view(B, -1, 1, 1)
            view_patches = [vp * (1.0 + gamma) + beta for vp in view_patches]

        # 1a. Optional DA3-style cross-view attention over the native DINO
        # patch grid. Runs only when --cross_view_layers > 0 and N>1.
        if self.cross_view_attn is not None:
            view_patches = self.cross_view_attn(view_patches)
        patch0 = view_patches[0]

        # 1b. Per-view feat_head → bilinear-up to pred_size → 5-layer
        # refine CNN. Same recipe as ``_pixel_features`` but split so we
        # could insert the cross-view stage in between.
        # v4: per-view adaptive z range computed from each view's cam pose.
        # Scene cam is high+static so cap is inactive; wrist cam rides the
        # EE so its z_hi drops to just under cam_z each frame.
        view_bin_centers: list[torch.Tensor] = []   # each (B, Z)
        view_pe_z: list[torch.Tensor] = []           # each (B, Z, height_enc_dim)
        for k in range(N):
            T_c2w_k = torch.inverse(T_w2c[k])
            cam_z_k = T_c2w_k[:, 2, 3]                                  # (B,)
            z_hi_k = torch.minimum(
                cam_z_k - float(self.cam_margin_m),
                torch.full_like(cam_z_k, float(self.z_hi)))
            z_hi_k = torch.clamp(z_hi_k, min=float(self.z_lo) + 0.01)
            bin_b = _v4_per_view_bin_centers(self.z_lo, z_hi_k, self.n_z)
            pe_b = _v4_fourier_pe_meters(
                self.z_lo, self.z_hi, bin_b, self.height_enc_dim)
            view_bin_centers.append(bin_b)
            view_pe_z.append(pe_b)

        view_feats: list[torch.Tensor] = []
        view_voxel_pos: list[torch.Tensor] = []
        for k in range(N):
            F0 = self.feat_head(view_patches[k])
            F1 = F.interpolate(F0, size=(self.pred_size, self.pred_size),
                                mode="bilinear", align_corners=False)
            F_scene = self.refine_cnn(F1)
            if self.refine_extra is not None:                        # extra language-aware capacity (residual, identity at init)
                F_scene = F_scene + self.refine_extra(F_scene)
            view_feats.append(F_scene)
            view_voxel_pos.append(self._unproject_grid(
                K_in[k], T_w2c[k], view_bin_centers[k]))
        # Fuse the CLS tokens that feed the grip/rot head. "concat" makes
        # the global MLP see every view's global descriptor; "primary"
        # falls back to the original behaviour (matches old ckpts).
        cls_fused = (torch.cat(view_cls, dim=-1)
                      if self.cls_fusion == "concat" and N > 1
                      else view_cls[0])

        # Combined voxel positions (B, N*Z, P, P, 3) — stack along Z.
        voxel_pos = torch.cat(view_voxel_pos, dim=1)
        NZ = N * Z
        V = NZ * P * P

        # 2. For each voxel (in the combined volume), sample features from
        #    every view. The features for view k include its OWN unprojected
        #    grid features (direct tile, no projection error) for that view's
        #    slice [k*Z : (k+1)*Z], and projected-sampled features for the
        #    other views' slices.
        # Flatten voxel_pos for sampling.
        voxel_flat = voxel_pos.reshape(B, V, 3)                   # (B, V, 3)

        per_view_feat: list[torch.Tensor] = []                    # each (B, V, F)
        for k in range(N):
            per_view_feat.append(self._sample_view_feat(
                view_feats[k], voxel_flat, K_in[k], T_w2c[k], self.img_size))
        all_view_feat = torch.cat(per_view_feat, dim=-1)          # (B, V, N*F)

        # 3. v4: per-view per-batch sin/cos PE of the world-Z bin centres
        # (the centres differ per view because of the adaptive z range).
        # Stack into (B, NZ, h_enc) → broadcast over (P, P) → flat (B, V, h_enc).
        # view_pe_z[k] is (B, Z, h_enc); concat across views gives (B, NZ, h_enc).
        h_per_view = torch.cat(view_pe_z, dim=1)                    # (B, NZ, h_enc)
        h_enc = h_per_view.shape[-1]
        h_emb = (h_per_view.view(B, NZ, 1, 1, h_enc)
                  .expand(B, NZ, P, P, h_enc)
                  .reshape(B, V, h_enc))                            # (B, V, h_enc)

        # 4. Voxel MLP: concat [features..., height_emb] → feat_dim.
        voxel_in = torch.cat([all_view_feat, h_emb], dim=-1)      # (B, V, N*F + h_enc)
        voxel_feat = self.voxel_mlp(voxel_in)                     # (B, V, F)

        # 5. Flat global MLP query — eef sampled from PRIMARY view's feature map.
        F_prim = view_feats[0]
        scale = P / S
        sx = (start_pix[..., 0] * scale).long().clamp(0, P - 1)
        sy = (start_pix[..., 1] * scale).long().clamp(0, P - 1)
        b_idx = torch.arange(B, device=rgb[0].device)
        eef_feat = F_prim[b_idx, :, sy, sx]                       # (B, F)
        # Optional past-trajectory conditioning. Both tensors are (B, past_n).
        # We flatten + encode + concat into the global query input so the
        # MLP knows "current gripper / EE-height" before predicting next chunk.
        global_in = [eef_feat, cls_fused]
        if self.past_encoder is not None:
            assert past_grip is not None and past_z is not None, \
                "past_n > 0 but past_grip / past_z not provided"
            past_flat = torch.cat([past_grip, past_z], dim=-1)            # (B, 2*past_n)
            global_in.append(self.past_encoder(past_flat))                # (B, past_enc_dim)
        if self.lang_global is not None and lang_emb is not None:
            global_in.append(self.lang_global(lang_emb))                  # (B, lang_global_dim)
        q_in = self.input_proj(torch.cat(global_in, dim=-1))
        hglob = self.global_mlp(q_in)                              # (B, hidden)
        q = self.q_head(hglob).reshape(B, T, F_d)                  # (B, T, F)

        # 6. Dot-product score per voxel.
        vol = torch.einsum("btc,bvc->btv", q, voxel_feat)         # (B, T, V)
        vol = vol.view(B, T, NZ, P, P)                             # (B, T, NZ, P, P)

        # Border mask applies BEFORE both the per-voxel grip/rot argmax
        # AND the downstream xyz lookup so the two stay consistent.
        # Training never passes a positive ``border_mask_px`` so the
        # volume loss still sees every voxel.
        if border_mask_px > 0 and border_mask_px * 2 < P:
            keep = torch.zeros(P, P, dtype=torch.bool, device=vol.device)
            keep[border_mask_px:P - border_mask_px,
                  border_mask_px:P - border_mask_px] = True
            vol = vol.masked_fill(~keep.view(1, 1, 1, P, P), -1e9)

        grip_logits = self.grip_head(hglob).reshape(B, T, self.n_gripper_bins)
        rot_logits = self.rot_head(hglob).reshape(B, T, self.n_rot_clusters)

        # Per-voxel grip/rot path — overrides the global heads above when
        # enabled. Same teacher-force-at-GT / argmax-at-infer pattern the
        # original production PARA used for grip/rot, but now gathering
        # from the cross-view-fused per-voxel features.
        if self.per_voxel_grip_rot:
            vol_flat = vol.reshape(B, T, V)
            if target_xyz is not None:
                # Argmin world-distance per (B, T) → GT voxel index.
                # d² = |a|² + |b|² − 2 a·b (avoids materialising (B,T,V,3)).
                v_norm_sq = (voxel_flat * voxel_flat).sum(-1)                # (B, V)
                t_norm_sq = (target_xyz * target_xyz).sum(-1)                # (B, T)
                dot = torch.einsum("btc,bvc->btv", target_xyz, voxel_flat)
                dist_sq = (t_norm_sq.unsqueeze(-1)
                            + v_norm_sq.unsqueeze(1) - 2 * dot)              # (B, T, V)
                idx_t = dist_sq.argmin(dim=2)                                # (B, T)
            else:
                idx_t = vol_flat.argmax(dim=2)                               # (B, T)
            gather_idx = idx_t.unsqueeze(-1).expand(B, T, F_d)
            temp_feat = torch.gather(voxel_feat, 1, gather_idx)              # (B, T, F)
            if self.temporal_head_kind == "flat_mlp":
                flat_in = temp_feat.reshape(B, T * F_d)
                flat_out = self.temporal_mlp(flat_in)                        # (B, T*(n_grip+n_rot))
                gr = flat_out.reshape(B, T, self.n_gripper_bins + self.n_rot_clusters)
                grip_logits = gr[:, :, :self.n_gripper_bins]
                rot_logits = gr[:, :, self.n_gripper_bins:]
            else:  # temporal_cnn
                x_tc = self.temporal_cnn(temp_feat.transpose(1, 2)).transpose(1, 2)
                grip_logits = self.pv_grip_head(x_tc)                        # (B, T, n_grip)
                rot_logits = self.pv_rot_head(x_tc)                          # (B, T, n_rot)

        return {
            "volume_logits": vol,
            "voxel_positions": voxel_pos,                          # (B, NZ, P, P, 3)
            "grip_logits": grip_logits,
            "rot_logits": rot_logits,
            "F_refined": F_prim,                                   # primary view's
            "patch_features": patch0,
            "views": self.views,
            "view_feat_maps": view_feats,                          # list of (B, F, P, P)
        }


def multiview_losses(
    out: dict,
    target_yx: torch.Tensor,
    target_z: torch.Tensor,
    target_grip: torch.Tensor,
    target_rot: torch.Tensor,
    target_xyz: torch.Tensor,
) -> dict:
    """CE loss over the FULL N×Z×P×P voxel set with target = argmin world
    distance from GT-XYZ. Same identity-trick as single-view to avoid
    materialising a per-voxel diff tensor."""
    vol = out["volume_logits"]                                  # (B, T, NZ, P, P)
    voxel_pos = out["voxel_positions"]                           # (B, NZ, P, P, 3)
    B, T, NZ, P, _ = vol.shape
    V = NZ * P * P
    voxel_flat = voxel_pos.reshape(B, V, 3)

    with torch.no_grad():
        gt = target_xyz                                          # (B, T, 3)
        voxel_sq = (voxel_flat ** 2).sum(dim=-1)                 # (B, V)
        gt_sq = (gt ** 2).sum(dim=-1)                            # (B, T)
        dot = torch.einsum("btc,bvc->btv", gt, voxel_flat)       # (B, T, V)
        dist_sq = (gt_sq[..., None]
                   + voxel_sq[:, None, :]
                   - 2.0 * dot)
        target = dist_sq.argmin(dim=-1)

    vol_flat = vol.reshape(B * T, V)
    loss_vol = F.cross_entropy(vol_flat, target.reshape(B * T))
    loss_grip = F.cross_entropy(
        out["grip_logits"].reshape(B * T, -1), target_grip.reshape(B * T))
    loss_rot = F.cross_entropy(
        out["rot_logits"].reshape(B * T, -1), target_rot.reshape(B * T))
    return {
        "loss/volume": loss_vol,
        "loss/grip": loss_grip,
        "loss/rot": loss_rot,
        "loss/total": loss_vol + loss_grip + loss_rot,
    }
