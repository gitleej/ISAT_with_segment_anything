# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoderHQ, PromptEncoder, Sam, TwoWayTransformer, TinyViT


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

from typing import Any, Dict, Optional, Tuple

# ---------- 辅助函数 ----------
def _safe_load_checkpoint(path_or_file: str):
    """
    安全加载 checkpoint，兼容多种保存方式并尽量绕过 persistent_id 问题。
    返回加载得到的原始对象（可能是 state_dict、dict、nn.Module 等）。
    """
    last_exc = None
    # 1) 直接用 torch.load(path)
    try:
        return torch.load(path_or_file, map_location=torch.device('cpu'))
    except Exception as e:
        last_exc = e

    # 2) try file object with torch.load
    try:
        with open(path_or_file, "rb") as f:
            return torch.load(f, map_location=torch.device("cpu"))
    except Exception as e:
        last_exc = e

    # 3) try pickle.load
    try:
        with open(path_or_file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        last_exc = e

    # 4) try Unpickler ignoring persistent_load (best-effort)
    class _IgnorePersistentUnpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            # 忽略 persistent id，返回 None（可能导致部分对象为 None）
            return None

    try:
        with open(path_or_file, "rb") as f:
            return _IgnorePersistentUnpickler(f).load()
    except Exception:
        # 抛出最初的异常，便于调试
        raise last_exc

def _extract_state_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """
    尝试从加载对象中提取 state_dict（若可能）。
    返回 state_dict 或 None（无法提取）。
    """
    if obj is None:
        return None

    # 如果是 nn.Module，直接取 state_dict
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()

    # 如果是 dict：检查常见 key
    if isinstance(obj, dict):
        # 常见字段名
        for k in ('state_dict', 'model', 'model_state_dict', 'net', 'state'):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # 若 dict 的 value 大多为 tensor，则可能已经是 state_dict
        tensor_like = sum(1 for v in obj.values() if isinstance(v, (torch.Tensor,)))
        if len(obj) > 0 and tensor_like / len(obj) > 0.5:
            return obj
        # 否则无法确定
        return None

    # 其他情况暂不支持
    return None

def _strip_module_prefix(state_dict: Dict[str, Any], prefix: str = "module.") -> Dict[str, Any]:
    """
    如果大多数 key 带有 prefix，则移除它。
    """
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    with_prefix = sum(1 for k in keys if k.startswith(prefix))
    if with_prefix / len(keys) > 0.5:
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_sd[k[len(prefix):]] = v
            else:
                new_sd[k] = v
        return new_sd
    return state_dict

def build_sam_vit_t(checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoderHQ(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                vit_dim=160,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()

    # -------- 如果提供 checkpoint，尝试加载 --------
    if checkpoint is not None:
        print(f"[build_sam_vit_t] Loading checkpoint from: {checkpoint}")
        # 1) 读取对象
        ckpt_obj = _safe_load_checkpoint(checkpoint)

        # 2) 尝试提取 state_dict
        state_dict = _extract_state_dict(ckpt_obj)
        if state_dict is None:
            # 若无法直接提取，尝试把顶层 dict 作为 state_dict（最后一招）
            if isinstance(ckpt_obj, dict):
                state_dict = ckpt_obj
            else:
                raise RuntimeError(
                    f"[build_sam_vit_t] Cannot extract state_dict from checkpoint object of type {type(ckpt_obj)}")

        # 3) 处理 'module.' 前缀
        state_dict = _strip_module_prefix(state_dict, prefix="module.")

        # 4) 尝试加载到模型
        try:
            info = mobile_sam.load_state_dict(state_dict, strict=False)
            print("[build_sam_vit_t] load_state_dict result:", info)
        except Exception as e:
            # 如果直接加载失败，尝试按 key 后缀匹配（最简单的启发式尝试）
            print(f"[build_sam_vit_t] load_state_dict raised: {e}. Trying heuristic key matching...")
            model_sd = mobile_sam.state_dict()
            new_sd = {}
            ckpt_keys = list(state_dict.keys())
            for mk in model_sd.keys():
                # 找第一个以 mk 结尾的 ckpt key
                candidates = [k for k in ckpt_keys if k.endswith(mk)]
                if candidates:
                    new_sd[mk] = state_dict[candidates[0]]
            info = mobile_sam.load_state_dict(new_sd, strict=False)
            print("[build_sam_vit_t] heuristic load result:", info)
    # if checkpoint is not None:
    #     with open(checkpoint, "rb") as f:
    #         state_dict = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    #     info = mobile_sam.load_state_dict(state_dict, strict=False)
    #     print(info)
    for n, p in mobile_sam.named_parameters():
        if 'hf_token' not in n and 'hf_mlp' not in n and 'compress_vit_feat' not in n and 'embedding_encoder' not in n and 'embedding_maskfeature' not in n:
            p.requires_grad = False
    return mobile_sam

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_tiny": build_sam_vit_t
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderHQ(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=encoder_embed_dim,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()

    # -------- 如果提供 checkpoint，尝试加载 --------
    if checkpoint is not None:
        print(f"[build_sam_vit_t] Loading checkpoint from: {checkpoint}")
        # 1) 读取对象
        ckpt_obj = _safe_load_checkpoint(checkpoint)

        # 2) 尝试提取 state_dict
        state_dict = _extract_state_dict(ckpt_obj)
        if state_dict is None:
            # 若无法直接提取，尝试把顶层 dict 作为 state_dict（最后一招）
            if isinstance(ckpt_obj, dict):
                state_dict = ckpt_obj
            else:
                raise RuntimeError(
                    f"[build_sam_vit_t] Cannot extract state_dict from checkpoint object of type {type(ckpt_obj)}")

        # 3) 处理 'module.' 前缀
        state_dict = _strip_module_prefix(state_dict, prefix="module.")

        # 4) 尝试加载到模型
        try:
            info = sam.load_state_dict(state_dict, strict=False)
            print("[build_sam_vit_t] load_state_dict result:", info)
        except Exception as e:
            # 如果直接加载失败，尝试按 key 后缀匹配（最简单的启发式尝试）
            print(f"[build_sam_vit_t] load_state_dict raised: {e}. Trying heuristic key matching...")
            model_sd = sam.state_dict()
            new_sd = {}
            ckpt_keys = list(state_dict.keys())
            for mk in model_sd.keys():
                # 找第一个以 mk 结尾的 ckpt key
                candidates = [k for k in ckpt_keys if k.endswith(mk)]
                if candidates:
                    new_sd[mk] = state_dict[candidates[0]]
            info = sam.load_state_dict(new_sd, strict=False)
            print("[build_sam_vit_t] heuristic load result:", info)
    # if checkpoint is not None:
    #     with open(checkpoint, "rb") as f:
    #         state_dict = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    #     info = sam.load_state_dict(state_dict, strict=False)
    #     print(info)
    for n, p in sam.named_parameters():
        if 'hf_token' not in n and 'hf_mlp' not in n and 'compress_vit_feat' not in n and 'embedding_encoder' not in n and 'embedding_maskfeature' not in n:
            p.requires_grad = False

    return sam
