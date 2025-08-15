# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pickle

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, TinyViT


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

from typing import Any, Dict, Optional

def _safe_torch_load(path: str):
    """
    尝试多种方式加载 checkpoint，兼容老/新版 PyTorch。
    返回原始 obj（可能是 state_dict / dict / nn.Module 等）。
    """
    # 1) 直接 torch.load（最常见）
    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        # 可能是传了 weights_only 等新参数（这段通常不会触发，但保留）
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            last_exc = e
    except Exception as e:
        last_exc = e

    # 2) 尝试使用 pickle.load（在某些场景下能绕过 torch 的 persistent id）
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        last_exc = e

    # 3) 最后尝试用一个忽略 persistent_id 的 Unpickler（尽量不丢数据，但可能不完美）
    class _IgnorePersistentUnpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            # 忽略 persistent id，返回 None（这会让部分对象变成 None，需要后续处理）
            return None

    try:
        with open(path, "rb") as f:
            return _IgnorePersistentUnpickler(f).load()
    except Exception:
        # 最后抛出最先捕获到的异常
        raise last_exc

def _extract_state_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """
    从 torch.load/pickle.load 的结果中提取 state_dict（若可能）。
    返回 None 表示无法确定 state_dict。
    """
    if obj is None:
        return None

    # 1) 如果本身是 dict，检查常见 key
    if isinstance(obj, dict):
        # 常见字段名
        candidates = ['state_dict', 'model', 'model_state_dict', 'net', 'state']
        for k in candidates:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]

        # 如果 dict 的 value 大多数是 Tensor/ndarray，极可能就是 state_dict
        # 判断一下 value 类型
        tensor_like_count = 0
        total = 0
        for v in obj.values():
            total += 1
            # torch.Tensor 或 numpy array/torch storage
            if isinstance(v, (torch.Tensor,)):
                tensor_like_count += 1
        if total > 0 and tensor_like_count / total > 0.5:
            return obj

        # 有时候 mmcv 保存会把 key 前面加上 module.，我们在 load 时处理即可
        return None

    # 2) 如果 obj 是 nn.Module 实例
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()

    # 3) 其它类型暂不处理
    return None

def _maybe_strip_prefix(state_dict: Dict[str, Any], prefix: str = "module.") -> Dict[str, Any]:
    """
    如果 keys 大多以 prefix 开头，则移除该前缀（处理 DistributedDataParallel 保存的情况）。
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

def build_sam_vit_t(checkpoint: Optional[str] = None):
    import torch.nn as nn
    # 你的模型构建（保持不变）
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
            mask_decoder=MaskDecoder(
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
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()

    # 如果提供检查点，尝试加载（兼容老版 PyTorch）
    if checkpoint is not None:
        # 1) 尝试用多种方式读取 checkpoint
        ckpt_obj = None
        try:
            ckpt_obj = _safe_torch_load(checkpoint)
        except Exception as e:
            # 明确打印异常便于调试，但不要直接中断（后面会 raise）
            print(f"[build_sam_vit_t] failed to load checkpoint via safe loader: {e}")
            raise

        # 2) 提取 state_dict
        state_dict = _extract_state_dict(ckpt_obj)
        if state_dict is None:
            # 有时候加载出来的 obj 是更复杂的结构（或我们的忽略 persistent id 导致部分 None）
            # 直接尝试把 ckpt_obj 当作 state_dict 使用（可能失败）
            if isinstance(ckpt_obj, dict):
                state_dict = ckpt_obj  # 让后面的加载尝试匹配 key
            else:
                raise RuntimeError("Unable to extract state_dict from checkpoint. "
                                   "The loaded object type is: {}".format(type(ckpt_obj)))

        # 3) 清理 keys（例如移除 'module.' 前缀）
        state_dict = _maybe_strip_prefix(state_dict, prefix="module.")

        # 4) 尝试加载 state_dict 到模型
        missing_keys, unexpected_keys = None, None
        try:
            model_dict = mobile_sam.state_dict()
            # 4a) 直接尝试严格加载（strict=False 更宽容）
            load_res = mobile_sam.load_state_dict(state_dict, strict=False)
            # torch >=1.6 返回 NamedTuple，有 missing_keys/unexpected_keys 属性
            # 兼容处理：
            if isinstance(load_res, tuple) or hasattr(load_res, "missing_keys"):
                # PyTorch 1.12 返回 a NamedTuple/ dict-like from load_state_dict?
                try:
                    missing_keys = load_res.missing_keys if hasattr(load_res, "missing_keys") else load_res[0]
                    unexpected_keys = load_res.unexpected_keys if hasattr(load_res, "unexpected_keys") else load_res[1]
                except Exception:
                    # fallback: ignore
                    missing_keys = None
                    unexpected_keys = None
            else:
                # older returns dict? just ignore
                missing_keys = None
                unexpected_keys = None
        except Exception as e:
            # 如果直接加载失败，尝试把 state_dict 中的 key 逐个映射到 model 的 key（更强的容错）
            print(f"[build_sam_vit_t] load_state_dict failed: {e}. Will try heuristic key matching.")
            # 尝试按 key 前缀匹配 —— 简单实现：对每个 model key，寻找最相似的 ckpt key（按后缀）
            sd_new = {}
            model_keys = list(mobile_sam.state_dict().keys())
            ckpt_keys = list(state_dict.keys())
            for mk in model_keys:
                # 找到以 mk 结尾的 ckpt key（常见 distributed 情况）
                candidates = [k for k in ckpt_keys if k.endswith(mk)]
                if candidates:
                    sd_new[mk] = state_dict[candidates[0]]
            # 尝试加载部分匹配的 sd_new
            load_res = mobile_sam.load_state_dict(sd_new, strict=False)
            print(f"[build_sam_vit_t] heuristic load result: {load_res}")
        # 打印缺失/多余信息，便于调试
        if missing_keys:
            print("[build_sam_vit_t] missing keys:", missing_keys)
        if unexpected_keys:
            print("[build_sam_vit_t] unexpected keys:", unexpected_keys)

    return mobile_sam
# def build_sam_vit_t(checkpoint=None):
#     prompt_embed_dim = 256
#     image_size = 1024
#     vit_patch_size = 16
#     image_embedding_size = image_size // vit_patch_size
#     mobile_sam = Sam(
#             image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
#                 embed_dims=[64, 128, 160, 320],
#                 depths=[2, 2, 6, 2],
#                 num_heads=[2, 4, 5, 10],
#                 window_sizes=[7, 7, 14, 7],
#                 mlp_ratio=4.,
#                 drop_rate=0.,
#                 drop_path_rate=0.0,
#                 use_checkpoint=False,
#                 mbconv_expand_ratio=4.0,
#                 local_conv_size=3,
#                 layer_lr_decay=0.8
#             ),
#             prompt_encoder=PromptEncoder(
#             embed_dim=prompt_embed_dim,
#             image_embedding_size=(image_embedding_size, image_embedding_size),
#             input_image_size=(image_size, image_size),
#             mask_in_chans=16,
#             ),
#             mask_decoder=MaskDecoder(
#                     num_multimask_outputs=3,
#                     transformer=TwoWayTransformer(
#                     depth=2,
#                     embedding_dim=prompt_embed_dim,
#                     mlp_dim=2048,
#                     num_heads=8,
#                 ),
#                 transformer_dim=prompt_embed_dim,
#                 iou_head_depth=3,
#                 iou_head_hidden_dim=256,
#             ),
#             pixel_mean=[123.675, 116.28, 103.53],
#             pixel_std=[58.395, 57.12, 57.375],
#         )
#
#     mobile_sam.eval()
#     if checkpoint is not None:
#         with open(checkpoint, "rb") as f:
#             try:
#                 state_dict = torch.load(f, map_location="cpu", weights_only=False)
#             except TypeError:
#                 state_dict = torch.load(f, map_location="cpu", pickle_module=pickle)
#         mobile_sam.load_state_dict(state_dict)
#     return mobile_sam


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
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
        mask_decoder=MaskDecoder(
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
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu", weights_only=False)
        sam.load_state_dict(state_dict)
    return sam


