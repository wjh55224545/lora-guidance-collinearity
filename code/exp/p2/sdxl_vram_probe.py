"""SDXL LoRA 在 8 GB 上的显存可行性探测。

不加载权重：按 SDXL UNet 的真实配置随机初始化 fp16 模型，加 rank-8 LoRA，
在 512 分辨率（latent 64x64）上跑一次前反向，报告各阶段显存峰值。
目的只是回答"能不能训"，不是训出东西。

用法：
  D:/python/python.exe exp/p2/sdxl_vram_probe.py --res 512
"""
import argparse
import gc
import os
import sys
import time

import torch

GB = 2 ** 30


def mem(tag):
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    print(f"[{tag:<26}] 进程占用 {alloc/GB:5.2f} | 峰值 {peak/GB:5.2f} | "
          f"整卡可用 {free/GB:5.2f} / {total/GB:5.2f} GB", flush=True)
    return alloc, peak


def build_unet(dtype):
    from diffusers import UNet2DConditionModel

    cfg = dict(
        sample_size=128,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(320, 640, 1280),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=2048,
        addition_embed_type="text_time",
        addition_time_embed_dim=256,
        projection_class_embeddings_input_dim=2816,
        transformer_layers_per_block=(1, 2, 10),
        norm_num_groups=32,
        attention_head_dim=(5, 10, 20),
    )
    t = time.time()
    # CPU 只有 3.6 GB 可用，fp32 构造（10.3 GB）不可行。
    # 用 meta 设备构造（零内存）并把默认 dtype 临时设成 fp16，
    # 再 to_empty 直接在 GPU 上按 fp16 分配，全程不经过 CPU 副本。
    # 权重是未初始化的，前向会得到 nan，但显存占用与真实情况一致——本脚本只测显存。
    from accelerate import init_empty_weights
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with init_empty_weights():
            unet = UNet2DConditionModel(**cfg)
    finally:
        torch.set_default_dtype(old_dtype)
    unet = unet.to_empty(device="cuda")
    print(f"UNet 构建完成 {time.time()-t:.1f}s | 参数 "
          f"{sum(p.numel() for p in unet.parameters())/1e9:.3f} B（meta→GPU 直分）", flush=True)
    return unet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--no-gc", action="store_true", help="关闭梯度检查点")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16",
                    help="bf16 是训练要用的那个；fp16 只用于对照")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("无 CUDA")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    mem("起手")

    dt = torch.bfloat16 if a.dtype == 'bf16' else torch.float16
    unet = build_unet(dt)

    # 所有参数锁定为只读（训练只更新 LoRA）
    unet.requires_grad_(False)

    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(
        r=a.rank,
        lora_alpha=a.rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lcfg)
    n_train = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"LoRA 可训练参数 {n_train/1e6:.2f} M", flush=True)

    try:
        unet = unet.to("cuda", dtype=dt)
    except torch.cuda.OutOfMemoryError:
        print("!!! 致命：UNet + LoRA 权重本身放不进显存", flush=True)
        raise
    mem("放到 GPU")

    if not a.no_gc:
        unet.enable_gradient_checkpointing()
        print("梯度检查点已开", flush=True)

    unet.train()
    # 只让 LoRA 参数参与优化器状态
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2)
    mem("优化器建立")

    lat = a.res // 8
    gen = torch.Generator(device="cpu").manual_seed(0)
    ok = True
    try:
        for i in range(a.steps):
            x = torch.randn(1, 4, lat, lat, generator=gen, dtype=dt).cuda()
            enc = torch.randn(1, 77, 2048, generator=gen, dtype=dt).cuda()
            # SDXL 的 added_cond_kwargs：text_embeds (B,1280) + time_ids (B,6)
            tid = torch.randn(1, 6, generator=gen, dtype=dt).cuda()
            te = torch.randn(1, 1280, generator=gen, dtype=dt).cuda()
            ts = torch.tensor([999], device="cuda")
            noise = torch.randn_like(x)
            out = unet(x, ts, encoder_hidden_states=enc, added_cond_kwargs={
                "text_embeds": te, "time_ids": tid,
            }).sample
            loss = torch.nn.functional.mse_loss(out.float(), noise.float())
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            mem(f"第 {i+1} 步 前反向完")
            print(f"    loss = {loss.item():.4f}", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        ok = False
        print(f"!!! OOM 在 {a.res}px / rank {a.rank} / gc={not a.no_gc}", flush=True)
        print(str(e)[:400], flush=True)

    print("\n=== 判定 ===")
    if ok:
        _, peak = mem("最终")
        print(f"PASS：{a.res}px 可以训，峰值 {peak/GB:.2f} GB")
    else:
        print(f"FAIL：{a.res}px 放不下")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
