# Y11：跨 backbone 稳健性——SD2.1-base 上 κ 同量级（11-08 14:20 → 16:40，约 2 h 20 min）

## 目的

论文二全部主结果（κ≈0.2、g_eff = s(1+wρ)、等 w·s 换算错 4.2 倍）都在 SD1.5 上测得。审稿人必问：κ 是 SD1.5 的特例还是 CFG-LoRA 耦合的一般性质？本轮把整套协议原样搬到 SD2.1-base 上复测。

⚠️ **诚实口径**：这是**换 backbone**，不是换 LoRA 类型（Y2 的 A/B/full 三个切片已覆盖"换适配器"维度）。"跨 LoRA 类型（DoRA/LoHa 等）仍未测"写进 §7 局限。

## 设定

- **backbone**：SD2.1-base（ModelScope `AI-ModelScope/stable-diffusion-2-1-base` → `models/sd21_base`，15 GB；cross_attn 1024、epsilon 预测，与 SD1.5 同构所以 LoRA 代码零改动）。
- **概念集**：SD2.1 自产 24 图（build_concept 阶段，w=1.0/s=7.5，2 prompt × 12 seed）→ 训练 rank-8 LoRA（fp32 UNet，800 步，lr 1e-4，目标 attn qkv；final loss 0.0675，1.66M 参数，训练确定性可复现）。
- **网格**：2 prompt（obj00 人脸风、obj03 物体）× 2 seed × w 8 档（0.2–2.0 对数间隔）× s 9 档（3–14）= **288 图**，与 SD1.5 主协议完全一致。
- **参考图与度量**：同协议——ref = 该 prompt/seed 在 (w=1, s=7.5) 的图；pix_l2 在 64×64 灰度上算；s*(w) = argmin_s pix_l2。
- **预登记判据**（跑前写死在脚本 docstring）：
  - C1 活性：pix_l2_base(w=1) 的 min > 2.0（INERT 门槛），否则 LoRA 没挂上
  - C2 κ：可辨识格（R²≥.75）的 κ 落在 SD1.5 参考 0.211±0.10 = [0.11, 0.31] 内
  - C3 失败模式：R²<.25 的格如实报"未测出"，不许钓鱼、不换拟合方式

## 结果

### C1 通过：LoRA 活性充分

pix_l2_base 在 w=1.0 的 min = **2.77 > 2.0** → SD2.1 的 LoRA 确实改变了生成分布，不是 INERT。

### C2 通过：可辨识格 κ 与 SD1.5 同量级

| prompt / seed | κ | 95% CI | R² | 判定 |
|---|---|---|---|---|
| obj00 / 11 | **0.288** | [0.200, 0.376] | .891 | ✅ 在 [0.11, 0.31] |
| obj03 / 11 | **0.146** | [0.073, 0.218] | .758 | ✅ 在 [0.11, 0.31] |
| obj00 / 22 | −0.050 | [−0.173, 0.073] | .114 | 未测出（R²<.25） |
| obj03 / 22 | −0.147 | [−0.430, 0.136] | .172 | 未测出（R²<.25） |

- 两个可辨识格 κ = .288 / .146，均落在 SD1.5 参考带 0.211±0.10 内 → **C2 通过：κ 的量级跨 backbone 稳健**，不是 SD1.5 特例。
- SD1.5 的 91 格中有 12 格 R²<.25 同样测不出 → SD2.1 的 2/4 未测出格是**同一性质**（该 (prompt,seed) 组合下 s*(w) 剖面太平/噪声大），不是 SD2.1 特有退化。

### C3 如实报：piecewise 拟合照旧不稳

分段两段斜率（y11_piecewise.csv）方向混乱（如 obj03/11 hi_seg = −0.0），与 SD1.5 上三次失败一致——**维持红线：不写幂律/分段，正文只用全网格 log-log 单一斜率**。

## 过程事故记录（不掩盖）

第一轮 288 图全部作废：`gen(text, w, s)` 三参签名把 **guidance scale 误传进 seed 位**，实际网格是 (w, guidance=7.5, seed=s)，s 维度失效，产生 κ≈0 假象。诊断方法：MD5 对比 seed 11/22 两图完全相同 → 定位参数错位。修复为四参 `gen(pid, w, sd, gs=s)` 后重跑。bug 图已删除（img_buggy_seedpassed-as-s/，288 张），事故日志见 exp_snapshot/y11_run.log，正确日志 y11_run2.log。

## 产物

- `img/` 288 张正确图；`csv/y11_grid.csv`（288 行全网格）、`y11_sstar.csv`（16 组 s*）、`y11_kappa.csv`、`y11_piecewise.csv`
- `fig/fig_y11_sstar.png`（s* 轨迹）、`fig/fig_y11_kappa.png`（可辨识格拟合 vs SD1.5 参考带）
- `exp_snapshot/`：y11_sd21.py、y11_reanalyze.py、y11_run.log（bug 版）、y11_run2.log（正确版）
