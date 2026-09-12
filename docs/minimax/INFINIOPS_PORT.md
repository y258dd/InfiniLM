# 迁移 `lightning_attention` 到 InfiniOps（新架构）

> 配套补丁：`docs/minimax/infiniops-lightning-attention.patch`（CPU + NVIDIA + Ascend + pytest，7 文件 / +1019 行）
> 旧架构补丁（归档基线用）：`docs/minimax/lightning-attention-infinicore.patch`

## 1. 为什么要迁到 InfiniOps

InfiniCore 已完成重构（`26f7382d refactor!: reduce InfiniCore to unified component architecture (#1406)`，2026-09-11）：顶层只剩 `.gitmodules` / `CONTRIBUTING.md` / `LICENSE` / `README.md` / `submodules`，算子实现归属 **InfiniOps**。

- 旧的 `infinicore::` C++ API（`op::*`、`nn::*`、`Tensor`、graph）在新版顶层已不存在。
- **InfiniLM 主干尚未迁移**（上游 `270feb3e`），因此现在无法把 minimax **模型**迁到新版；能且应该迁移的是**算子**。
- 本补丁交付 `lightning_attention_infinilm` 的 **CPU + NVIDIA CUDA + 昇腾 ACLNN** 后端与 pytest。昇腾部分为 correctness-first 的 aclnn 组合实现，需在昇腾真机完成最终编译与回归。

## 2. 补丁内容（7 个文件）

| 文件 | 说明 |
|---|---|
| `src/base/lightning_attention_infinilm.h` | 算子类（接口 + 元数据 + assert 校验），所有后端共享 |
| `src/native/cpu/ops/lightning_attention_infinilm/lightning_attention_infinilm.h` | CPU 参考实现（float32 累加，支持 f32/f16/bf16 与任意 stride） |
| `src/native/cuda/ops/lightning_attention_infinilm/kernel.cuh` | CUDA device kernel（每 `(batch, head)` 一块，每线程一列状态） |
| `src/native/cuda/ops/lightning_attention_infinilm/kernel.h` | CUDA launcher（`CudaLightningAttentionInfinilm<Backend>`，按 dtype/index dtype 分发） |
| `src/native/cuda/nvidia/ops/lightning_attention_infinilm/kernel.h` | NVIDIA vendor 绑定（`Operator<..., kNvidia>`） |
| `src/native/ascend/ops/lightning_attention_infinilm/kernel.h` | 昇腾 ACLNN 组合实现（Mul + Matmul + Add，工作区状态池，支持 f32/f16/bf16 与 int32/int64 索引） |
| `tests/test_lightning_attention_infinilm.py` | pytest：4 组形状 × 3 种 dtype，分别断言输出与状态池 |

> 如果服务器已经应用过旧版 6 文件补丁（CPU + NVIDIA），不要重复应用累计补丁；直接应用 `infiniops-ascend-lightning-attention.patch` 即可。

## 3. 接口与语义设计

### 3.1 对齐目标与命名

按 `docs/operator-api-alignment.md` 的对齐顺序（PyTorch → vLLM → SGLang → ONNX → 库级公开封装 → CUDA/vendor → custom），lightning attention 在 PyTorch 中没有对应算子，最接近的公开实现是 Flash-Linear-Attention 的 `fused_recurrent_lightning_attn`。由于本算子带 InfiniLM 专有的"索引状态池"契约，按仓库既有 10+ 个先例（`paged_attention_infinilm`、`causal_softmax_infinilm`…）命名为 **`LightningAttentionInfinilm`**，注释中写明与 `dexp = exp(-slope)` 的对应关系。

### 3.2 参数顺序（InfiniOps 规范：输入 → 属性 → 输出）

```cpp
LightningAttentionInfinilm(q, k, v, slope,
                           initial_state, initial_state_indices, final_state_indices,
                           out)
```

### 3.3 递推语义

```
ratio[h] = exp(-slope[h])
S        = ratio[h] * S + outer(k_t[h], v_t[h])     # 先更新状态
out_t[h] = q_t[h] @ S                                # 再读状态（当前 token 权重为 1）
```

请求 `b` 从 `initial_state[initial_state_indices[b]]` 读状态，最终状态写回 `initial_state[final_state_indices[b]]`。

**关键契约：初始行不会被修改。** 读/写行相同时表现为就地更新；不同时等价于"先复制到目标行、再累加"。CPU 与 CUDA 实现都遵守，测试特意让读行 ≠ 写行来覆盖该契约（CUDA 侧通过在 kernel 开头把初始行协作拷贝到目标行、再在目标行上累加来保证）。

## 4. 服务器上需要下载/安装什么

### 4.1 基础依赖（必装）

| 依赖 | 版本/用途 | 安装方式 |
|---|---|---|
| Linux x86_64 | Ubuntu 22.04 等 | 租的机器自带 |
| **CMake** | ≥ 3.18（两个仓库都是 CMake 工程） | NGC 镜像通常自带，先 `cmake --version`；没有则 `pip install cmake` 或 `apt-get install -y cmake` |
| **C++ 编译器** | gcc-11+ / clang-16+（C++17） | `apt-get install -y build-essential` |
| **OpenMP** | CPU 后端 `find_package(OpenMP REQUIRED)` | `apt-get install -y libgomp1`（多半自带） |
| **Python** | ≥ 3.10 + pip | 镜像自带 |
| Python 包 | `torch`（测试参考实现）、`pytest`、`scikit-build-core` | `pip install torch pytest`；`.[dev]` 会带上构建依赖 |
| **InfiniRT** | InfiniOps 的前置依赖，**必须先装** | `git clone --recursive https://github.com/InfiniTensor/InfiniRT.git` + CMake 安装 |
| **InfiniOps** | 我们的补丁落在这里 | `git clone https://github.com/InfiniTensor/InfiniOps.git` |
| 网络 | configure 阶段访问 GitHub | CUDA 构建会 FetchContent 下载 CUTLASS（固定 commit + SHA256）；CPU 构建不需要 |

### 4.2 测 NVIDIA GPU 额外需要

| 依赖 | 说明 |
|---|---|
| **CUDA Toolkit（nvcc）** | ≥ 12；NGC PyTorch 镜像自带。注意 **InfiniRT 也要用 `-DWITH_NVIDIA=ON` 重新编译安装** |
| **CUTLASS** | 由 CMake `FetchContent` 自动下载，**不需要手动装**；网络受限时要预置或配代理 |
| **PyTorch（CUDA 版）** | 测试参考实现需要；NGC 镜像自带 |

### 4.3 测 Ascend 额外需要

| 依赖 | 说明 |
|---|---|
| **CANN Toolkit** | 建议使用与你租用的昇腾机器、驱动/固件匹配的版本；安装后 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，并确认 `ASCEND_HOME_PATH` 指向 toolkit 根目录 |
| **昇腾驱动/固件** | `npu-smi info` 能看到设备，容器内能看到 `/dev/davinci*` 和 `/dev/davinci_manager` |
| **PyTorch + torch_npu** | 测试参考实现和 `torch.npu` device/stream 需要；版本必须与 Python、CANN 匹配 |
| **ACLNN 算子库** | 本实现使用 `aclnnMul`、`aclnnAdd`、`aclnnMatmul`，由 CANN 自带；不需要额外安装 |
| **AscendC 编译器** | 本算子走 ACLNN，不走 AscendC；若只验证本算子，可用 `-DBUILD_ASCEND_CUSTOM=OFF` 跳过 `ccec` 构建 |

最小构建命令（服务器上可直接逐行粘贴）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
npu-smi info
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

```bash
git clone --recursive https://github.com/InfiniTensor/InfiniRT.git
cmake -S InfiniRT -B build-rt -DCMAKE_INSTALL_PREFIX=$HOME/infinirt -DWITH_CPU=ON -DWITH_ASCEND=ON
cmake --build build-rt -j
cmake --install build-rt
```

```bash
git clone https://github.com/InfiniTensor/InfiniOps.git
cd InfiniOps
git apply /path/to/docs/minimax/infiniops-ascend-lightning-attention.patch
python -m pip install ".[dev]" --break-system-packages --config-settings=cmake.define.INFINI_RT_ROOT=$HOME/infinirt --config-settings=cmake.define.WITH_CPU=ON --config-settings=cmake.define.WITH_ASCEND=ON --config-settings=cmake.define.BUILD_ASCEND_CUSTOM=OFF --config-settings=cmake.define.INFINI_OPS_OPS=lightning_attention_infinilm
pytest tests/test_lightning_attention_infinilm.py --devices ascend -v
```

- 上面第 2 段命令按“服务器已经应用过旧版 CPU/NVIDIA 补丁”的情况使用昇腾增量补丁；如果是干净 clone，则改用累计补丁 `infiniops-lightning-attention.patch`。
- 昇腾真机首次验证时，优先先跑 `--devices cpu` 确认测试本身和参考实现正常，再跑 `--devices ascend`。
- 若同一个 conda/venv 里同时有 CUDA 版 PyTorch 和 torch_npu，必须按华为官方要求处理冲突；最简单是单独的昇腾环境。

### 4.4 完整命令（NVIDIA Linux 服务器）

```bash
# 0) 自检
cmake --version          # >= 3.18
g++ --version            # >= 11
python3 --version        # >= 3.10
nvcc --version           # 只有 GPU 机器需要

# 1) 编译安装 InfiniRT（GPU 机器同时打开 WITH_NVIDIA）
git clone --recursive https://github.com/InfiniTensor/InfiniRT.git
cmake -S InfiniRT -B build-rt \
      -DCMAKE_INSTALL_PREFIX=$HOME/infinirt \
      -DWITH_CPU=ON -DWITH_NVIDIA=ON
cmake --build build-rt -j
cmake --install build-rt

# 2) 取 InfiniOps 并应用补丁
git clone https://github.com/InfiniTensor/InfiniOps.git
cd InfiniOps
git apply /path/to/docs/minimax/infiniops-ascend-lightning-attention.patch

# 3) 构建安装 InfiniOps（CPU + NVIDIA）
python -m pip install ".[dev]" \
  --config-settings=cmake.define.INFINI_RT_ROOT=$HOME/infinirt \
  --config-settings=cmake.define.WITH_CPU=ON \
  --config-settings=cmake.define.WITH_NVIDIA=ON

# 4) 跑本算子的 pytest（device fixture 会自动覆盖 cpu 与 cuda）
pytest tests/test_lightning_attention_infinilm.py -v
```

- 只想先验 CPU：把第 1、3 步的 `-DWITH_NVIDIA=ON` 去掉即可（更快，也不需要 CUDA/CUTLASS）。
- 想只编本算子加速 configure/build：追加 `--config-settings=cmake.define.INFINI_OPS_OPS=lightning_attention_infinilm`。

## 5. 实测记录

### 5.1 NVIDIA 真机验证

| 项目 | 结果 |
|---|---|
| 环境 | Ubuntu 24.04 容器；CUDA Toolkit 12.8.61；CMake 3.31.4；g++ 13.3.0；Python 3.12.3 |
| GPU | NVIDIA GeForce RTX 5090（**sm_120 / Blackwell**），驱动 610.43.02（CUDA UMD 13.3） |
| InfiniRT | 编译安装成功（`-DWITH_CPU=ON -DWITH_NVIDIA=ON -DCMAKE_CUDA_ARCHITECTURES=120`） |
| InfiniOps | wheel 构建成功：`infiniops-0.1.0-cp312-cp312-linux_x86_64.whl`（2.75 MB）并安装；全部 CUDA 源码在 sm_120 下编译通过（含依赖 CUTLASS 的算子） |
| 本算子测试 | `pytest tests/test_lightning_attention_infinilm.py -v` → **48 passed** |
| 覆盖 | 后端 CPU + CUDA；dtype float32/float16/bfloat16；4 组形状（`seq_len` 为 1 与 5，`batch` 为 1~3）；「输出」与「状态池」两个断言用例；读行≠写行的「初始行只读」契约 |

> 构建时必须显式指定 `CMAKE_CUDA_ARCHITECTURES=120`：InfiniOps 默认按 sm_80 生成 cubin，而 sm_80 的二进制在 sm_120 上无法运行（会报 `no kernel image is available for execution on the device`）。

联调期间发现并修复的三处问题（均已包含在补丁中）：

1. `initial_state` 曾声明为 `const Tensor`，而该张量需要被就地更新 → 编译期 `invalid type conversion`（已改为非常量视图，`q/k/v/slope/indices` 保持 const）；
2. 测试曾使用 `final_index = initial_index + 1`，使请求 `b=0` 的**写入行**成为请求 `b=1` 的**读取行**——CPU 串行语义与 CUDA 并行语义因此不一致（已改为跨请求读写行互不相交，并把「请求独立、可并发执行」写入算子契约）；
3. torch 参考实现未模拟「状态按张量 dtype 存储」的舍入，低估了 bf16 误差（参考实现已同步每步舍入；bf16 容差调为 `4e-2`，理由是递推累积）。

### 5.2 昇腾后端实现状态（静态完成，待真机验证）

昇腾目录：`src/native/ascend/ops/lightning_attention_infinilm/kernel.h`。

实现思路：CANN 没有现成的 fused lightning-attention 算子，因此组合 ACLNN 算子完成递推：

```
ratio   = exp(-slope)                              # 每个 head 一次
decayed = Mul(state, ratio)                        # ratio 形状 [H, 1, 1]
outer   = Matmul(k_t [H, D, 1], v_t [H, 1, D])
state   = Add(decayed, outer)
out_t   = Matmul(q_t [H, 1, D], state)
```

每个 request 先在 workspace 行里暂存初始状态，在该行上跑完所有 token，最后异步写回 `final_state_indices` 指定的目标行。因此：

- `initial_state_indices` 指向的源行在整个 request 内只读，读行和写行不同不会被提前污染；
- 不同 request 的 state workspace 在同一 stream 上按顺序复用，跨 request 可并发的前提仍是“目的行不能是另一个 request 的源行”；
- 状态工作区、decay 临时区、outer 临时区和 ratio 区都通过 `GetWorkspacePool().Ensure()` 获取，避免在算子内部直接管理临时内存；
- 输入索引和 slope 在开始时统一同步后读回 host，状态 staging/writeback 使用 `aclrtMemcpyAsync(..., stream)`，保持算子正常异步语义。

支持范围与测试矩阵一致：`float32` / `float16` / `bfloat16`，索引 `int32` / `int64`，输出与状态池分别断言。

> 当前只在 Windows 上用 mock CANN/ACLNN 头做了 C++17 语法检查，并做了补丁反向校验；**尚未在昇腾真机编译或运行**。真机上若报错，优先检查 `/usr/local/Ascend/.../set_env.sh`、`ASCEND_HOME_PATH`、torch_npu 与 CANN 版本，以及 `aclrtMemcpyAsync` 的流/指针合法性。

## 6. 已知限制与下一步

- 本机（Windows，无 CANN）**未做真机编译**；已完成 C++17 mock-header 语法检查、补丁反向校验（`git apply --check -R` 通过）和 `git diff --check`。NVIDIA 真机回归为 48/48 通过，昇腾真机回归仍待执行。
- CUDA kernel 假定 `head_dim` 能放进一个 block（`head_dim <= Backend::max_block_size`，launcher 里有 assert），典型 MiniMax `head_dim = 128` 满足。
- **Ascend 后端**：ACLNN 组合实现已落地，下一步是在昇腾真机执行第 4.3 节命令并修正平台差异；若性能不达标，再考虑 AscendC fused kernel。
- **InfiniLM 侧适配**：等上游完成 InfiniLM → 新栈迁移后，把 `MiniMaxLightningAttention` 的调用点换成 `infini::ops::LightningAttentionInfinilm`（一处调用 + 状态池形状对齐），模型其余代码不动。