#!/bin/bash
# GEHRL 项目-单任务运行脚本 (传参版)
# 运行方式：bash run_gehrl_single.sh [GPU卡号] [实验编号-e] [随机种子-r]
# 示例：bash run_gehrl_single.sh 0 1 1  对应 → CUDA=0 -e=1 -r=1
# 日志路径: EduSim/Experiment_logs/KESassist15/Experiment_<e>/<r>.txt
set -euo pipefail  # 出错立即停止，未定义变量报错，管道错误捕获，安全优先

# ===================== 固定配置区 (无需修改，一次配置永久生效) =====================
PROJECT_DIR="/home/wrz/code/GEHRL"
CUDA_VERSION="11.6"
CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
CONDA_ENV_NAME="uv"
SIMULATOR="KESassist15"
AGENT="HRL"
MAX_STEPS=20
MAX_EPISODE_NUM=15000

# ===================== 1. 传参校验 & 获取参数 =====================
if [ $# -ne 3 ]; then
    echo -e "\033[31m【参数错误】正确运行格式：\033[0m bash $0 [GPU卡号] [实验编号-e] [随机种子-r]"
    echo -e "\033[33m【示例】\033[0m bash $0 0 1 1  → CUDA_VISIBLE_DEVICES=0 -e=1 -r=1"
    echo -e "\033[33m【示例】\033[0m bash $0 1 1 2  → CUDA_VISIBLE_DEVICES=1 -e=1 -r=2"
    echo -e "\033[33m【示例】\033[0m bash $0 2 1 3  → CUDA_VISIBLE_DEVICES=2 -e=1 -r=3"
    exit 1
fi

# 接收外部传入的3个核心参数
GPU_ID=$1          # 第1个参数：GPU卡号 → 赋值给CUDA_VISIBLE_DEVICES
EXP_IDX=$2         # 第2个参数：实验编号 → 对应 -e 参数
REPEAT_NUM=$3      # 第3个参数：随机种子 → 对应 -r 参数

# ===================== 2. 进入项目目录 =====================
echo -e "\033[32m====================================\033[0m"
echo -e "\033[32m[1] 进入项目目录: ${PROJECT_DIR}\033[0m"
cd ${PROJECT_DIR} || { echo -e "\033[31m错误: 项目目录不存在\033[0m"; exit 1; }

# ===================== 3. 激活Python venv虚拟环境 =====================
echo -e "\033[32m[2] 激活Python虚拟环境\033[0m"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source .venv/bin/activate
else
    echo -e "\033[36m提示: 已激活Python虚拟环境，无需重复激活\033[0m"
fi

# ===================== 4. 配置CUDA环境变量 (最高优先级+LD_PRELOAD兜底) =====================
echo -e "\033[32m[3] 配置CUDA ${CUDA_VERSION} 环境变量(强制生效)\033[0m"
export CUDA_HOME=${CUDA_HOME}
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export LD_PRELOAD=${CUDA_HOME}/lib64/libcudnn.so.8:${CUDA_HOME}/lib64/libcuda.so

# ===================== 5. 激活conda环境 uv =====================
echo -e "\033[32m[4] 激活conda环境: ${CONDA_ENV_NAME}\033[0m"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV_NAME}
else
    echo -e "\033[36m提示: 已激活conda环境 ${CONDA_ENV_NAME}，无需重复激活\033[0m"
fi

# ===================== 6. 核心环境变量配置 =====================
export GEHRL_DEVICE_TARGET=GPU
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# ===================== ✅ 醒目打印本次运行的所有核心参数 (重点！) =====================
echo -e "\033[33m====================================\033[0m"
echo -e "\033[31m【本次运行核心参数】\033[0m"
echo -e "\033[31mCUDA_VISIBLE_DEVICES = ${GPU_ID}\033[0m"
echo -e "\033[31m实验编号  -e         = ${EXP_IDX}\033[0m"
echo -e "\033[31m随机种子  -r         = ${REPEAT_NUM}\033[0m"
echo -e "\033[33m====================================\033[0m"

# ===================== 执行核心运行命令 =====================
echo -e "\033[32m[5] 开始执行任务 → uv run runSim.py\033[0m"
uv run runSim.py -s ${SIMULATOR} -e ${EXP_IDX} -a ${AGENT} -r ${REPEAT_NUM} -m ${MAX_STEPS} --max_episode_num ${MAX_EPISODE_NUM}

# 任务完成提示
echo -e "\033[32m✅ 任务执行完成！日志文件：EduSim/Experiment_logs/${SIMULATOR}/Experiment_${EXP_IDX}/${REPEAT_NUM}.txt\033[0m"
echo -e "\033[32m====================================\033[0m"