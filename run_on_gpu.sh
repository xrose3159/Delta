#!/bin/bash

set -e

echo "=========================================="
echo "DPO课程学习"
echo "=========================================="

# 修复 GCC 编译环境问题（用于 Triton/DeepSpeed 编译）
# 需要包含：include（stddef.h等）、include-fixed（limits.h等系统修正的头文件）
GCC_BASE="/mnt/petrelfs/share/gcc/gcc-8.5.0"
GCC_LIB="${GCC_BASE}/lib/gcc/x86_64-pc-linux-gnu/8.5.0"
export C_INCLUDE_PATH="${GCC_LIB}/include:${GCC_LIB}/include-fixed:${GCC_BASE}/include:${C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="${GCC_LIB}/include:${GCC_LIB}/include-fixed:${GCC_BASE}/include:${CPLUS_INCLUDE_PATH}"
export CPATH="${GCC_LIB}/include:${GCC_LIB}/include-fixed:${GCC_BASE}/include:${CPATH}"

# GPU 资源配置
export GPU_COUNT=${GPU_COUNT:-8}                    # 需要的 GPU 数量
export PARTITION=${PARTITION:-"raise"}              # 分区名称
export QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}         # quota 类型: reserved 或 spot

# Apptainer 容器配置
export USE_APPTAINER_FOR_VLLM=${USE_APPTAINER_FOR_VLLM:-true}
export APPTAINER_IMAGE=${APPTAINER_IMAGE:-"/mnt/petrelfs/zhuyun/vllm-cu128.sif"}

# 训练配置
export MODEL_PATH=${MODEL_PATH:-"/mnt/dhwfile/raise/user/zhuyun/Qwen3-VL-8B-Instruct"}
# export DATASET_PATH=${DATASET_PATH:-"/mnt/petrelfs/shangxiaoran/math_generation6/adaptive_training/dataset/wemath_standard_235b.json"}
# export DATASET_PATH=${DATASET_PATH:-"/mnt/dhwfile/raise/user/shangxiaoran/filter/round_1/filtered_dataset.json"}  # Euclid30K.json 过滤完
# export DATASET_PATH=${DATASET_PATH:-"/mnt/dhwfile/raise/user/shangxiaoran/dataset/Euclid30K_filter_no_null.json"}  # Euclid30K.json 打完level
# export DATASET_PATH=${DATASET_PATH:-"/mnt/dhwfile/raise/user/shangxiaoran/dataset/Euclid30K_filter_last.json"}
export DATASET_PATH=${DATASET_PATH:-"/mnt/dhwfile/raise/user/shangxiaoran/dataset/wemath_difficulty.json"}
export WORKSPACE_DIR=${WORKSPACE_DIR:-"/mnt/dhwfile/raise/user/shangxiaoran/new_pipeline_sft"}
export PIL_MAX_IMAGE_PIXELS=${PIL_MAX_IMAGE_PIXELS:-0}        # 0/none 表示无限制，避免 Pillow 解压炸弹告警

# 数据集配置
export ROUND_TOTAL_SAMPLES=${ROUND_TOTAL_SAMPLES:--1}        # 每轮使用的样本数
export MAX_ROUNDS=${MAX_ROUNDS:-5}                            # 训练轮数
export START_ROUND=${START_ROUND:-4}                          # 从第一轮开始
export MANUAL_SKIP_TO_UPGRADE=false
# export MANUAL_SKIP_MODEL_PATH="/mnt/dhwfile/raise/user/shangxiaoran/new_pipeline_sft/round_3/sft/sft_model_round3"
export RESUME_FROM_STEP=1                                     # 从第二步(问题分离+HES筛选)开始

# SFT 配置（替代 DPO）
export ENABLE_SFT=${ENABLE_SFT:-true}                         # 启用 SFT 训练（替代 DPO）
export SFT_NUM_TRAIN_EPOCHS=${SFT_NUM_TRAIN_EPOCHS:-3.0}     # SFT 训练轮数
export SFT_PER_DEVICE_TRAIN_BATCH_SIZE=${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-1}  # 每个设备的批次大小
export SFT_GRADIENT_ACCUMULATION_STEPS=${SFT_GRADIENT_ACCUMULATION_STEPS:-2}  # 梯度累积步数
export SFT_LEARNING_RATE=${SFT_LEARNING_RATE:-5e-6}          # SFT 学习率
export SFT_WARMUP_RATIO=${SFT_WARMUP_RATIO:-0.1}             # 预热比例
export SFT_LR_SCHEDULER_TYPE=${SFT_LR_SCHEDULER_TYPE:-cosine}  # 学习率调度器
export SFT_LOGGING_STEPS=${SFT_LOGGING_STEPS:-10}             # 日志步数
export SFT_SAVE_STEPS=${SFT_SAVE_STEPS:-100}                 # 保存步数
export SFT_CUTOFF_LEN=${SFT_CUTOFF_LEN:-10240}               # 最大序列长度
export SFT_FREEZE_VISION_TOWER=${SFT_FREEZE_VISION_TOWER:-true}   # 是否冻结视觉编码器 (true=冻结/不训练, false=训练)
export SFT_FREEZE_PROJECTOR=${SFT_FREEZE_PROJECTOR:-true}        # 是否冻结投影层 (true=冻结/不训练, false=训练)
export SFT_FREEZE_LLM=${SFT_FREEZE_LLM:-false}                     # 是否冻结语言模型 (true=冻结/不训练, false=训练)
export SFT_LR_FIRST_ROUND=${SFT_LR_FIRST_ROUND:-5e-6}
export SFT_LR_OTHER_ROUNDS=${SFT_LR_OTHER_ROUNDS:-1e-5}
export SFT_ENABLE_PACKING=${SFT_ENABLE_PACKING:-true}

# HES (High-Entropy Sum) 配置 - 筛选高熵正确题
export ENABLE_HES_FILTERING=${ENABLE_HES_FILTERING:-true}         # 是否启用 HES 筛选
export HES_TOP_RATIO=${HES_TOP_RATIO:-0.2}                        # 取 HES 前 20% 的正确题
export HES_PERCENTILE_CUTOFF=${HES_PERCENTILE_CUTOFF:-0.005}      # 每个样本取 top 0.5% 高熵 token
export HES_BATCH_SIZE=${HES_BATCH_SIZE:-8}                        # HES 计算批次大小
export HES_TENSOR_PARALLEL_SIZE=${HES_TENSOR_PARALLEL_SIZE:-8}    # HES 计算使用的 GPU 数量

# 保留 DPO 配置（用于内存等待时间等通用设置）
export DPO_MEMORY_WAIT_TIME=${DPO_MEMORY_WAIT_TIME:-20}      # 训练前等待 GPU 内存释放的时间（秒）

# LLM Judge 配置
export USE_LLM_JUDGE=${USE_LLM_JUDGE:-true}
export JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:-"/mnt/dhwfile/raise/user/shangxiaoran/models/Qwen3-30B-A3B-Instruct-2507"}
export JUDGE_TENSOR_PARALLEL_SIZE=${JUDGE_TENSOR_PARALLEL_SIZE:-8} 
export JUDGE_GPU_MEMORY_UTILIZATION=${JUDGE_GPU_MEMORY_UTILIZATION:-0.6}
export JUDGE_MAX_MODEL_LEN=${JUDGE_MAX_MODEL_LEN:-65536}     

# LLM Generator 配置（用于推理和错误分析的 vLLM 部署）
export USE_LLM_GENERATOR=${USE_LLM_GENERATOR:-true}
export LLM_GENERATOR_MODEL_PATH=${LLM_GENERATOR_MODEL_PATH:-"/mnt/dhwfile/raise/user/zhuyun/Qwen3-VL-235B-A22B-Thinking"}
# export LLM_GENERATOR_MODEL_PATH=${LLM_GENERATOR_MODEL_PATH:-"/mnt/dhwfile/raise/user/zhuyun/Qwen3-VL-30B-A3B-Thinking"}
export LLM_GENERATOR_TENSOR_PARALLEL_SIZE=${LLM_GENERATOR_TENSOR_PARALLEL_SIZE:-8}  
export LLM_GENERATOR_GPU_MEMORY_UTILIZATION=${LLM_GENERATOR_GPU_MEMORY_UTILIZATION:-0.8}  
export LLM_GENERATOR_MAX_MODEL_LEN=${LLM_GENERATOR_MAX_MODEL_LEN:-32768}  # 输入+输出
# Evaluation 配置
export EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.0}
export EVAL_TOP_P=${EVAL_TOP_P:-1.0}
export EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-16384}  
export EVAL_FREQUENCY_PENALTY=${EVAL_FREQUENCY_PENALTY:-0}  
export EVAL_TENSOR_PARALLEL_SIZE=${EVAL_TENSOR_PARALLEL_SIZE:-8}  

# Corrected CoT 生成配置（用于 DPO 训练的正确答案生成）
export CORRECTED_COT_MAX_TOKENS=${CORRECTED_COT_MAX_TOKENS:-32768} 

# Gemini API 配置（用于生成新题）
export GEMINI_API_KEY=${GEMINI_API_KEY:-"sk-Aqo44ZAdY4y5J9ro4uLIq5jaNXBDo5tavIS3WY0KiLpJrQt6"}
export GEMINI_MODEL=${GEMINI_MODEL:-"gemini-3-flash-preview-thinking"}
export GEMINI_BASE_URL=${GEMINI_BASE_URL:-"http://35.220.164.252:3888/v1/"}
export GEMINI_MAX_TOKENS=${GEMINI_MAX_TOKENS:-65536}              # Gemini 最大 token 数（输入+输出）
export GEMINI_MAX_OUTPUT_TOKENS=${GEMINI_MAX_OUTPUT_TOKENS:-65536}  # Gemini 最大输出 token 数

# Corrected CoT API 配置（用于生成错题的正确CoT，代替本地vLLM部署）
export CORRECTED_COT_API_KEY=${CORRECTED_COT_API_KEY:-"sk-Aqo44ZAdY4y5J9ro4uLIq5jaNXBDo5tavIS3WY0KiLpJrQt6"}
export CORRECTED_COT_MODEL=${CORRECTED_COT_MODEL:-"qwen3-vl-235b-a22b-thinking"}
export CORRECTED_COT_BASE_URL=${CORRECTED_COT_BASE_URL:-"http://35.220.164.252:3888/v1/"}
export CORRECTED_COT_MAX_WORKERS=${CORRECTED_COT_MAX_WORKERS:-30}  # API并发数 (降低以避免超时)

# 错误分析和自适应难度升级配置
export ENABLE_ERROR_ANALYSIS=${ENABLE_ERROR_ANALYSIS:-true}
export ERROR_ANALYZER_MODEL_PATH=${ERROR_ANALYZER_MODEL_PATH:-"/mnt/dhwfile/raise/user/zhuyun/Qwen3-VL-30B-A3B-Thinking"}
export ERROR_ANALYZER_TENSOR_PARALLEL_SIZE=${ERROR_ANALYZER_TENSOR_PARALLEL_SIZE:-8} 
export ERROR_ANALYZER_GPU_MEMORY_UTILIZATION=${ERROR_ANALYZER_GPU_MEMORY_UTILIZATION:-0.75}  # 错误分析模型的显存利用率（Qwen3-VL-30B需要更多显存）
export ERROR_ANALYZER_MAX_MODEL_LEN=${ERROR_ANALYZER_MAX_MODEL_LEN:-65536}  # 错误分析模型的最大上下文长度
export ERROR_ANALYZER_MAX_TOKENS=${ERROR_ANALYZER_MAX_TOKENS:-8192}  # 错误分析的最大输出token数

echo ""
echo "配置信息:"
echo "  GPU 数量: $GPU_COUNT"
echo "  分区: $PARTITION"
echo "  Quota 类型: $QUOTA_TYPE"
echo "  模型: $MODEL_PATH"
echo "  数据集: $DATASET_PATH"
echo "  样本数: $ROUND_TOTAL_SAMPLES"
echo "  训练轮数: $MAX_ROUNDS"
echo "  Judge LLM: $JUDGE_MODEL_PATH"
echo "  Generator LLM: $LLM_GENERATOR_MODEL_PATH"
echo "  错误分析LLM: $ERROR_ANALYZER_MODEL_PATH"
echo "  升级错题LLM: $GEMINI_MODEL"
echo ""

# ============================================
# 提交到 GPU 节点运行
# ============================================

cd /mnt/petrelfs/shangxiaoran/math_generation6

# 创建日志目录
LOG_DIR="$WORKSPACE_DIR/logs"
mkdir -p $LOG_DIR

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/adaptive_train_dpo_${TIMESTAMP}.log"

echo "日志文件: $LOG_FILE"
echo ""

srun -p $PARTITION \
     --gres=gpu:$GPU_COUNT \
     --quotatype=$QUOTA_TYPE \
     --job-name=update_sft \
     --output=$LOG_FILE \
     bash -c 'echo "==========================================" && \
              echo "🖥️  节点信息: $(hostname)" && \
              echo "🎮 分配的 GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" && \
              echo "📊 GPU 内存状态:" && \
              nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv && \
              echo "==========================================" && \
              source /mnt/petrelfs/shangxiaoran/anaconda3/bin/activate math  && \
              cd /mnt/petrelfs/shangxiaoran/math_generation6 && python -m adaptive_training'

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "结果位置: $WORKSPACE_DIR"
echo "日志文件: $LOG_FILE"
echo ""

