#!/bin/bash -l
#SBATCH --job-name=transform
#SBATCH -A hfdrake_lab
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/gtrans_%A_%a.out
#SBATCH --error=logs/gtrans_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G

# 检查是否提供起始年月
if [ -z "$1" ]; then
    echo "错误：请提供 START_YYYYMM 参数（例如 sbatch --array=0-XX submit_wmb_monthly.sh 201101）"
    exit 1
fi

# 起始年月
START_YM=$1
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# 解析起始年月
START_YEAR=${START_YM:0:4}
START_MONTH=${START_YM:4:2}

# 计算当前子任务的目标年月
YEAR=$((START_YEAR + (10#$START_MONTH + TASK_ID - 1) / 12))
MONTH=$(((10#$START_MONTH + TASK_ID - 1) % 12 + 1))

# 格式化为 YYYYMM
printf -v THIS_YM "%04d%02d" $YEAR $MONTH

# 验证范围（可选）
if ! [[ $YEAR -ge 1992 && $YEAR -le 2017 && $MONTH -ge 1 && $MONTH -le 12 ]]; then
    echo "错误：计算得到的年月不在有效范围内: $THIS_YM"
    exit 1
fi

echo "当前任务 ID: $TASK_ID -> 正在处理: $THIS_YM"

# 加载 Python 环境
source ~/.bashrc
conda activate docs_env_xwmb

# 运行 Python 脚本
python monthly_wmb.py $THIS_YM
