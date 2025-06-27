#!/bin/bash

START_YM=199201
END_YM=201712

START_YEAR=${START_YM:0:4}
START_MONTH=${START_YM:4:2}
END_YEAR=${END_YM:0:4}
END_MONTH=${END_YM:4:2}

NUM_MONTHS=$(( (END_YEAR - START_YEAR)*12 + (10#$END_MONTH - 10#$START_MONTH + 1) ))

echo "起始年月: $START_YM, 终止年月: $END_YM, 总任务数: $NUM_MONTHS"

# 提交任务数组，每个 task 自行计算年月
sbatch --array=0-$(($NUM_MONTHS - 1))%120 submit_wmb_monthly.sh $START_YM