#!/bin/bash
# GEMM 配置调优脚本
# 用法:
#   ./tune_gemm_config.sh show          # 显示当前配置
#   ./tune_gemm_config.sh set_fp8  N K STAGES   # 设置 FP8 配置
#   ./tune_gemm_config.sh set_bf16 N K STAGES   # 设置 BF16 配置
#   ./tune_gemm_config.sh set_all  BF16_N BF16_K BF16_STAGES FP8_N FP8_K FP8_STAGES  # 一次性设置全部

CONFIG_FILE="python/triton_dist/function/nvidia/common.py"

# 获取脚本所在目录的项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_FILE}"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: 找不到配置文件: $CONFIG_PATH"
    exit 1
fi

show_config() {
    echo "============================================"
    echo "  当前 GEMM 配置 (from common.py)"
    echo "============================================"
    echo ""
    
    # BF16 配置
    BF16_N=$(grep -oP "FWD_GEMM_BLOCK_SIZE_N=\K[0-9]+" "$CONFIG_PATH")
    BF16_K=$(grep -oP "BF16_FWD_GEMM_BLOCK_SIZE_K=\K[0-9]+" "$CONFIG_PATH")
    BF16_STAGES=$(grep -oP "BF16_FWD_GEMM_NUM_STAGES=\K[0-9]+" "$CONFIG_PATH")
    
    echo "  [BF16]"
    echo "    BLOCK_SIZE_N  = ${BF16_N:-未设置}"
    echo "    BLOCK_SIZE_K  = ${BF16_K:-未设置}"
    echo "    NUM_STAGES    = ${BF16_STAGES:-未设置}"
    echo ""
    
    # FP8 配置
    FP8_N=$(grep -oP "FP8_FWD_GEMM_BLOCK_SIZE_N=\K[0-9]+" "$CONFIG_PATH")
    FP8_K=$(grep -oP "FP8_FWD_GEMM_BLOCK_SIZE_K=\K[0-9]+" "$CONFIG_PATH")
    FP8_STAGES=$(grep -oP "FP8_FWD_GEMM_NUM_STAGES=\K[0-9]+" "$CONFIG_PATH")
    
    echo "  [FP8]"
    echo "    BLOCK_SIZE_N  = ${FP8_N:-未设置}"
    echo "    BLOCK_SIZE_K  = ${FP8_K:-未设置}"
    echo "    NUM_STAGES    = ${FP8_STAGES:-未设置}"
    echo ""
    echo "  BLOCK_SIZE_M    = 128 (固定，不可调)"
    echo "============================================"
}

set_fp8() {
    local N=$1
    local K=$2
    local STAGES=$3
    
    if [ -z "$N" ] || [ -z "$K" ] || [ -z "$STAGES" ]; then
        echo "用法: $0 set_fp8 <BLOCK_SIZE_N> <BLOCK_SIZE_K> <NUM_STAGES>"
        echo "示例: $0 set_fp8 128 128 4"
        exit 1
    fi
    
    echo "设置 FP8 配置: N=$N, K=$K, stages=$STAGES"
    
    sed -i "s/FP8_FWD_GEMM_BLOCK_SIZE_N=[0-9]*/FP8_FWD_GEMM_BLOCK_SIZE_N=${N}/" "$CONFIG_PATH"
    sed -i "s/FP8_FWD_GEMM_BLOCK_SIZE_K=[0-9]*/FP8_FWD_GEMM_BLOCK_SIZE_K=${K}/" "$CONFIG_PATH"
    sed -i "s/FP8_FWD_GEMM_NUM_STAGES=[0-9]*/FP8_FWD_GEMM_NUM_STAGES=${STAGES}/" "$CONFIG_PATH"
    
    echo "✓ FP8 配置已更新"
    echo ""
    show_config
}

set_bf16() {
    local N=$1
    local K=$2
    local STAGES=$3
    
    if [ -z "$N" ] || [ -z "$K" ] || [ -z "$STAGES" ]; then
        echo "用法: $0 set_bf16 <BLOCK_SIZE_N> <BLOCK_SIZE_K> <NUM_STAGES>"
        echo "示例: $0 set_bf16 256 64 3"
        exit 1
    fi
    
    echo "设置 BF16 配置: N=$N, K=$K, stages=$STAGES"
    
    sed -i "s/FWD_GEMM_BLOCK_SIZE_N=[0-9]*/FWD_GEMM_BLOCK_SIZE_N=${N}/" "$CONFIG_PATH"
    sed -i "s/BF16_FWD_GEMM_BLOCK_SIZE_K=[0-9]*/BF16_FWD_GEMM_BLOCK_SIZE_K=${K}/" "$CONFIG_PATH"
    sed -i "s/BF16_FWD_GEMM_NUM_STAGES=[0-9]*/BF16_FWD_GEMM_NUM_STAGES=${STAGES}/" "$CONFIG_PATH"
    
    echo "✓ BF16 配置已更新"
    echo ""
    show_config
}

set_all() {
    local BF16_N=$1
    local BF16_K=$2
    local BF16_STAGES=$3
    local FP8_N=$4
    local FP8_K=$5
    local FP8_STAGES=$6
    
    if [ -z "$FP8_STAGES" ]; then
        echo "用法: $0 set_all <BF16_N> <BF16_K> <BF16_STAGES> <FP8_N> <FP8_K> <FP8_STAGES>"
        echo "示例: $0 set_all 256 64 3 128 128 4"
        exit 1
    fi
    
    echo "设置全部配置:"
    echo "  BF16: N=$BF16_N, K=$BF16_K, stages=$BF16_STAGES"
    echo "  FP8:  N=$FP8_N, K=$FP8_K, stages=$FP8_STAGES"
    
    sed -i "s/FWD_GEMM_BLOCK_SIZE_N=[0-9]*/FWD_GEMM_BLOCK_SIZE_N=${BF16_N}/" "$CONFIG_PATH"
    sed -i "s/BF16_FWD_GEMM_BLOCK_SIZE_K=[0-9]*/BF16_FWD_GEMM_BLOCK_SIZE_K=${BF16_K}/" "$CONFIG_PATH"
    sed -i "s/BF16_FWD_GEMM_NUM_STAGES=[0-9]*/BF16_FWD_GEMM_NUM_STAGES=${BF16_STAGES}/" "$CONFIG_PATH"
    sed -i "s/FP8_FWD_GEMM_BLOCK_SIZE_N=[0-9]*/FP8_FWD_GEMM_BLOCK_SIZE_N=${FP8_N}/" "$CONFIG_PATH"
    sed -i "s/FP8_FWD_GEMM_BLOCK_SIZE_K=[0-9]*/FP8_FWD_GEMM_BLOCK_SIZE_K=${FP8_K}/" "$CONFIG_PATH"
    sed -i "s/FP8_FWD_GEMM_NUM_STAGES=[0-9]*/FP8_FWD_GEMM_NUM_STAGES=${FP8_STAGES}/" "$CONFIG_PATH"
    
    echo "✓ 全部配置已更新"
    echo ""
    show_config
}

# 主入口
case "${1}" in
    show)
        show_config
        ;;
    set_fp8)
        set_fp8 "$2" "$3" "$4"
        ;;
    set_bf16)
        set_bf16 "$2" "$3" "$4"
        ;;
    set_all)
        set_all "$2" "$3" "$4" "$5" "$6" "$7"
        ;;
    *)
        echo "GEMM 配置调优工具"
        echo ""
        echo "用法:"
        echo "  $0 show                                        显示当前配置"
        echo "  $0 set_fp8  <N> <K> <STAGES>                   设置 FP8 配置"
        echo "  $0 set_bf16 <N> <K> <STAGES>                   设置 BF16 配置"
        echo "  $0 set_all  <BF16_N> <BF16_K> <BF16_S> <FP8_N> <FP8_K> <FP8_S>  设置全部"
        echo ""
        echo "示例:"
        echo "  $0 show"
        echo "  $0 set_fp8 128 128 4"
        echo "  $0 set_bf16 256 64 3"
        echo "  $0 set_all 256 64 3 128 128 4"
        echo ""
        echo "可选的常见配置组合:"
        echo "  BF16: N=256,K=64,stages=3 | N=256,K=64,stages=4 | N=128,K=128,stages=3"
        echo "  FP8:  N=128,K=128,stages=4 | N=128,K=128,stages=3 | N=256,K=128,stages=3"
        ;;
esac
