#!/bin/bash

# Flash Attention 커널 튜닝 스크립트 - V Local Caching 최적화 버전
# BLOCK_M, BLOCK_N 조정으로 LDS 사용량과 occupancy 튜닝

KERNEL_FILE="ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl"

function set_config() {
    local block_m=$1
    local block_n=$2
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "설정: BLOCK_M=$block_m, BLOCK_N=$block_n"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    sed -i "23s/.*/\#define BLOCK_M $block_m/" "$KERNEL_FILE"
    sed -i "27s/.*/\#define BLOCK_N $block_n/" "$KERNEL_FILE"
    
    # LDS 사용량 계산
    local lds_kb=$(( (32 + 32) * block_n * 8 / 1024 ))
    echo "LDS 사용량: ${lds_kb}KB"
    echo "✓ 설정 완료"
}

function show_config() {
    echo "==================================="
    echo "현재 설정:"
    grep "^#define BLOCK_M" "$KERNEL_FILE"
    grep "^#define BLOCK_N" "$KERNEL_FILE"
    
    # LDS 계산
    local block_n=$(grep "^#define BLOCK_N" "$KERNEL_FILE" | awk '{print $3}')
    local lds_kb=$(( (32 + 32) * block_n * 8 / 1024 ))
    echo "LDS 사용량: ${lds_kb}KB"
    echo "==================================="
}

case "$1" in
    "combo_A")
        echo "🔧 combo_A: 기본 (BLOCK_M=64, BLOCK_N=32)"
        echo "   - LDS: 16KB (안전)"
        echo "   - Occupancy: 높음"
        set_config 64 32
        ;;
    "combo_B")
        echo "🔧 combo_B: 큰 타일 (BLOCK_M=64, BLOCK_N=64)"
        echo "   - LDS: 32KB (중간)"
        echo "   - Barrier 절반, 더 빠를 수 있음"
        set_config 64 64
        ;;
    "combo_C")
        echo "🔧 combo_C: 작은 Query (BLOCK_M=32, BLOCK_N=32)"
        echo "   - LDS: 16KB"
        echo "   - Work-group 많음"
        set_config 32 32
        ;;
    "combo_D")
        echo "🔧 combo_D: 작은 Query, 큰 타일 (BLOCK_M=32, BLOCK_N=64)"
        echo "   - LDS: 32KB"
        echo "   - 균형잡힌 설정"
        set_config 32 64
        ;;
    "combo_AGGRESSIVE")
        echo "🔧 공격적 설정 (BLOCK_M=64, BLOCK_N=128)"
        echo "   ⚠️  LDS: 64KB - 위험할 수 있음!"
        set_config 64 128
        ;;
    "show")
        show_config
        ;;
    "custom")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "사용법: ./test_configs.sh custom [BLOCK_M] [BLOCK_N]"
            exit 1
        fi
        set_config "$2" "$3"
        ;;
    *)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Flash Attention 최적화 커널 튜닝 스크립트"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "핵심 최적화: K + V 모두 Local Memory 캐시!"
        echo "  → Global memory 접근 1000배 감소"
        echo "  → 예상 성능: 3-5배 향상"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "권장 테스트 순서:"
        echo "  combo_A  - 기본 (64×32, LDS 16KB) ← 여기서 시작"
        echo "  combo_B  - 큰 타일 (64×64, LDS 32KB) ← 추천!"
        echo "  combo_C  - 작은 Query (32×32, LDS 16KB)"
        echo "  combo_D  - 균형 (32×64, LDS 32KB)"
        echo "  combo_AGGRESSIVE - 공격적 (64×128, LDS 64KB)"
        echo ""
        echo "기타:"
        echo "  show            - 현재 설정 확인"
        echo "  custom [M] [N]  - 커스텀 설정"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "파라미터 설명:"
        echo "  BLOCK_M: Query 블록 크기 (클수록 재사용↑, work-group↓)"
        echo "  BLOCK_N: KV 타일 크기 (클수록 LDS↑, barrier↓)"
        echo ""
        echo "LDS 사용량:"
        echo "  BLOCK_N=32:  16KB (안전)"
        echo "  BLOCK_N=64:  32KB (보통)"
        echo "  BLOCK_N=128: 64KB (위험할 수 있음)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        show_config
        ;;
esac
