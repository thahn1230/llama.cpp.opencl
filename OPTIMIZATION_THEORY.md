# Flash Attention 최적화 이론 배경

## 🎯 핵심 문제: 레지스터 Spill

### GPU 메모리 계층 구조

```
┌─────────────────────────────────────────┐
│ 레지스터 (Register File)                │  ← 가장 빠름 (1 cycle)
│ - 각 스레드 전용                         │  ← 크기: 64-256 registers/thread
│ - 가장 제한적                            │  ← 여기가 부족하면 큰일남!
└─────────────────────────────────────────┘
              ↓ spill
┌─────────────────────────────────────────┐
│ Local Memory (LDS/Shared)               │  ← 빠름 (20-40 cycles)
│ - Work-group 공유                       │  ← 크기: 32-64KB
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ Global Memory (VRAM)                    │  ← 느림 (200-400 cycles)
│ - 모든 스레드 공유                       │  ← 크기: 수 GB
└─────────────────────────────────────────┘
```

### Spill이란?

**정의**: 레지스터가 부족해서 변수를 Global Memory에 저장하는 현상

**예시**:
```c
// 원래 의도 (레지스터에 저장)
float4 o_acc[32];  // 512 bytes
o_acc[i] = o_acc[i] + value;  // 1 cycle

// Spill 발생 시 (메모리에 저장)
// 컴파일러가 자동으로 이렇게 바꿈:
o_acc[i] = load_from_memory(addr) + value;  // 200 cycles
store_to_memory(addr, o_acc[i]);            // 200 cycles
// 총 400 cycles!
```

### 실제 영향

```
레지스터 사용량:
- DK=128, DV=128
- q_priv[32] = 32 × 16 bytes = 512 bytes
- o_acc[32] = 32 × 16 bytes = 512 bytes
- 총: 1024 bytes per thread

GPU 레지스터 한계:
- Adreno: 보통 64-128 registers = 256-512 bytes/thread
- 초과!

Spill 발생:
- o_acc 접근: 매 KV 토큰마다 수십 번
- n_kv=2048이면: 수만 번 spill
- 각 spill: 400 cycles
- 총 overhead: 수백만 cycles → 초 단위 지연!
```

---

## ✅ 해결책: DV Tiling

### 아이디어

**"큰 배열을 한 번에 들지 말고, 작은 조각씩 순차 처리"**

### 기존 방식
```c
// 한 타일의 전체 output 동시 계산
float4 o_acc[32];  // 512 bytes

for (k_tile in KV) {
    load K[BLOCK_N][32];
    load V[BLOCK_N][32];
    
    for (j in BLOCK_N) {
        score = Q · K[j];
        o_acc += score * V[j];  // 전체 DV=128 업데이트
    }
}

write(o_acc);  // 한 번에 전체 write
```

### DV Tiling 방식
```c
// DV를 4조각으로 나눔 (DV_TILE=32)
for (dv_tile = 0; dv_tile < 4; ++dv_tile) {
    float4 o_tile[8];  // 128 bytes만! (75% 감소)
    
    for (k_tile in KV) {
        load K[BLOCK_N][32];
        load V[BLOCK_N][8];  // 이 타일의 V만
        
        for (j in BLOCK_N) {
            score = Q · K[j];
            o_tile += score * V[j][dv_tile];  // 일부만 업데이트
        }
    }
    
    write(o_tile, offset=dv_tile*32);  // 부분 write
}
```

### Trade-off 분석

| 항목 | 기존 | DV Tiling (4 tiles) | 차이 |
|------|------|---------------------|------|
| **레지스터** | 512B | 128B | **75% 감소 ✓** |
| **Spill** | 발생 | 없음 | **제거 ✓** |
| **KV 로드** | 1회 | 4회 | **4배 증가 ✗** |
| **연산량** | N | N | 동일 |

**수치 예시**:
```
기존 (Spill 발생):
  KV load: 100ms
  Compute: 50ms
  Spill overhead: 2000ms (!)
  총: 2150ms

DV Tiling (4 tiles):
  KV load: 400ms (4×)
  Compute: 50ms
  Spill overhead: 0ms
  총: 450ms

→ 2150ms → 450ms = 4.8배 빠름!
```

**왜 효과적인가?**
- KV 로드는 sequential access → 캐시 효율 좋음
- Spill은 random access → 캐시 미스 많음
- Spill은 매 변수마다 발생 → 수만 번
- KV 재로드는 타일 당 1번 → 수 번

---

## 🔍 파라미터 튜닝 이론

### BLOCK_M (Query 타일 크기)

**정의**: 한 work-group이 동시에 처리하는 query row 수

```
BLOCK_M=32:
  - Work-group 수: n_q / 32
  - 메모리 재사용: K,V 타일을 32개 query가 공유
  - LDS 사용량: K[BLOCK_N][DK_VEC] (BLOCK_M 무관)

BLOCK_M=64:
  - Work-group 수: n_q / 64 (절반)
  - 메모리 재사용: K,V 타일을 64개 query가 공유 (2배)
  - LDS 사용량: 동일
```

**Trade-off**:
- ↑ BLOCK_M → 메모리 재사용 ↑ (K,V 타일당 더 많은 query)
- ↑ BLOCK_M → Work-group 수 ↓ → Occupancy ↓ (잠재적)
- ↑ BLOCK_M → 레지스터/private 메모리 사용량 동일 (각 스레드는 1 query만)

**권장**:
- GPU에 여유 있으면: BLOCK_M=64 (메모리 재사용 최대화)
- Occupancy 낮으면: BLOCK_M=32 (Work-group 많이)

---

### BLOCK_N (KV 타일 크기)

**정의**: 한 번에 local memory에 로드하는 K/V row 수

```
BLOCK_N=32:
  - LDS 사용량: K[32][32] = 8KB (half4 기준)
  - Barrier 빈도: n_kv / 32 회
  - 메모리 대역폭: 작은 chunk 여러 번

BLOCK_N=64:
  - LDS 사용량: K[64][32] = 16KB
  - Barrier 빈도: n_kv / 64 회 (절반)
  - 메모리 대역폭: 큰 chunk 적게
```

**Trade-off**:
- ↑ BLOCK_N → Barrier overhead ↓
- ↑ BLOCK_N → LDS 사용량 ↑ → Occupancy ↓ (잠재적)
- ↑ BLOCK_N → Coalesced access 효율 ↑ (큰 transaction)

**LDS 한계**:
```
Adreno LDS: 보통 32-64KB
BLOCK_N=64: K[64][32] × sizeof(half4) = 64×32×8 = 16KB
여유: 충분 ✓

BLOCK_N=128: K[128][32] × 8 = 32KB
여유: 괜찮지만 다른 변수도 있으므로 위험
```

**권장**:
- 32 → 64 순서로 테스트
- 128은 LDS 부족 위험

---

### DV_TILE (Output 타일 크기)

**정의**: 한 번에 계산하는 output 차원 크기

```
DV_TILE=32 (4 tiles):
  - 레지스터: o_acc[8] = 128 bytes
  - KV 재로드: 4회
  - Spill: 없음 (충분히 작음)

DV_TILE=64 (2 tiles):
  - 레지스터: o_acc[16] = 256 bytes
  - KV 재로드: 2회
  - Spill: GPU 따라 다름

DV_TILE=128 (1 tile = 타일링 없음):
  - 레지스터: o_acc[32] = 512 bytes
  - KV 재로드: 1회
  - Spill: 거의 확실 (기존 문제)
```

**Trade-off**:
- ↓ DV_TILE → 레지스터 ↓ → Spill 방지 ✓
- ↓ DV_TILE → KV 재로드 ↑ → 메모리 대역폭 ↑
- ↓ DV_TILE → Loop overhead ↑ (미미함)

**Sweet Spot 찾기**:
```
목표: "Spill이 발생하지 않는 최대 크기"

레지스터 한계: R bytes/thread
현재 사용량: q_priv + o_tile + 기타
- q_priv[32] = 512 bytes (고정)
- o_tile[DV_TILE/4] = DV_TILE × 4 bytes
- 기타 (스칼라 등): ~100 bytes

여유: R - 612 bytes
o_tile 한계: (R - 612) / 4 floats

예시 (R=1024):
  여유: 1024 - 612 = 412 bytes
  o_tile: 412 / 4 = 103 floats = 25 float4
  DV_TILE: 25 × 4 = 100
  → DV_TILE=64 안전, DV_TILE=128 위험
```

**권장 순서**:
1. DV_TILE=32 (가장 안전, baseline)
2. DV_TILE=64 (2배 큰 타일, 보통 안전)
3. DV_TILE=16 (극도로 안전, overhead 많음)
4. DV_TILE=128 (spill 테스트용, 느릴 것)

---

## 📊 성능 모델링

### Roofline 분석

**Compute Intensity** = FLOPs / Bytes

```
Flash Attention (한 query):
  FLOPs: 2 × n_kv × DK (QK) + 2 × n_kv × DV (PV)
        ≈ 4 × n_kv × 128
        
  Bytes (기존):
    K load: n_kv × DK × 2 (half)
    V load: n_kv × DV × 2
    Q load: DK × 4 (float)
    Spill: o_acc 왕복 × n_kv = 512 × n_kv
    총: 4×128×n_kv + 512 + 512×n_kv ≈ 1536×n_kv
  
  Intensity (기존):
    (4×128×n_kv) / (1536×n_kv) = 0.33 FLOPs/Byte
    → 메모리 bound! (Spill 지배적)

  Bytes (DV Tiling, 4 tiles):
    K load: n_kv × DK × 2 × 4 (4회 재로드)
    V load: n_kv × DV × 2 × 4
    Q load: DK × 4
    Spill: 0
    총: 4×(4×128×n_kv) + 512 = 2048×n_kv + 512
  
  Intensity (Tiling):
    (4×128×n_kv) / (2048×n_kv) = 0.25 FLOPs/Byte
    → 여전히 메모리 bound, 하지만 Spill 제거!
```

**예상 속도**:
```
Spill overhead ≈ 512×n_kv cycles
KV 재로드 overhead ≈ 4×256×n_kv cycles (sequential, 캐시 효율)

Spill: 512×n_kv / 1 (random) = 512×n_kv
재로드: 1024×n_kv / 4 (cached) = 256×n_kv

→ Spill이 2배 더 느림!
```

---

## 🎓 왜 이 해결책이 도출되었는가?

### 1. 문제 진단 과정

```
관찰:
  - Prefill 커널이 전체 시간의 절반
  - Work-group 수는 512개 (충분함)
  - Occupancy는 낮지 않음

가설 1: Global memory 대역폭 부족?
  → 아니면 V도 local 캐시하면 개선?
  → 실험: LDS 계산 → 용량 부족
  → 기각

가설 2: Barrier overhead?
  → BLOCK_N 키우면 개선?
  → 수치 계산: n_kv/BLOCK_N = 64회 정도
  → 너무 적어서 주요 병목 아님
  → 기각

가설 3: 레지스터 Spill?
  → 스레드당 메모리 계산: 1KB+
  → GPU 레지스터 한계: 256-512B
  → Spill 확실!
  → 채택! ✓
```

### 2. 해결책 탐색

```
목표: 레지스터 사용량 줄이기

방법 A: Q를 shared에?
  → Q는 쿼리마다 다름 → 공유 불가
  → 기각

방법 B: O를 shared에?
  → O도 쿼리마다 다름 → 공유 불가
  → 기각

방법 C: 차원 병렬화 (decode처럼)?
  → 각 스레드가 DV의 일부만 담당
  → Reduction 필요 → 복잡
  → 잠재적 해결책, 하지만 복잡

방법 D: DV 타일링?
  → 시간 축으로 분할
  → KV 재로드 필요하지만 단순
  → 채택! ✓
```

### 3. Trade-off 검증

```
질문: "KV 재로드 overhead가 Spill보다 작은가?"

계산:
  Spill overhead: 매 o_acc 접근마다 200 cycles
    - 접근 빈도: 내부 루프마다 (수천 번)
    - 총: 수십만 cycles
  
  재로드 overhead: 타일당 KV 전체 로드
    - 빈도: 타일 개수 (4회)
    - 메모리 크기: 4 × n_kv × 256B
    - Sequential access → 캐시 효율 높음
    - 총: 수만 cycles
  
→ 재로드가 훨씬 낫다! ✓
```

### 4. 구현 설계

```
선택:
  - K만 local 캐시 (LDS 절약)
  - V는 global 직접 (타일별로 다른 영역 접근)
  - Q는 register 유지 (모든 타일에서 재사용)
  - O만 타일링 (레지스터 압박 해소)

이유:
  - K는 모든 타일에서 동일 영역 재사용 → local 효율적
  - V는 타일별로 다른 영역 → local 캐시 비효율적
  - Q는 크지만(512B), V보다는 작고 재사용 많음
  - O가 레지스터 압박 주범 → 타일링 대상
```

---

## 🚀 최종 정리

### 최적화의 핵심

**"GPU 병렬성이 부족한 게 아니라, 레지스터 부족이 문제였다"**

### 해결 전략

1. **진짜 병목 찾기**: Occupancy → Spill
2. **단순한 해결책**: DV 타일링
3. **Trade-off 계산**: 재로드 < Spill
4. **체계적 튜닝**: 여러 타일 크기 실험

### 예상 결과

```
최소 (안전한 타일링): 1.5-2배 향상
보통 (최적 타일 찾기): 2-3배 향상
최대 (Spill 완전 제거): 4-5배 향상
```

---

**이것이 "왜 이런 해결 방식이 도출되었는지"에 대한 완전한 답변입니다!**

