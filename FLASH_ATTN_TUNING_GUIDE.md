# Flash Attention Kernel 최적화 가이드

## 🎯 문제 진단 요약

### 원래 병목 원인
1. **레지스터 Spill (가장 심각)**
   - `float4 q_priv[DK_VEC]` = `float4 q_priv[32]` = 512 bytes
   - `float4 o_acc[DV_VEC]` = `float4 o_acc[32]` = 512 bytes
   - **총 1KB+ per thread** → GPU 레지스터 한계 초과 → 메모리 spill 발생
   - Spill은 global memory 왕복이므로 **10-100배 느림**

2. **Occupancy 문제 (부차적)**
   - Prefill: 512 work-groups (충분함)
   - Decode: 32 work-groups (낮지만 구조적 한계)

3. **메모리 접근 패턴**
   - V를 매번 global에서 읽음 (local 캐시 없음)
   - K는 local 캐시하지만, tile 단위 재사용만 함

---

## 🔧 해결 방식 (DV 타일링)

### 핵심 아이디어
**"스레드당 state를 최소화 → 레지스터 spill 제거"**

```
기존:
  - 한 스레드가 output 전체(DV=128) 담당
  - o_acc[32] 필요 → 512 bytes
  
최적화:
  - DV를 타일(예: 32씩)로 분할
  - 각 타일마다 전체 KV를 순회
  - o_acc[8]만 필요 → 128 bytes (75% 감소!)
```

### Trade-off
- **장점**: 레지스터 사용량 대폭 감소 → spill 제거 → 속도 향상
- **단점**: KV를 여러 번 읽음 (타일 개수만큼)
  - 하지만 spill이 더 느리므로 **net positive**

### 왜 이게 효과적인가?
1. **메모리 계층 구조**:
   ```
   레지스터 접근: 1 cycle
   Global memory: 200-400 cycles
   Spill to global: 매 변수마다 왕복
   ```

2. **실제 병목**:
   - Spill 발생 시: o_acc 접근마다 global 왕복 (수천 번)
   - Tiling 후: KV를 4번 더 읽음 (하지만 sequential access, 캐시 효율 좋음)

3. **수치 예시** (대략적):
   ```
   기존:
     - KV read: 1000ms
     - Compute: 500ms
     - Spill overhead: 3000ms
     총: 4500ms
   
   최적화 (DV_TILE=32, 4 tiles):
     - KV read: 4000ms (4배)
     - Compute: 500ms
     - Spill overhead: 0ms (제거!)
     총: 4500ms → 2000ms (2.25배 빠름)
   ```

---

## 📋 튜닝 조합표

### Prefill 커널 파라미터

| 파라미터 | 의미 | 기본값 | 영향 |
|---------|------|--------|------|
| `BLOCK_M` | Query 타일 크기 (work-group당 처리할 query row 수) | 32 | ↑ → 메모리 재사용 증가, LDS 사용량 증가 |
| `BLOCK_N` | KV 타일 크기 (한 번에 로드할 K/V 행 수) | 32 | ↑ → LDS 사용량 증가, barrier 빈도 감소 |
| `DV_TILE` | Output 차원 타일 크기 | 32 | ↑ → 레지스터 사용량 증가, KV 재로드 감소 |

### Decode 커널 파라미터

| 파라미터 | 의미 | 기본값 | 영향 |
|---------|------|--------|------|
| `DEC_WG_SIZE` | Work-group 크기 | 64 | = sub-group 크기에 맞춰야 함 |

---

## 🧪 테스트 조합 (우선순위 순)

### **Tier 1: 안전한 조합 (먼저 테스트)**

#### 조합 A (현재 기본값)
```bash
# 파일: flash_attn_f32_f16.cl
# 수정 위치: 42-52번 줄

#define BLOCK_M 32
#define BLOCK_N 32
#define DV_TILE 32
```

**예상 결과**:
- 레지스터 사용량: 중간 (o_acc[8])
- LDS 사용량: 낮음 (K 타일만, 4KB)
- KV 재로드: 4번 (NUM_DV_TILES = 128/32 = 4)
- **기대**: 기존 대비 1.5-2배 향상

---

#### 조합 B (더 큰 DV 타일)
```bash
#define BLOCK_M 32
#define BLOCK_N 32
#define DV_TILE 64  # ← 변경
```

**예상 결과**:
- 레지스터 사용량: 중간-높음 (o_acc[16])
- KV 재로드: 2번 (NUM_DV_TILES = 128/64 = 2)
- **Trade-off**: 레지스터 더 쓰되, KV 재로드 절반
- **기대**: 레지스터 여유 있으면 조합 A보다 20-30% 빠름

---

#### 조합 C (더 큰 KV 타일)
```bash
#define BLOCK_M 32
#define BLOCK_N 64  # ← 변경
#define DV_TILE 32
```

**예상 결과**:
- LDS 사용량: 중간 (K 타일, 8KB)
- Barrier 빈도: 절반
- **Trade-off**: LDS 더 쓰되, barrier overhead 감소
- **기대**: LDS 여유 있으면 조합 A보다 10-20% 빠름

---

### **Tier 2: 공격적 조합 (Tier 1 결과 보고 테스트)**

#### 조합 D (큰 타일들)
```bash
#define BLOCK_M 32
#define BLOCK_N 64
#define DV_TILE 64
```

**예상 결과**:
- 레지스터: 높음, LDS: 높음
- KV 재로드: 2번, Barrier: 적음
- **위험**: Occupancy 감소 가능
- **기대**: 하드웨어 여유 있으면 최고 성능 (2-2.5배)

---

#### 조합 E (BLOCK_M 증가)
```bash
#define BLOCK_M 64  # ← 변경
#define BLOCK_N 32
#define DV_TILE 32
```

**예상 결과**:
- Work-group당 처리량: 2배
- LDS 사용량: 동일 (K 타일 크기는 BLOCK_N에만 의존)
- **Trade-off**: Work-group 수 절반, 하지만 각 work-group이 더 효율적
- **주의**: global size 계산 방식 변경 필요할 수 있음
- **기대**: 조합 A와 비슷하거나 약간 빠름

---

#### 조합 F (모든 타일 최대)
```bash
#define BLOCK_M 64
#define BLOCK_N 64
#define DV_TILE 64
```

**예상 결과**:
- **최고 위험**: LDS, 레지스터 모두 최대 사용
- **기대**: 하드웨어가 감당하면 2.5-3배, 못 하면 더 느림

---

### **Tier 3: 극단적 조합 (성능 한계 측정용)**

#### 조합 G (DV 타일링 완전 제거)
```bash
#define BLOCK_M 32
#define BLOCK_N 32
#define DV_TILE 128  # ← DV 전체 (타일링 없음)
```

**예상 결과**:
- 레지스터 사용량: 최대 (o_acc[32], 원본 수준)
- KV 재로드: 1번 (타일링 없음)
- **위험**: Spill 재발 가능
- **목적**: "타일링 효과"를 측정하는 baseline

---

#### 조합 H (최소 타일)
```bash
#define BLOCK_M 32
#define BLOCK_N 16  # ← 최소
#define DV_TILE 16  # ← 최소
```

**예상 결과**:
- 레지스터: 최소, LDS: 최소
- KV 재로드: 8번, Barrier: 많음
- **기대**: Occupancy 최대, 하지만 overhead 많아서 느릴 수 있음

---

## 🔨 코드 수정 방법

### 방법 1: 코드 직접 수정 (간단)

**파일**: `/data/thahn1230/llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl`

**수정 위치**: 42-52번 줄

```c
// 현재 (42-52번 줄):
#ifndef BLOCK_M
#define BLOCK_M 32
#endif

#ifndef BLOCK_N
#define BLOCK_N 32
#endif

#ifndef DV_TILE
#define DV_TILE 32  // DV를 타일 단위로 처리 (DV=128이면 4개 타일)
#endif

// ↓ 아래처럼 직접 값 변경:
#ifndef BLOCK_M
#define BLOCK_M 64  // ← 이 숫자만 바꾸면 됨
#endif

#ifndef BLOCK_N
#define BLOCK_N 64  // ← 이 숫자만 바꾸면 됨
#endif

#ifndef DV_TILE
#define DV_TILE 64  // ← 이 숫자만 바꾸면 됨
#endif
```

**재컴파일 후 테스트**

---

### 방법 2: 컴파일 옵션으로 설정 (고급)

빌드 시스템에서 `-DBLOCK_M=64` 같은 옵션을 추가하면 코드 수정 없이 테스트 가능.

CMakeLists.txt나 빌드 스크립트에서:
```cmake
add_definitions(-DBLOCK_M=64 -DBLOCK_N=64 -DDV_TILE=64)
```

---

## 📊 성능 측정 방법

### 1. 프로파일링 데이터 확인
```bash
# cl_profiling.csv에서 다음 확인:
- flash_attn_f32_f16 실행 시간
- flash_attn_f32_f16_q1 실행 시간
- 전체 대비 비율
```

### 2. 각 조합별로 기록
```
조합 A: flash_attn_f32_f16 = 1234ms
조합 B: flash_attn_f32_f16 = 987ms  (20% 향상!)
조합 C: flash_attn_f32_f16 = 1100ms
...
```

### 3. Occupancy 측정 (가능하면)
GPU 프로파일러로:
- Active wave 수
- Register spill 발생 여부
- LDS 사용량

---

## 🎓 왜 이런 숫자들인가?

### BLOCK_M, BLOCK_N이 2의 거듭제곱인 이유
- GPU는 power-of-2에 최적화
- 메모리 정렬(alignment) 효율성
- Sub-group 크기(64)와 잘 맞아떨어짐

### DV_TILE이 32, 64, 128인 이유
- DV=128의 약수여야 함
- float4 단위 (4의 배수)
- 너무 작으면: overhead, 너무 크면: spill

### BLOCK_N 범위 (16-64)
- 너무 작으면: barrier가 너무 자주 (overhead)
- 너무 크면: LDS 부족
- Adreno LDS 크기 제한: 보통 32-64KB

---

## ⚡ 빠른 시작 가이드

### Step 1: 조합 A 테스트 (현재 기본값)
```bash
# 코드 수정 없음
# 빌드 & 실행
# 시간 측정: T_A = ???ms
```

### Step 2: 조합 B 테스트
```bash
# flash_attn_f32_f16.cl 51번 줄:
#define DV_TILE 64

# 빌드 & 실행
# 시간 측정: T_B = ???ms
# 비교: (T_A - T_B) / T_A * 100 = ??% 향상
```

### Step 3: 조합 C, D 순서대로 테스트

### Step 4: 가장 빠른 조합 선택!

---

## 🐛 문제 발생 시

### 컴파일 에러
```
error: local memory size exceeded
```
→ BLOCK_N 줄이기 (64 → 32)

### 실행 에러 / 느려짐
```
기존보다 더 느림
```
→ DV_TILE 줄이기 (64 → 32 → 16)
→ 레지스터 spill 재발 가능성

### 결과 불일치
```
출력값이 이상함
```
→ NUM_DV_TILES 계산 확인 (DV % DV_TILE == 0 이어야 함)

---

## 📈 예상 성능 향상

```
최악의 경우 (조합 잘못 선택): 0% (현상 유지)
보통의 경우 (조합 A-C):      50-100% 향상
최선의 경우 (조합 D, 하드웨어 여유): 150-200% 향상
이론적 최대 (spill 완전 제거): 300-400% 향상
```

실제 결과는 GPU 하드웨어 스펙(레지스터 개수, LDS 크기)에 크게 의존합니다!

