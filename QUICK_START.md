# ⚡ 빠른 시작 가이드 - 5분 안에 테스트하기

## 📝 요약: 무엇을 하는가?

**문제**: Flash Attention 커널이 레지스터 spill로 인해 매우 느림  
**해결**: DV 타일링으로 스레드당 메모리 사용량 75% 감소  
**목표**: 여러 타일 크기 조합을 테스트해서 최적값 찾기

---

## 🎯 3단계 프로세스

### Step 1: 설정 변경
```bash
./test_configs.sh combo_A
```

### Step 2: 재컴파일
```bash
make -j$(nproc)
# 또는 빌드 시스템에 맞게
```

### Step 3: 실행 & 시간 측정
```bash
./your_inference_command
# cl_profiling.csv 확인
grep "flash_attn_f32_f16" cl_profiling.csv
```

---

## 📊 테스트 순서 (권장)

```bash
# 1. 기본 (현재 설정)
./test_configs.sh combo_A
make && ./run_test
# 시간 기록: _____ms

# 2. DV 타일 키우기
./test_configs.sh combo_B
make && ./run_test
# 시간 기록: _____ms

# 3. KV 타일 키우기
./test_configs.sh combo_C
make && ./run_test
# 시간 기록: _____ms

# 4. 둘 다 키우기
./test_configs.sh combo_D
make && ./run_test
# 시간 기록: _____ms

# 5. 가장 빠른 것 선택!
```

---

## 🔍 코드 수정 위치 (수동으로 할 경우)

**파일**: `ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl`

```c
// ============================================
// 📍 이 부분만 수정하면 됨! (42-52번 줄)
// ============================================

#ifndef BLOCK_M
#define BLOCK_M 32        // ← 이 숫자 변경 (32, 64 테스트)
#endif

#ifndef BLOCK_N
#define BLOCK_N 32        // ← 이 숫자 변경 (16, 32, 64 테스트)
#endif

#ifndef DV_TILE
#define DV_TILE 32        // ← 이 숫자 변경 (16, 32, 64, 128 테스트)
#endif

// ============================================
// 다른 부분은 건드리지 마세요!
// ============================================
```

---

## 📈 기대 결과

### 성공 사례
```
조합 A: 2000ms  ← 기준
조합 B: 1600ms  ← 20% 빠름! ✓
조합 C: 1800ms  ← 10% 빠름
조합 D: 1400ms  ← 30% 빠름! ✓✓
```

### 실패 사례 (정상임)
```
조합 F: 2500ms  ← 더 느림 (LDS 부족으로 occupancy 떨어짐)
조합 G: 4000ms  ← 훨씬 느림 (spill 재발)
```

**→ 가장 빠른 조합을 찾는 것이 목표!**

---

## 🛠️ 각 파라미터의 의미 (간단 버전)

### BLOCK_M (32 or 64)
- **작으면**: Work-group 많음, 메모리 덜 씀
- **크면**: Work-group 적음, 메모리 재사용 좋음

**추천**: 32 → 64 순서로 테스트

---

### BLOCK_N (16, 32, 64)
- **작으면**: LDS 적게 씀, barrier 자주
- **크면**: LDS 많이 씀, barrier 가끔

**추천**: 32 → 64 → 16 순서로 테스트

---

### DV_TILE (16, 32, 64, 128)
- **작으면**: 레지스터 거의 안 씀, KV 여러번 읽음
- **크면**: 레지스터 많이 씀, KV 적게 읽음

**추천**: 32 → 64 → 16 → 128 순서로 테스트

**주의**: 128 = 타일링 없음 (baseline 측정용)

---

## ⚠️ 트러블슈팅

### 컴파일이 안 됨
```
error: local memory size exceeded
```
**해결**: BLOCK_N을 줄이세요 (64 → 32 → 16)

---

### 더 느려짐
```
기존: 2000ms
새로운: 2500ms
```
**해결**: 
1. DV_TILE을 줄이세요 (64 → 32)
2. BLOCK_M을 줄이세요 (64 → 32)

---

### 결과가 이상함 (숫자가 틀림)
```
출력이 NaN이거나 이상한 값
```
**해결**: 
1. DV_TILE이 DV(128)의 약수인지 확인 (16, 32, 64, 128만 가능)
2. 코드 수정 실수가 없는지 확인

---

## 📞 디버깅 체크리스트

- [ ] 스크립트 실행했나? `./test_configs.sh combo_X`
- [ ] 재컴파일했나? `make`
- [ ] 현재 설정 확인했나? `./test_configs.sh show`
- [ ] 시간 측정했나? `grep "flash_attn" cl_profiling.csv`
- [ ] BENCHMARK_RESULTS.md에 기록했나?

---

## 🎓 왜 이게 빨라지는가? (1분 설명)

### 기존 문제
```
한 스레드가 output 전체(128 floats) 담당
→ 레지스터 부족
→ GPU가 자동으로 메모리에 저장(spill)
→ 매번 메모리 왕복
→ 100배 느림!
```

### 해결책
```
output을 4조각(32씩)으로 나눔
→ 한 번에 1조각만 처리
→ 레지스터에 다 들어감!
→ spill 없음
→ 빠름!
```

**Trade-off**: KV를 4번 읽어야 하지만, spill보다 훨씬 빠름

---

## 🚀 지금 바로 시작하세요!

```bash
cd /data/thahn1230/llama.cpp
./test_configs.sh combo_A
make
# 여러분의 테스트 실행
```

**Happy tuning! 🎉**

