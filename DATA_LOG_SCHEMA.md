# data_log.jsonl — Field Reference

`data_log.jsonl`은 AdScope v6가 15초 단위로 기록하는 JSONL 파일입니다.  
한 줄 = 한 집계 윈도우의 JSON 객체.

---

## 식별 정보

| 필드 | 타입 | 예시 | 설명 |
|---|---|---|---|
| `board_id` | string | `"board_gangnam_01"` | 광고판 식별자. 복수 카메라 운영 시 구분용 |
| `window_start` | string | `"2026-06-26 15:31:41"` | 집계 윈도우 시작 시각 (로컬 타임) |
| `window_end` | string | `"2026-06-26 15:31:56"` | 집계 윈도우 종료 시각 |

---

## 핵심 KPI — 고유 인원 기반

> 동일인을 한 번만 카운트하는 메인 지표. 광고 효과 측정의 핵심.

| 필드 | 타입 | 예시 | 설명 |
|---|---|---|---|
| `unique_total` | int | `3` | 윈도우 내 감지된 고유 인원 수. 같은 사람이 여러 번 프레임에 잡혀도 1명으로 집계 |
| `unique_looked` | int | `2` | 그 중 광고판 방향을 바라본 고유 인원 수 |
| `unique_attention_rate` | float | `66.7` | 주목률 (%) = `unique_looked / unique_total × 100` |

---

## 성별 / 연령

> InsightFace genderage 모델(antelopev2)이 추정한 값. 트랙(사람) 단위로 한 번 추정 후 캐싱.

| 필드 | 타입 | 예시 | 설명 |
|---|---|---|---|
| `unique_male` | int | `2` | 윈도우 내 남성으로 추정된 고유 인원 수 |
| `unique_female` | int | `1` | 윈도우 내 여성으로 추정된 고유 인원 수 |
| `avg_age` | float \| null | `32.4` | 추정 연령 평균. 연령 추정이 한 건도 없으면 `null` |
| `age_distribution` | object | `{"10s":0,"20s":1,"30s":2,"40s":0,"50plus":0}` | 연령대별 고유 인원 분포 (10대 / 20대 / 30대 / 40대 / 50대 이상) |

> **참고:** `unique_male + unique_female ≤ unique_total`.  
> 얼굴 크기가 너무 작거나(< 25px) 추정에 실패한 인원은 성별/연령 미집계.

---

## 보조 지표 — 프레임 누적

> AI가 처리한 모든 프레임의 감지 결과를 누적한 값. 트래픽 밀도·체류 시간 분석에 활용.

| 필드 | 타입 | 예시 | 설명 |
|---|---|---|---|
| `frame_detections` | int | `259` | 윈도우 내 전체 프레임에서 얼굴이 감지된 누적 건수 (동일인 중복 포함) |
| `frame_looking` | int | `139` | 그 중 광고 방향을 바라보고 있던 누적 프레임 건수 |
| `frame_attention_rate` | float | `53.7` | 프레임 기준 주목률 (%) = `frame_looking / frame_detections × 100` |
| `avg_attention_score` | float | `34.8` | 주목(LOOK) 판정된 프레임들의 Attention Score 평균 (0~100). 정면에 가까울수록 높음 |
| `peak_persons` | int | `2` | 윈도우 내 단일 프레임에서 동시에 감지된 최대 인원 수 |
| `frame_count` | int | `300` | AI 처리가 수행된 총 프레임 수 (`Config.PROCESS_EVERY_N` 간격으로 샘플링된 값) |

---

## 지표 해석 가이드

```
unique_attention_rate  ← 광고 효과 핵심 지표 (몇 %가 봤는가)
frame_attention_rate   ← 주목 지속 시간 반영 (얼마나 오래 봤는가)
avg_attention_score    ← 주목 품질 (정면으로 얼마나 집중했는가)
peak_persons           ← 동시 체류 밀도 (광고판 앞 혼잡도)
```

### unique vs frame 차이

- `unique_attention_rate 100%` + `frame_attention_rate 50%`  
  → 지나간 모든 사람이 한 번씩은 봤지만, 각자 체류 시간의 절반만 광고를 주목

- `unique_attention_rate 30%` + `frame_attention_rate 90%`  
  → 소수만 봤지만 본 사람은 매우 오래 집중해서 봄
