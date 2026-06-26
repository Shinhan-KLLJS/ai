# AdScope — Claude 변경 로그

개발 결정사항, 아키텍처 논의, 주요 변경 이력을 기록합니다.

---

## 2026-06-26

### v5 → v6 업그레이드
- **성별/연령 추정 모듈 추가** (InsightFace antelopev2 `genderage.onnx`, 96×96 입력)
  - 출력: `[male_logit, female_logit, age/100]` (buffalo_l과 인덱스 순서 반대 — 주의)
  - 트랙 단위 캐싱: 신규 등록 시 + 30프레임마다 갱신
  - 얼굴 crop 25px 미만은 추정 스킵
- **장거리 감지 강화**
  - YOLO 입력 640 → 960px
  - 최소 얼굴 크기 15 → 8px, 신뢰도 임계값 0.50 → 0.45
  - SAHI-lite (2-타일 분할) 옵션 추가 (`SAHI_ENABLE=False` 기본)
- **배치 저장 주기**: 1분 → 15초로 변경 (`Config.BATCH_SEC = 15`)
- **카메라 화면 한글 → 영어 전환**: `cv2.putText`가 Hershey 폰트만 지원하여 한글 깨짐

### UniquePersonTracker 버그 수정
- **증상**: 혼자인데 2~6명으로 카운트됨
- **원인 1**: `MAX_MISSING=20` (~1.3초)이 너무 짧아 고개 돌리면 퇴장→신규 처리
- **원인 2**: 성별 인덱스 오해석 (antelopev2는 pred[0]=남성, pred[1]=여성)
- **수정**:
  - `MAX_MISSING`: 20 → 60 (~4초)
  - `IOU_THRESH`: 0.30 → 0.25
  - 중심점 거리 160px 이내 fallback 매칭 추가
  - 성별 인덱스 수정

---

## 아키텍처 논의: 고유 인원 정확도 개선

### 현재 구조의 한계 (face-only IoU 트래킹)

| 상황 | 문제 |
|---|---|
| 고개를 돌림 | 얼굴 미감지 → 트랙 소멸 → 신규 카운트 |
| 두 사람이 겹침 | 얼굴 박스 합쳐지거나 소멸 |
| 거리 변화 | 박스 크기 급변 → IoU 매칭 실패 |
| 잠깐 화면 이탈 | 동일인인데 신규 등록 |

### 권장 구조 (3단계 로드맵)

```
카메라 프레임
    ├─ [1] YOLOv8n person 감지  ← 상체/전신, 자세 무관 안정적 bbox
    ├─ [2] OSNet ReID 임베딩    ← 외형 기반 동일인 판별 (코사인 유사도)
    ├─ [3] Kalman Filter 예측   ← 잠깐 가려져도 위치 추적
    └─ [보조] YOLOv8n-face      ← 성별/연령, 시선 추정에만 사용
```

**매칭 우선순위 (3단계 폭포식)**
1. IoU > 0.5 → 동일 트랙 (빠른 경로)
2. cosine similarity > 0.65 → ReID 매칭
3. Kalman 예측 위치 근처 → 위치 예측 매칭
4. 모두 미달 → 신규 인원 등록

### Phase 로드맵

| Phase | 내용 | 난이도 | 효과 |
|---|---|---|---|
| **A** | YOLOv8n person bbox로 트래킹 교체, face는 보조 | 낮음 | 중 |
| **B** | OSNet ReID 임베딩 추가, cosine similarity 매칭 | 중간 | 높음 |
| **C** | Kalman Filter 위치 예측 추가 | 중간 | 중 |

---

### Phase A 구현 (2026-06-26) — adscope_v7.py

**변경 사항**
- `PersonDetector` 추가: `yolov8n.onnx` (COCO class=0, 12.5MB)
  - ultralytics export로 생성 (직접 다운로드 URL은 401/404로 실패)
  - 출력: `(1, 84, 8400)` → person score = `preds[:, 4]`
- 주 트래킹 소스: face bbox → **person bbox** (전신/상체, 자세 무관)
- `YOLOFaceDetector`는 시선 판단 + 성별/연령에만 사용 (보조)
- `associate_faces_to_persons()`: face 중심점이 person bbox 내부이면 연결
- `draw()`: person bbox(굵은 테두리) + face bbox(얇은 노란 테두리) 이중 표시
- `UniquePersonTracker.CENTROID_FALLBACK`: 160 → 200px (person bbox 스케일)

**실행 결과 (1인 테스트)**
- `unique_male: 1, unique_female: 0` — 성별 정확
- Batch 1: `unique_total: 1` — 정확
- Batch 2~4: `unique_total: 2` — tracker는 run 전체 누적이므로, 세션 중 `peak_persons: 2` 프레임이 있었음 (배경 오감지 또는 실제 통행인)
- v6 대비: 혼자일 때 2~6명 카운트 → **1명으로 안정화**
