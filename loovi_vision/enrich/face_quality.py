# 얼굴 샘플 한 장의 "품질 가중치"를 계산한다.
# 성별/연령은 프레임마다 누적 투표로 수렴시키는데(track_state.add_genderage),
# 이때 작고 흐릿한 얼굴이 크고 선명한 정면 얼굴과 똑같이 1표를 행사하면 결과가 오염된다.
# 그래서 각 샘플에 품질 가중치를 줘, 신뢰할 만한 얼굴의 표를 더 무겁게 만든다.

import cv2


def sharpness_score(face_region):
    # 선명도 지표 = 라플라시안 분산. 초점이 나갔거나 움직임으로 흐린 얼굴일수록 낮다.
    # 얼굴 영역이 비어 있으면(잘림 등) 0으로 본다.
    if face_region is None or getattr(face_region, "size", 0) == 0:
        return 0.0
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if face_region.ndim == 3 else face_region
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_weight(conf, area, face_region, sharp_ref=0.0):
    # 품질 가중치 = 검출확신도 × 얼굴면적 × 선명도계수.
    #   - conf, area: 검출부가 이미 주는 신뢰 신호. 멀고 작은 얼굴은 area가 작아 자연히 가벼워진다.
    #   - 선명도계수: sharp_ref>0 일 때만 적용. min(1, 선명도/기준)으로 0~1 범위로 눌러
    #     선명한 얼굴이 과도하게 지배하지 않게 하고, 흐린 얼굴만 선택적으로 깎는다.
    #   - sharp_ref<=0(기본): 선명도 미적용 → 가중치 = conf × area (매직 상수 없는 기본 동작).
    weight = max(0.0, float(conf)) * max(0.0, float(area))
    if sharp_ref and sharp_ref > 0:
        factor = min(1.0, sharpness_score(face_region) / sharp_ref)
        weight *= factor
    return weight
