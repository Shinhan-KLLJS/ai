"""경계선(입구) 기반 유동인구 카운터.

계수 정의: "화면 밖에서 안으로 들어온 사람"(재입장도 별도로 셈)
        = track이 '화면 가장자리'에서 처음 생기는 사건.

반대로 화면 '중앙'에서 새로 생긴 track은 가림에 의한 파편(같은 사람이 잠깐 가려졌다 재획득된
경우)이므로 세지 않는다. 이렇게 하면 고유 ID 과다계수(파편)를 걷어내고 실제 입장만 센다.
- 파편은 중앙에서 태어나므로 제외 → 과다계수 억제
- 진짜 입장/재입장은 가장자리에서 태어나므로 포함 → 사용자 정의와 일치
"""
from collections import defaultdict

from loovi_vision.tracking.ultralytics_tracker import UltralyticsTracker


def count_entries(settings, frames_dets, frame_w, frame_h, margin_ratio=0.03, min_hits=3):
    """캐시된 검출을 트래커에 먹여, 가장자리에서 태어난 track만 입장으로 센다."""
    tracker = UltralyticsTracker(settings)
    first_box = {}                 # track_id -> 최초 관측 bbox(x1,y1,x2,y2)
    hits = defaultdict(int)        # track_id -> 관측 프레임 수

    for dets in frames_dets:
        det_to_track = tracker.update(dets, None)
        for det_idx, tid in det_to_track.items():
            hits[tid] += 1
            if tid not in first_box:
                x, y, w, h = dets[det_idx]["bbox"]
                first_box[tid] = (x, y, x + w, y + h)

    # 가장자리 판정용 여유 폭(프레임 크기 비율).
    mx, my = margin_ratio * frame_w, margin_ratio * frame_h
    entries = 0        # 가장자리 출생 = 입장
    interior = 0       # 중앙 출생 = 파편(제외)
    for tid, (x1, y1, x2, y2) in first_box.items():
        if hits[tid] < min_hits:   # 너무 짧게 관측된 유령 track 제외
            continue
        at_border = x1 <= mx or y1 <= my or x2 >= frame_w - mx or y2 >= frame_h - my
        if at_border:
            entries += 1
        else:
            interior += 1

    return {
        "entries": entries,
        "interior_births": interior,
        "confirmed_tracks": entries + interior,
        "margin_ratio": margin_ratio,
    }
