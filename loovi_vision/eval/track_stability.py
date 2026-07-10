"""라벨(GT) 없이 재는 트래킹 안정성 지표.

정답이 없어도 "트래커가 같은 사람의 ID를 얼마나 안 흔들리게 유지하나"는 잴 수 있다.
핵심 아이디어: 검출을 한 번만 돌려 프레임별로 캐시하고, 같은 검출을 여러 설정(예: match_thresh)
트래커에 먹여 비교한다 → 검출 조건이 동일하므로 차이는 순수하게 트래커 연관 품질이다.

해석(같은 검출 기준):
  - unique_ids ↓  : 같은 사람이 새 ID로 덜 튐(끊긴 track 재발급 감소) = 더 안정적
  - short_tracks ↓: 몇 프레임 반짝하고 사라지는 유령 track 감소
  - fragmentations↓: 한 track이 중간에 끊겼다 이어지는 횟수 감소
  - avg_concurrent: 프레임당 동시 track 수(밀도 참고값, 설정 간 거의 동일해야 정상)
"""
from collections import defaultdict

import cv2

from loovi_vision.detectors.person import PersonDetector
from loovi_vision.tracking.ultralytics_tracker import UltralyticsTracker

# 이 프레임 수 미만으로만 관측된 track은 "깜빡임(유령)"으로 본다.
SHORT_TRACK_HITS = 5


def cache_detections(settings, video_path, start=0, end=None, step=1, log=print):
    """영상 구간에 person 검출을 한 번 돌려 프레임별 detection 리스트를 캐시한다."""
    detector = PersonDetector(settings)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = total if end is None else min(end, total)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames_dets = []
    idx = start
    while idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % step == 0:
            frames_dets.append(detector.detect(frame))
        idx += 1
    cap.release()
    log(f"  검출 캐시: {len(frames_dets)}프레임 (구간 {start}~{end}, step {step})")
    return frames_dets


def track_stability(settings, frames_dets):
    """캐시된 검출을 현재 settings의 트래커에 먹여 안정성 지표를 계산한다."""
    tracker = UltralyticsTracker(settings)
    appear = defaultdict(list)          # track_id -> 등장한 프레임 순번 리스트
    per_frame_active = []
    for fi, dets in enumerate(frames_dets):
        det_to_track = tracker.update(dets, None)
        ids = set(det_to_track.values())
        per_frame_active.append(len(ids))
        for tid in ids:
            appear[tid].append(fi)

    frags = 0
    short = 0
    for frames in appear.values():
        if len(frames) < SHORT_TRACK_HITS:
            short += 1
        # 연속 등장 사이의 공백 = fragmentation
        frags += sum(1 for a, b in zip(frames, frames[1:]) if b - a > 1)

    n = len(per_frame_active) or 1
    return {
        "unique_ids": len(appear),
        "confirmed_unique": tracker.total_unique,
        "short_tracks": short,
        "fragmentations": frags,
        "avg_concurrent": round(sum(per_frame_active) / n, 2),
        "frames": len(frames_dets),
    }


def compare_thresholds(settings, frames_dets, thresholds):
    """같은 검출 캐시로 여러 match_thresh를 비교한다. [{thresh, stats}] 반환."""
    rows = []
    for thresh in thresholds:
        settings.track_match_thresh = thresh   # 런타임 오버라이드(검출은 그대로 재사용)
        rows.append({"match_thresh": thresh, "stats": track_stability(settings, frames_dets)})
    return rows
