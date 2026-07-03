import json
from collections import defaultdict

from loovi_vision.enrich.gaze import facing_from_angles

# 시청시간 분포 버킷. Attention 대상은 LTS(누적 응시 >= lts_min=1초) 인원뿐이라
# "1초 미만"은 정의상 항상 0 → 제외하고 [1~2초 / 2초 이상] 2구간으로 확정했다.
# realtime/summary.py와 공유하는 단일 정의(라벨 드리프트 방지).
DWELL_BUCKETS = ("1_to_2s", "over_2s")


def load_pose_records(poses_path):
    # COLD raw(JSONL) 로드. 깨진 줄은 건너뛴다.
    records = []
    try:
        with open(poses_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return records


def grade_of(duration, settings):
    # 세션 하나의 지속시간을 등급화. 임계값은 settings 주입 (raw 재저장 없이 재해석 가능).
    # glance_sec 미만은 노이즈(1프레임 스침)로 보고 세션에서 제외한다(최소 유효 시청).
    if duration < settings.gaze_grade_glance_sec:
        return None
    if duration >= settings.gaze_grade_dwell_sec:
        return "dwell"
    if duration >= settings.gaze_grade_view_sec:
        return "view"
    return "glance"


def dwell_bucket(total_sec, dwell_sec):
    # 개인의 누적 시청 시간을 리포트 구간으로 분류. Attention 대상은 LTS 인원(누적 >= lts_min=1초)뿐이라
    # "1초 미만" 버킷은 정의상 항상 0 → 제거하고 [1~2초 / 2초 이상] 2구간으로 확정한다.
    if total_sec >= dwell_sec:
        return "over_2s"
    return "1_to_2s"


def _sessions_for_track(records, settings, center_fn):
    # 한 track 의 pose 샘플들을 시간순으로 훑어 "연속 응시 구간(세션)"을 뽑는다.
    #   - facing(광고 향함) 샘플만 본다. 아닌 샘플은 건너뛴다.
    #   - 응시가 이어지다가 직전 응시 샘플과의 간격이 gap_tol 을 넘으면(잠깐 딴 데 봄/검출 끊김)
    #     거기서 한 세션을 끊고 새 세션을 시작한다. gap_tol 이내면 같은 세션으로 이어붙인다.
    #   세션 시청시간 = 마지막 응시 샘플 시각 − 첫 응시 샘플 시각.
    sessions = []
    start = last = None          # start=현재 세션 시작 시각, last=직전 응시 샘플 시각
    for rec in records:
        # center_fn 이 있으면 카메라-매체 상대위치 기반으로 정면 기준각을 보정(3차, 기본 미사용).
        yaw_c, pitch_c = center_fn(rec) if center_fn else (settings.gaze_yaw_center, settings.gaze_pitch_center)
        facing = facing_from_angles(rec["yaw"], rec["pitch"], yaw_c, pitch_c,
                                    settings.gaze_yaw_tol, settings.gaze_pitch_tol)
        if rec.get("low_conf") and settings.gaze_low_conf_policy == "exclude":
            facing = False       # 신뢰 낮은(얼굴 작은) 샘플은 응시로 치지 않음
        if not facing:
            continue
        ts = float(rec["timestamp_sec"])
        if start is None:
            start = ts                                        # 첫 응시 → 세션 시작
        elif ts - last > settings.gaze_gap_tol_sec:
            _append_session(sessions, start, last, settings)  # 간격이 커 세션 종료 후
            start = ts                                        # 새 세션 시작
        last = ts
    if start is not None:
        _append_session(sessions, start, last, settings)      # 마지막까지 열려 있던 세션 마감
    return sessions


def _append_session(sessions, start, end, settings):
    # (시작~끝) 한 세션을 확정해 목록에 추가한다. 단 glance_sec 미만(스침)은 노이즈로 버린다.
    duration = round(end - start, 3)
    grade = grade_of(duration, settings)
    if grade is None:            # 최소 유효 시청(0.2초) 미만 → 세션으로 세지 않음
        return
    sessions.append({
        "start_ts": round(start, 3),
        "end_ts": round(end, 3),
        "duration_sec": duration,
        "grade": grade,
    })


def analyze_gaze_sessions(poses_path, settings, center_fn=None):
    # COLD 사후 분석: 저장된 raw pose(JSONL) 전체를 읽어 Attention 지표를 낸다.
    # 실시간 경로(realtime/summary.py)와 독립이며, 임계값(gaze.*)만 바꿔 재실행하면
    # raw 재수집 없이 결과가 바뀐다. 세션 요약 manifest·리뷰 대시보드가 이 결과를 쓴다.
    # center_fn(rec)->(yaw_center,pitch_center): 거리/위치 기반 보정 훅 (3차, 기본 None).
    records = load_pose_records(poses_path)
    by_track = defaultdict(list)
    for rec in records:
        by_track[rec["track_id"]].append(rec)

    # track 별로 응시 세션과 총 시청시간(total_gaze_sec = 세션 시청시간의 합)을 구한다.
    per_track, all_sessions = [], []
    for track_id, recs in by_track.items():
        recs.sort(key=lambda r: r["timestamp_sec"])   # 시간순 정렬 후 세션 추출
        sessions = _sessions_for_track(recs, settings, center_fn)
        if not sessions:
            continue
        per_track.append({
            "track_id": track_id,
            "gaze_sessions": sessions,
            "total_gaze_sec": round(sum(s["duration_sec"] for s in sessions), 3),
            "gaze_count": len(sessions),
        })
        all_sessions.extend(sessions)

    # Attention 확정 정의: 대상=LTS 인원(누적 응시 >= lts_min_sec), 평균=인원 기준.
    # 즉 한 사람의 여러 세션을 total_gaze_sec 로 합쳐 1인 1대표값으로 평균/분포를 낸다.
    lts_min = settings.gaze_lts_min_sec
    lts_totals = [p["total_gaze_sec"] for p in per_track if p["total_gaze_sec"] >= lts_min]
    dwell_distribution = {b: 0 for b in DWELL_BUCKETS}
    for total in lts_totals:
        dwell_distribution[dwell_bucket(total, settings.gaze_grade_dwell_sec)] += 1
    return {
        "gaze_total_sessions": len(all_sessions),          # 전체 응시 구간 수 (참고)
        "gazers_count": len(per_track),                    # 한 번이라도 응시한 인원 (참고)
        "attention_count": sum(1 for t in lts_totals if t >= settings.gaze_grade_dwell_sec),  # 2초+ 응시자
        "avg_dwell_sec": round(sum(lts_totals) / len(lts_totals), 3) if lts_totals else 0.0,  # 인원 기준
        "dwell_distribution": dwell_distribution,          # LTS 인원의 시청시간 분포
        "per_track_gaze": sorted(per_track, key=lambda p: p["track_id"]),
    }
