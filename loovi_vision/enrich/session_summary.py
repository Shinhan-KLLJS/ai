# 주목자의 성별·연령 분포와 세션 전체 요약(per_track 포함)을 만든다.
# (realtime/summary.py의 5초 구간 스냅샷과 구분하기 위해 session_summary로 명명.)

AGE_BUCKETS = ["under_10", "10s", "20s", "30s", "40s", "50s", "60_plus"]

# 서버 전송 스키마용 연령 키 (내부 구간 -> 스냅샷/summary 표기).
SERVER_AGE_KEYS = {
    "under_10": "under10", "10s": "10s", "20s": "20s", "30s": "30s",
    "40s": "40s", "50s": "50s", "60_plus": "60plus",
}


def empty_age_dist():
    # 연령대별 0으로 초기화한 딕셔너리(서버 스키마 키). 성별별 age 블록 공통.
    return {key: 0 for key in SERVER_AGE_KEYS.values()}


def demographics_of(states):
    # 주어진 track 상태 묶음의 성별/연령 분포를 서버 스키마(v2) 키로 집계한다.
    # OTS(얼굴 잡힌 통행자)·LTS(응시자) 어느 쪽이든 같은 함수로 쓴다.
    # v2 구조: 성별별로 {count, age} 를 중첩한다 → 성별 미상이면 어느 쪽에도 넣지 않는다.
    count = {"male": 0, "female": 0}
    age = {"male": empty_age_dist(), "female": empty_age_dist()}
    for state in states:
        label = gender_label(state)
        if label is None:
            continue                       # 성별 미상 → 성별·연령 어느 쪽에도 집계하지 않음
        count[label] += 1
        bucket = age_bucket(state.age)
        if bucket is None:
            continue                       # 연령 미상 → 성별만 집계(count ≥ age 합)
        age[label][SERVER_AGE_KEYS[bucket]] += 1
    # v2: {"male": {"count": N, "age": {...}}, "female": {"count": N, "age": {...}}}
    return {g: {"count": count[g], "age": age[g]} for g in ("male", "female")}


def age_bucket(age):
    # insightface는 정수 나이를 반환한다. 오차가 있으므로 대략적 구간으로만 묶는다.
    if age is None:
        return None
    if age < 10:
        return "under_10"
    if age < 20:
        return "10s"
    if age < 30:
        return "20s"
    if age < 40:
        return "30s"
    if age < 50:
        return "40s"
    if age < 60:
        return "50s"
    return "60_plus"


def gender_label(state):
    # 얼굴 없는 사람에게는 성별을 추정하지 않는다 (1차 범위 밖) -> None.
    if not state.attended or state.gender is None:
        return None
    return "male" if state.gender == 1 else "female"


def build_summary(registry, min_hits=1):
    # 통행/주목 분리 집계: 모든 track은 분모, 얼굴 잡힌 track만 분자.
    # 노이즈(짧게 스친) track 제외를 위해 frames_seen >= min_hits 만 집계한다.
    confirmed = [s for s in registry.all() if s.frames_seen >= min_hits]
    total_unique = len(confirmed)
    attended = [s for s in confirmed if s.attended]
    attended_count = len(attended)

    gender_dist = {"male": 0, "female": 0}
    age_dist = {bucket: 0 for bucket in AGE_BUCKETS}
    for state in attended:
        label = gender_label(state)
        if label is not None:
            gender_dist[label] += 1
        bucket = age_bucket(state.age)
        if bucket is not None:
            age_dist[bucket] += 1

    per_track = []
    for state in sorted(confirmed, key=lambda s: s.track_id):
        seen = state.frames_seen
        per_track.append({
            "track_id": state.track_id,
            "frames_seen": seen,
            "frames_face_visible": state.frames_face_visible,
            "face_ratio": round(state.frames_face_visible / seen, 4) if seen else 0.0,
            "attended": state.attended,
            "gender": gender_label(state),                  # 미상이면 null
            "age": state.age if state.attended else None,   # 미상이면 null
        })

    return {
        "total_unique": total_unique,
        "attended_count": attended_count,
        "attention_rate": round(attended_count / total_unique, 4) if total_unique else 0.0,
        "gender_dist": gender_dist,
        "age_dist": age_dist,
        "per_track": per_track,
    }
