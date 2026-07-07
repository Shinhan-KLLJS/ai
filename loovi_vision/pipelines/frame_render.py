# HUD 통계 집계와 overlay 프레임 생성을 담당한다. run() 루프에서 분리해 재사용/가독성을 높인다.
from loovi_vision.pipelines.overlay import draw


def make_stats(detections, batch, tracker, enricher=None, gazers=None, lts=None):
    stats = {
        "now_count": len(detections),
        "peak_persons": batch.peak_persons,
        "active_tracks": len(tracker.tracks),
        "unique_total": tracker.total_unique,
        "frame_count": batch.frame_count,
    }
    if enricher:
        # face.enable=true일 때만 실시간 주목/성별 카운트를 HUD에 더한다.
        stats.update(enricher.live_counts())
    if gazers is not None:
        stats["gazers"] = gazers
    if lts is not None:
        # gaze.enable=true일 때만 OTS(=unique_total)/LTS를 HUD에 표시한다.
        stats["lts"] = lts
    return stats


def collect_overlay_state(enricher, gaze_runtime, now_sec):
    # overlay 렌더에 필요한 얼굴/응시 상태를 한 번에 모은다.
    # enricher·gaze_runtime이 없으면 전부 None → 1차(person-only) 동작과 100% 동일.
    return {
        "face_boxes": enricher.last_face_boxes if enricher else None,
        "face_labels": enricher.face_labels if enricher else None,
        "attended_ids": enricher.attended_ids() if enricher else None,
        "gazing_ids": gaze_runtime.gazing_ids(now_sec) if gaze_runtime else None,
        "gazers": gaze_runtime.current_gazers(now_sec) if gaze_runtime else None,
        "lts": gaze_runtime.lts_count() if gaze_runtime else None,
        "poses": gaze_runtime.poses() if gaze_runtime else None,
        "gaze_secs": gaze_runtime.gaze_secs() if gaze_runtime else None,
    }


def render_display_frame(frame, detections, settings, det_to_track, batch, tracker, enricher, overlay):
    # HUD/overlay를 그린 프레임을 만든다(show_window 또는 record_overlay일 때만 호출).
    stats = make_stats(detections, batch, tracker, enricher, overlay["gazers"], overlay["lts"])
    return draw(frame.copy(), detections, stats, settings, det_to_track,
                overlay["face_boxes"], overlay["face_labels"], overlay["attended_ids"],
                overlay["gazing_ids"], overlay["poses"], overlay["gaze_secs"])


def render_from_snapshot(frame, snapshot, settings):
    # DetectionWorker가 발행한 스냅샷(detections/stats/overlay 미리 계산됨)으로 오버레이 프레임을 만든다.
    # 원본 frame은 raw 저장을 위해 보존해야 하므로 copy 위에 그린다.
    o = snapshot["overlay"]
    return draw(frame.copy(), snapshot["detections"], snapshot["stats"], settings,
                snapshot["det_to_track"], o["face_boxes"], o["face_labels"],
                o["attended_ids"], o["gazing_ids"], o["poses"], o["gaze_secs"])
