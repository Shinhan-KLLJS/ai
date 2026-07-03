import argparse
import json
import time
from datetime import datetime

import cv2
import numpy as np

from loovi_vision.config import Settings, load_config
from loovi_vision.detectors import PersonDetector
from loovi_vision.tracking import create_tracker
from loovi_vision.pipelines.batch import PersonOnlyBatch
from loovi_vision.pipelines.frame_render import collect_overlay_state, render_display_frame
from loovi_vision.pipelines.gaze_runtime import attach_performance, build_gaze_runtime
from loovi_vision.pipelines.session_io import (
    build_manifest,
    make_run_id,
    open_capture,
    open_video_writer,
    output_path,
    pace_realtime,
    session_path,
    video_path,
    write_session_manifest,
)


def append_row(out_path, payload):
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run(config_path="loovi_vision/configs/person_only.yaml"):
    # person-only 파이프라인 실행 단위: 설정, 모델, 카메라, 저장, 루프를 묶는다.
    settings = Settings(load_config(config_path))
    run_id = make_run_id(settings)
    run_started_at = time.time()
    run_started_text = datetime.fromtimestamp(run_started_at).strftime("%Y-%m-%d %H:%M:%S")
    out_path = output_path(settings, run_id)
    rec_path = video_path(settings, run_id) if settings.record_video else None
    meta_path = session_path(settings, run_id)
    manifest = build_manifest(settings, run_id, out_path, rec_path, run_started_text)
    write_session_manifest(meta_path, manifest)
    print(f"[Person Only] config: {config_path}")
    print(f"[Person Only] run_id: {run_id}")
    print(f"[Person Only] output: {out_path}")
    if rec_path:
        print(f"[Person Only] video: {rec_path}")
    print(f"[Person Only] session: {meta_path}")
    print(f"[Person Only] model: {settings.person_onnx}")

    # CUDA 사용 시 첫 추론이 느리므로 더미 프레임으로 provider/model warmup을 먼저 수행한다.
    detector = PersonDetector(settings)
    if settings.enable_cuda:
        print("  CUDA warmup started...", flush=True)
        detector.detect(np.zeros((settings.frame_h, settings.frame_w, 3), dtype=np.uint8))
        print("  CUDA warmup done", flush=True)

    enricher = None
    if settings.face_enable:
        # face.enable=false면 enricher를 만들지 않아 기존 동작과 100% 동일하다.
        from loovi_vision.enrich.face_enricher import FaceEnricher

        enricher = FaceEnricher(settings)

    # gaze.enable=false면 None을 반환해 1차와 100% 동일하게 동작한다.
    gaze_runtime = build_gaze_runtime(settings, enricher, run_id)

    tracker = create_tracker(settings)
    # 입력 소스: video_path 있으면 사전 촬영 영상(source_fps>0), 없으면 웹캠(source_fps=0).
    cap, source_fps = open_capture(settings)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open input: {settings.camera_video or settings.camera_id}")
        return
    writer = None

    batch = PersonOnlyBatch(settings, run_id, run_started_at, face_enabled=settings.face_enable)
    frame_id = 0
    detections = []
    det_to_track = {}
    proc_frames = 0
    max_dt = 0.0
    last_proc = run_started_at

    while True:
        # 카메라 프레임은 매번 읽지만, detector/tracker는 process_every_n마다 실행한다.
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        pace_realtime(frame_id, source_fps, run_started_at)  # 사전 촬영 영상은 원래 속도로 재생
        now_sec = time.time() - run_started_at   # run 시작 기준 elapsed (시계열 통일)

        if rec_path and writer is None:
            writer = open_video_writer(settings, rec_path, frame.shape)

        if frame_id % settings.process_every_n == 0:
            detections = detector.detect(frame)
            det_to_track = tracker.update(detections, frame)
            seen = faced = None
            if enricher:
                # track당 person crop을 얼굴 분석에 넣어 통행/주목을 분리 집계한다.
                seen, faced = enricher.process(frame, detections, det_to_track, frame_id, now_sec)
            if gaze_runtime:
                # HOT: 1초 경계마다 스냅샷 집계 → 버퍼/전송 (COLD raw는 on_face에서 기록).
                gaze_runtime.tick(seen, now_sec)
            batch.add(frame_id, detections, tracker, det_to_track, seen, faced)
            proc_frames += 1
            dt = (run_started_at + now_sec) - last_proc
            max_dt = max(max_dt, dt)
            last_proc = run_started_at + now_sec

        # show_window가 꺼져도 overlay 영상 저장이 켜져 있으면 display_frame을 생성한다.
        display_frame = frame
        if settings.show_window or (writer and settings.record_overlay):
            overlay = collect_overlay_state(enricher, gaze_runtime, now_sec)
            display_frame = render_display_frame(frame, detections, settings, det_to_track,
                                                 batch, tracker, enricher, overlay)
            if settings.show_window:
                cv2.imshow("Loovi Person Only (q: quit)", display_frame)

        if writer:
            writer.write(display_frame if settings.record_overlay else frame)

        if batch.should_flush():
            # batch_sec마다 aggregate row를 JSONL에 append한다.
            extra = gaze_runtime.row_extra(now_sec, tracker.total_unique) if gaze_runtime else None
            append_row(out_path, batch.flush(extra))
            print(f"  saved: {out_path}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if batch.frame_count:
        # 종료 직전 남은 partial batch도 버리지 않고 저장한다.
        extra = gaze_runtime.row_extra(time.time() - run_started_at, tracker.total_unique) if gaze_runtime else None
        append_row(out_path, batch.flush(extra))
    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

    if enricher:
        # 세션 종료 시 best_face로 성별/연령을 1회 판정하고 세션 요약을 만든다.
        from loovi_vision.enrich.session_summary import build_summary

        manifest.update(build_summary(enricher.finalize(), settings.track_min_hits))

    # 성능 로그 + (gaze 활성 시) COLD 응시 분석/전송 지표를 manifest에 병합한다.
    attach_performance(manifest, proc_frames, max(1e-6, time.time() - run_started_at), max_dt, gaze_runtime)

    manifest["status"] = "completed"
    manifest["ended_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_session_manifest(meta_path, manifest)
    print("\nDone. person-only log saved.")
    print("Review with: python -m loovi_vision.review.server --port 8765")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="loovi_vision/configs/person_only.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
