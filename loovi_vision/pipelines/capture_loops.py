import time

import cv2

from loovi_vision.pipelines.frame_render import render_from_snapshot
from loovi_vision.pipelines.session_io import open_video_writer, pace_realtime


def _open_writer_if_needed(settings, writer, rec_path, frame):
    # VideoWriter는 첫 프레임 크기를 알아야 열 수 있어 첫 프레임 도착 시 지연 생성한다.
    if rec_path and writer is None:
        return open_video_writer(settings, rec_path, frame.shape)
    return writer


def _display_and_record(settings, frame, snapshot, writer):
    # 스냅샷으로 오버레이 프레임을 만들어 창 표시 + 영상 저장을 처리한다.
    display_frame = frame
    if settings.show_window or (writer and settings.record_overlay):
        display_frame = render_from_snapshot(frame, snapshot, settings)
        if settings.show_window:
            cv2.imshow("Loovi Person Only (q: quit)", display_frame)
    if writer:
        writer.write(display_frame if settings.record_overlay else frame)


def _tick_display_fps(settings, state):
    # 메인 루프(=표시/submit) 실효 FPS를 2초마다 출력. worker FPS와 비교해 병목 위치를 가른다.
    if not settings.perf_log:
        return
    state["n"] += 1
    elapsed = time.perf_counter() - state["t0"]
    if elapsed >= 2.0:
        print(f"  [perf/display] fps:{state['n'] / elapsed:5.1f}", flush=True)
        state["n"], state["t0"] = 0, time.perf_counter()


def live_loop(settings, cap, worker, rec_path, run_started_at):
    # 라이브(웹캠): 검출은 워커 스레드가 최신 프레임으로 돌리고, 메인은 캡처·표시에 집중한다.
    # → 화면은 카메라 FPS로 부드럽게 흐르고, bbox는 마지막 검출 결과라 빠른 움직임엔 살짝 지연된다.
    worker.start()
    writer = None
    frame_id = 0
    fps_state = {"n": 0, "t0": time.perf_counter()}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        now_sec = time.time() - run_started_at
        writer = _open_writer_if_needed(settings, writer, rec_path, frame)
        worker.submit(frame, now_sec, frame_id)          # 검출은 워커에 맡기고
        _display_and_record(settings, frame, worker.snapshot(), writer)  # 최신 스냅샷으로 즉시 표시
        _tick_display_fps(settings, fps_state)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    worker.stop()
    return writer


def sync_loop(settings, cap, worker, rec_path, run_started_at, source_fps):
    # 영상 파일/스레드 비활성: 기존처럼 process_every_n 마다 동기로 검출한다(결정론적 재생 유지).
    writer = None
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        pace_realtime(frame_id, source_fps, run_started_at)  # 사전 촬영 영상은 원래 속도로 재생
        now_sec = time.time() - run_started_at
        writer = _open_writer_if_needed(settings, writer, rec_path, frame)
        if frame_id % settings.process_every_n == 0:
            worker.process(frame, now_sec, frame_id)
        _display_and_record(settings, frame, worker.snapshot(), writer)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    return writer
