# 영상 스트리밍/프레임 추출의 순수 계산 로직. HTTP 전송은 handler가 담당한다.


def parse_byte_range(range_header, file_size):
    # HTTP Range 헤더를 (start, end, status)로 해석한다. 유효하지 않으면 None(→ 416).
    start, end, status = 0, file_size - 1, 200
    if range_header and range_header.startswith("bytes="):
        status = 206
        spec = range_header.removeprefix("bytes=").split(",", 1)[0]
        left, _, right = spec.partition("-")
        if left:
            start = int(left)
        if right:
            end = int(right)
        end = min(end, file_size - 1)
    if start < 0 or end < start or start >= file_size:
        return None
    return start, end, status


def video_seek_plan(sec, timeline_sec, fps, frame_count, video_sec):
    # 리뷰 시간축(sec)을 MP4 내부 위치로 변환해 seek 방법을 정한다.
    # 반환: ("frame", frame_index) 또는 ("msec", milliseconds).
    target_sec = sec
    if timeline_sec > 0 and video_sec > 0:
        # 요청 t는 review 시간축 기준이므로 MP4 내부 시간으로 비율 변환한다.
        target_sec = max(0.0, min(sec, timeline_sec)) * (video_sec / timeline_sec)
    if frame_count > 0 and video_sec > 0:
        target_sec = max(0.0, min(target_sec, max(0.0, video_sec - (1.0 / max(fps, 1.0)))))
        frame_index = min(frame_count - 1, max(0, int(round(target_sec * fps))))
        return "frame", frame_index
    return "msec", max(0.0, target_sec) * 1000.0
