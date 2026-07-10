from loovi_vision.pipelines.session_io import open_video_writer


class Recorders:
    """오버레이(인식결과) 영상과 원본 영상을 동시에(또는 각각) 저장하는 writer 묶음.

    각 VideoWriter는 첫 프레임 크기를 알아야 열 수 있으므로, 경로가 지정된 writer만
    첫 write 시점에 지연 생성한다. overlay_path/raw_path 중 None인 쪽은 만들지 않아
    한쪽만 저장할 때 기존 단일 저장과 동일하게 동작한다.
    """

    def __init__(self, settings, overlay_path, raw_path):
        self.settings = settings
        self.overlay_path = overlay_path        # 오버레이 영상 경로(없으면 저장 안 함)
        self.raw_path = raw_path                # 원본 영상 경로(없으면 저장 안 함)
        self.overlay_writer = None
        self.raw_writer = None

    @property
    def active(self):
        # 저장 대상이 하나라도 있으면 True.
        return bool(self.overlay_path or self.raw_path)

    @property
    def wants_overlay(self):
        # 오버레이 영상을 저장하는지(오버레이 렌더가 필요한지 판단용).
        return bool(self.overlay_path)

    def write(self, overlay_frame, raw_frame):
        # 지정된 경로가 있는 writer만 지연 생성 후 각자 프레임을 기록한다.
        # overlay_frame 은 렌더된 오버레이, raw_frame 은 캡처 원본 프레임.
        if self.overlay_path:
            if self.overlay_writer is None:
                self.overlay_writer = open_video_writer(
                    self.settings, self.overlay_path, overlay_frame.shape)
            self.overlay_writer.write(overlay_frame)
        if self.raw_path:
            if self.raw_writer is None:
                self.raw_writer = open_video_writer(
                    self.settings, self.raw_path, raw_frame.shape)
            self.raw_writer.write(raw_frame)

    def release(self):
        # 열려 있는 writer를 모두 정리한다(스레드 writer는 남은 프레임을 마저 인코딩).
        for writer in (self.overlay_writer, self.raw_writer):
            if writer:
                writer.release()
