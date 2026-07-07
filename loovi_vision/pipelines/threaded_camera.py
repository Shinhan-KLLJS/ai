import threading


class ThreadedCamera:
    """웹캠 캡처를 백그라운드 스레드로 분리해 '항상 최신 프레임'만 소비하게 하는 래퍼.

    무선 웹캠(Iriun 등)은 OpenCV 내부 버퍼에 프레임이 쌓이면 cap.read()가
    오래된 프레임부터 반환해 지연이 누적되고 라이브 화면이 끊긴다. 이 클래스는
    별도 스레드에서 쉬지 않고 프레임을 읽어 최신 한 장만 유지하고, 소비자(메인 루프)는
    추론 속도와 무관하게 가장 최근 프레임을 받는다 → 캡처와 표시를 디커플링해 버벅임 제거.

    cv2.VideoCapture와 동일한 isOpened()/read()/release() 인터페이스를 제공해
    기존 루프에 그대로 끼워 쓸 수 있다.
    """

    def __init__(self, cap, read_timeout=2.0):
        self._cap = cap
        self._read_timeout = read_timeout          # 새 프레임 대기 최대 시간(초). 초과 시 종료로 간주
        self._lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._running = True
        self._new_frame = threading.Event()        # 새 프레임 도착 신호(소비자 대기용)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        # 백그라운드 루프: 카메라에서 계속 읽어 최신 프레임만 덮어쓴다.
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                # 원본 루프와 동일하게 read 실패는 스트림 종료로 본다.
                with self._lock:
                    self._ret = False
                self._running = False
                self._new_frame.set()
                break
            with self._lock:
                self._ret = True
                self._frame = frame
            self._new_frame.set()

    def isOpened(self):
        return self._cap.isOpened()

    def read(self):
        # 새 프레임이 도착할 때까지 대기한 뒤 최신 프레임을 반환한다.
        # 소비자가 느리면 그 사이 쌓인 중간 프레임은 버려지고 최신만 남아 지연이 쌓이지 않는다.
        got = self._new_frame.wait(self._read_timeout)
        with self._lock:
            self._new_frame.clear()
            if not got:
                # 타임아웃: 소스가 멈춘 것으로 보고 종료 신호를 준다.
                return False, None
            return self._ret, self._frame

    def release(self):
        # 스레드를 멈추고 원본 VideoCapture 자원을 해제한다.
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()
