import json
import mimetypes
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2

from .media import parse_byte_range, video_seek_plan
from .paths import safe_run_id


PACKAGE_DIR = Path(__file__).resolve().parent
INDEX_PATH = PACKAGE_DIR / "static" / "index.html"


class ReviewHandler(BaseHTTPRequestHandler):
    # 표준 라이브러리 HTTP server로 정적 HTML, JSON API, 영상/프레임을 제공한다.
    server_version = "LooviReview/1.0"

    @property
    def state(self):
        return self.server.state

    def do_GET(self):
        # 라우팅은 단순 path prefix 기반으로 유지해 외부 web framework 의존성을 피한다.
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in {"/", "/index.html"}:
            self.send_static_file(INDEX_PATH)
            return
        if path == "/api/sessions":
            self.send_json({"sessions": self.state.list_sessions()})
            return
        if path.startswith("/api/session/"):
            self.send_session(path.removeprefix("/api/session/"))
            return
        if path.startswith("/media/") and path.endswith(".mp4"):
            self.send_video(path.removeprefix("/media/").removesuffix(".mp4"))
            return
        if path.startswith("/frame/"):
            frame_id = path.removeprefix("/frame/").split("?", 1)[0].removesuffix(".jpg")
            self.send_frame(frame_id, parsed.query)
            return

        self.send_error(404, "Not found")

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def send_json(self, payload, status=200):
        # API 응답은 브라우저 캐시 없이 매번 최신 파일 상태를 보게 한다.
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_static_file(self, path):
        # 현재는 단일 HTML 파일을 직접 서빙한다.
        if not path.exists():
            self.send_error(404, "Static file not found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _resolve_video(self, run_id):
        # run_id 검증 → manifest → 실제 영상 경로까지 해석한다. 실패 시 에러를 보내고 None 반환.
        if not safe_run_id(run_id):
            self.send_error(400, "Invalid run_id")
            return None, None
        manifest = self.state.manifest_for(run_id)
        if not manifest:
            self.send_error(404, "Session not found")
            return None, None
        video_path = self.state.resolve_stored_path(manifest.get("video_path"))
        if not video_path or not video_path.exists():
            self.send_error(404, "Video not found")
            return None, None
        return manifest, video_path

    def send_session(self, run_id):
        # 한 세션의 manifest, JSONL rows, timing metadata, 영상 URL을 묶어 반환한다.
        if not safe_run_id(run_id):
            self.send_error(400, "Invalid run_id")
            return
        manifest = self.state.manifest_for(run_id)
        if not manifest:
            self.send_error(404, "Session not found")
            return

        video_path = self.state.resolve_stored_path(manifest.get("video_path"))
        rows = self.state.load_rows(manifest)
        video_meta = self.state.video_metadata(manifest)
        payload = {
            "manifest": manifest,
            "rows": rows,
            "timing": {
                "timeline_duration_sec": self.state.timeline_duration(rows),
                "video_duration_sec": video_meta["duration_sec"],
                "video_fps": video_meta["fps"],
                "video_frame_count": video_meta["frame_count"],
            },
            "video_url": f"/media/{run_id}.mp4" if video_path and video_path.exists() else None,
        }
        self.send_json(payload)

    def send_video(self, run_id):
        # 브라우저 video 태그가 seek할 수 있도록 Range 요청을 지원한다.
        _, video_path = self._resolve_video(run_id)
        if video_path is None:
            return

        file_size = video_path.stat().st_size
        parsed = parse_byte_range(self.headers.get("Range"), file_size)
        if parsed is None:
            self.send_error(416, "Invalid range")
            return
        start, end, status = parsed

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        with video_path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def send_frame(self, run_id, query):
        # 브라우저에서 MP4 코덱 재생이 실패할 때 쓰는 JPEG 프레임 fallback.
        manifest, video_path = self._resolve_video(run_id)
        if video_path is None:
            return

        params = parse_qs(query)
        try:
            sec = max(0.0, float(params.get("t", ["0"])[0]))
        except ValueError:
            sec = 0.0

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.send_error(500, "Cannot open video")
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_sec = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        rows = self.state.load_rows(manifest)
        timeline_sec = self.state.timeline_duration(rows)
        seek_mode, seek_value = video_seek_plan(sec, timeline_sec, fps, frame_count, video_sec)
        if seek_mode == "frame":
            cap.set(cv2.CAP_PROP_POS_FRAMES, seek_value)
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_value)

        ok, frame = cap.read()
        if (not ok or frame is None) and frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
            ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            self.send_error(404, "Frame not found")
            return

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            self.send_error(500, "Cannot encode frame")
            return

        body = encoded.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
