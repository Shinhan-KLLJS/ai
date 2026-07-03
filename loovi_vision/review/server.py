import argparse
from http.server import ThreadingHTTPServer

from .handler import ReviewHandler
from .state import ReviewState


def main():
    # CLI에서 경로를 바꿀 수 있지만 기본값은 현재 data 하위 구조다.
    parser = argparse.ArgumentParser(description="Loovi local review server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-dir", default="data/jsonl")
    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--session-dir", default="data/sessions")
    args = parser.parse_args()

    state = ReviewState(args.root, args.data_dir, args.video_dir, args.session_dir)
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    server.state = state
    print(f"Loovi review server: http://{args.host}:{args.port}")
    print(f"data: {state.data_dir}")
    print(f"videos: {state.video_dir}")
    print(f"sessions: {state.session_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
