import argparse

from loovi_vision.pipelines.person_only import run

# 실행 모드는 config 파일로 결정한다.
#   기본값 attention.yaml : person + face + gaze + realtime 전체 파이프라인
#   person_only.yaml      : 사람 검출/추적 + 로컬 기록만 하는 순수 baseline
# 예) 순수 person-only 실행:
#   python main.py --config loovi_vision/configs/person_only.yaml
DEFAULT_CONFIG = "loovi_vision/configs/attention.yaml"


def main():
    parser = argparse.ArgumentParser(description="Loovi Vision 파이프라인 실행")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="사용할 YAML config 경로 (기본: 전체 파이프라인 attention.yaml)",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
