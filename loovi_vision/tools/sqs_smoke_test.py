"""SQS 연동 스모크 테스트.

카메라/모델 없이 설정된 큐로 샘플 summary 1건을 동기 전송해
AWS 환경(자격증명·리전·큐 URL)이 올바른지 즉시(성공/실패 원인까지) 확인한다.

사용 예:
  python -m loovi_vision.tools.sqs_smoke_test
  python -m loovi_vision.tools.sqs_smoke_test --dry-run
  python -m loovi_vision.tools.sqs_smoke_test --config loovi_vision/configs/person_only.yaml

큐 URL/리전은 config(realtime.sqs_queue_url / sqs_region) 또는
환경변수(SQS_QUEUE_URL 또는 LOOVI_SQS_QUEUE_URL / AWS_REGION)로 주입한다.
"""
import argparse
import json
from pathlib import Path
import sys

from loovi_vision.config import Settings, load_config
from loovi_vision.enrich.session_summary import empty_age_dist
from loovi_vision.realtime.sqs_sender import resolve_sqs_target


def sample_summary(settings):
    # 실제 파이프라인 출력과 동일한 구조의 더미 summary(전송 경로 검증용).
    male_age = empty_age_dist()
    male_age["20s"] = 1
    demo = {"gender": {"male": 1, "female": 0}, "male_age": male_age, "female_age": empty_age_dist()}
    return {
        "device_id": settings.rt_device_id,
        "board_id": settings.board_id,
        "seq": 1,
        "timestamp": "1970-01-01T00:00:00Z",   # 고정값 → 수신 측에서 테스트 메시지로 식별 가능
        "interval_sec": 5.0,
        "ots_count": 1,
        "lts_count": 1,
        "ots_demographics": demo,
        "lts_demographics": demo,
        "attention": {
            "avg_dwell_sec": 2.0,
            "dwell_sum_sec": 2.0,
            "dwell_distribution": {"1_to_2s": 0, "2_to_3s": 1, "3_to_4s": 0, "over_4s": 0},
        },
    }


def main():
    parser = argparse.ArgumentParser(description="SQS 연동 스모크 테스트")
    parser.add_argument("--config", default="loovi_vision/configs/person_only.yaml")
    parser.add_argument("--dry-run", action="store_true", help="스키마/설정만 검증하고 SQS 전송은 하지 않음")
    args = parser.parse_args()

    settings = Settings(load_config(args.config))
    queue_url, region = resolve_sqs_target(settings)

    if not queue_url:
        print("[FAIL] 큐 URL 미설정: realtime.sqs_queue_url 또는 환경변수 SQS_QUEUE_URL 를 지정하세요.")
        return 1
    if any(ch.isspace() for ch in queue_url) or "|" in queue_url:
        print("[FAIL] 큐 URL 형식 오류: SQS_QUEUE_URL 에 공백이나 '|' 문자가 섞여 있습니다.")
        return 1
    if not queue_url.startswith("https://sqs."):
        print("[FAIL] 큐 URL 형식 오류: SQS_QUEUE_URL 은 https://sqs.<region>.amazonaws.com/... 형식이어야 합니다.")
        return 1

    try:
        import boto3
    except Exception as exc:
        print(f"[FAIL] boto3 임포트 실패({exc}). 'pip install boto3' 후 재시도하세요.")
        return 1

    message = sample_summary(settings)

    try:
        import jsonschema

        schema_path = Path("docs/vision-summary-schema.json")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(message, schema)
        print("  스키마    : OK")
    except Exception as exc:
        print(f"[FAIL] 스키마 검증 실패: {exc}")
        return 1

    print(f"  대상 큐   : {queue_url}")
    print(f"  리전      : {region or '(boto3 기본 해석)'}")
    if args.dry_run:
        print("[OK] dry-run 통과: SQS 전송은 생략했습니다.")
        print(json.dumps(message, ensure_ascii=False, indent=2))
        return 0

    try:
        client = boto3.client("sqs", region_name=region or None)
        resp = client.send_message(QueueUrl=queue_url, MessageBody=json.dumps(message, ensure_ascii=False))
    except Exception as exc:
        # 리전/자격증명/큐URL 오설정은 여기서 원인 그대로 드러난다.
        print(f"[FAIL] 전송 실패: {exc}")
        return 1

    print(f"[OK] 전송 성공 MessageId={resp['MessageId']}")
    print(json.dumps(message, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
