# Vision Summary → SQS 연동 가이드 (AI 담당자용)

카메라(엣지 디바이스)에서 5초마다 집계한 통행/응시(OTS/LTS) 요약 데이터를 SQS로 전송하는 방법을 설명합니다.

## 1. SQS가 뭔지 (한 줄 요약)

SQS는 메시지를 넣어두는 큐(대기열)입니다. 이 프로젝트에서는:

- **AI 담당자(당신)**: 5초마다 요약 데이터를 만들어서 큐에 `SendMessage`로 "넣기"만 함
- **백엔드팀**: 큐에서 메시지를 꺼내서(`ReceiveMessage`) DB에 저장

당신은 큐를 읽거나 지울 수 없고, **오직 넣기(SendMessage)만 가능**합니다. 큐 안에 데이터가 쌓여도 신경 쓸 필요 없습니다 — 백엔드가 알아서 가져갑니다.

## 2. 전달받아야 할 정보 (백엔드팀에게 요청하세요)

아래 값들을 백엔드 담당자에게 받아야 합니다:

| 값 | 용도 |
|---|---|
| `AWS_ACCESS_KEY_ID` | - |
| `AWS_SECRET_ACCESS_KEY` | - |
| `SQS_QUEUE_URL` | - |
| `AWS_REGION` | `ap-northeast-2` (고정) |
| `device_id` | 이 카메라의 고유 식별자 (예: `adscope-cam-01`) |
| `board_id` | 이 카메라가 설치된 광고 매체 식별자 (예: `MEDIA_001`) |

**주의**: 이 자격증명은 `sqs:SendMessage`(+ 필요 시 `s3:PutObject`)만 가능하도록 제한되어 있어서, 실수로 잘못 써도 다른 AWS 리소스에 영향을 줄 수 없습니다. 그래도 Slack/이메일 평문으로 주고받지 말고, 1Password 등 안전한 방법으로 전달받으세요. 코드나 Git에 절대 커밋하지 마세요 (환경변수나 `.env` 파일로 관리하고 `.gitignore`에 추가).

## 3. 설치

```bash
pip install boto3
```

## 4. 자격증명 설정

환경변수로 설정하는 걸 권장합니다 (코드에 하드코딩하지 마세요):

```bash
export AWS_ACCESS_KEY_ID="전달받은 값"
export AWS_SECRET_ACCESS_KEY="전달받은 값"
export AWS_DEFAULT_REGION="ap-northeast-2"
```

Windows PowerShell이라면:
```powershell
$env:AWS_ACCESS_KEY_ID = "전달받은 값"
$env:AWS_SECRET_ACCESS_KEY = "전달받은 값"
$env:AWS_DEFAULT_REGION = "ap-northeast-2"
```

이 저장소의 런타임은 프로젝트 루트의 `.env`도 자동으로 읽습니다. 로컬 실행 시에는 아래 형식으로 `.env`를 두고 값을 채우면 됩니다.

```dotenv
AWS_ACCESS_KEY_ID=전달받은_값
AWS_SECRET_ACCESS_KEY=전달받은_값
SQS_QUEUE_URL=https://sqs.ap-northeast-2.amazonaws.com/계정ID/큐이름
AWS_REGION=ap-northeast-2
device_id=adscope-cam-01
board_id=MEDIA_001
```

`.env`는 이미 `.gitignore`에 포함되어 있으므로 Git에 올리지 마세요. 기존 코드와의 호환을 위해 큐 URL은 `LOOVI_SQS_QUEUE_URL`도 지원하지만, 이 문서 기준으로는 `SQS_QUEUE_URL`을 쓰면 됩니다.

## 5. 메시지 스키마 (확정본)

**아래 구조를 절대 임의로 바꾸지 마세요.** 필드 이름, 타입, 중첩 구조가 모두 백엔드 DB 저장 로직과 1:1로 맞물려 있습니다. 필드를 추가/변경/삭제해야 하면 반드시 백엔드팀과 먼저 상의하세요.

```json
{
  "device_id": "adscope-cam-01",
  "board_id": "MEDIA_001",
  "seq": 1,
  "timestamp": "2026-07-22T05:05:00Z",
  "interval_sec": 5.0,
  "ots_count": 30,
  "lts_count": 9,
  "ots_demographics": {
    "gender": { "male": 16, "female": 12 },
    "male_age": { "under10": 0, "10s": 0, "20s": 4, "30s": 3, "40s": 2, "50s": 0, "60plus": 0 },
    "female_age": { "under10": 0, "10s": 0, "20s": 4, "30s": 3, "40s": 2, "50s": 0, "60plus": 0 }
  },
  "lts_demographics": {
    "gender": { "male": 5, "female": 4 },
    "male_age": { "under10": 0, "10s": 0, "20s": 4, "30s": 3, "40s": 2, "50s": 0, "60plus": 0 },
    "female_age": { "under10": 0, "10s": 0, "20s": 4, "30s": 3, "40s": 2, "50s": 0, "60plus": 0 }
  },
  "attention": {
    "avg_dwell_sec": 2.4,
    "dwell_sum_sec": 4.8,
    "dwell_distribution": { "1_to_2s": 6, "2_to_3s": 6, "3_to_4s": 6, "over_4s": 3 }
  }
}
```

### 필드별 유의사항

- **`timestamp`**: 반드시 **UTC**, ISO-8601 형식(`Z` 접미사). 로컬 타임존(KST) 그대로 보내면 안 됩니다.
- **`interval_sec`**: 항상 5.0이 아닐 수 있습니다 — 마지막 window가 짧게 끊기는 경우 실제 길이를 그대로 넣으세요.
- **`seq`**: 이 값은 **summary 메시지 전용 카운터**입니다(1부터 시작, 매 window마다 +1). 프로그램이 재시작되면 1부터 다시 시작해도 됩니다 — 백엔드는 `device_id + seq`가 아니라 `device_id + timestamp` 기준으로 순서를 판단하니, seq는 "중복 전송 감지용" 보조 값 정도로 취급하세요.
- **`ots_demographics` / `lts_demographics`의 `gender` 합**: 반드시 `male + female ≤ ots_count`(또는 `lts_count`) — 얼굴이 안 잡힌 사람은 성별/연령 집계에서 빠지기 때문에 부등호입니다. 합이 count를 넘는 경우는 버그이니 보내기 전에 확인하세요.
- **필드 이름의 숫자 시작 키** (`10s`, `20s` 등): JSON 키라서 문제없지만, 언어에 따라 다루는 방식이 다를 수 있으니 각별히 신경 써서 그대로 사용하세요.

## 6. 스키마 자체 검증 (보내기 전에 꼭 해보세요)

`docs/vision-summary-schema.json`에 이 스키마를 JSON Schema 형식으로 만들어뒀습니다. 보내기 전에 로컬에서 검증하면 필드 이름 오타나 구조 실수를 미리 잡을 수 있습니다.

```bash
pip install jsonschema
```

```python
import json
import jsonschema

with open("vision-summary-schema.json", encoding="utf-8") as f:
    schema = json.load(f)

def validate_message(message: dict):
    jsonschema.validate(message, schema)  # 틀리면 여기서 예외 발생
```

## 7. 전송 코드 예시

```python
import boto3
import json
import time
import logging
from datetime import datetime, timezone

REGION = "ap-northeast-2"
QUEUE_URL = "전달받은 SQS_QUEUE_URL"
DEVICE_ID = "adscope-cam-01"   # 전달받은 값으로 교체
BOARD_ID = "MEDIA_001"          # 전달받은 값으로 교체

sqs = boto3.client("sqs", region_name=REGION)
_seq = 0

def send_summary(window_data: dict, max_retries: int = 3):
    """
    window_data는 device_id/board_id/seq를 뺀 나머지 필드
    (timestamp, interval_sec, ots_count, ... attention)를 담은 dict
    """
    global _seq
    _seq += 1

    message = {
        "device_id": DEVICE_ID,
        "board_id": BOARD_ID,
        "seq": _seq,
        **window_data,
    }

    body = json.dumps(message, ensure_ascii=False)

    for attempt in range(1, max_retries + 1):
        try:
            sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=body)
            return
        except Exception as e:
            logging.warning(f"SQS 전송 실패 (시도 {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                logging.error(f"SQS 전송 최종 실패, 이 window는 유실됨: {message['seq']}")
            else:
                time.sleep(1 * attempt)  # 1초, 2초 백오프


# 5초마다 정확히 호출되도록 하는 예시 루프 (드리프트 보정 포함)
def run_loop(collect_window_fn):
    """collect_window_fn(): 5초 동안 집계한 결과를 위 스키마의 필드들로 돌려주는 함수"""
    next_tick = time.monotonic()
    while True:
        window_data = collect_window_fn()
        send_summary(window_data)
        next_tick += 5.0
        sleep_time = next_tick - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # 처리 시간이 5초를 넘긴 경우 — drift 누적 방지를 위해 다음 tick을 현재 시각 기준으로 재설정
            next_tick = time.monotonic()
```

### 전송 관련 유의사항

- **정확히 5초 간격**: 단순히 `time.sleep(5)`만 반복하면 처리 시간만큼 누적 지연(drift)이 생깁니다. 위 예시처럼 `next_tick` 기준으로 보정하세요.
- **재시도**: 네트워크 문제로 전송이 실패할 수 있습니다. 2~3회 정도 짧은 backoff로 재시도하고, 그래도 실패하면 해당 window는 포기(로그만 남김) — 5초 단위 데이터라 하나 유실돼도 전체 통계에 큰 영향 없습니다. 굳이 로컬에 큐잉해서 나중에 재전송하는 로직까지는 필요 없습니다.
- **메시지 크기**: 이 스키마는 1KB도 안 되므로 SQS 제한(256KB)은 전혀 문제되지 않습니다.
- **중복 전송 가능성**: SQS 자체가 "최소 한 번 전달(at-least-once)"이라, 백엔드 쪽에서 같은 메시지를 두 번 받을 수도 있습니다. 이건 백엔드가 처리할 부분이니 AI 쪽에서 신경 쓸 필요 없습니다.
- **전송 순서**: SQS(Standard 큐)는 순서를 100% 보장하지 않습니다. `timestamp`가 진짜 순서 기준이니, 전송 순서가 살짝 뒤바뀌어도 문제없습니다.

## 8. 확인 체크리스트 (첫 연동 시)

- [ ] `pip install boto3 jsonschema` 완료
- [ ] 환경변수로 자격증명 설정 완료 (코드에 하드코딩 안 함)
- [ ] `vision-summary-schema.json`으로 샘플 메시지 검증 통과
- [ ] 로컬 dry-run 통과: `python -m loovi_vision.tools.sqs_smoke_test --dry-run`
- [ ] 실제 SQS 테스트 메시지 1건 전송 통과: `python -m loovi_vision.tools.sqs_smoke_test`
- [ ] 5초 간격 전송 테스트 (드리프트 없이 몇 분간 안정적으로 도는지 확인)
- [ ] 백엔드팀에 "테스트 메시지 보냈다" 알리고, DB에 정상적으로 쌓이는지 같이 확인
