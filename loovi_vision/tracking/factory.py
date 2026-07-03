from .custom import CustomTracker
from .ultralytics_tracker import UltralyticsTracker


def create_tracker(settings):
    # 설정된 backend를 우선 사용하고, 실패하면 custom tracker로 계속 실행한다.
    if settings.tracker_backend in ("botsort", "bytetrack"):
        try:
            return UltralyticsTracker(settings)
        except Exception as exc:
            print(f"  Tracker backend {settings.tracker_backend} failed ({exc}); using custom")
    return CustomTracker(settings)
