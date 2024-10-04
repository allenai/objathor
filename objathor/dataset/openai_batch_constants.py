from enum import Enum


class RequestStatus(str, Enum):
    VALIDATING = "validating"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    NOT_FOUND = "not_found"

    def is_complete(self):
        return self == RequestStatus.COMPLETED

    def is_in_progress(self):
        return self in [
            RequestStatus.VALIDATING,
            RequestStatus.IN_PROGRESS,
            RequestStatus.FINALIZING,
        ]

    def is_fail(self):
        return self in [
            RequestStatus.FAILED,
            RequestStatus.EXPIRED,
            RequestStatus.CANCELLING,
            RequestStatus.CANCELLED,
            RequestStatus.NOT_FOUND,
        ]
