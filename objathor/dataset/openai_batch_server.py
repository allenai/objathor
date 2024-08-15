import json
import logging
import os.path
import sqlite3
import threading
import traceback
import uuid
from argparse import ArgumentParser
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Sequence, Tuple, List, Union

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, request, jsonify
from openai import OpenAI


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


class OpenAIBatchServer:
    def __init__(
        self,
        save_dir: str,
        batch_after_minutes: float,
        batch_after_mb: float,
        batch_after_num: int,
        check_batch_status_interval: int = 1,
    ):
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.batch_after_minutes = batch_after_minutes
        self.batch_after_mb = batch_after_mb
        self.batch_after_num = batch_after_num

        assert batch_after_num > 0
        assert batch_after_mb > 0
        assert batch_after_minutes > 0

        self.check_batch_status_interval = check_batch_status_interval

        self.db_lock = threading.RLock()
        self.openai_client = OpenAI()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.conn = sqlite3.connect(
            os.path.join(self.save_dir, "requests.db"), check_same_thread=False
        )
        self.create_table()

        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            func=self.batch_requests,
            trigger="interval",
            minutes=self.batch_after_minutes,
            id="batch_requests",
        )
        self.scheduler.add_job(
            func=self.check_batch_status,
            trigger="interval",
            minutes=self.check_batch_status_interval,
            id="check_batch_status",
        )
        self.scheduler.start()

        self.app = Flask(__name__)
        self.setup_routes()

        self._num_recent_requests = 0
        self._num_recent_mb = 0
        self._time_of_last_batch = datetime.now()

        self._trigger_check_batch_status()
        self._trigger_batch_requests()

    def setup_routes(self):
        @self.app.route("/put", methods=["POST"])
        def put():
            request_data = request.json
            uid = self.put(request_data)
            return jsonify({"uid": uid})

        @self.app.route("/check_status/<uid>", methods=["GET"])
        def check_status(uid):
            status = self.check_status(uid)
            return jsonify({"status": status.value})

        @self.app.route("/get/<uid>", methods=["GET"])
        def get(uid):
            output = self.get(uid)
            return jsonify({"output": output})

        @self.app.route("/ping/", methods=["GET"])
        def ping():
            return jsonify({"output": "pong"})

        @self.app.route("/delete/<uid>", methods=["DELETE"])
        def delete(uid):
            self.delete(uid)
            return "", 204

        @self.app.route("/delete_older_than", methods=["DELETE"])
        def delete_older_than():
            date = datetime.fromisoformat(request.json["date"])
            deleted_count = self.delete_older_than(date)
            return jsonify({"deleted_count": deleted_count})

    def run(self, host="0.0.0.0", port=5000, **flask_kwargs):
        self.app.run(host=host, port=port, **flask_kwargs)

    def create_table(self):
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    uid TEXT PRIMARY KEY,
                    request TEXT,
                    request_mb REAL,
                    status TEXT,
                    file_id TEXT,
                    batch_id TEXT,
                    created_at TIMESTAMP,
                    output TEXT
                )
            """
            )
            self.conn.commit()
        self.logger.info("Database table created or verified")

    def put(self, request_data: Dict[str, Any]):
        uid = str(uuid.uuid4())
        request_str = json.dumps(request_data)
        request_mb = len(request_str) / (1024 * 1024)

        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO requests (uid, request, request_mb, status, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    uid,
                    request_str,
                    request_mb,
                    RequestStatus.VALIDATING.value,
                    datetime.now(),
                ),
            )
            self.conn.commit()
            self._num_recent_requests += 1
            self._num_recent_mb += request_mb

        self.logger.info(f"New request added with UID: {uid}")

        if (self._num_recent_requests >= self.batch_after_num) or (
            self._num_recent_mb >= self.batch_after_mb
        ):
            self._trigger_batch_requests()

        return uid

    def _trigger_scheduled_job(self, id: str):
        for job in self.scheduler.get_jobs():
            if job.id == id:
                job.modify(next_run_time=datetime.now())
                break

    def _trigger_check_batch_status(self):
        self._trigger_scheduled_job("check_batch_status")

    def _trigger_batch_requests(self):
        self._trigger_scheduled_job("batch_requests")

    def check_status(self, uid: str):
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT status FROM requests WHERE uid = ?", (uid,))
            result = cursor.fetchone()
        status = RequestStatus(result[0]) if result else RequestStatus.NOT_FOUND
        self.logger.info(f"Status check for UID {uid}: {status}")
        return status

    def get(self, uid: str):
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT output FROM requests WHERE uid = ? AND status = ?",
                (uid, RequestStatus.COMPLETED.value),
            )
            result = cursor.fetchone()

        if result:
            self.logger.info(f"Retrieved output for UID: {uid}")
            return json.loads(result[0])
        else:
            self.logger.warning(f"No completed output found for UID: {uid}")
            return None

    def delete(self, uid: Union[str, List[str], Tuple[str, ...]]):
        with self.db_lock:
            cursor = self.conn.cursor()

            if isinstance(uid, str):
                uids = (uid,)
            else:
                uids = uid

            file_id_batch_id_tuples = set()
            deleted_uids = []
            for uid in uids:
                cursor.execute(
                    "SELECT file_id, batch_id FROM requests WHERE uid = ?", (uid,)
                )
                file_id_batch_id_tuple = cursor.fetchone()
                if file_id_batch_id_tuple:
                    deleted_uids.append(uid)
                    file_id_batch_id_tuples.add(file_id_batch_id_tuple)
                    cursor.execute("DELETE FROM requests WHERE uid = ?", (uid,))
                else:
                    self.logger.warning(f"No request found for UID: {uid}")

            if len(deleted_uids) > 0:
                self.conn.commit()
                for uid in deleted_uids:
                    self.logger.info(f"Deleted request with UID: {uid}")

            # Only need to cleanup files, batches, and commit the results if something was actually deleted.
            if len(file_id_batch_id_tuples) > 1:
                for file_id, batch_id in file_id_batch_id_tuples:
                    self.cleanup_file_and_batch(
                        input_file_id=file_id,
                        batch_id=batch_id,
                        delete_only_if_no_references=True,
                    )

    def delete_older_than(self, date: datetime):
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT uid FROM requests WHERE created_at < ?",
                (date,),
            )
            uids = [uid for (uid,) in cursor.fetchall()]

            self.delete(uids)

        self.logger.info(f"Deleted all requests (=={len(uids)}) older than: {date}")
        return len(uids)

    def cleanup_file_and_batch(
        self,
        input_file_id: str,
        batch_id: str,
        delete_only_if_no_references: bool = False,
    ):
        with self.db_lock:
            should_delete = True

            if delete_only_if_no_references:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM requests WHERE file_id = ? OR batch_id = ?",
                    (input_file_id, batch_id),
                )
                count = cursor.fetchone()[0]
                should_delete = count == 0

            if should_delete:
                if batch_id:
                    try:
                        batch_info = self.openai_client.batches.retrieve(batch_id)
                        if batch_info.status not in [
                            RequestStatus.COMPLETED.value,
                            RequestStatus.FAILED.value,
                            RequestStatus.EXPIRED.value,
                            RequestStatus.CANCELLED.value,
                            RequestStatus.FINALIZING.value,
                        ]:
                            self.openai_client.batches.cancel(batch_id)
                            self.logger.info(f"Cancelled batch: {batch_id}")
                    except Exception:
                        self.logger.error(
                            f"Error cancelling batch {batch_id}: {traceback.format_exc()}"
                        )
                if input_file_id:
                    try:
                        # Running retrieve checks that input_file_id exists
                        self.openai_client.files.retrieve(input_file_id)
                        # If the above didn't error, we now attempt to delete
                        self.openai_client.files.delete(input_file_id)
                        self.logger.info(f"Deleted file: {input_file_id}")
                    except Exception:
                        self.logger.error(
                            f"Error deleting file {input_file_id}: {traceback.format_exc()}"
                        )

    def batch_requests(self):
        self.logger.info("Checking if there are pending requests to batch and process.")

        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT uid, request FROM requests WHERE status = ?",
                (RequestStatus.VALIDATING.value,),
            )
            uid_and_request_tuples = cursor.fetchall()

        enough_num = len(uid_and_request_tuples) >= max(self.batch_after_num, 1)
        enough_mb = self.get_total_size(uid_and_request_tuples) >= self.batch_after_mb
        enough_time = (datetime.now() - self._time_of_last_batch).total_seconds() >= (
            60 * self.batch_after_minutes
        ) * 0.99  # 0.99 to avoid timing mismatch between the scheduler and the batch_after_minutes

        if enough_num or enough_mb or (enough_time and len(uid_and_request_tuples) > 0):
            why_logging_message = (
                f"Processing batch of {len(uid_and_request_tuples)} requests as"
            )
            if enough_num:
                why_logging_message += f" {len(uid_and_request_tuples)} >= {max(self.batch_after_num, 1)} requests."
            elif enough_mb:
                why_logging_message += f" {self.get_total_size(uid_and_request_tuples)} >= {self.batch_after_mb} MB of requests."
            else:
                why_logging_message += (
                    f" {(datetime.now() - self._time_of_last_batch).total_seconds() / 60}"
                    f" >= {0.99 * self.batch_after_minutes} minutes have elapsed."
                )

            self.logger.info(why_logging_message)
            self.process_batch(uid_and_request_tuples)

            self._time_of_last_batch = datetime.now()
            self._num_recent_requests = 0
            self._num_recent_mb = 0
        else:
            self.logger.info(
                f"Too few requests ({len(uid_and_request_tuples)}) to batch."
            )

    def get_total_size(self, uid_and_request_tuples: Sequence[Tuple[str, str]]):
        return sum(
            len(json.dumps(json.loads(req))) for uid, req in uid_and_request_tuples
        ) / (1024 * 1024)

    def process_batch(self, uid_and_request_tuples: Sequence[Tuple[str, str]]):
        if len(uid_and_request_tuples) == 0:
            self.logger.info("Attempted to process batch with no requests.")
            return

        batch_file_content = "\n".join(
            json.dumps(
                {
                    "custom_id": uid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": json.loads(body_str),
                }
            )
            for uid, body_str in uid_and_request_tuples
        )

        try:
            file = self.openai_client.files.create(
                file=batch_file_content.encode(), purpose="batch"
            )
            self.logger.info(f"Created batch file with ID: {file.id}")

            batch = self.openai_client.batches.create(
                input_file_id=file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            self.logger.info(f"Created batch with ID: {batch.id}")

            with self.db_lock:
                cursor = self.conn.cursor()
                for uid, _ in uid_and_request_tuples:
                    cursor.execute(
                        "UPDATE requests SET status = ?, file_id = ?, batch_id = ? WHERE uid = ?",
                        (RequestStatus.IN_PROGRESS.value, file.id, batch.id, uid),
                    )
                self.conn.commit()
            self.logger.info(
                f"Updated {len(uid_and_request_tuples)} requests to IN_PROGRESS status"
            )

        except Exception:
            self.logger.error(f"Error processing batch: {traceback.format_exc()}")
            with self.db_lock:
                cursor = self.conn.cursor()
                for uid, _ in uid_and_request_tuples:
                    cursor.execute(
                        "UPDATE requests SET status = ? WHERE uid = ?",
                        (RequestStatus.FAILED.value, uid),
                    )
                self.conn.commit()
            self.logger.info(
                f"Updated {len(uid_and_request_tuples)} requests to FAILED status due to batch processing error"
            )

    def check_batch_status(self):
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT DISTINCT batch_id FROM requests WHERE status = ?",
                (RequestStatus.IN_PROGRESS.value,),
            )
            batch_ids = cursor.fetchall()

        for batch_id in batch_ids:
            try:
                batch = self.openai_client.batches.retrieve(batch_id[0])
                self.logger.info(
                    f"Checked status of batch {batch_id[0]}: {batch.status}."
                )
                if batch.status == RequestStatus.COMPLETED.value:
                    if batch.request_counts.failed > 0:
                        self.logger.info(
                            f"Batch {batch.id} has {batch.request_counts.failed} failed requests"
                            f" (of {batch.request_counts.total} total requests), processing failed batch."
                        )
                        self.process_failed_batch(batch)
                    else:
                        self.process_completed_batch(batch)
                elif batch.status in [
                    RequestStatus.FAILED.value,
                    RequestStatus.EXPIRED.value,
                    RequestStatus.CANCELLED.value,
                ]:
                    self.process_failed_batch(batch)
            except Exception:
                self.logger.error(
                    f"Error checking batch status: {traceback.format_exc()}"
                )

    def process_completed_batch(self, batch):
        output_file = self.openai_client.files.content(
            file_id=batch.output_file_id
        ).content.decode()

        outputs = [json.loads(line) for line in output_file.split("\n") if line]

        with self.db_lock:
            cursor = self.conn.cursor()
            for output in outputs:
                cursor.execute(
                    "UPDATE requests SET status = ?, output = ? WHERE uid = ?",
                    (
                        RequestStatus.COMPLETED.value,
                        json.dumps(output),
                        output["custom_id"],
                    ),
                )
            self.conn.commit()
        self.logger.info(
            f"Processed completed batch {batch.id}: {len(outputs)} requests updated"
        )

        self.cleanup_file_and_batch(
            input_file_id=batch.input_file_id, batch_id=batch.id
        )

    def process_failed_batch(self, batch):
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE requests SET status = ? WHERE batch_id = ?",
                (RequestStatus.FAILED.value, batch.id),
            )
            self.conn.commit()
        self.logger.info(
            f"Processed failed batch {batch.id}: all requests updated to FAILED status"
        )

        self.cleanup_file_and_batch(
            input_file_id=batch.input_file_id, batch_id=batch.id
        )


def main():
    parser = ArgumentParser(
        description="Starts a OpenAIBatchServer running on this machine."
    )
    parser.add_argument(
        "--batch_after_minutes",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--batch_after_mb",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--batch_after_num",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
    )

    args = parser.parse_args()

    server = OpenAIBatchServer(
        batch_after_minutes=args.batch_after_minutes,
        batch_after_mb=args.batch_after_mb,
        batch_after_num=args.batch_after_num,
        save_dir=args.save_dir,
    )
    server.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":

    main()
