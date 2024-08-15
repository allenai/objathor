import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union

import requests

from objathor.dataset.openai_batch_server import RequestStatus


class OpenAIBatchClientError(Exception):
    pass


class OpenAIBatchClient:
    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"

        if not self.ping():
            raise OpenAIBatchClientError(
                f"Could not connect to OpenAIBatchServer at {self.base_url} in 10 seconds."
            )

    def _make_request(self, method, endpoint, data=None, timeout: Optional[int] = None):
        url = f"{self.base_url}/{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, timeout=timeout)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=timeout)
            elif method == "DELETE":
                response = requests.delete(url, json=data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()

            if len(response.content) == 0:
                assert (
                    method == "DELETE"
                ), f"Response content is empty for method {method}"
                return None

            return response.json()

        except requests.RequestException as e:
            print(f"Request failed: {traceback.format_exc()}")
            raise OpenAIBatchClientError(
                f"Request to {endpoint} failed: {traceback.format_exc()}"
            )

    def ping(self) -> bool:
        # print(f"Pinging server at {self.base_url}")
        try:
            response = self._make_request("GET", "ping", timeout=10)
        except requests.Timeout:
            return False

        if response["output"] != "pong":
            raise OpenAIBatchClientError("Server did not respond with 'pong'")

        return True

    def put(self, request: Dict[str, Any]):
        request_str_short = str(request)
        if len(request_str_short) > 100:
            request_str_short = request_str_short[:100] + "..."
        print(f"Putting request: {request_str_short}")
        response = self._make_request("POST", "put", request)
        return response["uid"]

    def check_status(self, uid: str):
        print(f"Checking status for UID: {uid}")
        response = self._make_request("GET", f"check_status/{uid}")
        return RequestStatus(response["status"])

    def get(self, uid: str):
        print(f"Getting result for UID: {uid}")
        response = self._make_request("GET", f"get/{uid}")
        return response["output"]

    def delete(self, uid: str):
        print(f"Deleting request with UID: {uid}")
        self._make_request("DELETE", f"delete/{uid}")

    def delete_older_than(self, date: Union[str, datetime]):
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        print(f"Deleting requests older than: {date}")
        self._make_request("DELETE", "delete_older_than", {"date": date.isoformat()})


if __name__ == "__main__":
    client = OpenAIBatchClient(host="0.0.0.0", port=5000)

    client.delete_older_than("2024-08-16")
