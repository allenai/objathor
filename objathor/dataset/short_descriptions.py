import os
import base64
import io
import json
import time
import signal
from contextlib import contextmanager
from typing import Tuple, Any, Dict, Optional, Iterator, List, cast
from abc import ABC, abstractmethod
import gzip
from pathlib import Path

# pip install tqdm prior lmdb pillow openai pydantic

from tqdm import tqdm
import prior

try:
    import lmdb
except ImportError:
    raise ImportError("Missing dependency lmdb. Please install it with `pip install lmdb` and try again.")

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise ImportError(f"Missing dependency PIL. Please install it with `pip install pillow` and try again.")

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(f"Missing dependency openai. Please install it with `pip install openai` and try again.")

try:
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise ImportError(
        f"Missing dependency torch. Please install it, e.g. following"
        f" https://pytorch.org/get-started/previous-versions, and try again."
    )

try:
    import pydantic
except ImportError:
    raise ImportError(
        f"Missing dependency pydantic. Please install it with `pip install pydantic` and try again."
    )


ASSETS_VERSION = "2025_06_10"
OBJAVERSE_AND_DB_DIR = Path("/weka/prior/datasets/holodeck_data") / ASSETS_VERSION


# max batch size in bytes and max requests per batch,
# from https://platform.openai.com/docs/guides/batch#rate-limits
MAX_BATCH_SIZE_BYTES = 200_000_000
MAX_BATCH_REQUESTS = 50_000

DEFAULT_SYS_PROMPT = "You are a helpful assistant."


MAP_SIZE = 10_000_000_000  # 10 GiB

# Note: reusing database from 2024_08_16 for reannotated (thor properties, texture replacement) subset 2025_06_10
COMPLETE_KEY = (
    f"__SharedJSONDatabase"
    f"__{ASSETS_VERSION if ASSETS_VERSION != '2025_06_10' else '2024_08_16'}"
    f"__dump_complete__"
)


class DescriptionOutput(pydantic.BaseModel):
    one_word: str
    two_words: str
    three_words: str
    four_words: str
    five_words: str


class SharedJSONDatabase:
    def __init__(
        self, db_path=str(OBJAVERSE_AND_DB_DIR), readonly=True, map_size=MAP_SIZE
    ):
        """
        Args:
            db_path (str): Path to LMDB database directory.
            readonly (bool): Open the database in readonly mode (for workers).
            map_size (int): Max size in bytes (needed only for writer mode).
        """
        self.pid = None
        self.db_path = db_path
        self.readonly = readonly
        self.map_size = map_size
        self.env = None

    def _open_env(self):
        self.env = lmdb.open(
            self.db_path,
            readonly=self.readonly,
            map_size=self.map_size,
            max_readers=2048,
            lock=not self.readonly,  # No lock needed for read-only
            readahead=True,
            meminit=False,
        )

    def _check_pid(self):
        if os.getpid() != self.pid:
            # Fork detected: close and reopen
            self.close()
            self.pid = os.getpid()
            self._open_env()

    @classmethod
    def database_exists(cls, db_path=str(OBJAVERSE_AND_DB_DIR), map_size=MAP_SIZE):
        if not os.path.exists(db_path):
            return False

        could_open = False
        db = cls(db_path, map_size=map_size)
        try:
            db._check_pid()
            could_open = True

            complete_key = db.get(COMPLETE_KEY)

            if complete_key is None:
                return False
            if complete_key != COMPLETE_KEY:
                raise ValueError(
                    f"database complete key {complete_key} differs from expected {COMPLETE_KEY}"
                )

            return True
        except lmdb.Error:
            return False
        finally:
            if could_open:
                db.close()

    @classmethod
    def from_json(cls, json_data, db_path=str(OBJAVERSE_AND_DB_DIR), map_size=MAP_SIZE):
        def put(key, value):
            k = key.encode("utf-8")
            # Compress the serialized JSON
            raw_json = json.dumps(value).encode("utf-8")
            compressed = gzip.compress(raw_json)
            txn.put(k, compressed)

        """Create a new LMDB database from a big JSON dictionary."""
        os.makedirs(db_path, exist_ok=True)
        instance = cls(db_path, readonly=False, map_size=map_size)
        instance._check_pid()

        with instance.env.begin(write=True) as txn:
            for key, value in tqdm(json_data.items(), "Dumping dataset"):
                put(key, value)
            put(COMPLETE_KEY, COMPLETE_KEY)
            instance.env.sync()

        instance.close()

        return cls(db_path, map_size=map_size)

    def get(self, key, alt=None):
        self._check_pid()
        """Retrieve a value by key (str)."""
        with self.env.begin() as txn:
            raw = txn.get(key.encode("utf-8"))
            if raw is None:
                return alt
            # Decompress before loading JSON
            decompressed = gzip.decompress(raw)
            return json.loads(decompressed.decode("utf-8"))

    def __getitem__(self, key):
        return self.get(key)

    def __len__(self):
        self._check_pid()
        with self.env.begin() as txn:
            stats = txn.stat()
            return stats["entries"]

    def keys(self):
        self._check_pid()
        """Yield all keys one by one."""
        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                for key, _ in cursor:
                    yield key.decode("utf-8")

    def items(self):
        self._check_pid()
        """Yield (key, decompressed value) pairs."""
        with self.env.begin() as txn:
            with txn.cursor() as cursor:
                for key, value in cursor:
                    decompressed = gzip.decompress(value)
                    yield key.decode("utf-8"), json.loads(decompressed.decode("utf-8"))

    def close(self):
        if self.env is not None:
            self.env.close()
        self.env = None
        self.pid = None


@contextmanager
def block_signals(signals: list[int]):
    previous_blocked = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, signals)
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_blocked)


def create_collage(
    views: list[Image.Image],
    num_rows_cols: Tuple[int, int],
    target_resolution: Tuple[int, int] = None,
    ensure_same_resolution_views=True,
    border_color="black",
    border_width=2,
):
    nrows, ncols = num_rows_cols
    assert len(views) == nrows * ncols

    resolution = views[0].size
    if ensure_same_resolution_views:
        assert all(view.size == resolution for view in views)

    target_resolution = target_resolution or resolution

    collage = Image.new(
        "RGB", (target_resolution[0] * ncols, target_resolution[1] * nrows)
    )
    for i, view in enumerate(views):
        view = view.resize(target_resolution)
        collage.paste(
            view, (i % ncols * target_resolution[0], i // ncols * target_resolution[1])
        )

    # Draw borders between images
    draw = ImageDraw.Draw(collage)
    # Vertical lines
    for x in range(1, ncols):
        draw.line(
            [
                (x * target_resolution[0], 0),
                (x * target_resolution[0], target_resolution[1] * nrows),
            ],
            fill=border_color,
            width=border_width,
        )
    # Horizontal lines
    for y in range(1, nrows):
        draw.line(
            [
                (0, y * target_resolution[1]),
                (target_resolution[0] * ncols, y * target_resolution[1]),
            ],
            fill=border_color,
            width=border_width,
        )

    return collage


def encode_png(image: Image.Image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return base64.b64encode(image_bytes.getvalue()).decode("utf-8")


def create_query(
    custom_id: str,
    query_message: Optional[str] = None,
    query_image: Optional[Image.Image] = None,
    sys_prompt: str = DEFAULT_SYS_PROMPT,
    model="gpt-4.1",
    max_tokens=8192,
    image_detail="high",
):
    content = []

    if query_message is not None:
        content.append(
            {
                "type": "text",
                "text": query_message,
            },
        )

    if query_image is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_png(query_image)}",
                    "detail": image_detail,
                },
            },
        )

    assert len(content), f"{custom_id} has no text or image queries"

    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "DescriptionOutput",
                    "schema": DescriptionOutput.model_json_schema(),
                }
            },
        },
    }

    return request


class AbstractQueryDataset(Dataset, ABC):
    encode_query: bool

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def make_query_dict(self, index: int) -> Tuple[str, Optional[Dict[str, Any]]]:
        raise NotImplementedError

    def __getitem__(self, index) -> Tuple[str, Optional[Any]]:
        custom_id, maybe_query = self.make_query_dict(index)

        if not self.encode_query or maybe_query is None:
            return custom_id, maybe_query

        return custom_id, (json.dumps(maybe_query) + "\n").encode("utf-8")


class QueryDataSource:
    def __init__(
        self,
        dataset: AbstractQueryDataset,
        data_load_batch_size: int = 64,
        data_load_num_workers: int = 16,
    ):
        self.dataset = dataset
        self.batch_size = data_load_batch_size
        self.num_workers = data_load_num_workers

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def do_not_collate_valid(batch):
        return [datum for datum in batch if datum[1] is not None]

    def iterator(self) -> Iterator[Tuple[str, Optional[Any]]]:
        for batch in DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.do_not_collate_valid,
        ):
            for custom_id, query_or_bytes in batch:
                yield custom_id, query_or_bytes


class BatchActions:
    def __init__(
        self,
        dataset: Optional[AbstractQueryDataset] = None,
        data_load_batch_size: int = 1,
        data_load_num_workers: int = 0,
        **openai_kwargs: Any,
    ):
        self.dataset = dataset
        self.data_load_batch_size = data_load_batch_size
        self.data_load_num_workers = data_load_num_workers
        self.openai_kwargs = openai_kwargs

        self._data_source = None
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(**self.openai_kwargs)
        return self._client

    @property
    def data_source(self):
        if self._data_source is None:
            self._data_source = QueryDataSource(
                self.dataset, self.data_load_batch_size, self.data_load_num_workers
            )
        return self._data_source

    def immediate(self) -> List[Dict[str, str]]:
        responses = []

        for custom_id, query_dict in tqdm(
            self.data_source.iterator(), total=len(self.data_source), desc="Processing"
        ):
            response = self.client.chat.completions.create(**query_dict["body"])
            responses.append(
                dict(custom_id=custom_id, message=response.choices[0].message.content)
            )

        return responses

    def submit(self, batch_ids_path: str):
        if os.path.exists(batch_ids_path):
            print(f"Batch IDs file {batch_ids_path} exists, skipping resubmission")
            return

        batch_file = io.BytesIO()
        total_queries = 0
        batch_queries = 0
        batch_file_ids = []

        def add_file():
            batch_file.seek(0)
            batch_file_ids.append(
                self.client.files.create(file=batch_file, purpose="batch").id
            )
            batch_file.close()

        try:
            for uuid, query_bytes in tqdm(
                self.data_source.iterator(),
                total=len(self.data_source),
                desc="Compiling and uploading files",
            ):
                if (
                    batch_file.tell() + len(query_bytes) > MAX_BATCH_SIZE_BYTES
                    or batch_queries == MAX_BATCH_REQUESTS
                ):
                    add_file()

                    batch_file = io.BytesIO()
                    batch_queries = 0

                batch_file.write(query_bytes)
                batch_queries += 1
                total_queries += 1

            if batch_queries > 0:
                add_file()

        except:
            print(
                "\nException with uploaded batch file ids:\n",
                batch_file_ids,
                flush=True,
            )
            raise

        print(f"Total annotations: {total_queries}, total files: {len(batch_file_ids)}")

        with block_signals([signal.SIGINT]):
            # make sure no preemption during batch submission
            batch_ids = []
            try:
                for batch_file_id in tqdm(batch_file_ids, "Submitting batches"):
                    batch = self.client.batches.create(
                        input_file_id=batch_file_id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                    )
                    batch_ids.append(batch.id)

            except:
                print(
                    "\nException with submitted batch id(s):\n", batch_ids, flush=True
                )
                raise

            print(
                f"Submitted {len(batch_ids)} batch job(s) with ID(s): {' '.join(batch_ids)}"
            )

            with open(batch_ids_path, "w") as f:
                f.write("\n".join(batch_ids))

    def retrieve(self, batch_ids_path: str, out_dir: str, save_source: bool = False):
        os.makedirs(out_dir, exist_ok=True)

        done_statuses = ["completed", "expired", "cancelled", "failed"]

        with open(batch_ids_path, "r") as f:
            batch_ids = f.read().strip().splitlines()

        out_file = f"{out_dir}/raw_{os.path.basename(batch_ids_path)}.jsonl"

        for batch_id in batch_ids:
            time_to_wait = 5
            while (
                batch := self.client.batches.retrieve(batch_id)
            ).status not in done_statuses:
                print(
                    f"Batch job {batch_id} is {batch.status}, waiting {time_to_wait} seconds before checking again"
                )
                time.sleep(time_to_wait)
                time_to_wait = min(time_to_wait * 2, 10 * 60)  # cap at 10 minutes

            if batch.status != "completed":
                print(f"Batch job {batch_id} did not complete successfully!")

            if save_source:
                task_file = self.client.files.content(batch.input_file_id)
                with open(f"{out_dir}/source__{batch.input_file_id}", "w") as f:
                    f.write(task_file.text)

            try:
                batch_file = self.client.files.content(batch.output_file_id)
                batch_file_lines = batch_file.content.decode("utf-8").splitlines()
                with open(out_file, "a") as f:
                    for line in batch_file_lines:
                        result = json.loads(line)
                        custom_id: str = result["custom_id"]
                        ans = result["response"]["body"]["choices"][0]["message"][
                            "content"
                        ]
                        res = json.dumps({custom_id: ans})
                        f.write(res + "\n")

                print(f"Appended {batch_id} results to {out_file}")
            except KeyboardInterrupt:
                raise
            except:
                print(
                    f"Failed to append {batch_id} results with error info"
                    f"\n{self.client.files.content(batch.error_file_id).text}"
                )

    def submit_and_retrieve(
        self,
        batch_ids_path: str,
        out_dir: Optional[str] = None,
        save_source: bool = False,
    ):
        if not os.path.exists(batch_ids_path):
            self.submit(batch_ids_path)

        if out_dir is not None:
            self.retrieve(
                batch_ids_path=batch_ids_path, out_dir=out_dir, save_source=save_source
            )


_split_to_whitelisted_assets = None


def get_split_to_whitelisted_assets() -> Dict[str, List[str]]:
    if ASSETS_VERSION not in [
        "2025_06_10",
        "2024_08_16",
    ]:
        raise NotImplementedError(
            "holodeck-scale-data (which includes whitelisted assets"
            " and their assigned splits) is only available for"
            "\nASSETS_VERSION in [2024_08_16, 2025_06_10]"
        )

    global _split_to_whitelisted_assets

    if _split_to_whitelisted_assets is None:
        _split_to_whitelisted_assets = cast(
            Dict[str, List[str]],
            prior.load_dataset(
                "holodeck-scale-data",
                modality="uuid splits",
                offline=False,
            ),
        )

    return _split_to_whitelisted_assets


class MergeDescriptionQueryDataset(AbstractQueryDataset):
    model = "gpt-4.1-mini"

    merge_description_sys_prompt = """
Using the specified amount of words, describe the object in simple words keeping the essentials of the visual appearance (e.g. the dominant color). Use at least one noun. Do not add any additional comments.
    """.strip()

    def __init__(
        self,
        encode_query: bool = True,
        num_queries: int | None = None,
        first_query: int = 0,
    ):
        self.encode_query = encode_query
        self.annotation = SharedJSONDatabase()

        valid_ids = set(sum([split_items for split, split_items in get_split_to_whitelisted_assets().items()], []))
        self.filtered_ids = sorted(valid_ids & set(self.annotation.keys()))

        if num_queries is not None:
            assert num_queries > 0, "num_queries must be larger than 0 if given"
            self.filtered_ids = self.filtered_ids[
                first_query : first_query + num_queries
            ]

        print(len(self.filtered_ids), "ids to process")

    def __len__(self):
        return len(self.filtered_ids)

    def make_query_dict(self, index: int) -> Tuple[str, Optional[Dict[str, Any]]]:
        custom_id = f"asset_{index}"
        anno = self.annotation[self.filtered_ids[index]]
        descriptions = [value for key, value in anno.items() if "description" in key]
        message = "\n - ".join(["Descriptions:"] + descriptions)

        if not self.encode_query:
            print(message)

        return custom_id, create_query(
            custom_id,
            query_message=message,
            sys_prompt=self.merge_description_sys_prompt,
            model=self.model,
        )


if __name__ == "__main__":
    import argparse

    def get_args():
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)

        submit_parser = subparsers.add_parser("submit")
        submit_parser.add_argument(
            "batch_ids_file", help="Write batch IDs to this file"
        )
        submit_parser.add_argument(
            "--out_dir",
            help="If given, wait for batch to finish and save results to this directory",
            default=None,
        )
        submit_parser.add_argument(
            "--num_queries",
            help="if given, number of queries to submit",
            default=None,
            type=int,
        )

        retrieve_parser = subparsers.add_parser("retrieve")
        retrieve_parser.add_argument(
            "batch_ids_file", help="Read batch IDs from this file"
        )
        retrieve_parser.add_argument(
            "out_dir",
            help="Wait for batch to finish and save results to this directory",
        )

        immediate_parser = subparsers.add_parser("immediate")
        immediate_parser.add_argument(
            "--first_query", help="first query to submit", default=0, type=int
        )
        immediate_parser.add_argument(
            "--num_queries", help="number of queries to submit", default=1, type=int
        )

        return parser.parse_args()

    def main():
        args = get_args()

        if args.command == "immediate":
            actions = BatchActions(
                MergeDescriptionQueryDataset(
                    encode_query=False,
                    num_queries=args.num_queries,
                    first_query=args.first_query,
                )
            )
            responses = actions.immediate()
            for response in responses:
                print(response["custom_id"], response["message"])

        elif args.command == "submit":
            actions = BatchActions(
                MergeDescriptionQueryDataset(encode_query=True, num_queries=args.num_queries)
            )
            actions.submit_and_retrieve(
                args.batch_ids_file, args.out_dir, save_source=True
            )

        elif args.command == "retrieve":
            actions = BatchActions()
            actions.retrieve(args.batch_ids_file, args.out_dir, save_source=True)

        else:
            raise NotImplementedError

    main()
    print("DONE")
