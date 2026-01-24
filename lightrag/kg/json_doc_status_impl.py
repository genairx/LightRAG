from dataclasses import dataclass
import os
import time
import fcntl
import asyncio
import contextlib
from typing import Any, Union, final

from lightrag.base import (
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from .json_common import JsonStorageLockMixin
from lightrag.utils import (
    load_json,
    logger,
    write_json,
    get_pinyin_sort_key,
    compute_mdhash_id,
)
from lightrag.exceptions import StorageNotInitializedError
from .shared_storage import (
    get_namespace_data,
    get_storage_lock,
    get_data_init_lock,
    get_update_flag,
    set_all_update_flags,
    clear_all_update_flags,
    try_initialize_namespace,
    reset_namespace_initialization,
)


@final
@dataclass
class JsonDocStatusStorage(DocStatusStorage, JsonStorageLockMixin):
    """JSON implementation of document status storage"""

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        working_dir_hash = compute_mdhash_id(working_dir, prefix="")
        if self.workspace:
            # Include workspace in the file path for data isolation
            workspace_dir = os.path.join(working_dir, self.workspace)
            self.final_namespace = (
                f"{self.workspace}_{working_dir_hash}_{self.namespace}"
            )
        else:
            # Default behavior when workspace is empty
            self.final_namespace = f"{working_dir_hash}_{self.namespace}"
            self.workspace = "_"
            workspace_dir = working_dir

        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"kv_store_{self.namespace}.json")
        self._data = None
        self._storage_lock = None
        self.storage_updated = None
        self.init_lock_stats()

    async def initialize(self):
        """Initialize storage data"""
        self._storage_lock = get_storage_lock()
        self.storage_updated = await get_update_flag(self.final_namespace)
        async with get_data_init_lock():
            # check need_init must before get_namespace_data
            need_init = await try_initialize_namespace(self.final_namespace)
            self._data = await get_namespace_data(self.final_namespace)
            if need_init:
                # Use shared lock to read initial data safely
                lock_file = self._file_name + ".lock"
                with open(lock_file, "w+") as fd:
                    async with self._file_lock(fd, fcntl.LOCK_SH, lock_file):
                        loaded_data = load_json(self._file_name) or {}

                async with self._storage_lock:
                    self._data.update(loaded_data)
                    logger.info(
                        f"[{self.workspace}] Process {os.getpid()} doc status load {self.namespace} with {len(loaded_data)} records"
                    )

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return keys that should be processed (not in storage or not successfully processed)"""
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            return set(keys) - set(self._data.keys())

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            return await self._get_by_ids_no_lock(ids)

    async def _get_by_ids_no_lock(self, ids: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for id in ids:
            data = self._data.get(id, None)
            if data:
                result.append(data)
        return result

    async def get_status_counts(self) -> dict[str, int]:
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            return await self._get_status_counts_no_lock()

    async def _get_status_counts_no_lock(self) -> dict[str, int]:
        counts = {status.value: 0 for status in DocStatus}
        for doc in self._data.values():
            counts[doc["status"]] += 1
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        async with self._storage_lock:
            return await self._get_docs_by_status_no_lock(status)

    async def _get_docs_by_status_no_lock(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        result = {}
        for k, v in self._data.items():
            if v["status"] == status.value:
                try:
                    data = v.copy()
                    data.pop("content", None)
                    data.pop("multimodal_processed", None)
                    if "file_path" not in data:
                        data["file_path"] = "no-file-path"
                    if "metadata" not in data:
                        data["metadata"] = {}
                    if "error_msg" not in data:
                        data["error_msg"] = None
                    result[k] = DocProcessingStatus(**data)
                except KeyError as e:
                    logger.error(
                        f"[{self.workspace}] Missing required field for document {k}: {e}"
                    )
                    continue
        return result

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        async with self._storage_lock:
            return await self._get_docs_by_track_id_no_lock(track_id)

    async def _get_docs_by_track_id_no_lock(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        result = {}
        for k, v in self._data.items():
            if v.get("track_id") == track_id:
                try:
                    data = v.copy()
                    data.pop("content", None)
                    if "file_path" not in data:
                        data["file_path"] = "no-file-path"
                    if "metadata" not in data:
                        data["metadata"] = {}
                    if "error_msg" not in data:
                        data["error_msg"] = None
                    result[k] = DocProcessingStatus(**data)
                except KeyError as e:
                    logger.error(
                        f"[{self.workspace}] Missing required field for document {k}: {e}"
                    )
                    continue
        return result

    async def finalize(self):
        """Finalize storage resources"""
        # Reset init flag
        await reset_namespace_initialization(self.final_namespace)

    @contextlib.asynccontextmanager
    async def transaction(self):
        """
        Context manager for atomic read-modify-write operations.
        Acquires an exclusive lock on the storage file, ensuring isolation.
        """
        start_time = time.time()
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        lock_file = self._file_name + ".lock"

        # Acquire asyncio lock to coordinate with local tasks
        async with self._storage_lock:
            # Acquire file lock to coordinate with other processes
            with open(lock_file, "w+") as fd:
                async with self._file_lock(fd, fcntl.LOCK_EX, lock_file):
                    # 1. Reload from disk
                    disk_data = load_json(self._file_name) or {}

                    # 2. Merge logic using updated_at timestamp (Same as index_done_callback)
                    for k, v in self._data.items():
                        if k in disk_data:
                            disk_v = disk_data[k]
                            # Compare updated_at
                            disk_time = disk_v.get("updated_at", "")
                            my_time = v.get("updated_at", "")
                            # Simple string comparison for ISO format works
                            if my_time >= disk_time:
                                disk_data[k] = v
                        else:
                            disk_data[k] = v

                    # Update memory
                    self._data = disk_data

                    yield self._TransactionWrapper(self)

                    # 3. Persist changes
                    write_json(self._data, self._file_name)
                    await clear_all_update_flags(self.final_namespace)

        # Record transaction stats
        end_time = time.time()
        self._update_tx_stats(end_time - start_time, self._file_name)

    class _TransactionWrapper:
        def __init__(self, storage):
            self._storage = storage

        def __getattr__(self, name):
            return getattr(self._storage, name)

        async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
            return await self._storage._get_by_id_no_lock(id)

        async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
            return await self._storage._get_by_ids_no_lock(ids)

        async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
            return await self._storage._upsert_no_lock(data)

        async def get_status_counts(self) -> dict[str, int]:
            return await self._storage._get_status_counts_no_lock()

        async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
            return await self._storage._get_docs_by_status_no_lock(status)

        async def get_docs_by_track_id(self, track_id: str) -> dict[str, DocProcessingStatus]:
            return await self._storage._get_docs_by_track_id_no_lock(track_id)

        async def get_docs_paginated(self, *args, **kwargs):
            return await self._storage._get_docs_paginated_no_lock(*args, **kwargs)

        async def get_doc_by_file_path(self, file_path: str):
            return await self._storage._get_doc_by_file_path_no_lock(file_path)

        async def delete(self, ids: list[str]) -> None:
            return await self._storage._delete_no_lock(ids)

    async def index_done_callback(self) -> None:
        lock_file = self._file_name + ".lock"
        
        async with self._storage_lock:
            if self.storage_updated.value:
                # Acquire file lock to prevent race conditions
                with open(lock_file, "w+") as fd:
                    async with self._file_lock(fd, fcntl.LOCK_EX, lock_file):
                        # 1. Reload from disk
                        disk_data = load_json(self._file_name) or {}
                        
                        # 2. Merge logic using updated_at timestamp
                        for k, v in self._data.items():
                            if k in disk_data:
                                disk_v = disk_data[k]
                                # Compare updated_at
                                disk_time = disk_v.get("updated_at", "")
                                my_time = v.get("updated_at", "")
                                # Simple string comparison for ISO format works
                                if my_time >= disk_time:
                                    disk_data[k] = v
                            else:
                                disk_data[k] = v
                        
                        # 3. Write merged data
                        data_count = len(disk_data)
                        logger.debug(
                            f"[{self.workspace}] Process {os.getpid()} doc status writing {data_count} records to {self.namespace}"
                        )
                        write_json(disk_data, self._file_name)
                        
                        # 4. Update memory
                        self._data = disk_data
                        
                        await clear_all_update_flags(self.final_namespace)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            await self._upsert_no_lock(data)
        
        await self.index_done_callback()

    async def _upsert_no_lock(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        logger.debug(
            f"[{self.workspace}] Inserting {len(data)} records to {self.namespace}"
        )
        # Ensure chunks_list field exists for new documents
        for doc_id, doc_data in data.items():
            if "chunks_list" not in doc_data:
                doc_data["chunks_list"] = []
        self._data.update(data)
        await set_all_update_flags(self.final_namespace)

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        async with self._storage_lock:
            return self._data.get(id)

    async def _get_by_id_no_lock(self, id: str) -> Union[dict[str, Any], None]:
        return self._data.get(id)

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        async with self._storage_lock:
            return await self._get_docs_paginated_no_lock(
                status_filter, page, page_size, sort_field, sort_direction
            )

    async def _get_docs_paginated_no_lock(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        # Validate parameters
        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200

        if sort_field not in ["created_at", "updated_at", "id", "file_path"]:
            sort_field = "updated_at"

        if sort_direction.lower() not in ["asc", "desc"]:
            sort_direction = "desc"

        # For JSON storage, we load all data and sort/filter in memory
        all_docs = []

        for doc_id, doc_data in self._data.items():
            # Apply status filter
            if (
                status_filter is not None
                and doc_data.get("status") != status_filter.value
            ):
                continue

            try:
                # Prepare document data
                data = doc_data.copy()
                data.pop("content", None)
                if "file_path" not in data:
                    data["file_path"] = "no-file-path"
                if "metadata" not in data:
                    data["metadata"] = {}
                if "error_msg" not in data:
                    data["error_msg"] = None

                doc_status = DocProcessingStatus(**data)

                # Add sort key for sorting
                if sort_field == "id":
                    doc_status._sort_key = doc_id
                elif sort_field == "file_path":
                    # Use pinyin sorting for file_path field to support Chinese characters
                    file_path_value = getattr(doc_status, sort_field, "")
                    doc_status._sort_key = get_pinyin_sort_key(file_path_value)
                else:
                    doc_status._sort_key = getattr(doc_status, sort_field, "")

                all_docs.append((doc_id, doc_status))

            except KeyError as e:
                logger.error(
                    f"[{self.workspace}] Error processing document {doc_id}: {e}"
                )
                continue

        # Sort documents
        reverse_sort = sort_direction.lower() == "desc"
        all_docs.sort(
            key=lambda x: getattr(x[1], "_sort_key", ""), reverse=reverse_sort
        )

        # Remove sort key from documents
        for doc_id, doc in all_docs:
            if hasattr(doc, "_sort_key"):
                delattr(doc, "_sort_key")

        total_count = len(all_docs)

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_docs = all_docs[start_idx:end_idx]

        return paginated_docs, total_count

    async def get_all_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status for all documents

        Returns:
            Dictionary mapping status names to counts, including 'all' field
        """
        counts = await self.get_status_counts()

        # Add 'all' field with total count
        total_count = sum(counts.values())
        counts["all"] = total_count

        return counts

    async def delete(self, doc_ids: list[str]) -> None:
        async with self._storage_lock:
            await self._delete_no_lock(doc_ids)

    async def _delete_no_lock(self, doc_ids: list[str]) -> None:
        any_deleted = False
        for doc_id in doc_ids:
            result = self._data.pop(doc_id, None)
            if result is not None:
                any_deleted = True

        if any_deleted:
            await set_all_update_flags(self.final_namespace)

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")
        async with self._storage_lock:
            return await self._get_doc_by_file_path_no_lock(file_path)

    async def _get_doc_by_file_path_no_lock(self, file_path: str) -> Union[dict[str, Any], None]:
        for doc_id, doc_data in self._data.items():
            if doc_data.get("file_path") == file_path:
                return doc_data
        return None

    async def drop(self) -> dict[str, str]:
        try:
            async with self._storage_lock:
                await self._drop_no_lock()

            await self.index_done_callback()
            logger.info(
                f"[{self.workspace}] Process {os.getpid()} drop {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}

    async def _drop_no_lock(self) -> None:
        self._data.clear()
        await set_all_update_flags(self.final_namespace)
