import os
import time
import fcntl
import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any, final

from lightrag.base import (
    BaseKVStorage,
)
from .json_common import JsonStorageLockMixin
from lightrag.utils import (
    load_json,
    logger,
    write_json,
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
class JsonKVStorage(BaseKVStorage, JsonStorageLockMixin):
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
            workspace_dir = working_dir
            self.final_namespace = f"{working_dir_hash}_{self.namespace}"
            self.workspace = "_"

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
                    # Migrate legacy cache structure if needed
                    if self.namespace.endswith("_cache"):
                        loaded_data = await self._migrate_legacy_cache_structure(
                            loaded_data
                        )

                    self._data.update(loaded_data)
                    data_count = len(loaded_data)

                    logger.info(
                        f"[{self.workspace}] Process {os.getpid()} KV load {self.namespace} with {data_count} records"
                    )
            else:
                # If already initialized, check if we need to sync updates from disk
                # This handles the case where other processes updated the storage
                await self.index_done_callback()

    async def index_done_callback(self) -> None:
        lock_file = self._file_name + ".lock"
        
        async with self._storage_lock:
            if not self.storage_updated.value:
                return

            # Acquire file lock to prevent race conditions
            with open(lock_file, "w+") as fd:
                async with self._file_lock(fd, fcntl.LOCK_EX, lock_file):
                    # 1. Reload from disk to get latest state from other processes
                    disk_data = load_json(self._file_name) or {}
                    
                    # 2. Merge logic using timestamps
                    for k, v in self._data.items():
                        if k in disk_data:
                            disk_v = disk_data[k]
                            # Handle dictionary data with timestamps
                            if isinstance(disk_v, dict) and isinstance(v, dict):
                                disk_time = disk_v.get("update_time", 0)
                                my_time = v.get("update_time", 0)
                                if my_time >= disk_time:
                                    disk_data[k] = v
                            else:
                                # Fallback for non-dict data: overwrite
                                disk_data[k] = v
                        else:
                            # New key
                            disk_data[k] = v
                    
                    # 3. Write merged data back
                    data_count = len(disk_data)
                    logger.debug(
                        f"[{self.workspace}] Process {os.getpid()} KV writing {data_count} records to {self.namespace}"
                    )
                    write_json(disk_data, self._file_name)
                    
                    # 4. Update memory to match merged state
                    self._data = disk_data
                    
                    await clear_all_update_flags(self.final_namespace)

    async def get_all(self) -> dict[str, Any]:
        """Get all data from storage"""
        async with self._storage_lock:
            return await self._get_all_no_lock()

    async def _get_all_no_lock(self) -> dict[str, Any]:
        result = {}
        for key, value in self._data.items():
            if value:
                data = dict(value)
                data.setdefault("create_time", 0)
                data.setdefault("update_time", 0)
                result[key] = data
            else:
                result[key] = value
        return result

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._storage_lock:
            return await self._get_by_id_no_lock(id)

    async def _get_by_id_no_lock(self, id: str) -> dict[str, Any] | None:
        result = self._data.get(id)
        if result:
            result = dict(result)
            result.setdefault("create_time", 0)
            result.setdefault("update_time", 0)
            result["_id"] = id
        return result

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        async with self._storage_lock:
            return await self._get_by_ids_no_lock(ids)

    async def _get_by_ids_no_lock(self, ids: list[str]) -> list[dict[str, Any]]:
        results = []
        for id in ids:
            data = self._data.get(id, None)
            if data:
                result = {k: v for k, v in data.items()}
                result.setdefault("create_time", 0)
                result.setdefault("update_time", 0)
                result["_id"] = id
                results.append(result)
            else:
                results.append(None)
        return results

    async def filter_keys(self, keys: set[str]) -> set[str]:
        async with self._storage_lock:
            return set(keys) - set(self._data.keys())

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonKVStorage")
        async with self._storage_lock:
            await self._upsert_no_lock(data)

    async def _upsert_no_lock(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        import time

        current_time = int(time.time())

        logger.debug(
            f"[{self.workspace}] Inserting {len(data)} records to {self.namespace}"
        )

        # Add timestamps to data based on whether key exists
        for k, v in data.items():
            # For text_chunks namespace, ensure llm_cache_list field exists
            if self.namespace.endswith("text_chunks"):
                if "llm_cache_list" not in v:
                    v["llm_cache_list"] = []

            # Add timestamps based on whether key exists
            if k in self._data:  # Key exists, only update update_time
                v["update_time"] = current_time
            else:  # New key, set both create_time and update_time
                v["create_time"] = current_time
                v["update_time"] = current_time

            v["_id"] = k

        self._data.update(data)
        await set_all_update_flags(self.final_namespace)

    async def delete(self, ids: list[str]) -> None:
        async with self._storage_lock:
            await self._delete_no_lock(ids)

    async def _delete_no_lock(self, ids: list[str]) -> None:
        any_deleted = False
        for doc_id in ids:
            result = self._data.pop(doc_id, None)
            if result is not None:
                any_deleted = True

        if any_deleted:
            await set_all_update_flags(self.final_namespace)

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage and clean up resources
           This action will persistent the data to disk immediately.

        This method will:
        1. Clear all data from memory
        2. Update flags to notify other processes
        3. Trigger index_done_callback to save the empty state

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            async with self._storage_lock:
                self._data.clear()
                await set_all_update_flags(self.final_namespace)

            await self.index_done_callback()
            logger.info(
                f"[{self.workspace}] Process {os.getpid()} drop {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}

    async def _migrate_legacy_cache_structure(self, data: dict) -> dict:
        """Migrate legacy nested cache structure to flattened structure

        Args:
            data: Original data dictionary that may contain legacy structure

        Returns:
            Migrated data dictionary with flattened cache keys
        """
        from lightrag.utils import generate_cache_key

        # Early return if data is empty
        if not data:
            return data

        # Check first entry to see if it's already in new format
        first_key = next(iter(data.keys()))
        if ":" in first_key and len(first_key.split(":")) == 3:
            # Already in flattened format, return as-is
            return data

        migrated_data = {}
        migration_count = 0

        for key, value in data.items():
            # Check if this is a legacy nested cache structure
            if isinstance(value, dict) and all(
                isinstance(v, dict) and "return" in v for v in value.values()
            ):
                # This looks like a legacy cache mode with nested structure
                mode = key
                for cache_hash, cache_entry in value.items():
                    cache_type = cache_entry.get("cache_type", "extract")
                    flattened_key = generate_cache_key(mode, cache_type, cache_hash)
                    migrated_data[flattened_key] = cache_entry
                    migration_count += 1
            else:
                # Keep non-cache data or already flattened cache data as-is
                migrated_data[key] = value

        if migration_count > 0:
            logger.info(
                f"[{self.workspace}] Migrated {migration_count} legacy cache entries to flattened structure"
            )
            # Persist migrated data immediately
            write_json(migrated_data, self._file_name)

        return migrated_data

    async def finalize(self):
        """Finalize storage resources
        Persistence cache data to disk before exiting
        """
        if self.namespace.endswith("_cache"):
            await self.index_done_callback()

        # Reset init flag so next initialize loads from disk
        await reset_namespace_initialization(self.final_namespace)

    @contextlib.asynccontextmanager
    async def transaction(self):
        """
        Context manager for atomic read-modify-write operations.
        Acquires an exclusive lock on the storage file, ensuring isolation.
        Yields a wrapper that allows lock-free access to storage methods to prevent deadlocks.
        """
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonKVStorage")

        lock_file = self._file_name + ".lock"
        start_time = time.time()

        # Acquire asyncio lock to coordinate with local tasks
        async with self._storage_lock:
            # Acquire file lock to coordinate with other processes
            with open(lock_file, "w+") as fd:
                async with self._file_lock(fd, fcntl.LOCK_EX, lock_file):
                    # 1. Reload from disk
                    disk_data = load_json(self._file_name) or {}

                    # 2. Merge logic using timestamps (Same as index_done_callback)
                    for k, v in self._data.items():
                        if k in disk_data:
                            disk_v = disk_data[k]
                            if isinstance(disk_v, dict) and isinstance(v, dict):
                                disk_time = disk_v.get("update_time", 0)
                                my_time = v.get("update_time", 0)
                                if my_time >= disk_time:
                                    disk_data[k] = v
                            else:
                                disk_data[k] = v
                        else:
                            disk_data[k] = v

                    self._data = disk_data

                    # Yield wrapper that bypasses lock
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

        async def get_by_id(self, id: str) -> dict[str, Any] | None:
            return await self._storage._get_by_id_no_lock(id)

        async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
            return await self._storage._get_by_ids_no_lock(ids)

        async def get_all(self) -> dict[str, Any]:
            return await self._storage._get_all_no_lock()

        async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
            return await self._storage._upsert_no_lock(data)

        async def delete(self, ids: list[str]) -> None:
            return await self._storage._delete_no_lock(ids)
