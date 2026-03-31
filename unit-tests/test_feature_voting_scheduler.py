import builtins
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class _StubLoader:
    def __init__(self, module_attrs):
        self._module_attrs = module_attrs

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        for name, value in self._module_attrs.items():
            setattr(module, name, value)


class _StubConnectionPool:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


class _StubPostgresSaver:
    def __init__(self, *args, **kwargs):
        pass

    def setup(self):
        pass


class _NoOpTqdm:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, amount):
        pass


def _install_import_stubs(monkeypatch):
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = _NoOpTqdm
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)

    psycopg_pool_module = types.ModuleType("psycopg_pool")
    psycopg_pool_module.ConnectionPool = _StubConnectionPool
    monkeypatch.setitem(sys.modules, "psycopg_pool", psycopg_pool_module)

    langgraph_module = types.ModuleType("langgraph")
    checkpoint_module = types.ModuleType("langgraph.checkpoint")
    postgres_module = types.ModuleType("langgraph.checkpoint.postgres")
    postgres_module.PostgresSaver = _StubPostgresSaver
    checkpoint_module.postgres = postgres_module
    langgraph_module.checkpoint = checkpoint_module
    monkeypatch.setitem(sys.modules, "langgraph", langgraph_module)
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint", checkpoint_module)
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", postgres_module)


def _load_feature_voting_module(monkeypatch):
    _install_import_stubs(monkeypatch)

    target_path = REPO_ROOT / "experiments" / "feature-voting" / "run_voting_queries.py"
    original_spec_from_file_location = importlib.util.spec_from_file_location

    def fake_spec_from_file_location(name, location, *args, **kwargs):
        if str(location).endswith("gpuFLOPBench-agentic/agents/llm_models.py"):
            module_attrs = {
                "build_openrouter_llm": lambda settings: settings,
                "OpenRouterLLMSettings": type(
                    "OpenRouterLLMSettings",
                    (),
                    {"__init__": lambda self, model_name: setattr(self, "model_name", model_name)},
                ),
            }
            return importlib.util.spec_from_loader(name, _StubLoader(module_attrs))

        if str(location).endswith("experiments/feature-voting/graph.py"):
            module_attrs = {
                "build_graph": lambda: None,
            }
            return importlib.util.spec_from_loader(name, _StubLoader(module_attrs))

        if str(location).endswith("experiments/feature-voting/db_manager.py"):
            module_attrs = {
                "CheckpointDBParser": object,
                "QueryAttemptTracker": object,
                "setup_default_database": lambda: "postgresql://unused",
                "ensure_postgres_running": lambda: None,
                "wipe_database": lambda: None,
                "restore_database_from_dump": lambda path: "postgresql://unused",
                "dump_database": lambda path: path,
                "delete_thread_history": lambda db_uri, thread_ids: None,
            }
            return importlib.util.spec_from_loader(name, _StubLoader(module_attrs))

        return original_spec_from_file_location(name, location, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", fake_spec_from_file_location)

    spec = original_spec_from_file_location("feature_voting_run_queries", target_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeParser:
    def __init__(self, db_uri):
        self.db_uri = db_uri

    def fetch_all_checkpoints(self):
        return []

    def calculate_database_run_statistics(self, trials, thread_ids):
        return {
            "total_checkpoint_entries": 0,
            "completed_threads": 0,
            "runs_with_all_trials_completed": 0,
        }

    def close(self):
        pass


class _FakeAttemptTracker:
    def __init__(self, db_uri):
        self.db_uri = db_uri
        self.failures = []

    def fetch_attempts(self, thread_ids):
        return {}

    def mark_attempt_failure(self, thread_id, error):
        self.failures.append((thread_id, error))

    def close(self):
        pass


class _FakeFuture:
    def __init__(self, sequence, query, executor):
        self.sequence = sequence
        self.query = query
        self.executor = executor

    def __hash__(self):
        return hash(self.sequence)

    def result(self):
        return {
            "thread_id": self.query["thread_id"],
            "status": "completed",
            "final_state": {
                "program_name": self.query["state"]["program_name"],
                "kernel_mangled_name": self.query["state"]["kernel_mangled_name"],
                "kernel_demangled_name": self.query["state"]["kernel_demangled_name"],
                "prediction": {},
                "total_tokens": 1,
            },
            "cost_usd": 0.0,
            "error": None,
        }


class _FakeExecutor:
    last_instance = None

    def __init__(self, max_workers, mp_context):
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.events = []
        self.active_futures = []
        self.max_inflight = 0
        _FakeExecutor.last_instance = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, db_uri, query, print_prompts, max_timeout):
        future = _FakeFuture(len(self.events), query, self)
        self.events.append(f"submit:{query['thread_id']}")
        self.active_futures.append(future)
        self.max_inflight = max(self.max_inflight, len(self.active_futures))
        return future


def test_run_queries_refills_open_slots_without_waiting_for_full_batch(monkeypatch):
    module = _load_feature_voting_module(monkeypatch)

    dataset = {
        "prog-cuda": {
            "sources": ["demo.cu"],
            "exeArgs": "",
            "kernels": {
                f"kernel{i}": {"demangledName": f"kernel{i}()"}
                for i in range(5)
            },
        }
    }

    monkeypatch.setattr(module, "load_dataset", lambda path: dataset)
    monkeypatch.setattr(module, "_ensure_checkpoint_schema", lambda db_uri: None)
    monkeypatch.setattr(module, "CheckpointDBParser", _FakeParser)
    monkeypatch.setattr(module, "QueryAttemptTracker", _FakeAttemptTracker)
    monkeypatch.setattr(module, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(module.multiprocessing, "get_context", lambda method: object())
    monkeypatch.setattr(module, "_maybe_print_kernel_consensus", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "print_run_result", lambda state: None)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "")

    def fake_wait(futures, return_when):
        next_future = min(futures, key=lambda future: future.sequence)
        next_future.executor.events.append(f"complete:{next_future.query['thread_id']}")
        next_future.executor.active_futures.remove(next_future)
        return {next_future}, set(futures) - {next_future}

    monkeypatch.setattr(module, "wait", fake_wait)

    module.run_queries(
        "postgresql://unused",
        "/tmp/dataset.json",
        ["provider/model"],
        trials=1,
        single_dry_run=False,
        verbose=False,
        print_prompts=False,
        max_timeout=30,
        max_queries=None,
        cli_config=None,
        max_failed_attempts=3,
        skip_completed_check=False,
        max_spend=None,
        query_batch_size=2,
    )

    executor = _FakeExecutor.last_instance
    assert executor is not None
    assert executor.max_inflight == 2

    submit_events = [event for event in executor.events if event.startswith("submit:")]
    completion_events = [event for event in executor.events if event.startswith("complete:")]

    assert len(submit_events) == 5
    assert len(completion_events) == 5
    assert executor.events[0:4] == [
        submit_events[0],
        submit_events[1],
        completion_events[0],
        submit_events[2],
    ]