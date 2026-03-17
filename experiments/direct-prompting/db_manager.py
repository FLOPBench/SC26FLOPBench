import psycopg
import subprocess
import time
import re
import os
from psycopg.errors import DuplicateDatabase, UndefinedTable
import json
from typing import Dict, Any, List


def _discover_postgres_cluster(port: int = 5432) -> dict | None:
    result = subprocess.run(
        ["pg_lsclusters", "--no-header"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to inspect PostgreSQL clusters: {result.stderr.strip()}")

    clusters = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        clusters.append(
            {
                "version": parts[0],
                "name": parts[1],
                "port": int(parts[2]),
                "status": parts[3],
            }
        )

    if not clusters:
        return None

    for cluster in clusters:
        if cluster["port"] == port:
            return cluster
    return clusters[0]


def _can_connect_to_postgres(host: str, port: int, user: str, password: str, db_name: str = "postgres") -> bool:
    try:
        with psycopg.connect(
            f"postgresql://{user}:{password}@{host}:{port}/{db_name}",
            connect_timeout=2,
        ):
            return True
    except psycopg.OperationalError:
        return False


def ensure_postgres_running(
    host: str = "localhost",
    port: int = 5432,
    user: str = "postgres",
    password: str = "postgres",
) -> None:
    cluster = _discover_postgres_cluster(port)
    if cluster is None:
        raise RuntimeError("No local PostgreSQL cluster was found. Install or initialize PostgreSQL first.")

    status = cluster["status"]
    if status == "online" and _can_connect_to_postgres(host, port, user, password):
        return

    action = "start" if status != "online" else "restart"
    print(
        f"PostgreSQL cluster {cluster['version']}/{cluster['name']} on port {cluster['port']} "
        f"is {status}; attempting to {action} it..."
    )
    result = subprocess.run(
        ["pg_ctlcluster", cluster["version"], cluster["name"], action],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"pg_ctlcluster {action} failed")

    deadline = time.time() + 20
    while time.time() < deadline:
        if _can_connect_to_postgres(host, port, user, password):
            print("PostgreSQL is ready.")
            return
        time.sleep(1)

    raise RuntimeError(f"PostgreSQL did not become ready on {host}:{port} after {action}.")

def setup_default_database(
    db_name: str = "gpuflops_db",
    user: str = "postgres",
    password: str = "postgres",
    host: str = "localhost",
    port: int = 5432
) -> str:
    """
    Attempts to connect to the default postgres server and create the target database if it doesn't exist.
    Returns the constructed DB_URI for the target database.
    """
    base_uri = f"postgresql://{user}:{password}@{host}:{port}/postgres"
    target_uri = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"

    try:
        # Connect to default 'postgres' database to create the new one
        with psycopg.connect(base_uri, autocommit=True) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(f"CREATE DATABASE {db_name};")
                    print(f"Created default database '{db_name}'.")
                except DuplicateDatabase:
                    # Database already exists, this is fine
                    pass
    except Exception as e:
        print(f"Note: Could not automatically setup database (ensure PostgreSQL is running at {host}:{port}): {e}")

    return target_uri


def _postgres_command_env(password: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    return env


def wipe_database(
    db_name: str = "gpuflops_db",
    user: str = "postgres",
    password: str = "postgres",
    host: str = "localhost",
    port: int = 5432,
) -> None:
    base_uri = f"postgresql://{user}:{password}@{host}:{port}/postgres"
    with psycopg.connect(base_uri, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
                """,
                (db_name,),
            )
            cur.execute(f"DROP DATABASE IF EXISTS {db_name};")


def dump_database(
    dump_file_path: str,
    db_name: str = "gpuflops_db",
    user: str = "postgres",
    password: str = "postgres",
    host: str = "localhost",
    port: int = 5432,
) -> str:
    dump_file_abspath = os.path.abspath(dump_file_path)
    os.makedirs(os.path.dirname(dump_file_abspath), exist_ok=True)

    result = subprocess.run(
        [
            "pg_dump",
            "-h",
            host,
            "-p",
            str(port),
            "-U",
            user,
            "-d",
            db_name,
            "-Fc",
            "-f",
            dump_file_abspath,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_postgres_command_env(password),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "pg_dump failed")

    return dump_file_abspath


def restore_database_from_dump(
    dump_file_path: str,
    db_name: str = "gpuflops_db",
    user: str = "postgres",
    password: str = "postgres",
    host: str = "localhost",
    port: int = 5432,
) -> str:
    dump_file_abspath = os.path.abspath(dump_file_path)
    if not os.path.exists(dump_file_abspath):
        raise FileNotFoundError(f"Database dump file not found: {dump_file_abspath}")

    wipe_database(db_name=db_name, user=user, password=password, host=host, port=port)
    target_uri = setup_default_database(db_name=db_name, user=user, password=password, host=host, port=port)

    result = subprocess.run(
        [
            "pg_restore",
            "-h",
            host,
            "-p",
            str(port),
            "-U",
            user,
            "-d",
            db_name,
            "--no-owner",
            "--no-privileges",
            dump_file_abspath,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_postgres_command_env(password),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "pg_restore failed")

    return target_uri

class CheckpointDBParser:
    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self.conn = psycopg.connect(self.db_uri)

    def fetch_all_checkpoints(self) -> List[Dict[str, Any]]:
        query = (
            "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint "
            "FROM checkpoints"
        )
        checkpoints = []
        with self.conn.cursor() as cur:
            try:
                cur.execute(query)
            except UndefinedTable:
                # A fresh database will not have LangGraph checkpoint tables yet.
                self.conn.rollback()
                return []
            for row in cur.fetchall():
                thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint_data = row
                
                # Checkpointer stores data, we parse it
                if isinstance(checkpoint_data, (bytes, bytearray, memoryview)):
                    checkpoint_dict = json.loads(checkpoint_data.decode('utf-8'))
                else:
                    checkpoint_dict = checkpoint_data
                    
                checkpoints.append({
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "parent_checkpoint_id": parent_checkpoint_id,
                    "checkpoint": checkpoint_dict
                })
        return checkpoints

    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculates summary statistics across all valid graph executions.
        Extracts tokens, execution time, and estimated cost.
        """
        checkpoints = self.fetch_all_checkpoints()
        
        total_runs = 0
        total_tokens = 0
        total_cost = 0.0
        total_time = 0.0
        
        for cp in checkpoints:
            state = cp["checkpoint"]
            
            if "channel_values" not in state:
                continue
                
            channel_values = state["channel_values"]
            
            # Identify ending state where validator properties exist
            if "query_time" in channel_values and "total_tokens" in channel_values:
                total_runs += 1
                total_tokens += channel_values.get("total_tokens", 0)
                total_cost += channel_values.get("cost_usd", 0.0)
                total_time += channel_values.get("query_time", 0.0)
                
        if total_runs == 0:
            return {"status": "No completed runs found"}

        return {
            "total_runs": total_runs,
            "total_tokens_used": total_tokens,
            "total_cost_usd": total_cost,
            "total_execution_time_sec": total_time,
            "avg_tokens_per_run": total_tokens / total_runs,
            "avg_cost_per_run": total_cost / total_runs,
            "avg_time_per_run": total_time / total_runs
        }

    def calculate_database_run_statistics(self, trials_per_run: int, thread_ids: List[str] | None = None) -> Dict[str, Any]:
        checkpoints = self.fetch_all_checkpoints()
        if thread_ids is not None:
            thread_id_filter = set(thread_ids)
            checkpoints = [cp for cp in checkpoints if cp["thread_id"] in thread_id_filter]

        total_checkpoint_entries = len(checkpoints)

        completed_threads = set()
        for cp in checkpoints:
            state = cp["checkpoint"]
            channel_values = state["channel_values"]
            if "total_tokens" in channel_values:
                completed_threads.add(cp["thread_id"])

        runs_with_all_trials_completed = 0
        if trials_per_run > 0:
            completed_trials_by_run: Dict[str, set[int]] = {}
            for thread_id in completed_threads:
                match = re.search(r"_trial(\d+)(?:_DRYRUN\d+)?$", thread_id)
                if not match:
                    continue

                trial_index = int(match.group(1))
                base_thread_id = thread_id[:match.start()]
                completed_trials_by_run.setdefault(base_thread_id, set()).add(trial_index)

            runs_with_all_trials_completed = sum(
                1
                for completed_trial_indices in completed_trials_by_run.values()
                if all(trial_index in completed_trial_indices for trial_index in range(trials_per_run))
            )

        return {
            "total_checkpoint_entries": total_checkpoint_entries,
            "completed_threads": len(completed_threads),
            "runs_with_all_trials_completed": runs_with_all_trials_completed,
        }

    def close(self):
        self.conn.close()


class QueryAttemptTracker:
    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self.conn = psycopg.connect(self.db_uri, autocommit=True)
        self.setup()

    def setup(self):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS query_attempts (
                    thread_id TEXT PRIMARY KEY,
                    total_attempts INTEGER NOT NULL DEFAULT 0,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    last_status TEXT NOT NULL DEFAULT 'never-run',
                    last_error TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    def fetch_attempts(self, thread_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not thread_ids:
            return {}

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT thread_id, total_attempts, failed_attempts, last_status, last_error, updated_at
                FROM query_attempts
                WHERE thread_id = ANY(%s)
                """,
                (thread_ids,),
            )
            rows = cur.fetchall()

        return {
            thread_id: {
                "total_attempts": total_attempts,
                "failed_attempts": failed_attempts,
                "last_status": last_status,
                "last_error": last_error,
                "updated_at": updated_at,
            }
            for thread_id, total_attempts, failed_attempts, last_status, last_error, updated_at in rows
        }

    def fetch_all_attempts(self) -> Dict[str, Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT thread_id, total_attempts, failed_attempts, last_status, last_error, updated_at
                FROM query_attempts
                """
            )
            rows = cur.fetchall()

        return {
            thread_id: {
                "total_attempts": total_attempts,
                "failed_attempts": failed_attempts,
                "last_status": last_status,
                "last_error": last_error,
                "updated_at": updated_at,
            }
            for thread_id, total_attempts, failed_attempts, last_status, last_error, updated_at in rows
        }

    def mark_attempt_started(self, thread_id: str):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_attempts (thread_id, total_attempts, failed_attempts, last_status, last_error)
                VALUES (%s, 1, 0, 'running', NULL)
                ON CONFLICT (thread_id) DO UPDATE
                SET total_attempts = query_attempts.total_attempts + 1,
                    last_status = 'running',
                    last_error = NULL,
                    updated_at = NOW()
                """,
                (thread_id,),
            )

    def mark_attempt_success(self, thread_id: str):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_attempts (thread_id, total_attempts, failed_attempts, last_status, last_error)
                VALUES (%s, 1, 0, 'completed', NULL)
                ON CONFLICT (thread_id) DO UPDATE
                SET last_status = 'completed',
                    last_error = NULL,
                    updated_at = NOW()
                """,
                (thread_id,),
            )

    def mark_attempt_failure(self, thread_id: str, error_message: str):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_attempts (thread_id, total_attempts, failed_attempts, last_status, last_error)
                VALUES (%s, 1, 1, 'failed', %s)
                ON CONFLICT (thread_id) DO UPDATE
                SET failed_attempts = query_attempts.failed_attempts + 1,
                    last_status = 'failed',
                    last_error = EXCLUDED.last_error,
                    updated_at = NOW()
                """,
                (thread_id, error_message[:4000]),
            )

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    ensure_postgres_running()
    db_uri = setup_default_database()
    parser = CheckpointDBParser(db_uri)
    try:
        stats = parser.calculate_summary_statistics()
        print(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"Error accessing database: {e}")
    finally:
        parser.close()
