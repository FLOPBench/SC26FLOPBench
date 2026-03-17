import psycopg
import subprocess
import time
from psycopg.errors import DuplicateDatabase
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

class CheckpointDBParser:
    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self.conn = psycopg.connect(self.db_uri)

    def fetch_all_checkpoints(self) -> List[Dict[str, Any]]:
        query = "SELECT thread_id, checkpoint_ns, checkpoint_id, checkpoint FROM checkpoints"
        checkpoints = []
        with self.conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                thread_id, checkpoint_ns, checkpoint_id, checkpoint_data = row
                
                # Checkpointer stores data, we parse it
                if isinstance(checkpoint_data, (bytes, bytearray, memoryview)):
                    checkpoint_dict = json.loads(checkpoint_data.decode('utf-8'))
                else:
                    checkpoint_dict = checkpoint_data
                    
                checkpoints.append({
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
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
