SCHEMA_VERSION = 1

_TABLES = {
    "meta":("version",),
    "revisions":("cohort_id","sample_id","rev","payload"),
    "current":("cohort_id","sample_id","rev"),
    "batches":("batch_id","payload","result"),
    "tasks":("task_id","payload"),
    "source_links":("task_id","ordinal","cohort_id","sample_id","rev","row_digest"),
    "results":("result_id","kind","payload"),
    "result_index":("request_key","result_id"),
    "parents":("result_id","parent_kind","parent_id"),
}

def initialize_schema(connection):
    connection.execute("CREATE TABLE IF NOT EXISTS meta(version INTEGER NOT NULL)")
    versions = connection.execute("SELECT version FROM meta").fetchall()
    if versions and versions != [(SCHEMA_VERSION,)]:
        raise ValueError("unsupported database schema version")
    if not versions:
        connection.execute("INSERT INTO meta VALUES(?)",(SCHEMA_VERSION,))
    statements = [
        "CREATE TABLE IF NOT EXISTS revisions(cohort_id TEXT,sample_id TEXT,rev INTEGER,payload TEXT NOT NULL,PRIMARY KEY(cohort_id,sample_id,rev))",
        "CREATE TABLE IF NOT EXISTS current(cohort_id TEXT,sample_id TEXT,rev INTEGER,PRIMARY KEY(cohort_id,sample_id))",
        "CREATE TABLE IF NOT EXISTS batches(batch_id TEXT PRIMARY KEY,payload TEXT NOT NULL,result TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS tasks(task_id TEXT PRIMARY KEY,payload TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS source_links(task_id TEXT,ordinal INTEGER,cohort_id TEXT,sample_id TEXT,rev INTEGER,row_digest TEXT,PRIMARY KEY(task_id,ordinal))",
        "CREATE TABLE IF NOT EXISTS results(result_id TEXT PRIMARY KEY,kind TEXT NOT NULL,payload TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS result_index(request_key TEXT PRIMARY KEY,result_id TEXT UNIQUE NOT NULL)",
        "CREATE TABLE IF NOT EXISTS parents(result_id TEXT PRIMARY KEY,parent_kind TEXT NOT NULL,parent_id TEXT NOT NULL)",
    ]
    for statement in statements:
        connection.execute(statement)
    verify_schema(connection)

def verify_schema(connection):
    versions = connection.execute("SELECT version FROM meta").fetchall()
    if versions != [(SCHEMA_VERSION,)]:
        raise ValueError("schema version mismatch")
    for table,expected in _TABLES.items():
        actual = tuple(row[1] for row in connection.execute("PRAGMA table_info("+table+")"))
        if actual != expected:
            raise ValueError("database structure mismatch: "+table)
