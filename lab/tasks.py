from .observation_rules import identity
from .cohorts import select_current_rows
from .task_rules import normalize_selection, validate_task_binding
from .snapshots import build_snapshot, decode_snapshot
from .source_links import bind_source_links, verify_source_links
from .codecs import encode_payload, decode_payload, clone_payload, IntegrityError

class TaskService:
    def __init__(self,storage):
        self.storage = storage

    def freeze(self,task_id,selection):
        identity(task_id)
        selection = normalize_selection(selection)
        with self.storage.transaction():
            rows = select_current_rows(self.storage,selection)
            candidate = build_snapshot(task_id,selection,rows)
            saved = self.storage.fetch_one("SELECT payload FROM tasks WHERE task_id=?",(task_id,))
            if saved:
                old = decode_snapshot(decode_payload(saved[0]))
                validate_task_binding(old,candidate)
                verify_source_links(self.storage,old)
                return old
            self.storage.execute("INSERT INTO tasks VALUES(?,?)",(task_id,encode_payload(candidate)))
            self.storage.checkpoint("task.snapshot")
            bind_source_links(self.storage,candidate)
            self.storage.checkpoint("task.links")
        return clone_payload(candidate)

    def read_frozen(self,task_id):
        saved = self.storage.fetch_one("SELECT payload FROM tasks WHERE task_id=?",(task_id,))
        if saved is None:
            raise KeyError("unknown task: "+task_id)
        task = decode_snapshot(decode_payload(saved[0]))
        if task["task_id"] != task_id:
            raise IntegrityError("task identity mismatch")
        verify_source_links(self.storage,task)
        return task
