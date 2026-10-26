"""SQLite experiment tasks; each public operation uses a short transaction.

Task keys are global and immutable. Use a new, versioned key when inputs change.
Dependencies are task IDs or {"task_id": ..., "result_id": ...} references.
The latter pins a score (or other consumer) to an exact immutable result.
Payloads contain execution inputs, never gold/labels; evaluators read label refs.
No connection or write transaction is held while a worker performs its work.
"""

from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from threading import Lock, RLock
import time
from typing import Any, Iterable
import uuid


_WRITER_LOCKS_GUARD = Lock()
_WRITER_LOCKS: dict[str, RLock] = {}


def _writer_lock(path):
    key = os.path.normcase(os.path.realpath(path))
    with _WRITER_LOCKS_GUARD:
        return _WRITER_LOCKS.setdefault(key, RLock())


class TaskConflict(ValueError):
    """An existing key has different execution inputs or dependencies."""


class OwnershipError(RuntimeError):
    """The worker no longer owns this running task."""


class DependencyError(ValueError):
    """A dependency or an exact result reference is invalid."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    stage TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending','running','succeeded','failed')),
    owner TEXT,
    attempt_id TEXT REFERENCES attempts(id),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    CHECK ((status = 'running') = (owner IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id),
    owner TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('running','succeeded','failed','interrupted')),
    started_at REAL NOT NULL,
    finished_at REAL,
    response TEXT,
    usage TEXT,
    validation TEXT,
    error TEXT,
    uncertain INTEGER NOT NULL DEFAULT 0 CHECK (uncertain IN (0,1))
);
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL UNIQUE REFERENCES tasks(id),
    attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(id),
    result TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS task_dependencies (
    task_id TEXT NOT NULL REFERENCES tasks(id),
    dependency_id TEXT NOT NULL REFERENCES tasks(id),
    requested_result_id TEXT REFERENCES results(id),
    bound_result_id TEXT REFERENCES results(id),
    PRIMARY KEY (task_id, dependency_id),
    CHECK (task_id != dependency_id)
);
CREATE TABLE IF NOT EXISTS result_dependencies (
    result_id TEXT NOT NULL REFERENCES results(id),
    dependency_id TEXT NOT NULL REFERENCES tasks(id),
    dependency_result_id TEXT NOT NULL REFERENCES results(id),
    PRIMARY KEY (result_id, dependency_id)
);
CREATE INDEX IF NOT EXISTS tasks_ready ON tasks(status, stage, created_at);
CREATE INDEX IF NOT EXISTS attempts_task ON attempts(task_id, started_at);
CREATE TRIGGER IF NOT EXISTS results_no_update BEFORE UPDATE ON results
BEGIN SELECT RAISE(ABORT, 'results are immutable'); END;
CREATE TRIGGER IF NOT EXISTS results_no_delete BEFORE DELETE ON results
BEGIN SELECT RAISE(ABORT, 'results are immutable'); END;
CREATE TRIGGER IF NOT EXISTS result_dependencies_no_update BEFORE UPDATE ON result_dependencies
BEGIN SELECT RAISE(ABORT, 'result dependencies are immutable'); END;
CREATE TRIGGER IF NOT EXISTS result_dependencies_no_delete BEFORE DELETE ON result_dependencies
BEGIN SELECT RAISE(ABORT, 'result dependencies are immutable'); END;
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


@contextmanager
def coordinator_lock(path: str | Path):
    """Hold an OS lock, including on Windows; stale filenames never imply ownership."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another experiment coordinator owns this artifact root") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _task(store, phase, inputs, *, dependencies=()):
    digest = hashlib.sha256(_json(inputs).encode("utf-8")).hexdigest()
    return store.enqueue(phase + ":" + digest, phase, digest, {"phase": phase, **inputs}, dependencies)


def _save(store, task, value):
    old = store.get_result(task["key"])
    if old:
        return old
    claimed = store.claim(task["stage"], task_id=task["id"])
    if not claimed:
        raise ValueError("Task is interrupted/failed; explicit reconciliation required")
    store.record_attempt(task["id"], claimed["owner"], response=value)
    store.succeed(task["id"], claimed["owner"], value)
    return store.get_result(task["key"])


def _ref(result):
    return {"task_id": result["task_id"], "result_id": result["id"]}


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _check_payload(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in {'gold', 'gold_answer', 'ground_truth', 'label', 'labels'}:
                raise ValueError('task payload must not contain gold or labels; use an evaluator reference')
            _check_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_payload(item)


class TaskStore:
    """File-backed SQLite store, safe for workers using separate connections.

    One execution attempt is created per claim. record_attempt saves its raw
    response/usage/validation before succeed or fail; provider transport retries
    may additionally be recorded in the provider's own attempt store.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = os.fspath(path)
        if self.path == ':memory:' or not self.path:
            raise ValueError('TaskStore requires a persistent database path')
        with self._connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript(_SCHEMA)

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA busy_timeout=5000')
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _transaction(self, *, write: bool = False):
        # Queue this process's writers before opening a SQLite transaction.
        # Many completed provider calls can otherwise exhaust busy_timeout
        # while competing with each other and lose their stage commit.
        with (_writer_lock(self.path) if write else nullcontext()), self._connection() as db:
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def close(self) -> None:
        """No-op: connections close at the end of each operation."""

    def enqueue(self, key: str, stage: str, input_hash: str, payload: Any,
                dependencies: Iterable[str | dict[str, str]] = ()) -> dict:
        """Return an existing identical task or atomically insert a new one.

        Dependency order is not semantic; repeated dependency IDs are rejected.
        Dependencies must already exist. This also prevents dependency cycles.
        A failed task stays failed on enqueue; retry must be explicit via fail.
        """
        _text(key, 'key')
        _text(stage, 'stage')
        _text(input_hash, 'input_hash')
        _check_payload(payload)
        encoded = _json(payload)
        refs = []
        for dependency in dependencies:
            if isinstance(dependency, str):
                refs.append((_text(dependency, 'dependency task ID'), None))
            elif isinstance(dependency, dict) and set(dependency) == {'task_id', 'result_id'}:
                refs.append((_text(dependency['task_id'], 'dependency task ID'),
                             _text(dependency['result_id'], 'dependency result ID')))
            else:
                raise DependencyError('dependencies must be task IDs or task_id/result_id references')
        refs.sort(key=lambda ref: ref[0])
        if len({task_id for task_id, _ in refs}) != len(refs):
            raise DependencyError('duplicate dependency task ID')
        with self._transaction(write=True) as db:
            existing = db.execute('SELECT * FROM tasks WHERE key=?', (key,)).fetchone()
            if existing is not None:
                stored_refs = [(r['dependency_id'], r['requested_result_id']) for r in db.execute(
                    'SELECT dependency_id, requested_result_id FROM task_dependencies '
                    'WHERE task_id=? ORDER BY dependency_id', (existing['id'],))]
                if (existing['stage'], existing['input_hash'], existing['payload'], stored_refs) != (
                        stage, input_hash, encoded, refs):
                    raise TaskConflict(f'key {key!r} already has different inputs; use a versioned key')
                return self._task(db, existing)
            for dependency_id, result_id in refs:
                if db.execute('SELECT 1 FROM tasks WHERE id=?', (dependency_id,)).fetchone() is None:
                    raise DependencyError(f'unknown dependency task {dependency_id}')
                if result_id is not None and db.execute(
                        'SELECT 1 FROM results WHERE id=? AND task_id=?',
                        (result_id, dependency_id)).fetchone() is None:
                    raise DependencyError(f'result {result_id} does not belong to task {dependency_id}')
            task_id, now = str(uuid.uuid4()), time.time()
            db.execute('INSERT INTO tasks(id,key,stage,input_hash,payload,status,created_at,updated_at) '
                       "VALUES (?,?,?,?,?,'pending',?,?)", (task_id, key, stage, input_hash, encoded, now, now))
            db.executemany('INSERT INTO task_dependencies(task_id,dependency_id,requested_result_id) '
                           'VALUES (?,?,?)', [(task_id, dep, result) for dep, result in refs])
            return self._task(db, db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone())

    def claim(self, stage: str | None = None, *, task_id: str | None = None) -> dict | None:
        """Atomically claim a ready task, optionally selecting its exact ID."""
        with self._transaction(write=True) as db:
            predicate = "t.id=?" if task_id is not None else "(? IS NULL OR t.stage=?)"
            parameters = (task_id,) if task_id is not None else (stage, stage)
            row = db.execute("""
                SELECT t.* FROM tasks t
                WHERE t.status='pending' AND """ + predicate + """
                  AND NOT EXISTS (
                    SELECT 1 FROM task_dependencies d
                    JOIN tasks parent ON parent.id=d.dependency_id
                    LEFT JOIN results r ON r.task_id=parent.id
                    WHERE d.task_id=t.id AND (parent.status!='succeeded' OR r.id IS NULL
                        OR (d.requested_result_id IS NOT NULL AND d.requested_result_id!=r.id))
                  )
                ORDER BY t.created_at, t.rowid LIMIT 1
            """, parameters).fetchone()
            if row is None or (stage is not None and row['stage'] != stage):
                return None
            task_id, attempt_id, owner = row['id'], str(uuid.uuid4()), str(uuid.uuid4())
            now = time.time()
            db.execute("INSERT INTO attempts(id,task_id,owner,status,started_at) VALUES (?,?,?,'running',?)",
                       (attempt_id, task_id, owner, now))
            db.execute("UPDATE tasks SET status='running',owner=?,attempt_id=?,updated_at=? WHERE id=?",
                       (owner, attempt_id, now, task_id))
            db.execute('UPDATE task_dependencies SET bound_result_id='
                       '(SELECT id FROM results WHERE task_id=dependency_id) WHERE task_id=?', (task_id,))
            return self._task(db, db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone())

    @staticmethod
    def _owned(db, task_id: str, owner: str):
        row = db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
        if row is None or row['status'] != 'running' or not owner or row['owner'] != owner:
            raise OwnershipError(f'worker does not own running task {task_id}')
        return row

    def record_attempt(self, task_id: str, owner: str, *, response: Any,
                       usage: Any = None, validation: Any = None) -> str:
        """Persist the current attempt's response before validating/committing a result.

        Identical repeat writes are allowed. A different response cannot replace
        one already recorded. Validation/usage can be added after response save.
        """
        response_json = _json(response)
        usage_json = None if usage is None else _json(usage)
        validation_json = None if validation is None else _json(validation)
        with self._transaction(write=True) as db:
            task = self._owned(db, task_id, owner)
            attempt = db.execute('SELECT * FROM attempts WHERE id=?', (task['attempt_id'],)).fetchone()
            for field, value in [('response', response_json), ('usage', usage_json), ('validation', validation_json)]:
                if value is not None and attempt[field] is not None and attempt[field] != value:
                    raise TaskConflict(f'attempt {field} is already recorded')
            db.execute('UPDATE attempts SET response=?,usage=COALESCE(?,usage),'
                       'validation=COALESCE(?,validation) WHERE id=?',
                       (response_json, usage_json, validation_json, task['attempt_id']))
            return task['attempt_id']

    def succeed(self, task_id: str, owner: str, result: Any) -> str:
        """Commit one immutable result and its exact dependency result IDs."""
        encoded = _json(result)
        with self._transaction(write=True) as db:
            task = self._owned(db, task_id, owner)
            refs = db.execute('SELECT d.*,t.status,r.id AS actual_result_id FROM task_dependencies d '
                              'JOIN tasks t ON t.id=d.dependency_id LEFT JOIN results r ON r.task_id=t.id '
                              'WHERE d.task_id=?', (task_id,)).fetchall()
            for ref in refs:
                if (ref['status'] != 'succeeded' or ref['bound_result_id'] is None
                        or ref['bound_result_id'] != ref['actual_result_id']
                        or (ref['requested_result_id'] is not None
                            and ref['requested_result_id'] != ref['bound_result_id'])):
                    raise DependencyError('dependency result changed since claim')
            result_id, now = str(uuid.uuid4()), time.time()
            db.execute('INSERT INTO results(id,task_id,attempt_id,result,created_at) VALUES (?,?,?,?,?)',
                       (result_id, task_id, task['attempt_id'], encoded, now))
            db.executemany('INSERT INTO result_dependencies(result_id,dependency_id,dependency_result_id) '
                           'VALUES (?,?,?)', [(result_id, r['dependency_id'], r['bound_result_id']) for r in refs])
            db.execute("UPDATE attempts SET status='succeeded',finished_at=? WHERE id=?", (now, task['attempt_id']))
            db.execute("UPDATE tasks SET status='succeeded',owner=NULL,updated_at=? WHERE id=?", (now, task_id))
            return result_id

    def fail(self, task_id: str, owner: str, error: Any, retry: bool = False) -> None:
        """Keep the failed attempt; explicit retry immediately returns task to pending.

        Retry budgets/backoff and provider availability belong to the executor.
        """
        encoded = _json(error)
        with self._transaction(write=True) as db:
            task = self._owned(db, task_id, owner)
            now = time.time()
            db.execute("UPDATE attempts SET status='failed',finished_at=?,error=? WHERE id=?",
                       (now, encoded, task['attempt_id']))
            db.execute('UPDATE tasks SET status=?,owner=NULL,updated_at=? WHERE id=?',
                       ('pending' if retry else 'failed', now, task_id))

    def recover_interrupted(self) -> int:
        """Requeue all running tasks and preserve interrupted, uncertain attempts.

        PRECONDITION: the caller holds the single coordinator's exclusive run
        lock and has established that the old coordinator/workers are inactive.
        This store neither acquires that lock nor kills/checks processes. Never
        call this concurrently with a live coordinator.
        """
        with self._transaction(write=True) as db:
            rows = db.execute("SELECT id,attempt_id FROM tasks WHERE status='running'").fetchall()
            now = time.time()
            for row in rows:
                db.execute("UPDATE attempts SET status='interrupted',finished_at=?,uncertain=1,error=? WHERE id=?",
                           (now, _json({'kind': 'interrupted', 'reason': 'coordinator recovery; outcome uncertain'}),
                            row['attempt_id']))
                db.execute("UPDATE tasks SET status='pending',owner=NULL,updated_at=? WHERE id=?", (now, row['id']))
                db.execute('UPDATE task_dependencies SET bound_result_id=NULL WHERE task_id=?', (row['id'],))
            return len(rows)

    @staticmethod
    def _task(db, row) -> dict:
        task = dict(row)
        task['payload'] = json.loads(task['payload'])
        task['dependencies'] = [dict(ref) for ref in db.execute(
            'SELECT d.dependency_id AS task_id,d.requested_result_id,'
            'COALESCE(d.bound_result_id,d.requested_result_id,r.id) AS result_id,t.status '
            'FROM task_dependencies d JOIN tasks t ON t.id=d.dependency_id '
            'LEFT JOIN results r ON r.task_id=t.id WHERE d.task_id=? ORDER BY d.dependency_id', (row['id'],))]
        task['blocked_by'] = []
        task['waiting_for'] = []
        if task['status'] == 'pending':
            task['blocked_by'] = [dict(failed) for failed in db.execute('''
                WITH RECURSIVE ancestors(id) AS (
                    SELECT dependency_id FROM task_dependencies WHERE task_id=?
                    UNION
                    SELECT d.dependency_id FROM task_dependencies d JOIN ancestors a ON d.task_id=a.id
                )
                SELECT t.id AS task_id,t.key,t.stage,a.error FROM ancestors p JOIN tasks t ON t.id=p.id
                LEFT JOIN attempts a ON a.id=t.attempt_id WHERE t.status='failed' ORDER BY t.key
            ''', (row['id'],))]
            for failed in task['blocked_by']:
                failed['error'] = json.loads(failed['error']) if failed['error'] is not None else None
            if task['blocked_by']:
                task['status'] = 'blocked'
            else:
                task['waiting_for'] = [ref['task_id'] for ref in task['dependencies'] if ref['status'] != 'succeeded']
        return task

    def get_task(self, task_id: str) -> dict | None:
        with self._transaction() as db:
            row = db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
            return None if row is None else self._task(db, row)

    def list_tasks(self, *, stage: str | None = None, status: str | None = None) -> list[dict]:
        """Read task state, including computed blocked reasons and pending waits."""
        with self._transaction() as db:
            tasks = [self._task(db, row) for row in db.execute(
                'SELECT * FROM tasks WHERE (? IS NULL OR stage=?) ORDER BY created_at,rowid', (stage, stage))]
            return [task for task in tasks if status is None or task['status'] == status]

    @staticmethod
    def _result(db, row) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        result['result'] = json.loads(result['result'])
        result['dependencies'] = [dict(ref) for ref in db.execute(
            'SELECT dependency_id AS task_id,dependency_result_id AS result_id FROM result_dependencies '
            'WHERE result_id=? ORDER BY dependency_id', (row['id'],))]
        return result

    def get_result(self, key: str) -> dict | None:
        with self._transaction() as db:
            return self._result(db, db.execute('SELECT r.* FROM results r JOIN tasks t ON t.id=r.task_id '
                                               'WHERE t.key=?', (key,)).fetchone())

    def get_result_by_id(self, result_id: str) -> dict | None:
        with self._transaction() as db:
            return self._result(db, db.execute('SELECT * FROM results WHERE id=?', (result_id,)).fetchone())

    def list_attempts(self, task_id: str | None = None) -> list[dict]:
        with self._transaction() as db:
            attempts = []
            for row in db.execute('SELECT * FROM attempts WHERE (? IS NULL OR task_id=?) '
                                  'ORDER BY started_at,rowid', (task_id, task_id)):
                attempt = dict(row)
                for field in ('response', 'usage', 'validation', 'error'):
                    attempt[field] = json.loads(attempt[field]) if attempt[field] is not None else None
                attempt['uncertain'] = bool(attempt['uncertain'])
                attempts.append(attempt)
            return attempts

    def stats(self, *, stage: str | None = None) -> dict[str, int]:
        with self._transaction() as db:
            counts = dict.fromkeys(('pending', 'running', 'succeeded', 'failed', 'blocked'), 0)
            rows = db.execute('SELECT * FROM tasks WHERE (? IS NULL OR stage=?)', (stage, stage)).fetchall()
            for row in rows:
                counts[self._task(db, row)['status']] += 1
            counts['total'] = len(rows)
            for table in ('attempts', 'results'):
                counts[table] = db.execute(f'SELECT COUNT(*) FROM {table} x JOIN tasks t ON t.id=x.task_id '
                                          'WHERE (? IS NULL OR t.stage=?)', (stage, stage)).fetchone()[0]
            return counts


class _ReadOnlyStore(TaskStore):
    def __init__(self, path):
        self.path = str(path)

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(Path(self.path).resolve().as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()
