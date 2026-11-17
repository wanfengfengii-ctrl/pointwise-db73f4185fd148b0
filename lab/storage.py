import sqlite3
from contextlib import contextmanager
from .schema import initialize_schema, verify_schema

class Storage:
    def __init__(self,path=":memory:"):
        self.connection = sqlite3.connect(path,isolation_level=None)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.failures = set()
        try:
            self.initialize()
        except BaseException:
            self.close()
            raise

    def initialize(self):
        with self.transaction():
            initialize_schema(self.connection)
        verify_schema(self.connection)

    @contextmanager
    def transaction(self):
        if self.connection.in_transaction:
            raise RuntimeError("nested transaction")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self
            self.checkpoint("commit")
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    @contextmanager
    def failing(self,*stages):
        old = self.failures.copy()
        self.failures.update(stages)
        try:
            yield self
        finally:
            self.failures = old

    def checkpoint(self,stage):
        if stage in self.failures:
            raise sqlite3.OperationalError("injected publication failure: "+stage)

    def fetch_one(self,sql,parameters=()):
        return self.connection.execute(sql,parameters).fetchone()

    def fetch_all(self,sql,parameters=()):
        return self.connection.execute(sql,parameters).fetchall()

    def execute(self,sql,parameters=()):
        return self.connection.execute(sql,parameters)

    def close(self):
        self.connection.close()
