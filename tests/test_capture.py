import copy
import sqlite3
import unittest
from lab.workflow import Workflow
from samples import exact_rows,E,L,R,I

class TestCapture(unittest.TestCase):
    def setUp(self):
        self.w = Workflow()

    def tearDown(self):
        self.w.close()

    def test_revision_freeze(self):
        rows = exact_rows()
        rows[0]["x"] = [L(.2),I(-1,.4)]
        rows[1]["x"] = [R(-.2),E(1)]
        original = self.w.put_batch("first",rows)
        replay = copy.deepcopy(original)
        changed = copy.deepcopy(rows[0])
        changed["rev"],changed["x"] = 3,[I(-.8,.1),R(.2)]
        self.w.put_batch("third",[changed])
        frozen = self.w.freeze("task",["exact"])
        self.assertEqual(len(frozen["input"]["rows"]),6)
        self.assertEqual(frozen["input"]["rows"][0],changed)
        late = copy.deepcopy(changed)
        late["rev"],late["x"] = 2,[E(8),L(.1)]
        self.w.put_batch("late",[late])
        self.assertEqual(self.w.put_batch("first",rows),replay)
        self.assertEqual(self.w.freeze("task",["exact"]),frozen)
        newer = copy.deepcopy(changed)
        newer["rev"],newer["count"] = 4,7
        self.w.put_batch("fourth",[newer])
        self.assertEqual(self.w.tasks.read_frozen("task"),frozen)
        with self.assertRaises(ValueError):
            self.w.freeze("task",["exact"])
        current = self.w.freeze("new",["exact"])
        self.assertEqual(current["input"]["rows"][0],newer)
        self.assertEqual(self.w.observations.revisions.read_revision("exact","0",1),rows[0])
        self.assertEqual(self.w.observations.revisions.read_revision("exact","0",2),late)

    def test_invalid_batch(self):
        self.w.put_batch("base",exact_rows())
        before = list(self.w.storage.connection.iterdump())
        invalid = [
            {"kind":"interval","lower":1,"upper":0},
            {"kind":"exact","value":None},
            {"kind":"exact","value":1,"upper":2},
            {"kind":"left","upper":float("inf")},
            {"kind":"right","lower":float("nan")},
            {"kind":"interval","lower":0,"upper":0},
            {"kind":"left"},
        ]
        for constraint in invalid:
            bad = exact_rows("bad")
            bad[1]["x"][0] = constraint
            with self.assertRaises(ValueError):
                self.w.put_batch("bad",bad)
            self.assertEqual(list(self.w.storage.connection.iterdump()),before)
        for field in ("count","rev"):
            for value in (0,-1,True,1.5,None):
                bad = exact_rows("bad")
                bad[0][field] = value
                with self.assertRaises(ValueError):
                    self.w.put_batch("bad",bad)
        conflict = exact_rows()
        conflict[0]["count"] = 2
        with self.assertRaises(ValueError):
            self.w.put_batch("conflict",conflict)
        with self.assertRaises(ValueError):
            self.w.put_batch("base",exact_rows("other"))
        self.assertEqual(list(self.w.storage.connection.iterdump()),before)

    def test_duplicate_and_atomic_acceptance(self):
        rows = exact_rows()
        repeated = [*rows,copy.deepcopy(rows[0])]
        result = self.w.put_batch("base",repeated)
        self.assertEqual(len(result["revisions"]),6)
        self.assertEqual(self.w.put_batch("base",list(reversed(rows))),result)
        bad = exact_rows("other")
        different = copy.deepcopy(bad[0])
        different["count"] = 4
        with self.assertRaises(ValueError):
            self.w.put_batch("conflict",bad+[different])
        with self.w.storage.failing("batch"):
            with self.assertRaises(sqlite3.OperationalError):
                self.w.put_batch("failed",bad)
        with self.assertRaises(KeyError):
            self.w.freeze("absent",["other"])
        self.w.put_batch("failed",bad)
        self.assertEqual(len(self.w.freeze("present",["other"])["input"]["rows"]),6)

class TestTasks(unittest.TestCase):
    def test_identity_replay(self):
        with Workflow() as w:
            w.put_batch("a",exact_rows("A"))
            w.put_batch("b",exact_rows("B"))
            request = ["B","A","B"]
            task = w.freeze("t",request)
            self.assertEqual(w.freeze("t",["A","B"]),task)
            self.assertEqual(len(task["input"]["rows"]),12)
            request.clear()
            task["input"]["rows"][0]["count"] = 999
            saved = w.tasks.read_frozen("t")
            self.assertEqual(saved["input"]["rows"][0]["count"],1)
            with self.assertRaises(ValueError):
                w.freeze("t",["A"])
            self.assertEqual(w.tasks.read_frozen("t"),saved)
            with self.assertRaises(KeyError):
                w.freeze("unknown",["missing"])
            with self.assertRaises(ValueError):
                w.freeze("empty",[])
            for stage in ("task.snapshot","task.links","commit"):
                with w.storage.failing(stage):
                    with self.assertRaises(sqlite3.OperationalError):
                        w.freeze("failed",["A"])
                self.assertIsNone(w.storage.fetch_one("SELECT 1 FROM tasks WHERE task_id='failed'"))
                self.assertEqual(w.storage.fetch_one("SELECT count(*) FROM source_links WHERE task_id='failed'"),(0,))
            w.freeze("failed",["A"])
