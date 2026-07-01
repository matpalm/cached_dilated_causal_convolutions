import sqlite3
import numpy as np
from typing import List


class SampleDB(object):

    def __init__(self, db_file="sample.db", check_same_thread=True):
        self.conn = sqlite3.connect(db_file, check_same_thread=check_same_thread)
        self.create_if_required()

    def create_if_required(self):
        c = self.conn.cursor()
        try:
            c.execute("""create table cv_values (
                            run text,
                            idx int,
                            cv0 real,
                            cv1 real,
                            cv2 real,
                            cv3 real,
                            captured bool
                        )""")
            c.execute("create index idx_cv_values_run on cv_values(run)")
            c.execute("create index idx_cv_values_run_idx on cv_values(run, idx)")
            c.execute("create index idx_cv_values_captured on cv_values(captured)")
        except sqlite3.OperationalError:
            # assume table already exists? clumsy...
            pass

    def set_cv_values(self, run: str, idx: int, cv: List[float]):
        run = str(run)
        c = self.conn.cursor()
        c.execute(
            """
            insert into cv_values
             (run, idx, cv0, cv1, cv2, cv3, captured)
             values (?, ?, ?, ?, ?, ?, 0)
            """,
            (run, idx, cv[0], cv[1], cv[2], cv[3]),
        )
        self.conn.commit()

    def set_cv_values_from_npy(self, run: str, cv_values: np.array):
        run = str(run)
        assert len(cv_values.shape) == 2
        assert cv_values.shape[-1] == 4
        c = self.conn.cursor()
        for idx, cv in enumerate(cv_values):
            c.execute(
                """
                insert into cv_values
                 (run, idx, cv0, cv1, cv2, cv3, captured)
                 values (?, ?, ?, ?, ?, ?, 0)
                """,
                (run, idx, cv[0], cv[1], cv[2], cv[3]),
            )
        self.conn.commit()

    def cv_values_for(self, run: str, idx: int = None):
        run = str(run)
        c = self.conn.cursor()
        if idx is None:
            # all idxs
            cvss = []
            c.execute(
                """
                select cv0, cv1, cv2, cv3
                from cv_values
                where run=?
                order by idx
                """,
                (run,),
            )
            for cvs in c.fetchall():
                cvss.append(cvs)
            return np.array(cvss)
        else:
            c.execute(
                """
                select cv0, cv1, cv2, cv3
                from cv_values
                where run=? and idx=?
                order by idx
                """,
                (run, idx),
            )
            return np.array(c.fetchone())

    def captured_stats_for(self, run: str):
        run = str(run)
        c = self.conn.cursor()
        c.execute(
            """
            select captured, count(*) as c
            from cv_values
            where run=?
            group by captured
            """,
            (run,),
        )
        vals = c.fetchall()
        if vals == []:
            raise Exception(f"no entries for run=[{run}] ?")
        captured = {True: 0, False: 0}
        for cb, count in vals:
            captured[bool(cb)] += count
        return captured

    def idxs_to_capture(self, run: str):
        run = str(run)
        c = self.conn.cursor()
        c.execute(
            """
            select idx
            from cv_values
            where run=? and captured=0
            order by idx
            """,
            (run,),
        )
        return [r[0] for r in c.fetchall()]

    def set_captured(self, run: str, idx: int):
        run = str(run)
        c = self.conn.cursor()
        c.execute(
            """
            update cv_values
            set captured=1
            where run=? and idx=?
            """,
            (run, idx),
        )
        self.conn.commit()
