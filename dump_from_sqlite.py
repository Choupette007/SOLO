import sqlite3
import json
import sys
from pathlib import Path

# Edit DB_PATH if your DB is elsewhere
DB_PATH = Path(r"C:\Users\Admin\AppData\Local\SOLOTradingBot\tokens.sqlite3")

def find_rows(db_path: Path, target: str):
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    results = []
    for tbl in tables:
        try:
            cols = [c[1] for c in cur.execute(f"PRAGMA table_info('{tbl}')").fetchall()]
            if not cols:
                continue
            # Build a query that checks each TEXT column for the address substring
            qcols = []
            for c in cols:
                qcols.append(f"CAST({c} AS TEXT) LIKE ?")
            q = f"SELECT * FROM {tbl} WHERE {' OR '.join(qcols)} LIMIT 20"
            args = ["%"+target+"%"] * len(cols)
            rows = cur.execute(q, args).fetchall()
            for r in rows:
                rd = {k: r[k] for k in r.keys()}
                # attempt to json-decode JSON-like strings
                for k, v in list(rd.items()):
                    if isinstance(v, str) and v.strip() and (v.strip().startswith("{") or v.strip().startswith("[")):
                        try:
                            rd[k] = json.loads(v)
                        except Exception:
                            pass
                results.append({"table": tbl, "row": rd})
        except Exception:
            # skip tables we can't query
            continue
    con.close()
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dump_from_sqlite.py <address_prefix_or_full>")
        sys.exit(1)
    target = sys.argv[1]
    found = find_rows(DB_PATH, target)
    if not found:
        print("No matching rows found for:", target)
        sys.exit(0)
    print(json.dumps(found, indent=2, ensure_ascii=False))