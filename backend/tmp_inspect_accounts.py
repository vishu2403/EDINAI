from app.postgres import get_pg_cursor

with get_pg_cursor() as cur:
    cur.execute("SELECT enrollment_number, password_hash, last_login FROM student_accounts ORDER BY enrollment_number")
    print("accounts:")
    for row in cur.fetchall():
        print(row)

with get_pg_cursor() as cur:
    cur.execute("SELECT enrollment_number, auto_password FROM student_roster_entries ORDER BY enrollment_number LIMIT 10")
    print("roster:")
    for row in cur.fetchall():
        print(row)
