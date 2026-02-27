# set up the .pgpass file
# then you don't need to type in password
"""Create/update ~/.pgpass for WRDS.

This avoids re-typing your password for PostgreSQL connections (e.g., DuckDB postgres_query).
"""

import os

import wrds


def main() -> None:
	wrds_username = os.environ.get("WRDS_USERNAME")
	if not wrds_username:
		wrds_username = input("WRDS username: ").strip()
	if not wrds_username:
		raise SystemExit("WRDS username is required (set WRDS_USERNAME or type it).")

	db = wrds.Connection(wrds_username=wrds_username)
	db.create_pgpass_file()
	db.close()

	db = wrds.Connection(wrds_username=wrds_username)
	db.close()


if __name__ == "__main__":
	main()
