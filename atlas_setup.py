import json

from db import atlas_enabled, ensure_atlas_schema, get_db


def main():
    if not atlas_enabled():
        print("Atlas disabled: MONGODB_URI not found.")
        return

    status = ensure_atlas_schema()
    print("Atlas schema initialized:")
    print(json.dumps(status, indent=2))

    db = get_db()
    print(f"Database: {db.name}")
    print("Collections:", sorted(db.list_collection_names()))


if __name__ == "__main__":
    main()
