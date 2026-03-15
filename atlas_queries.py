import argparse
import json
import random

from db import atlas_enabled, get_db


def run_mission_summary(mission_id: str):
    db = get_db()
    pipeline = [
        {"$match": {"mission_id": mission_id}},
        {
            "$group": {
                "_id": "$drone_id",
                "samples": {"$sum": 1},
                "avg_velocity": {"$avg": "$velocity_ms"},
                "max_gpr": {"$max": "$gpr_score"},
                "avg_altitude": {"$avg": "$altitude_m"},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    rows = list(db.telemetry_raw.aggregate(pipeline))
    print(json.dumps(rows, indent=2))


def run_geospatial_hazards(mission_id: str, x: float, y: float, radius_m: float):
    db = get_db()
    query = {
        "mission_id": mission_id,
        "location": {
            "$near": {
                "$geometry": {"type": "Point", "coordinates": [x, y]},
                "$maxDistance": float(radius_m),
            }
        },
    }
    docs = list(db.hazards.find(query, {"_id": 0}).limit(20))
    print(json.dumps(docs, indent=2))


def run_text_search(term: str):
    db = get_db()
    pipeline = [
        {
            "$search": {
                "index": "vision_text_search",
                "text": {"query": term, "path": ["error", "status", "source", "decision_label"]},
            }
        },
        {"$limit": 20},
        {"$project": {"_id": 0, "mission_id": 1, "status": 1, "error": 1, "source": 1, "decision_label": 1}},
    ]
    try:
        docs = list(db.vision_events.aggregate(pipeline))
        print(json.dumps(docs, indent=2))
    except Exception as exc:
        print(f"Text search unavailable ({exc})")


def run_vector_search(mission_id: str, dims: int):
    db = get_db()
    query_vec = [random.uniform(-0.01, 0.01) for _ in range(max(4, int(dims)))]
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vision_embedding_vector",
                "path": "embedding",
                "queryVector": query_vec,
                "numCandidates": 50,
                "limit": 10,
                "filter": {"mission_id": mission_id},
            }
        },
        {"$project": {"_id": 0, "mission_id": 1, "mine_id": 1, "status": 1, "decision_label": 1}},
    ]
    try:
        docs = list(db.vision_events.aggregate(pipeline))
        print(json.dumps(docs, indent=2))
    except Exception as exc:
        print(f"Vector search unavailable ({exc})")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_summary = sub.add_parser("summary")
    p_summary.add_argument("--mission_id", type=str, required=True)

    p_geo = sub.add_parser("geo")
    p_geo.add_argument("--mission_id", type=str, required=True)
    p_geo.add_argument("--x", type=float, required=True)
    p_geo.add_argument("--y", type=float, required=True)
    p_geo.add_argument("--radius_m", type=float, default=60.0)

    p_search = sub.add_parser("search")
    p_search.add_argument("--term", type=str, required=True)

    p_vector = sub.add_parser("vector")
    p_vector.add_argument("--mission_id", type=str, required=True)
    p_vector.add_argument("--dims", type=int, default=768)

    args = parser.parse_args()
    if not atlas_enabled():
        print("Atlas disabled: MONGODB_URI not found.")
        return

    if args.cmd == "summary":
        run_mission_summary(args.mission_id)
    elif args.cmd == "geo":
        run_geospatial_hazards(args.mission_id, args.x, args.y, args.radius_m)
    elif args.cmd == "search":
        run_text_search(args.term)
    elif args.cmd == "vector":
        run_vector_search(args.mission_id, args.dims)


if __name__ == "__main__":
    main()
