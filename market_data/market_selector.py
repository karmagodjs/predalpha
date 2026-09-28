import requests


GAMMA_API = "https://gamma-api.polymarket.com/markets"


def get_candidate_markets(limit=100):
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
    }

    response = requests.get(
        GAMMA_API,
        params=params,
        timeout=10,
    )

    response.raise_for_status()

    markets = response.json()

    candidates = []

    for market in markets:
        tokens = market.get("clobTokenIds")

        if not tokens:
            continue

        candidates.append({
            "question": market.get("question"),
            "condition_id": market.get("conditionId"),
            "tokens": tokens,
        })

    return candidates


if __name__ == "__main__":
    markets = get_candidate_markets()

    print(f"Candidate markets: {len(markets)}")

    for i, market in enumerate(markets[:20], 1):
        print("\n" + "=" * 70)
        print(f"#{i}")
        print("Question:", market["question"])
        print("Condition:", market["condition_id"])
        print("Tokens:", market["tokens"])