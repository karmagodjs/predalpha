import time
import requests
from collections import defaultdict


GAMMA_API = "https://gamma-api.polymarket.com/markets"


def get_active_markets(limit=100):
    response = requests.get(
        GAMMA_API,
        params={
            "active": "true",
            "closed": "false",
            "limit": limit,
        },
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def extract_candidates(markets):
    candidates = []

    for market in markets:
        tokens = market.get("clobTokenIds")

        if not tokens:
            continue

        for token_id in tokens:
            candidates.append({
                "market_id": market.get("conditionId"),
                "token_id": token_id,
                "question": market.get("question"),
            })

    return candidates


def inspect_market(token_id, duration=30):
    """
    Placeholder for live order-book inspection.

    The collector should provide:
        best_bid
        best_ask
        mid_price
        timestamp
    """

    # This function will be connected to your existing
    # WebSocket/order-book implementation.
    return {
        "token_id": token_id,
        "events": 0,
        "unique_mid_prices": 0,
        "price_changes": 0,
        "avg_spread": None,
    }


def quality_score(stats):
    if stats["events"] == 0:
        return 0

    score = 0

    if stats["unique_mid_prices"] > 1:
        score += 40

    if stats["price_changes"] > 0:
        score += 40

    if stats["events"] >= 100:
        score += 20

    return score


def main():

    markets = get_active_markets()

    candidates = extract_candidates(markets)

    print("Candidate tokens:", len(candidates))

    results = []

    for i, candidate in enumerate(candidates[:20], 1):

        print(
            f"\n[{i}/{min(20, len(candidates))}] "
            f"{candidate['question']}"
        )

        stats = inspect_market(
            candidate["token_id"],
            duration=30,
        )

        stats["market_id"] = candidate["market_id"]
        stats["question"] = candidate["question"]
        stats["score"] = quality_score(stats)

        results.append(stats)

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    print("\n" + "=" * 80)
    print("MARKET QUALITY")
    print("=" * 80)

    for result in results:
        print(
            f"Score={result['score']:3d} | "
            f"Events={result['events']:5d} | "
            f"UniqueMid={result['unique_mid_prices']:4d} | "
            f"Changes={result['price_changes']:4d} | "
            f"{result['question']}"
        )


if __name__ == "__main__":
    main()