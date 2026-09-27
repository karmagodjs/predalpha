import requests


GAMMA_API = "https://gamma-api.polymarket.com/markets"


def get_active_markets(limit=20):

    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
    }

    response = requests.get(
        GAMMA_API,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    return response.json()


def print_markets(markets):

    for i, market in enumerate(markets):

        print("\n" + "=" * 60)

        print(f"Market #{i + 1}")
        print("Question:", market.get("question"))
        print("Condition ID:", market.get("conditionId"))
        print("Token IDs:", market.get("clobTokenIds"))


if __name__ == "__main__":

    markets = get_active_markets(limit=10)

    print(f"Found {len(markets)} active markets")

    print_markets(markets)