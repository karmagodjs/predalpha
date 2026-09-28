import requests


GAMMA_API = "https://gamma-api.polymarket.com/markets"


def get_active_markets(limit=100):
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

    return response.json()


def print_markets(markets):

    for i, market in enumerate(markets):

        print("\n" + "=" * 70)

        print(f"#{i + 1}")
        print("Question:", market.get("question"))
        print("Condition:", market.get("conditionId"))

        tokens = market.get("clobTokenIds")

        print("Tokens:", tokens)

        print(
            "Active:",
            market.get("active"),
            "| Closed:",
            market.get("closed"),
        )


if __name__ == "__main__":

    markets = get_active_markets(100)

    print("Markets found:", len(markets))

    print_markets(markets)