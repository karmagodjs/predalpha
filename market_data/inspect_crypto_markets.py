import requests

URL = "https://gamma-api.polymarket.com/markets"
KEYWORDS = ("bitcoin", "btc", "ethereum", "eth", "solana", "sol", "up or down")

response = requests.get(
    URL,
    params={
        "active": "true",
        "closed": "false",
        "order": "volume24hr",
        "ascending": "false",
        "limit": 100,
    },
    timeout=20,
)
response.raise_for_status()
markets = response.json()

print("Fetched:", len(markets))

matches = []
for market in markets:
    question = str(market.get("question", ""))
    slug = str(market.get("slug", ""))
    searchable = f"{question} {slug}".lower()

    if any(word in searchable for word in KEYWORDS):
        matches.append(market)

print("Crypto keyword matches:", len(matches))

for market in matches:
    print("\n" + "-" * 60)
    print("Question:", market.get("question"))
    print("Slug:", market.get("slug"))
    print("End date:", market.get("endDate"))
    print("Volume 24h:", market.get("volume24hr"))
    print("Volume CLOB:", market.get("volume24hrClob"))
    print("Active:", market.get("active"))
    print("Closed:", market.get("closed"))