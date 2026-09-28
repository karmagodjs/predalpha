class Portfolio:
    def __init__(self, initial_cash=10_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0

    def buy(self, price, quantity, fee):
        cost = price * quantity + fee

        if cost > self.cash:
            return False

        previous_position = self.position

        self.cash -= cost

        if previous_position + quantity > 0:
            self.avg_entry_price = (
                self.avg_entry_price * previous_position
                + price * quantity
            ) / (previous_position + quantity)

        self.position += quantity

        return True

    def sell(self, price, quantity, fee):
        if quantity > self.position:
            return False

        proceeds = price * quantity - fee

        self.cash += proceeds

        pnl = (
            price - self.avg_entry_price
        ) * quantity - fee

        self.realized_pnl += pnl
        self.position -= quantity

        if self.position == 0:
            self.avg_entry_price = 0.0

        return True

    def equity(self, market_price):
        return self.cash + self.position * market_price