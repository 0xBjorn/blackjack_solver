"""
Deck/Shoe management for Blackjack simulation.
Supports both finite 8-deck shoe and infinite deck modes.
"""

import random
from typing import List, Optional
import numpy as np


# Card values: 2-10 face value, J/Q/K = 10, A = 11 (handled specially)
CARD_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]  # 2-10, J, Q, K, A


class Shoe:
    """
    8-deck shoe with configurable penetration.
    Shuffles when penetration threshold is reached.
    """

    def __init__(self, num_decks: int = 8, penetration: float = 0.5):
        """
        Initialize shoe.

        Args:
            num_decks: Number of decks in shoe (default 8)
            penetration: Fraction of shoe dealt before reshuffle (default 0.5)
        """
        self.num_decks = num_decks
        self.penetration = penetration
        self.cards: List[int] = []
        self.cut_card_position: int = 0
        self.shuffle()

    def shuffle(self) -> None:
        """Shuffle all cards back into the shoe."""
        # Each deck has 4 of each card value (4 suits)
        single_deck = CARD_VALUES * 4
        self.cards = single_deck * self.num_decks
        random.shuffle(self.cards)
        # Cut card position - reshuffle after this many cards dealt
        self.cut_card_position = int(len(self.cards) * self.penetration)

    def needs_shuffle(self) -> bool:
        """Check if shoe needs reshuffling based on penetration."""
        cards_dealt = (self.num_decks * 52) - len(self.cards)
        return cards_dealt >= self.cut_card_position

    def draw(self) -> int:
        """Draw a card from the shoe."""
        if not self.cards:
            self.shuffle()
        return self.cards.pop()

    def draw_specific(self, value: int) -> Optional[int]:
        """
        Draw a specific card value from the shoe (for testing/setup).
        Returns None if card not available.
        """
        if value in self.cards:
            self.cards.remove(value)
            return value
        return None


class InfiniteDeck:
    """
    Infinite deck - each card drawn independently with equal probability.
    Used for computing base strategy (no card counting effects).
    """

    # Probability weights for each card value
    # 2-9: 1/13 each, 10/J/Q/K: 4/13, A: 1/13
    WEIGHTS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=np.float64)
    WEIGHTS = WEIGHTS / WEIGHTS.sum()
    VALUES = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    def __init__(self):
        """Initialize infinite deck."""
        pass

    def shuffle(self) -> None:
        """No-op for infinite deck."""
        pass

    def needs_shuffle(self) -> bool:
        """Infinite deck never needs shuffle."""
        return False

    def draw(self) -> int:
        """Draw a random card with correct probability distribution."""
        return int(np.random.choice(self.VALUES, p=self.WEIGHTS))

    def draw_specific(self, value: int) -> int:
        """For infinite deck, always returns the requested value."""
        return value


def hand_value(cards: List[int]) -> tuple:
    """
    Calculate the value of a hand.

    Args:
        cards: List of card values (Ace = 11)

    Returns:
        (total, is_soft): Total value and whether hand is soft
    """
    total = sum(cards)
    aces = cards.count(11)

    # Convert aces from 11 to 1 as needed to avoid bust
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    # Hand is soft if it contains an ace counted as 11
    is_soft = aces > 0 and total <= 21

    return total, is_soft


def is_blackjack(cards: List[int]) -> bool:
    """Check if hand is a natural blackjack (Ace + 10-value on first 2 cards)."""
    if len(cards) != 2:
        return False
    total, _ = hand_value(cards)
    return total == 21


def is_bust(cards: List[int]) -> bool:
    """Check if hand is busted (over 21)."""
    total, _ = hand_value(cards)
    return total > 21


def is_pair(cards: List[int]) -> bool:
    """Check if hand is a splittable pair."""
    return len(cards) == 2 and cards[0] == cards[1]


def get_state(player_cards: List[int], dealer_upcard: int) -> tuple:
    """
    Get the state tuple for strategy lookup.

    Args:
        player_cards: Player's cards
        dealer_upcard: Dealer's visible card

    Returns:
        (player_total, dealer_upcard, is_soft, is_pair)
    """
    total, is_soft = hand_value(player_cards)
    pair = is_pair(player_cards)

    return (total, dealer_upcard, is_soft, pair)
