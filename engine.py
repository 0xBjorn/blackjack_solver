"""
Monte Carlo Blackjack simulation engine.
Handles all game logic and EV calculations.
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from deck import InfiniteDeck, hand_value, is_blackjack, is_bust


class Action(Enum):
    """Possible player actions."""
    HIT = "H"
    STAND = "S"
    DOUBLE = "D"
    SPLIT = "P"
    SURRENDER = "R"


@dataclass
class ActionStats:
    """Statistics for a single action in a state."""
    n: int = 0
    sum_x: float = 0.0
    sum_x_squared: float = 0.0

    def update(self, result: float) -> None:
        """Update statistics with a new result."""
        self.n += 1
        self.sum_x += result
        self.sum_x_squared += result * result

    def ev(self) -> float:
        """Calculate expected value (mean return)."""
        if self.n == 0:
            return float('-inf')  # Unsimulated actions should never be selected
        return self.sum_x / self.n

    def variance(self) -> float:
        """Calculate variance."""
        if self.n < 2:
            return float('inf')
        mean = self.sum_x / self.n
        return (self.sum_x_squared / self.n) - (mean * mean)

    def sem(self) -> float:
        """Calculate Standard Error of the Mean."""
        if self.n < 2:
            return float('inf')
        var = self.variance()
        if var < 0:  # Numerical stability
            var = 0
        return math.sqrt(var / self.n)

    def merge(self, other: 'ActionStats') -> None:
        """Merge statistics from another ActionStats object."""
        self.n += other.n
        self.sum_x += other.sum_x
        self.sum_x_squared += other.sum_x_squared


@dataclass
class StateStats:
    """Statistics for all actions in a given state."""
    actions: Dict[Action, ActionStats] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize stats for all actions
        for action in Action:
            if action not in self.actions:
                self.actions[action] = ActionStats()

    def best_action(self) -> Tuple[Action, float]:
        """Return the action with highest EV and its value."""
        best = max(self.actions.items(), key=lambda x: x[1].ev())
        return best[0], best[1].ev()

    def all_converged(self, target_sem: float) -> bool:
        """Check if all actions have converged to target SEM."""
        return all(stats.sem() < target_sem for stats in self.actions.values())

    def needs_simulation(self, action: Action, target_sem: float) -> bool:
        """Check if an action still needs more simulation."""
        return self.actions[action].sem() >= target_sem


class BlackjackEngine:
    """
    Monte Carlo simulation engine for Blackjack.
    Implements Evolution Gaming rules (S17, DAS, ENHC).
    """

    def __init__(self, use_infinite_deck: bool = True):
        """
        Initialize engine.

        Args:
            use_infinite_deck: If True, use infinite deck (recommended for base strategy)
        """
        self.deck = InfiniteDeck()
        self.use_infinite_deck = use_infinite_deck

    def dealer_play(self, dealer_cards: List[int]) -> List[int]:
        """
        Play out dealer's hand according to S17 rules.
        Dealer stands on all 17s (including soft 17).

        Args:
            dealer_cards: Dealer's starting cards

        Returns:
            Final dealer hand
        """
        cards = dealer_cards.copy()

        while True:
            total, is_soft = hand_value(cards)

            # S17: Dealer stands on all 17s
            if total >= 17:
                break

            # Draw another card
            cards.append(self.deck.draw())

        return cards

    def play_hand_hit(self, player_cards: List[int], dealer_upcard: int,
                      dealer_hole: int) -> float:
        """
        Simulate hitting (drawing one card and continuing optimally).
        For Monte Carlo, we use a simple strategy after the hit.

        Args:
            player_cards: Current player cards
            dealer_upcard: Dealer's visible card
            dealer_hole: Dealer's hole card

        Returns:
            Result of the hand (-1 to +1.5)
        """
        cards = player_cards.copy()
        cards.append(self.deck.draw())

        if is_bust(cards):
            return -1.0

        # After hitting, continue with approximate basic strategy
        while True:
            total, is_soft = hand_value(cards)

            # Always stand on 17+
            if total >= 17:
                break

            # Soft hand strategy
            if is_soft:
                # Stand on soft 18+ (already covered by >= 17 check for soft 19+)
                if total >= 18:
                    break
                # Hit soft 17 or less
            else:
                # Hard hand strategy
                if total >= 17:
                    break
                # Stand on 12-16 vs dealer 2-6
                if total >= 12 and dealer_upcard in [2, 3, 4, 5, 6]:
                    break
                # Hit on 12-16 vs dealer 7+ (continue loop)
                # Always hit on 11 or less (continue loop)

            cards.append(self.deck.draw())
            if is_bust(cards):
                return -1.0

        return self._resolve_vs_dealer(cards, dealer_upcard, dealer_hole)

    def play_hand_stand(self, player_cards: List[int], dealer_upcard: int,
                        dealer_hole: int) -> float:
        """
        Simulate standing.

        Args:
            player_cards: Current player cards
            dealer_upcard: Dealer's visible card
            dealer_hole: Dealer's hole card

        Returns:
            Result of the hand (-1 to +1)
        """
        return self._resolve_vs_dealer(player_cards, dealer_upcard, dealer_hole)

    def play_hand_double(self, player_cards: List[int], dealer_upcard: int,
                         dealer_hole: int) -> float:
        """
        Simulate doubling down.

        Args:
            player_cards: Current player cards
            dealer_upcard: Dealer's visible card
            dealer_hole: Dealer's hole card

        Returns:
            Result of the hand (-2 to +2)
        """
        cards = player_cards.copy()
        cards.append(self.deck.draw())

        if is_bust(cards):
            return -2.0

        result = self._resolve_vs_dealer(cards, dealer_upcard, dealer_hole)
        return result * 2.0

    def play_hand_split(self, player_cards: List[int], dealer_upcard: int,
                        dealer_hole: int, can_double: bool = True) -> float:
        """
        Simulate splitting a pair.
        Only one split allowed (2 hands max).
        Split aces receive only 1 card each.

        Args:
            player_cards: Current player cards (must be a pair)
            dealer_upcard: Dealer's visible card
            dealer_hole: Dealer's hole card
            can_double: Whether doubling is allowed on split hands (DAS)

        Returns:
            Combined result of both hands
        """
        if len(player_cards) != 2 or player_cards[0] != player_cards[1]:
            raise ValueError("Cannot split non-pair")

        split_card = player_cards[0]
        is_aces = (split_card == 11)

        total_result = 0.0

        for _ in range(2):
            hand = [split_card, self.deck.draw()]

            if is_aces:
                # Split aces: only one card, no further action
                result = self._resolve_vs_dealer(hand, dealer_upcard, dealer_hole)
            else:
                # Play the hand with basic strategy
                result = self._play_split_hand(hand, dealer_upcard, dealer_hole, can_double)

            total_result += result

        return total_result

    def _play_split_hand(self, hand: List[int], dealer_upcard: int,
                         dealer_hole: int, can_double: bool) -> float:
        """
        Play a single split hand with simplified strategy.
        After split, play using basic hit/stand/double logic.
        """
        total, is_soft = hand_value(hand)

        # Check if we should double (only on 2 cards with DAS)
        if can_double and len(hand) == 2:
            # Simple doubling logic: double on 9, 10, 11, soft 16-18
            should_double = False
            if not is_soft and total in [9, 10, 11]:
                should_double = True
            elif is_soft and total in [16, 17, 18]:
                should_double = True

            if should_double:
                hand.append(self.deck.draw())
                if is_bust(hand):
                    return -2.0
                return self._resolve_vs_dealer(hand, dealer_upcard, dealer_hole) * 2.0

        # Hit until we reach standing threshold
        while True:
            total, is_soft = hand_value(hand)

            # Standing thresholds
            if is_soft:
                if total >= 18:
                    break
            else:
                if total >= 17:
                    break
                # Stand on 12-16 vs dealer 2-6
                if total >= 12 and dealer_upcard in [2, 3, 4, 5, 6]:
                    break

            hand.append(self.deck.draw())
            if is_bust(hand):
                return -1.0

        return self._resolve_vs_dealer(hand, dealer_upcard, dealer_hole)

    def _resolve_vs_dealer(self, player_cards: List[int], dealer_upcard: int,
                           dealer_hole: int) -> float:
        """
        Resolve player hand against dealer after player is done.
        Implements ENHC (European No Hole Card) rule.

        Args:
            player_cards: Final player hand
            dealer_upcard: Dealer's visible card
            dealer_hole: Dealer's hole card

        Returns:
            Result: +1.5 (BJ), +1 (win), 0 (push), -1 (loss)
        """
        player_total, _ = hand_value(player_cards)

        # ENHC: Check if dealer has blackjack
        dealer_cards = [dealer_upcard, dealer_hole]
        if is_blackjack(dealer_cards):
            # Player loses everything (including doubles/splits)
            # The -1 here represents the base bet loss
            # Double/split multipliers are handled in calling functions
            return -1.0

        # Play out dealer's hand
        dealer_cards = self.dealer_play(dealer_cards)
        dealer_total, _ = hand_value(dealer_cards)

        # Compare hands
        if is_bust(dealer_cards):
            return 1.0
        elif player_total > dealer_total:
            return 1.0
        elif player_total < dealer_total:
            return -1.0
        else:
            return 0.0

    def simulate_action(self, player_cards: List[int], dealer_upcard: int,
                        action: Action) -> float:
        """
        Simulate a single hand with the given action.

        Args:
            player_cards: Player's initial cards
            dealer_upcard: Dealer's up card
            action: Action to take

        Returns:
            Result of the hand
        """
        # Draw dealer's hole card
        dealer_hole = self.deck.draw()

        # Check for player blackjack (only relevant for initial deal)
        if len(player_cards) == 2 and is_blackjack(player_cards):
            # Check dealer blackjack
            dealer_cards = [dealer_upcard, dealer_hole]
            if is_blackjack(dealer_cards):
                return 0.0  # Push
            return 1.5  # Blackjack pays 3:2

        if action == Action.HIT:
            return self.play_hand_hit(player_cards, dealer_upcard, dealer_hole)
        elif action == Action.STAND:
            return self.play_hand_stand(player_cards, dealer_upcard, dealer_hole)
        elif action == Action.DOUBLE:
            return self.play_hand_double(player_cards, dealer_upcard, dealer_hole)
        elif action == Action.SPLIT:
            return self.play_hand_split(player_cards, dealer_upcard, dealer_hole)
        elif action == Action.SURRENDER:
            # Late surrender: lose half bet
            # With ENHC, if dealer has BJ, player loses full bet even on surrender
            dealer_cards = [dealer_upcard, dealer_hole]
            if is_blackjack(dealer_cards):
                return -1.0  # Lose full bet to dealer blackjack
            return -0.5  # Normal surrender

        raise ValueError(f"Unknown action: {action}")

    def simulate_batch(self, player_cards: List[int], dealer_upcard: int,
                       action: Action, batch_size: int = 10000) -> ActionStats:
        """
        Simulate a batch of hands for a given state-action pair.

        Args:
            player_cards: Player's initial cards
            dealer_upcard: Dealer's up card
            action: Action to take
            batch_size: Number of hands to simulate

        Returns:
            ActionStats with results
        """
        stats = ActionStats()

        for _ in range(batch_size):
            result = self.simulate_action(player_cards.copy(), dealer_upcard, action)
            stats.update(result)

        return stats


def generate_all_states() -> List[Tuple[int, int, bool, bool]]:
    """
    Generate all possible player states.

    Returns:
        List of (player_total, dealer_upcard, is_soft, is_pair) tuples
    """
    states = []

    # Hard totals: 5-21 (we skip 4 as it's only 2,2 which is a pair)
    for total in range(5, 22):
        for dealer in range(2, 12):  # 2-10, 11 (Ace)
            states.append((total, dealer, False, False))

    # Soft totals: 13-20 (A,2 through A,9)
    # Note: Soft 21 (A,10) is blackjack - no decision needed
    for total in range(13, 21):
        for dealer in range(2, 12):
            states.append((total, dealer, True, False))

    # Pairs: 2,2 through A,A
    for card in range(2, 12):  # 2-10, 11 (Ace)
        pair_total = card * 2 if card != 11 else 12  # A,A = soft 12
        for dealer in range(2, 12):
            states.append((pair_total, dealer, card == 11, True))

    return states


def get_cards_for_state(total: int, is_soft: bool, is_pair: bool) -> List[int]:
    """
    Generate cards that create a given state.

    Args:
        total: Player's hand total
        is_soft: Whether hand is soft
        is_pair: Whether hand is a pair

    Returns:
        List of cards creating this state
    """
    if is_pair:
        # For pairs, return the two identical cards
        if is_soft:  # A,A
            return [11, 11]
        else:
            card = total // 2
            return [card, card]

    if is_soft:
        # Soft hands: Ace + (total - 11)
        other = total - 11
        if other < 2:
            # Can't make this soft total with 2 cards, use 3
            return [11, 2, other - 2] if other > 2 else [11, other]
        return [11, other]

    # Hard hands: various combinations
    if total <= 11:
        # Small totals
        return [2, total - 2] if total >= 4 else [total]
    elif total <= 19:
        # Use 10 + something
        return [10, total - 10]
    else:
        # 20, 21
        if total == 20:
            return [10, 10]
        else:  # 21
            return [10, 10, 1]  # Need 3 cards for hard 21


def get_valid_actions(is_pair: bool, num_cards: int = 2) -> List[Action]:
    """
    Get valid actions for a state.

    Args:
        is_pair: Whether hand is a pair
        num_cards: Number of cards in hand

    Returns:
        List of valid actions
    """
    actions = [Action.HIT, Action.STAND]

    if num_cards == 2:
        actions.append(Action.DOUBLE)
        actions.append(Action.SURRENDER)  # Late surrender on initial hand

    if is_pair and num_cards == 2:
        actions.append(Action.SPLIT)

    return actions
