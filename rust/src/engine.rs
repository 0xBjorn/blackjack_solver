//! Monte Carlo Blackjack simulation engine.
//! Handles all game logic and EV calculations.

use crate::deck::{hand_value, is_blackjack, is_bust, InfiniteDeck, PlayerState};

/// Possible player actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Hit,
    Stand,
    Double,
    Split,
    Surrender,
}

impl Action {
    pub fn symbol(&self) -> &'static str {
        match self {
            Action::Hit => "H",
            Action::Stand => "S",
            Action::Double => "D",
            Action::Split => "P",
            Action::Surrender => "R",
        }
    }

    /// Get all valid actions for a state
    pub fn valid_actions(is_pair: bool) -> Vec<Action> {
        let mut actions = vec![Action::Hit, Action::Stand, Action::Double, Action::Surrender];
        if is_pair {
            actions.push(Action::Split);
        }
        actions
    }
}

/// Statistics for a single action in a state
#[derive(Debug, Clone, Default)]
pub struct ActionStats {
    pub n: u64,
    pub sum_x: f64,
    pub sum_x_squared: f64,
}

impl ActionStats {
    pub fn new() -> Self {
        ActionStats::default()
    }

    #[inline]
    pub fn update(&mut self, result: f64) {
        self.n += 1;
        self.sum_x += result;
        self.sum_x_squared += result * result;
    }

    pub fn ev(&self) -> f64 {
        if self.n == 0 {
            f64::NEG_INFINITY
        } else {
            self.sum_x / self.n as f64
        }
    }

    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            f64::INFINITY
        } else {
            let mean = self.sum_x / self.n as f64;
            (self.sum_x_squared / self.n as f64) - (mean * mean)
        }
    }

    pub fn sem(&self) -> f64 {
        if self.n < 2 {
            f64::INFINITY
        } else {
            let var = self.variance().max(0.0);
            (var / self.n as f64).sqrt()
        }
    }

    pub fn merge(&mut self, other: &ActionStats) {
        self.n += other.n;
        self.sum_x += other.sum_x;
        self.sum_x_squared += other.sum_x_squared;
    }
}

/// Blackjack simulation engine
pub struct BlackjackEngine {
    deck: InfiniteDeck,
}

impl BlackjackEngine {
    pub fn new() -> Self {
        BlackjackEngine {
            deck: InfiniteDeck::new(),
        }
    }

    /// Play out dealer's hand according to S17 rules
    fn dealer_play(&mut self, dealer_cards: &mut Vec<u8>) {
        loop {
            let hv = hand_value(dealer_cards);
            // S17: Dealer stands on all 17s
            if hv.total >= 17 {
                break;
            }
            dealer_cards.push(self.deck.draw());
        }
    }

    /// Simulate hitting
    fn play_hand_hit(&mut self, player_cards: &[u8], dealer_upcard: u8, dealer_hole: u8) -> f64 {
        let mut cards = player_cards.to_vec();
        cards.push(self.deck.draw());

        if is_bust(&cards) {
            return -1.0;
        }

        // Continue with approximate basic strategy
        loop {
            let hv = hand_value(&cards);

            if hv.total >= 17 {
                break;
            }

            if hv.is_soft {
                if hv.total >= 18 {
                    break;
                }
            } else {
                // Stand on 12-16 vs dealer 2-6
                if hv.total >= 12 && (2..=6).contains(&dealer_upcard) {
                    break;
                }
            }

            cards.push(self.deck.draw());
            if is_bust(&cards) {
                return -1.0;
            }
        }

        self.resolve_vs_dealer(&cards, dealer_upcard, dealer_hole)
    }

    /// Simulate standing
    fn play_hand_stand(&mut self, player_cards: &[u8], dealer_upcard: u8, dealer_hole: u8) -> f64 {
        self.resolve_vs_dealer(player_cards, dealer_upcard, dealer_hole)
    }

    /// Simulate doubling down
    fn play_hand_double(&mut self, player_cards: &[u8], dealer_upcard: u8, dealer_hole: u8) -> f64 {
        let mut cards = player_cards.to_vec();
        cards.push(self.deck.draw());

        if is_bust(&cards) {
            return -2.0;
        }

        self.resolve_vs_dealer(&cards, dealer_upcard, dealer_hole) * 2.0
    }

    /// Simulate splitting a pair
    fn play_hand_split(&mut self, player_cards: &[u8], dealer_upcard: u8, dealer_hole: u8) -> f64 {
        if player_cards.len() != 2 || player_cards[0] != player_cards[1] {
            return -999.0; // Invalid split
        }

        let split_card = player_cards[0];
        let is_aces = split_card == 11;

        let mut total_result = 0.0;

        for _ in 0..2 {
            let mut hand = vec![split_card, self.deck.draw()];

            let result = if is_aces {
                // Split aces: only one card, no further action
                self.resolve_vs_dealer(&hand, dealer_upcard, dealer_hole)
            } else {
                self.play_split_hand(&mut hand, dealer_upcard, dealer_hole)
            };

            total_result += result;
        }

        total_result
    }

    /// Play a single split hand with basic strategy
    fn play_split_hand(&mut self, hand: &mut Vec<u8>, dealer_upcard: u8, dealer_hole: u8) -> f64 {
        let hv = hand_value(hand);

        // Check if we should double (DAS)
        if hand.len() == 2 {
            let should_double = if !hv.is_soft {
                matches!(hv.total, 9 | 10 | 11)
            } else {
                matches!(hv.total, 16 | 17 | 18)
            };

            if should_double {
                hand.push(self.deck.draw());
                if is_bust(hand) {
                    return -2.0;
                }
                return self.resolve_vs_dealer(hand, dealer_upcard, dealer_hole) * 2.0;
            }
        }

        // Hit until we reach standing threshold
        loop {
            let hv = hand_value(hand);

            if hv.is_soft {
                if hv.total >= 18 {
                    break;
                }
            } else {
                if hv.total >= 17 {
                    break;
                }
                if hv.total >= 12 && (2..=6).contains(&dealer_upcard) {
                    break;
                }
            }

            hand.push(self.deck.draw());
            if is_bust(hand) {
                return -1.0;
            }
        }

        self.resolve_vs_dealer(hand, dealer_upcard, dealer_hole)
    }

    /// Resolve player hand against dealer (ENHC rules)
    fn resolve_vs_dealer(&mut self, player_cards: &[u8], dealer_upcard: u8, dealer_hole: u8) -> f64 {
        let player_total = hand_value(player_cards).total;

        // ENHC: Check if dealer has blackjack
        let dealer_cards = vec![dealer_upcard, dealer_hole];
        if is_blackjack(&dealer_cards) {
            return -1.0;
        }

        // Play out dealer's hand
        let mut dealer_hand = dealer_cards;
        self.dealer_play(&mut dealer_hand);
        let dealer_total = hand_value(&dealer_hand).total;

        // Compare hands
        if is_bust(&dealer_hand) {
            1.0
        } else if player_total > dealer_total {
            1.0
        } else if player_total < dealer_total {
            -1.0
        } else {
            0.0
        }
    }

    /// Simulate a single hand with the given action
    pub fn simulate_action(&mut self, player_cards: &[u8], dealer_upcard: u8, action: Action) -> f64 {
        let dealer_hole = self.deck.draw();

        // Check for player blackjack
        if player_cards.len() == 2 && is_blackjack(player_cards) {
            let dealer_cards = vec![dealer_upcard, dealer_hole];
            if is_blackjack(&dealer_cards) {
                return 0.0; // Push
            }
            return 1.5; // Blackjack pays 3:2
        }

        match action {
            Action::Hit => self.play_hand_hit(player_cards, dealer_upcard, dealer_hole),
            Action::Stand => self.play_hand_stand(player_cards, dealer_upcard, dealer_hole),
            Action::Double => self.play_hand_double(player_cards, dealer_upcard, dealer_hole),
            Action::Split => self.play_hand_split(player_cards, dealer_upcard, dealer_hole),
            Action::Surrender => {
                // Late surrender with ENHC
                let dealer_cards = vec![dealer_upcard, dealer_hole];
                if is_blackjack(&dealer_cards) {
                    -1.0 // Lose full bet to dealer blackjack
                } else {
                    -0.5 // Normal surrender
                }
            }
        }
    }

    /// Simulate a batch of hands for a given state-action pair
    pub fn simulate_batch(
        &mut self,
        player_cards: &[u8],
        dealer_upcard: u8,
        action: Action,
        batch_size: u32,
    ) -> ActionStats {
        let mut stats = ActionStats::new();

        for _ in 0..batch_size {
            let result = self.simulate_action(player_cards, dealer_upcard, action);
            stats.update(result);
        }

        stats
    }
}

impl Default for BlackjackEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate all possible player states
pub fn generate_all_states() -> Vec<PlayerState> {
    let mut states = Vec::new();

    // Hard totals: 5-21
    for total in 5..=21 {
        for dealer in 2..=11 {
            states.push(PlayerState::new(total, dealer, false, false));
        }
    }

    // Soft totals: 13-20 (A,2 through A,9; A,10 is blackjack)
    for total in 13..=20 {
        for dealer in 2..=11 {
            states.push(PlayerState::new(total, dealer, true, false));
        }
    }

    // Pairs: 2,2 through A,A
    for card in 2..=11 {
        let pair_total = if card == 11 { 12 } else { card * 2 };
        let is_soft = card == 11;
        for dealer in 2..=11 {
            states.push(PlayerState::new(pair_total, dealer, is_soft, true));
        }
    }

    states
}
