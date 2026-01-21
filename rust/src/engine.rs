//! Monte Carlo Blackjack simulation engine.
//! Optimized for speed with inlined functions and no heap allocations.

use crate::deck::{hand_value, is_blackjack, is_bust, get_hand_for_state, Hand, InfiniteDeck, PlayerState};

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
    #[inline(always)]
    pub fn symbol(&self) -> &'static str {
        match self {
            Action::Hit => "H",
            Action::Stand => "S",
            Action::Double => "D",
            Action::Split => "P",
            Action::Surrender => "R",
        }
    }

    pub fn valid_actions(is_pair: bool) -> &'static [Action] {
        if is_pair {
            &[Action::Hit, Action::Stand, Action::Double, Action::Surrender, Action::Split]
        } else {
            &[Action::Hit, Action::Stand, Action::Double, Action::Surrender]
        }
    }
}

/// Statistics for a single action
#[derive(Debug, Clone, Default)]
pub struct ActionStats {
    pub n: u64,
    pub sum_x: f64,
    pub sum_x_squared: f64,
}

impl ActionStats {
    #[inline(always)]
    pub fn new() -> Self {
        ActionStats { n: 0, sum_x: 0.0, sum_x_squared: 0.0 }
    }

    #[inline(always)]
    pub fn update(&mut self, result: f64) {
        self.n += 1;
        self.sum_x += result;
        self.sum_x_squared += result * result;
    }

    #[inline(always)]
    pub fn ev(&self) -> f64 {
        if self.n == 0 { f64::NEG_INFINITY } else { self.sum_x / self.n as f64 }
    }

    #[inline(always)]
    pub fn sem(&self) -> f64 {
        if self.n < 2 {
            f64::INFINITY
        } else {
            let mean = self.sum_x / self.n as f64;
            let var = (self.sum_x_squared / self.n as f64) - (mean * mean);
            (var.max(0.0) / self.n as f64).sqrt()
        }
    }

    #[inline(always)]
    pub fn merge(&mut self, other: &ActionStats) {
        self.n += other.n;
        self.sum_x += other.sum_x;
        self.sum_x_squared += other.sum_x_squared;
    }
}

/// Blackjack simulation engine - zero heap allocations in hot path
pub struct BlackjackEngine {
    deck: InfiniteDeck,
}

impl BlackjackEngine {
    #[inline(always)]
    pub fn new() -> Self {
        BlackjackEngine { deck: InfiniteDeck::new() }
    }

    /// Dealer plays according to S17 rules
    #[inline(always)]
    fn dealer_play(&mut self, hand: &mut Hand) {
        loop {
            let (total, _) = hand_value(hand);
            if total >= 17 { break; }
            hand.push(self.deck.draw());
        }
    }

    /// Play hand after hitting
    #[inline(always)]
    fn play_hand_hit(&mut self, hand: &mut Hand, dealer_upcard: u8, dealer_hole: u8) -> f64 {
        hand.push(self.deck.draw());
        if is_bust(hand) { return -1.0; }

        // Continue with basic strategy
        loop {
            let (total, is_soft) = hand_value(hand);
            if total >= 17 { break; }
            if is_soft && total >= 18 { break; }
            if !is_soft && total >= 12 && dealer_upcard <= 6 { break; }

            hand.push(self.deck.draw());
            if is_bust(hand) { return -1.0; }
        }

        self.resolve_vs_dealer(hand, dealer_upcard, dealer_hole)
    }

    /// Play hand after standing
    #[inline(always)]
    fn play_hand_stand(&mut self, hand: &Hand, dealer_upcard: u8, dealer_hole: u8) -> f64 {
        self.resolve_vs_dealer(hand, dealer_upcard, dealer_hole)
    }

    /// Play hand after doubling
    #[inline(always)]
    fn play_hand_double(&mut self, hand: &mut Hand, dealer_upcard: u8, dealer_hole: u8) -> f64 {
        hand.push(self.deck.draw());
        if is_bust(hand) { return -2.0; }
        self.resolve_vs_dealer(hand, dealer_upcard, dealer_hole) * 2.0
    }

    /// Play hand after splitting
    #[inline(always)]
    fn play_hand_split(&mut self, split_card: u8, dealer_upcard: u8, dealer_hole: u8) -> f64 {
        let is_aces = split_card == 11;
        let mut total_result = 0.0;

        for _ in 0..2 {
            let mut hand = Hand::from_cards(split_card, self.deck.draw());

            let result = if is_aces {
                self.resolve_vs_dealer(&hand, dealer_upcard, dealer_hole)
            } else {
                self.play_split_hand(&mut hand, dealer_upcard, dealer_hole)
            };
            total_result += result;
        }

        total_result
    }

    /// Play a single split hand with basic strategy (DAS allowed)
    #[inline(always)]
    fn play_split_hand(&mut self, hand: &mut Hand, dealer_upcard: u8, dealer_hole: u8) -> f64 {
        let (total, is_soft) = hand_value(hand);

        // Check for DAS
        if hand.len() == 2 {
            let should_double = if !is_soft {
                matches!(total, 9 | 10 | 11)
            } else {
                matches!(total, 16 | 17 | 18)
            };

            if should_double {
                hand.push(self.deck.draw());
                if is_bust(hand) { return -2.0; }
                return self.resolve_vs_dealer(hand, dealer_upcard, dealer_hole) * 2.0;
            }
        }

        // Hit until threshold
        loop {
            let (total, is_soft) = hand_value(hand);
            if is_soft && total >= 18 { break; }
            if !is_soft {
                if total >= 17 { break; }
                if total >= 12 && dealer_upcard <= 6 { break; }
            }

            hand.push(self.deck.draw());
            if is_bust(hand) { return -1.0; }
        }

        self.resolve_vs_dealer(hand, dealer_upcard, dealer_hole)
    }

    /// Resolve player hand vs dealer (ENHC rules)
    #[inline(always)]
    fn resolve_vs_dealer(&mut self, player_hand: &Hand, dealer_upcard: u8, dealer_hole: u8) -> f64 {
        let (player_total, _) = hand_value(player_hand);

        // Check dealer blackjack (ENHC)
        let dealer_hand = Hand::from_cards(dealer_upcard, dealer_hole);
        if is_blackjack(&dealer_hand) { return -1.0; }

        // Dealer plays out
        let mut dealer = dealer_hand;
        self.dealer_play(&mut dealer);
        let (dealer_total, _) = hand_value(&dealer);

        if is_bust(&dealer) {
            1.0
        } else if player_total > dealer_total {
            1.0
        } else if player_total < dealer_total {
            -1.0
        } else {
            0.0
        }
    }

    /// Simulate a single hand with given action
    #[inline(always)]
    pub fn simulate_action(&mut self, initial_hand: &Hand, dealer_upcard: u8, action: Action) -> f64 {
        let dealer_hole = self.deck.draw();

        // Check player blackjack
        if initial_hand.len() == 2 && is_blackjack(initial_hand) {
            let dealer = Hand::from_cards(dealer_upcard, dealer_hole);
            if is_blackjack(&dealer) { return 0.0; }
            return 1.5;
        }

        match action {
            Action::Hit => {
                let mut hand = *initial_hand;
                self.play_hand_hit(&mut hand, dealer_upcard, dealer_hole)
            }
            Action::Stand => self.play_hand_stand(initial_hand, dealer_upcard, dealer_hole),
            Action::Double => {
                let mut hand = *initial_hand;
                self.play_hand_double(&mut hand, dealer_upcard, dealer_hole)
            }
            Action::Split => {
                let split_card = initial_hand.first();
                self.play_hand_split(split_card, dealer_upcard, dealer_hole)
            }
            Action::Surrender => {
                let dealer = Hand::from_cards(dealer_upcard, dealer_hole);
                if is_blackjack(&dealer) { -1.0 } else { -0.5 }
            }
        }
    }

    /// Simulate a batch of hands
    #[inline]
    pub fn simulate_batch(&mut self, state: &PlayerState, action: Action, batch_size: u32) -> ActionStats {
        let initial_hand = get_hand_for_state(state.total, state.is_soft, state.is_pair);
        let mut stats = ActionStats::new();

        for _ in 0..batch_size {
            let result = self.simulate_action(&initial_hand, state.dealer_upcard, action);
            stats.update(result);
        }

        stats
    }
}

impl Default for BlackjackEngine {
    fn default() -> Self { Self::new() }
}

/// Generate all possible player states
pub fn generate_all_states() -> Vec<PlayerState> {
    let mut states = Vec::with_capacity(350);

    // Hard totals: 5-21
    for total in 5..=21 {
        for dealer in 2..=11 {
            states.push(PlayerState::new(total, dealer, false, false));
        }
    }

    // Soft totals: 13-20
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
