//! Deck/Shoe management for Blackjack simulation.
//! Supports infinite deck mode for base strategy calculation.

use rand::prelude::*;

/// Card values for infinite deck drawing
/// Probabilities: 2-9 = 1/13 each, 10/J/Q/K = 4/13, A = 1/13
const CARD_VALUES: [u8; 10] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const CARD_WEIGHTS: [u32; 10] = [1, 1, 1, 1, 1, 1, 1, 1, 4, 1]; // 10-value cards have weight 4

/// Infinite deck - each card drawn independently with correct probability
#[derive(Clone)]
pub struct InfiniteDeck {
    rng: ThreadRng,
    cumulative_weights: [u32; 10],
    total_weight: u32,
}

impl InfiniteDeck {
    pub fn new() -> Self {
        let mut cumulative = [0u32; 10];
        let mut sum = 0u32;
        for (i, &w) in CARD_WEIGHTS.iter().enumerate() {
            sum += w;
            cumulative[i] = sum;
        }
        InfiniteDeck {
            rng: thread_rng(),
            cumulative_weights: cumulative,
            total_weight: sum,
        }
    }

    /// Draw a random card with correct probability distribution
    #[inline]
    pub fn draw(&mut self) -> u8 {
        let r = self.rng.gen_range(0..self.total_weight);
        for (i, &cw) in self.cumulative_weights.iter().enumerate() {
            if r < cw {
                return CARD_VALUES[i];
            }
        }
        CARD_VALUES[9] // Ace (shouldn't reach here)
    }
}

impl Default for InfiniteDeck {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of hand value calculation
#[derive(Debug, Clone, Copy)]
pub struct HandValue {
    pub total: u8,
    pub is_soft: bool,
}

/// Calculate the value of a hand
#[inline]
pub fn hand_value(cards: &[u8]) -> HandValue {
    let mut total: u16 = cards.iter().map(|&c| c as u16).sum();
    let mut aces = cards.iter().filter(|&&c| c == 11).count();

    // Convert aces from 11 to 1 as needed to avoid bust
    while total > 21 && aces > 0 {
        total -= 10;
        aces -= 1;
    }

    HandValue {
        total: total as u8,
        is_soft: aces > 0 && total <= 21,
    }
}

/// Check if hand is a natural blackjack (Ace + 10-value on first 2 cards)
#[inline]
pub fn is_blackjack(cards: &[u8]) -> bool {
    cards.len() == 2 && hand_value(cards).total == 21
}

/// Check if hand is busted (over 21)
#[inline]
pub fn is_bust(cards: &[u8]) -> bool {
    hand_value(cards).total > 21
}

/// Check if hand is a splittable pair
#[inline]
pub fn is_pair(cards: &[u8]) -> bool {
    cards.len() == 2 && cards[0] == cards[1]
}

/// Player state for strategy lookup
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerState {
    pub total: u8,
    pub dealer_upcard: u8,
    pub is_soft: bool,
    pub is_pair: bool,
}

impl PlayerState {
    pub fn new(total: u8, dealer_upcard: u8, is_soft: bool, is_pair: bool) -> Self {
        PlayerState {
            total,
            dealer_upcard,
            is_soft,
            is_pair,
        }
    }
}

/// Generate cards that create a given state
pub fn get_cards_for_state(total: u8, is_soft: bool, is_pair: bool) -> Vec<u8> {
    if is_pair {
        if is_soft {
            // A,A
            return vec![11, 11];
        } else {
            let card = total / 2;
            return vec![card, card];
        }
    }

    if is_soft {
        // Soft hands: Ace + (total - 11)
        let other = total - 11;
        return vec![11, other];
    }

    // Hard hands
    if total <= 11 {
        if total >= 4 {
            vec![2, total - 2]
        } else {
            vec![total]
        }
    } else if total <= 19 {
        vec![10, total - 10]
    } else if total == 20 {
        vec![10, 10]
    } else {
        // Hard 21 needs 3 cards
        vec![10, 10, 1]
    }
}
