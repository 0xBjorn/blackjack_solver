//! Deck/Shoe management for Blackjack simulation.
//! Optimized for speed with fixed-size arrays and fast RNG.

use fastrand::Rng;

/// Maximum cards in a hand (5 cards + safety margin)
pub const MAX_HAND_SIZE: usize = 12;

/// Fixed-size hand to avoid heap allocations
#[derive(Clone, Copy)]
pub struct Hand {
    cards: [u8; MAX_HAND_SIZE],
    len: u8,
}

impl Hand {
    #[inline(always)]
    pub fn new() -> Self {
        Hand {
            cards: [0; MAX_HAND_SIZE],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn from_cards(c1: u8, c2: u8) -> Self {
        let mut h = Hand::new();
        h.cards[0] = c1;
        h.cards[1] = c2;
        h.len = 2;
        h
    }

    #[inline(always)]
    pub fn push(&mut self, card: u8) {
        self.cards[self.len as usize] = card;
        self.len += 1;
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline(always)]
    pub fn cards(&self) -> &[u8] {
        &self.cards[..self.len as usize]
    }

    #[inline(always)]
    pub fn first(&self) -> u8 {
        self.cards[0]
    }

    #[inline(always)]
    pub fn second(&self) -> u8 {
        self.cards[1]
    }
}

impl Default for Hand {
    fn default() -> Self {
        Self::new()
    }
}

/// Infinite deck with fast RNG
/// Uses lookup table for O(1) card drawing
pub struct InfiniteDeck {
    rng: Rng,
}

// Lookup table: maps random value 0-12 to card value
// 0-7 -> 2-9, 8-11 -> 10, 12 -> 11 (Ace)
const CARD_LOOKUP: [u8; 13] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11];

impl InfiniteDeck {
    #[inline(always)]
    pub fn new() -> Self {
        InfiniteDeck {
            rng: Rng::new(),
        }
    }

    /// Draw a random card - O(1) with lookup table
    #[inline(always)]
    pub fn draw(&mut self) -> u8 {
        CARD_LOOKUP[self.rng.usize(0..13)]
    }
}

impl Default for InfiniteDeck {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate hand value - optimized with early exit
#[inline(always)]
pub fn hand_value(hand: &Hand) -> (u8, bool) {
    let cards = hand.cards();
    let mut total: u16 = 0;
    let mut aces: u8 = 0;

    for &card in cards {
        total += card as u16;
        aces += (card == 11) as u8;
    }

    // Convert aces from 11 to 1 as needed
    while total > 21 && aces > 0 {
        total -= 10;
        aces -= 1;
    }

    (total as u8, aces > 0)
}

/// Check if hand is a natural blackjack
#[inline(always)]
pub fn is_blackjack(hand: &Hand) -> bool {
    hand.len() == 2 && {
        let (total, _) = hand_value(hand);
        total == 21
    }
}

/// Check if hand is busted
#[inline(always)]
pub fn is_bust(hand: &Hand) -> bool {
    let (total, _) = hand_value(hand);
    total > 21
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
    #[inline(always)]
    pub fn new(total: u8, dealer_upcard: u8, is_soft: bool, is_pair: bool) -> Self {
        PlayerState { total, dealer_upcard, is_soft, is_pair }
    }
}

/// Generate starting hand for a state
#[inline(always)]
pub fn get_hand_for_state(total: u8, is_soft: bool, is_pair: bool) -> Hand {
    if is_pair {
        if is_soft {
            Hand::from_cards(11, 11) // A,A
        } else {
            let card = total / 2;
            Hand::from_cards(card, card)
        }
    } else if is_soft {
        Hand::from_cards(11, total - 11)
    } else if total <= 11 {
        if total >= 4 {
            Hand::from_cards(2, total - 2)
        } else {
            Hand::from_cards(total, 0) // edge case
        }
    } else if total <= 19 {
        Hand::from_cards(10, total - 10)
    } else {
        Hand::from_cards(10, 10) // 20
    }
}
