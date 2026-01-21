# Blackjack Strategy Optimizer

Monte Carlo simulation to find the optimal Blackjack strategy for **Evolution Live Blackjack** rules using both Python and Rust implementations.

## Rules Implemented (Evolution Gaming Standard)

| Rule | Setting |
|------|---------|
| Decks | 8 (infinite deck approximation) |
| Dealer | Stands on all 17s (S17) |
| Double Down | Any initial 2 cards |
| Double After Split | Allowed |
| Splitting | Once per hand (max 2 hands) |
| Split Aces | One card only |
| Surrender | Late surrender allowed |
| Hole Card | **No Peek (ENHC)** - dealer doesn't check for blackjack |

### ENHC (European No Hole Card) Impact

With ENHC rules, if the dealer has blackjack, the player loses their **entire wager** including doubles and splits. This significantly affects optimal strategy:

- **Don't double 11 vs 10/A** - Hit instead
- **Don't double 10 vs 10/A** - Hit instead
- **Don't split 8,8 vs 10/A** - Surrender instead
- **Don't split A,A vs A** - Hit instead

## Performance

| Implementation | Runtime | Speedup |
|----------------|---------|---------|
| Python | ~540s | 1x |
| **Rust** | **~0.27s** | **2000x** |

Both use parallel processing (16 threads) with Monte Carlo simulation converging to SEM < 0.005.

### Rust Optimizations

- `fastrand` instead of cryptographic RNG
- Fixed-size stack arrays instead of heap-allocated `Vec`
- Lookup table for O(1) card drawing
- Aggressive inlining with `#[inline(always)]`
- Lock-free parallel collection

## Quick Start

### Rust (Recommended)

```bash
cd rust
cargo run --release
```

### Python

```bash
pip install numpy
python main.py
```

## Optimal Strategy Tables

### Hard Totals

| Hand | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|------|---|---|---|---|---|---|---|---|---|---|
| **17** | S | S | S | S | S | S | S | S | S | S |
| **16** | S | S | S | S | S | H | H | R | R | R |
| **15** | S | S | S | S | S | H | H | H | R | H |
| **14** | S | S | S | S | S | H | H | H | H | H |
| **13** | S | S | S | S | S | H | H | H | H | H |
| **12** | H | H | S | S | S | H | H | H | H | H |
| **11** | D | D | D | D | D | D | D | D | H | H |
| **10** | D | D | D | D | D | D | D | D | H | H |
| **9** | H | D | D | D | D | H | H | H | H | H |
| **8** | H | H | H | H | H | H | H | H | H | H |

### Soft Totals

| Hand | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|------|---|---|---|---|---|---|---|---|---|---|
| **A,9** | S | S | S | S | S | S | S | S | S | S |
| **A,8** | S | S | S | S | S | S | S | S | S | S |
| **A,7** | D | D | D | D | D | S | S | H | H | H |
| **A,6** | D | D | D | D | D | H | H | H | H | H |
| **A,5** | H | H | D | D | D | H | H | H | H | H |
| **A,4** | H | H | D | D | D | H | H | H | H | H |
| **A,3** | H | H | H | D | D | H | H | H | H | H |
| **A,2** | H | H | H | D | D | H | H | H | H | H |

### Pairs

| Hand | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|------|---|---|---|---|---|---|---|---|---|---|
| **A,A** | P | P | P | P | P | P | P | P | P | H |
| **10,10** | S | S | S | S | S | S | S | S | S | S |
| **9,9** | P | P | P | P | P | S | P | P | S | S |
| **8,8** | P | P | P | P | P | P | P | P | R | R |
| **7,7** | P | P | P | P | P | P | H | H | H | H |
| **6,6** | P | P | P | P | P | H | H | H | H | H |
| **5,5** | D | D | D | D | D | D | D | D | H | H |
| **4,4** | H | H | H | P | P | H | H | H | H | H |
| **3,3** | P | P | P | P | P | P | H | H | H | H |
| **2,2** | H | P | P | P | P | P | H | H | H | H |

### Legend

- **H** = Hit
- **S** = Stand
- **D** = Double (if not allowed, Hit)
- **P** = Split
- **R** = Surrender (if not allowed, Hit)

## Close Decisions

These hands have very small EV differences between best and second-best actions:

| State | Best | EV | 2nd | EV | Diff |
|-------|------|-----|-----|-----|------|
| 2,2 vs 2 | H | -0.123 | P | -0.123 | 0.0004 |
| Hard 12 vs 4 | S | -0.209 | H | -0.211 | 0.0015 |
| A,7 vs 2 | D | +0.124 | S | +0.120 | 0.0038 |
| Hard 15 vs 10 | R | -0.540 | H | -0.544 | 0.0041 |
| Hard 16 vs 9 | R | -0.500 | H | -0.514 | 0.0135 |

## Project Structure

```
blackjack_solver/
├── deck.py          # Python: Card/deck management
├── engine.py        # Python: Monte Carlo simulation engine
├── main.py          # Python: Parallel runner & output
├── analyze_ev.py    # Python: EV analysis tool
└── rust/
    ├── Cargo.toml   # Rust dependencies
    └── src/
        ├── deck.rs      # Rust: Card management
        ├── engine.rs    # Rust: Simulation engine
        └── main.rs      # Rust: Parallel runner
```

## Algorithm

1. **State Space**: All combinations of (player_total, dealer_upcard, is_soft, is_pair)
2. **Actions**: Hit, Stand, Double, Split, Surrender
3. **Simulation**: For each state-action pair, run 10,000 hand batches
4. **Convergence**: Stop when Standard Error of Mean (SEM) < 0.005
5. **Result**: Select action with highest Expected Value (EV) for each state

## License

MIT
