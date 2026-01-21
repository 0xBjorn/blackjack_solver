"""
Main entry point for Blackjack strategy optimization.
Uses multiprocessing to parallelize Monte Carlo simulations.
"""

import multiprocessing as mp
from multiprocessing import Manager
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import sys
from collections import defaultdict

from deck import hand_value
from engine import (
    BlackjackEngine, Action, ActionStats, StateStats,
    generate_all_states, get_cards_for_state, get_valid_actions
)


# Configuration
TARGET_SEM = 0.005
BATCH_SIZE = 10000
MAX_ITERATIONS = 1000  # Safety limit per state


@dataclass
class SimulationTask:
    """A task for the worker pool."""
    state: Tuple[int, int, bool, bool]  # (total, dealer_upcard, is_soft, is_pair)
    action: Action
    player_cards: List[int]
    dealer_upcard: int


@dataclass
class SimulationResult:
    """Result from a simulation batch."""
    state: Tuple[int, int, bool, bool]
    action: Action
    stats: ActionStats


def worker_simulate(task: SimulationTask) -> SimulationResult:
    """
    Worker function to simulate a batch of hands.

    Args:
        task: SimulationTask with state and action info

    Returns:
        SimulationResult with statistics
    """
    engine = BlackjackEngine(use_infinite_deck=True)
    stats = engine.simulate_batch(
        task.player_cards,
        task.dealer_upcard,
        task.action,
        BATCH_SIZE
    )
    return SimulationResult(task.state, task.action, stats)


def run_simulation(num_workers: Optional[int] = None, verbose: bool = True) -> Dict:
    """
    Run the full Monte Carlo simulation to find optimal strategy.

    Args:
        num_workers: Number of worker processes (default: CPU count)
        verbose: Print progress updates

    Returns:
        Dictionary mapping states to StateStats
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    if verbose:
        print(f"Starting Monte Carlo simulation with {num_workers} workers")
        print(f"Target SEM: {TARGET_SEM}, Batch size: {BATCH_SIZE}")
        print()

    # Generate all states
    all_states = generate_all_states()
    if verbose:
        print(f"Total states to analyze: {len(all_states)}")

    # Initialize state statistics
    state_stats: Dict[Tuple, StateStats] = {}
    for state in all_states:
        state_stats[state] = StateStats()

    # Track which state-actions need more simulation
    pending_tasks = []
    for state in all_states:
        total, dealer_upcard, is_soft, is_pair = state
        player_cards = get_cards_for_state(total, is_soft, is_pair)
        valid_actions = get_valid_actions(is_pair)

        for action in valid_actions:
            task = SimulationTask(state, action, player_cards, dealer_upcard)
            pending_tasks.append(task)

    if verbose:
        print(f"Total state-action pairs: {len(pending_tasks)}")
        print()

    # Run simulation in parallel
    start_time = time.time()
    iteration = 0
    converged_count = 0

    with mp.Pool(num_workers) as pool:
        while pending_tasks and iteration < MAX_ITERATIONS:
            iteration += 1

            if verbose and iteration % 5 == 1:
                elapsed = time.time() - start_time
                remaining = len(pending_tasks)
                total_pairs = sum(
                    len(get_valid_actions(s[3])) for s in all_states
                )
                converged = total_pairs - remaining
                print(f"Iteration {iteration}: {converged}/{total_pairs} converged "
                      f"({100*converged/total_pairs:.1f}%), "
                      f"elapsed: {elapsed:.1f}s")

            # Run batch in parallel
            results = pool.map(worker_simulate, pending_tasks)

            # Update statistics
            for result in results:
                state_stats[result.state].actions[result.action].merge(result.stats)

            # Filter out converged state-actions
            new_pending = []
            for task in pending_tasks:
                stats = state_stats[task.state].actions[task.action]
                if stats.sem() >= TARGET_SEM:
                    new_pending.append(task)

            pending_tasks = new_pending

    elapsed = time.time() - start_time

    if verbose:
        print()
        print(f"Simulation complete in {elapsed:.1f} seconds")
        print(f"Final iteration: {iteration}")

        # Check convergence
        not_converged = []
        for state, ss in state_stats.items():
            for action, stats in ss.actions.items():
                if stats.n > 0 and stats.sem() >= TARGET_SEM:
                    not_converged.append((state, action, stats.sem()))

        if not_converged:
            print(f"Warning: {len(not_converged)} state-actions did not converge")
        else:
            print("All state-actions converged to target SEM")

    return state_stats


def format_strategy_tables(state_stats: Dict) -> str:
    """
    Format the optimal strategy as Markdown tables.

    Args:
        state_stats: Dictionary mapping states to StateStats

    Returns:
        Markdown formatted strategy tables
    """
    output = []

    # Dealer upcards header
    dealer_cards = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]

    # Action symbols
    action_symbols = {
        Action.HIT: "H",
        Action.STAND: "S",
        Action.DOUBLE: "D",
        Action.SPLIT: "P",
        Action.SURRENDER: "R"
    }

    # Hard totals table
    output.append("## Hard Totals Strategy")
    output.append("")
    output.append("| Hand | " + " | ".join(dealer_cards) + " |")
    output.append("|------|" + "|".join(["---"] * 10) + "|")

    for total in range(17, 4, -1):  # 17 down to 5
        row = [f"**{total}**"]
        for dealer in range(2, 12):
            dealer_up = 11 if dealer == 11 else dealer
            state = (total, dealer_up, False, False)
            if state in state_stats:
                best_action, ev = state_stats[state].best_action()
                symbol = action_symbols[best_action]
                row.append(symbol)
            else:
                row.append("-")
        output.append("| " + " | ".join(row) + " |")

    output.append("")

    # Soft totals table
    output.append("## Soft Totals Strategy")
    output.append("")
    output.append("| Hand | " + " | ".join(dealer_cards) + " |")
    output.append("|------|" + "|".join(["---"] * 10) + "|")

    for total in range(20, 12, -1):  # A,9 down to A,2 (A,10 is blackjack)
        row = [f"**A,{total-11}**"]
        for dealer in range(2, 12):
            dealer_up = 11 if dealer == 11 else dealer
            state = (total, dealer_up, True, False)
            if state in state_stats:
                best_action, ev = state_stats[state].best_action()
                symbol = action_symbols[best_action]
                row.append(symbol)
            else:
                row.append("-")
        output.append("| " + " | ".join(row) + " |")

    output.append("")

    # Pairs table
    output.append("## Pairs Strategy")
    output.append("")
    output.append("| Hand | " + " | ".join(dealer_cards) + " |")
    output.append("|------|" + "|".join(["---"] * 10) + "|")

    pair_order = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]  # A,A down to 2,2
    for card in pair_order:
        if card == 11:
            label = "**A,A**"
            total = 12  # Soft 12 for aces
            is_soft = True
        else:
            label = f"**{card},{card}**"
            total = card * 2
            is_soft = False

        row = [label]
        for dealer in range(2, 12):
            dealer_up = 11 if dealer == 11 else dealer
            state = (total, dealer_up, is_soft, True)
            if state in state_stats:
                best_action, ev = state_stats[state].best_action()
                symbol = action_symbols[best_action]
                row.append(symbol)
            else:
                row.append("-")
        output.append("| " + " | ".join(row) + " |")

    output.append("")

    # Legend
    output.append("## Legend")
    output.append("")
    output.append("- **H** = Hit")
    output.append("- **S** = Stand")
    output.append("- **D** = Double (if not allowed, Hit)")
    output.append("- **P** = Split")
    output.append("- **R** = Surrender (if not allowed, Hit)")
    output.append("")
    output.append("### Rules Used")
    output.append("")
    output.append("- 8 Decks (Infinite deck approximation)")
    output.append("- Dealer Stands on All 17s (S17)")
    output.append("- Double After Split (DAS) allowed")
    output.append("- Late Surrender allowed")
    output.append("- No Peek / European No Hole Card (ENHC)")
    output.append("- Split once only (max 2 hands)")
    output.append("- One card only to split Aces")

    return "\n".join(output)


def print_ev_details(state_stats: Dict) -> None:
    """Print detailed EV information for each state-action."""
    print("\n## Detailed EV Analysis")
    print()

    # Hard totals
    print("### Hard Totals EV")
    print()

    for total in range(17, 4, -1):
        print(f"**Hard {total}:**")
        for dealer in [2, 7, 11]:  # Sample dealer upcards
            state = (total, dealer, False, False)
            if state in state_stats:
                ss = state_stats[state]
                dealer_str = "A" if dealer == 11 else str(dealer)
                evs = []
                for action in [Action.HIT, Action.STAND, Action.DOUBLE]:
                    stats = ss.actions[action]
                    if stats.n > 0:
                        evs.append(f"{action.value}={stats.ev():.4f}")
                print(f"  vs {dealer_str}: {', '.join(evs)}")
        print()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Blackjack Strategy Optimizer")
    print("Evolution Live Blackjack Rules (S17, DAS, ENHC)")
    print("=" * 60)
    print()

    # Run simulation
    state_stats = run_simulation(verbose=True)

    print()
    print("=" * 60)
    print("OPTIMAL STRATEGY TABLES")
    print("=" * 60)
    print()

    # Generate and print strategy tables
    tables = format_strategy_tables(state_stats)
    print(tables)

    # Save to file
    with open("strategy_output.md", "w") as f:
        f.write("# Optimal Blackjack Strategy\n\n")
        f.write("Evolution Live Blackjack Rules\n\n")
        f.write(tables)

    print()
    print("Strategy saved to: strategy_output.md")

    return state_stats


if __name__ == "__main__":
    # Freeze support for Windows multiprocessing
    mp.freeze_support()
    main()
