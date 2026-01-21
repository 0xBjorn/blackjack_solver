"""
Analyze close EV decisions from the Blackjack simulation.
"""

from typing import Dict, List, Tuple
from engine import Action, ActionStats, StateStats, get_valid_actions
from main import run_simulation


def format_state(state: Tuple[int, int, bool, bool]) -> str:
    """Format a state tuple as a readable string."""
    total, dealer, is_soft, is_pair = state
    dealer_str = "A" if dealer == 11 else str(dealer)

    if is_pair:
        if is_soft:  # A,A
            return f"A,A vs {dealer_str}"
        else:
            card = total // 2
            return f"{card},{card} vs {dealer_str}"
    elif is_soft:
        other = total - 11
        return f"A,{other} vs {dealer_str}"
    else:
        return f"Hard {total} vs {dealer_str}"


def analyze_close_decisions(state_stats: Dict, threshold: float = 0.02) -> List[dict]:
    """
    Find decisions where EV difference between best and second-best is small.

    Args:
        state_stats: Dictionary of state -> StateStats
        threshold: Maximum EV difference to consider "close"

    Returns:
        List of close decision details
    """
    close_decisions = []

    for state, ss in state_stats.items():
        total, dealer, is_soft, is_pair = state

        # Get valid actions for this state
        valid_actions = get_valid_actions(is_pair)

        # Get EVs for valid actions only
        evs = []
        for action in valid_actions:
            stats = ss.actions[action]
            if stats.n > 0:
                evs.append((action, stats.ev(), stats.sem(), stats.n))

        if len(evs) < 2:
            continue

        # Sort by EV descending
        evs.sort(key=lambda x: x[1], reverse=True)

        best_action, best_ev, best_sem, best_n = evs[0]
        second_action, second_ev, second_sem, second_n = evs[1]

        ev_diff = best_ev - second_ev

        if ev_diff < threshold:
            close_decisions.append({
                'state': state,
                'state_str': format_state(state),
                'best': best_action,
                'best_ev': best_ev,
                'best_sem': best_sem,
                'second': second_action,
                'second_ev': second_ev,
                'second_sem': second_sem,
                'ev_diff': ev_diff,
                'all_evs': evs
            })

    return close_decisions


def print_close_decisions(close_decisions: List[dict], max_show: int = 50) -> None:
    """Print close decisions in a formatted table."""

    # Sort by EV difference
    close_decisions.sort(key=lambda x: x['ev_diff'])

    print("=" * 90)
    print("CLOSE DECISIONS (EV difference < 0.02)")
    print("=" * 90)
    print()
    print(f"{'State':<20} {'Best':>6} {'EV':>8} {'2nd':>6} {'EV':>8} {'Diff':>8} {'All EVs'}")
    print("-" * 90)

    for d in close_decisions[:max_show]:
        all_evs_str = " | ".join([f"{a.value}:{ev:+.4f}" for a, ev, sem, n in d['all_evs']])
        print(f"{d['state_str']:<20} {d['best'].value:>6} {d['best_ev']:>+8.4f} "
              f"{d['second'].value:>6} {d['second_ev']:>+8.4f} {d['ev_diff']:>8.4f}   {all_evs_str}")

    if len(close_decisions) > max_show:
        print(f"\n... and {len(close_decisions) - max_show} more close decisions")


def print_all_ev_table(state_stats: Dict) -> None:
    """Print complete EV table for all states."""

    dealer_cards = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]

    print("\n" + "=" * 100)
    print("COMPLETE EV TABLE - HARD TOTALS")
    print("=" * 100)

    for total in range(17, 4, -1):
        print(f"\n--- Hard {total} ---")
        print(f"{'Dealer':<8} {'Hit':>10} {'Stand':>10} {'Double':>10} {'Surr':>10} {'Best':>8}")
        print("-" * 62)

        for dealer in range(2, 12):
            dealer_str = "A" if dealer == 11 else str(dealer)
            state = (total, dealer, False, False)

            if state in state_stats:
                ss = state_stats[state]
                h_ev = ss.actions[Action.HIT].ev()
                s_ev = ss.actions[Action.STAND].ev()
                d_ev = ss.actions[Action.DOUBLE].ev()
                r_ev = ss.actions[Action.SURRENDER].ev()

                best = max([(Action.HIT, h_ev), (Action.STAND, s_ev),
                           (Action.DOUBLE, d_ev), (Action.SURRENDER, r_ev)],
                          key=lambda x: x[1] if x[1] != float('-inf') else -999)

                h_str = f"{h_ev:+.4f}" if h_ev != float('-inf') else "N/A"
                s_str = f"{s_ev:+.4f}" if s_ev != float('-inf') else "N/A"
                d_str = f"{d_ev:+.4f}" if d_ev != float('-inf') else "N/A"
                r_str = f"{r_ev:+.4f}" if r_ev != float('-inf') else "N/A"

                print(f"{dealer_str:<8} {h_str:>10} {s_str:>10} {d_str:>10} {r_str:>10} {best[0].value:>8}")

    print("\n" + "=" * 100)
    print("COMPLETE EV TABLE - SOFT TOTALS")
    print("=" * 100)

    for total in range(20, 12, -1):  # A,9 down to A,2 (A,10 is blackjack)
        other = total - 11
        print(f"\n--- Soft {total} (A,{other}) ---")
        print(f"{'Dealer':<8} {'Hit':>10} {'Stand':>10} {'Double':>10} {'Surr':>10} {'Best':>8}")
        print("-" * 62)

        for dealer in range(2, 12):
            dealer_str = "A" if dealer == 11 else str(dealer)
            state = (total, dealer, True, False)

            if state in state_stats:
                ss = state_stats[state]
                h_ev = ss.actions[Action.HIT].ev()
                s_ev = ss.actions[Action.STAND].ev()
                d_ev = ss.actions[Action.DOUBLE].ev()
                r_ev = ss.actions[Action.SURRENDER].ev()

                best = max([(Action.HIT, h_ev), (Action.STAND, s_ev),
                           (Action.DOUBLE, d_ev), (Action.SURRENDER, r_ev)],
                          key=lambda x: x[1] if x[1] != float('-inf') else -999)

                h_str = f"{h_ev:+.4f}" if h_ev != float('-inf') else "N/A"
                s_str = f"{s_ev:+.4f}" if s_ev != float('-inf') else "N/A"
                d_str = f"{d_ev:+.4f}" if d_ev != float('-inf') else "N/A"
                r_str = f"{r_ev:+.4f}" if r_ev != float('-inf') else "N/A"

                print(f"{dealer_str:<8} {h_str:>10} {s_str:>10} {d_str:>10} {r_str:>10} {best[0].value:>8}")

    print("\n" + "=" * 100)
    print("COMPLETE EV TABLE - PAIRS")
    print("=" * 100)

    pair_order = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    for card in pair_order:
        if card == 11:
            label = "A,A"
            total = 12
            is_soft = True
        else:
            label = f"{card},{card}"
            total = card * 2
            is_soft = False

        print(f"\n--- {label} ---")
        print(f"{'Dealer':<8} {'Hit':>10} {'Stand':>10} {'Double':>10} {'Split':>10} {'Surr':>10} {'Best':>8}")
        print("-" * 74)

        for dealer in range(2, 12):
            dealer_str = "A" if dealer == 11 else str(dealer)
            state = (total, dealer, is_soft, True)

            if state in state_stats:
                ss = state_stats[state]
                h_ev = ss.actions[Action.HIT].ev()
                s_ev = ss.actions[Action.STAND].ev()
                d_ev = ss.actions[Action.DOUBLE].ev()
                p_ev = ss.actions[Action.SPLIT].ev()
                r_ev = ss.actions[Action.SURRENDER].ev()

                evs = [(Action.HIT, h_ev), (Action.STAND, s_ev),
                       (Action.DOUBLE, d_ev), (Action.SPLIT, p_ev),
                       (Action.SURRENDER, r_ev)]
                best = max(evs, key=lambda x: x[1] if x[1] != float('-inf') else -999)

                h_str = f"{h_ev:+.4f}" if h_ev != float('-inf') else "N/A"
                s_str = f"{s_ev:+.4f}" if s_ev != float('-inf') else "N/A"
                d_str = f"{d_ev:+.4f}" if d_ev != float('-inf') else "N/A"
                p_str = f"{p_ev:+.4f}" if p_ev != float('-inf') else "N/A"
                r_str = f"{r_ev:+.4f}" if r_ev != float('-inf') else "N/A"

                print(f"{dealer_str:<8} {h_str:>10} {s_str:>10} {d_str:>10} {p_str:>10} {r_str:>10} {best[0].value:>8}")


def main():
    """Run analysis."""
    import multiprocessing as mp
    mp.freeze_support()

    print("Running simulation to gather EV data...")
    print()

    state_stats = run_simulation(verbose=True)

    # Find and print close decisions
    close = analyze_close_decisions(state_stats, threshold=0.02)
    print()
    print_close_decisions(close)

    # Print full EV tables
    print_all_ev_table(state_stats)


if __name__ == "__main__":
    main()
