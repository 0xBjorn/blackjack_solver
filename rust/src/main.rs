//! Blackjack Strategy Optimizer
//! Monte Carlo simulation for Evolution Live Blackjack rules (S17, DAS, ENHC)

mod deck;
mod engine;

use engine::{generate_all_states, Action, ActionStats, BlackjackEngine};
use deck::{get_cards_for_state, PlayerState};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use std::fs::File;

const TARGET_SEM: f64 = 0.005;
const BATCH_SIZE: u32 = 10_000;
const MAX_ITERATIONS: u32 = 1000;

/// Task for simulation
#[derive(Clone)]
struct SimulationTask {
    state: PlayerState,
    action: Action,
    player_cards: Vec<u8>,
}

/// Result from simulation
struct SimulationResult {
    state: PlayerState,
    action: Action,
    stats: ActionStats,
}

fn main() {
    println!("============================================================");
    println!("Blackjack Strategy Optimizer (Rust)");
    println!("Evolution Live Blackjack Rules (S17, DAS, ENHC)");
    println!("============================================================");
    println!();

    let num_threads = rayon::current_num_threads();
    println!("Starting Monte Carlo simulation with {} threads", num_threads);
    println!("Target SEM: {}, Batch size: {}", TARGET_SEM, BATCH_SIZE);
    println!();

    // Generate all states
    let all_states = generate_all_states();
    println!("Total states to analyze: {}", all_states.len());

    // Initialize state statistics
    let state_stats: HashMap<PlayerState, HashMap<Action, Mutex<ActionStats>>> = all_states
        .iter()
        .map(|&state| {
            let actions = Action::valid_actions(state.is_pair);
            let action_stats: HashMap<Action, Mutex<ActionStats>> = actions
                .into_iter()
                .map(|a| (a, Mutex::new(ActionStats::new())))
                .collect();
            (state, action_stats)
        })
        .collect();

    // Generate initial tasks
    let mut pending_tasks: Vec<SimulationTask> = Vec::new();
    for state in &all_states {
        let player_cards = get_cards_for_state(state.total, state.is_soft, state.is_pair);
        let valid_actions = Action::valid_actions(state.is_pair);
        for action in valid_actions {
            pending_tasks.push(SimulationTask {
                state: *state,
                action,
                player_cards: player_cards.clone(),
            });
        }
    }

    let total_pairs = pending_tasks.len();
    println!("Total state-action pairs: {}", total_pairs);
    println!();

    let start_time = Instant::now();
    let converged_count = AtomicUsize::new(0);

    for iteration in 1..=MAX_ITERATIONS {
        if pending_tasks.is_empty() {
            break;
        }

        if iteration % 5 == 1 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let conv = converged_count.load(Ordering::Relaxed);
            println!(
                "Iteration {}: {}/{} converged ({:.1}%), elapsed: {:.1}s",
                iteration,
                conv,
                total_pairs,
                100.0 * conv as f64 / total_pairs as f64,
                elapsed
            );
        }

        // Run batch in parallel
        let results: Vec<SimulationResult> = pending_tasks
            .par_iter()
            .map(|task| {
                let mut engine = BlackjackEngine::new();
                let stats = engine.simulate_batch(
                    &task.player_cards,
                    task.state.dealer_upcard,
                    task.action,
                    BATCH_SIZE,
                );
                SimulationResult {
                    state: task.state,
                    action: task.action,
                    stats,
                }
            })
            .collect();

        // Update statistics
        for result in results {
            if let Some(action_map) = state_stats.get(&result.state) {
                if let Some(stats_mutex) = action_map.get(&result.action) {
                    let mut stats = stats_mutex.lock().unwrap();
                    stats.merge(&result.stats);
                }
            }
        }

        // Filter out converged state-actions
        let mut new_pending = Vec::new();
        for task in pending_tasks {
            if let Some(action_map) = state_stats.get(&task.state) {
                if let Some(stats_mutex) = action_map.get(&task.action) {
                    let stats = stats_mutex.lock().unwrap();
                    if stats.sem() >= TARGET_SEM {
                        new_pending.push(task);
                    } else {
                        converged_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        pending_tasks = new_pending;
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    println!();
    println!("Simulation complete in {:.1} seconds", elapsed);
    println!("All state-actions converged to target SEM");

    // Convert to final format
    let final_stats: HashMap<PlayerState, HashMap<Action, ActionStats>> = state_stats
        .into_iter()
        .map(|(state, action_map)| {
            let actions: HashMap<Action, ActionStats> = action_map
                .into_iter()
                .map(|(action, mutex)| (action, mutex.into_inner().unwrap()))
                .collect();
            (state, actions)
        })
        .collect();

    // Print and save results
    println!();
    println!("============================================================");
    println!("OPTIMAL STRATEGY TABLES");
    println!("============================================================");
    println!();

    let output = format_strategy_tables(&final_stats);
    println!("{}", output);

    // Save to file
    let mut file = File::create("strategy_output.md").expect("Failed to create file");
    writeln!(file, "# Optimal Blackjack Strategy\n").unwrap();
    writeln!(file, "Evolution Live Blackjack Rules\n").unwrap();
    write!(file, "{}", output).unwrap();
    println!("\nStrategy saved to: strategy_output.md");

    // Print close decisions
    println!();
    print_close_decisions(&final_stats);
}

fn get_best_action(actions: &HashMap<Action, ActionStats>) -> (Action, f64) {
    actions
        .iter()
        .filter(|(_, stats)| stats.n > 0)
        .max_by(|(_, a), (_, b)| a.ev().partial_cmp(&b.ev()).unwrap())
        .map(|(&action, stats)| (action, stats.ev()))
        .unwrap_or((Action::Stand, f64::NEG_INFINITY))
}

fn format_strategy_tables(state_stats: &HashMap<PlayerState, HashMap<Action, ActionStats>>) -> String {
    let mut output = String::new();
    let dealer_cards = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"];

    // Hard totals table
    output.push_str("## Hard Totals Strategy\n\n");
    output.push_str("| Hand | ");
    output.push_str(&dealer_cards.join(" | "));
    output.push_str(" |\n");
    output.push_str("|------|");
    output.push_str(&vec!["---"; 10].join("|"));
    output.push_str("|\n");

    for total in (5..=17).rev() {
        output.push_str(&format!("| **{}** |", total));
        for dealer in 2..=11 {
            let state = PlayerState::new(total, dealer, false, false);
            if let Some(actions) = state_stats.get(&state) {
                let (best_action, _) = get_best_action(actions);
                output.push_str(&format!(" {} |", best_action.symbol()));
            } else {
                output.push_str(" - |");
            }
        }
        output.push('\n');
    }
    output.push('\n');

    // Soft totals table
    output.push_str("## Soft Totals Strategy\n\n");
    output.push_str("| Hand | ");
    output.push_str(&dealer_cards.join(" | "));
    output.push_str(" |\n");
    output.push_str("|------|");
    output.push_str(&vec!["---"; 10].join("|"));
    output.push_str("|\n");

    for total in (13..=20).rev() {
        let other = total - 11;
        output.push_str(&format!("| **A,{}** |", other));
        for dealer in 2..=11 {
            let state = PlayerState::new(total, dealer, true, false);
            if let Some(actions) = state_stats.get(&state) {
                let (best_action, _) = get_best_action(actions);
                output.push_str(&format!(" {} |", best_action.symbol()));
            } else {
                output.push_str(" - |");
            }
        }
        output.push('\n');
    }
    output.push('\n');

    // Pairs table
    output.push_str("## Pairs Strategy\n\n");
    output.push_str("| Hand | ");
    output.push_str(&dealer_cards.join(" | "));
    output.push_str(" |\n");
    output.push_str("|------|");
    output.push_str(&vec!["---"; 10].join("|"));
    output.push_str("|\n");

    let pair_order = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2];
    for card in pair_order {
        let (label, total, is_soft) = if card == 11 {
            ("A,A".to_string(), 12, true)
        } else {
            (format!("{},{}", card, card), card * 2, false)
        };
        output.push_str(&format!("| **{}** |", label));
        for dealer in 2..=11 {
            let state = PlayerState::new(total, dealer, is_soft, true);
            if let Some(actions) = state_stats.get(&state) {
                let (best_action, _) = get_best_action(actions);
                output.push_str(&format!(" {} |", best_action.symbol()));
            } else {
                output.push_str(" - |");
            }
        }
        output.push('\n');
    }
    output.push('\n');

    // Legend
    output.push_str("## Legend\n\n");
    output.push_str("- **H** = Hit\n");
    output.push_str("- **S** = Stand\n");
    output.push_str("- **D** = Double (if not allowed, Hit)\n");
    output.push_str("- **P** = Split\n");
    output.push_str("- **R** = Surrender (if not allowed, Hit)\n\n");
    output.push_str("### Rules Used\n\n");
    output.push_str("- 8 Decks (Infinite deck approximation)\n");
    output.push_str("- Dealer Stands on All 17s (S17)\n");
    output.push_str("- Double After Split (DAS) allowed\n");
    output.push_str("- Late Surrender allowed\n");
    output.push_str("- No Peek / European No Hole Card (ENHC)\n");
    output.push_str("- Split once only (max 2 hands)\n");
    output.push_str("- One card only to split Aces\n");

    output
}

fn print_close_decisions(state_stats: &HashMap<PlayerState, HashMap<Action, ActionStats>>) {
    println!("============================================================");
    println!("CLOSE DECISIONS (EV difference < 0.02)");
    println!("============================================================");
    println!();
    println!("{:<20} {:>6} {:>10} {:>6} {:>10} {:>10}",
             "State", "Best", "EV", "2nd", "EV", "Diff");
    println!("{}", "-".repeat(70));

    let mut close_decisions: Vec<(String, Action, f64, Action, f64, f64)> = Vec::new();

    for (state, actions) in state_stats {
        let mut evs: Vec<(Action, f64)> = actions
            .iter()
            .filter(|(_, stats)| stats.n > 0)
            .map(|(&action, stats)| (action, stats.ev()))
            .collect();

        if evs.len() < 2 {
            continue;
        }

        evs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let (best_action, best_ev) = evs[0];
        let (second_action, second_ev) = evs[1];
        let diff = best_ev - second_ev;

        if diff < 0.02 {
            let state_str = format_state(state);
            close_decisions.push((state_str, best_action, best_ev, second_action, second_ev, diff));
        }
    }

    close_decisions.sort_by(|a, b| a.5.partial_cmp(&b.5).unwrap());

    for (state_str, best, best_ev, second, second_ev, diff) in close_decisions.iter().take(30) {
        println!(
            "{:<20} {:>6} {:>+10.4} {:>6} {:>+10.4} {:>10.4}",
            state_str,
            best.symbol(),
            best_ev,
            second.symbol(),
            second_ev,
            diff
        );
    }
}

fn format_state(state: &PlayerState) -> String {
    let dealer_str = if state.dealer_upcard == 11 { "A" } else { &state.dealer_upcard.to_string() };

    if state.is_pair {
        if state.is_soft {
            format!("A,A vs {}", dealer_str)
        } else {
            let card = state.total / 2;
            format!("{},{} vs {}", card, card, dealer_str)
        }
    } else if state.is_soft {
        let other = state.total - 11;
        format!("A,{} vs {}", other, dealer_str)
    } else {
        format!("Hard {} vs {}", state.total, dealer_str)
    }
}
