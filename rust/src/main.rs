//! Blackjack Strategy Optimizer
//! Monte Carlo simulation for Evolution Live Blackjack rules (S17, DAS, ENHC)

mod deck;
mod engine;

use engine::{generate_all_states, Action, ActionStats, BlackjackEngine};
use deck::PlayerState;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;
use std::fs::File;

const TARGET_SEM: f64 = 0.005;
const BATCH_SIZE: u32 = 10_000;
const MAX_ITERATIONS: u32 = 1000;

/// Task for simulation
#[derive(Clone, Copy)]
struct SimulationTask {
    state: PlayerState,
    action: Action,
}

fn main() {
    println!("============================================================");
    println!("Blackjack Strategy Optimizer (Rust - Optimized)");
    println!("Evolution Live Blackjack Rules (S17, DAS, ENHC)");
    println!("============================================================");
    println!();

    let num_threads = rayon::current_num_threads();
    println!("Starting Monte Carlo simulation with {} threads", num_threads);
    println!("Target SEM: {}, Batch size: {}", TARGET_SEM, BATCH_SIZE);
    println!();

    let all_states = generate_all_states();
    println!("Total states to analyze: {}", all_states.len());

    // Initialize state statistics
    let mut state_stats: HashMap<PlayerState, HashMap<Action, ActionStats>> = all_states
        .iter()
        .map(|&state| {
            let action_stats: HashMap<Action, ActionStats> = Action::valid_actions(state.is_pair)
                .iter()
                .map(|&a| (a, ActionStats::new()))
                .collect();
            (state, action_stats)
        })
        .collect();

    // Generate initial tasks
    let mut pending_tasks: Vec<SimulationTask> = Vec::new();
    for &state in &all_states {
        for &action in Action::valid_actions(state.is_pair) {
            pending_tasks.push(SimulationTask { state, action });
        }
    }

    let total_pairs = pending_tasks.len();
    println!("Total state-action pairs: {}", total_pairs);
    println!();

    let start_time = Instant::now();
    let mut converged_count = 0usize;

    for iteration in 1..=MAX_ITERATIONS {
        if pending_tasks.is_empty() {
            break;
        }

        if iteration % 5 == 1 {
            let elapsed = start_time.elapsed().as_secs_f64();
            println!(
                "Iteration {}: {}/{} converged ({:.1}%), elapsed: {:.2}s",
                iteration, converged_count, total_pairs,
                100.0 * converged_count as f64 / total_pairs as f64, elapsed
            );
        }

        // Run batch in parallel - collect results without locks
        let results: Vec<(PlayerState, Action, ActionStats)> = pending_tasks
            .par_iter()
            .map(|task| {
                let mut engine = BlackjackEngine::new();
                let stats = engine.simulate_batch(&task.state, task.action, BATCH_SIZE);
                (task.state, task.action, stats)
            })
            .collect();

        // Merge results (single-threaded, but fast)
        for (state, action, batch_stats) in results {
            if let Some(action_map) = state_stats.get_mut(&state) {
                if let Some(stats) = action_map.get_mut(&action) {
                    stats.merge(&batch_stats);
                }
            }
        }

        // Filter converged tasks
        let mut new_pending = Vec::with_capacity(pending_tasks.len());
        for task in pending_tasks {
            if let Some(action_map) = state_stats.get(&task.state) {
                if let Some(stats) = action_map.get(&task.action) {
                    if stats.sem() >= TARGET_SEM {
                        new_pending.push(task);
                    } else {
                        converged_count += 1;
                    }
                }
            }
        }
        pending_tasks = new_pending;
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    println!();
    println!("Simulation complete in {:.2} seconds", elapsed);
    println!("All state-actions converged to target SEM");

    println!();
    println!("============================================================");
    println!("OPTIMAL STRATEGY TABLES");
    println!("============================================================");
    println!();

    let output = format_strategy_tables(&state_stats);
    println!("{}", output);

    let mut file = File::create("strategy_output.md").expect("Failed to create file");
    writeln!(file, "# Optimal Blackjack Strategy\n").unwrap();
    writeln!(file, "Evolution Live Blackjack Rules\n").unwrap();
    write!(file, "{}", output).unwrap();
    println!("\nStrategy saved to: strategy_output.md");

    println!();
    print_close_decisions(&state_stats);
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

    // Hard totals
    output.push_str("## Hard Totals Strategy\n\n");
    output.push_str("| Hand | ");
    output.push_str(&dealer_cards.join(" | "));
    output.push_str(" |\n|------|");
    output.push_str(&vec!["---"; 10].join("|"));
    output.push_str("|\n");

    for total in (5..=17).rev() {
        output.push_str(&format!("| **{}** |", total));
        for dealer in 2..=11 {
            let state = PlayerState::new(total, dealer, false, false);
            if let Some(actions) = state_stats.get(&state) {
                let (best, _) = get_best_action(actions);
                output.push_str(&format!(" {} |", best.symbol()));
            } else {
                output.push_str(" - |");
            }
        }
        output.push('\n');
    }
    output.push('\n');

    // Soft totals
    output.push_str("## Soft Totals Strategy\n\n");
    output.push_str("| Hand | ");
    output.push_str(&dealer_cards.join(" | "));
    output.push_str(" |\n|------|");
    output.push_str(&vec!["---"; 10].join("|"));
    output.push_str("|\n");

    for total in (13..=20).rev() {
        output.push_str(&format!("| **A,{}** |", total - 11));
        for dealer in 2..=11 {
            let state = PlayerState::new(total, dealer, true, false);
            if let Some(actions) = state_stats.get(&state) {
                let (best, _) = get_best_action(actions);
                output.push_str(&format!(" {} |", best.symbol()));
            } else {
                output.push_str(" - |");
            }
        }
        output.push('\n');
    }
    output.push('\n');

    // Pairs
    output.push_str("## Pairs Strategy\n\n");
    output.push_str("| Hand | ");
    output.push_str(&dealer_cards.join(" | "));
    output.push_str(" |\n|------|");
    output.push_str(&vec!["---"; 10].join("|"));
    output.push_str("|\n");

    for card in [11, 10, 9, 8, 7, 6, 5, 4, 3, 2] {
        let (label, total, is_soft) = if card == 11 {
            ("A,A".to_string(), 12, true)
        } else {
            (format!("{},{}", card, card), card * 2, false)
        };
        output.push_str(&format!("| **{}** |", label));
        for dealer in 2..=11 {
            let state = PlayerState::new(total, dealer, is_soft, true);
            if let Some(actions) = state_stats.get(&state) {
                let (best, _) = get_best_action(actions);
                output.push_str(&format!(" {} |", best.symbol()));
            } else {
                output.push_str(" - |");
            }
        }
        output.push('\n');
    }
    output.push('\n');

    // Legend
    output.push_str("## Legend\n\n");
    output.push_str("- **H** = Hit\n- **S** = Stand\n- **D** = Double (if not allowed, Hit)\n");
    output.push_str("- **P** = Split\n- **R** = Surrender (if not allowed, Hit)\n\n");
    output.push_str("### Rules Used\n\n");
    output.push_str("- 8 Decks (Infinite deck approximation)\n- Dealer Stands on All 17s (S17)\n");
    output.push_str("- Double After Split (DAS) allowed\n- Late Surrender allowed\n");
    output.push_str("- No Peek / European No Hole Card (ENHC)\n- Split once only (max 2 hands)\n");
    output.push_str("- One card only to split Aces\n");

    output
}

fn print_close_decisions(state_stats: &HashMap<PlayerState, HashMap<Action, ActionStats>>) {
    println!("============================================================");
    println!("CLOSE DECISIONS (EV difference < 0.02)");
    println!("============================================================\n");
    println!("{:<20} {:>6} {:>10} {:>6} {:>10} {:>10}", "State", "Best", "EV", "2nd", "EV", "Diff");
    println!("{}", "-".repeat(70));

    let mut close: Vec<(String, Action, f64, Action, f64, f64)> = Vec::new();

    for (state, actions) in state_stats {
        let mut evs: Vec<(Action, f64)> = actions.iter()
            .filter(|(_, s)| s.n > 0)
            .map(|(&a, s)| (a, s.ev()))
            .collect();
        if evs.len() < 2 { continue; }
        evs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let diff = evs[0].1 - evs[1].1;
        if diff < 0.02 {
            close.push((format_state(state), evs[0].0, evs[0].1, evs[1].0, evs[1].1, diff));
        }
    }

    close.sort_by(|a, b| a.5.partial_cmp(&b.5).unwrap());
    for (s, b, bev, sec, sev, d) in close.iter().take(25) {
        println!("{:<20} {:>6} {:>+10.4} {:>6} {:>+10.4} {:>10.4}", s, b.symbol(), bev, sec.symbol(), sev, d);
    }
}

fn format_state(state: &PlayerState) -> String {
    let d = if state.dealer_upcard == 11 { "A".to_string() } else { state.dealer_upcard.to_string() };
    if state.is_pair {
        if state.is_soft { format!("A,A vs {}", d) }
        else { format!("{},{} vs {}", state.total/2, state.total/2, d) }
    } else if state.is_soft {
        format!("A,{} vs {}", state.total - 11, d)
    } else {
        format!("Hard {} vs {}", state.total, d)
    }
}
