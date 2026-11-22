use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Standard stopwords excluding any date or numerical time tokens.
/// Date-related tokens ("november", "2026", "14", etc.) have been completely removed
/// to prevent false positives in time-sensitive evaluation (Bucket 1).
const STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractorOutput {
    pub needs_lookup: bool,
    pub query: String,
    pub entity_preserved: bool,
}

/// The exact pinned Rust contract struct for input JSONL lines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRecord {
    pub id: String,
    pub bucket: u8,
    pub question: String,
    pub wrapper_variant: String,
    pub level_k: u8,
    pub skeleton_text: String,
    pub skeleton_token_len: usize,
    pub extractor_json: ExtractorOutput,
    pub tool_called: bool,
    pub tool_query: Option<String>,
    pub tool_output: Option<String>,
    pub final_answer: Option<String>,
    pub latency_breakdown: Option<HashMap<String, f64>>,
    pub gold_query: String,
    pub gold_entity: String,
    pub expected_needs_lookup: bool,
    pub tool_returned_answer: bool,
    pub planner_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Classification {
    Success,
    TokenizerFailure,
    DecoderFailure,
    OverTrigger,
    RareCase,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SliceMetrics {
    pub total: usize,
    pub success_count: usize,
    pub tokenizer_failure_count: usize,
    pub decoder_failure_count: usize,
    pub over_trigger_count: usize,
    pub rare_case_count: usize,
    pub recovery_rate: f64,
    pub query_token_overlap_avg: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScoreReport {
    /// Flattened key "{level_k}__{wrapper_variant}" to preserve JSON serialization compatibility.
    pub slice_metrics: HashMap<String, HashMap<u8, SliceMetrics>>,
    pub overall_metrics: SliceMetrics,
    pub classification_breakdown: HashMap<String, usize>,
    pub overall_recovery_rate: f64,
}

/// Helper function to format flat slice keys for clean JSON serialization
pub fn make_slice_key(level_k: u8, wrapper: &str) -> String {
    format!("{level_k}__{wrapper}")
}

fn month_name(mm: &str) -> &str {
    match mm {
        "01" | "1" | "jan" | "january" => "january",
        "02" | "2" | "feb" | "february" => "february",
        "03" | "3" | "mar" | "march" => "march",
        "04" | "4" | "apr" | "april" => "april",
        "05" | "5" | "may" => "may",
        "06" | "6" | "jun" | "june" => "june",
        "07" | "7" | "jul" | "july" => "july",
        "08" | "8" | "aug" | "august" => "august",
        "09" | "9" | "sep" | "sept" | "september" => "september",
        "10" | "oct" | "october" => "october",
        "11" | "nov" | "november" => "november",
        "12" | "dec" | "december" => "december",
        other => other,
    }
}

/// Normalizes dates (e.g. YYYY-MM-DD -> YYYY month DD) and month names
pub fn normalize_and_expand_dates(text: &str) -> String {
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let mut expanded = String::new();
    let mut i = 0;

    // Expand YYYY-MM-DD
    while i < chars.len() {
        if i + 10 <= chars.len()
            && chars[i..i + 4].iter().all(|c| c.is_ascii_digit())
            && chars[i + 4] == '-'
            && chars[i + 5..i + 7].iter().all(|c| c.is_ascii_digit())
            && chars[i + 7] == '-'
            && chars[i + 8..i + 10].iter().all(|c| c.is_ascii_digit())
        {
            let y: String = chars[i..i + 4].iter().collect();
            let m: String = chars[i + 5..i + 7].iter().collect();
            let d_str: String = chars[i + 8..i + 10].iter().collect();
            let d = d_str.trim_start_matches('0');
            let m_name = month_name(&m);
            expanded.push_str(&format!("{} {} {} ", y, m_name, d));
            i += 10;
        } else {
            expanded.push(chars[i]);
            i += 1;
        }
    }

    let words: Vec<&str> = expanded.split_whitespace().collect();
    let mut mapped = Vec::new();
    for w in words {
        let trimmed = w.trim_matches(|c: char| !c.is_alphanumeric());
        mapped.push(month_name(trimmed).to_string());
    }
    mapped.join(" ")
}

pub fn tokenize_clean(text: &str) -> HashSet<String> {
    let stopset: HashSet<&str> = STOPWORDS.iter().cloned().collect();
    let normalized = normalize_and_expand_dates(text);
    normalized
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .filter(|t| !stopset.contains(t.as_str()))
        .collect()
}

/// Token-set recall approach.
/// Checks if text_to_check contains the entity tokens from gold_entity.
pub fn entity_match(gold_entity: &str, text_to_check: &str) -> bool {
    let gold_tokens = tokenize_clean(gold_entity);
    if gold_tokens.is_empty() {
        return false;
    }
    let target_tokens = tokenize_clean(text_to_check);

    let matched_count = gold_tokens
        .iter()
        .filter(|tok| target_tokens.contains(*tok))
        .count();

    if gold_tokens.len() <= 2 {
        matched_count == gold_tokens.len()
    } else {
        (matched_count as f64 / gold_tokens.len() as f64) >= 0.70
    }
}

pub fn calculate_query_overlap(gold_query: &str, query: &str) -> f64 {
    let gold_tokens = tokenize_clean(gold_query);
    if gold_tokens.is_empty() {
        return 0.0;
    }
    let query_tokens = tokenize_clean(query);
    let intersection = gold_tokens.intersection(&query_tokens).count();
    intersection as f64 / gold_tokens.len() as f64
}

/// Evaluates row into one of five mutually exclusive research categories:
/// - Success: Entity successfully recovered in query/skeleton and correctly routed.
/// - TokenizerFailure: NCM skeleton corrupted or produced wrong entity.
/// - DecoderFailure: Skeleton preserved entity, but extractor failed to produce a valid query matching the entity.
/// - OverTrigger: Question did not require external lookup (static fact), but pipeline triggered search anyway.
/// - RareCase: Expected external lookup, but pipeline failed to trigger search.
pub fn classify_record(entry: &LogRecord) -> Classification {
    let expected_lookup = entry.expected_needs_lookup;
    let predicted_lookup = entry.extractor_json.needs_lookup;
    let query_matched = entity_match(&entry.gold_entity, &entry.extractor_json.query);
    let skeleton_matched = entity_match(&entry.gold_entity, &entry.skeleton_text);
    let entity_preserved = entry.extractor_json.entity_preserved;

    if !expected_lookup && predicted_lookup {
        Classification::OverTrigger
    } else if expected_lookup && !predicted_lookup {
        Classification::RareCase
    } else if !skeleton_matched && !entity_preserved {
        Classification::TokenizerFailure
    } else if (skeleton_matched || entity_preserved) && !query_matched {
        Classification::DecoderFailure
    } else {
        Classification::Success
    }
}

pub fn score_file(path: &Path) -> Result<ScoreReport, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut report = ScoreReport::default();
    // Initialize all canonical classification keys with 0 to prevent downstream KeyError
    report.classification_breakdown.insert("success".to_string(), 0);
    report.classification_breakdown.insert("tokenizer_failure".to_string(), 0);
    report.classification_breakdown.insert("decoder_failure".to_string(), 0);
    report.classification_breakdown.insert("over_trigger".to_string(), 0);
    report.classification_breakdown.insert("rare_case".to_string(), 0);

    let mut total_success = 0;

    for (line_idx, line) in reader.lines().enumerate() {
        let line_content = line?;
        let trimmed = line_content.trim();
        if trimmed.is_empty() {
            continue;
        }

        let entry: LogRecord = match serde_json::from_str(trimmed) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("Error deserializing line {}: {}", line_idx + 1, err);
                return Err(Box::new(err));
            }
        };

        let classification = classify_record(&entry);
        let class_name = match classification {
            Classification::Success => "success",
            Classification::TokenizerFailure => "tokenizer_failure",
            Classification::DecoderFailure => "decoder_failure",
            Classification::OverTrigger => "over_trigger",
            Classification::RareCase => "rare_case",
        };

        *report
            .classification_breakdown
            .entry(class_name.to_string())
            .or_insert(0) += 1;

        let slice_key = make_slice_key(entry.level_k, &entry.wrapper_variant);
        let bucket = entry.bucket;

        let bucket_stat = report
            .slice_metrics
            .entry(slice_key)
            .or_default()
            .entry(bucket)
            .or_default();

        bucket_stat.total += 1;
        report.overall_metrics.total += 1;

        match classification {
            Classification::Success => {
                bucket_stat.success_count += 1;
                report.overall_metrics.success_count += 1;
                total_success += 1;
            }
            Classification::TokenizerFailure => {
                bucket_stat.tokenizer_failure_count += 1;
                report.overall_metrics.tokenizer_failure_count += 1;
            }
            Classification::DecoderFailure => {
                bucket_stat.decoder_failure_count += 1;
                report.overall_metrics.decoder_failure_count += 1;
            }
            Classification::OverTrigger => {
                bucket_stat.over_trigger_count += 1;
                report.overall_metrics.over_trigger_count += 1;
            }
            Classification::RareCase => {
                bucket_stat.rare_case_count += 1;
                report.overall_metrics.rare_case_count += 1;
            }
        }

        let overlap = calculate_query_overlap(&entry.gold_query, &entry.extractor_json.query);
        bucket_stat.query_token_overlap_avg += overlap;
        report.overall_metrics.query_token_overlap_avg += overlap;
    }

    // Finalize slice recovery rates
    for slice_map in report.slice_metrics.values_mut() {
        for stat in slice_map.values_mut() {
            if stat.total > 0 {
                stat.recovery_rate = stat.success_count as f64 / stat.total as f64;
                stat.query_token_overlap_avg /= stat.total as f64;
            }
        }
    }

    if report.overall_metrics.total > 0 {
        report.overall_metrics.query_token_overlap_avg /= report.overall_metrics.total as f64;
        report.overall_metrics.recovery_rate =
            total_success as f64 / report.overall_metrics.total as f64;
        report.overall_recovery_rate = report.overall_metrics.recovery_rate;
    }

    Ok(report)
}

fn print_help() {
    println!("ncm-probe scorer");
    println!("Usage: score [OPTIONS]");
    println!("\nOptions:");
    println!("  --input <PATH>   Path to JSONL execution logs to score (default: logs/logs.jsonl)");
    println!("  -h, --help       Print this help message");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input_path = "logs/logs.jsonl";

    let mut i = 1;
    while i < args.len() {
        if args[i] == "-h" || args[i] == "--help" {
            print_help();
            return;
        } else if args[i] == "--input" && i + 1 < args.len() {
            input_path = &args[i + 1];
            i += 1;
        } else if !args[i].starts_with('-') {
            input_path = &args[i];
        }
        i += 1;
    }

    let target_path = Path::new(input_path);
    match score_file(target_path) {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
        Err(e) => {
            eprintln!("Scorer failed: {}", e);
            std::process::exit(1);
        }
    }
}
