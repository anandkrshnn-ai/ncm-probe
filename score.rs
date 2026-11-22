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

/// The exact contract struct for input JSONL lines (emitted by src/assemble_logs.py)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLogEntry {
    pub id: String,
    pub bucket: u8,
    pub wrapper_variant: String,
    pub level_k: u8,
    pub question: String,
    pub gold_query: String,
    pub gold_entity: String,
    pub skeleton_text: String,
    pub needs_lookup: bool,
    pub query: String,
    pub entity_preserved: bool,
    pub execution_response: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BucketStats {
    pub total: usize,
    pub entity_preserved_count: usize,
    pub lookup_needed_count: usize,
    pub entity_matched_count: usize,
    pub query_token_overlap_avg: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScoreReport {
    /// Flattened key "{level_k}__{wrapper_variant}" to preserve JSON serialization compatibility.
    pub slice_metrics: HashMap<String, HashMap<u8, BucketStats>>,
    pub overall_stats: BucketStats,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RateReport {
    /// Flattened key "{level_k}__{wrapper_variant}" to preserve JSON serialization compatibility.
    pub rates_by_slice: HashMap<String, f64>,
    pub overall_recovery_rate: f64,
}

/// Helper function to format flat slice keys for clean JSON serialization
pub fn make_slice_key(level_k: u8, wrapper: &str) -> String {
    format!("{level_k}__{wrapper}")
}

/// Normalizes month names (e.g., "Nov" -> "november") to prevent false negatives
/// when comparing abbreviations with full month names across gold and extracted text.
pub fn normalize_months(text: &str) -> String {
    let lower = text.to_lowercase();
    let replacements = [
        ("january", "january"),
        ("february", "february"),
        ("march", "march"),
        ("april", "april"),
        ("may", "may"),
        ("june", "june"),
        ("july", "july"),
        ("august", "august"),
        ("september", "september"),
        ("october", "october"),
        ("november", "november"),
        ("december", "december"),
        ("jan", "january"),
        ("feb", "february"),
        ("mar", "march"),
        ("apr", "april"),
        ("jun", "june"),
        ("jul", "july"),
        ("aug", "august"),
        ("sep", "september"),
        ("sept", "september"),
        ("oct", "october"),
        ("nov", "november"),
        ("dec", "december"),
    ];

    let mut result = lower;
    for (short_m, full_m) in replacements {
        let pattern = format!(r"\b{}\b", short_m);
        // Replace boundary-delimited occurrences safely
        if let Ok(re) = regex_replace(&pattern, &result, full_m) {
            result = re;
        }
    }
    result
}

fn regex_replace(pat: &str, text: &str, rep: &str) -> Result<String, ()> {
    // Basic regex-free or simple word boundary replacement
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut mapped = Vec::new();
    let clean_pat = pat.trim_start_matches(r"\b").trim_end_matches(r"\b");
    for w in words {
        let trimmed = w.trim_matches(|c: char| !c.is_alphanumeric());
        if trimmed == clean_pat {
            mapped.push(rep.to_string());
        } else {
            mapped.push(w.to_string());
        }
    }
    Ok(mapped.join(" "))
}

pub fn tokenize_clean(text: &str) -> HashSet<String> {
    let stopset: HashSet<&str> = STOPWORDS.iter().cloned().collect();
    let normalized = normalize_months(text);
    normalized
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .filter(|t| !stopset.contains(t.as_str()))
        .collect()
}

/// Token-set recall approach with normalized months to handle "Nov 14" vs "November 14".
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

    // High recall match: at least 70% of gold entity tokens or all tokens if <= 2
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

pub fn score_file(path: &Path) -> Result<(ScoreReport, RateReport), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut report = ScoreReport::default();
    let mut rate_report = RateReport::default();

    let mut slice_tallies: HashMap<String, (usize, usize)> = HashMap::new();
    let mut total_count = 0;
    let mut total_recovered = 0;

    for (line_idx, line) in reader.lines().enumerate() {
        let line_content = line?;
        let trimmed = line_content.trim();
        if trimmed.is_empty() {
            continue;
        }

        let entry: ExecutionLogEntry = match serde_json::from_str(trimmed) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("Error deserializing line {}: {}", line_idx + 1, err);
                return Err(Box::new(err));
            }
        };

        let slice_key = make_slice_key(entry.level_k, &entry.wrapper_variant);
        let bucket = entry.bucket;

        let bucket_stat = report
            .slice_metrics
            .entry(slice_key.clone())
            .or_default()
            .entry(bucket)
            .or_default();

        bucket_stat.total += 1;
        report.overall_stats.total += 1;

        if entry.entity_preserved {
            bucket_stat.entity_preserved_count += 1;
            report.overall_stats.entity_preserved_count += 1;
        }

        if entry.needs_lookup {
            bucket_stat.lookup_needed_count += 1;
            report.overall_stats.lookup_needed_count += 1;
        }

        let is_matched = entity_match(&entry.gold_entity, &entry.query)
            || entity_match(&entry.gold_entity, &entry.skeleton_text)
            || entity_match(&entry.gold_entity, &entry.execution_response);

        if is_matched {
            bucket_stat.entity_matched_count += 1;
            report.overall_stats.entity_matched_count += 1;
        }

        let overlap = calculate_query_overlap(&entry.gold_query, &entry.query);
        bucket_stat.query_token_overlap_avg += overlap;
        report.overall_stats.query_token_overlap_avg += overlap;

        let tally = slice_tallies.entry(slice_key).or_insert((0, 0));
        tally.0 += 1;
        total_count += 1;
        if is_matched || entry.entity_preserved {
            tally.1 += 1;
            total_recovered += 1;
        }
    }

    if report.overall_stats.total > 0 {
        report.overall_stats.query_token_overlap_avg /= report.overall_stats.total as f64;
    }

    for (slice_k, (count, recovered)) in slice_tallies {
        let rate = if count > 0 {
            recovered as f64 / count as f64
        } else {
            0.0
        };
        rate_report.rates_by_slice.insert(slice_k, rate);
    }

    rate_report.overall_recovery_rate = if total_count > 0 {
        total_recovered as f64 / total_count as f64
    } else {
        0.0
    };

    Ok((report, rate_report))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_path = Path::new("logs/logs.jsonl");
    let target_path = if args.len() > 1 {
        Path::new(&args[1])
    } else if Path::new("fixtures/skeletons_3row.jsonl").exists() {
        Path::new("fixtures/skeletons_3row.jsonl")
    } else {
        default_path
    };

    println!("Scoring file: {:?}", target_path);
    match score_file(target_path) {
        Ok((score_rep, rate_rep)) => {
            println!("\n=== Score Report ===");
            println!("{}", serde_json::to_string_pretty(&score_rep).unwrap());
            println!("\n=== Rate Report ===");
            println!("{}", serde_json::to_string_pretty(&rate_rep).unwrap());
        }
        Err(e) => {
            eprintln!("Scorer failed: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_match_month_normalization() {
        assert!(entity_match("NVDA 2026-11-14", "NVDA Nov 14 2026"));
        assert!(entity_match("NVDA 2026-11-14", "NVDA November 14 2026"));
    }

    #[test]
    fn test_slice_key() {
        assert_eq!(make_slice_key(2, "answer"), "2__answer");
        assert_eq!(make_slice_key(3, "passage"), "3__passage");
    }

    #[test]
    fn test_stopwords_do_not_contain_dates() {
        let date_tokens = ["november", "nov", "2026", "14", "october", "2024"];
        for d in date_tokens {
            assert!(!STOPWORDS.contains(&d), "STOPWORDS should not contain date token: {}", d);
        }
    }
}
