//! Inverted index for fast text search with TF-IDF scoring.

use std::collections::{HashMap, HashSet};

/// A document in the index.
#[derive(Debug, Clone)]
pub struct Document {
    pub id: u64,
    pub text: String,
    pub tokens: Vec<String>,
}

impl Document {
    pub fn new(id: u64, text: &str) -> Self {
        let tokens = tokenize(text);
        Self {
            id,
            text: text.to_string(),
            tokens,
        }
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Posting list entry: (doc_id, term_frequency).
type PostingList = Vec<(u64, usize)>;

/// Inverted index with TF-IDF scoring.
#[derive(Debug, Clone)]
pub struct InvertedIndex {
    postings: HashMap<String, PostingList>,
    docs: HashMap<u64, Document>,
    doc_lengths: HashMap<u64, usize>,
    total_docs: usize,
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            docs: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_docs: 0,
        }
    }

    /// Add a document to the index.
    pub fn add(&mut self, doc: Document) {
        let id = doc.id;
        let len = doc.tokens.len();
        self.doc_lengths.insert(id, len);
        self.total_docs += 1;

        let mut term_counts: HashMap<String, usize> = HashMap::new();
        for token in &doc.tokens {
            *term_counts.entry(token.clone()).or_insert(0) += 1;
        }

        for (term, count) in &term_counts {
            self.postings
                .entry(term.clone())
                .or_default()
                .push((id, *count));
        }

        self.docs.insert(id, doc);
    }

    /// Remove a document by ID.
    pub fn remove(&mut self, doc_id: u64) -> bool {
        if self.docs.remove(&doc_id).is_none() {
            return false;
        }
        self.doc_lengths.remove(&doc_id);
        self.total_docs = self.total_docs.saturating_sub(1);

        for posting in self.postings.values_mut() {
            posting.retain(|(id, _)| *id != doc_id);
        }
        self.postings.retain(|_, v| !v.is_empty());
        true
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.total_docs
    }

    pub fn is_empty(&self) -> bool {
        self.total_docs == 0
    }

    /// Get all terms in the index.
    pub fn terms(&self) -> Vec<&str> {
        self.postings.keys().map(|s| s.as_str()).collect()
    }

    /// IDF (inverse document frequency) for a term.
    pub fn idf(&self, term: &str) -> f64 {
        let df = self.postings.get(term).map(|p| p.len()).unwrap_or(0) as f64;
        if df == 0.0 {
            return 0.0;
        }
        (1.0 + self.total_docs as f64 / (1.0 + df)).ln() + 1.0
    }

    /// TF-IDF score for a (term, doc) pair.
    pub fn tfidf(&self, term: &str, doc_id: u64) -> f64 {
        let tf = self
            .postings
            .get(term)
            .and_then(|p| {
                p.iter()
                    .find(|(id, _)| *id == doc_id)
                    .map(|(_, c)| *c as f64)
            })
            .unwrap_or(0.0);
        if tf == 0.0 {
            return 0.0;
        }
        let idf = self.idf(term);
        tf * idf
    }

    /// Search for documents matching all query terms, ranked by cosine similarity.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(u64, f64)> {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return vec![];
        }

        // Find candidate docs (containing at least one query term)
        let mut candidates: HashSet<u64> = HashSet::new();
        for term in &query_tokens {
            if let Some(posting) = self.postings.get(term) {
                for &(id, _) in posting {
                    candidates.insert(id);
                }
            }
        }

        // Score each candidate
        let query_vec: HashMap<&str, f64> =
            query_tokens.iter().map(|t| (t.as_str(), 1.0)).collect();
        let mut scored: Vec<(u64, f64)> = candidates
            .iter()
            .map(|&doc_id| {
                let score = self.cosine_similarity(&query_vec, doc_id);
                (doc_id, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(limit);
        scored
    }

    fn cosine_similarity(&self, query_vec: &HashMap<&str, f64>, doc_id: u64) -> f64 {
        let mut dot = 0.0;
        let mut q_norm = 0.0;
        let mut d_norm = 0.0;

        for (term, &q_weight) in query_vec {
            let d_weight = self.tfidf(term, doc_id);
            dot += q_weight * d_weight;
            q_norm += q_weight * q_weight;
        }

        // Compute doc norm over all terms in the doc
        if let Some(doc) = self.docs.get(&doc_id) {
            let unique: HashSet<&str> = doc.tokens.iter().map(|s| s.as_str()).collect();
            for term in &unique {
                let w = self.tfidf(term, doc_id);
                d_norm += w * w;
            }
        }

        let denom = q_norm.sqrt() * d_norm.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            dot / denom
        }
    }

    /// Prefix search: find all terms starting with the given prefix.
    pub fn prefix_search(&self, prefix: &str) -> Vec<&str> {
        self.postings
            .keys()
            .filter(|t| t.starts_with(prefix))
            .map(|s| s.as_str())
            .collect()
    }

    /// Get the posting list for a term.
    pub fn get_postings(&self, term: &str) -> Option<&[(u64, usize)]> {
        self.postings.get(term).map(|v| v.as_slice())
    }

    /// Fuzzy term frequency: count how many query tokens appear in a doc.
    pub fn fuzzy_match(&self, query: &str, doc_id: u64) -> usize {
        let query_tokens = tokenize(query);
        let doc_terms: HashSet<&str> = self
            .docs
            .get(&doc_id)
            .map(|d| d.tokens.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default();
        query_tokens
            .iter()
            .filter(|t| doc_terms.contains(t.as_str()))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_search() {
        let mut idx = InvertedIndex::new();
        idx.add(Document::new(1, "hello world"));
        idx.add(Document::new(2, "hello rust"));
        idx.add(Document::new(3, "world of rust"));

        let results = idx.search("hello", 10);
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 2));
        assert!(!results.iter().any(|(id, _)| *id == 3));
    }

    #[test]
    fn test_tfidf_scoring() {
        let mut idx = InvertedIndex::new();
        idx.add(Document::new(1, "rust rust rust"));
        idx.add(Document::new(2, "rust once"));

        // Doc 1 has more "rust" terms → higher TF → higher score
        let tf1 = idx.tfidf("rust", 1);
        let tf2 = idx.tfidf("rust", 2);
        assert!(tf1 > tf2);
    }

    #[test]
    fn test_remove_document() {
        let mut idx = InvertedIndex::new();
        idx.add(Document::new(1, "hello world"));
        assert!(idx.remove(1));
        assert!(idx.is_empty());
        let results = idx.search("hello", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_prefix_search() {
        let mut idx = InvertedIndex::new();
        idx.add(Document::new(1, "programming program programmer"));
        let results = idx.prefix_search("prog");
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_idf_rare_term_higher() {
        let mut idx = InvertedIndex::new();
        idx.add(Document::new(1, "common word"));
        idx.add(Document::new(2, "common rare"));
        idx.add(Document::new(3, "common thing"));

        let common_idf = idx.idf("common");
        let rare_idf = idx.idf("rare");
        assert!(rare_idf > common_idf);
    }

    #[test]
    fn test_empty_search() {
        let idx = InvertedIndex::new();
        assert!(idx.search("hello", 10).is_empty());
    }

    #[test]
    fn test_fuzzy_match() {
        let mut idx = InvertedIndex::new();
        idx.add(Document::new(1, "hello world"));
        let count = idx.fuzzy_match("hello world test", 1);
        assert_eq!(count, 2); // "hello" and "world" match
    }
}
