# flux-index-rs

Inverted index for fast text search with TF-IDF scoring, cosine similarity ranking, prefix autocomplete, and fuzzy matching.

## What This Gives You

- **Inverted index** — Posting lists for O(1) term lookup
- **TF-IDF scoring** — Term frequency × inverse document frequency ranking
- **Cosine similarity** — Vector-space ranking for query relevance
- **Prefix search** — Autocomplete-style prefix matching
- **Fuzzy matching** — Approximate string matching for typo tolerance
- **Zero dependencies** — Pure Rust, `std` + `HashMap` only

## Quick Start

```rust
use flux_index::{InvertedIndex, Document};

let mut idx = InvertedIndex::new();
idx.add(Document::new(1, "the quick brown fox"));
idx.add(Document::new(2, "the lazy brown dog"));
idx.add(Document::new(3, "a quick red fox"));

// TF-IDF search with cosine similarity ranking
let results = idx.search("quick fox", 10);
for hit in &results {
    println!("doc {} score: {:.4}", hit.id, hit.score);
}

// Prefix autocomplete
let suggestions = idx.prefix_search("bro", 5);
// → ["brown"]

// Fuzzy matching
let fuzzy = idx.fuzzy_search("quikc", 1, 5);  // max edit distance 1
// → matches "quick"
```

## API Reference

### `InvertedIndex`

| Method | Description |
|--------|-------------|
| `new()` | Create empty index |
| `add(document)` | Index a document |
| `search(query, limit)` | TF-IDF cosine similarity search |
| `prefix_search(prefix, limit)` | Autocomplete prefix matching |
| `fuzzy_search(term, max_dist, limit)` | Approximate matching |

### `Document`

```rust
Document::new(id, text)    // Create document with ID and text content
```

### `SearchResult`

| Field | Description |
|-------|-------------|
| `id` | Document ID |
| `score` | Cosine similarity score |
| `terms_matched` | Number of query terms found |

## How It Fits

- **[caching-service-rs](https://github.com/SuperInstance/caching-service-rs)** — Cache frequent search results for sub-millisecond latency
- **[cocapn-health-rs](https://github.com/SuperInstance/cocapn-health-rs)** — Search service logs and health reports
- **[constraint-dsl](https://github.com/SuperInstance/constraint-dsl)** — Index constraint definitions for pipeline lookup
- **[flux-genome](https://github.com/SuperInstance/flux-genome)** — Search musical tradition genomes by feature similarity

## Testing

7 tests covering indexing, TF-IDF scoring, cosine similarity ranking, prefix search, and fuzzy matching.

```bash
cargo test
```

## Installation

```toml
[dependencies]
flux-index = { git = "https://github.com/SuperInstance/flux-index-rs" }
```

```bash
git clone https://github.com/SuperInstance/flux-index-rs.git
cd flux-index-rs
cargo build
```

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance) ecosystem.
