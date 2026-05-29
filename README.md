# flux-index-rs

Rust port of [flux-index](https://github.com/SuperInstance/flux-index) — inverted index for fast text search.

## Features

- **TF-IDF scoring** with cosine similarity ranking
- **Inverted index** with posting lists
- **Prefix search** for autocomplete
- **Fuzzy matching** for loose queries
- Zero dependencies

## Usage

```rust
use flux_index::{InvertedIndex, Document};

let mut idx = InvertedIndex::new();
idx.add(Document::new(1, "the quick brown fox"));
idx.add(Document::new(2, "the lazy brown dog"));
idx.add(Document::new(3, "a quick red fox"));

let results = idx.search("quick fox", 10);
// Returns docs ranked by cosine similarity
```

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
