from __future__ import annotations

import argparse
from pathlib import Path

from .numeric_extractor import format_numeric_expressions
from .searcher import HybridSearcher
from .numeric_extractor import debug_numeric_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("parsed_data/search_index"),
        help="Папка поискового индекса",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Показывать полный текст chunk, а не короткий snippet",
    )
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=700,
        help="Сколько символов показывать в коротком режиме",
    )

    parser.add_argument(
        "--show-query-plan",
        action="store_true",
        help="Показать перевод запроса, dense_query, bm25_query и найденные синонимы",
    )

    args = parser.parse_args()

    searcher = HybridSearcher(index_dir=args.index_dir)
    results = searcher.search(args.query, top_k=args.top_k)

    print()
    print(f"INDEX: {args.index_dir}")
    print(f"QUERY: {args.query}")
    print("=" * 100)

    if args.show_query_plan:
        query_plan = searcher.get_query_plan(args.query)

        print()
        print("QUERY PLAN:")
        print(f"original_query: {query_plan['original_query']}")
        print(f"is_english_query: {query_plan['is_english_query']}")
        print(f"translated_query: {query_plan['translated_query']}")
        print(f"dense_query: {query_plan['dense_query']}")
        print(f"bm25_query: {query_plan['bm25_query']}")

        print()
        print("MATCHED SYNONYMS:")
        for match in query_plan["matched_synonyms"]:
            print(
                f"- {match['canonical_name']} "
                f"({match['canonical_id']}): "
                f"{', '.join(match['matched_aliases'])}"
            )

    for result in results:
        print()
        print(f"[{result['rank']}] {result['title']}")
        print(f"chunk_id: {result['chunk_id']}")
        print(
            f"score: fused={result['fused_score']:.4f} | "
            f"dense={result['dense_score']:.4f} | "
            f"bm25={result['bm25_score']:.4f}"
        )
        print(f"chars: {result['char_start']}–{result['char_end']}")
        print(f"numbers: {format_numeric_expressions(result.get('numeric_expressions', []))}")

        #print("numeric candidates:")
        #for row in debug_numeric_candidates(result["text"]):
        #    print(f"  {row}")
        #print("-" * 100)

        if args.full_text:
            print(result["text"])
        else:
            text = result["text"].replace("\n", " ")
            text = " ".join(text.split())

            if len(text) > args.snippet_chars:
                text = text[:args.snippet_chars].rstrip() + "..."

            print(text)


if __name__ == "__main__":
    main()