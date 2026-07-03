from __future__ import annotations

from collections import defaultdict

from .models import GraphReasoningContext


class GraphContextBuilder:
    def build_text_context(self, context: GraphReasoningContext, max_edges: int = 40) -> str:
        node_by_id = context.node_by_id()
        lines: list[str] = []

        lines.append("Graph context")
        lines.append(f"Seed entities: {', '.join(context.seed_entities) or 'none'}")

        if context.edges:
            lines.append("Facts:")
            for edge in sorted(context.edges, key=lambda item: item.confidence, reverse=True)[:max_edges]:
                src = node_by_id.get(edge.source_id)
                tgt = node_by_id.get(edge.target_id)
                src_name = src.name if src else edge.source_id
                tgt_name = tgt.name if tgt else edge.target_id
                quote = f" Quote: {edge.quote}" if edge.quote else ""
                doc = f" Source: {edge.source_document}" if edge.source_document else ""
                lines.append(
                    f"- {src_name} --{edge.relation_type}--> {tgt_name} "
                    f"(confidence={edge.confidence:.2f}).{doc}{quote}"
                )

        grouped = defaultdict(list)
        for gap in context.gaps:
            grouped[gap.severity].append(gap.message)
        if grouped:
            lines.append("Knowledge gaps:")
            for severity, messages in sorted(grouped.items()):
                for message in messages:
                    lines.append(f"- [{severity}] {message}")

        if context.contradictions:
            lines.append("Contradictions:")
            for edge in context.contradictions:
                src = node_by_id.get(edge.source_id)
                tgt = node_by_id.get(edge.target_id)
                src_name = src.name if src else edge.source_id
                tgt_name = tgt.name if tgt else edge.target_id
                lines.append(f"- {src_name} contradicts {tgt_name}: {edge.quote}")

        return "\n".join(lines)
