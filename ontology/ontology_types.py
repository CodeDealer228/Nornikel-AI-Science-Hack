"""Ontology types and labels for the Nornikel Knowledge Graph.

The ontology is the source of truth for:

* **EntityType** — the conceptual types a node can have (Material,
  Process, Equipment, …). Used by the NER/RE prompt and by the
  Python typing throughout the pipeline.
* **RelationType** — the conceptual relationship types (uses_material,
  operates_at_condition, …). Same dual use.
* **EntityLabel** — the Neo4j label used in the graph. **One
  EntityType maps to one EntityLabel** so the graph has proper
  per-type labels (not a flat ``(e:Entity {type})``).
* **NodeLabel** — auxiliary node labels (Document, Chunk, etc.).
* **RelationshipLabel** — auxiliary relationship labels.

Adding a new type
-----------------
1. Add the value to ``EntityType`` (StrEnum).
2. Add the same value to ``EntityLabel`` (the Neo4j label is
   identical to the conceptual type by convention).
3. Optionally add a corresponding relation type to ``RelationType``.
4. Re-run ingestion — the loader will create the new label and
   indexes automatically.
"""

from enum import StrEnum


class EntityType(StrEnum):
    MATERIAL = "Material"
    SUBSTANCE = "Substance"
    PROCESS = "Process"
    EQUIPMENT = "Equipment"
    PROPERTY = "Property"
    PARAMETER = "Parameter"
    CONDITION = "Condition"
    EXPERIMENT = "Experiment"
    PUBLICATION = "Publication"
    TECHNOLOGY_SOLUTION = "TechnologySolution"
    RESULT = "Result"
    CONCLUSION = "Conclusion"
    LIMITATION = "Limitation"
    FACILITY = "Facility"
    ORGANIZATION = "Organization"
    EXPERT = "Expert"
    # New (post-flat-schema): geographic, temporal, and numeric anchors.
    GEOGRAPHY = "Geography"
    YEAR = "Year"
    NUMERIC_VALUE = "NumericValue"


class RelationType(StrEnum):
    HAS_SUBPROCESS = "has_subprocess"
    REPLACED_BY = "replaced_by"
    AFFECTS_PROPERTY = "affects_property"
    HAS_LIMITATION = "has_limitation"
    HAS_QUALITY_REQUIREMENT = "has_quality_requirement"
    HAS_DISTRIBUTION_REQUIREMENT = "has_distribution_requirement"
    HAS_EXPECTED_RESULT = "has_expected_result"
    BASED_ON = "based_on"
    APPLIES_TO = "applies_to"
    USES_TECHNOLOGY = "uses_technology"
    PRODUCES_OUTPUT = "produces_output"
    REQUIRES_EXPERTISE = "requires_expertise"
    USES_EQUIPMENT = "uses_equipment"
    STUDIES = "studies"
    FED_THROUGH = "fed_through"
    DEPENDS_ON = "depends_on"
    OPERATES_AT_CONDITION = "operates_at_condition"
    MEASURED_PROPERTY = "measured_property"
    OPERATES_BETWEEN = "operates_between"
    VALIDATED_BY = "validated_by"
    PERFORMED_BY = "performed_by"
    USES_MATERIAL = "uses_material"
    DESCRIBED_IN = "described_in"
    SUPPORTED_BY = "supported_by"
    AUTHORED_BY = "authored_by"
    AFFILIATED_WITH = "affiliated_with"
    USED_IN_FACILITY = "used_in_facility"
    CONTRADICTS = "contradicts"
    # New (post-flat-schema): anchors to geography/year/numeric values.
    HAS_GEOGRAPHY = "has_geography"
    PUBLISHED_IN_YEAR = "published_in_year"
    HAS_NUMERIC_VALUE = "has_numeric_value"


class EntityLabel(StrEnum):
    """Neo4j node labels — one per ``EntityType``."""

    MATERIAL = "Material"
    SUBSTANCE = "Substance"
    PROCESS = "Process"
    EQUIPMENT = "Equipment"
    PROPERTY = "Property"
    PARAMETER = "Parameter"
    CONDITION = "Condition"
    EXPERIMENT = "Experiment"
    PUBLICATION = "Publication"
    TECHNOLOGY_SOLUTION = "TechnologySolution"
    RESULT = "Result"
    CONCLUSION = "Conclusion"
    LIMITATION = "Limitation"
    FACILITY = "Facility"
    ORGANIZATION = "Organization"
    EXPERT = "Expert"
    GEOGRAPHY = "Geography"
    YEAR = "Year"
    NUMERIC_VALUE = "NumericValue"


class NodeLabel(StrEnum):
    DOCUMENT = "Document"
    CHUNK = "Chunk"
    ALIAS = "Alias"


class RelationshipLabel(StrEnum):
    MENTIONS = "MENTIONS"
    HAS_CHUNK = "HAS_CHUNK"
    KNOWN_AS = "KNOWN_AS"
    SUPPORTS = "SUPPORTS"
    HAS_GEOGRAPHY = "HAS_GEOGRAPHY"
    PUBLISHED_IN_YEAR = "PUBLISHED_IN_YEAR"
    HAS_NUMERIC_VALUE = "HAS_NUMERIC_VALUE"


# Numeric operator vocabulary used in Cypher / query helpers.
class NumericOperator(StrEnum):
    LTE = "<="
    GTE = ">="
    EQ = "="
    LT = "<"
    GT = ">"
    RANGE = "range"


# Geography vocabulary used for filtering domestic vs worldwide practice.
class GeographyKind(StrEnum):
    RUSSIA = "Russia"
    CIS = "CIS"
    EUROPE = "Europe"
    ASIA = "Asia"
    NORTH_AMERICA = "NorthAmerica"
    SOUTH_AMERICA = "SouthAmerica"
    AFRICA = "Africa"
    AUSTRALIA = "Australia"
    WORLDWIDE = "Worldwide"
    UNKNOWN = "Unknown"


# Quick lookup: which Cypher label to use for a given EntityType.
ENTITY_TYPE_TO_LABEL: dict[EntityType, EntityLabel] = {
    etype: EntityLabel(etype.value) for etype in EntityType
}


def label_for(entity_type: EntityType | str) -> EntityLabel:
    """Return the Neo4j label for an ``EntityType`` (or its string value)."""
    if isinstance(entity_type, EntityType):
        return ENTITY_TYPE_TO_LABEL[entity_type]
    return EntityLabel(entity_type)


__all__ = [
    "EntityType",
    "RelationType",
    "EntityLabel",
    "NodeLabel",
    "RelationshipLabel",
    "NumericOperator",
    "GeographyKind",
    "ENTITY_TYPE_TO_LABEL",
    "label_for",
]
