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


class EntityLabel(StrEnum):
    ENTITY = "Entity"


class NodeLabel(StrEnum):
    DOCUMENT = "Document"
    CHUNK = "Chunk"


class RelationshipLabel(StrEnum):
    MENTIONS = "MENTIONS"
    HAS_CHUNK = "HAS_CHUNK"
    KNOWN_AS = "KNOWN_AS"
    SUPPORTS = "SUPPORTS"
