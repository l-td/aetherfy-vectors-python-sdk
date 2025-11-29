"""
Schema validation and management for Aetherfy Vectors SDK.

Provides schema definition, validation, and type detection functionality
to enforce data quality in vector collections.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field as dataclass_field


def detect_type(value: Any) -> str:
    """Detect the precise type of a value.

    Args:
        value: Value to detect type of.

    Returns:
        Type name: 'null', 'boolean', 'string', 'integer', 'float', 'array', 'object', 'unknown'.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


@dataclass
class FieldDefinition:
    """Definition of a single field in a schema."""

    type: str
    required: bool
    element_type: Optional[str] = None  # For arrays
    fields: Optional[Dict[str, "FieldDefinition"]] = None  # For objects

    def to_dict(self) -> Dict[str, Any]:
        """Convert field definition to dictionary format."""
        result = {"type": self.type, "required": self.required}
        if self.element_type:
            result["element_type"] = self.element_type
        if self.fields:
            result["fields"] = {k: v.to_dict() for k, v in self.fields.items()}
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldDefinition":
        """Create FieldDefinition from dictionary."""
        fields = None
        if "fields" in data:
            fields = {k: cls.from_dict(v) for k, v in data["fields"].items()}

        return cls(
            type=data["type"],
            required=data["required"],
            element_type=data.get("element_type"),
            fields=fields,
        )


@dataclass
class Schema:
    """Schema definition for a collection's payload structure."""

    fields: Dict[str, FieldDefinition]

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary format."""
        return {"fields": {k: v.to_dict() for k, v in self.fields.items()}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Schema":
        """Create Schema from dictionary."""
        fields = {k: FieldDefinition.from_dict(v) for k, v in data["fields"].items()}
        return cls(fields=fields)


@dataclass
class ValidationError:
    """Represents a single validation error."""

    field: str
    code: str
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {"field": self.field, "code": self.code, "message": self.message}
        if self.expected:
            result["expected"] = self.expected
        if self.actual:
            result["actual"] = self.actual
        return result


def validate_payload(
    payload: Dict[str, Any], schema: Schema, path: str = ""
) -> List[ValidationError]:
    """Validate a payload against a schema.

    Args:
        payload: Payload dictionary to validate.
        schema: Schema to validate against.
        path: Current field path (for nested validation).

    Returns:
        List of validation errors (empty if valid).
    """
    errors = []

    # Handle None/null payload
    if payload is None:
        payload = {}

    for field_name, field_def in schema.fields.items():
        field_path = f"{path}.{field_name}" if path else field_name
        value = payload.get(field_name)

        # Check required fields
        if field_def.required and value is None:
            errors.append(
                ValidationError(
                    field=field_path,
                    code="REQUIRED_FIELD_MISSING",
                    message=f"Required field '{field_path}' is missing",
                )
            )
            continue

        # Skip validation for optional missing fields
        if value is None:
            continue

        # Check type
        actual_type = detect_type(value)
        if actual_type != field_def.type:
            errors.append(
                ValidationError(
                    field=field_path,
                    code="TYPE_MISMATCH",
                    message=f"Field '{field_path}' expected {field_def.type}, got {actual_type}",
                    expected=field_def.type,
                    actual=actual_type,
                )
            )
            continue

        # Check array element types
        if (
            field_def.type == "array"
            and field_def.element_type
            and isinstance(value, list)
        ):
            for i, element in enumerate(value):
                element_type = detect_type(element)
                if element_type != field_def.element_type:
                    errors.append(
                        ValidationError(
                            field=f"{field_path}[{i}]",
                            code="ARRAY_ELEMENT_TYPE_MISMATCH",
                            message=f"Array element at '{field_path}[{i}]' expected {field_def.element_type}, got {element_type}",
                            expected=field_def.element_type,
                            actual=element_type,
                        )
                    )

        # Recursively validate nested objects
        if field_def.type == "object" and field_def.fields and isinstance(value, dict):
            nested_schema = Schema(fields=field_def.fields)
            nested_errors = validate_payload(value, nested_schema, field_path)
            errors.extend(nested_errors)

    return errors


@dataclass
class VectorValidationError:
    """Represents validation errors for a single vector."""

    index: int
    id: Union[str, int]
    errors: List[ValidationError]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "index": self.index,
            "id": self.id,
            "errors": [e.to_dict() for e in self.errors],
        }


def validate_vectors(vectors: List[Any], schema: Schema) -> List[VectorValidationError]:
    """Validate multiple vectors against a schema.

    Args:
        vectors: List of vector dictionaries or Point objects with payloads.
        schema: Schema to validate against.

    Returns:
        List of validation errors per vector.
    """
    all_errors = []

    for i, vector in enumerate(vectors):
        # Handle both dict and Point objects
        if hasattr(vector, "payload"):  # Point object
            payload = vector.payload
            vector_id = vector.id
        else:  # Dictionary
            payload = vector.get("payload", {})
            vector_id = vector.get("id", "unknown")

        errors = validate_payload(payload, schema)

        if errors:
            all_errors.append(
                VectorValidationError(index=i, id=vector_id, errors=errors)
            )

    return all_errors


@dataclass
class AnalysisResult:
    """Result of schema analysis on a collection."""

    collection: str
    sample_size: int
    total_points: int
    fields: Dict[str, Any]
    suggested_schema: Schema
    processing_time_ms: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Create AnalysisResult from dictionary."""
        return cls(
            collection=data["collection"],
            sample_size=data["sample_size"],
            total_points=data["total_points"],
            fields=data["fields"],
            suggested_schema=Schema.from_dict(data["suggested_schema"]),
            processing_time_ms=data["processing_time_ms"],
        )
