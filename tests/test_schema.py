"""
Tests for schema functionality in Aetherfy Vectors SDK.

Tests schema detection, validation, analysis, and client integration.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.schema import (
    detect_type,
    validate_payload,
    validate_vectors,
    Schema,
    FieldDefinition,
    AnalysisResult,
    ValidationError,
)
from aetherfy_vectors.models import Point
from aetherfy_vectors.exceptions import (
    SchemaValidationError,
    SchemaNotFoundError,
    ValidationError as ClientValidationError,
)


class TestTypeDetection:
    """Test type detection function."""

    def test_detect_null(self):
        """Test null type detection."""
        assert detect_type(None) == 'null'

    def test_detect_boolean(self):
        """Test boolean type detection."""
        assert detect_type(True) == 'boolean'
        assert detect_type(False) == 'boolean'

    def test_detect_integer(self):
        """Test integer type detection."""
        assert detect_type(42) == 'integer'
        assert detect_type(0) == 'integer'
        assert detect_type(-100) == 'integer'

    def test_detect_float(self):
        """Test float type detection."""
        assert detect_type(3.14) == 'float'
        assert detect_type(-2.5) == 'float'
        assert detect_type(0.1) == 'float'

    def test_detect_string(self):
        """Test string type detection."""
        assert detect_type('hello') == 'string'
        assert detect_type('') == 'string'
        assert detect_type('123') == 'string'

    def test_detect_array(self):
        """Test array type detection."""
        assert detect_type([]) == 'array'
        assert detect_type([1, 2, 3]) == 'array'
        assert detect_type(['a', 'b']) == 'array'

    def test_detect_object(self):
        """Test object type detection."""
        assert detect_type({}) == 'object'
        assert detect_type({'key': 'value'}) == 'object'

    def test_distinguish_integer_from_float(self):
        """Test that integers and floats are properly distinguished."""
        assert detect_type(42) == 'integer'
        assert detect_type(42.0) == 'float'  # Python 42.0 is still a float
        assert detect_type(3.14) == 'float'


class TestFieldDefinition:
    """Test FieldDefinition class."""

    def test_field_definition_creation(self):
        """Test basic field definition creation."""
        field = FieldDefinition(type='string', required=True)
        assert field.type == 'string'
        assert field.required is True
        assert field.element_type is None
        assert field.fields is None

    def test_field_definition_with_element_type(self):
        """Test field definition with element_type for arrays."""
        field = FieldDefinition(type='array', required=True, element_type='string')
        assert field.type == 'array'
        assert field.element_type == 'string'

    def test_field_definition_with_nested_fields(self):
        """Test field definition with nested fields for objects."""
        nested_fields = {
            'source': FieldDefinition(type='string', required=True)
        }
        field = FieldDefinition(type='object', required=True, fields=nested_fields)
        assert field.type == 'object'
        assert 'source' in field.fields
        assert field.fields['source'].type == 'string'

    def test_field_definition_to_dict(self):
        """Test field definition to dictionary conversion."""
        field = FieldDefinition(type='integer', required=False)
        result = field.to_dict()
        assert result == {'type': 'integer', 'required': False}

    def test_field_definition_to_dict_with_element_type(self):
        """Test field definition to dict with element_type."""
        field = FieldDefinition(type='array', required=True, element_type='integer')
        result = field.to_dict()
        assert result == {
            'type': 'array',
            'required': True,
            'element_type': 'integer'
        }

    def test_field_definition_to_dict_with_nested_fields(self):
        """Test field definition to dict with nested fields."""
        field = FieldDefinition(
            type='object',
            required=True,
            fields={
                'name': FieldDefinition(type='string', required=True),
                'age': FieldDefinition(type='integer', required=False)
            }
        )
        result = field.to_dict()
        assert result['type'] == 'object'
        assert result['required'] is True
        assert 'fields' in result
        assert result['fields']['name'] == {'type': 'string', 'required': True}
        assert result['fields']['age'] == {'type': 'integer', 'required': False}

    def test_field_definition_from_dict(self):
        """Test field definition from dictionary creation."""
        data = {'type': 'string', 'required': True}
        field = FieldDefinition.from_dict(data)
        assert field.type == 'string'
        assert field.required is True

    def test_field_definition_from_dict_with_element_type(self):
        """Test field definition from dict with element_type."""
        data = {'type': 'array', 'required': True, 'element_type': 'string'}
        field = FieldDefinition.from_dict(data)
        assert field.type == 'array'
        assert field.element_type == 'string'

    def test_field_definition_from_dict_with_nested_fields(self):
        """Test field definition from dict with nested fields."""
        data = {
            'type': 'object',
            'required': True,
            'fields': {
                'source': {'type': 'string', 'required': True}
            }
        }
        field = FieldDefinition.from_dict(data)
        assert field.type == 'object'
        assert 'source' in field.fields
        assert field.fields['source'].type == 'string'


class TestSchema:
    """Test Schema class."""

    def test_schema_creation(self):
        """Test basic schema creation."""
        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True),
            'age': FieldDefinition(type='integer', required=False)
        })
        assert 'name' in schema.fields
        assert 'age' in schema.fields
        assert schema.fields['name'].type == 'string'

    def test_schema_to_dict(self):
        """Test schema to dictionary conversion."""
        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True),
            'price': FieldDefinition(type='integer', required=True)
        })
        result = schema.to_dict()
        assert 'fields' in result
        assert result['fields']['name'] == {'type': 'string', 'required': True}
        assert result['fields']['price'] == {'type': 'integer', 'required': True}

    def test_schema_from_dict(self):
        """Test schema from dictionary creation."""
        data = {
            'fields': {
                'name': {'type': 'string', 'required': True},
                'tags': {'type': 'array', 'required': False, 'element_type': 'string'}
            }
        }
        schema = Schema.from_dict(data)
        assert 'name' in schema.fields
        assert 'tags' in schema.fields
        assert schema.fields['name'].type == 'string'
        assert schema.fields['tags'].element_type == 'string'

    def test_schema_with_nested_objects(self):
        """Test schema with nested object fields."""
        schema = Schema(fields={
            'metadata': FieldDefinition(
                type='object',
                required=True,
                fields={
                    'source': FieldDefinition(type='string', required=True)
                }
            )
        })
        result = schema.to_dict()
        assert result['fields']['metadata']['type'] == 'object'
        assert 'fields' in result['fields']['metadata']
        assert result['fields']['metadata']['fields']['source']['type'] == 'string'


class TestValidatePayload:
    """Test payload validation function."""

    def test_validate_valid_payload(self):
        """Test validation of a valid payload."""
        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True),
            'age': FieldDefinition(type='integer', required=False)
        })
        payload = {'name': 'John', 'age': 30}
        errors = validate_payload(payload, schema)
        assert len(errors) == 0

    def test_validate_missing_required_field(self):
        """Test validation detects missing required field."""
        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True)
        })
        payload = {}
        errors = validate_payload(payload, schema)
        assert len(errors) == 1
        assert errors[0].code == 'REQUIRED_FIELD_MISSING'
        assert errors[0].field == 'name'

    def test_validate_type_mismatch(self):
        """Test validation detects type mismatch."""
        schema = Schema(fields={
            'price': FieldDefinition(type='integer', required=True)
        })
        payload = {'price': '100'}
        errors = validate_payload(payload, schema)
        assert len(errors) == 1
        assert errors[0].code == 'TYPE_MISMATCH'
        assert errors[0].field == 'price'
        assert 'integer' in errors[0].message
        assert 'string' in errors[0].message

    def test_validate_optional_field_missing(self):
        """Test validation allows missing optional fields."""
        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True),
            'description': FieldDefinition(type='string', required=False)
        })
        payload = {'name': 'Product'}
        errors = validate_payload(payload, schema)
        assert len(errors) == 0

    def test_validate_extra_fields_allowed(self):
        """Test validation allows extra fields not in schema."""
        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True)
        })
        payload = {'name': 'Product', 'extra': 'field'}
        errors = validate_payload(payload, schema)
        assert len(errors) == 0

    def test_validate_array_elements(self):
        """Test validation of array element types."""
        schema = Schema(fields={
            'tags': FieldDefinition(type='array', required=True, element_type='string')
        })
        payload = {'tags': ['a', 'b', 'c']}
        errors = validate_payload(payload, schema)
        assert len(errors) == 0

    def test_validate_array_element_type_mismatch(self):
        """Test validation detects array element type mismatch."""
        schema = Schema(fields={
            'tags': FieldDefinition(type='array', required=True, element_type='string')
        })
        payload = {'tags': ['a', 123, 'c']}
        errors = validate_payload(payload, schema)
        assert len(errors) == 1
        assert errors[0].code == 'ARRAY_ELEMENT_TYPE_MISMATCH'
        assert 'tags[1]' in errors[0].field

    def test_validate_nested_object(self):
        """Test validation of nested objects."""
        schema = Schema(fields={
            'metadata': FieldDefinition(
                type='object',
                required=True,
                fields={
                    'source': FieldDefinition(type='string', required=True)
                }
            )
        })
        payload = {'metadata': {'source': 'api'}}
        errors = validate_payload(payload, schema)
        assert len(errors) == 0

    def test_validate_nested_object_missing_field(self):
        """Test validation detects missing nested required field."""
        schema = Schema(fields={
            'metadata': FieldDefinition(
                type='object',
                required=True,
                fields={
                    'source': FieldDefinition(type='string', required=True)
                }
            )
        })
        payload = {'metadata': {}}
        errors = validate_payload(payload, schema)
        assert len(errors) == 1
        assert errors[0].code == 'REQUIRED_FIELD_MISSING'
        assert 'metadata.source' in errors[0].field

    def test_validate_multiple_errors(self):
        """Test validation detects multiple errors."""
        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True),
            'price': FieldDefinition(type='integer', required=True),
            'description': FieldDefinition(type='string', required=True)
        })
        payload = {'price': '100'}  # Wrong type, missing name and description
        errors = validate_payload(payload, schema)
        assert len(errors) == 3
        codes = [e.code for e in errors]
        assert 'REQUIRED_FIELD_MISSING' in codes
        assert 'TYPE_MISMATCH' in codes

    def test_validate_null_payload(self):
        """Test validation handles null payload."""
        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True)
        })
        errors = validate_payload(None, schema)
        assert len(errors) > 0


class TestValidateVectors:
    """Test batch vector validation function."""

    def test_validate_valid_vectors(self):
        """Test validation of valid vectors."""
        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True)
        })
        vectors = [
            Point(id='1', vector=[0.1, 0.2], payload={'name': 'A'}),
            Point(id='2', vector=[0.3, 0.4], payload={'name': 'B'})
        ]
        errors = validate_vectors(vectors, schema)
        assert len(errors) == 0

    def test_validate_vectors_with_errors(self):
        """Test validation detects errors in specific vectors."""
        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True)
        })
        vectors = [
            Point(id='1', vector=[0.1, 0.2], payload={'name': 'A'}),
            Point(id='2', vector=[0.3, 0.4], payload={'name': 123}),  # Wrong type
            Point(id='3', vector=[0.5, 0.6], payload={})  # Missing field
        ]
        errors = validate_vectors(vectors, schema)
        assert len(errors) == 2
        assert errors[0].index == 1
        assert errors[0].id == '2'
        assert errors[1].index == 2
        assert errors[1].id == '3'

    def test_validate_vectors_includes_all_errors_per_vector(self):
        """Test validation includes all errors for each vector."""
        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True),
            'price': FieldDefinition(type='integer', required=True)
        })
        vectors = [
            Point(id='1', vector=[0.1, 0.2], payload={'name': 123, 'price': 'wrong'})
        ]
        errors = validate_vectors(vectors, schema)
        assert len(errors) == 1
        assert len(errors[0].errors) == 2

    def test_validate_empty_vector_list(self):
        """Test validation handles empty vector list."""
        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True)
        })
        errors = validate_vectors([], schema)
        assert len(errors) == 0

    def test_validate_vectors_with_dict_format(self):
        """Test validation works with dictionary format vectors."""
        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True)
        })
        vectors = [
            {'id': '1', 'vector': [0.1, 0.2], 'payload': {'name': 'A'}},
            {'id': '2', 'vector': [0.3, 0.4], 'payload': {}}  # Missing field
        ]
        errors = validate_vectors(vectors, schema)
        assert len(errors) == 1
        assert errors[0].index == 1


class TestAnalysisResult:
    """Test AnalysisResult class."""

    def test_analysis_result_from_dict(self):
        """Test creating AnalysisResult from API response."""
        data = {
            'collection': 'test_collection',
            'sample_size': 100,
            'total_points': 1000,
            'fields': {
                'name': {
                    'presence': 1.0,
                    'types': {'string': 1.0},
                    'warnings': []
                },
                'price': {
                    'presence': 0.9,
                    'types': {'integer': 0.8, 'string': 0.2},
                    'warnings': ['MIXED_TYPES']
                }
            },
            'suggested_schema': {
                'fields': {
                    'name': {'type': 'string', 'required': True},
                    'price': {'type': 'integer', 'required': False}
                }
            },
            'processing_time_ms': 42
        }
        result = AnalysisResult.from_dict(data)
        assert result.collection == 'test_collection'
        assert result.sample_size == 100
        assert result.total_points == 1000
        assert 'name' in result.fields
        assert 'price' in result.fields
        assert result.fields['price']['warnings'] == ['MIXED_TYPES']
        assert result.suggested_schema.fields['name'].type == 'string'
        assert result.suggested_schema.fields['price'].required is False


class TestClientSchemaOperations:
    """Test client schema management methods."""

    def test_get_schema_success(self, client, mock_requests, mock_successful_response):
        """Test successful schema retrieval."""
        schema_data = {
            'schema': {
                'fields': {
                    'name': {'type': 'string', 'required': True},
                    'price': {'type': 'integer', 'required': True}
                }
            },
            'enforcement_mode': 'strict',
            'etag': 'abc123'
        }
        mock_response = mock_successful_response(schema_data, 200)
        mock_response.headers = {'etag': 'abc123'}
        mock_requests.request.return_value = mock_response

        schema = client.get_schema('test_collection')

        assert schema is not None
        assert 'name' in schema.fields
        assert 'price' in schema.fields
        assert schema.fields['name'].type == 'string'

        # Verify it was cached
        assert 'test_collection' in client._payload_schema_cache
        assert client._payload_schema_cache['test_collection']['etag'] == 'abc123'

    def test_get_schema_not_found(self, client, mock_requests, mock_error_response):
        """Test schema not found returns None."""
        mock_requests.request.return_value = mock_error_response(
            message="Schema not found",
            status_code=404,
            error_code="SCHEMA_NOT_FOUND"
        )

        schema = client.get_schema('nonexistent_collection')
        assert schema is None

    def test_set_schema_success(self, client, mock_requests, mock_successful_response):
        """Test successful schema creation."""
        mock_requests.request.return_value = mock_successful_response({
            'success': True,
            'etag': 'new_etag_123'
        }, 200)

        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True),
            'price': FieldDefinition(type='integer', required=True)
        })

        etag = client.set_schema('test_collection', schema, enforcement='strict')

        assert etag == 'new_etag_123'

        # Verify request
        args, kwargs = mock_requests.request.call_args
        assert kwargs['method'] == 'PUT'
        assert 'test_collection' in kwargs['url']
        assert kwargs['json']['schema']['fields']['name']['type'] == 'string'
        assert kwargs['json']['enforcement_mode'] == 'strict'

    def test_set_schema_with_default_enforcement(self, client, mock_requests, mock_successful_response):
        """Test schema creation with default enforcement mode."""
        mock_requests.request.return_value = mock_successful_response({
            'success': True,
            'etag': 'etag_456'
        }, 200)

        schema = Schema(fields={
            'name': FieldDefinition(type='string', required=True)
        })

        etag = client.set_schema('test_collection', schema)

        # Verify default enforcement is 'off'
        args, kwargs = mock_requests.request.call_args
        assert kwargs['json']['enforcement_mode'] == 'off'

    def test_delete_schema_success(self, client, mock_requests, mock_successful_response):
        """Test successful schema deletion."""
        # First cache a schema
        client._payload_schema_cache['test_collection'] = {
            'schema': Schema(fields={}),
            'enforcement': 'strict',
            'etag': 'old_etag'
        }

        mock_requests.request.return_value = mock_successful_response({
            'success': True
        }, 200)

        result = client.delete_schema('test_collection')

        assert result is True
        # Verify cache was cleared
        assert 'test_collection' not in client._payload_schema_cache

        # Verify request
        args, kwargs = mock_requests.request.call_args
        assert kwargs['method'] == 'DELETE'
        assert 'test_collection' in kwargs['url']

    def test_delete_schema_not_found(self, client, mock_requests, mock_error_response):
        """Test schema deletion when schema doesn't exist."""
        mock_requests.request.return_value = mock_error_response(
            message="Schema not found",
            status_code=404,
            error_code="SCHEMA_NOT_FOUND"
        )

        with pytest.raises(SchemaNotFoundError):
            client.delete_schema('nonexistent_collection')

    def test_analyze_schema_success(self, client, mock_requests, mock_successful_response):
        """Test successful schema analysis."""
        analysis_data = {
            'collection': 'test_collection',
            'sample_size': 100,
            'total_points': 1000,
            'fields': {
                'name': {
                    'presence': 1.0,
                    'types': {'string': 1.0},
                    'warnings': []
                },
                'price': {
                    'presence': 0.95,
                    'types': {'integer': 1.0},
                    'warnings': []
                }
            },
            'suggested_schema': {
                'fields': {
                    'name': {'type': 'string', 'required': True},
                    'price': {'type': 'integer', 'required': True}
                }
            },
            'processing_time_ms': 42
        }
        mock_requests.request.return_value = mock_successful_response(analysis_data, 200)

        result = client.analyze_schema('test_collection', sample_size=100)

        assert result.collection == 'test_collection'
        assert result.sample_size == 100
        assert result.total_points == 1000
        assert 'name' in result.fields
        assert 'price' in result.fields
        assert result.suggested_schema.fields['name'].type == 'string'

        # Verify request
        args, kwargs = mock_requests.request.call_args
        assert kwargs['method'] == 'POST'
        assert 'test_collection/analyze' in kwargs['url']
        assert kwargs['json']['sample_size'] == 100

    def test_analyze_schema_with_default_sample_size(self, client, mock_requests, mock_successful_response):
        """Test schema analysis with default sample size."""
        mock_requests.request.return_value = mock_successful_response({
            'collection': 'test_collection',
            'sample_size': 1000,
            'total_points': 5000,
            'fields': {},
            'suggested_schema': {'fields': {}},
            'processing_time_ms': 42
        }, 200)

        result = client.analyze_schema('test_collection')

        # Verify default sample_size of 1000 was used
        args, kwargs = mock_requests.request.call_args
        assert kwargs['json']['sample_size'] == 1000

    def test_refresh_schema(self, client, mock_requests, mock_successful_response):
        """Test schema cache refresh."""
        # Cache an old schema
        client._payload_schema_cache['test_collection'] = {
            'schema': Schema(fields={'old': FieldDefinition(type='string', required=True)}),
            'enforcement': 'strict',
            'etag': 'old_etag'
        }

        # Mock the new schema response
        new_schema_data = {
            'schema': {
                'fields': {
                    'new': {'type': 'integer', 'required': True}
                }
            },
            'enforcement_mode': 'warn',
            'etag': 'new_etag'
        }
        mock_response = mock_successful_response(new_schema_data, 200)
        mock_response.headers = {'etag': 'new_etag'}
        mock_requests.request.return_value = mock_response

        client.refresh_schema('test_collection')

        # Verify cache was updated
        assert 'test_collection' in client._payload_schema_cache
        assert client._payload_schema_cache['test_collection']['etag'] == 'new_etag'
        assert 'new' in client._payload_schema_cache['test_collection']['schema'].fields


class TestClientValidationIntegration:
    """Test client-side validation integration in upsert."""

    def test_upsert_with_valid_data_and_strict_schema(self, client, mock_requests, mock_successful_response):
        """Test upsert with valid data passes validation."""
        # Mock GET /collections/{name} - vector config
        collection_response = mock_successful_response({
            'result': {
                'config': {
                    'params': {
                        'vectors': {'size': 2, 'distance': 'Cosine'}
                    }
                }
            },
            'schema_version': 'vec_etag'
        }, 200)

        # Mock GET /api/v1/schema/{name} - payload schema
        schema_data = {
            'schema': {
                'fields': {
                    'name': {'type': 'string', 'required': True},
                    'price': {'type': 'integer', 'required': True}
                }
            },
            'enforcement_mode': 'strict',
            'etag': 'schema_etag'
        }
        schema_response = mock_successful_response(schema_data, 200)
        schema_response.headers = {'etag': 'schema_etag'}

        # Mock PUT /collections/{name}/points - upsert
        upsert_response = mock_successful_response({'status': 'ok'}, 200)

        # Set up mock to return responses in order
        mock_requests.request.side_effect = [
            collection_response,  # GET vector config
            schema_response,      # GET payload schema
            upsert_response       # PUT upsert
        ]

        points = [
            Point(id='1', vector=[0.1, 0.2], payload={'name': 'Product A', 'price': 100}),
            Point(id='2', vector=[0.3, 0.4], payload={'name': 'Product B', 'price': 200})
        ]

        result = client.upsert('test_collection', points)
        assert result is True

    def test_upsert_with_invalid_data_raises_validation_error(self, client, mock_requests, mock_successful_response):
        """Test upsert with invalid data raises SchemaValidationError."""
        # Mock GET /collections/{name} - vector config
        collection_response = mock_successful_response({
            'result': {
                'config': {
                    'params': {
                        'vectors': {'size': 2, 'distance': 'Cosine'}
                    }
                }
            },
            'schema_version': 'vec_etag'
        }, 200)

        # Mock GET /api/v1/schema/{name} - payload schema
        schema_data = {
            'schema': {
                'fields': {
                    'name': {'type': 'string', 'required': True},
                    'price': {'type': 'integer', 'required': True}
                }
            },
            'enforcement_mode': 'strict',
            'etag': 'schema_etag'
        }
        schema_response = mock_successful_response(schema_data, 200)
        schema_response.headers = {'etag': 'schema_etag'}

        mock_requests.request.side_effect = [collection_response, schema_response]

        # Invalid data: price is string instead of integer
        points = [
            Point(id='1', vector=[0.1, 0.2], payload={'name': 'Product A', 'price': 'invalid'})
        ]

        with pytest.raises(SchemaValidationError) as exc_info:
            client.upsert('test_collection', points)

        # Verify error details
        assert len(exc_info.value.errors) == 1
        assert exc_info.value.errors[0]['index'] == 0
        assert exc_info.value.errors[0]['id'] == '1'

    def test_upsert_without_schema_allows_any_data(self, client, mock_requests, mock_error_response, mock_successful_response):
        """Test upsert without schema allows any data."""
        # Mock GET /collections/{name} - vector config
        collection_response = mock_successful_response({
            'result': {
                'config': {
                    'params': {
                        'vectors': {'size': 2, 'distance': 'Cosine'}
                    }
                }
            },
            'schema_version': 'vec_etag'
        }, 200)

        # Mock GET /api/v1/schema/{name} - no schema found (404 exception)
        from aetherfy_vectors.exceptions import AetherfyVectorsException
        schema_404_error = AetherfyVectorsException("Schema not found", status_code=404)

        # Mock PUT /collections/{name}/points - upsert
        upsert_response = mock_successful_response({'status': 'ok'}, 200)

        mock_requests.request.side_effect = [
            collection_response,  # GET vector config
            schema_404_error,     # GET payload schema (404)
            upsert_response       # PUT upsert
        ]

        # Any data should be allowed
        points = [
            Point(id='1', vector=[0.1, 0.2], payload={'anything': 'goes'})
        ]

        result = client.upsert('test_collection', points)
        assert result is True

    def test_upsert_with_warn_mode_allows_invalid_data(self, client, mock_requests, mock_successful_response):
        """Test upsert in warn mode allows invalid data."""
        # Mock GET /collections/{name} - vector config
        collection_response = mock_successful_response({
            'result': {
                'config': {
                    'params': {
                        'vectors': {'size': 2, 'distance': 'Cosine'}
                    }
                }
            },
            'schema_version': 'vec_etag'
        }, 200)

        # Mock GET /api/v1/schema/{name} - payload schema with warn enforcement
        schema_data = {
            'schema': {
                'fields': {
                    'name': {'type': 'string', 'required': True}
                }
            },
            'enforcement_mode': 'warn',
            'etag': 'schema_etag'
        }
        schema_response = mock_successful_response(schema_data, 200)
        schema_response.headers = {'etag': 'schema_etag'}

        # Mock PUT /collections/{name}/points - upsert
        upsert_response = mock_successful_response({'status': 'ok'}, 200)

        mock_requests.request.side_effect = [
            collection_response,  # GET vector config
            schema_response,      # GET payload schema
            upsert_response       # PUT upsert
        ]

        # Invalid data should be allowed in warn mode
        points = [
            Point(id='1', vector=[0.1, 0.2], payload={'name': 123})  # Wrong type
        ]

        result = client.upsert('test_collection', points)
        assert result is True

    def test_upsert_handles_412_etag_mismatch(self, client, mock_requests, mock_successful_response, mock_error_response):
        """Test upsert handles 412 response by refreshing schema and retrying."""
        # Mock GET /collections/{name} - vector config
        collection_response = mock_successful_response({
            'result': {
                'config': {
                    'params': {
                        'vectors': {'size': 2, 'distance': 'Cosine'}
                    }
                }
            },
            'schema_version': 'vec_etag'
        }, 200)

        # Mock initial GET /api/v1/schema/{name} - payload schema
        old_schema_data = {
            'schema': {
                'fields': {
                    'name': {'type': 'string', 'required': True}
                }
            },
            'enforcement_mode': 'strict',
            'etag': 'old_etag'
        }
        old_schema_response = mock_successful_response(old_schema_data, 200)
        old_schema_response.headers = {'etag': 'old_etag'}

        # Mock 412 response on first upsert attempt
        precondition_failed = mock_error_response(
            message="Schema has changed",
            status_code=412,
            error_code="PRECONDITION_FAILED"
        )

        # Mock refreshed GET /api/v1/schema/{name} - updated payload schema
        new_schema_data = {
            'schema': {
                'fields': {
                    'name': {'type': 'string', 'required': True},
                    'description': {'type': 'string', 'required': False}
                }
            },
            'enforcement_mode': 'strict',
            'etag': 'new_etag'
        }
        new_schema_response = mock_successful_response(new_schema_data, 200)
        new_schema_response.headers = {'etag': 'new_etag'}

        # Mock successful upsert on retry
        success_response = mock_successful_response({'status': 'ok'}, 200)

        mock_requests.request.side_effect = [
            collection_response,   # GET vector config
            old_schema_response,   # GET payload schema (initial)
            precondition_failed,   # PUT upsert attempt (412)
            new_schema_response,   # GET payload schema (refresh)
            success_response       # PUT upsert (retry)
        ]

        points = [
            Point(id='1', vector=[0.1, 0.2], payload={'name': 'Product A'})
        ]

        result = client.upsert('test_collection', points)
        assert result is True

        # Verify schema was refreshed
        assert client._payload_schema_cache['test_collection']['etag'] == 'new_etag'

    def test_upsert_with_off_enforcement_skips_validation(self, client, mock_requests, mock_successful_response):
        """Test upsert with 'off' enforcement skips client-side validation."""
        # Mock GET /collections/{name} - vector config
        collection_response = mock_successful_response({
            'result': {
                'config': {
                    'params': {
                        'vectors': {'size': 2, 'distance': 'Cosine'}
                    }
                }
            },
            'schema_version': 'vec_etag'
        }, 200)

        # Mock GET /api/v1/schema/{name} - payload schema with off enforcement
        schema_data = {
            'schema': {
                'fields': {
                    'name': {'type': 'string', 'required': True}
                }
            },
            'enforcement_mode': 'off',
            'etag': 'schema_etag'
        }
        schema_response = mock_successful_response(schema_data, 200)
        schema_response.headers = {'etag': 'schema_etag'}

        # Mock PUT /collections/{name}/points - upsert
        upsert_response = mock_successful_response({'status': 'ok'}, 200)

        mock_requests.request.side_effect = [
            collection_response,  # GET vector config
            schema_response,      # GET payload schema
            upsert_response       # PUT upsert
        ]

        # Invalid data should be allowed when enforcement is off
        points = [
            Point(id='1', vector=[0.1, 0.2], payload={'name': 123})  # Wrong type
        ]

        result = client.upsert('test_collection', points)
        assert result is True
