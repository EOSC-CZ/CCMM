from __future__ import annotations

from pathlib import Path

import rich_click as click
import dataclasses
import xmltodict
import logging
import json
import sys
from typing import Optional, Any, Iterator
from ruamel.yaml import YAML
import copy

log = logging.getLogger(Path(sys.argv[0]).name)

singular_to_plural_element_names = {
    "LowerCorner": "LowerCorners",
    "UpperCorner": "UpperCorners",
    "label": "labels",
    "dataBox": "dataBoxes",
    "email": "emails",
    "phone": "phones",
    "address": "addresses",
    "external_identifiers": "external_identifiers",
    "contact_point": "contact_points",
    "date_updated": "dates_updated",
    "language": "languages",
    "qualified_relation": "qualified_relations",
    "original_repository": "original_repositories",
    "conforms_to_standard": "conforms_to_standards",
    "alternate_name": "alternate_names",
    "given_name": "given_names",
    "affiliation": "affiliations",
    "funder": "funders",
    "time_reference": "time_references",
    "location_name": "location_names",
    "dataset_relations": "dataset_relations",
    "related_object_identifier": "related_object_identifiers",
    "endpoint_url": "endpoint_urls",
    "description": "descriptions",
    "documentation": "documentations",
    "specification": "specifications",
    "access_service": "access_services",
    "conforms_to_schema": "conforms_to_schemas",
    "access_url": "access_urls",
    "download_url": "download_urls",
    "access_rights": "access_rights",
    "contact_person": "contact_persons",
    "other_language": "other_languages",
    "alternate_title": "alternate_titles",
    "is_described_by": "is_described_by",
    "identifier": "identifiers",
    "location": "locations",
    "provenance": "provenances",
    "subject": "subjects",
    "validation_result": "validation_results",
    "distribution": "distributions",
    "funding_reference": "funding_references",
    "terms_of_use": "terms_of_use",
    "related_resource": "related_resources",
    "contact": "contacts",
    "distribution_data_service": "distribution_data_services",
    "distribution_downloadable_file": "distribution_downloadable_files",
}


@dataclasses.dataclass
class SchemaElement:
    name: str
    schema_type_name: str
    schema_type: Optional["SchemaType"] = None
    ref: Optional[str] = None
    documentation: list[str] | tuple[str, ...] = ()
    min_occurs: int = 1
    max_occurs: int | None = 1
    substitution_group: str = ""
    substitutions: list[SchemaElement] = dataclasses.field(default_factory=list)

    def resolve(self, schema: Schema):
        self.check_plural_name()

        if not self.schema_type:
            if self.ref:
                self.schema_type_name = self.ref
                self.schema_type = schema.elements[self.ref].schema_type
            elif not is_simple_type(self.schema_type_name):
                if self.schema_type_name not in schema.types:
                    raise ValueError(
                        f"Unknown type '{self.schema_type_name}' for element {self}"
                    )
                self.schema_type = schema.types[self.schema_type_name]

        if self.schema_type:
            self.schema_type.resolve(schema)

    def check_plural_name(self):
        if self.max_occurs is None or self.max_occurs > 1:
            name = self.name.strip()
            if name not in singular_to_plural_element_names.values():
                self.name = singular_to_plural_element_names[name]

    def resolve_substitutions(self, schema: Schema):
        if self.substitution_group:
            self.schema_type = self.schema_type.copy()
            self.schema_type.elements.update(
                schema.elements[self.substitution_group].schema_type.elements
            )

        if self.ref:
            for s in schema.elements[self.ref].substitutions:
                if s not in self.substitutions:
                    self.substitutions.append(s)

        if self.schema_type:
            self.schema_type.resolve_substitutions(schema)

    def get_invenio_definition(
        self, ignored_elements: set[tuple[str, str]]
    ) -> dict[str, Any]:
        ret: dict[str, Any] = convert_schema_to_invenio(self.schema_type_name)

        if self.max_occurs is None or self.max_occurs > 1:
            ret = {
                "type": "array",
                "items": ret,
            }

        if self.documentation:
            if len(self.documentation) >= 1:
                ret["label.en"] = self.documentation[0]
            if len(self.documentation) >= 2:
                ret["help.en"] = self.documentation[1]

        return ret


def is_simple_type(type_name: str) -> bool:
    return type_name.startswith("xs:")


def convert_schema_to_invenio(schema_type_name: str) -> dict[str, str]:
    match schema_type_name:
        case schema_type_name if schema_type_name.startswith("xs:"):
            return {
                "type": {
                    "xs:string": "keyword",
                    "xs:integer": "integer",
                    "xs:decimal": "float",
                    "xs:boolean": "boolean",
                    "xs:date": "date",
                    "xs:dateTime": "datetime",
                    "xs:gYear": "integer",
                    "xs:anyURI": "url",
                    "xs:positiveInteger": "integer",
                }[schema_type_name]
            }
        case "langString":
            return {"type": "i18nstr"}
        case _:
            return {
                "use": "/$defs/" + schema_type_name,
            }


class IRI(SchemaElement):
    def __init__(self):
        super().__init__(name="iri", schema_type_name="xs:string")

    def resolve(self, schema: Schema):
        pass


@dataclasses.dataclass
class SchemaType:
    name: str
    elements: dict[str, SchemaElement]
    documentation: list[str] | tuple[str, ...] = ()
    base_name: Optional[str] = None

    def copy(self) -> SchemaType:
        return type(self)(
            name=self.name,
            elements={**self.elements},
            documentation=[*self.documentation],
            base_name=self.base_name,
        )

    def resolve(self, schema: Schema):
        for el in self.elements.values():
            el.resolve(schema)

    def get_invenio_definition(self, ignored_elements: set[tuple[str, str]]):
        props: dict[str, Any] = {}
        required: list[str] = []

        for el_name, el in list(self.elements.items()):
            # TODO: does not solve cardinalities etc.
            if el.substitutions:
                del self.elements[el_name]
                for sub in el.substitutions:
                    self.elements[sub.name] = copy.deepcopy(sub)
                    self.elements[sub.name].min_occurs = el.min_occurs
                    self.elements[sub.name].max_occurs = el.max_occurs
                    self.elements[sub.name].check_plural_name()

        for el in self.elements.values():
            if (self.name, el.name) in ignored_elements:
                continue
            props[el.name] = el.get_invenio_definition(ignored_elements)
            if el.min_occurs > 0:
                required.append(el.name)

        if "iri" in required:
            required.remove("iri")

        return remove_empty(
            {
                "properties": props,
                "required": required,
            }
        )

    def resolve_substitutions(self, schema: Schema):
        for el in self.elements.values():
            el.resolve_substitutions(schema)


class OWSBoundingBox(SchemaType):
    def __init__(self):
        super().__init__(
            "ows:BoundingBoxType",
            {
                "LowerCorner": SchemaElement(
                    name="LowerCorner",
                    schema_type_name="xs:decimal",
                    min_occurs=1,
                    max_occurs=None,
                ),
                "UpperCorner": SchemaElement(
                    name="UpperCorner",
                    schema_type_name="xs:decimal",
                    min_occurs=1,
                    max_occurs=None,
                ),
                "crs": SchemaElement(
                    name="crs", schema_type_name="xs:anyURI", min_occurs=0
                ),
                "dimensions": SchemaElement(
                    name="dimensions",
                    schema_type_name="xs:positiveInteger",
                    min_occurs=0,
                ),
            },
            [
                "BoundingBoxType",
                "XML encoded minimum rectangular bounding box (or region) parameter, surrounding all the associated data. "
                "This type is adapted from the EnvelopeType of GML 3.1, with modified contents and documentation "
                "for encoding a MINIMUM size box SURROUNDING all associated data.",
            ],
        )


known_types = {
    "langString": SchemaType("langString", {}, []),
    "ows:BoundingBoxType": OWSBoundingBox(),
}
known_elements: dict[str, SchemaElement] = {"c:iri": IRI()}
ignored_elements: set[tuple[str, str]] = {
    ("dataset", "qualified_relation"),
    ("metadata_record", "qualified_relation"),
    ("resource", "qualified_relation"),
    ("terms_of_use", "contact_person"),
}


class Schema:
    def __init__(
        self,
        root_schema_location: Path,
        root_element_name: str,
        known_types: dict[str, SchemaType] = known_types,
        known_elements: dict[str, SchemaElement] = known_elements,
        ignored_elements: set[tuple[str, str]] = ignored_elements,
    ):
        self.root_schema_location = root_schema_location
        self.root_element_name = root_element_name
        self.elements: dict[str, SchemaElement] = {}
        self.types: dict[str, SchemaType] = {}
        self.known_types = known_types
        self.known_elements = known_elements
        self.ignored_elements = ignored_elements

        self.types.update(self.known_types)
        self.elements.update(self.known_elements)

        self.parsed_locations: set[Path] = set()
        self.parse(root_schema_location)

    def parse(self, schema_location: Path):
        schema_location = schema_location.resolve()
        if schema_location in self.parsed_locations:
            return
        self.parsed_locations.add(schema_location)
        log.info("Parsing %s", schema_location)
        with open(schema_location) as f:
            schema = xmltodict.parse(f.read(), process_namespaces=True)
            schema_items = schema["http://www.w3.org/2001/XMLSchema:schema"]

            pop_list(schema_items, "http://www.w3.org/2001/XMLSchema:import", [])
            pop_string(schema_items, "@targetNamespace", "")
            pop_string(
                schema_items,
                "@http://www.w3.org/2007/XMLSchema-versioning:minVersion",
                "",
            )
            pop_string(
                schema_items,
                "@elementFormDefault",
                "",
            )
            pop_with_error(schema_items, "@xmlns", "")

            self.parse_imports(schema_location, schema_items)
            self.parse_elements(schema_items, self.elements)
            self.parse_types(schema_items, self.types)

            assert_empty(schema_items)

    def parse_imports(self, parent_schema_location: Path, schema: DictSchema):
        for item in pop_list(schema, "http://www.w3.org/2001/XMLSchema:include", []):
            schema_location = pop_string(item, "@schemaLocation")
            schema_location_path = parent_schema_location.parent.joinpath(
                schema_location
            )
            self.parse(schema_location_path)

    def parse_elements(self, schema: DictSchema, elements: dict[str, SchemaElement]):
        for item in pop_list(schema, "http://www.w3.org/2001/XMLSchema:element", []):
            el: SchemaElement = self.parse_element(item)
            elements[el.name] = el

    def parse_element(self, item: DictSchema) -> SchemaElement:
        name = pop_string(item, "@name", "")
        ref = pop_string(item, "@ref", "")
        documentation = self.parse_documentation(item)
        min_occurs = int(pop_string(item, "@minOccurs", "1"))
        max_occurs = to_max_occurs(pop_string(item, "@maxOccurs", "1"))
        substitution_group = pop_string(item, "@substitutionGroup", "")
        pop_string(item, "@http://www.w3.org/ns/sawsdl:modelReference", "")

        if "http://www.w3.org/2001/XMLSchema:complexType" in item:
            schema_type = self.parse_complex_type(
                pop_dict(item, "http://www.w3.org/2001/XMLSchema:complexType", {})
            )
        else:
            schema_type = None

        try:
            if ref in self.known_elements:
                return self.known_elements[ref]
            else:
                if ref and not name:
                    name = ref
                # p(item)
                return SchemaElement(
                    name=name,
                    ref=ref,
                    schema_type_name=pop_string(item, "@type", ""),
                    schema_type=schema_type,
                    documentation=documentation,
                    min_occurs=min_occurs,
                    max_occurs=max_occurs,
                    substitution_group=substitution_group,
                )
        finally:
            assert_empty(item)

    def parse_types(self, schema: DictSchema, types: dict[str, SchemaType]):
        for item in pop_list(
            schema, "http://www.w3.org/2001/XMLSchema:complexType", []
        ):
            ct: SchemaType = self.parse_complex_type(item)
            types[ct.name] = ct

    def parse_complex_type(self, item: DictSchema) -> SchemaType:
        pop_string(item, "@http://www.w3.org/ns/sawsdl:modelReference", "")
        pop_string(item, "@abstract", "")
        documentation = self.parse_documentation(item)

        name = pop_string(item, "@name", "")
        if name in self.known_types:
            return self.known_types[name]
        if not name:
            print(f"Missing name on complexType: {item}")
            raise Exception()
        try:
            # p(item)
            seq = pop_dict(item, "http://www.w3.org/2001/XMLSchema:sequence", {})
            local_elements: dict[str, SchemaElement] = {}
            self.parse_elements(seq, local_elements)
            assert_empty(seq)

            complex_content = pop_dict(
                item, "http://www.w3.org/2001/XMLSchema:complexContent", {}
            )
            extension = pop_dict(
                complex_content, "http://www.w3.org/2001/XMLSchema:extension", {}
            )
            base_name = pop_string(extension, "@base", "")

            seq = pop_dict(extension, "http://www.w3.org/2001/XMLSchema:sequence", {})
            self.parse_elements(seq, local_elements)

            assert_empty(seq)
            assert_empty(extension)
            assert_empty(complex_content)

            return SchemaType(
                name=name,
                elements=local_elements,
                documentation=documentation,
                base_name=base_name,
            )
        finally:
            assert_empty(item)

    def parse_documentation(self, item: DictSchema) -> list[str]:
        annotation = pop_dict(item, "http://www.w3.org/2001/XMLSchema:annotation", {})
        documentation = pop_list(
            annotation, "http://www.w3.org/2001/XMLSchema:documentation", []
        )

        assert_empty(annotation)
        ret: list[str] = []
        for doc in documentation:
            if doc["@http://www.w3.org/XML/1998/namespace:lang"] == "en":
                ret.append(doc["#text"])
        return ret

    def gather_types(self) -> Iterator[SchemaType]:
        seen_types: set[str] = set()

        def gather_type(t: SchemaType) -> Iterator[SchemaType]:
            if not t.name:
                raise Exception(f"Do not have a type name for {t}")
            if t.name not in seen_types:
                seen_types.add(t.name)
                yield t
                for el in t.elements.values():
                    if (t.name, el.name) in self.ignored_elements:
                        continue
                    yield from gather_element(el)

        def gather_element(el: SchemaElement) -> Iterator[SchemaType]:
            if el.substitutions:
                for sub in el.substitutions:
                    yield from gather_element(sub)

            if el.schema_type:
                yield from gather_type(el.schema_type)

        yield from gather_element(self.elements[self.root_element_name])


def p(data: Any):
    print(json.dumps(data, indent=2))


type DictSchemaValue = None | str | list[str] | "DictSchema" | list["DictSchema"]
type DictSchema = dict[str, DictSchemaValue]


def as_list(value: DictSchemaValue) -> list[DictSchema]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [value]
    return value  # type: ignore


def pop_with_error[T](data: Any, key: str, default: T | None = None) -> T:
    if key not in data:
        if default is not None:
            return default
        raise ValueError(f"Missing key {key} in:\n{json.dumps(data, indent=2)}")
    return data.pop(key)


def pop_string(data: Any, key: str, default: str | None = None) -> str:
    value: Any = pop_with_error(data, key, default)

    if not isinstance(value, str):
        raise ValueError(f"Expected string, got {value}")
    return value


def assert_empty(items: DictSchemaValue):
    if not isinstance(items, dict):
        raise ValueError(f"Expected dict, got {items}")
    if items:
        raise ValueError(f"Unprocessed items: {items}")


def pop_dict(data: Any, key: str, default: DictSchema | None = None) -> DictSchema:
    value: Any = pop_with_error(data, key, default)

    if not isinstance(value, dict):
        raise ValueError(f"Expected dict, got {value}")
    return value


def pop_list(
    data: Any, key: str, default: list[DictSchema] | None = None
) -> list[DictSchema]:
    value: Any = pop_with_error(data, key, default)
    if not isinstance(value, list):
        value = [value]
    return value


def to_max_occurs(value: str) -> int | None:
    if value == "unbounded":
        return None
    return int(value)


def remove_empty(d: dict[str, Any]):
    return {k: v for k, v in d.items() if v is not None and v != {} and v != []}


@click.command()
@click.argument("xsd_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path(), default="ccmm.yaml")
def main(xsd_file: str, output_file: str):
    logging.basicConfig(level=logging.INFO)
    schema = Schema(Path(xsd_file), "dataset")

    for el in schema.elements.values():
        if el.substitution_group:
            schema.elements[el.substitution_group].substitutions.append(el)

    for t in schema.types.values():
        t.resolve(schema)

    for el in schema.elements.values():
        el.resolve(schema)

    for el in schema.elements.values():
        el.resolve_substitutions(schema)

    ret = {
        "$defs": {
            t.name: t.get_invenio_definition(schema.ignored_elements)
            for t in sorted(schema.gather_types(), key=lambda x: x.name)
        }
    }

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.dump(ret, open(output_file, "w"))


if __name__ == "__main__":
    main()
