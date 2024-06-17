import ast
from typing import List

from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)


class MetaExprVisitor(ast.NodeVisitor):
    """AST visitor class to collect comparisons and conditions from an expression."""

    def __init__(self):
        self.comparisons = []
        self.conditions = []

    def visit_Compare(self, node):
        """Visit a comparison node and extract details."""
        comparison = {"key": None, "value": None, "operator": None}
        if isinstance(node.left, ast.Name):
            comparison["key"] = node.left.id
        if isinstance(node.comparators[0], ast.Constant):
            comparison["value"] = node.comparators[0].value
        elif isinstance(node.comparators[0], ast.List):
            comparison["value"] = [elt.value for elt in node.comparators[0].elts]
        comparison["operator"] = node.ops[0]
        self.comparisons.append(comparison)
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        """Visit a boolean operation node and extract the condition."""
        self.generic_visit(node)
        self.conditions.append(type(node.op).__name__)


class MetaExprEngine:
    """
    Engine to parse complex filter expressions into structured metadata filters
    compatible with vector stores.

    The engine supports various filter operators and logical conditions as defined
    by FilterOperator and FilterCondition enums:

    FilterOperators:
    - == (equal)
    - > (greater than)
    - < (less than)
    - != (not equal)
    - >= (greater than or equal to)
    - <= (less than or equal to)
    - in (in array)
    - nin (not in array)
    - any (contains any)
    - all (contains all)
    - text_match (full text match)
    - contains (metadata array contains value)

    FilterConditions:
    - and
    - or

    It also supports:
    - Single values (strings, integers, and floats)
    - Lists of strings, integers, and floats

    Example of a supported expression:
    'page gt 1 and title eq "Some title" or tags in ["tag1", "tag2"]'
    """

    def __init__(self) -> None:
        pass

    def parse_expression(self, expression: str) -> List[MetadataFilters]:
        """Parse an expression string into a list of MetadataFilters."""
        tree = ast.parse(expression, mode="eval")

        visitor = MetaExprVisitor()
        visitor.visit(tree)

        metadata_filters_list = []
        current_metadata_filters = MetadataFilters(filters=[], condition=None)
        last_condition = None

        for i, comparison in enumerate(visitor.comparisons):
            key = comparison["key"]
            value = comparison["value"]
            operator = self._map_operator(comparison["operator"])

            current_metadata_filters.filters.append(
                MetadataFilter(
                    key=key,
                    value=value,
                    operator=operator,
                )
            )

            if i < len(visitor.conditions):
                condition = self._map_condition(visitor.conditions[i])
                if last_condition and last_condition != condition:
                    metadata_filters_list.append(current_metadata_filters)
                    current_metadata_filters = MetadataFilters(
                        filters=[], condition=condition
                    )
                else:
                    current_metadata_filters.condition = condition
                last_condition = condition

        if current_metadata_filters.filters:
            metadata_filters_list.append(current_metadata_filters)

        return metadata_filters_list

    @staticmethod
    def _map_operator(operator: type) -> FilterOperator:
        """Map AST operator type to FilterOperator enum."""
        operator_str = type(operator).__name__.upper()
        if operator_str == "NOTEQ":
            operator_str = "NE"
        try:
            return FilterOperator[operator_str]
        except KeyError:
            raise ValueError(f"Unsupported operator: {operator_str}")

    @staticmethod
    def _map_condition(condition: str) -> FilterCondition:
        """Map AST condition string to FilterCondition enum."""
        try:
            return FilterCondition[condition.upper()]
        except KeyError:
            raise ValueError(f"Unsupported condition: {condition}")


if __name__ == "__main__":
    expression = 'tags in ["tag1", "tag2"] and page > 1 or title == "Some title"'

    engine = MetaExprEngine()
    filters = engine.parse_expression(expression)

    for metadata_filter in filters:
        print(metadata_filter)
