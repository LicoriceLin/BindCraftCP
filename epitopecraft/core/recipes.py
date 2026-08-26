"""Validated filter and residue-bias recipes with explicit Python hooks."""

from __future__ import annotations

import ast
import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from .paths import resolve_package_path


class RecipeError(ValueError):
    """Base error for malformed or unevaluable recipes."""


class RecipeSecurityError(RecipeError):
    """Raised when an expression requests an unsafe Python operation."""


def _mean(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values:
        raise ValueError("mean() expects at least one value")
    return sum(values) / len(values)


SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "all": all,
    "any": any,
    "len": len,
    "max": max,
    "mean": _mean,
    "min": min,
    "sorted": sorted,
    "sum": sum,
}

ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Subscript,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.Call,
    ast.Attribute,
    ast.IfExp,
)


class _ExpressionValidator(ast.NodeVisitor):
    """Reject expression syntax outside the recipe evaluator's allowlist."""

    def generic_visit(self, node):
        """Visit a node only when its syntax class is allowed."""
        if not isinstance(node, ALLOWED_NODES):
            raise RecipeSecurityError(
                f"Expression operation is not allowed: {node.__class__.__name__}"
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        """Reject private or dunder-style names."""
        if node.id.startswith("_"):
            raise RecipeSecurityError(f"Private name is not allowed: {node.id}")

    def visit_Attribute(self, node: ast.Attribute):
        """Allow only safe mapping inspection methods."""
        if node.attr not in {"values", "keys", "items"}:
            raise RecipeSecurityError(f"Attribute is not allowed: {node.attr}")
        if not isinstance(node.value, ast.Name) or node.value.id.startswith("_"):
            raise RecipeSecurityError("Only mapping values/keys/items access is allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Validate calls against safe functions and mapping methods."""
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_FUNCTIONS:
                raise RecipeSecurityError(f"Function is not allowed: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            self.visit_Attribute(node.func)
        else:
            raise RecipeSecurityError("Only allowlisted functions may be called")
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)


class SafeExpression:
    """A small expression DSL compiled after AST validation."""

    def __init__(self, expression: str):
        source = expression.strip()
        self.row_argument: str | None = None
        parsed = ast.parse(source, mode="eval")
        body = parsed.body
        if isinstance(body, ast.Lambda):
            if len(body.args.args) != 1 or body.args.vararg or body.args.kwarg:
                raise RecipeSecurityError("Selector lambdas must take exactly one row")
            self.row_argument = body.args.args[0].arg
            body = body.body
            parsed = ast.Expression(body=body)
            ast.fix_missing_locations(parsed)
        _ExpressionValidator().visit(parsed)
        self.expression = expression
        self._code = compile(parsed, "<recipe>", "eval")

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        """Evaluate with no builtins and only allowlisted helpers."""

        environment = {**SAFE_FUNCTIONS, **dict(values)}
        return eval(self._code, {"__builtins__": {}}, environment)

    def matches(self, row: Mapping[str, Any]) -> bool:
        """Evaluate a legacy lambda body or expression against one row."""

        if self.row_argument is None:
            environment = dict(row)
        else:
            environment = {self.row_argument: row}
        return bool(self.evaluate(environment))


class TrustedCallableRegistry:
    """Explicit registry for scientific rules that exceed the YAML DSL."""

    def __init__(self):
        self._callables: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, function: Callable[..., Any]) -> None:
        """Register a named callable and reject accidental replacement."""

        if name in self._callables:
            raise RecipeError(f"Callable already registered: {name}")
        self._callables[name] = function

    def resolve(self, name: str, *, allow_import: bool = False) -> Callable[..., Any]:
        """Resolve a registered name or an explicitly allowed module path."""

        if name in self._callables:
            return self._callables[name]
        if allow_import and ":" in name:
            module_name, attribute = name.split(":", 1)
            function = getattr(importlib.import_module(module_name), attribute)
            if not callable(function):
                raise RecipeError(f"Python hook is not callable: {name}")
            return function
        raise RecipeError(f"Unknown trusted callable: {name}")


def _load_mapping(path: str | Path) -> dict[str, Any]:
    resolved = resolve_package_path(path)
    with resolved.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise RecipeError(f"Recipe root must be a mapping: {resolved}")
    return value


def _get_metric(metrics: Mapping[str, Any], path: str) -> Any:
    if path in metrics:
        return metrics[path]
    value: Any = metrics
    for part in path.split(":"):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _is_filter_leaf(value: Any) -> bool:
    return isinstance(value, Mapping) and (
        "higher" in value
        or "func" in value
        or "expr" in value
        or "predicate_hook" in value
    )


def _flatten_filter_tree(
    value: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Mapping[str, Any]]:
    leaves: dict[str, Mapping[str, Any]] = {}
    for key, child in value.items():
        path = f"{prefix}:{key}" if prefix else str(key)
        if _is_filter_leaf(child):
            leaves[path] = child
        elif isinstance(child, Mapping):
            leaves.update(_flatten_filter_tree(child, path))
        else:
            raise RecipeError(f"Invalid filter rule at {path}")
    return leaves


def _tree_get(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split(":"):
        if not isinstance(current, Mapping) or part not in current:
            raise RecipeError(f"Unknown filter recipe element: {path}")
        current = current[part]
    return current


@dataclass(frozen=True)
class FilterDecision:
    """Per-rule decisions plus the combined pass/fail result."""

    passed: bool
    rules: Mapping[str, bool]


class FilterRecipeBook:
    """Named filter stages resolved from flexible threshold trees."""

    def __init__(
        self,
        thresholds: Mapping[str, Any],
        recipes: Mapping[str, Any],
        *,
        callable_registry: TrustedCallableRegistry | None = None,
        allow_python_hooks: bool = False,
    ):
        self.thresholds = dict(thresholds)
        self.recipes = dict(recipes)
        self.callable_registry = callable_registry or TrustedCallableRegistry()
        self.allow_python_hooks = allow_python_hooks

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        callable_registry: TrustedCallableRegistry | None = None,
        allow_python_hooks: bool = False,
    ) -> "FilterRecipeBook":
        """Load current or v2 filter YAML."""

        data = _load_mapping(path)
        return cls(
            data.get("thresholds", {}),
            data.get("recipes", {}),
            callable_registry=callable_registry,
            allow_python_hooks=allow_python_hooks,
        )

    def rules_for(self, recipe: str) -> dict[str, Mapping[str, Any]]:
        """Expand a named stage while preserving metric-path rule ids."""

        if recipe == "all":
            return _flatten_filter_tree(self.thresholds)
        if recipe not in self.recipes:
            raise RecipeError(f"Unknown filter recipe: {recipe}")
        rules: dict[str, Mapping[str, Any]] = {}
        for element in self.recipes[recipe]:
            subtree = _tree_get(self.thresholds, element)
            if _is_filter_leaf(subtree):
                rules[element] = subtree
            elif isinstance(subtree, Mapping):
                rules.update(_flatten_filter_tree(subtree, element))
            else:
                raise RecipeError(f"Invalid filter recipe element: {element}")
        return rules

    def evaluate(self, recipe: str, metrics: Mapping[str, Any]) -> FilterDecision:
        """Evaluate every selected rule; missing metrics fail closed."""

        decisions: dict[str, bool] = {}
        for rule_id, rule in self.rules_for(recipe).items():
            predicate_hook = rule.get("predicate_hook")
            if predicate_hook is not None:
                if not self.allow_python_hooks:
                    raise RecipeSecurityError(
                        f"Filter rule {rule_id} requests a Python hook; enable it explicitly"
                    )
                predicate = self.callable_registry.resolve(
                    str(predicate_hook),
                    allow_import=True,
                )
                decisions[rule_id] = bool(predicate(metrics))
                continue
            expression = rule.get("expr", rule.get("func"))
            if expression is not None:
                metric_spec = rule.get("metrics", {})
                if isinstance(metric_spec, str):
                    resolved = {"value": _get_metric(metrics, metric_spec)}
                elif isinstance(metric_spec, list):
                    resolved = {
                        f"m{index}": _get_metric(metrics, path)
                        for index, path in enumerate(metric_spec)
                    }
                elif isinstance(metric_spec, Mapping):
                    resolved = {
                        name: _get_metric(metrics, path)
                        for name, path in metric_spec.items()
                    }
                else:
                    raise RecipeError(f"Invalid metrics mapping for {rule_id}")
                if any(value is None for value in resolved.values()):
                    decisions[rule_id] = False
                else:
                    decisions[rule_id] = bool(
                        SafeExpression(str(expression)).evaluate(
                            {
                                "metrics": resolved,
                                "value": (
                                    next(iter(resolved.values()))
                                    if len(resolved) == 1
                                    else None
                                ),
                            }
                        )
                    )
                continue

            value = _get_metric(metrics, rule_id)
            if value is None:
                decisions[rule_id] = False
            elif isinstance(value, bool):
                decisions[rule_id] = value == bool(rule["higher"])
            else:
                decisions[rule_id] = bool(rule["higher"]) == (
                    value >= rule["threshold"]
                )
        return FilterDecision(passed=all(decisions.values()), rules=decisions)


AMINO_ACIDS = tuple("ARNDCQEGHILKMFPSTWYV")
DEFAULT_PENALTIES = {
    "Sp": math.log(0.5),
    "Hp": math.log(0.25),
    "No": -1_000_000.0,
    "Null": 0.0,
    "Sr": math.log(2.0),
    "Hr": math.log(4.0),
}


@dataclass(frozen=True)
class BiasPlan:
    """Backend-neutral per-position amino-acid log biases and mutable mask."""

    amino_acids: tuple[str, ...]
    bias: tuple[tuple[float, ...], ...]
    mutable: tuple[bool, ...]


@dataclass(frozen=True)
class _PenaltyRule:
    name: str
    selector: Callable[[Mapping[str, Any]], bool]
    penalties_aas: Mapping[str, str]
    penalties: Mapping[str, float]
    reward_input_sequence: float | None = None


class PenaltyRecipeBook:
    """Flexible residue selectors and composable ProteinMPNN log biases."""

    def __init__(self, rules: Iterable[_PenaltyRule]):
        self.rules = tuple(rules)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        callable_registry: TrustedCallableRegistry | None = None,
        allow_python_hooks: bool = False,
    ) -> "PenaltyRecipeBook":
        """Load JSON or YAML through the same safe mapping parser."""

        return cls.from_mapping(
            _load_mapping(path),
            callable_registry=callable_registry,
            allow_python_hooks=allow_python_hooks,
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        callable_registry: TrustedCallableRegistry | None = None,
        allow_python_hooks: bool = False,
    ) -> "PenaltyRecipeBook":
        """Compile legacy and v2 selector forms without raw ``eval``."""

        registry = callable_registry or TrustedCallableRegistry()
        rules: list[_PenaltyRule] = []
        for name, raw_rule in mapping.items():
            if not isinstance(raw_rule, Mapping):
                raise RecipeError(f"Penalty rule {name} must be a mapping")
            selector_hook = raw_rule.get("selector_hook")
            if selector_hook is not None:
                if not allow_python_hooks:
                    raise RecipeSecurityError(
                        f"Penalty rule {name} requests a Python hook; enable it explicitly"
                    )
                selector = registry.resolve(str(selector_hook), allow_import=True)
            else:
                expression = raw_rule.get(
                    "where",
                    raw_rule.get("select_func_expr", "lambda s: True"),
                )
                compiled = SafeExpression(str(expression))
                selector = compiled.matches
            penalties = dict(DEFAULT_PENALTIES)
            penalties.update(raw_rule.get("penalties_values", {}))
            reward = None
            if raw_rule.get("type") == "reward_input_sequence":
                reward = float(raw_rule.get("reward_value", penalties["Sr"]))
            rules.append(
                _PenaltyRule(
                    name=str(name),
                    selector=selector,
                    penalties_aas=dict(raw_rule.get("penalties_aas", {})),
                    penalties=penalties,
                    reward_input_sequence=reward,
                )
            )
        return cls(rules)

    def apply(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        amino_acids: tuple[str, ...] = AMINO_ACIDS,
    ) -> BiasPlan:
        """Apply every rule in order and return a backend-neutral bias plan."""

        rows = tuple(rows)
        indices = {amino_acid: index for index, amino_acid in enumerate(amino_acids)}
        bias = [[0.0 for _ in amino_acids] for _ in rows]
        mutable = [False for _ in rows]
        for rule in self.rules:
            for position, row in enumerate(rows):
                if not rule.selector(row):
                    continue
                if rule.reward_input_sequence is not None:
                    sequence_aa = row.get("seq")
                    if sequence_aa in indices:
                        bias[position][indices[sequence_aa]] += rule.reward_input_sequence
                if "fix" in rule.penalties_aas:
                    mutable[position] = False
                    continue
                for penalty_name, amino_acid_group in rule.penalties_aas.items():
                    penalty = float(rule.penalties.get(penalty_name, 0.0))
                    for amino_acid in amino_acid_group:
                        if amino_acid not in indices:
                            raise RecipeError(
                                f"Rule {rule.name} uses unknown amino acid {amino_acid!r}"
                            )
                        bias[position][indices[amino_acid]] += penalty
                    mutable[position] = True
        return BiasPlan(
            amino_acids=amino_acids,
            bias=tuple(tuple(row) for row in bias),
            mutable=tuple(mutable),
        )
