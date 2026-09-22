import pytest

from ..nodes import logic_math_nodes


EXPECTED_NODE_IDS = {
    "UC_LogicIF",
    "UC_LogicAND",
    "UC_LogicOR",
    "UC_LogicNOT",
    "UC_LogicXOR",
    "UC_MathAdd",
    "UC_MathSubtract",
    "UC_MathMultiply",
    "UC_MathDivide",
    "UC_MathPower",
    "UC_MathFloor",
    "UC_MathCeil",
    "UC_MathRound",
    "UC_MathModulo",
    "UC_MathAbs",
    "UC_MathSqrt",
    "UC_MathSin",
    "UC_MathCos",
    "UC_MathTan",
    "UC_MathMin",
    "UC_MathMax",
    "UC_MathClamp",
    "UC_MathNumberConvert",
    "UC_StringToNumber",
    "UC_NumberToString",
    "UC_MathCompare",
    "UC_MathOperation",
    "UC_MathAspectRatio",
}


def _value(node_output):
    return node_output.args[0]


def test_logic_math_node_ids_are_prefixed_and_complete():
    node_ids = {
        node.define_schema().node_id
        for node in logic_math_nodes.LOGIC_MATH_NODES
    }
    assert node_ids == EXPECTED_NODE_IDS


def test_logic_nodes_preserve_boolean_behavior():
    assert _value(logic_math_nodes.UC_LogicIF.execute(True, "yes", "no")) == "yes"
    assert _value(logic_math_nodes.UC_LogicIF.execute(False, "yes", "no")) == "no"
    assert _value(
        logic_math_nodes.UC_LogicAND.execute({"input0": True, "input1": False})
    ) is False
    assert _value(
        logic_math_nodes.UC_LogicOR.execute({"input0": False, "input1": True})
    ) is True
    assert _value(logic_math_nodes.UC_LogicNOT.execute(True)) is False
    assert _value(
        logic_math_nodes.UC_LogicXOR.execute(
            {"input0": True, "input1": True, "input2": True}
        )
    ) is True


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (logic_math_nodes.UC_MathAdd, 15),
        (logic_math_nodes.UC_MathSubtract, 5),
        (logic_math_nodes.UC_MathMultiply, 50),
    ],
)
def test_variadic_math_nodes_preserve_input_order(node, expected):
    assert _value(node.execute({"operand0": 10, "operand1": 5})) == expected


def test_divide_and_zero_handling_are_preserved():
    assert _value(
        logic_math_nodes.UC_MathDivide.execute(
            {"operand0": 12, "operand1": 3},
            True,
        )
    ) == 4
    assert _value(
        logic_math_nodes.UC_MathDivide.execute(
            {"operand0": 12, "operand1": 0},
            True,
        )
    ) == 0
    assert (
        logic_math_nodes.UC_MathDivide.validate_inputs(
            {"operand0": 12, "operand1": 0},
            False,
        )
        == "Division by zero is not allowed"
    )


def test_numeric_operations_preserve_results():
    assert _value(logic_math_nodes.UC_MathPower.execute(2, 3)) == 8
    assert _value(logic_math_nodes.UC_MathFloor.execute(2.9)) == 2
    assert _value(logic_math_nodes.UC_MathCeil.execute(2.1)) == 3
    assert _value(logic_math_nodes.UC_MathRound.execute(2.345, 2)) == 2.35
    assert _value(logic_math_nodes.UC_MathModulo.execute(10, 3)) == 1
    assert _value(logic_math_nodes.UC_MathModulo.execute(10, 0)) == 0
    assert _value(logic_math_nodes.UC_MathAbs.execute(-3.5)) == 3.5
    assert _value(logic_math_nodes.UC_MathSqrt.execute(-1)) == 0.0
    assert _value(logic_math_nodes.UC_MathSqrt.execute(9)) == 3.0


def test_trigonometry_and_min_max_preserve_results():
    assert _value(logic_math_nodes.UC_MathSin.execute(90, "Degrees")) == pytest.approx(1)
    assert _value(logic_math_nodes.UC_MathCos.execute(0, "Radians")) == pytest.approx(1)
    assert _value(logic_math_nodes.UC_MathTan.execute(45, "Degrees")) == pytest.approx(1)
    assert _value(
        logic_math_nodes.UC_MathMin.execute({"value0": 5, "value1": 2})
    ) == 2
    assert _value(
        logic_math_nodes.UC_MathMax.execute({"value0": 5, "value1": 2})
    ) == 5


def test_conversion_comparison_and_aspect_ratio_nodes():
    assert _value(logic_math_nodes.UC_MathClamp.execute(12, 0, 10)) == 10
    assert logic_math_nodes.UC_MathNumberConvert.execute(3.5).args == (3, 3.5)
    assert _value(logic_math_nodes.UC_StringToNumber.execute("12")) == 12
    assert _value(logic_math_nodes.UC_StringToNumber.execute("1.5")) == 1.5
    assert _value(logic_math_nodes.UC_StringToNumber.execute("bad", 7)) == 7
    assert _value(logic_math_nodes.UC_NumberToString.execute(12.5)) == "12.5"
    assert _value(logic_math_nodes.UC_MathCompare.execute(3, 2, "Greater Than"))
    assert _value(logic_math_nodes.UC_MathCompare.execute(3, 2, "unknown")) is False
    assert _value(logic_math_nodes.UC_MathOperation.execute(3, 2, "Multiply")) == 6
    assert logic_math_nodes.UC_MathAspectRatio.execute(1920, 1080).args == (16, 9)


def test_logic_math_schemas_preserve_autogrow_limits():
    for node in (
        logic_math_nodes.UC_LogicAND,
        logic_math_nodes.UC_MathAdd,
        logic_math_nodes.UC_MathMin,
    ):
        dynamic_input = node.define_schema().inputs[0]
        assert dynamic_input.template.min == 2
        assert dynamic_input.template.max == 10
