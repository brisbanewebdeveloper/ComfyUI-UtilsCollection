import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_from_list_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_from_list_test.nodes.utils_nodes import UC_FromList
finally:
    cli_args.cpu = prior_cpu


def test_schema_accepts_and_returns_a_type_preserving_list():
    schema = UC_FromList.define_schema()

    assert schema.node_id == "UC_FromList"
    assert schema.display_name == "From List"
    assert schema.is_input_list is True
    assert [value.id for value in schema.inputs] == [
        "items",
        "start_index",
        "number_of_entries",
    ]
    assert schema.inputs[1].default == 0
    assert schema.inputs[2].default == 1
    assert schema.outputs[0].is_output_list is True


def test_execute_returns_requested_consecutive_entries():
    output = UC_FromList.execute(
        ["zero", "one", "two", "three"],
        start_index=[1],
        number_of_entries=[2],
    )

    assert output.args == (["one", "two"],)


def test_execute_stops_cleanly_at_end_of_list():
    output = UC_FromList.execute(
        [10, 20, 30],
        start_index=[2],
        number_of_entries=[4],
    )

    assert output.args == ([30],)
