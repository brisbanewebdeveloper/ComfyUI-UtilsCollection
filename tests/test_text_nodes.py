import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_text_nodes_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_text_nodes_test.nodes import text_nodes


def test_text_concatenate_autogrow_schema_uses_wildcard_links():
    schema = text_nodes.UC_TextConcatenateAutogrow.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    text_inputs = inputs["text_inputs"]

    assert schema.node_id == "UC_TextConcatenateAutogrow"
    assert schema.display_name == "Concatenate Text (Autogrow)"
    assert schema.category == "advanced/text"
    assert inputs["delimiter"].get_io_type() == "*"
    assert inputs["delimiter"].optional is True
    assert text_inputs.optional is True
    assert text_inputs.template.input.get_io_type() == "*"
    assert text_inputs.template.input.optional is True
    assert text_inputs.template.names == [
        f"text_{index}" for index in range(1, 101)
    ]
    assert text_inputs.template.min == 0
    assert schema.outputs[0].get_io_type() == "STRING"
    assert schema.outputs[0].display_name == "concatenated_text"


def test_text_concatenate_autogrow_joins_in_numeric_socket_order():
    output = text_nodes.UC_TextConcatenateAutogrow.execute(
        delimiter=" | ",
        text_inputs={
            "text_10": "ten",
            "text_2": "two",
            "text_1": "one",
        },
    )

    assert output.args == ("one | two | ten",)


def test_text_concatenate_autogrow_converts_arbitrary_values_to_strings():
    output = text_nodes.UC_TextConcatenateAutogrow.execute(
        delimiter=0,
        text_inputs={
            "text_1": 12,
            "text_2": True,
            "text_3": None,
        },
    )

    assert output.args == ("120True0None",)


def test_text_concatenate_autogrow_supports_empty_and_newline_delimiters():
    compact = text_nodes.UC_TextConcatenateAutogrow.execute(
        delimiter="",
        text_inputs={"text_1": "alpha", "text_2": "beta"},
    )
    multiline = text_nodes.UC_TextConcatenateAutogrow.execute(
        delimiter="\n",
        text_inputs={"text_1": "alpha", "text_2": "beta"},
    )

    assert compact.args == ("alphabeta",)
    assert multiline.args == ("alpha\nbeta",)


def test_text_concatenate_autogrow_uses_empty_delimiter_when_disconnected():
    output = text_nodes.UC_TextConcatenateAutogrow.execute(
        text_inputs={"text_1": "alpha", "text_2": "beta"},
    )

    assert output.args == ("alphabeta",)


def test_text_concatenate_autogrow_accepts_no_inputs():
    output = text_nodes.UC_TextConcatenateAutogrow.execute()

    assert output.args == ("",)


def test_text_concatenate_lists_schema_receives_and_returns_complete_lists():
    schema = text_nodes.UC_TextConcatenateListsAutogrow.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert schema.node_id == "UC_TextConcatenateListsAutogrow"
    assert schema.display_name == "Concatenate Text Lists (Autogrow)"
    assert schema.is_input_list is True
    assert inputs["delimiter"].get_io_type() == "*"
    assert inputs["text_inputs"].template.input.get_io_type() == "*"
    assert schema.outputs[0].get_io_type() == "STRING"
    assert schema.outputs[0].is_output_list is True


def test_text_concatenate_lists_broadcasts_scalars_and_aligns_lists():
    output = text_nodes.UC_TextConcatenateListsAutogrow.execute(
        delimiter=[" "],
        text_inputs={
            "text_1": ["prefix"],
            "text_2": ["one", "two", "three"],
            "text_3": ["suffix"],
        },
    )

    assert output.args == (
        ["prefix one suffix", "prefix two suffix", "prefix three suffix"],
    )


def test_text_concatenate_lists_aligns_delimiters_and_repeats_final_values():
    output = text_nodes.UC_TextConcatenateListsAutogrow.execute(
        delimiter=["/", " | "],
        text_inputs={
            "text_2": ["A", "B"],
            "text_1": [1, 2, 3],
        },
    )

    assert output.args == (["1/A", "2 | B", "3 | B"],)


def test_text_concatenate_lists_accepts_no_inputs():
    output = text_nodes.UC_TextConcatenateListsAutogrow.execute()

    assert output.args == ([""],)


def test_newline_node_has_no_inputs_and_outputs_one_newline():
    schema = text_nodes.UC_Newline.define_schema()

    assert schema.node_id == "UC_Newline"
    assert schema.display_name == r"\n"
    assert schema.inputs == []
    assert schema.outputs[0].get_io_type() == "STRING"
    assert schema.outputs[0].display_name == r"\n"
    assert text_nodes.UC_Newline.execute().args == ("\n",)
