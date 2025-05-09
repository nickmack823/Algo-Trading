import ast
import os

import astor


def extract_indicator_dicts(node):
    """
    Recursively extract all dictionary definitions that have a 'name' and 'function' key.
    """
    dicts = []
    for child in ast.walk(node):
        if isinstance(child, ast.Dict):
            keys = [k.s for k in child.keys if isinstance(k, ast.Str)]
            if "name" in keys and "function" in keys:
                dicts.append(child)
    return dicts


def make_function_name(name):
    return f"{name.lower()}_func".replace(" ", "_").replace("-", "_")


def lambda_to_function(name, lambda_node):
    func_name = make_function_name(name)
    args = lambda_node.args
    body = lambda_node.body
    func_def = ast.FunctionDef(
        name=func_name, args=args, body=[ast.Return(value=body)], decorator_list=[]
    )
    return func_def, func_name


def replace_lambda_in_dict(dict_node, func_name):
    for i, key in enumerate(dict_node.keys):
        if isinstance(key, ast.Str) and key.s == "function":
            dict_node.values[i] = ast.Name(id=func_name, ctx=ast.Load())


def refactor_file(filepath):
    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    indicator_dicts = extract_indicator_dicts(tree)
    new_functions = []

    for d in indicator_dicts:
        # Find the 'name' and 'function' values
        name = None
        lambda_func = None
        for k, v in zip(d.keys, d.values):
            if isinstance(k, ast.Str) and k.s == "name":
                name = v.s
            if (
                isinstance(k, ast.Str)
                and k.s == "function"
                and isinstance(v, ast.Lambda)
            ):
                lambda_func = v

        if name and lambda_func:
            func_def, func_name = lambda_to_function(name, lambda_func)
            replace_lambda_in_dict(d, func_name)
            new_functions.append(func_def)

    # Prepend function defs to the module
    tree.body = new_functions + tree.body

    refactored_source = astor.to_source(tree)
    return refactored_source


if __name__ == "__main__":
    input_path = "scripts/indicators.py"
    output_path = "indicators_refactored.py"

    new_code = refactor_file(input_path)
    with open(output_path, "w") as f:
        f.write(new_code)

    print(f"Refactored file written to {output_path}")
