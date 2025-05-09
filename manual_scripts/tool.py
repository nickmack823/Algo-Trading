import ast
import inspect

import astor
import talib as ta
from indicator_functions import *  # So custom functions like KVO, SSL are visible

TA_INTEGER_DEFAULT = -2147483648


def extract_params(func):
    sig = inspect.signature(func)
    ignore = {"open", "high", "low", "close", "volume", "price_data", "df", "real"}
    defaults = {}
    for name, param in sig.parameters.items():
        if name.lower() in ignore:
            continue
        default = param.default
        defaults[name] = default if default is not inspect.Parameter.empty else None
    return defaults


def get_function_object(node):
    # node is a lambda, so its body will be a Call node
    if isinstance(node, ast.Lambda) and isinstance(node.body, ast.Call):
        func_name_node = node.body.func
        if isinstance(func_name_node, ast.Name):
            return func_name_node.id
        elif isinstance(func_name_node, ast.Attribute):
            return f"{func_name_node.value.id}.{func_name_node.attr}"
    return None


print("Paste your indicator list (end with an empty line):")
lines = []
while True:
    line = input()
    if line.strip() == "":
        break
    lines.append(line)
code_snippet = "\n".join(lines)

tree = ast.parse(code_snippet)


class Transformer(ast.NodeTransformer):
    def visit_List(self, node):
        for elt in node.elts:
            if isinstance(elt, ast.Dict):
                keys = [k.s for k in elt.keys if isinstance(k, ast.Str)]
                values = elt.values
                name_node = values[keys.index("name")] if "name" in keys else None
                func_node = (
                    values[keys.index("function")] if "function" in keys else None
                )
                if isinstance(name_node, ast.Constant) and isinstance(
                    func_node, ast.Lambda
                ):
                    func_expr = get_function_object(func_node)
                    try:
                        func_obj = eval(func_expr, globals())
                        params = extract_params(func_obj)

                        # Inject raw_function
                        elt.keys.append(ast.Constant(value="raw_function"))
                        elt.values.append(ast.parse(func_expr).body[0].value)

                        # Inject parameters
                        elt.keys.append(ast.Constant(value="parameters"))
                        elt.values.append(ast.parse(str(params)).body[0].value)

                    except Exception as e:
                        print(
                            f"Warning: Couldn't resolve function `{func_expr}` for {name_node.value}: {e}"
                        )
        return node


# Add parent links
for node in ast.walk(tree):
    for child in ast.iter_child_nodes(node):
        child.parent = node

modified = Transformer().visit(tree)

print("\n--- Modified Indicator List ---\n")
print(astor.to_source(modified))
