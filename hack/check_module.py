import inspect

def recursively_list_module_contents(module, indent=0):
    """
    Recursively print all classes, methods, functions, and submodules in a pybind11 module.
    
    Args:
        module: The pybind11 module to explore.
        indent: Indentation level for printing (used for recursive calls).
    """
    prefix = " " * indent
    print(f"{prefix}Module: {module.__name__}")
    
    # Get all members of the module
    for name in dir(module):
        if name.startswith("_"):
            continue
        member = getattr(module, name)
        
        # Check if it's a class
        if inspect.isclass(member) and member.__module__ == module.__name__:
            print(f"{prefix}  Class: {name}")
            # List all methods of the class
            methods = inspect.getmembers(member)
            for method_name, member_ in methods:
                if method_name.startswith("_"):
                    continue

                if isinstance(member_, property):
                    print(f"{prefix}    Property: {method_name}")
                else:
                    print(f"{prefix}    Method?: {method_name}")
        
        # Check if it's a function
        elif inspect.isfunction(member):
            print(f"{prefix}  Function: {name}")
        
        # Check if it's a submodule
        elif inspect.ismodule(member) and member.__name__.startswith(module.__name__):
            # Recurse into submodules
            recursively_list_module_contents(member, indent + 2)

        elif inspect.isbuiltin(member):
            print(f"{prefix}  Builtin: {name}")
        
        # Otherwise, it's an attribute or variable
        else:
            print(f"{prefix}  Attribute: {name}")

# 示例使用
import pygloo  # 替换为你的 pybind11 模块名称
recursively_list_module_contents(pygloo)