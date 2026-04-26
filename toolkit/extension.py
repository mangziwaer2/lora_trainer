import os
import importlib
import pkgutil
from typing import List

from toolkit.paths import TOOLKIT_ROOT


class Extension(object):
    """Base class for extensions.

    Extensions are registered with the ExtensionManager, which is
    responsible for calling the extension's load() and unload()
    methods at the appropriate times.

    """

    name: str = None
    uid: str = None

    @classmethod
    def get_process(cls):
        # extend in subclass
        pass


def get_all_extensions() -> List[Extension]:
    extension_folders = ['extensions', 'extensions_built_in']

    # This will hold the classes from all extension modules
    all_extension_classes: List[Extension] = []

    # Iterate over all directories (i.e., packages) in the "extensions" directory
    for sub_dir in extension_folders:
        extensions_dir = os.path.join(TOOLKIT_ROOT, sub_dir)
        for (_, name, _) in pkgutil.iter_modules([extensions_dir]):
            # try:
                # Import the module
                module = importlib.import_module(f"{sub_dir}.{name}")
                # Get the value of the AI_TOOLKIT_EXTENSIONS variable
                extensions = getattr(module, "AI_TOOLKIT_EXTENSIONS", None)
                # Check if the value is a list
                if isinstance(extensions, list):
                    # Iterate over the list and add the classes to the main list
                    all_extension_classes.extend(extensions)
            # except ImportError as e:
            #     print(f"Failed to import the {name} module. Error: {str(e)}")

    # Some deployment/packaging paths fail to enumerate namespace-package
    # directories with pkgutil, which leaves the process registry empty.
    # Register the built-in trainer extension explicitly as a fallback.
    known_uids = {extension.uid for extension in all_extension_classes}
    if "sd_trainer" not in known_uids:
        try:
            module = importlib.import_module("extensions_built_in.sd_trainer")
            extensions = getattr(module, "AI_TOOLKIT_EXTENSIONS", None)
            if isinstance(extensions, list):
                all_extension_classes.extend(
                    extension for extension in extensions
                    if extension.uid not in known_uids
                )
        except ImportError:
            pass

    return all_extension_classes


def get_all_extensions_process_dict():
    all_extensions = get_all_extensions()
    process_dict = {}
    for extension in all_extensions:
        try:
            process_dict[extension.uid] = extension.get_process()
        except Exception as e:
            print(f"Warning: failed to load extension process {extension.uid}: {e}")
    return process_dict
