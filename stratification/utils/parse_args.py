import json
import argparse
from jsonargparse import ActionJsonSchema, namespace_to_dict, ArgumentParser
import json


from .utils import ScientificNotationDecoder, convert_value, set_by_dotted_path
from .schema import schema


def get_config(args_list=None):
    """
    """
    # load and validate config file
    print(args_list)
    # return json.loads(args_list)
    #parser = argparse.ArgumentParser()
    parser = ArgumentParser()
    parser.add_argument('config', action=ActionJsonSchema(schema=schema))
    parser.add_argument('updates', nargs='*')
    print(parser)
    args = parser.parse_args(args_list)
    args = namespace_to_dict(args)

    # convert config to json-serializable dict object
    config = args['config']
    if '__path__' in config:
        config = {**config, '__path__': config['__path__'].abs_path}
    config = json.loads(json.dumps(config), cls=ScientificNotationDecoder)

    # update config in-place with commandline arguments
    update_config(config, args['updates'])
    return config


def update_config(config, updates):
    for update in updates:
        path, sep, value = update.partition("=")
        if sep == "=":
            path = path.strip()  # get rid of surrounding whitespace
            value = value.strip()  # get rid of surrounding whitespace
            set_by_dotted_path(config, path, convert_value(value))
