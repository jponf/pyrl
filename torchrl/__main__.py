# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
__version__ = "0.0.1"

import importlib
# import sys
# import traceback
# import warnings

import click

# ...
import torchrl.util.logging
import torchrl.cli.ddpg
import torchrl.cli.her_ddpg
import torchrl.cli.td3
import torchrl.cli.her_td3
import torchrl.cli.list_envs

###############################################################################

click.disable_unicode_literals_warning = True

_LOG = torchrl.util.logging.get_logger()


###############################################################################

# Modify warning to show traceback
# def warn_with_traceback(message, category, filename, lineno,
#                         stream=None, line=None):

#     log = stream if hasattr(stream, 'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename,
#                                      lineno, line))


# warnings.showwarning = warn_with_traceback


###############################################################################

@click.group()
@click.option("--verbose",
              type=click.Choice(torchrl.util.logging.LEVELS.keys()),
              help="Verbosity level",
              default="INFO")
def main(verbose):
    torchrl.util.logging.set_up_logger(verbose)

    try:
        importlib.import_module("pybullet_envs")
        _LOG.info("PyBullet environments found")
    except ImportError:
        _LOG.info("PyBullet environments not found")


if __name__ == "__main__":
    main.add_command(torchrl.cli.list_envs.cli_list_envs)

    main.add_command(torchrl.cli.ddpg.cli_ddpg_train)
    main.add_command(torchrl.cli.ddpg.cli_ddpg_test)

    main.add_command(torchrl.cli.her_ddpg.cli_her_ddpg_train)
    main.add_command(torchrl.cli.her_ddpg.cli_her_ddpg_test)

    main.add_command(torchrl.cli.td3.cli_td3_train)
    main.add_command(torchrl.cli.td3.cli_td3_test)

    main.add_command(torchrl.cli.her_td3.cli_her_td3_train)
    main.add_command(torchrl.cli.her_td3.cli_her_td3_optimize)
    main.add_command(torchrl.cli.her_td3.cli_her_td3_test)

    main(prog_name="robotrl")
