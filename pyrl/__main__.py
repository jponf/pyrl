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
import pyrl.util.logging
import pyrl.cli.ddpg
import pyrl.cli.her_ddpg
import pyrl.cli.sac
import pyrl.cli.her_sac
import pyrl.cli.td3
import pyrl.cli.her_td3
import pyrl.cli.list_envs

###############################################################################

click.disable_unicode_literals_warning = True

_LOG = pyrl.util.logging.get_logger()


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
              type=click.Choice(pyrl.util.logging.LEVELS.keys()),
              help="Verbosity level",
              default="INFO")
def main(verbose):
    pyrl.util.logging.set_up_logger(verbose)

    try:
        importlib.import_module("pybullet_envs")
        _LOG.info("PyBullet environments found")
    except ImportError:
        _LOG.info("PyBullet environments not found")


def run_main():
    """Utility function that runs the CLI main routine."""
    main.add_command(pyrl.cli.list_envs.cli_list_envs)

    main.add_command(pyrl.cli.ddpg.cli_ddpg_train)
    main.add_command(pyrl.cli.ddpg.cli_ddpg_test)

    main.add_command(pyrl.cli.her_ddpg.cli_her_ddpg_train)
    main.add_command(pyrl.cli.her_ddpg.cli_her_ddpg_test)

    main.add_command(pyrl.cli.sac.cli_sac_train)
    main.add_command(pyrl.cli.sac.cli_sac_test)
    main.add_command(pyrl.cli.her_sac.cli_her_sac_train)
    main.add_command(pyrl.cli.her_sac.cli_her_sac_test)

    main.add_command(pyrl.cli.td3.cli_td3_train)
    main.add_command(pyrl.cli.td3.cli_td3_test)

    main.add_command(pyrl.cli.her_td3.cli_her_td3_train)
    main.add_command(pyrl.cli.her_td3.cli_her_td3_test)

    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    main(prog_name="torchrl")


if __name__ == "__main__":
    run_main()
