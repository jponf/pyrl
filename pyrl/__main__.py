# -*- coding: utf-8 -*-

import importlib
import typer

import pyrl.cli.her_sac
import pyrl.util.logging
from pyrl.cli import cli_ddpg, cli_her_ddpg, cli_her_td3, cli_list, cli_sac, cli_td3
from pyrl.util.logging import LoggingLevelName

# import sys
# import traceback
# import warnings


###############################################################################

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


def pyrl_cli_callback(
    verbose: LoggingLevelName = typer.Option(
        LoggingLevelName.INFO,
        case_sensitive=False,
    ),
):
    pyrl.util.logging.set_up_logger(verbose)

    try:
        importlib.import_module("pybullet_envs")
        _LOG.info("PyBullet environments found")
    except ImportError:
        _LOG.info("PyBullet environments not found")


pyrl_cli = typer.Typer(
    no_args_is_help=True,
    callback=pyrl_cli_callback,
)

pyrl_cli.add_typer(cli_ddpg.app)
pyrl_cli.add_typer(cli_her_ddpg.app)
pyrl_cli.add_typer(cli_her_td3.app)
pyrl_cli.add_typer(cli_list.app)
pyrl_cli.add_typer(cli_sac.app)
pyrl_cli.add_typer(cli_td3.app)


# def run_main():
#     """Utility function that runs the CLI main routine."""
#     main.add_command(pyrl.cli.list.cli_list_envs)

#     main.add_command(pyrl.cli.ddpg.cli_ddpg_train)
#     main.add_command(pyrl.cli.ddpg.cli_ddpg_test)

#     main.add_command(pyrl.cli.her_ddpg.cli_her_ddpg_train)
#     main.add_command(pyrl.cli.her_ddpg.cli_her_ddpg_test)

#     main.add_command(pyrl.cli.sac.cli_sac_train)
#     main.add_command(pyrl.cli.sac.cli_sac_test)
#     main.add_command(pyrl.cli.her_sac.cli_her_sac_train)
#     main.add_command(pyrl.cli.her_sac.cli_her_sac_test)

#     main.add_command(pyrl.cli.td3.cli_td3_train)
#     main.add_command(pyrl.cli.td3.cli_td3_test)

#     main.add_command(pyrl.cli.her_td3.cli_her_td3_train)
#     main.add_command(pyrl.cli.her_td3.cli_her_td3_test)

#     # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
#     main(prog_name="torchrl")


if __name__ == "__main__":
    pyrl_cli()
