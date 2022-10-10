# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import gym
import sys
import typer

###############################################################################

app = typer.Typer(
    name="list",
    no_args_is_help=True,
    help="List environments and agents.",
)

###############################################################################


@app.command(name="envs", help="List environments.")
def cli_list_envs():
    """Lists the environments registered by the robotrl package."""
    environments = [env_spec.id for env_spec in gym.envs.registry.all()]
    environments.sort()

    for env_id in environments:
        print("{}".format(env_id))

    print("Listed", len(environments), "environment(s)", file=sys.stderr)

    sys.exit(0)
