# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import click
import gym
import sys


###############################################################################

click.disable_unicode_literals_warning = True


###############################################################################


@click.command(name="list-envs")
@click.option("-a", "--all", is_flag=True)
def cli_list_envs(all):
    """Lists the environments registered by the robotrl package."""
    environments = [env_spec.id for env_spec in gym.envs.registry.all()
                    if all or env_spec.entry_point.startswith("robotrl")]
    environments.sort()

    if environments:
        for env_id in environments:
            print("{}".format(env_id))
    else:
        print("robotrl environments not registered!")

    print("Listed", len(environments), "environment(s)", file=sys.stderr)

    sys.exit(0)
