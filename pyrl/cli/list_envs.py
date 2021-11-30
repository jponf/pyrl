# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division, unicode_literals

import click
import gym
import sys


###############################################################################

if hasattr(click, "disable_unicode_literals_warning"):
    setattr(click, "disable_unicode_literals_warning", True)


###############################################################################


@click.command(name="list-envs")
def cli_list_envs():
    """Lists the environments registered by the robotrl package."""
    environments = [env_spec.id for env_spec in gym.envs.registry.all()]
    environments.sort()

    for env_id in environments:
        print("{}".format(env_id))

    print("Listed", len(environments), "environment(s)", file=sys.stderr)

    sys.exit(0)
