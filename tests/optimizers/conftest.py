#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests"""

import pytest

from pyswarms.backend.topology import Pyramid, Random, Ring, Star, VonNeumann
from pyswarms.single import GeneralOptimizerPSO
from pyswarms.utils.functions.single_obj import sphere


@pytest.fixture(scope="module")
def general_opt_history(topology):
    """Returns a GeneralOptimizerPSO instance run for 1000 iterations for checking
    history"""
    pso = GeneralOptimizerPSO(
        10, 2, {"c1": 0.5, "c2": 0.7, "w": 0.5}, topology=topology
    )
    pso.optimize(sphere, 1000)
    return pso


@pytest.fixture(scope="module")
def general_opt_reset(topology):
    """Returns a GeneralOptimizerPSO instance that has been run and reset to check
    default value"""
    pso = GeneralOptimizerPSO(
        10, 2, {"c1": 0.5, "c2": 0.7, "w": 0.5}, topology=topology
    )
    pso.optimize(sphere, 10, verbose=0)
    pso.reset()
    return pso




# fmt: off
@pytest.fixture(params=[
                Star,
                Ring,
                Pyramid,
                Random,
                VonNeumann
                ])
# fmt: on
def topology(request):
    """Parametrized topology parameter"""
    topology_ = request.param
    return topology_
