#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests"""

import pytest

from pyswarms.backend.topology import Pyramid, Random, Ring, Star, VonNeumann
from pyswarms.discrete import BinaryPSO
from pyswarms.single import GeneralOptimizerPSO, GlobalBestPSO, LocalBestPSO
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


@pytest.fixture(scope="module")
def gbest_history():
    """Returns a GlobalBestPSO instance run for 1000 iterations for checking
    history"""
    pso = GlobalBestPSO(10, 2, {"c1": 0.5, "c2": 0.7, "w": 0.5})
    pso.optimize(sphere, 1000)
    return pso


@pytest.fixture(scope="module")
def gbest_reset():
    """Returns a GlobalBestPSO instance that has been run and reset to check
    default value"""
    pso = GlobalBestPSO(10, 2, {"c1": 0.5, "c2": 0.7, "w": 0.5})
    pso.optimize(sphere, 10)
    pso.reset()
    return pso


@pytest.fixture(scope="module")
def lbest_history():
    """Returns a LocalBestPSO instance run for 1000 iterations for checking
    history"""
    pso = LocalBestPSO(10, 2, {"c1": 0.5, "c2": 0.7, "w": 0.5, "k": 2, "p": 2})
    pso.optimize(sphere, 1000)
    return pso


@pytest.fixture(scope="module")
def lbest_reset():
    """Returns a LocalBestPSO instance that has been run and reset to check
    default value"""
    pso = LocalBestPSO(10, 2, {"c1": 0.5, "c2": 0.7, "w": 0.5, "k": 2, "p": 2})
    pso.optimize(sphere, 10)
    pso.reset()
    return pso


@pytest.fixture(scope="module")
def binary_history():
    """Returns a BinaryPSO instance run for 1000 iterations for checking
    history"""
    pso = BinaryPSO(10, 2, {"c1": 0.5, "c2": 0.7, "w": 0.5, "k": 2, "p": 2})
    pso.optimize(sphere, 1000)
    return pso


@pytest.fixture(scope="module")
def binary_reset():
    """Returns a BinaryPSO instance that has been run and reset to check
    default value"""
    pso = BinaryPSO(10, 2, {"c1": 0.5, "c2": 0.7, "w": 0.5, "k": 2, "p": 2})
    pso.optimize(sphere, 10)
    pso.reset()
    return pso


@pytest.fixture
def options():
    """Default options dictionary for most PSO use-cases"""
    options_ = {"c1": 0.5, "c2": 0.7, "w": 0.5, "k": 2, "p": 2, "r": 1}
    return options_


# fmt: off
@pytest.fixture(params=[
                Star(),
                Ring(static=False), Ring(static=True),
                Pyramid(static=False), Pyramid(static=True),
                Random(static=False), Random(static=True),
                VonNeumann()
                ])
# fmt: on
def topology(request):
    """Parametrized topology parameter"""
    topology_ = request.param
    return topology_
