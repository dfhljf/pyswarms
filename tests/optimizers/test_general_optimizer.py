#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest


import pyswarms.backend.topology as t
from .abc_test_optimizer import ABCTestOptimizer
from pyswarms.single import GeneralOptimizerPSO
from pyswarms.utils.functions.single_obj import sphere

topologies = [
        t.Star(),
        t.Pyramid()
        ]


class TestGeneralOptimizer(ABCTestOptimizer):

    @pytest.fixture
    def optimizer(self):
        return GeneralOptimizerPSO

    @pytest.fixture
    def optimizer_history(self, request):
        opt = GeneralOptimizerPSO(n_particles=10, dimensions=2,
                                  options={"c1": 0.5, "c2": 0.7, "w": 0.5},
                                  topology=request.param)
        opt.optimize(sphere, 1000)
        return opt

    @pytest.fixture(params=topologies)
    def optimizer_reset(self, request):
        opt = GeneralOptimizerPSO(n_particles=10, dimensions=2,
                                  options={"c1": 0.5, "c2": 0.7, "w": 0.5},
                                  topology=request.param)
        opt.optimize(sphere, 1000)
        opt.reset()
        return opt
