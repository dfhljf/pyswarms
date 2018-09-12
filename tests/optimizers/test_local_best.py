#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from .abc_test_optimizer import ABCTestOptimizer
from pyswarms.single import LocalBestPSO
from pyswarms.utils.functions.single_obj import sphere

class TestLocalBestOptimizer(ABCTestOptimizer):

    @pytest.fixture
    def optimizer(self):
        return LocalBestPSO

    @pytest.fixture
    def optimizer_history(self, options):
        pso = LocalBestPSO(10, 2, options)
        pso.optimize(sphere, 1000)
        return pso

    @pytest.fixture
    def optimizer_reset(self, options):
        pso = LocalBestPSO(10, 2, options)
        pso.optimize(sphere, 10)
        pso.reset()
        return pso
