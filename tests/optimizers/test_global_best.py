#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from .abc_test_optimizer import ABCTestOptimizer
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere


class TestGlobalBestOptimizer(ABCTestOptimizer):
    @pytest.fixture
    def optimizer(self):
        return GlobalBestPSO

    @pytest.fixture
    def optimizer_history(self, options):
        pso = GlobalBestPSO(10, 2, options=options)
        pso.optimize(sphere, 1000)
        return pso

    @pytest.fixture
    def optimizer_reset(self, options):
        pso = GlobalBestPSO(10, 2, options=options)
        pso.optimize(sphere, 10)
        pso.reset()
        return pso
