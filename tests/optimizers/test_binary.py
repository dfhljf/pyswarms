#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest

from .abc_test_discrete_optimizer import ABCTestDiscreteOptimizer
from pyswarms.discrete import BinaryPSO
from pyswarms.utils.functions.single_obj import sphere


class TestDiscreteOptimizer(ABCTestDiscreteOptimizer):
    @pytest.fixture
    def optimizer(self):
        return BinaryPSO

    @pytest.fixture
    def optimizer_history(self, options):
        pso = BinaryPSO(10, 2, options=options)
        pso.optimize(sphere, 1000)
        return pso

    @pytest.fixture
    def optimizer_reset(self, options):
        pso = BinaryPSO(10, 2, options=options)
        pso.optimize(sphere, 10)
        pso.reset()
        return pso
