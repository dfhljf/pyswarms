#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from .abc_test_optimizer import ABCTestOptimizer
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere


class TestGlobalBest(ABCTestOptimizer):

    @pytest.fixture
    def optimizer(self):
        return GlobalBestPSO

    @pytest.fixture
    def optimizer_history(self):
        pso = GlobalBestPSO(10, 2, {"c1": 0.5, "c2": 0.7, "w": 0.5})
        pso.optimize(sphere, 1000)
        return pso

    @pytest.fixture
    def optimizer_reset(self):
        pso = GlobalBestPSO(10, 2, {"c1": 0.5, "c2": 0.7, "w": 0.5})
        pso.optimize(sphere, 10)
        pso.reset()
        return pso
