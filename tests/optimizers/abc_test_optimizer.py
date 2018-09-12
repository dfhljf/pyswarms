#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pyswarms.utils.functions.single_obj import sphere

class ABCTestOptimizer(object):
    """Abstract class that defines various tests for high-level optimizers

    Whenever an optimizer implementation inherits from ABCTestOptimizer,
    you don't need to write down all tests anymore. Instead, you can just
    specify all required fixtures in the test suite.
    """

    @pytest.fixture
    def optimizer(self):
        """Return an instance of the optimizer"""
        raise NotImplementedError("NotImplementedError::optimizer")

    @pytest.fixture
    def optimizer_history(self):
        """Run the optimizer for 1000 iterations and return its instance"""
        raise NotImplementedError("NotImplementedError::optimizer_history")


    @pytest.fixture
    def optimizer_reset(self):
        """Reset the optimizer and return its instance"""
        raise NotImplementedError("NotImplementedError::optimizer_reset")

    @pytest.fixture
    def options(self):
        """Default options dictionary for most PSO use-cases"""
        return {"c1": 0.5, "c2": 0.7, "w": 0.5, "k": 2, "p": 2, "r": 1}

    @pytest.mark.parametrize(
        "history, expected_shape",
        [
            ("cost_history", (1000,)),
            ("mean_pbest_history", (1000,)),
            ("mean_neighbor_history", (1000,)),
            ("pos_history", (1000, 10, 2)),
            ("velocity_history", (1000, 10, 2)),
        ],
    )
    def test_training_history_shape(self, optimizer_history, history, expected_shape):
        """Test if training histories are of expected shape"""
        opt = vars(optimizer_history)
        assert np.array(opt[history]).shape == expected_shape


    def test_reset_default_values(self, optimizer_reset):
        """Test if best cost and best pos are set properly when the reset()
        method is called"""
        assert optimizer_reset.swarm.best_cost == np.inf
        assert set(optimizer_reset.swarm.best_pos) == set(np.array([]))


    def test_ftol_effect(self, options, optimizer):
        """Test if setting the ftol breaks the optimization process accodingly"""
        opt = optimizer(10, 2, options=options, ftol=1e-1)
        opt.optimize(sphere, 2000)
        assert np.array(opt.cost_history).shape != (2000,)
