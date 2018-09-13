#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np


import pyswarms.backend.topology as t
from .abc_test_optimizer import ABCTestOptimizer
from pyswarms.single import GeneralOptimizerPSO
from pyswarms.utils.functions.single_obj import sphere

# Just integration test, supplying
# random values
topologies = [t.Star(), t.Pyramid()]


class TestGeneralOptimizer(ABCTestOptimizer):
    @pytest.fixture(params=topologies)
    def optimizer(self, request):
        return GeneralOptimizerPSO(
            n_particles=10,
            dimensions=2,
            options={"c1": 0.5, "c2": 0.7, "w": 0.5},
            topology=request.param,
        )

    @pytest.fixture
    def optimizer_history(self, request):
        opt = GeneralOptimizerPSO(
            n_particles=10,
            dimensions=2,
            options={"c1": 0.5, "c2": 0.7, "w": 0.5},
            topology=request.param,
        )
        opt.optimize(sphere, 1000)
        return opt

    @pytest.fixture(params=topologies)
    def optimizer_reset(self, request):
        opt = GeneralOptimizerPSO(
            n_particles=10,
            dimensions=2,
            options={"c1": 0.5, "c2": 0.7, "w": 0.5},
            topology=request.param,
        )
        opt.optimize(sphere, 1000)
        opt.reset()
        return opt

    def test_ftol_effect(self, optimizer):
        """Test if setting the ftol breaks the optimization process"""
        optimizer.optimize(sphere, 2000)
        assert np.array(optimizer.cost_history).shape != (2000,)

    def test_obj_with_kwargs(self, obj_with_args, optimizer, options):
        """Test if kwargs are passed properly in objfunc"""
        x_max = 10 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        opt = optimizer(100, 2, options=options, bounds=bounds)
        cost, pos = opt.optimize(obj_with_args, 1000, a=1, b=100)
        assert np.isclose(cost, 0, rtol=1e-03)
        assert np.isclose(pos[0], 1.0, rtol=1e-03)
        assert np.isclose(pos[1], 1.0, rtol=1e-03)

    def test_obj_unnecessary_kwargs(
        self, obj_without_args, optimizer, options
    ):
        """Test if error is raised given unnecessary kwargs"""
        x_max = 10 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        opt = optimizer(100, 2, options=options, bounds=bounds)
        with pytest.raises(TypeError):
            # kwargs `a` should not be supplied
            cost, pos = opt.optimize(obj_without_args, 1000, a=1)

    def test_obj_missing_kwargs(self, obj_with_args, optimizer, options):
        """Test if error is raised with incomplete kwargs"""
        x_max = 10 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        opt = optimizer(100, 2, options=options, bounds=bounds)
        with pytest.raises(TypeError):
            # kwargs `b` is missing here
            cost, pos = opt.optimize(obj_with_args, 1000, a=1)

    def test_obj_incorrect_kwargs(self, obj_with_args, optimizer, options):
        """Test if error is raised with wrong kwargs"""
        x_max = 10 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        opt = optimizer(100, 2, options=options, bounds=bounds)
        with pytest.raises(TypeError):
            # Wrong kwargs
            cost, pos = opt.optimize(obj_with_args, 1000, c=1, d=100)
