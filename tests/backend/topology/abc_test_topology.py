#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc

import numpy as np
import pytest


class ABCTestTopology(abc.ABC):
    """Abstract class that defines various tests for topologies

    Whenever a topology inherits from ABCTestTopology,
    you don't need to write down all tests anymore. Instead, you can just
    specify all required fixtures in the test suite.
    """

    @pytest.fixture
    def topology(self):
        """Return an instance of the topology"""
        raise NotImplementedError("NotImplementedError::topology")

    @pytest.fixture
    def options(self):
        """Return a dictionary of options"""
        raise NotImplementedError("NotImplementedError::options")

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize("clamp", [None, (0, 1), (-1, 1)])
    def test_compute_velocity_return_values(self, topology, options, swarm, clamp, static):
        """Test if compute_velocity() gives the expected shape and range"""
        topo = topology(static=static, **options)
        v = topo.compute_velocity(swarm, clamp)
        assert v.shape == swarm.position.shape
        if clamp is not None:
            assert (clamp[0] <= v).all() and (clamp[1] >= v).all()


    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize(
        "bounds",
        [None, ([-5, -5, -5], [5, 5, 5]), ([-10, -10, -10], [10, 10, 10])],
    )
    def test_compute_position_return_values(self, topology, options, swarm, bounds, static):
        """Test if compute_position() gives the expected shape and range"""
        topo = topology(static=static, **options)
        p = topo.compute_position(swarm, bounds)
        assert p.shape == swarm.velocity.shape
        if bounds is not None:
            assert (bounds[0] <= p).all() and (bounds[1] >= p).all()


    @pytest.mark.parametrize("static", [True, False])
    def test_neighbor_idx(self, topology, options, swarm, static):
        """Test if the neighbor_idx attribute is assigned"""
        topo = topology(static=static, **options)
        topo.compute_gbest(swarm)
        assert topo.neighbor_idx is not None
