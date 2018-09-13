#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from .abc_test_topology import ABCTestTopology
from pyswarms.backend.topology import Random


class TestRandomTopology(ABCTestTopology):
    @pytest.fixture
    def topology(self):
        return Random

    @pytest.fixture
    def options(self):
        return {"k": 2}

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize("k", [1, 2])
    def test_compute_gbest_return_values(self, swarm, topology, k, static):
        """Test if update_gbest_neighborhood gives the expected return values"""
        topo = topology(static=static, k=k)
        pos, cost = topo.compute_gbest(swarm)
        expected_pos = np.array(
            [9.90438476e-01, 2.50379538e-03, 1.87405987e-05]
        )
        expected_cost = 1.0002528364353296
        assert cost == pytest.approx(expected_cost)
        assert pos == pytest.approx(expected_pos)

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize("k", [1, 2])
    def test_compute_neighbors_return_values(self, swarm, topology, k, static):
        """Test if __compute_neighbors() gives the expected shape and symmetry"""
        topo = topology(static=static, k=k)
        adj_matrix = topo._Random__compute_neighbors(swarm, k=k)
        assert adj_matrix.shape == (swarm.n_particles, swarm.n_particles)
        assert np.allclose(
            adj_matrix, adj_matrix.T, atol=1e-8
        )  # Symmetry test

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize("k", [1])
    def test_compute_neighbors_adjacency_matrix(
        self, swarm, topology, k, static
    ):
        """Test if __compute_neighbors() gives the expected matrix"""
        np.random.seed(1)
        topo = topology(static=static, k=k)
        adj_matrix = topo._Random__compute_neighbors(swarm, k=k)
        # fmt: off
        comparison_matrix = np.array([[1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                      [0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
                                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                      [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
                                      [0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
                                      [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
                                      [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                                      [1, 1, 1, 0, 1, 1, 1, 0, 0, 1]])
        assert np.allclose(adj_matrix, comparison_matrix, atol=1e-8)
        # fmt: on
