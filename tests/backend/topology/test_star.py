#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from .abc_test_topology import ABCTestTopology
from pyswarms.backend.topology import Star


class TestStarTopology(ABCTestTopology):
    @pytest.fixture
    def topology(self):
        return Star

    @pytest.fixture
    def options(self):
        return {}

    def test_compute_gbest_return_values(self, swarm, topology):
        """Test if compute_gbest() gives the expected return values"""
        topo = topology()
        expected_cost = 1.0002528364353296
        expected_pos = np.array(
            [9.90438476e-01, 2.50379538e-03, 1.87405987e-05]
        )
        pos, cost = topo.compute_gbest(swarm)
        assert cost == pytest.approx(expected_cost)
        assert pos == pytest.approx(expected_pos)
