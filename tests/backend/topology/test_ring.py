#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from .abc_test_topology import ABCTestTopology
from pyswarms.backend.topology import Ring


class TestRingTopology(ABCTestTopology):
    @pytest.fixture
    def topology(self):
        return Ring

    @pytest.fixture
    def options(self):
        return {"p":1, "k":2}

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize("k", [i for i in range(1, 10)])
    @pytest.mark.parametrize("p", [1, 2])
    def test_compute_gbest_return_values(self, swarm, topology, p, k, static):
        """Test if update_gbest_neighborhood gives the expected return values"""
        topo = topology(static=static, p=p, k=k)
        pos, cost = topo.compute_gbest(swarm)
        expected_pos = np.array([9.90438476e-01, 2.50379538e-03, 1.87405987e-05])
        expected_cost = 1.0002528364353296
        assert cost == pytest.approx(expected_cost)
        assert pos == pytest.approx(expected_pos)

