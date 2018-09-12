#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from .abc_test_optimizer import ABCTestOptimizer
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere


class TestGlobalBest(ABCTestOptimizer):

    def test_another_stuff(self):
        assert True
