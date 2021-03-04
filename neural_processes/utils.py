#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def unravel(idx):
    """
    pytorch will apply function vectorization, s.t. batch processing is possible
    unravel linear index [0, 28*28) to cartesian coordinates
    """

    col = idx % 28
    row = idx // 28

    return row, col
