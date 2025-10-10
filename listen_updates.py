#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin wrapper around ``listen_start.main`` for backward compatibility.

The main polling loop now persists offsets to ``data/offset.txt`` so this
wrapper stays available for existing deployments without extra wiring.
"""

from listen_start import main

if __name__ == "__main__":
    main()
