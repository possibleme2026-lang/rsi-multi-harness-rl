"""Multi-harness agentic RL experiment.

Measures the cross-harness generalization gap:
    gap = mean(reward | train harnesses) - mean(reward | held-out harness)
"""
