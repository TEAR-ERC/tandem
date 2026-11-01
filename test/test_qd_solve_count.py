"""
Linear solves of a non-viscoelastic QD run with a scheme without FSAL.

Runs bp1_ref (Poisson, QD) with classic RK4 at a fixed step and no output, see
scripts/2D/qd_solve_count.sh. Each RK stage solves once and the initial condition
solves twice. With no writer the monitor needs no solve, and an operator without
viscoelastic strain history must not solve after an accepted step either: that
post-step solve exists to update the strain history, and for an elastic run it
was one extra global linear solve per step (5 instead of 4 per RK4 step).
"""

RK4_STAGES = 4
INITIAL_CONDITION_SOLVES = 2


def test_qd_solve_count(temp_results_path, read_tandem_summary):
    summary = read_tandem_summary(temp_results_path / "qd_solve_count.log")
    steps = summary["time_steps"]
    assert steps > 0
    assert summary["step_rejections"] == 0

    expected = INITIAL_CONDITION_SOLVES + RK4_STAGES * steps
    assert summary["linear_solves"] == expected, (
        f"linear_solves = {summary['linear_solves']:.0f}, expected "
        f"{INITIAL_CONDITION_SOLVES} + {RK4_STAGES} x {steps:.0f} steps = {expected:.0f}"
    )
    # A non-viscoelastic operator never depends on dt.
    assert summary["operator_reassemblies"] == 0
