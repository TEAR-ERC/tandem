Run model
=========

Tandem simulation can be run with a syntax of :code:`$ tandem parameter_file.toml {tandem options} --petsc {PETSc options}`.
E.g.,

.. code:: console

   $ ./tandem ../examples/tandem/2d/tutorial.toml --discrete_green yes --petsc -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -ts_type rk -ts_rk_type 5dp -ts_rtol 1e-8 -ts_atol 1e-50 -ts_adapt_wnormtype infinity -ts_dt 0.0001 -ts_monitor

:code:`--discrete_green yes` is equivalent to adding :code:`mode=QDGreen` in the paramter file.


Alternatively, to make the command line neater, multiple PETSc options can be written in a file and loaded using the :code:`-options_file` option.
E.g.,

.. code:: console

   $ ./tandem ../examples/tandem/2d/tutorial.toml --discrete_green yes --petsc -options_file solver.cfg


where :code:`solver.cfg` becomes:

.. code:: bash
   
   -ksp_type preonly
   -pc_type lu
   -pc_factor_mat_solver_type mumps

   -ts_type rk
   -ts_rk_type 5dp
   -ts_rtol 1e-8
   -ts_atol 1e-50
   -ts_adapt_wnormtype infinity

   -ts_dt 0.0001
   -ts_monitor