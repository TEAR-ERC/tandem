mpiexec -bind-to core -n 128 tandem parameter.toml --petsc -options_file lu_mumps.cfg -options_file rk45.cfg -ts_monitor > messages.log &
