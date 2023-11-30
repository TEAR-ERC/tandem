package.path = package.path .. ";/home/jyun/Tandem"
local ridgecrest54 = require "matfric_Fourier_main_perturb"

-------------------- List of libraries
vs340_41450 = ridgecrest54.new{model_n='vert_slow',strike=340,fcoeff=0.4,dt=0.01,init_time=1.381263359374180436e+06} -- testcase
vs340_8791850 = ridgecrest54.new{model_n='vert_slow',strike=340,fcoeff=0.4,dt=0.01,init_time=1.402901087547306519e+11}
vs340_1515450 = ridgecrest54.new{model_n='vert_slow',strike=340,fcoeff=0.4,dt=0.01,init_time=2.712094552433102798e+10}
vs340_7176480 = ridgecrest54.new{model_n='vert_slow',strike=340,fcoeff=0.4,dt=0.01,init_time=1.162875225174398804e+11}
vs340_1515453 = ridgecrest54.new{model_n='vert_slow',strike=340,fcoeff=0.4,dt=0.01,init_time=2.712094562139096069e+10}
vs340_8908134 = ridgecrest54.new{model_n='vert_slow',strike=340,fcoeff=0.4,dt=0.01,init_time=1.421668527168912659e+11}
vs340_4915642 = ridgecrest54.new{model_n='vert_slow',strike=340,fcoeff=0.4,dt=0.01,init_time=8.267655560204478455e+10}
vf330_4915642 = ridgecrest54.new{model_n='vert_fast',strike=330,fcoeff=0.4,dt=0.01,init_time=8.267655560204478455e+10}
ds350_4915642 = ridgecrest54.new{model_n='dipping_slow',strike=350,fcoeff=0.4,dt=0.01,init_time=8.267655560204478455e+10}
vs340_2055768 = ridgecrest54.new{model_n='vert_slow',strike=340,fcoeff=0.4,dt=0.01,init_time=3.721969163349252319e+10}
vsX2_340_1515453 = ridgecrest54.new{model_n='vert_slow_X2',strike=340,fcoeff=0.4,dt=0.01,init_time=2.712094562139096069e+10}
vsX3_340_1515453 = ridgecrest54.new{model_n='vert_slow_X3',strike=340,fcoeff=0.4,dt=0.01,init_time=2.712094562139096069e+10}
vsX5_340_1515453 = ridgecrest54.new{model_n='vert_slow_X5',strike=340,fcoeff=0.4,dt=0.01,init_time=2.712094562139096069e+10}
vsX10_340_1515453 = ridgecrest54.new{model_n='vert_slow_X10',strike=340,fcoeff=0.4,dt=0.01,init_time=2.712094562139096069e+10}
vsX30_340_1515453 = ridgecrest54.new{model_n='vert_slow_X30',strike=340,fcoeff=0.4,dt=0.01,init_time=2.712094562139096069e+10}