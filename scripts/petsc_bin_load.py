'''
Reads in petsc binary files
By Jeena Yun, modified from Dave May
Last modification: 2023.11.25.
'''
import numpy as np
import PetscBinaryIO as pio

def PetscVecLoadFromFile(fname, **kwags):
  io = pio.PetscBinaryIO(**kwags) # Instantiate a petsc binary loader
  with open(fname) as fp:
    objecttype = io.readObjectType(fp)
    v = io.readVec(fp)
  return v

correct,total = 0,0
base_dir = 'BP1/trajectories'
dir1 = 'outputcheck_base_QDG'
dir2 = 'outputcheck_mid1_QDG'
dir3 = 'outputcheck_mid2_QDG'
for i in range(101):
  x1 = PetscVecLoadFromFile('%s/%s/TS-%06d.bin'%(base_dir,dir1,i))
  if i < 50:
    x2 = PetscVecLoadFromFile('%s/%s/TS-%06d.bin'%(base_dir,dir2,i))
    crit = [np.all(x1==x2)]
  elif i > 50:
    x3 = PetscVecLoadFromFile('%s/%s/TS-%06d.bin'%(base_dir,dir3,i))
    crit = [np.all(x1==x3)]
  elif i == 50:
    x2 = PetscVecLoadFromFile('%s/%s/TS-%06d.bin'%(base_dir,dir2,i))
    x3 = PetscVecLoadFromFile('%s/%s/TS-%06d.bin'%(base_dir,dir3,i))
    crit = [np.all(x1==x2),np.all(x1==x3)]
  if len(crit) == 1:
    print('(i = %d) All components the same? -->'%(i),crit[0])
    if crit[0]: correct += 1
  else:
    print('(i = %d) All components the same? -->'%(i),crit[0],'(for mid1);',crit[1],'(for mid1)')
    if crit[0] and crit[1]: correct += 1
  total += 1 
print('=================================')
print('In summary, %d out of %d checkpoints are completely agreeing'%(correct,total))

# x1 = PetscVecLoadFromFile('%s/%s/TS-%06d.bin'%(base_dir,dir1,35))
# x2 = PetscVecLoadFromFile('%s/%s/TS-%06d.bin'%(base_dir,dir2,34))
# crit = np.all(x1==x2)
# print(crit)
