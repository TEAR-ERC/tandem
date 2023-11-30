
import numpy as np
import PetscBinaryIO as pio

def PetscVecLoadFromFile(fname, **kwags):
  io = pio.PetscBinaryIO(**kwags) # Instantiate a petsc binary loader
  with open(fname) as fp:
    objecttype = io.readObjectType(fp)
    v = io.readVec(fp)
  return v


x = PetscVecLoadFromFile('TS-000056.bin')
print(x)
