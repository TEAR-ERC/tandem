#!/usr/bin/env python3
"""
Tools for making varitions from the base model
Last modification: 2023.06.15.
by Jeena Yun
"""

import numpy as np
import setup_shortcut

sc = setup_shortcut.setups()

class variate:
    def __init__(self):
        self.yr2sec = 365*24*60*60
        self.wk2sec = 7*24*60*60
        self.setup_dir = sc.setup_dir

    def load_parameter(self,prefix):
        fsigma,ff0,fab,fdc,newb,newL,dz = self.what_is_varied(prefix)
        y,Hs,a,b,a_b,tau0,sigma0,L,others = self.fractal_param(prefix,fsigma,fab,fdc,ff0)
        y,Hs,a,b,a_b,tau0,sigma0,L,others = self.change_uniform(y,Hs,a,b,a_b,tau0,sigma0,L,others,newb,newL)
        return y,Hs,a,b,a_b,tau0,sigma0,L,others

    def same_length(self,y,input_var,target_y):
        from scipy import interpolate
        f_var = interpolate.interp1d(y,input_var,bounds_error=False,fill_value=input_var[0])
        output_var = f_var(target_y)
        return output_var

    def get_model_n(self,prefix,indicator):
        if 'BP1' in prefix:
            model_n = None
        elif len(prefix.split('/')[-1].split(indicator)) > 1:
            if len(prefix.split('/')[-1].split(indicator)[-1].split('_')) > 1:
                try:
                    model_n = int(prefix.split('/')[-1].split(indicator)[-1].split('_')[0])
                except ValueError:
                    print('might not be a fractal model - returning None')
                    model_n = None
            else:
                try:
                    model_n = int(prefix.split('/')[-1].split(indicator)[-1])
                except ValueError:
                    print('might not be a fractal model - returning None')
                    model_n = None
        else:
            model_n = None

        if 'hetero_stress' in prefix and indicator == 'v' and model_n is None:
            model_n = 0
            
        return model_n

    def what_is_varied(self,prefix):
        fsigma = self.get_model_n(prefix,'v')
        ff0 = self.get_model_n(prefix,'f0')
        fab = self.get_model_n(prefix,'ab')
        fdc = self.get_model_n(prefix,'Dc')
        newb = self.get_model_n(prefix,'B')
        newL = self.get_model_n(prefix,'L')
        dz = self.get_model_n(prefix,'DZ')

        if fsigma is not None:
            print('Fractal normal stress model ver.%d'%(fsigma))
        if ff0 is not None:
            print('Fractal f0 model ver.%d'%(ff0))
        if fab is not None:
            print('Fractal a-b model ver.%d'%(fab))
        if fdc is not None:
            print('Fractal Dc model ver.%d'%(fdc))
        if dz is not None:
            print('DZ included')

        if fsigma is None and ff0 is None and fab is None and newb is None and fdc is None and newL is None and dz is None:
            print('No parameters changed - returning regular output')   

        return fsigma,ff0,fab,fdc,newb,newL,dz

    def change_uniform(self,y,Hs,_a,_b,a_b,tau0,sigma0,_L,others,newb=None,newL=None):
        if newb is not None:
            print('Value for b changed:',_b[0],'->',newb/1000)
            b = newb / 1000 * np.ones(len(_b))
            a = a_b + b
        else:
            b = _b
            a = _a

        if newL is not None:
            print('Value for Dc homogeneously changed:',_L[0],'->',newL/10000)
            L = newL / 10000 * np.ones(len(_L))
        else:
            L = _L

        # if newHs is not None:
        #     print('Value for b changed:',_b[0],'->',newb)
        #     Hs = newHs
        # else:
        #     Hs = _Hs
        
        return y,Hs,a,b,a_b,tau0,sigma0,L,others

    def read_fractal_file(self,fname):
        print('Using file %s'%(fname.split('/')[-1]))
        fid = open(fname,'r')
        lines = fid.readlines()
        mesh_y = []
        het_var = []
        c = 0
        for line in lines:
            if line[0].isspace():
                continue
            if len(line.strip()) == 0:
                continue
            
            _y, _sn = line.split('\t')
            mesh_y.append(float(_y)); het_var.append(float(_sn.strip()))
            c += 1
        fid.close()
        mesh_y = np.array(mesh_y)
        het_var = np.array(het_var)

        return [het_var,mesh_y]

    def fractal_param(self,prefix,fsigma=None,fab=None,fdc=None,ff0=None):
        y,Hs,_a,_b,_ab,tau0,_sigma0,_L,_others = sc.params(prefix)
        others = _others

        if fsigma is not None:
            if fsigma == 0:
                fname = '%s/Thakur20_hetero_stress/fractal_snpre'%(self.setup_dir)
            elif 'litho' in prefix:
                fname = '%s/lithostatic_sn/fractal_litho_snpre_%02d'%(self.setup_dir,fsigma)
            else:
                fname = '%s/Thakur20_hetero_stress/fractal_snpre_%02d'%(self.setup_dir,fsigma)
            sigma0 = self.read_fractal_file(fname)
        else:
            sigma0 = _sigma0

        if fab is not None:
            fname = '%s/Thakur20_various_fractal_profiles/fractal_ab_%02d'%(self.setup_dir,fab)
            a_b = self.read_fractal_file(fname)
            b = self.same_length(y,_b,a_b[1])
            a = a_b[0] + b
        else:
            a_b = _ab
            a = _a
            b = _b

        if ff0 is not None:
            fname = '%s/Thakur20_various_fractal_profiles/fractal_f0_%02d'%(self.setup_dir,ff0)
            het_f0 = self.read_fractal_file(fname)
            others[-1] = het_f0

        if fdc is not None:
            fname = '%s/Thakur20_various_fractal_profiles/fractal_Dc_%02d'%(self.setup_dir,fdc)
            het_dc = self.read_fractal_file(fname)
            L = het_dc
        else:
            L = _L

        return y,Hs,a,b,a_b,tau0,sigma0,L,others
    
    def GenerateRoughTopography(self, mesh_y, lambdaMin, lambdaMax, L, H, targetHrms=None):
        # Modified from Thomas's original code - application in 1-D
        # Fourier transform method: see Andrews & Barall (2011) and Shi & Day (2013)
        N = len(mesh_y)
        Nmax = int(L / lambdaMin) + 1
        if Nmax >= N / 2:
            ValueError(f"Nmax={Nmax}>=N/2({N/2}), increase N")

        np.random.seed(3)
        a = np.zeros(N, dtype=complex)
        beta = 2 * H + 1.0
        for i in range(0, Nmax + 1):
            randPhase = np.random.rand() * np.pi * 2.0
            if i == 0:
                continue
            elif (lambdaMax * i / L) ** 2 < 1.0:
                # remove lambda>lambdaMaxS
                fac = 0.0
            elif (lambdaMin * i / L) ** 2 > 1.0:
                # remove lambda<lambdaMin
                fac = 0.0
            else:
                fac = np.power(i, -beta)

            a[i] = fac * np.exp(randPhase * 1.0j)
            if i != 0:
                a[N - i] = np.conjugate(a[i])

        a = a * N
        h = np.real(np.fft.ifft(a))

        dx = 1.0 * L / (N - 1)
        x = np.arange(0, L + dx, dx)
        nx = x.shape[0]
        h = h[0:nx]

        if targetHrms is not None:
            print('Scaled by target Hrms: %2.4f'%(targetHrms))
            hrms = np.std(h)
            h = h * targetHrms / hrms

        return h
    
    def generate_mesh_points(self,based_on_mesh,start_y=0,end_y=24,npoints=2000):
        # ----- Mesh points where normal stress will be evaluated
        if based_on_mesh:
            print('Method 1) Read in actaul mesh points')  # - not used anymore
            fid = open('%s/Thakur20_hetero_stress/meshpoints.txt'%(self.setup_dir),'r')
            lines = fid.readlines()
            mesh_x = []
            mesh_y = []
            c = 0
            for line in lines:
                if line[0].isspace():
                    continue
                if line[0:4] == 'Comp':
                    break
                
                _x, _y = line.split('\t')
                mesh_x.append(float(_x)); mesh_y.append(float(_y.strip()))
                c += 1

            print('Total %d points'%c)
            fid.close()
            idx = np.argsort(np.array(mesh_y))
            mesh_x = np.array(mesh_x)[idx]; mesh_y = np.array(mesh_y)[idx]
        else:
            print('Method 2) evenly distributed dense points throughout the fault')
            mesh_y = np.linspace(start_y,-abs(end_y),npoints)
            
        return mesh_y
    
    def version_info(self,prefix):
        fsigma,ff0,fab,fdc,newb,newL,dz = self.what_is_varied(prefix)
        ver_info = ''
        if fsigma is not None:
            ver_info += 'Stress ver.%d '%fsigma
        if ff0 is not None: 
            ver_info += '+ f0 ver.%d '%fsigma
        if fab is not None:
            ver_info += '+ a-b ver.%d '%fab
        if fdc is not None:
            ver_info += '+ Dc ver.%d '%fdc
        if newb is not None:
            ver_info += '+ b = %2.4f '%newb
        if newL is not None:
            ver_info += '+ L = %2.4f'%newL/10000
        if dz is not None:
            ver_info += '+ DZ'

        if ver_info[:2] == '+ ':
            ver_info = ver_info[2:]
            
        return ver_info