#!/usr/bin/env python3
"""
Tools for making varitions from the base model
Last modification: 2023.11.08.
by Jeena Yun
"""
import numpy as np
import setup_shortcut
import os

sc = setup_shortcut.setups()

class variate:
    def __init__(self):
        self.yr2sec = 365*24*60*60
        self.wk2sec = 7*24*60*60

    def get_setup_dir(self):
        if 'j4yun' in os.getcwd(): # local
            setup_dir = '/Users/j4yun/Dropbox/Codes/Ridgecrest_CSC/jeena-tandem/setup_files/supermuc'
        elif 'di75weg' in os.getcwd(): # supermuc
            setup_dir = '/hppfs/work/pn49ha/di75weg/jeena-tandem/setup_files/supermuc'
        elif 'jyun' in os.getcwd(): # LMU server
            setup_dir = '/home/jyun/Tandem'
        return setup_dir
    
    def extract_prefix(self,save_dir):
        if 'models' in save_dir: # local
            prefix = save_dir.split('models/')[-1]
        elif 'di75weg' in save_dir: # supermuc
            prefix = save_dir.split('di75weg/')[-1]
        elif 'jyun' in save_dir: # LMU server
            prefix = save_dir.split('jyun/')[-1]
        return prefix
    
    def load_parameter(self,prefix,print_on=True):
        fsigma,ff0,fab,fdc,newb,newL,dz = self.what_is_varied(prefix,print_on)
        y,Hs,a,b,a_b,tau0,sigma0,L,others = self.fractal_param(prefix,fsigma,fab,fdc,ff0,print_on)
        y,Hs,a,b,a_b,tau0,sigma0,L,others = self.change_uniform(y,Hs,a,b,a_b,tau0,sigma0,L,others,newb,newL,print_on)
        return y,Hs,a,b,a_b,tau0,sigma0,L,others

    def same_length(self,y,input_var,target_y):
        from scipy import interpolate
        f_var = interpolate.interp1d(y,input_var,bounds_error=False,fill_value=input_var[0])
        output_var = f_var(target_y)
        return output_var

    def get_model_n(self,prefix,indicator,print_on=True):
        if 'BP1' in prefix:
            model_n = None
        elif indicator == 'DZ' and indicator in prefix:
            model_n = 0
        elif len(prefix.split('/')[-1].split(indicator)) > 1:
            if len(prefix.split('/')[-1].split(indicator)[-1].split('_')) > 1:
                try:
                    model_n = int(prefix.split('/')[-1].split(indicator)[-1].split('_')[0])
                except ValueError:
                    if print_on: print('might not be a fractal model - returning None')
                    model_n = None
            else:
                try:
                    model_n = int(prefix.split('/')[-1].split(indicator)[-1])
                except ValueError:
                    if print_on: print('might not be a fractal model - returning None')
                    model_n = None
        else:
            model_n = None

        if 'hetero_stress' in prefix and indicator == 'v' and model_n is None:
            model_n = 0
        
        if 'perturb_stress' in prefix:
            if indicator == 'v':
                model_n = 6
            if indicator == 'ab':
                model_n = 2
            if indicator == 'Dc':
                model_n = 2

        return model_n

    def what_is_varied(self,prefix,print_on=True):
        fsigma = self.get_model_n(prefix,'v')
        ff0 = self.get_model_n(prefix,'f0')
        fab = self.get_model_n(prefix,'ab')
        fdc = self.get_model_n(prefix,'Dc')
        newb = self.get_model_n(prefix,'B')
        newL = self.get_model_n(prefix,'L')
        dz = self.get_model_n(prefix,'DZ')

        if print_on:
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

    def change_uniform(self,y,Hs,_a,_b,a_b,tau0,sigma0,_L,others,newb=None,newL=None,print_on=True):
        if newb is not None:
            if print_on: print('Value for b changed:',_b[0],'->',newb/1000)
            b = newb / 1000 * np.ones(len(_b))
            a = a_b + b
        else:
            b = _b
            a = _a

        if newL is not None:
            if print_on: print('Value for Dc homogeneously changed:',_L[0],'->',newL/10000)
            L = newL / 10000 * np.ones(len(_L))
        else:
            L = _L

        # if newHs is not None:
        #     print('Value for b changed:',_b[0],'->',newb)
        #     Hs = newHs
        # else:
        #     Hs = _Hs
        
        return y,Hs,a,b,a_b,tau0,sigma0,L,others

    def read_fractal_file(self,fname,print_on=True):
        if print_on: print('Using file %s'%(fname.split('/')[-1]))
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

    def fractal_param(self,prefix,fsigma=None,fab=None,fdc=None,ff0=None,print_on=True):
        y,Hs,_a,_b,_ab,tau0,_sigma0,_L,_others = sc.params(prefix)
        others = _others

        if fsigma is not None:
            if fsigma == 0:
                fname = '%s/Thakur20_hetero_stress/fractal_snpre'%(self.get_setup_dir())
            elif 'litho' in prefix:
                fname = '%s/lithostatic_sn/fractal_litho_snpre_%02d'%(self.get_setup_dir(),fsigma)
            else:
                fname = '%s/Thakur20_hetero_stress/fractal_snpre_%02d'%(self.get_setup_dir(),fsigma)
            sigma0 = self.read_fractal_file(fname,print_on)
        else:
            sigma0 = _sigma0

        if fab is not None:
            fname = '%s/Thakur20_various_fractal_profiles/fractal_ab_%02d'%(self.get_setup_dir(),fab)
            a_b = self.read_fractal_file(fname,print_on)
            b = self.same_length(y,_b,a_b[1])
            a = a_b[0] + b
        else:
            a_b = _ab
            a = _a
            b = _b

        if ff0 is not None:
            fname = '%s/Thakur20_various_fractal_profiles/fractal_f0_%02d'%(self.get_setup_dir(),ff0)
            het_f0 = self.read_fractal_file(fname,print_on)
            others[-1] = het_f0

        if fdc is not None:
            fname = '%s/Thakur20_various_fractal_profiles/fractal_Dc_%02d'%(self.get_setup_dir(),fdc)
            het_dc = self.read_fractal_file(fname,print_on)
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

        np.random.seed(3) # all profiles except Dc_04,05,06,07 or snpre_07
        # rs = np.random.randint(100); print(rs) # fractal_Dc_04, 07
        # np.random.seed(rs)
        # np.random.seed(50) # fractal_Dc_04 and 07 / 98 for Dc_05, 30 for Dc_06
        # np.random.seed(12) # fractal_snpre_07 / 62 for snpre_08

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
            fid = open('%s/Thakur20_hetero_stress/meshpoints.txt'%(self.get_setup_dir()),'r')
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
        fsigma,ff0,fab,fdc,newb,newL,dz = self.what_is_varied(prefix,print_on=False)
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
            ver_info += '+ L = %2.4f'%(newL/10000)
        if dz is not None:
            ver_info += '+ DZ'

        if ver_info[:2] == '+ ':
            ver_info = ver_info[2:]
            
        return ver_info
    
    def make_uniform(self,prefix,print_on=True):
        y,Hs,a,b,a_b,tau0,sigma0,Dc,others = self.load_parameter(prefix,print_on)
        l1,l2,l3=0,0,0
        if len(sigma0) == 2:
            if print_on: print('Heterogeneous normal stress profile')
            l1 = len(sigma0[1])
        if len(a_b) == 2:
            if print_on: print('Heterogeneous a-b profile')
            l2 = len(a_b[1])
        if len(Dc) == 2:
            if print_on: print('Heterogeneous Dc profile')
            l3 = len(Dc[1])
        if sum([l1,l2,l3]) > 0:
            base = np.argmax([l1,l2,l3])
            if print_on: print('Lengths:',l1,l2,l3,'-> base:',base)
            meshs = [sigma0[1],a_b[1],Dc[1]]
            mesh_y = meshs[base]
        else:
            if print_on: print('Do not need to rebase')
            mesh_y = y

        if l1 > 0 and 0 == base:
            if print_on: print('base = sigma0')
            _sigma = sigma0[0]
        elif l1 > 0:
            _sigma = self.same_length(sigma0[1],sigma0[0],mesh_y)
        else:
            _sigma = self.same_length(y,sigma0,mesh_y)

        if l2 > 0 and 1 == base:
            if print_on: print('base = a-b')
            _ab = a_b[0]
            _a = a
            _b = b
        elif l2 > 0:
            _ab = self.same_length(a_b[1],a_b[0],mesh_y)
            _a = self.same_length(a_b[1],a,mesh_y)
            _b = self.same_length(a_b[1],b,mesh_y)
        else:
            _ab = self.same_length(y,a_b,mesh_y)
            _a = self.same_length(y,a,mesh_y)
            _b = self.same_length(y,b,mesh_y)

        if l3 > 0 and 2 == base:
            if print_on: print('base = Dc')
            _Dc = Dc[0]
        elif l3 > 0:
            _Dc = self.same_length(Dc[1],Dc[0],mesh_y)
        else:
            _Dc = self.same_length(y,Dc,mesh_y)
            
        return mesh_y,_a,_b,_ab,_sigma,_Dc