import numpy as np
 
R = 8.314E-3 #MJ/(kmol K)
#A_0 = np.array([2.8E19,3.28E14,1.3E10,2.1e16,8.75E15,2.6E11,9.6E8,1.5E9,7.7E6,4.25E6])
A_0 = np.array([2.8E19,3.28E14,1.3E10,2.1e16,8.75E15,2.6E11,9.6E8,1.5E9,7.7E6,0])
EA_0 = np.array([242.4,196.5,150.5,186.7,202.4,145.7,107.6,143.8,111.4,108.0])
m0 = np.array([0.40177, #Cellulose
                0.23887, # hemicellulos
                0.30757, # lignin
                0, # active cellulose
                0, # active hemicellulose
                0, # active lignin
                0, # tar
                0, # gas
                0, # char
                ])

h = 400
cp = 2300 
rho = 540 #kg/m^3
MW_char = 0.1237 #kg/mol
MW_tar = 0.1237 #kg/mol, yes they are the same

Y_c = 0.35
Y_h = 0.6
Y_l = 0.75

ds_mean_integrated = np.array([4.18844032e-08, 5.36778057e-05])
ds_std_integrated = np.array([6.65054621e-08, 5.84519135e-05])
ds_mean = np.array([5.0, 4.0])
ds_std = np.array([5.0, 4.0])

S0 = np.concatenate([[623],m0])

y_ax_map = {'tar_mol':-3,'char_mol':-1}