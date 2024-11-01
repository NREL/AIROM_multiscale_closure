import numpy as np
from paraview import simple as pv
import vtk.numpy_interface.dataset_adapter as dsa 
import sys
from sys import argv


def pv_calc_gas_outlet(infile, outfile):
    
    solfoam = pv.OpenFOAMReader(FileName = infile) # just need to provide folder
    solfoam.CaseType = 'Reconstructed Case'
    solfoam.MeshRegions = ['outlet']
    solfoam.CellArrays = ['alpha.gas', 'U.gas', 'SGC.gas', 'TRC.gas', 'SGH.gas', 'SGL.gas', 'TRH.gas', 'TRL.gas', 'T.gas', 'thermo:rho.gas']
    t = np.array(solfoam.TimestepValues)
    N=t.size
    print(N)
    # Calculate SG species
    calculator1 = pv.Calculator(Input=solfoam)
    calculator1.AttributeType = 'Point Data'
    calculator1.ResultArrayName = 'sgcmflux'
    calculator1.Function = 'alpha.gas*SGC.gas*U.gas_Z*thermo:rho.gas'

    calculator2 = pv.Calculator(Input=calculator1)
    calculator2.AttributeType = 'Point Data'
    calculator2.ResultArrayName = 'sghmflux'
    calculator2.Function = 'alpha.gas*SGH.gas*U.gas_Z*thermo:rho.gas'

    calculator3 = pv.Calculator(Input=calculator2)
    calculator3.AttributeType = 'Point Data'
    calculator3.ResultArrayName = 'sglmflux'
    calculator3.Function = 'alpha.gas*SGL.gas*U.gas_Z*thermo:rho.gas'

    # Calculate TR species
    calculator4 = pv.Calculator(Input=calculator3)
    calculator4.AttributeType = 'Point Data'
    calculator4.ResultArrayName = 'trcmflux'
    calculator4.Function = 'alpha.gas*TRC.gas*U.gas_Z*thermo:rho.gas'

    calculator5 = pv.Calculator(Input=calculator4)
    calculator5.AttributeType = 'Point Data'
    calculator5.ResultArrayName = 'trhmflux'
    calculator5.Function = 'alpha.gas*TRH.gas*U.gas_Z*thermo:rho.gas'

    calculator6 = pv.Calculator(Input=calculator5)
    calculator6.AttributeType = 'Point Data'
    calculator6.ResultArrayName = 'trlmflux'
    calculator6.Function = 'alpha.gas*TRL.gas*U.gas_Z*thermo:rho.gas'

    # create a new 'Integrate Variables'
    int1 = pv.IntegrateVariables(Input=calculator6)
    
    outfile=open(outfile,"w")
    outfile.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ('t[i]','area','sgcmfluxint','sghmfluxint','sglmfluxint','trcmfluxint','trhmfluxint','trlmfluxint','mflux_sg_tot','mflux_tr_tot','alphaint/area\n'))
    for i in range(N):
        try:
            pv.UpdatePipeline(time=t[i], proxy=int1)
            idat    = dsa.WrapDataObject(pv.servermanager.Fetch(int1) )
            area       = idat.CellData['Area'].item()
            sgcmfluxint   = idat.PointData['sgcmflux'].item()
            sghmfluxint   = idat.PointData['sghmflux'].item()
            sglmfluxint   = idat.PointData['sglmflux'].item()
            trcmfluxint   = idat.PointData['trcmflux'].item()
            trhmfluxint   = idat.PointData['trhmflux'].item()
            trlmfluxint   = idat.PointData['trlmflux'].item()
            alphaint   = idat.PointData['alpha.gas'].item()
            mflux_sg_tot = sgcmfluxint + sghmfluxint + sglmfluxint
            mflux_tr_tot = trcmfluxint + trhmfluxint + trlmfluxint
            #print(f'SGC: {sgcmfluxint}, SGH: {sghmfluxint} SGL: {sglmfluxint} total: {mflux_sg_tot}\n')
            #print(f'TRC: {trcmfluxint}, TRH: {trhmfluxint} TRL: {trlmfluxint} total: {mflux_tr_tot}\n')
            print("processing time = %e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n" % (t[i],area,sgcmfluxint,sghmfluxint,sglmfluxint,trcmfluxint,trhmfluxint,trlmfluxint,mflux_sg_tot,mflux_tr_tot,alphaint/area))
            outfile.write("%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n" % (t[i],area,sgcmfluxint,sghmfluxint,sglmfluxint,trcmfluxint,trhmfluxint,trlmfluxint,mflux_sg_tot,mflux_tr_tot,alphaint/area))
        except AttributeError:
            print("Skipping: ",i)
    outfile.close()


def pv_calc_biomass_outlet(infile,outfile):
    # trace generated using paraview version 5.8.1
    #
    # To ensure correct image size when batch processing, please search 
    # for and uncomment the line `# renderView*.ViewSize = [*,*]`

    #### import the simple module from the paraview


    solfoam = pv.OpenFOAMReader(FileName =infile) # just need to provide folder
    solfoam.CaseType = 'Reconstructed Case'
    solfoam.MeshRegions = ['outlet']
    solfoam.CellArrays = ['alpha.particles', 'U.particles', 'thermo:rho.particles','alpha.gas', 'U.gas', 'thermo:rho.gas']
    t = np.array(solfoam.TimestepValues)
    N=t.size

    withsurfnormals = pv.GenerateSurfaceNormals(Input=solfoam)
    # Properties modified on calculator1
    calculator1 = pv.Calculator(Input=withsurfnormals)
    calculator1.AttributeType = 'Point Data'
    calculator1.ResultArrayName = 'mfluxb'
    calculator1.Function = 'alpha.particles*(U.particles_X*Normals_X+U.particles_Y*Normals_Y+U.particles_Z*Normals_Z)*thermo:rho.particles'

    Pstd=101325.0
    Tstd=273.15
    R=287.0
    std_dens=Pstd/Tstd/R
    calculator2 = pv.Calculator(Input=calculator1)
    calculator2.AttributeType = 'Point Data'
    calculator2.ResultArrayName = 'mfluxg'
    #in slm
    calculator2.Function = 'alpha.gas*(U.gas_X*Normals_X+U.gas_Y*Normals_Y+U.gas_Z*Normals_Z)*thermo:rho.gas/%e*1000.0*60.0'%(std_dens)

    # create a new 'Integrate Variables'
    int1 = pv.IntegrateVariables(Input=calculator2)

    outfile=open(outfile,"w")
    outfile.write("%s\t%s\t%s\t%s\t%s\t%s" % ('t[i]','area','mfluxbint','alphabint/area','mfluxgint','alphagint/are'))
    
    for i in range(N):
        try:
            pv.UpdatePipeline(time=t[i], proxy=int1)
            idat    = dsa.WrapDataObject(pv.servermanager.Fetch(int1) )
            area       = idat.CellData['Area'].item()
            mfluxbint   = idat.PointData['mfluxb'].item()
            alphabint   = idat.PointData['alpha.particles'].item()
            mfluxgint   = idat.PointData['mfluxg'].item()
            alphagint   = idat.PointData['alpha.gas'].item()
            print("processing time = %e\t%e\t%e\t%e\t%e\t%e" % (t[i],area,mfluxbint,alphabint/area,mfluxgint,alphagint/area))
            outfile.write("%e\t%e\t%e\t%e\t%e\t%e\n"%(t[i],area,mfluxbint,alphabint/area,mfluxgint,alphagint/area))
        except AttributeError:
            print("Skipping: ",i)

    outfile.close()
