# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:33:18 2024

@author: schillings
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd
import matplotlib.animation as ani
import time as systime
import scipy.constants as co
import scipy.integrate as inte
import ast
from sys import argv
plt.close("all")
plt.rc('legend',fontsize=14)
plt.rc('axes',labelsize=15,titlesize=15)
plt.rc("xtick",labelsize=12)
plt.rc("ytick",labelsize=12)
plt.rc('figure',figsize=(10,9))
total_start_time=systime.time()

ID=int(argv[1])
tag="1Sshift005f"
folder="1Stest"
isPlaneWave=True
produceDataOutput=False
produceTimeOutput=False

#~~~~~~~~~~~~~~parameters and constants~~~~~~~~~~~~~~#

G=co.gravitational_constant
M=211                           #Mirror mass of LF-ET
pi=np.pi

tmax=1                         #time of simulation
L=1000                         #Length of simulation box
Nx=25
Nt=100
dx=L/Nx                         #spacial stepwidth
dt=tmax/Nt                      #temporal stepwidth

grid_size=int(2*L/dx)
c_p=6000                        #sound velocity in rock  #6000 m/s
#c_s=4000

cavity_r=5
Awave=1
Awavemax=Awave                  #Amplitude of density-fluctuations of p-waves
Awavemin=Awave             #Amplitude of density-fluctuations of p-waves

rmin=L                          #minimum distance for waves to spawn
rmax=2*L                        #maximum distance for waves to spawn
fmax=1                         #frequency of seismic wave
fmin=10
sigmafmax=fmax/10               #width of frequency of gaussian package
sigmafmin=fmin/10

sChance=0
SNR=15
useDerivative=False
useSecondDerivative=False

NoE=10                          #Number of events per run
NoS=1                          #Number of seismometers
NoR=7500                          #Number of runs/realizations
NoT=1500                           #Number of runs without update of WF

WFdepth=1

x=np.linspace(-L,L-dx,grid_size)
y=np.linspace(-L,L-dx,grid_size)
z=np.linspace(-L,L-dx,grid_size)
xyz=np.meshgrid(x,y,z)
x3d=xyz[1]
y3d=xyz[0]
z3d=xyz[2]

dx_dis=L/10000

def speed_of_sound(x,f):
    csMatrix=np.ones(np.shape(x))
    csMatrix[:,:int(grid_size/2)]=c_p
    csMatrix[:,int(grid_size/2):]=c_p
    return csMatrix

def gauss_wave_package(x,t,x0,t0,f0,sigmaf,Awave=1,phi=0,steps=300,accuracy=3):
    wave_package=0
    for i in range(steps+1):
        f= f0 - accuracy*sigmaf + i/steps*2*accuracy*sigmaf
        wave_package+=Awave*1/steps* np.exp(-(f-f0)**2/2/sigmaf**2) * np.sin(2*np.pi*f*((x-x0)/speed_of_sound(x,f)-(t-t0))+phi)
    return wave_package

def gauss_wave_package_analytic(x,t,x0,t0,f0,sigmaf,Awave=1,phi=0):
    return np.sqrt(2*pi)*sigmaf*Awave*np.exp(-2*pi**2*sigmaf**2*((x-x0)/c_p-(t-t0))**2)*np.sin(2*pi*f0*((x-x0)/c_p-(t-t0))+phi)

def gauss_wave_package_einsteintest(x,t,x0,t0,f0,sigmaf,steps=100,accuracy=3):
    farray=np.linspace(f0-accuracy*sigmaf,f0+accuracy*sigmaf,steps+1)
    wave_package=1/steps*np.einsum("k,ijk->ij",np.exp(-(farray-f0)**2/2/sigmaf**2) , np.sin(np.einsum("k,ij->ijk",2*np.pi*farray,((x-x0)/speed_of_sound(x)-(t-t0)))))
    return wave_package #sadly no speed up :(

def calc_force(drho,R=0):
    F=0
    for ix in range(grid_size):
        for iy in range(grid_size):
            for iz in range(grid_size):
                r=np.sqrt(x[ix]**2+y[iy]**2+z[iz]**2)
                if r<=R:
                    continue
                F+=G*M*dx**3*np.cos(np.arctan2(y[iy],x[ix]))*np.sin(np.arccos(z[iz]/r))*drho[ix,iy,iz]/r**2
    return F


def calc_force_quick(drho,R=0):
    r=np.sqrt(x3d**2+y3d**2+z3d**2)+1e-20
    cavity_kernel=r>R
    F=G*M*dx**3*np.sum(np.cos(np.arctan2(y3d,x3d))*np.sin(np.arccos(z3d/r))*drho/r**2*cavity_kernel)
    return F

def derivative(x,f,dx):
    return (f(x)[1:]-f(x)[:-1])/dx

def secderivative(x,f,dx):
    return (derivative(x[1:],f)-derivative(x[:-1],f))/dx

def integrate(x,f,dx,x1=0):
    integral=[]
    for i in range(len(x)):
        if i==0:
            px=np.linspace(x1,x[0],int((x[0]-x1)/dx))
            integral.append(dx*np.sum(f(px)))
        else:
            px=np.linspace(x[i-1],x[i],int((x[i]-x[i-1])/dx))
            integral.append(dx*np.sum(f(px))+integral[i-1])
    return np.array(integral)

def integrate_quick(x,f,dx,x1=0):
    xbetter=np.linspace(np.min(x),np.max(x),int((np.max(x)-np.min(x))/dx))
    jump=(x[1]-x[0])/dx
    return inte.cumulative_trapezoid(f(xbetter),xbetter,initial=x1)[::int(jump)]


def single_integrate(x2,f,dx,x1=0):
    return dx*np.sum(f[x1:x2])

def seismometer_displacement_1D(x,f0,sigmaf,A=1,phi=0,dx=0.1,max_disdist=L*np.sqrt(2)):
    def rho_1D(x):
        return gauss_wave_package_analytic(x, 0, 0, 0, f0, sigmaf,A,phi)
    return -integrate_quick(x,rho_1D,dx,x1=-max_disdist)

def data_auto_correlation_unaveraged(data):
    data=data.reshape((NoS*3))
    return np.tensordot(data,data,axes=0)

def cross_correlation_unaveraged(data,signal):
    data=data.reshape((NoS*3))
    return data*signal

def data_auto_correlation_unaveraged_n(data,n):
    data=data.reshape((NoS*3*n))
    return np.tensordot(data,data,axes=0)

def cross_correlation_unaveraged_n(data,signal,n):
    data=data.reshape((NoS*3))
    signal.reshape((n))
    return np.tensordot(data,signal,axes=0).reshape((NoS*3*n))


#~~~~~~~~~~~~~~general generation~~~~~~~~~~~~~~#

data_auto_correlation_matrix=np.zeros((3*NoS*WFdepth,3*NoS*WFdepth))
cross_correlation_vector=np.zeros((3*NoS*WFdepth))
C=0

signal_data_CPSD=np.zeros((3*NoS),dtype = 'complex_')
data_self_CPSD=np.zeros((3*NoS,3*NoS),dtype = 'complex_')
signal_self_PSD=0
CFS=0


#Seismometer
useCustomState=True

if useCustomState:
    statestring=str(np.loadtxt("fComparison/p1end1Results"+str(ID)+str(ID)+".txt",dtype=str,skiprows=21,max_rows=1,delimiter="ö"))
    seismometer_positions=np.array(ast.literal_eval(statestring.split("(")[1].split(")")[0]))
    seismometer_positions[:,0]-=536.35
    
    seismometer_positions=np.array([[-536.35*0.05,0,0]])
    
else:
    seismometer_position_index=[]
    seismometer_positions=np.zeros((NoS,3))
    for s in range(NoS):
        seismometer_position_index.append([rd.randint(0,grid_size-1),rd.randint(0,grid_size-1),rd.randint(0,grid_size-1)])
    seismometer_position_index=np.array(seismometer_position_index)
    seismometer_positions[:,0],seismometer_positions[:,1],seismometer_positions[:,2]=x[seismometer_position_index[:,0]],y[seismometer_position_index[:,1]],z[seismometer_position_index[:,2]]

if produceTimeOutput or produceDataOutput:
    print(" seismometer positions\n",seismometer_positions,"\n wave data")
    
    
    
#prepare runs
residuals=[]
residualsWFf=[]
RFranc=[]
signalFranc=[]

for R in range(NoR):
    start_time=systime.time()
    time=np.linspace(0,tmax,int(tmax/dt)+1)
    force=[]
    density_fluctuation_array=[]
    seismometer_data=[]                 #x-, y- and z-displacement for each seismometer
    for s in range(NoS):
        seismometer_data.append([[],[],[]])
    
    data_vecs=[]
    for i in range(WFdepth-1):
        data_vecs.append(np.zeros((3*NoS*WFdepth)))
        
    #Wave events
    polar_angles=[]
    azimuthal_angles=[]
    phases=[0]*NoE
    x0s=[]
    t0s=[]
    kx3Ds=[]
    fs=[]
    sigmafs=[]
    As=[]
    displacement_arrays=[]
    max_disdists=[]
    is_s=[]
    s_polarisation=[]
    
    for n in range(NoE):
        
        polar_angles.append(rd.random()*2*pi)#rd.randint(0,359)*pi/180)
        azimuthal_angles.append(rd.random()*pi)
        #phases.append(rd.random()*2*pi)#rd.randint(0,359)*pi/180)
        
        x0s.append(-rd.randint(rmin,rmax))
        t0s.append(rd.random()*2*tmax-tmax)
        
        if isPlaneWave:
            fs.append(ID)
            sigmafs.append(1e-10)
        else:
            fs.append(rd.random()*(fmax-fmin)+fmin)
            sigmafs.append(rd.random()*(sigmafmax-sigmafmin)+sigmafmin)
        
        As.append(rd.random()*(Awavemax-Awavemin)+Awavemin)
    
        kx3Ds.append(np.cos(polar_angles[n])*np.sin(azimuthal_angles[n])*x3d+np.sin(polar_angles[n])*np.sin(azimuthal_angles[n])*y3d+np.cos(azimuthal_angles[n])*z3d)
        
        if isPlaneWave:
            max_disdists.append(c_p/fs[n])
            x_dis=np.linspace(-max_disdists[n],max_disdists[n],int(2*max_disdists[n]/dx_dis))
            displacement_arrays.append(seismometer_displacement_1D(x_dis, fs[n],sigmafs[n],As[n],phases[n],dx_dis/10,max_disdists[n]))
        else:
            max_disdists.append(3*c_p/sigmafs[n])
            x_dis=np.linspace(-max_disdists[n],max_disdists[n],int(2*max_disdists[n]/dx_dis))
            displacement_arrays.append(seismometer_displacement_1D(x_dis, fs[n],sigmafs[n],As[n],phases[n],dx_dis/10,max_disdists[n]))
    
        is_s.append(np.random.choice([True,False],p=[sChance,1-sChance]))
        s_polarisation.append(rd.random()*2*pi)
        
    wave_params=[polar_angles,azimuthal_angles,x0s,t0s,fs,sigmafs,As,phases,is_s,s_polarisation]
    wave_param_texts=["polar angle","azimuthal angle","x0","t0","f","sigma_f","A_wave","phase","is S-wave","S-wave polarization"]
    for i,a in enumerate(wave_params):
        b=np.round(np.array(a,dtype=np.float32),3)
        if produceDataOutput:
            print(wave_param_texts[i],"\t",b)
    
    if produceTimeOutput: 
        print("\npreparation time: "+str((systime.time()-start_time)/60)+" min")
        prep_time=systime.time()
    
    
    
    #~~~~~~~~~~~~~~simulation start~~~~~~~~~~~~~~#
    
    for t in time:
        #print(t)
        density_fluctuations=np.zeros((grid_size,grid_size,grid_size))
    
        #print("t1",systime.time())
        for n in range(NoE):
            if not is_s[n]:
                density_fluctuations+=gauss_wave_package_analytic(kx3Ds[n], t, x0s[n], t0s[n], fs[n], sigmafs[n], As[n],phases[n])
        #print("t2",systime.time())
        force.append(calc_force_quick(density_fluctuations,cavity_r))
        density_fluctuation_array.append(density_fluctuations)    
        #print("t3",systime.time())
        for s in range(NoS):
            xdisplacement=0
            ydisplacement=0
            zdisplacement=0
            for n in range(NoE):
                distanceFromWave=seismometer_positions[s][0]*np.cos(polar_angles[n])*np.sin(azimuthal_angles[n])+seismometer_positions[s][1]*np.sin(polar_angles[n])*np.sin(azimuthal_angles[n])+seismometer_positions[s][2]*np.cos(azimuthal_angles[n])-x0s[n]-c_p*(t-t0s[n])
                if np.abs(distanceFromWave)<max_disdists[n] and not isPlaneWave:
                    absoluteDisplacement=displacement_arrays[n][int((distanceFromWave+max_disdists[n])/dx_dis)]
                elif isPlaneWave:
                    absoluteDisplacement=displacement_arrays[n][min(int((distanceFromWave%(max_disdists[n])+max_disdists[n])/dx_dis),len(displacement_arrays[n])-1)]
                else:
                    absoluteDisplacement=0
                if not is_s[n]:
                    xdisplacement+=np.cos(polar_angles[n])*np.sin(azimuthal_angles[n])*absoluteDisplacement
                    ydisplacement+=np.sin(polar_angles[n])*np.sin(azimuthal_angles[n])*absoluteDisplacement
                    zdisplacement+=np.cos(azimuthal_angles[n])*absoluteDisplacement
                else:
                    xdisplacement+=(np.sin(s_polarisation[n])*(-np.sin(polar_angles[n])*np.sin(azimuthal_angles[n]))+np.cos(s_polarisation[n])*(-np.cos(polar_angles[n])*np.sin(azimuthal_angles[n])*np.cos(azimuthal_angles[n])))*absoluteDisplacement
                    ydisplacement+=(np.sin(s_polarisation[n])*(np.cos(polar_angles[n])*np.sin(azimuthal_angles[n]))+np.cos(s_polarisation[n])*(-np.sin(polar_angles[n])*np.sin(azimuthal_angles[n])*np.cos(azimuthal_angles[n])))*absoluteDisplacement
                    zdisplacement+=(np.cos(s_polarisation[n])*np.sin(azimuthal_angles[n])**2)*absoluteDisplacement
            seismometer_data[s][0].append(xdisplacement)#-single_integrate(seismometer_position_index[s,0], density_fluctuations[:,seismometer_position_index[s,1]],dx))#+density_fluctuations[0,seismometer_position_index[s,1]])
            seismometer_data[s][1].append(ydisplacement)#-single_integrate(seismometer_position_index[s,1], density_fluctuations[:,seismometer_position_index[s,0]],dx))#+density_fluctuations[seismometer_position_index[s,0],0])
            seismometer_data[s][2].append(zdisplacement)#-single_integrate(seismometer_position_index[s,1], density_fluctuations[:,seismometer_position_index[s,0]],dx))#+density_fluctuations[seismometer_position_index[s,0],0])
                
        
        if len(seismometer_data[0][0])>=WFdepth:
            if(R<NoR-NoT):
                data_auto_correlation_matrix=data_auto_correlation_matrix*C/(C+1)+data_auto_correlation_unaveraged_n(np.array(seismometer_data)[:,:,-WFdepth:],WFdepth)/(C+1)
                cross_correlation_vector=cross_correlation_vector*C/(C+1)+cross_correlation_unaveraged_n(np.array(seismometer_data)[:,:,-1:],np.array(force)[-WFdepth:],WFdepth)/(C+1)
                C+=1
            data_vecs.append(np.array(seismometer_data)[:,:,-WFdepth:].reshape(NoS*3*WFdepth))
                
            
    if produceTimeOutput:
        print("simulation time"+str(R)+": "+str((systime.time()-prep_time)/60)+" min")
    
    WF=np.dot(cross_correlation_vector,np.linalg.inv(data_auto_correlation_matrix))
    #data_vec=np.array(seismometer_data).reshape(3*NoS,Nt+1)
    if False:#WFdepth>1:
        estimate=np.sum(np.dot(WF,np.array(data_vecs).T),axis=1)
    else:
        estimate=np.dot(WF,np.array(data_vecs).T)
    
    """
    #~~~~~~~~~~~~~~saving the data~~~~~~~~~~~~~~#
    
    run_info=np.loadtxt("3Ddata/run_info.txt")
    run_id=0#int(run_info[-1][0])+1 #TODO: Make this all a function
    print("run ID: ", run_id)
    np.savetxt("3Ddata/eventdata"+str(run_id)+".txt",[polar_angles,azimuthal_angles,x0s,t0s,fs,sigmafs,As,phases])
    np.savetxt("3Ddata/seismometerPositions"+str(run_id)+".txt",seismometer_positions,fmt="%.4e")
    np.savetxt("3Ddata/forceOutput"+str(run_id)+".txt",force)
    np.savetxt("3Ddata/seismometerInput"+str(run_id)+".txt",[seismometer_data[int(i/NoS)][i%3] for i in range(3*NoS)])
    run_info=list(run_info)
    run_info.append(np.array([run_id,0,systime.time()-start_time]))
    np.savetxt("3Ddata/run_info.txt",run_info,fmt="%.3e")
    """
    seismometer_data = np.array(seismometer_data)
    if useDerivative or useSecondDerivative:
        seismometer_data[:,:,:-1] = (seismometer_data[:,:,1:]-seismometer_data[:,:,:-1])/dt
        seismometer_data[:,:,-1]=0
        if useSecondDerivative:
            seismometer_data[:,:,1:] = (seismometer_data[:,:,1:]-seismometer_data[:,:,:-1])/dt
            seismometer_data[:,:,0]=0
    seismometer_data+=np.random.normal(seismometer_data*0,np.abs(seismometer_data)/SNR)

    
    #~~~~~~~~~~~~~~data visualization~~~~~~~~~~~~~~#
    
    timestep=0
    z_slice=Nx
    dis_scale=1/np.max(seismometer_data)*L/30
    
    """
    #nice 2D-image+animation
    fig, ax=plt.subplots()    
    plt.title(r"density fluctuations in $(x,y)$ at $t=0$")
    im=plt.imshow(density_fluctuation_array[timestep][:,::-1,z_slice].T,extent=[-L,L,-L,L],label=r"$\delta\rho$")
    plt.colorbar(ax=ax,label=r"$\delta\rho$")
    plt.clim(-0.4*Awave,0.4*Awave)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    vec=plt.quiver(0,0,force[timestep],0,color="r",angles='xy', scale_units='xy',scale=max(force)/L)
    cavity=plt.Circle((0,0),cavity_r,fill=False,edgecolor="k",linewidth=1.5)
    ax.add_patch(cavity)
    
    seismometer_data=np.array(seismometer_data)
    scat=plt.scatter(seismometer_positions[:,0],seismometer_positions[:,1],marker="d",c="white",s=15)
    for s in range(NoS):
        plt.text(seismometer_positions[s,0],seismometer_positions[s,1],str(s))
    
    def update(i):
        t=i*dt
        ax.set_title(r"density fluctuations in $(x,y)$ at $t=$"+str(round(t,3)))
        im.set_array(density_fluctuation_array[i][:,::-1,z_slice].T)
        scat.set_offsets(np.array([seismometer_positions[:,0]+seismometer_data[:,0,i]*dis_scale,seismometer_positions[:,1]+seismometer_data[:,1,i]*dis_scale]).T)
        vec.set_UVC(force[i],0)
    
    #Force Plot
    forcefig=plt.figure()
    plt.xlabel(r"time $t$")
    plt.ylabel(r"Force $F$")
    plt.title("mirror x-Noise")
    plt.plot(time,force)
    
    #Seismometer Data
    seismometer_numbers=[0]
    plt.figure()
    plt.title("example displacement of seismometer")
    plt.xlabel(r"$t$")
    plt.ylabel(r"displacement $d$")
    for s in seismometer_numbers:
        if s<NoS:
            plt.plot(time,seismometer_data[s][0],label="seismometer "+str(s)+" x-data")
            plt.plot(time,seismometer_data[s][1],label="seismometer "+str(s)+" y-data")
            plt.plot(time,seismometer_data[s][2],label="seismometer "+str(s)+" z-data")
    plt.legend()
    
    def animate():
        anima=ani.FuncAnimation(fig=fig, func=update, frames=int(tmax/dt),interval=5)
        anima.save("testanimation.gif")
        forcefig.savefig("simple_force_test.pdf")
    """
    #full data
    if produceDataOutput:
        fullfig=plt.figure(figsize=(15,15))
        title=fullfig.suptitle("Density Fluctuations, Force and Seismometer Data",fontsize=16,y=0.95)
        
        ax1=plt.subplot(2,6,(1,3))
        plt.title(r"density fluctuations in $(x,y,z=$"+str(int(z[z_slice]))+r"$)$")
        im=plt.imshow(density_fluctuation_array[timestep][:,::-1,z_slice].T,extent=[-L,L,-L,L],label=r"$\delta\rho$")
        plt.colorbar(ax=ax1,label=r"$\delta\rho$")
        plt.clim(-0.4*Awave,0.4*Awave)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        if max(np.abs(force))>0:
            vec=plt.quiver(0,0,force[timestep],0,color="r",angles='xy', scale_units='xy',scale=max(force)/L)
        cavity=plt.Circle((0,0),cavity_r,fill=False,edgecolor="k",linewidth=1.5)
        ax1.add_patch(cavity)
        seismometer_data=np.array(seismometer_data)
        scat=plt.scatter(seismometer_positions[:,0],seismometer_positions[:,1],marker="d",c="white",s=70)
        for s in range(NoS):
            plt.text(seismometer_positions[s,0],seismometer_positions[s,1],str(s),fontsize=13.5)
            
        ax2=plt.subplot(2,6,(4,6))
        forceplot=ax2.plot(time,force,label="measured")[0]
        estimateplot=ax2.plot(time,estimate,label="estimate")[0]
        diffplot=ax2.plot(time,force-estimate,label="difference",color="red")[0]
        plt.xlim(0,np.max(time))
        if max(np.abs(force))>0:
            plt.ylim(-np.max(np.abs(force)),np.max(np.abs(force)))
        plt.title(r"force on mirror in $x$-direction")
        plt.ylabel(r"force")
        plt.xlabel(r"time")
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.legend()
        
        axarray=[[],[],[]]
        lines=[[],[],[]]
        plotspace=2
        titles=[r"seismometer $x$-displacement",r"seismometer $y$-displacement",r"seismometer $z$-displacement"]
        max_dim=3
        NoSplot=min(NoS,5)
        for s in range(NoSplot):
            for dim in range(max_dim):
                axarray[dim].append(plt.subplot(2*NoSplot+plotspace,6,(NoSplot*6+(int(6/max_dim)*dim+1)+(s+plotspace)*6,NoSplot*6+(int(6/max_dim)*dim+2)+(s+plotspace)*6)))
                color=[0,0,0]
                color[dim]=0.2+0.8/NoSplot*s
                lines[dim].append(axarray[dim][s].plot(time,seismometer_data[s][dim],color=color)[0])
                plt.xlim(0,np.max(time))
                plt.ylim(-np.max(np.abs(seismometer_data)),np.max(np.abs(seismometer_data)))
                if dim==0:
                    plt.ylabel(r"#"+str(s))
                if dim==1:
                    axarray[dim][s].set_yticklabels(())
                    axarray[dim][s].set_yticks(())
                if s==0:
                    plt.title(titles[dim])
                if s==NoSplot-1:
                    plt.xlabel(r"time")
                else:        
                    axarray[dim][s].set_xticklabels(())
                    axarray[dim][s].tick_params(axis="x",direction="in")
                if dim==max_dim-1:
                    axarray[max_dim-1][s].yaxis.set_label_position("right")
                    axarray[max_dim-1][s].yaxis.tick_right()
                    
        plt.subplots_adjust(hspace=0)
        #plt.savefig("lastTimeStep3D.svg")
        
        def animate_full():
            stepw=2
            def update_full(i):
                i=stepw*i
                t=i*dt
                title.set_text(r"Density Fluctuations, Force and Seismometer Data at $t=$"+str(round(t,3)))
                im.set_array(density_fluctuation_array[i][:,::-1,z_slice].T)
                scat.set_offsets(np.array([seismometer_positions[:,0]+seismometer_data[:,0,i]*dis_scale,seismometer_positions[:,1]+seismometer_data[:,1,i]*dis_scale]).T)
                if max(np.abs(force))>0:
                    vec.set_UVC(force[i],0)
                forceplot.set_xdata(time[:i+1])
                forceplot.set_ydata(force[:i+1])
                estimateplot.set_xdata(time[:i+1])
                estimateplot.set_ydata(estimate[:i+1])
                diffplot.set_xdata(time[:i+1])
                diffplot.set_ydata(force[:i+1]-estimate[:i+1])
                for s in range(NoSplot):
                    for dim in range(max_dim):
                        lines[dim][s].set_xdata(time[:i+1])
                        lines[dim][s].set_ydata(seismometer_data[s][dim][:i+1])
            ax1.set_title(r"density fluctuations in $(x,y,z$="+str(int(z[z_slice]))+r"$)$")
            anima=ani.FuncAnimation(fig=fullfig, func=update_full, frames=int((int(tmax/dt)+1)/stepw),interval=max(1,int(5/stepw)))
            anima.save("fullanimation3D"+str(ID)+"_"+str(R)+".gif")
        
        #animate_full()
    
    #Fourier-space
    freqs=np.fft.fftfreq(time.shape[-1])*time.shape[-1]/tmax
    signalFS=np.fft.fft(force)[np.argmax(freqs>=ID)]
    dataFS=np.fft.fft(seismometer_data)[:,:,np.argmax(freqs>=ID)].reshape(NoS*3)
    if produceDataOutput:
        print("signalFS:",signalFS)
        print("dataFS:",dataFS)
    if(R<NoR-NoT):
        signal_data_CPSD=signal_data_CPSD*CFS/(CFS+1)+np.conjugate(dataFS)*signalFS/(CFS+1)
        data_self_CPSD=data_self_CPSD*CFS/(CFS+1)+np.einsum("i,k->ik",np.conjugate(dataFS),dataFS)/(CFS+1)
        inv_data_self_CPSD=np.linalg.inv(data_self_CPSD)
        WF_FS=np.einsum("ij,i->j",inv_data_self_CPSD,signal_data_CPSD)
        signal_self_PSD=signal_self_PSD*CFS/(CFS+1)+signalFS*np.conjugate(signalFS)/(CFS+1)
        CFS+=1
        if produceDataOutput:
            print("signal-data-CPSD:",signal_data_CPSD)
            print("data-self-CPSD:",data_self_CPSD)
            print("data-self-CPSD-inverse:",np.linalg.inv(data_self_CPSD))
            print("WF:",WF_FS)
            print("CFS:",CFS)
    estimateFS=np.einsum("i,i->",WF_FS,dataFS)
    if produceDataOutput:
        print("estimateFS:",estimateFS)
    
    #save things
    residuals.append(np.abs(np.fft.fft(force-estimate))[np.argmax(freqs>=ID)]/np.abs(np.fft.fft(force))[np.argmax(freqs>=ID)])
    residualsWFf.append(np.abs(signalFS-estimateFS)/np.abs(signalFS))
    RFranc.append(np.abs(1-np.einsum("i,ij,j->",np.conjugate(signal_data_CPSD),inv_data_self_CPSD,signal_data_CPSD)/signal_self_PSD))
    signalFranc.append(np.abs(signalFS))
    
    
    #Plot Fourier space
    if produceDataOutput:
        plt.figure()
        plt.title("FFT")
        plt.xlabel(r"$f$")
        plt.ylabel(r"Amplitude")
        plt.plot(freqs[:int(len(freqs)/2+1)],np.abs(np.fft.fft(force))[:int(len(freqs)/2+1)],label="force")
        plt.plot(freqs[:int(len(freqs)/2+1)],np.abs(np.fft.fft(force-estimate))[:int(len(freqs)/2+1)],color="red",label="difference")
        plt.legend()
        plt.savefig("FFT"+str(ID)+"_"+str(R)+".jpg")
    
    
    
if produceTimeOutput:
    print("total time: "+str((systime.time()-total_start_time)/60)+" min\n")


if produceDataOutput or produceTimeOutput:
    print(residualsWFf)
    print("\n\n")
    
import os
if not os.path.exists(folder):  
    os.makedirs(folder)
if not os.path.exists(folder+"/dataFile"+tag+".txt"):
    f = open(folder+"/dataFile"+tag+".txt",'a+') 
    f.write("isPlaneWave = "+str(isPlaneWave)+"\n")
    f.write("t_max = "+str(tmax)+"\n")
    f.write("L = "+str(L)+"\n")
    f.write("Nx = "+str(Nx)+"\n")
    f.write("Nt = "+str(Nt)+"\n")
    f.write("cavity_r = "+str(cavity_r)+"\n")
    f.write("Awave_max = "+str(Awavemax)+"\n")
    f.write("Awave_min = "+str(Awavemin)+"\n")
    f.write("f_min = "+str(fmin)+"\n")
    f.write("f_max = "+str(fmax)+"\n")
    f.write("sigma_f_min = "+str(sigmafmin)+"\n")
    f.write("sigma_f_max = "+str(sigmafmax)+"\n")
    f.write("SNR = "+str(SNR)+"\n")
    f.write("useSecondDerivative = "+str(useSecondDerivative)+"\n")
    f.write("NoE = "+str(NoE)+"\n")
    f.write("NoS = "+str(NoS)+"\n")
    f.write("NoR = "+str(NoR)+"\n")
    f.write("NoT = "+str(NoT)+"\n")
    f.write("WF_depth = "+str(WFdepth)+"\n")
    f.write("seismometer_positions = np.array("+str(seismometer_positions.tolist())+")\n")
    f.write("#runtime = "+str(np.round((systime.time()-total_start_time)/60,2))+" min\n")
    f.close()
    
np.savetxt(folder+"/residuals"+tag+str(ID)+".txt",residuals)    
np.savetxt(folder+"/residualsf"+tag+str(ID)+".txt",residualsWFf)
np.savetxt(folder+"/resFranc"+tag+str(ID)+".txt",RFranc)
np.savetxt(folder+"/signal"+tag+str(ID)+".txt",signalFranc)
np.savetxt(folder+"/WF"+tag+str(ID)+".txt",WF)
np.savetxt(folder+"/WFf"+tag+str(ID)+".txt",WF_FS)
