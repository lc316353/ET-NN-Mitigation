# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:28:53 2023

@author: schillings
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd
import matplotlib.animation as ani
import time as systime
import scipy.constants as co
plt.close("all")
start_time=systime.time()



#~~~~~~~~~~~~~~parameters and constants~~~~~~~~~~~~~~#

G=co.gravitational_constant
M=211                           #Mirror mass of LF-ET
pi=np.pi

tmax=5                         #time of simulation
L=6000                         #Length of simulation box
Nx=40
Nt=500
dx=L/Nx                         #spacial stepwidth
dt=tmax/Nt                      #temporal stepwidth

grid_size=int(2*L/dx)
c_s=6000                        #sound velocity in rock  #6000 m/s

cavity_r=300
Awave=1
Awavemax=Awave                  #Amplitude of density-fluctuations of p-waves
Awavemin=Awave*0.25             #Amplitude of density-fluctuations of p-waves

rmin=L                          #minimum distance for waves to spawn
rmax=2*L                        #maximum distance for waves to spawn
fmax=10                         #frequency of seismic wave
fmin=1
sigmafmax=fmax/10               #width of frequency of gaussian package
sigmafmin=fmin/10

NoE=10                          #Number of events per run
NoS=4                          #Number of seismometers


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
    csMatrix[:,:int(grid_size/2)]=c_s
    csMatrix[:,int(grid_size/2):]=c_s
    return csMatrix

def gauss_wave_package(x,t,x0,t0,f0,sigmaf,Awave=1,phi=0,steps=300,accuracy=3):
    wave_package=0
    for i in range(steps+1):
        f= f0 - accuracy*sigmaf + i/steps*2*accuracy*sigmaf
        wave_package+=Awave*1/steps* np.exp(-(f-f0)**2/2/sigmaf**2) * np.sin(2*np.pi*f*((x-x0)/speed_of_sound(x,f)-(t-t0))+phi)
    return wave_package

def gauss_wave_package_analytic(x,t,x0,t0,f0,sigmaf,Awave=1,phi=0):
    return np.sqrt(2*pi)*sigmaf*Awave*np.exp(-2*pi**2*sigmaf**2*((x-x0)/c_s-(t-t0))**2)*np.sin(2*pi*f0*((x-x0)/c_s-(t-t0))+phi)

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
                F+=G*M*dx**2*np.cos(np.arctan2(y[iy],x[ix]))*np.sin(np.arccos(z[iz]/r))*drho[ix,iy,iz]/r**2
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

def single_integrate(x2,f,dx,x1=0):
    return dx*np.sum(f[x1:x2])

def seismometer_displacement_1D(x,f0,sigmaf,A=1,phi=0,dx=0.1,max_disdist=L*np.sqrt(2)):
    def rho_1D(x):
        return gauss_wave_package_analytic(x, 0, 0, 0, f0, sigmaf,A,phi)
    return -integrate(x,rho_1D,dx,x1=-max_disdist)


#~~~~~~~~~~~~~~general generation~~~~~~~~~~~~~~#

time=np.linspace(0,tmax,int(tmax/dt)+1)
force=[]
density_fluctuation_array=[]
seismometer_data=[]                 #x-, y- and z-displacement for each seismometer

#Seismometer
seismometer_position_index=[]
seismometer_positions=np.zeros((NoS,3))

for s in range(NoS):
    seismometer_position_index.append([rd.randint(0,grid_size-1),rd.randint(0,grid_size-1),rd.randint(0,grid_size-1)])
    seismometer_data.append([[],[],[]])
seismometer_position_index=np.array(seismometer_position_index)
seismometer_positions[:,0],seismometer_positions[:,1],seismometer_positions[:,2]=x[seismometer_position_index[:,0]],y[seismometer_position_index[:,1]],z[seismometer_position_index[:,2]]
print(" seismometer positions\n",seismometer_positions,"\n wave data")

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

for n in range(NoE):
    
    polar_angles.append(rd.random()*2*pi)#rd.randint(0,359)*pi/180)
    azimuthal_angles.append(rd.random()*pi)
    #phases.append(rd.random()*2*pi)#rd.randint(0,359)*pi/180)
    
    x0s.append(-rd.randint(rmin,rmax))
    t0s.append(rd.random()*2*tmax-tmax)
    
    fs.append(rd.random()*(fmax-fmin)+fmin)
    sigmafs.append(rd.random()*(sigmafmax-sigmafmin)+sigmafmin)
    
    As.append(rd.random()*(Awavemax-Awavemin)+Awavemin)

    kx3Ds.append(np.cos(polar_angles[n])*np.sin(azimuthal_angles[n])*x3d+np.sin(polar_angles[n])*np.sin(azimuthal_angles[n])*y3d+np.cos(azimuthal_angles[n])*z3d)
    
    max_disdists.append(3*c_s/sigmafs[n])
    x_dis=np.linspace(-max_disdists[n],max_disdists[n],int(2*max_disdists[n]/dx_dis))
    displacement_arrays.append(seismometer_displacement_1D(x_dis, fs[n],sigmafs[n],As[n],phases[n],dx_dis/10,max_disdists[n]))

wave_params=[polar_angles,azimuthal_angles,x0s,t0s,fs,sigmafs,As,phases]
wave_param_texts=["polar angle","azimuthal angle","x0","t0","f","sigma_f","A_wave","phase"]
for i,a in enumerate(wave_params):
    b=np.round(np.array(a),3)
    print(wave_param_texts[i],"\t",b)
    
print("\npreparation time: "+str((systime.time()-start_time)/60)+" min")
prep_time=systime.time()



#~~~~~~~~~~~~~~simulation start~~~~~~~~~~~~~~#

for t in time:
    #print(t)
    density_fluctuations=np.zeros((grid_size,grid_size,grid_size))

    #print("t1",systime.time())
    for n in range(NoE):
        density_fluctuations+=gauss_wave_package_analytic(kx3Ds[n], t, x0s[n], t0s[n], fs[n], sigmafs[n], As[n],phases[n])
    #print("t2",systime.time())
    force.append(calc_force(density_fluctuations,cavity_r))
    density_fluctuation_array.append(density_fluctuations)    
    #print("t3",systime.time())
    for s in range(NoS):
        xdisplacement=0
        ydisplacement=0
        zdisplacement=0
        for n in range(NoE):
            distanceFromWave=seismometer_positions[s][0]*np.cos(polar_angles[n])*np.sin(azimuthal_angles[n])+seismometer_positions[s][1]*np.sin(polar_angles[n])*np.sin(azimuthal_angles[n])+seismometer_positions[s][2]*np.cos(azimuthal_angles[n])-x0s[n]-c_s*(t-t0s[n])
            if np.abs(distanceFromWave)<max_disdists[n]:
                absoluteDisplacement=displacement_arrays[n][int((distanceFromWave+max_disdists[n])/dx_dis)]
            else:
                absoluteDisplacement=0
            xdisplacement+=np.cos(polar_angles[n])*np.sin(azimuthal_angles[n])*absoluteDisplacement
            ydisplacement+=np.sin(polar_angles[n])*np.sin(azimuthal_angles[n])*absoluteDisplacement
            zdisplacement+=np.cos(azimuthal_angles[n])*absoluteDisplacement
        seismometer_data[s][0].append(xdisplacement)#-single_integrate(seismometer_position_index[s,0], density_fluctuations[:,seismometer_position_index[s,1]],dx))#+density_fluctuations[0,seismometer_position_index[s,1]])
        seismometer_data[s][1].append(ydisplacement)#-single_integrate(seismometer_position_index[s,1], density_fluctuations[:,seismometer_position_index[s,0]],dx))#+density_fluctuations[seismometer_position_index[s,0],0])
        seismometer_data[s][2].append(zdisplacement)#-single_integrate(seismometer_position_index[s,1], density_fluctuations[:,seismometer_position_index[s,0]],dx))#+density_fluctuations[seismometer_position_index[s,0],0])

print("simulation time: "+str((systime.time()-prep_time)/60)+" min")




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




#~~~~~~~~~~~~~~data visualization~~~~~~~~~~~~~~#

#nice 2D-image+animation
timestep=0
z_slice=Nx
dis_scale=1/np.max(seismometer_data)*L/30
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

#full data
fullfig=plt.figure(figsize=(15,15))
title=fullfig.suptitle("Density Fluctuations, Force and Seismometer Data",fontsize=16,y=0.95)

ax1=plt.subplot(2,6,(1,3))
plt.title(r"density fluctuations in $(x,y,z=$"+str(int(z[z_slice]))+r"$)$ at $t=$"+str(time[timestep]))
im=plt.imshow(density_fluctuation_array[timestep][:,::-1,z_slice].T,extent=[-L,L,-L,L],label=r"$\delta\rho$")
plt.colorbar(ax=ax1,label=r"$\delta\rho$")
plt.clim(-0.4*Awave,0.4*Awave)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
vec=plt.quiver(0,0,force[timestep],0,color="r",angles='xy', scale_units='xy',scale=max(force)/L)
cavity=plt.Circle((0,0),cavity_r,fill=False,edgecolor="k",linewidth=1.5)
ax1.add_patch(cavity)
seismometer_data=np.array(seismometer_data)
scat=plt.scatter(seismometer_positions[:,0],seismometer_positions[:,1],marker="d",c="white",s=15)
for s in range(NoS):
    plt.text(seismometer_positions[s,0],seismometer_positions[s,1],str(s))
    
ax2=plt.subplot(2,6,(4,6))
forceplot=ax2.plot(time,force)[0]
plt.xlim(0,np.max(time))
plt.ylim(-np.max(np.abs(force)),np.max(np.abs(force)))
plt.title(r"force on mirror in $x$-direction")
plt.ylabel(r"force")
plt.xlabel(r"time")
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()

axarray=[[],[],[]]
lines=[[],[],[]]
plotspace=2
titles=[r"seismometer $x$-displacement",r"seismometer $y$-displacement",r"seismometer $z$-displacement"]
max_dim=3
for s in range(min(NoS,6)):
    for dim in range(max_dim):
        axarray[dim].append(plt.subplot(2*NoS+plotspace,6,(NoS*6+(int(6/max_dim)*dim+1)+(s+plotspace)*6,NoS*6+(int(6/max_dim)*dim+2)+(s+plotspace)*6)))
        color=[0,0,0]
        color[dim]=0.2+0.8/min(NoS,6)*s
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
        if s==min(NoS,6)-1:
            plt.xlabel(r"time")
        else:        
            axarray[dim][s].set_xticklabels(())
            axarray[dim][s].tick_params(axis="x",direction="in")
        if dim==max_dim-1:
            axarray[max_dim-1][s].yaxis.set_label_position("right")
            axarray[max_dim-1][s].yaxis.tick_right()
            
plt.subplots_adjust(hspace=0)

def animate_full():
    def update_full(i):
        t=i*dt
        title.set_text(r"Density Fluctuations, Force and Seismometer Data at $t=$"+str(round(t,3)))
        im.set_array(density_fluctuation_array[i][:,::-1,z_slice].T)
        scat.set_offsets(np.array([seismometer_positions[:,0]+seismometer_data[:,0,i]*dis_scale,seismometer_positions[:,1]+seismometer_data[:,1,i]*dis_scale]).T)
        vec.set_UVC(force[i],0)
        forceplot.set_xdata(time[:i+1])
        forceplot.set_ydata(force[:i+1])
        for s in range(min(NoS,6)):
            for dim in range(max_dim):
                lines[dim][s].set_xdata(time[:i+1])
                lines[dim][s].set_ydata(seismometer_data[s][dim][:i+1])
    ax1.set_title(r"density fluctuations in $(x,y,z$="+str(int(z[z_slice]))+r"$)$")
    anima=ani.FuncAnimation(fig=fullfig, func=update_full, frames=int(tmax/dt)+1,interval=5)
    anima.save("fullanimation3D.gif")
    
print("total time: "+str((systime.time()-start_time)/60)+" min")