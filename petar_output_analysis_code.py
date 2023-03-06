#!/usr/bin/env python
# coding: utf-8

# In[1]:


import petar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


path ='/Simulation directory/'


# In[3]:


def single_parameters(path, timestep):
    
    single = petar.Particle(interrupt_mode='bse/mobse')
    
    path_file_single = '{}{}.{}.{}'.format(path, 'data', timestep, 'single')
    
    single.loadtxt(path_file_single)
    
    with open(path_file_single) as file:
    
       for line in file:
    
            m_s = single.mass
            X_s = single.pos
            V_s = single.vel
            L_s = single.star.lum
            R_s = single.star.rad
            type_s = single.star.type
            N_s = len(m_s)
            
            x_s = X_s[:,0]
            y_s = X_s[:,1]
            z_s = X_s[:,2]
            
    return N_s, m_s, X_s, V_s, L_s, R_s, type_s


# In[4]:


def binary_parameters(path, timestep):
    
    binary = petar.Binary(member_particle_type=petar.Particle, interrupt_mode='bse/mobse', G=petar.G_MSUN_PC_MYR)
    
    path_file_binary = '{}{}.{}.{}'.format(path, 'data', timestep, 'binary')
    
    binary.loadtxt(path_file_binary)
    
    with open(path_file_binary) as file:
        
        for line in file:
            
            m_cm = binary.mass
            X_cm = binary.pos
            V_cm = binary.vel
            x_cm = X_cm[:,0]
            y_cm = X_cm[:,1]
            z_cm = X_cm[:,2]
            N_b = len(m_cm)
            a = binary.semi
            e = binary.ecc
            
            m_1 = binary.p1.mass
            X_1 = binary.p1.pos
            V_1 = binary.p1.vel
            L_1 = binary.p1.star.lum
            R_1 = binary.p1.star.rad
            type_1 = binary.p1.star.type        
        
            m_2 = binary.p2.mass
            X_2 = binary.p2.pos          
            V_2 = binary.p2.vel            
            L_2 = binary.p2.star.lum
            R_2 = binary.p2.star.rad
            type_2 = binary.p2.star.type
            
    return N_b, m_cm, X_cm, V_cm, a, e, m_1, m_2, X_1, X_2, V_1, V_2, L_1, L_2, R_1, R_2, type_1, type_2


# In[5]:


def stellar_numbers(N_s, N_b, timestep):
    
    N = N_s + N_b
    N_tot = N_s + 2*N_b
    f_b_timestep = N_b/N_tot
    
    return N, N_tot, f_b_timestep


# In[ ]:


def HRD(L_s, L_1, L_2, R_s, R_1, R_2, timestep, plot=True/False):
    
    # Luminosity
    
    L = np.concatenate((L_s,L_1,L_2), axis=0)
    L_tot = np.sum(L)
    
    # Temperature
    
    T_s = 5778*(L_s/R_s**2)**(1/4)
    T_1 = 5778*(L_1/R_1**2)**(1/4)
    T_2 = 5778*(L_2/R_2**2)**(1/4)
    T = np.concatenate((T_s,T_1,T_2), axis=0)
    
    # HRD
    
    plt.figure(figsize=(9,7), dpi=120)
    plt.scatter(T, L, color='black', alpha=1)
    plt.xlabel('T [K]', fontsize=20)
    plt.ylabel('L [L$_{\odot}$]', fontsize=20)
    plt.xlim(T.max(),T.min())
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
    return L, L_tot, T


# In[10]:


def get_data_versus_time(path):
    
    timesteps = np.arange(0,'final timestep + 1')
    
    f_b = []
    
    for timestep in timesteps:
        
        N_s, m_s, X_s, V_s, L_s, R_s, type_s = single_parameters(path, timestep)
        N_b, m_cm, X_cm, V_cm, a, e, m_1, m_2, X_1, X_2, V_1, V_2, L_1, L_2, R_1, R_2, type_1, type_2 = binary_parameters(path, timestep)
        N, N_tot, f_b_timestep = stellar_numbers(N_s, N_b, timestep)
        L, L_tot, T = HRD(L_s, L_1, L_2, R_s, R_1, R_2, timestep, plot=True/False)
    
        f_b.append(f_b_timestep)     
      
    f_b = np.array(f_b, dtype='object')
    
    return f_b


# In[ ]:


lagr = petar.LagrangianMultiple(interrupt_mode='no/bse/mobse', external_mode='no/galpy', calc_energy=True)
lagr.loadtxt(path+'data.lagr')

core = petar.Core()
core.loadtxt(path+'data.core')

status = petar.Status()
status.loadtxt(path+'data.status')

tidal = petar.Tidal()
tidal.loadtxt(path+'data.tidal')

esc_single = petar.SingleEscaper(interrupt_mode='no/bse/mobse', external_mode='no/galpy')
esc_single.loadtxt(path+'data.esc_single')
    
esc_binary = petar.BinaryEscaper(interrupt_mode='no/bse/mobse', external_mode='no/galpy')
esc_binary.loadtxt(path+'data.esc_binary')


# In[12]:


def evolutionary_time():
    
    time_myr = status.time
    
    return time_myr


# In[22]:


def dynamical_timescales(G=petar.G_MSUN_PC_MYR):
    
    N_hm = lagr.all.n[:,2] # Total number of stars inside the half-mass radius
    m_hm = lagr.all.m[:,2] # Mean mass inside the half-mass radius
    r_hm = lagr.all.r[:,2] # Half-mass radius
    
    M_hm = m_hm*N_hm # Total mass inside the half-mass radius
    M = 2*m_h*N_h # Total mass
    
    t_cr = petar.calcTcr(M, r_hm, G)
    
    t_rh = petar.calcTrh(N_hm, r_hm, m_hm, G, gamma=0.02)
    
    return t_cr, t_rh


# In[13]:


def lagrangian_radii(time_myr, plot=True/False):
    
    r_lagr = lagr.all.r
    f_m = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9]) # Mass fractions
    
    r_hm = r_lagr[:,2]
    r_core = r_lagr[:,5]

    plt.figure(figsize=(8,11), dpi=120)
    plt.plot(time_myr, r_lagr[:,0], color='pink', linewidth=1, alpha=1, label='$f_m$=0.1')
    plt.plot(time_myr, r_lagr[:,1], color='lime', linewidth=1, alpha=0.5, label='$f_m$=0.3')
    plt.plot(time_myr, r_lagr[:,2], color='orange', linewidth=1, alpha=1, label='$f_m$=0.5')
    plt.plot(time_myr, r_lagr[:,3], color='turquoise', linewidth=1, alpha=1, label='$f_m$=0.7')
    plt.plot(time_myr, r_lagr[:,4], color='orchid', linewidth=1, alpha=1, label='$f_m$=0.9')
    plt.xlabel('Time [Myr]', fontsize=20)
    plt.ylabel('$R_{lagr}$ [pc]', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.show()
    
    return r_lagr, r_hm, r_core


# In[14]:


def virial_ratio(time_myr, plot=True/False):

    Q = lagr.all.vr[:,2]/2

    fig = plt.figure(figsize=(5,6), dpi=120)
    plt.plot(time_myr, Q, color='black', linewidth=2, alpha=1)
    plt.xlabel('Time [Myr]', fontsize=17)
    plt.ylabel('Q', fontsize=17)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()
    
    return Q


# In[15]:


def energy(time_myr, Q, plot=True/False):

    E_pot = lagr.all.epot[:,2]
    #E_pot = lagr.all.epot[:,2] - lagr.all.epot_ext[:,2]
    
    E = (Q-1)/np.abs(E_pot)
    Delta_E_norm = (E-E[0])/E[0]
    
    energy_figure = plt.figure(figsize=(23,10), dpi=120)

    energy_figure.add_subplot(1,2,1)
    plt.plot(time_myr, E, color='black', linewidth=2, alpha=1)
    plt.xlabel('Time [Myr]', fontsize=22)
    plt.ylabel('E [erg]', fontsize=22)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)

    energy_figure.add_subplot(1,2,2)
    plt.plot(time_myr, Delta_E_norm, color='black', linewidth=2, alpha=1)
    plt.xlabel('Time [Myr]', fontsize=22)
    plt.ylabel('$\Delta E/E_0$', fontsize=22)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    
    plt.show()
    
    return E, Delta_E_norm


# In[16]:


def total_mass(time_myr, plot=True/False):
    
    N_hm = lagr.all.n[:,2]
    m_hm = lagr.all.m[:,2]
    M = 2*m_hm*N_hm

    fig = plt.figure(figsize=(5,6), dpi=120)
    plt.plot(time_myr, M, color='black', linewidth=2, alpha=1)
    plt.xlabel('Time [Myr]', fontsize=18)
    plt.ylabel('M [M$_\odot$]', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()
    
    return M


# In[17]:


def binary_fraction(time_myr, f_b, plot=True/False):
    
    fig = plt.figure(figsize=(5,6), dpi=120)
    plt.plot(time_myr, f_b, color='black', linewidth=2, alpha=1)
    plt.xlabel('Time [Myr]', fontsize=18)
    plt.ylabel('$f_b$', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()
    
    return 0


# In[ ]:


def center_coordinates():
    
    X = core.pos
    V = core.vel
    
    return X, V


# In[ ]:


def tidal(time_myr, plot=True/False):
    
    # Bound stars data
    
    r_tid = tidal.rtid
    m_bound = tidal.mass
    n_bound = tidal.n
    
    figure = plt.figure(figsize=(23,10), dpi=120)

    figure.add_subplot(1,2,1)
    plt.plot(time_myr, m_bound, color='black', linewidth=2, alpha=1)
    plt.xlabel('Time [Myr]', fontsize=22)
    plt.ylabel('$M_{bound}$ [M$_\odot$]', fontsize=22)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)

    figure.add_subplot(1,2,2)
    plt.plot(time_myr, n_bound, color='black', linewidth=2, alpha=1)
    plt.xlabel('Time [Myr]', fontsize=22)
    plt.ylabel('$N_{bound}$', fontsize=22)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    
    plt.show()
    
    # Expected time for system dissolution
    
    def linear_interpolation(time_myr, n_bound):

        X = np.zeros(shape=('N timesteps',2)) # N timesteps = number of timesteps
        X[:,0] = np.ones('N timesteps')
        X[:,1] = time_myr

        Z = np.matmul(np.linalg.inv(np.matmul(X.T,X)), X.T) # (X.T X)**(-1) X.T
        w = np.matmul(Z,n_bound)
    
        n_bound_end = 0
        time_myr_end = (n_bound_end - w[0])/w[1]
    
        return time_myr_end

    time_myr_end = linear_interpolation(time_myr, n_bound)
    
    return r_tid, m_bound, n_bound, time_myr_end


# In[ ]:


f_b = get_data_versus_time(path)

time_myr = evolutionary_time()

t_cr, t_rh = dynamical_timescales(G=petar.G_MSUN_PC_MYR)

r_lagr, r_hm, r_core = lagrangian_radii(time_myr, plot=True/False)

Q = virial_ratio(time_myr, plot=True/False)

E, Delta_E_norm = energy(time_myr, Q, plot=True/False)

M = total_mass(time_myr, plot=True/False)

binary_fraction(time_myr, f_b, plot=True/False)

X, V = center_coordinates()

r_tid, m_bound, n_bound, time_myr_end = tidal(time_myr, plot=True/False)

