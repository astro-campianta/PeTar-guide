#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(6438) # Random seed


# In[2]:


def Kroupa_IMF(N_masses, m_min, m_max):

    def Kroupa_IMF_extraction(N=1, m_min=0.01, m_max=100):
    
        alpha_1 = 0.3
        alpha_2 = 1.3
        alpha_3 = 2.3
    
        m_change_1 = 0.08
        m_change_2 = 0.5
    
        continuity_1 = m_change_1**(alpha_2-alpha_1)
        continuity_2 = continuity_1*m_change_2**(alpha_3-alpha_2)

        norm_1 = (m_change_1**(1-alpha_1)-m_min**(1-alpha_1))/(1-alpha_1)
        norm_2 = continuity_1*(m_change_2**(1-alpha_2)-m_change_1**(1-alpha_2))/(1-alpha_2)
        norm_3 = continuity_2*(m_max**(1-alpha_3)-m_change_2**(1-alpha_3))/(1-alpha_3)

        norm = norm_1 + norm_2 + norm_3

        P_m = np.random.uniform(0,1,N)
        mask_light_1 = (norm*P_m - norm_1 < 0)
        mask_light_2 = (norm*P_m - norm_2 - norm_1 < 0) & (norm*P_m - norm_1 > 0)
        mask_heavy = (norm*P_m - norm_2 - norm_1 > 0)

        P_light_1 = P_m[mask_light_1]
        P_light_2 = P_m[mask_light_2]
        P_heavy = P_m[mask_heavy]

        m_light_1 = ((1-alpha_1)*(norm*P_light_1) + m_min**(1-alpha_1))**(1/(1-alpha_1))
        m_light_2 = ((1-alpha_2)*(norm*P_light_2-norm_1)/continuity_1 + m_change_1**(1-alpha_2))**(1/(1-alpha_2))
        m_heavy = ((1-alpha_3)*(norm*P_heavy-norm_1-norm_2)/continuity_2 + m_change_2**(1-alpha_3))**(1/(1-alpha_3))

        m_tot = np.concatenate((m_light_1,m_light_2,m_heavy), axis=0)
        np.random.shuffle(m_tot)
    
        return m_tot

    masses = Kroupa_IMF_extraction(N=N_masses)
    
    while True:
        mask_masses = (masses < m_min) | (masses> m_max)
        masses_accepted_1 = masses[~mask_masses]
        N_rejected = mask_masses.sum()
        if N_rejected == 0:
            break
        masses_accepted_2 = Kroupa_IMF_extraction(N=N_rejected)
        masses = np.concatenate((masses_accepted_1,masses_accepted_2), axis=0)

    np.random.shuffle(masses)
    
    return masses


# In[3]:


masses = Kroupa_IMF(10**7, 0.1, 100)

m_mean = np.mean(masses)

print('Mean mass in the selected mass range:', m_mean)


# In[4]:


def stellar_masses(N_s, N_b, N_tot, masses, m_mean):
    
    m_s = masses[:N_s]
    m_s = m_s.reshape(N_s)
    m_b = masses[N_s:]
 
    m_1, m_2 = np.zeros(N_b), np.zeros(N_b)
    
    m_b = m_b.reshape(N_b, 2)
    mask_change = (m_b[:,0] - m_b[:,1] < 0)
    m_1[~mask_change], m_2[~mask_change] = m_b[~mask_change][:,0], m_b[~mask_change][:,1]
    m_1[mask_change], m_2[mask_change] = m_b[mask_change][:,1], m_b[mask_change][:,0]
    
    M = np.sum(np.concatenate((m_s,m_1,m_2), axis=0))
    M_tot = m_mean*N_tot
    
    correction = M_tot/M
    
    m_s = m_s*correction
    m_1 = m_1*correction
    m_2 = m_2*correction
    
    m_cm = m_1 + m_2
    m = np.concatenate((m_s,m_cm), axis=0)
    m_tot = np.concatenate((m_s,m_1,m_2), axis=0)
    
    M = int(np.sum(m_tot))
    
    return m_s, m_1, m_2, m_cm, m, m_tot, M


# In[5]:


def positions(R, N):
    
    P_r = np.random.uniform(0,1,N)
    P_theta = np.random.uniform(-1,1,N)
    P_phi = np.random.uniform(0,2*np.pi,N)

    theta = np.arccos(P_theta)
    phi = P_phi
    
    r = (1/np.sqrt(pow(P_r,-2/3) - 1))*R # pc
    
    X = np.zeros(shape=(N,3))

    X[:,0] = r*np.sin(theta)*np.cos(phi)
    X[:,1] = r*np.sin(theta)*np.sin(phi)
    X[:,2] = r*np.cos(theta)
    
    return r, X


# In[6]:


def single_positions(N_s, r, X):
    
    r_s = r[:N_s]
    X_s = X[:N_s]
    
    return X_s


# In[7]:


def binary_positions(N_s, N_b, r, X, m_1, m_2, a, e):
    
    r_cm = r[N_s:] # Positions of binary centers of mass
    X_cm = X[N_s:]
    
    r_apo = a*(1+e) # Apocenter radius

    nx = 2*np.random.uniform(0,1,N_b) - 1
    ny = 2*np.random.uniform(0,1,N_b) - 1
    nz = 2*np.random.uniform(0,1,N_b) - 1 
    
    nx_norm = nx/np.sqrt(nx**2+ny**2+nz**2)
    ny_norm = ny/np.sqrt(nx**2+ny**2+nz**2)
    nz_norm = nz/np.sqrt(nx**2+ny**2+nz**2)
    
    X_apo = np.zeros(shape=(len(r_apo),3))
    X_apo[:,0] = nx_norm*r_apo
    X_apo[:,1] = ny_norm*r_apo
    X_apo[:,2] = nz_norm*r_apo
    
    # Position components of primaries
    
    X_1 = np.zeros(shape=(len(r_apo),3))

    X_1[:,0] = X_cm[:,0] + m_2*X_apo[:,0]/(m_1+m_2)
    X_1[:,1] = X_cm[:,1] + m_2*X_apo[:,1]/(m_1+m_2)
    X_1[:,2] = X_cm[:,2] + m_2*X_apo[:,2]/(m_1+m_2)

    # Position components of secondaries
    
    X_2 = np.zeros(shape=(len(r_apo),3))
    
    X_2[:,0] = X_cm[:,0] - m_1*X_apo[:,0]/(m_1+m_2)
    X_2[:,1] = X_cm[:,1] - m_1*X_apo[:,1]/(m_1+m_2)
    X_2[:,2] = X_cm[:,2] - m_1*X_apo[:,2]/(m_1+m_2)

    return X_1, X_2


# In[8]:


def velocities(R, r, m, M, N):
    
    n_samples = 0
    v = []

    while n_samples < N:
        x_t = np.random.uniform(0,1)
        y_t = 0.1*np.random.uniform(0,1)
        
        if (x_t**2)*(1-x_t**2)**(3.5) > y_t: 
            v.append(x_t)
            n_samples +=1
            
    r = r/R
    ve = np.sqrt(2)*(1+r**2)**(-0.25)
    
    v = np.asarray(v)*ve*np.sqrt(G*M/R) # km/s
    v = 1.023*v # pc/Myr
        
    n1 = np.random.uniform(0,1,N)
    n2 = np.random.uniform(0,1,N)

    V = np.zeros(shape=(N,3))

    V[:,0] = (1 - 2*n1)*v
    V[:,1] = np.sqrt(pow(v,2) - pow(V[:,0],2))*np.sin(2*np.pi*n2)
    V[:,2] = np.sqrt(pow(v,2) - pow(V[:,0],2))*np.cos(2*np.pi*n2)

    return v, V


# In[9]:


def virial_ratio(R, G, m, M, X, v):

    T = np.sum(m*pow(v,2)/2)
    
    U = 0
    for i in range(len(X)-1):
        x_eff = X[i+1:,0]
        y_eff = X[i+1:,1]
        z_eff = X[i+1:,2]
        l = len(x_eff)
        
        x_i = X[i,0]*np.ones(l)
        y_i = X[i,1]*np.ones(l) 
        z_i = X[i,2]*np.ones(l)
        
        inverse_distance = 1/np.sqrt((x_i - x_eff)**2 + (y_i - y_eff)**2 +(z_i-z_eff)**2)
        U += -G*m[i]*np.sum(np.dot(m[i+1:],inverse_distance))
                
    Q = T/abs(U)
    
    return Q


# In[10]:


def single_velocities(N_s, v, V):
    
    v_s = v[:N_s] 
    V_s = V[:N_s]
    
    return V_s 


# In[11]:


def binary_velocities(N_s, N_b, v, V, m_1, m_2, a):
    
    v_cm = v[N_s:] # Velocities of binary centers of mass
    V_cm = V[N_s:]
    
    m_1.reshape(N_b)
    m_2.reshape(N_b)
      
    v_orb_mean = np.sqrt(G*(m_1+m_2)/a) # Mean orbital velocity

    nx = 2*np.random.uniform(0,1,N_b) - 1
    ny = 2*np.random.uniform(0,1,N_b) - 1
    nz = 2*np.random.uniform(0,1,N_b) - 1

    nx_norm = nx/np.sqrt(nx**2 + ny**2 + nz**2)
    ny_norm = ny/np.sqrt(nx**2 + ny**2 + nz**2)
    nz_norm = nz/np.sqrt(nx**2 + ny**2 + nz**2)
    
    V_apo = np.zeros(shape=(N_b,3))
    
    V_apo[:,0] = nx_norm*v_orb_mean
    V_apo[:,1] = ny_norm*v_orb_mean
    V_apo[:,2] = nz_norm*v_orb_mean
    
    # Velocity components of primaries
    
    V_1 = np.zeros(shape=(N_b,3))

    V_1[:,0] = V_cm[:,0] + m_2*V_apo[:,0]/(m_1+m_2)
    V_1[:,1] = V_cm[:,1] + m_2*V_apo[:,1]/(m_1+m_2)
    V_1[:,2] = V_cm[:,2] + m_2*V_apo[:,2]/(m_1+m_2)
    
    # Velocity components of secondaries
    
    V_2 = np.zeros(shape=(N_b,3))
    
    V_2[:,0] = V_cm[:,0] - m_1*V_apo[:,0]/(m_1+m_2)
    V_2[:,1] = V_cm[:,1] - m_1*V_apo[:,1]/(m_1+m_2)
    V_2[:,2] = V_cm[:,2] - m_1*V_apo[:,2]/(m_1+m_2)
    
    return V_1, V_2


# In[12]:


def semi_major_axis(N_b, a_au_min, a_au_max):
    
    X_a = np.random.uniform(0,1,N_b)
    norm_a = np.log(a_au_max/a_au_min)
     
    a_au = np.exp((norm_a*X_a)+np.log(a_au_min))
    a = a_au/206264.55529277
    
    return a_au, a


# In[13]:


def eccentricity(N_b, e_min, e_max):
    
    norm_e = pow(e_max,2) - pow(e_min,2)
    X_e = np.random.uniform(0,1,N_b)
    e = np.sqrt(norm_e*X_e + pow(e_min,2))

    return e


# In[39]:


def plot_distributions(plot=True):
    
    m_tot = Kroupa_IMF(10**5, 0.1, 100)
    a_au, a = semi_major_axis(10**5, 0.2, 100)
    e = eccentricity(10**5, 0, 1)
    
    distributions = plt.figure(figsize=(17,6), dpi=120)
    
    distributions.add_subplot(1,3,1)
    plt.hist(m_tot, bins=int(len(m_tot)/100), color='blue', alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('m', fontsize=15)
    plt.ylabel('f(m)', fontsize=15)
    plt.title('Kroupa IMF', fontsize=18)
    
    distributions.add_subplot(1,3,2)
    plt.hist(a, bins=int(len(a)/100), color='yellow', alpha=1)
    plt.xscale('log')
    plt.xlabel('a', fontsize=15)
    plt.ylabel('f(a)', fontsize=15)
    plt.title('Semi-major axis distribution', fontsize=18)
    
    distributions.add_subplot(1,3,3)
    plt.hist(e, bins=int(len(e)/100), color='green', alpha=0.5)
    plt.xlabel('e', fontsize=15)
    plt.ylabel('f(e)', fontsize=15)
    plt.title('Eccentricity distribution', fontsize=18)
    
    plt.show()
    
    return 0


# In[40]:


def main_program(N_tot, f_b, m_mean=0.64):
    
    f_b /= 100
    N_b = round(float(f_b*N_tot))
    N_s = N_tot - 2*N_b
    N = N_s + N_b
    
    masses = Kroupa_IMF(N_masses=N_tot, m_min=0.01, m_max=100)
    
    m_s, m_1, m_2, m_cm, m, m_tot, M = stellar_masses(N_s, N_b, N_tot, masses, m_mean)
    
    r, X = positions(1, N)
    
    X_s = single_positions(N_s, r, X)
    
    v, V = velocities(1, r, m, M, N)
    
    V_s = single_velocities(N_s, v, V)

    Q = virial_ratio(1, G, m, M, X, v)
    
    a_au, a = semi_major_axis(N_b, 0.2, 100)
    
    e = eccentricity(N_b, 0, 1)
    
    X_1, X_2 = binary_positions(N_s, N_b, r, X, m_1, m_2, a, e)
    
    V_1, V_2 = binary_velocities(N_s, N_b, v, V, m_1, m_2, a)
    
    plot_distributions(plot=True) # True to display the plot, False not to
    
    m_b = np.zeros(2*N_b) 

    X_b = np.zeros(shape=(2*N_b,3))

    V_b = np.zeros(shape=(2*N_b,3))

    for i in range(N_b):
    
        m_b[2*i] = m_1[i]
        m_b[2*i+1] = m_2[i]
        X_b[2*i] = X_1[i]
        X_b[2*i+1] = X_2[i]
        V_b[2*i] = V_1[i]
        V_b[2*i+1] = V_2[i]

    # Masses vector   
    
    masses = np.concatenate((m_b,m_s),axis=0)

    # Positions vector

    x_positions = np.concatenate((X_b[:,0],X_s[:,0]),axis=0) 
    y_positions = np.concatenate((X_b[:,1],X_s[:,1]),axis=0)
    z_positions = np.concatenate((X_b[:,2],X_s[:,2]),axis=0)

    # Velocities vector

    x_velocities = np.concatenate((V_b[:,0],V_s[:,0]),axis=0)
    y_velocities = np.concatenate((V_b[:,1],V_s[:,1]),axis=0)
    z_velocities = np.concatenate((V_b[:,2],V_s[:,2]),axis=0)

    # Dataframe to create the input file

    nbody_data = {'Masses': masses, 'Positions x': x_positions, 'Positions y': y_positions, 'Positions z': z_positions,
                  'Velocities x': x_velocities, 'Velocities y': y_velocities, 'Velocities z': z_velocities}
    nbody_df = pd.DataFrame(nbody_data)

    file_dat = nbody_df.to_csv(r'initial_conditions', sep=' ', index=False, header=False)
    
    return Q, N_b, file_dat


# In[41]:


G = 0.00449830997959438 # value for converting velocities from km/s to pc/Myr

Q, N_b, file_dat = main_program(10**3, 15, 0.64)


# In[17]:


# Useful information for the input file

print('Virial ratio :', Q)
print('Total number of binaries :', 2*N_b)

