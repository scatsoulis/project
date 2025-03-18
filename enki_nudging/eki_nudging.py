import numpy as np
from KS_solver import KS_step, KS_step_nudge
from jax_models import *
from enki import EKI
import pickle


# ENKI for nudging parameter \mu
# Learn optimal nudging parameter for given system
# Can change resolution of low-resolution model,
# the temporal frequency with which you nudge, 
# and other parameters

## Forward model G, all ensemble members
# list of mu
def ensembleG(mu, data, params):
   KS, KS_low, T, burn, eta, obst_cf = params
   N = len(mu)
   cf = KS.s//KS_low.s

   # Cut off initial burn-in period 
   data = data[burn:]
   std = np.std(data[:,0])

   g_nudge = np.zeros((N, len(data), KS_low.s))

   # Generate low-resolution data from same starting point as our high-resolution data
   # data_low = np.zeros((N, len(data[0]), KS_low.s))
   for i in range(N):
      KS_low.mu = 10**mu[i]
      obs = data[:,::cf]
      obs = obs + np.random.normal(scale=eta*std, size=obs.shape)
      # data_low[i,:,:] = KS_step(KS_low, len(data[i])-1, data[i,0,::cf]) 
      g_nudge[i,:,:] = KS_step_nudge(KS_low, len(data)-1, data[0,::cf], obs, obst_cf) 
   
   # for i in range(N):
   #    print(mu[i], np.count_nonzero(np.isnan(g_nudge[i,:,-1])))
   return g_nudge[:,-1,:]

# def ensembleG(ui, z0, y):
#     ## Parallel computing of forward model G
#     Gmatrix = np.zeros([ui.shape[0],44])
#     pool = multiprocessing.Pool(multiprocessing.cpu_count())
#     results = []
#     for iterN in range(ui.shape[0]):
#         results.append(pool.apply_async(G, (ui[iterN,:],z0,y)))
#     iterN = 0
#     for result in results:
#         Gmatrix[iterN,:] = result.get()
#         iterN = iterN+1
#     pool.close()
#     return Gmatrix


def main():

   path = '/central/groups/esm/reusebi/acm154_final'
   for i in range(1,51):

      # define our KS domain and system
      s = 256 # number of points in domain
      M=16
      l = 2 * np.pi * 16  # domain length
      dt = .25  # time step size
      T = 2000  # total simulation time
      burn = 400 #burn-in time
      KS = KuramotoSivashinsky(dt=dt, s=s, l=l, M=M)

      # Define parameters for nudging
      eta = 1e-1 # standard deviation of observation gaussian noise relative to std of KS output

      # Define KS characteristics for low-resolution system
      cf = 4 # Coarsen factor
      obst_cf = i # Let's coarsen the observations in time
      KS_low = KuramotoSivashinsky(dt=dt, s=s//cf, l=l, M=M)

      # params for running our forward model
      params = KS, KS_low, T, burn, eta, obst_cf

      # Generate our high resolution simulation data (our truth)
      x = np.arange(0,KS.l, KS.l/KS.s)
      data = np.zeros((T+1, KS.s))
      f = f_rand(KS.l)
      data = KS_step(KS, T, f(x))
      y = data[-1,::cf]
      
      ## Set up EKI to calibrate the parameters in error model
      DAsteps = 10 # num of EnKI steps
      numS = 200 # num of ensembles in EKI

      # covariance matrix on the observations
      # maybe this should actually be calculated as the covariances between
      # all the spatial points in the KS true high resolution output?
      cov = np.eye(KS_low.s)*0.001**2  
      logmus = np.ones((numS,1))
      # distribute mu evenly in log space between 10e-3 and 10e1.3
      logmus[:,0] = np.random.uniform(-3, 1, numS)
      
      
      ## Run EKI
      eki = EKI(logmus, y, cov, 1)
      print("Meanlogmu", round(np.mean(logmus), 3))
      print("Std logmu", round(np.std(logmus), 3))
      for iterN in range(DAsteps):
         print(iterN)
         print('DA step: ', iterN+1)
         eki.EnKI(ensembleG(eki.u[iterN][:,0], data, params))
         print("Error: ", eki.error[-1])
         print("Mean logmu", round(np.mean(eki.u[iterN+1]), 3))
         print("Std logmu", round(np.std(eki.u[iterN+1]), 3))
         pickle.dump(eki.u, open(f"{path}/more_results/cf{cf}_tcf{obst_cf}_mu.pkl", "wb"))
         pickle.dump(eki.g, open("g.pkl", "wb"))

      plt.hist(eki.u[-1])
      plt.savefig(f"{path}/mu.png")
   

def f_rand(L):
   nwaves = 5
   a = np.arange(nwaves)+1
   b = np.random.rand(nwaves)
   d = np.random.rand(nwaves)

   xi = np.linspace(0,L,1000)
   fx = xi*0
   for i in range(nwaves):
      fx = fx + b[i]*np.sin(2*a[i]*np.pi*(xi-d[i])/L)
      
   mina = np.min(fx)
   maxa = np.max(fx)
   maxx = max(np.abs(mina), maxa)

   def f(x):
      fx = x*0
      for i in range(nwaves):
         fx = fx + b[i]*np.sin(2*a[i]*np.pi*(x-d[i])/L)
         
      return fx/maxx

   return f

if __name__ == '__main__':
   main()