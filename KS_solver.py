import numpy as np

def KS_step(KS, num_steps, y0, all_steps=True):
    y_full = np.zeros((num_steps+1, len(y0)))
    y_full[0] = y0
    for i in range(num_steps):
        y_full[i+1] = KS.step(y_full[i])

    if all_steps:
        return y_full
    else:
        return y_full[-1] 

def KS_step_nudge(KS, num_steps, y0, obs, all_steps=True):
    y_full = np.zeros((num_steps+1, len(y0)))
    y_full[0] = y0
    for i in range(num_steps):
        y_full[i+1] = KS.step_nudge(y_full[i], obs[i])

    if all_steps:
        return y_full
    else:
        return y_full[-1] 
    
def ml_step(ys, nsteps, params, stencil_x, all_steps=True):
   nx = len(ys[0])
   params_x = params[:stencil_x]
   params_t = params[stencil_x:]
   A = make_matrix(params_x, nx, stencil_x)

   sol = solver(A, ys, nsteps, params_t, all_steps)
   return sol

def make_matrix(params, nx, stencil):
   A = np.eye(nx)*params[0]
   for i in range(1,len(params)):
      A += np.diag(np.ones(nx-i)*params[i], k=-i)
   if len(params)+1 == stencil:
      A += np.diag(np.ones(nx-stencil)*(1-np.sum(params)), k=-stencil)
   for i in range(stencil-1):
      A[i,nx-stencil+1+i:] = params[i+1:][::-1]
   return A

def solver(A, ys, nsteps, params_t=None, all_steps=True):
   if params_t is not None:
      nt = len(params_t)
      params_t = np.expand_dims(params_t, 1)
   else:
      nt = 0
   y_next = np.zeros((nsteps+nt+1,len(A)))
   y_next[:nt+1,:] = ys

   if nt == 0:
      for i in range(nt, nsteps+nt):
         y_next[i+1,:] = A @ y_next[i,:]
   else:
      for i in range(nt, nsteps+nt):
         y_next[i+1,:] = A @ y_next[i,:] + np.sum(params_t*y_next[i-nt:i,:], axis=0)
   if all_steps:
      return y_next
   else:
      return y_next[-1]