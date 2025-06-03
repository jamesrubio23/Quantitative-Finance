import numpy as np
import matplotlib.pyplot as plt


class SDEModel:
    def __init__(self, drift, volatility, X0, T, dt, n_paths, seed=15, store_processes = True, name="SDEModel"):
        self.drift = drift
        self.volatility = volatility
        self.X0 =X0
        assert self.X0 > 0, "X0 must be strictly positive for the Geometric Brownian Motion"
        
        self.T=T
        self.dt=dt
        self.n_paths = n_paths
        self.simulated_paths = False #To see if there was any simulation. If not it will simualte n_paths
        self.store_processes = store_processes #Variable that tells if we store or not Xt if later we want to plot them
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)

        self.name=name #To check which process we are using

        #If they are inserted as functions of time
        if isinstance(drift, (int, float)):
            self.drift = lambda t: drift
        else:
            self.drift = drift
        
        if isinstance(volatility, (int, float)):
            self.volatility = lambda t: volatility
        else:
            self.volatility = volatility

        #Calculate N and vector t
        self.N = int(self.T/self.dt)
        self.t = np.linspace(0, self.T, self.N + 1)

        #Initialize the values of the process and the integrals
        Xt =np.zeros(self.N+1)
        Xt[0] =X0
        self.Xt = Xt

        self.normal_integral = np.zeros(self.N+1)
        self.Ito_integral= np.zeros(self.N+1) 
        self.integral= np.zeros(self.N+1)

        self.store_Xts = np.zeros((self.n_paths,self.N+1))
        self.store_integrals = np.zeros((self.n_paths, self.N + 1))
        
        self.final_values_Xt=np.zeros(self.n_paths)
        self.final_value_integral = np.zeros(self.n_paths)



    ###########################################
    ###Simulation of Path and Multiple Paths###
    ###########################################


    


    def simulate_paths(self):
        
        for i in range(self.n_paths):
            self.simulate_path()
            
            self.final_values_Xt[i] = self.Xt[-1]
            self.final_value_integral[i] = self.integral[-1]

            if self.store_processes == True:
                self.store_Xts[i] = self.Xt
                self.store_integrals[i] = self.integral

        self.simulated_paths=True


    ##################
    ###Calculations###
    ##################


    def mean(self):
        if self.simulated_paths==False:
            print("Simulating Paths since no Paths were simulated")
            self.simulate_paths()
        return np.mean(self.final_values_Xt, axis=0)
    
    def standard_dev(self):
        if self.simulated_paths==False:
            print("Simulating Paths since no Paths were simulated")
            self.simulate_paths()
        return np.std(self.final_values_Xt, axis=0)
    


    ##############
    ###Graphics###
    ##############
    

    def plot_paths(self, bins=100):
        if self.simulated_paths==False:
            print("Simulating Paths since no Paths were simulated")
            self.simulate_paths()

        fig, axis = plt.subplots(1,2,figsize=(10,6))


        for i in range(self.n_paths):
            axis[0].plot(self.t, self.store_Xts[i])
            axis[0].set_title(self.name)
            axis[0].set_xlabel('Time')
            axis[0].set_ylabel('Xt')

        axis[1].hist(self.final_values_Xt, bins, edgecolor='black')
        axis[1].set_xlabel('Final Value of Xt')
        axis[1].set_ylabel('Frequency')
        plt.show()

    def plot_random_samples(self, sample=5):
        pass

    def plot_integrals(self, bins=100):
        if self.simulated_paths==False:
            print("Simulating Paths since no Paths were simulated")
            self.simulate_paths()

        fig, axis = plt.subplots(1,2,figsize=(10,6))

 

        for i in range(self.n_paths):
            axis[0].plot(self.t, self.store_integrals[i])
            axis[0].set_title(f'Value of Integral of {self.name}')
            axis[0].set_xlabel('Time')
            axis[0].set_ylabel('Integral value')

        axis[1].hist(self.final_value_integral, bins, edgecolor='black')
        axis[1].set_xlabel('Final Value of Integral')
        axis[1].set_ylabel('Frequency')
        plt.show()

class GBM(SDEModel):
    def __init__(self, drift, volatility, X0, T, dt, n_paths, seed=15, store_processes=True):
        self.drift = drift
        self.volatility = volatility
        self.seed = seed
        super().__init__(drift, volatility, X0, T, dt, n_paths, seed, store_processes, name="Geometric Brownian Motion")



    def simulate_path(self):
        dBt = np.random.normal(0, np.sqrt(self.dt), self.N)
        for i in range(1, self.N + 1):
            left_dt = self.drift(self.t[i - 1]) * self.Xt[i - 1] * self.dt
            right_Bt = self.volatility(self.t[i - 1]) * self.Xt[i - 1] * dBt[i - 1]
            self.normal_integral[i] = self.normal_integral[i - 1] + left_dt
            self.Ito_integral[i] = self.Ito_integral[i - 1] + right_Bt
            self.integral[i] = self.normal_integral[i] + self.Ito_integral[i]
            self.Xt[i] = self.X0 + self.integral[i]

class OrnsteinUhlenbeck(SDEModel):
    def __init__(self, drift, volatility,theta, mu, sigma, X0, T, dt, n_paths, seed, store_processes=True):
        self.drift = drift
        self.volatility = volatility
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.seed = seed
        super().__init__(drift, volatility, X0, T, dt, n_paths, seed, store_processes, name="Ornstein Uhlenbeck")

    def simulate_path(self):
        dBt = np.random.normal(0, np.sqrt(self.dt), self.N)
        for i in range(1, self.N + 1):
            drift = self.drift(self.t[i - 1], self.Xt[i - 1]) * self.dt
            diffusion = self.volatility(self.t[i - 1], self.Xt[i - 1]) * dBt[i - 1]

            self.normal_integral[i] = self.normal_integral[i - 1] + drift
            self.Ito_integral[i] = self.Ito_integral[i - 1] + diffusion
            self.integral[i] = self.normal_integral[i] + self.Ito_integral[i]
            self.Xt[i] = self.X0 + self.integral[i]

class CIR(SDEModel):
    def __init__(self, drift, volatility, theta, mu, sigma, X0, T, dt, n_paths, seed, store_processes=True):
        self.drift = drift
        self.volatility = volatility
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.seed = seed
        super().__init__(drift, volatility, X0, T, dt, n_paths, seed, store_processes, name="CIR")

    def simulate_path(self):
        dBt = np.random.normal(0, np.sqrt(self.dt), self.N)
        for i in range(1, self.N + 1):
            drift_term = self.drift(self.t[i - 1], self.Xt[i - 1]) * self.dt
            diffusion_term = self.volatility(self.t[i - 1], self.Xt[i - 1]) * dBt[i - 1]


            self.normal_integral[i] = self.normal_integral[i - 1] + drift_term
            self.Ito_integral[i] = self.Ito_integral[i - 1] + diffusion_term
            self.integral[i] = self.normal_integral[i] + self.Ito_integral[i]
            self.Xt[i] = max(0, self.X0 + self.integral[i]) 