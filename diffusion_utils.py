import torch

class DiffusionUtils:
    def __init__(self, n_timesteps:int, beta_min:float, beta_max:float, device:str='cuda', scheduler:str='linear'):
        assert scheduler in ['linear', 'cosine'], 'scheduler must be linear or cosine'

        self.n_timesteps = n_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = device
        self.scheduler = scheduler
        
        self.betas = self.betaSamples()
        self.alphas = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
    
    
    def betaSamples(self):
        if self.scheduler == 'linear':
            return torch.linspace(start=self.beta_min, end=self.beta_max, steps=self.n_timesteps).to(self.device)

        elif self.scheduler == 'cosine':
            betas = []
            for i in reversed(range(self.n_timesteps)):
                T = self.n_timesteps - 1
                beta = self.beta_min + 0.5*(self.beta_max - self.beta_min) * (1 + np.cos((i/T) * np.pi))
                betas.append(beta)
                
            return torch.Tensor(betas).to(self.device)
    
    
    def sampleTimestep(self, size:int):
        #the size argument will let you randomly sample a batch of timesteps
        #output shape: (N, )
        return torch.randint(low=1, high=self.n_timesteps, size=(size, )).to(self.device)
    
    
    def noiseImage(self, x:torch.Tensor, t:torch.LongTensor):
        #expected input is a batch of inputs.
        #image shape: (N, C, H, W)
        #t:torch.Tensor shape: (N, )
        assert len(x.shape) == 4, 'input must be 4 dimensions'
        alpha_hat_sqrts = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        one_mins_alpha_hat_sqrt = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x).to(self.device)
        return (alpha_hat_sqrts * x) + (one_mins_alpha_hat_sqrt * noise), noise
    
    
    # def sample(self, x:torch.Tensor, model:nn.Module):
    #     # NOTE : for inference
    #     #x shape: (N, C, H, W)
    #     assert len(x.shape) == 4, 'input must be 4 dimensions'
    #     model.eval()
        
    #     with torch.no_grad():
    #         iterations = range(1, self.n_timesteps)
    #         for i in tqdm.tqdm(reversed(iterations)):
    #             #batch of timesteps t
    #             t = (torch.ones(x.shape[0]) * i).long().to(self.device)
                
    #             #params
    #             alpha = self.alphas[t][:, None, None, None]
    #             beta = self.betas[t][:, None, None, None]
    #             alpha_hat = self.alpha_hat[t][:, None, None, None]
    #             one_minus_alpha = 1 - alpha
    #             one_minus_alpha_hat = 1 - alpha_hat
                
    #             #predict noise pertaining for a given timestep
    #             predicted_noise = model(x, t)
                
    #             if i > 1:noise = torch.randn_like(x).to(self.device)
    #             else:noise = torch.zeros_like(x).to(self.device)
                
    #             x = 1/torch.sqrt(alpha) * (x - ((one_minus_alpha / torch.sqrt(one_minus_alpha_hat)) * predicted_noise))
    #             x = x + (torch.sqrt(beta) * noise)
                
    #         return x