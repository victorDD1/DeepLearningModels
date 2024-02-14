import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Any, Mapping, Optional, Union

DEFAULT_BETA_START = 1e-4
DEFAULT_BETA_END = 0.02
DEFAULT_BETASCHEDULE = 'squaredcos_cap_v2'
DEFAULT_TIMESTEPS = 1000
DEFAULT_CLIPSAMPLE = False
DEFAULT_TRAINING_SHAPE = [0,0,0]

class DDPM(nn.Module):
    def __init__(self,
                 eps_model:nn.Module,
                 **kwargs) -> None:
        """
        Diffusion Probabilistic Model (DDPM) Constructor.

        Args:
            - eps_model: The epsilon model.
            - **kwargs: Additional optional parameters for DDPM.
        """

        super(DDPM, self).__init__()
        self.eps_model = eps_model

        self.optional_parameters = {
            **{
            "beta_start" : DEFAULT_BETA_START,
            "beta_end" : DEFAULT_BETA_END,
            "beta_schedule" : DEFAULT_BETASCHEDULE,
            "timesteps" : DEFAULT_TIMESTEPS,
            "clip_sample" : DEFAULT_CLIPSAMPLE,
            "training_shape" : DEFAULT_TRAINING_SHAPE,
            },
            **kwargs # kwargs overrides optional if provided
        }
        self._set_parameters()
        self._set_noise_scheduler()

    def _set_parameters(self):
        """
        Set optional parameters as attributes or buffers.
        """
        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in self.optional_parameters.items():
            try:
                v = torch.tensor(v)
            except:
                setattr(self, k, v)
                continue
            self.register_buffer(k, v)

    def _set_noise_scheduler(self):
        """
        Initialize and set the noise scheduler.
        """
        self.noise_scheduler = DDPMScheduler(
            beta_start=self.beta_start.item(),
            beta_end=self.beta_end.item(),
            beta_schedule=self.beta_schedule, # big impact on performance
            num_train_timesteps=self.timesteps.item(),            
            clip_sample=self.clip_sample.item(), # clip output to [-1,1] to improve stability
            prediction_type='epsilon' # our network predicts noise (instead of denoised action)
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict, assign)
        self._set_noise_scheduler()

    def cast_timesteps(self,
                       sample:torch.Tensor,
                       timesteps:Optional[int]):
        """
        Cast and broadcast timesteps.

        Args:
            - sample: The input sample.
            - timesteps: The diffusion steps.

        Returns:
            - The cast and broadcasted timesteps.
        """
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        return timesteps
        
    def forward(self,
                sample:torch.Tensor,
                timesteps:Optional[int] = None,
                global_cond:Optional[torch.Tensor] = None):
        """
        Perform forward diffusion process.

        Args:
            - sample: The input sample.
            - timesteps: The diffusion steps.
            - global_cond: The global conditioning tensor.

        Returns:
            - estimated_noise: Estimated noise.
            - noise: Generated noise.
        """
        if (self.training_shape == torch.zeros((3,), device=sample.device)).all():
            self.training_shape = torch.tensor(sample.shape, device=sample.device).long()
            
        if timesteps == None:
            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (sample.shape[0],),
                device=sample.device
            ).long()

        timesteps = self.cast_timesteps(sample, timesteps)

        # Forward diffusion process
        noise = torch.randn(sample.shape, device=sample.device)
        noisy_data_samples = self.noise_scheduler.add_noise(sample, noise, timesteps)
        estimated_noise = self.eps_model(noisy_data_samples, timesteps, global_cond)

        return estimated_noise, noise
    
    @torch.no_grad()
    def sample(self,
               size:Optional[torch.Size]=None,
               num_inference_steps:int=DEFAULT_TIMESTEPS,
               condition:Optional[torch.Tensor]=None,
               intermediate_steps:int=1,
               ):
        """
        Generate samples using backward diffusion process.

        Args:
            - size: The size of the generated samples.
            - num_inference_steps: Number of inference steps.
            - condition: The conditioning tensor.
            - intermediate_steps: Number of intermediate steps.

        Returns:
            - intermediate_generated_samples: Intermediate generated samples.
        """

        assert (hasattr(self, "training_shape") or size != None),\
            "Please set attribute sample_shape or provide size argument"
        
        device = "cpu" if not(torch.cuda.is_available()) else "cuda"
        device = condition.device if condition != None else device

        if size != None:
            noise_shape = size.tolist()
        elif hasattr(self, "training_shape"):
            noise_shape = self.training_shape.tolist()
        if condition != None:
            noise_shape[0] = condition.shape[0]
        noise_shape = torch.Size(noise_shape)

        generated_samples = torch.randn(noise_shape, device=device)
        self.noise_scheduler.set_timesteps(num_inference_steps, device)

        intermediate_generated_samples = torch.empty((intermediate_steps, *noise_shape), device=device)
        intermediate_t = torch.linspace(self.noise_scheduler.timesteps[0], 0, intermediate_steps).long()
        intermediate_t = torch.nn.Tanh()(intermediate_t * 0.2 / self.noise_scheduler.timesteps[0]) * 2. * self.noise_scheduler.timesteps[0]
        
        # Backward diffusion process
        additional_step = 0
        for t in self.noise_scheduler.timesteps:

            timesteps = self.cast_timesteps(generated_samples, t)
            noise_pred = self.eps_model(generated_samples, timesteps, condition)

            # Inverse diffusion step (remove noise)
            generated_samples = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=generated_samples
            ).prev_sample

            if t <= intermediate_t[additional_step]:
                intermediate_generated_samples[additional_step] = generated_samples
                additional_step += 1
        
        intermediate_generated_samples = intermediate_generated_samples.squeeze()
        return intermediate_generated_samples