"""RoboFollow opt-in runner: preserve generic and strict dual-camera paths."""
from pathlib import Path
import json
from runner.ge_trainer import Trainer
from ge_act.experiments.robofollow_loading import load_robofollow_weights, ensure_vae_channels

class RoboFollowTrainer(Trainer):
    def prepare_models(self):
        original=getattr(self.args,'load_diffusion_model_weights',True)
        self.args.load_diffusion_model_weights=False
        try:super().prepare_models()
        finally:self.args.load_diffusion_model_weights=original
        ensure_vae_channels(self.vae)
        if self.args.load_weights and original:
            report=load_robofollow_weights(self.diffusion_model,Path(self.args.diffusion_model['model_path']),set(self.raw_config.get('robofollow_allowed_new_keys',[])))
            if self.state.accelerator.is_main_process:
                (Path(self.save_folder)/'weight_transfer.json').write_text(json.dumps(report,indent=2)+'\n')
