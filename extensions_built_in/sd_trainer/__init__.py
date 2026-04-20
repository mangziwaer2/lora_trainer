# This is an example extension for custom training. It is great for experimenting with new ideas.
from toolkit.extension import Extension


class SDTrainerExtension(Extension):
    uid = "sd_trainer"
    name = "SD Trainer"

    @classmethod
    def get_process(cls):
        from .SDTrainer import SDTrainer

        return SDTrainer


class DiffusionTrainerExtension(Extension):
    uid = "diffusion_trainer"
    name = "Diffusion Trainer"

    @classmethod
    def get_process(cls):
        from .DiffusionTrainer import DiffusionTrainer

        return DiffusionTrainer


AI_TOOLKIT_EXTENSIONS = [
    SDTrainerExtension,
    DiffusionTrainerExtension,
]
