from .sync_trainer import SyncTrainer

TRAINER_REGISTRY = {
    'sync': SyncTrainer,
}