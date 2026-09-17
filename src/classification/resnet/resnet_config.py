"""
resnet_config.py
-----------------
Singleton configuration wrapper for the ResNet lesion-classification
pipeline (direct-image → Normal / Variation / OPMD).

Mirrors UNetConfig exactly:
  - Singleton via __new__ so the config is parsed only once per process.
  - _load() maps every .ini key to a typed Python attribute.

Usage
-----
    from utils.load_configuration import load_config
    from src.classification.resnet.resnet_config import ResNetConfig

    base_cfg = load_config()                                   # config.ini
    cls_ini  = load_config(base_cfg.get("CLASSIFY-RESNET", "resnet.config"))
    cfg      = ResNetConfig(cls_ini)

    # Subsequent import anywhere in the process — same instance, no re-parse:
    cfg2 = ResNetConfig(None)
    assert cfg is cfg2
"""


class ResNetConfig:
    _instance = None

    def __new__(cls, config):
        if cls._instance is None:
            if config is None:
                raise ValueError("config must be provided on first initialization")
            instance = super().__new__(cls)
            instance._load(config)
            cls._instance = instance
        return cls._instance

    def _load(self, config):
        # ── TRAINING ─────────────────────────────────────────────────
        train = config["TRAINING"]
        self.epochs = train.getint("epochs")
        self.batch_size = train.getint("batch_size")
        self.backbone_lr = train.getfloat("backbone_lr")
        self.head_lr = train.getfloat("head_lr")
        self.momentum = train.getfloat("momentum")
        self.weight_decay = train.getfloat("weight_decay")
        self.gradient_clip = train.getfloat("gradient_clip")
        self.val_split = train.getfloat("val_split")
        self.test_split = train.getfloat("test_split")
        self.loss_function = train.get("loss_function")
        # Supported values: "ce" | "weighted_ce" | "focal"
        self.focal_gamma = train.getfloat("focal_gamma")
        self.lr_scheduler = train.get("lr_scheduler")
        self.min_lr = train.getfloat("min_lr")
        self.lr_patience = train.getint("lr_patience")
        self.lr_factor = train.getfloat("lr_factor")
        self.num_workers = train.getint("num_workers")
        self.log_every = train.getint("log_every")
        self.val_every = train.getint("val_every")

        # ── MODEL ─────────────────────────────────────────────────────
        model = config["MODEL"]
        self.backbone = model.get("backbone")
        self.pretrained = model.getboolean("pretrained")
        self.num_classes = model.getint("num_classes")

        # ── SYSTEM ────────────────────────────────────────────────────
        system = config["SYSTEM"]
        self.device = system.get("device")
        self.seed = system.getint("seed")

        # ── LOGGING ───────────────────────────────────────────────────
        logging_ = config["LOGGING"]
        self.save_best_model = logging_.getboolean("save_best_model")

        # ── PATHS ─────────────────────────────────────────────────────
        paths = config["PATHS"]
        self.model_dir = paths.get("model_dir")
        self.output_dir = paths.get("output_dir")
        self.checkpoint_dir = paths.get("checkpoint_dir")

        # ── DATASET ───────────────────────────────────────────────────
        dataset = config["DATASET"]
        self.image_size = dataset.getint("image_size")
