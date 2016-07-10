class Args:
    def __init__(self, **kwargs):
        self.seed = 1
        self.gpu  = 1
        self.use_cpu = False
        self.save_models = False
        self.verbose = True
        self.one_epoch = False
        self.unk_cut = 0
        self.batch = 1
        self.src_dev = ""
        self.trg_dev = ""
        self.optimizer = "adam"
        self.debug = True
        self.init_model = ""
        self.gen_limit = 10 
        self.eos_disc = 0
        self.beam = 10
        self.align_out = None
        self.dict_caching = True
        self.dict_method = "bias"
        for key, value in kwargs.items():
            setattr(self, key, value)
