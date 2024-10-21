from torch.distributed import init_process_group, destroy_process_group
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloaders import HDF5Dataset
from report import ReportGenerator
from tqdm import tqdm
from optimizers import adamw_cosine_warmup_wd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import os


class DDPClassificationTrainer:
    def __init__(
        self,
        model,
        model_config,
        model_name,
        hdf5_dataset_train_config,
        train_data_frac,
        hdf5_dataset_val_config,
        val_data_frac,
        save_path,
        params_to_save,
        seed,
        batch_size,
        epochs,
        start_lr,
        ref_lr,
        final_lr,
        wd,
        final_wd,
        ipe_scale,
        warmup_epochs,
        opt_config,
        master_addr,
        master_port,
        backend,
        main_device,
        process_timeout,
    ):
        self.model = model
        self.model_config = model_config
        self.model_name = model_name
        self.hdf5_dataset_train_config = hdf5_dataset_train_config
        self.train_data_frac = train_data_frac
        self.hdf5_dataset_val_config = hdf5_dataset_val_config
        self.val_data_frac = val_data_frac
        self.save_path = save_path
        self.params_to_save = params_to_save
        self.seed = seed
        self.batch_size = batch_size
        self.epochs = epochs
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.wd = wd
        self.final_wd = final_wd
        self.ipe_scale = ipe_scale
        self.warmup_epochs = warmup_epochs
        self.opt_config = opt_config
        self.master_addr = master_addr
        self.master_port = master_port
        self.backend = backend
        self.main_device = main_device
        self.process_timeout = process_timeout

        self.init_model = None
        self.metrics = {
            "loss": ["train", "val"],
            "learning_rate": ["learning_rate"],
            "weight_decay": ["weight_decay"],
        }
        self.best_metrics_obj = {
            "loss/val": "min",
        }

        # self.mp_manager = mp.Manager()
        # self.best_models = self.mp_manager.dict()

        os.makedirs(self.save_path, exist_ok=True)

    def ddp_setup(self, rank, world_size):
        init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            timeout=(
                None
                if not self.process_timeout
                else timedelta(seconds=self.process_timeout)
            ),
        )

    def ddp_cleanup(self):
        destroy_process_group()

    def init_models(self):
        self.init_model = self.model(**self.model_config)

    def launch_ddp_models(self, device):
        model = self.model(**self.model_config).to(device)
        model.load_state_dict(self.init_model.state_dict())

        return DDP(model, device_ids=[device])

    def train_ddp(self, rank, world_size):
        self.ddp_setup(rank, world_size)

        self.train(
            rank=rank,
            world_size=world_size,
        )

        self.ddp_cleanup()

    def spawn_train_ddp(self):
        self.init_models()

        world_size = torch.cuda.device_count()

        print("---- Initiating DDP training ----")
        print(f"CUDA device count: {world_size}")
        print(f"Master address: {self.master_addr}")
        print(f"Master port: {self.master_port}")

        self.report_generator = ReportGenerator(
            save_path=self.save_path,
            main_device=self.main_device,
            metrics=self.metrics,
            trainer=self,
            params_to_save=self.params_to_save,
            best_metrics_obj=self.best_metrics_obj,
        )
        self.report_generator.save_params()
        self.report_generator.save_model_configs({self.model_name: self.model_config})

        mp.spawn(
            self.train_ddp,
            args=(world_size,),
            nprocs=world_size,
        )

    def train(
        self,
        rank,
        world_size,
    ):
        train_loader = HDF5Dataset.get_dataloader(
            self.hdf5_dataset_train_config,
            batch_size=self.batch_size,
            num_workers=world_size,
            world_size=world_size,
            rank=rank,
            shuffle=True,
            seed=self.seed,
            data_frac=self.train_data_frac,
        )

        val_loader = HDF5Dataset.get_dataloader(
            self.hdf5_dataset_val_config,
            batch_size=self.batch_size,
            num_workers=world_size,
            world_size=world_size,
            rank=rank,
            shuffle=True,
            seed=self.seed,
            data_frac=self.val_data_frac,
        )

        model = self.launch_ddp_models(rank)
        optimizer, _, scheduler, wd_scheduler = adamw_cosine_warmup_wd(
            model=model,
            iterations_per_epoch=len(train_loader),
            start_lr=self.start_lr,
            ref_lr=self.ref_lr,
            warmup=self.warmup_epochs,
            num_epochs=self.epochs,
            final_lr=self.final_lr,
            wd=self.wd,
            final_wd=self.final_wd,
            ipe_scale=self.ipe_scale,
            opt_config=self.opt_config,
        )
        criterion = F.cross_entropy

        for epoch in range(self.epochs):
            with tqdm(
                total=(len(train_loader) + len(val_loader)),
                desc=f"Epoch {epoch+1}",
                disable=(rank != self.main_device),
            ) as bar:
                self.report_generator.init_epoch_metrics_dict(
                    epoch=epoch,
                    device=rank,
                )

                model.train()
                for data_batch, label_batch in train_loader:
                    data_batch = data_batch.to(rank)
                    label_batch = label_batch.to(rank)

                    # 1. Zero grad
                    optimizer.zero_grad()

                    # 2. Fwd pass
                    model_out = model(data_batch)

                    # 3. Compute loss
                    loss = criterion(model_out, label_batch)

                    # 4. Backprop grad
                    loss.backward()

                    # 5. Update model
                    optimizer.step()

                    # 6. Update learning rate
                    lr = scheduler.step()

                    # 7. Update weight decay
                    wd = wd_scheduler.step()

                    self.report_generator.add_epoch_metric(
                        path="loss/train",
                        value=loss.item(),
                        device=rank,
                    )
                    self.report_generator.add_epoch_metric(
                        path="learning_rate",
                        value=lr,
                        device=rank,
                    )
                    self.report_generator.add_epoch_metric(
                        path="weight_decay",
                        value=wd,
                        device=rank,
                    )
                    bar.set_postfix(
                        {
                            "train_loss": self.report_generator.get_last_epoch_metric(
                                path="loss/train"
                            ),
                            "val_loss": self.report_generator.get_last_epoch_metric(
                                path="loss/val"
                            ),
                        }
                    )
                    bar.update(1)

                with torch.no_grad():
                    model.eval()
                    for data_batch, label_batch in val_loader:
                        data_batch = data_batch.to(rank)
                        label_batch = label_batch.to(rank)

                        model_out = model(data_batch)
                        loss = criterion(model_out, label_batch)

                        self.report_generator.add_epoch_metric(
                            path="loss/val",
                            value=loss.item(),
                            device=rank,
                        )
                        bar.set_postfix(
                            {
                                "train_loss": self.report_generator.get_last_epoch_metric(
                                    path="loss/train"
                                ),
                                "val_loss": self.report_generator.get_last_epoch_metric(
                                    path="loss/val"
                                ),
                            }
                        )
                        bar.update(1)

                self.report_generator.update_global_metrics(device=rank)
                self.report_generator.save_best_models(
                    models={self.model_name: model.module},
                    device=rank,
                )
                self.report_generator.save_metrics(device=rank)
                self.report_generator.save_plots(device=rank)
