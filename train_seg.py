import wandb
import torch

from tqdm import tqdm

from datasets.BaseDataset import BaseDataset

from config import Config

from trainer import trainer


def main(pairs: list, config: Config):
    print("Number of experiments: ", len(pairs))
    
    pbar = tqdm(pairs, desc="Training", position=0)
    for dataset, model_cls in pbar:

        dataset: BaseDataset = dataset(config=config)
        config.dataset.dataset_name = dataset.dataset_name

        print(f"\nTraining {model_cls.__name__} on {config.dataset.dataset_name}")

        if config.dataset.dataset_name in [
            "CHASEDB1", "HRF"
        ]:
            config.training.batch_size = 4
            config.training.valid_ratio = 0.1
        else:
            config.training.batch_size = 8
            config.training.valid_ratio = 0.2

        config.dataset.dataset_name = dataset.dataset_name
        
        config.model.model_name = model_cls.__name__
        config.model.num_classes = dataset.num_classes
        config.model.temperature = None
        
        criterion = torch.nn.BCEWithLogitsLoss() if config.model.num_classes == 1\
                else torch.nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam

        model = model_cls(
            in_channels=config.model.in_channels,
            n_classes=config.model.num_classes,
            criterion=criterion,
            optimizer=optimizer,
            config=config   
        )

        training_fail = False
        try:
            trainer(
                model=model,
                dataset=dataset,
                config=config
            )
        except KeyboardInterrupt:
            print("Training interrupted")
            training_fail = True
        except Exception as e:
            print("Training failed")
            training_fail = True
            from traceback import print_exc
            print_exc()
        else:
            print("Training completed")
            training_fail = False
        finally:
            wandb.finish(training_fail)
            torch.cuda.empty_cache()
            if training_fail:
                return

if __name__ == "__main__":
    main()