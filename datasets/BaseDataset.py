from torch.utils.data import Dataset as TorchDataset

class BaseDataset(TorchDataset):
    
    @property
    def dataset_name(self):
        return self.__class__.__name__
    
    @property
    def num_classes(self):
        raise NotImplementedError
    
    def __init__(
            self,
            root: str,
        ) -> None:
        super().__init__()

        self.root = root
        

    def __len__(self):
        raise NotImplementedError
    

    def __getitem__(self, idx):
        raise NotImplementedError
    

    def collate_fn(self, batch):
        raise NotImplementedError
    

    def get_img(self, idx):
        raise NotImplementedError
    

    def get_ann(self, idx):
        raise NotImplementedError


    def get_img_ann(self, idx):
        raise NotImplementedError
    
