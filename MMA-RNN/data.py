from torch.utils.data import Dataset

class MMECG(Dataset):
    def __init__(self, dataMat, labelMat, LenMat, SeqLabelMat, OrdMat, numMat, index_list):
        super().__init__()
        self.dataMat = dataMat
        self.labelMat = labelMat
        self.LenMat = LenMat
        self.SeqLabelMat = SeqLabelMat
        self.index_list = index_list
        self.ordMat = OrdMat
        self.numMat = numMat

    def __len__(self):
        return len(self.dataMat)

    def __getitem__(self, idx):
        id = self.index_list[idx]

        return {'id':self.numMat[id],
                'ord':self.ordMat[id],
                'dataMat': self.dataMat[id],
                'labelMat': self.labelMat[id],
                'lenMat': self.LenMat[id],
                'seqLabelMat': self.SeqLabelMat[id]}
