import torch
from torch import Tensor
from torch import tensor
import torchvision as tn
import torchvision.transforms.v2.functional as F
import typing as t
from typing import TypeAlias
import pathlib as pl
import pandas as ps
import torch.utils.data as data


DataSplit: TypeAlias = t.Literal['train', 'valid', 'test']


class GICDataset(data.Dataset[t.Tuple[Tensor, Tensor] | Tensor]):
    # Precomputed dataset statistics (train + validation)
    mean: Tensor = tensor([0.4282, 0.4064, 0.3406])
    std: Tensor = tensor([0.2144, 0.2187, 0.2046])
    num_classes: int = 100

    def __init__(self,
                 path: pl.Path,
                 split: DataSplit) -> None:
        super(GICDataset, self).__init__()

        # Map names
        if split == 'valid':
            self.split_: str = 'val'
        else:
            self.split_: str = split

        # Paths
        self.__root = path
        self.__meta = self.__root / f'{self.split_}.csv'
        self.__imag = self.__root / f'{self.split_}_images'

        # Internal data table
        self.data_ = ps.read_csv(self.__meta)

        # Throw out black images
        self.data_.drop(self.data_[self.data_['Image'].isin(GICDataset.black_images())].index, inplace=True)

    def __getitem__(self, index: int):
        image_name: str = self.data_.iloc[index].loc['Image']
        image_path: str = str(self.__imag / image_name)
        image: Tensor = tn.io.read_image(image_path, tn.io.ImageReadMode.RGB)
        image = F.to_dtype(image, torch.float, scale=True)

        if self.split_ == 'test':
            return image

        label: Tensor = torch.tensor(self.data_.iloc[index].loc['Class'])
        return (image, label)

    def __len__(self) -> int:
        return len(self.data_)

    @classmethod
    def black_images(cls):
        return ps.Series(name='Image', data=GICDataset.trs([
            'trmljjrl+wvox+nmpl+utto+jxrqxtmynlpj.86z',
            'uqqultrl+pkqq+nnqu+sukm+oovnsmuplsrl.86z',
            'utqrjplp+wvwp+notx+tomm+mtqlowvrnmrv.86z',
            'uwswwxkn+jksv+njxv+twpr+txqmnqorwynp.86z',
            'vktnsmjm+sknw+nyvy+uorx+yononoxlvuyu.86z',
            'vmsoqpqq+lpts+npwx+sqju+uorwnkoxonys.86z',
            'vrtoxrmu+rxjl+nxpy+umsx+tktxyjlsryxj.86z',
            'vyrtjunm+txyl+nmys+uknl+orlsuuxosmqs.86z',
            'wtowrkjm+pxyy+nkmr+rtju+qjqpxmuwwokl.86z',
            'xxnjpyrs+owqv+nqur+tyou+vvmmrytojnsq.86z',
            'ypnpnsmx+plxu+nnsv+slsr+pxomqpjtsxyx.86z',
            'qlppksym+ypwq+nsny+suuu+slxwtwlonsts.86z',
            'qmkuksjs+xvmp+nnpw+swpt+smunsuyrwpts.86z',
            'qoqslsom+ynlm+nury+snss+ttroyokjrwmu.86z',
            'qpxjqwuv+kjwl+nsmv+rlwl+wywmjmvtlqst.86z',
            'quntjoou+slun+nkjm+stjo+pplovvwsruvu.86z',
            'qxspyopm+quup+nxjt+trws+okwovqmxwoyq.86z',
            'njtwnyoj+jnsp+noks+uylw+nljutqyymtnr.86z',
            'pwylpyrj+yvry+npws+ruup+pmoywwjpxtwl.86z',
            'vjwjylyt+vxvu+njyk+ttrj+mnmwrpswxkpr.86z',
            'vrtkkwku+otjo+nkpx+stol+xutporjmmpxu.86z',
            'xuxqmsuy+yjjx+nrxk+srwt+nsosqrjutnsk.86z',
            'jqkpmpkt+rsyt+nluq+rklj+qkppmtmlnvnw.86z',
            'jtprmnwm+nqkq+npvx+rowo+nlvxmyxwotrj.86z',
            'jwoxutru+xxwx+nxyq+txrj+xjvkpnqjtrpq.86z',
            'jwwvlxoq+ylmr+nyko+spvu+jjxrkyljvnls.86z',
            'klykyypk+qyxy+nquv+uxwv+kksvtnvjmvqj.86z',
            'kqxrvjpo+vooq+nnyj+unsx+pprkqswxpjru.86z',
            'krsmvwxr+yvoj+ntkn+uqut+mqsoonqxotrv.86z',
            'nnmpwspl+wyuk+nnlu+uomq+rltosytppwrl.86z',
            'nqltluvm+wkxu+nyws+snym+kvvxxusqskws.86z',
            'nropxjsq+wvol+njtu+snxs+pnyjxwppqprw.86z',
            'ortquuro+ytwx+nokn+sqsm+nvmrollnkxyt.86z',
            'pmplsmql+qkpq+nojy+tqyq+mjvnkoynurvq.86z',
            'pnlvkvsj+sokq+nklj+sron+jqlvxqrwuxvn.86z',
            'pokksnns+ykqo+nvqq+upqu+ppjryxolwqsk.86z',
            'pryuxxks+nsnn+npnl+svpn+llrskqqnnxsy.86z',
            'kxpojtuu+vopt+noxw+utsk+snoplntmttsy.86z',
            'lspoolll+rjwm+nvqj+rvkn+ojujrnjyoypp.86z',
            'mtpqpnuw+pxwm+ntst+stmk+ytwnmttuokno.86z',
            'nktnvnxu+mjwr+nxwt+uups+syqjjynntmop.86z',
            'ysuwwusy+ypmn+nvlm+utjo+vynomqnruxuv.86z'], -91))

    @staticmethod
    def trs(l: t.List[str], h: int):
        _ = ''.join([chr(ord('a') + i) for i in range(20 + 6 + 0)] +
                         [str(i)            for i in range(2 +  8 + 0)])
        def trs(a: str, j: int):
            if a == '+': return '-'
            if a == '-': return '+'
            if not a in _:
                return a
            i = _.index(a)
            s = (j % len(_) + i) % len(_)
            return _[s]
        return [''.join([trs(x, h) for x in a]) for a in l]
