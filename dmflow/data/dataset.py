import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import os
from torch_geometric.data import Data
import pickle
import numpy as np
import chemparse
from diffcsp.common.utils import PROJECT_ROOT
from dmflow.data.disorder_utils import (
    DisorderRecorder,
    preprocess_disorder,
    recorders_to_vecs,
)

chemical_symbols = [
    # 0
    "X",
    # 1
    "H",
    "He",
    # 2
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    # 3
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    # 4
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    # 5
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    # 6
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    # 7
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


class SampleDataset(Dataset):
    def __init__(self, formula, num_evals):
        super().__init__()
        self.formula = formula
        self.num_evals = num_evals
        self.get_structure()

    def get_structure(self):
        self.composition = chemparse.parse_formula(self.formula)
        chem_list = []
        for elem in self.composition:
            num_int = int(self.composition[elem])
            chem_list.extend([chemical_symbols.index(elem)] * num_int)
        self.chem_list = chem_list

    def __len__(self) -> int:
        return self.num_evals

    def __getitem__(self, index):
        return Data(
            atom_types=torch.LongTensor(self.chem_list),
            num_atoms=len(self.chem_list),
            num_nodes=len(self.chem_list),
        )


train_dist = {
    "perov_5": [0, 0, 0, 0, 0, 1],
    "carbon_24": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3250697750779839,
        0.0,
        0.27795107535708424,
        0.0,
        0.15383352487276308,
        0.0,
        0.11246100804465604,
        0.0,
        0.04958134953209654,
        0.0,
        0.038745690362830404,
        0.0,
        0.019044491873255624,
        0.0,
        0.010178952552946971,
        0.0,
        0.007059596125430964,
        0.0,
        0.006074536200952225,
    ],
    "perov": [0, 0, 0, 0, 0, 1],
    "carbon": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3250697750779839,
        0.0,
        0.27795107535708424,
        0.0,
        0.15383352487276308,
        0.0,
        0.11246100804465604,
        0.0,
        0.04958134953209654,
        0.0,
        0.038745690362830404,
        0.0,
        0.019044491873255624,
        0.0,
        0.010178952552946971,
        0.0,
        0.007059596125430964,
        0.0,
        0.006074536200952225,
    ],
    "mp_20": [
        0.0,
        0.0021742334905660377,
        0.021079009433962265,
        0.019826061320754717,
        0.15271226415094338,
        0.047132959905660375,
        0.08464770047169812,
        0.021079009433962265,
        0.07808814858490566,
        0.03434551886792453,
        0.0972877358490566,
        0.013303360849056603,
        0.09669811320754718,
        0.02155807783018868,
        0.06522700471698113,
        0.014372051886792452,
        0.06703272405660378,
        0.00972877358490566,
        0.053176591981132074,
        0.010576356132075472,
        0.08995430424528301,
    ],
    "cod_sd": [
        0.0,
        0.0,
        0.062259483232545355,
        0.010032985156679494,
        0.015667949422759758,
        0.032297965915338095,
        0.024189114898295765,
        0.014431006047278724,
        0.01937877954920286,
        0.011819681143485432,
        0.05415063221550302,
        0.004672897196261682,
        0.033534909290819134,
        0.012369433754810335,
        0.16602528862012095,
        0.005360087960417812,
        0.024189114898295765,
        0.0072842221000549755,
        0.027899945024738866,
        0.0063221550302363936,
        0.09043430456294667,
        0.0030236393622869707,
        0.02075316107751512,
        0.006871907641561297,
        0.023227047828477187,
        0.0015118196811434853,
        0.03532160527762507,
        0.0013743815283122594,
        0.060472787245739415,
        0.0034359538207806486,
        0.014980758658603628,
        0.00041231445849367786,
        0.016355140186915886,
        0.0021990104452996153,
        0.008658603628367234,
        0.0015118196811434853,
        0.03559648158328752,
        0.0009620670698185816,
        0.013194062671797692,
        0.002473886750962067,
        0.026525563496426607,
        0.003710830126443101,
        0.028862012094557448,
        0.0016492578339747114,
        0.022127542605827378,
        0.0009620670698185816,
        0.010582737768004398,
        0.0006871907641561297,
        0.014431006047278724,
        0.002061572292468389,
        0.013743815283122594,
    ],
    "cod_pd": [
        0.0,
        0.0,
        0.0,
        0.002830188679245283,
        0.002830188679245283,
        0.0033018867924528303,
        0.005660377358490566,
        0.014150943396226415,
        0.00990566037735849,
        0.010377358490566037,
        0.010377358490566037,
        0.0047169811320754715,
        0.010849056603773584,
        0.02122641509433962,
        0.016037735849056604,
        0.007075471698113208,
        0.02688679245283019,
        0.007547169811320755,
        0.018867924528301886,
        0.011320754716981131,
        0.028773584905660378,
        0.014150943396226415,
        0.03018867924528302,
        0.022169811320754716,
        0.02311320754716981,
        0.007075471698113208,
        0.019339622641509433,
        0.013207547169811321,
        0.04150943396226415,
        0.008962264150943396,
        0.033962264150943396,
        0.005188679245283019,
        0.03349056603773585,
        0.008018867924528302,
        0.025943396226415096,
        0.011320754716981131,
        0.049528301886792456,
        0.010377358490566037,
        0.04622641509433962,
        0.013679245283018868,
        0.051415094339622644,
        0.00990566037735849,
        0.05471698113207547,
        0.009433962264150943,
        0.06556603773584906,
        0.016037735849056604,
        0.038679245283018866,
        0.012264150943396227,
        0.05660377358490566,
        0.005188679245283019,
        0.05,
    ],
    "cod_spd": [
        0.0,
        0.0,
        0.04821200510855683,
        0.008407833120476799,
        0.01277139208173691,
        0.025755640698169435,
        0.020008514261387826,
        0.014367816091954023,
        0.017241379310344827,
        0.011494252873563218,
        0.04427415921668795,
        0.004682843763303533,
        0.028416347381864625,
        0.014367816091954023,
        0.13218390804597702,
        0.005747126436781609,
        0.024797786292039166,
        0.007343550446998723,
        0.02586206896551724,
        0.00744997871434653,
        0.07641549595572585,
        0.005640698169433802,
        0.02288207747977863,
        0.010323541932737336,
        0.02320136228182205,
        0.002767134951042997,
        0.031715623669646656,
        0.004044274159216688,
        0.0561941251596424,
        0.004682843763303533,
        0.019263516389953172,
        0.001489995742869306,
        0.02022137079608344,
        0.00351213282247765,
        0.012558535547041293,
        0.003724989357173265,
        0.038739889314601955,
        0.0030864197530864196,
        0.02064708386547467,
        0.0050021285653469565,
        0.032141336739037886,
        0.005108556832694764,
        0.03469561515538527,
        0.0034057045551298426,
        0.031928480204342274,
        0.004363558961260111,
        0.016922094508301407,
        0.003299276287782035,
        0.023946360153256706,
        0.002767134951042997,
        0.02192422307364836,
    ],
    "cod_sd_20": [
        0.0,
        0.0,
        0.09868421052631579,
        0.017105263157894738,
        0.027192982456140352,
        0.05350877192982456,
        0.03793859649122807,
        0.022587719298245615,
        0.030921052631578946,
        0.019736842105263157,
        0.08552631578947369,
        0.007456140350877193,
        0.05482456140350877,
        0.019078947368421053,
        0.26337719298245615,
        0.008333333333333333,
        0.04013157894736842,
        0.009649122807017544,
        0.04671052631578947,
        0.01074561403508772,
        0.14649122807017545,
    ],
    "cod_pd_20": [
        0.0,
        0.0,
        0.0,
        0.014705882352941176,
        0.023529411764705882,
        0.01764705882352941,
        0.029411764705882353,
        0.07352941176470588,
        0.05588235294117647,
        0.07058823529411765,
        0.05588235294117647,
        0.03529411764705882,
        0.08529411764705883,
        0.12352941176470589,
        0.07941176470588235,
        0.047058823529411764,
        0.1411764705882353,
        0.04411764705882353,
        0.07352941176470588,
        0.029411764705882353,
    ],
    "cod_spd_20": [
        0.0,
        0.0,
        0.09183673469387756,
        0.01693877551020408,
        0.026938775510204082,
        0.05102040816326531,
        0.0373469387755102,
        0.026122448979591838,
        0.0326530612244898,
        0.02326530612244898,
        0.08346938775510204,
        0.009387755102040816,
        0.056938775510204084,
        0.026326530612244898,
        0.2506122448979592,
        0.011020408163265306,
        0.047142857142857146,
        0.012040816326530613,
        0.04857142857142857,
        0.012040816326530613,
        0.1363265306122449,
    ],
    "cod_sd_20_aug": [
        0.0,
        0.0,
        0.032880177047107176,
        0.02007587733164717,
        0.13597850142269996,
        0.049193803351248816,
        0.07907050268732216,
        0.022099272842238383,
        0.0725576983876067,
        0.03303825482137211,
        0.09889345558014544,
        0.012488144166930129,
        0.08925071134998419,
        0.021245652861207713,
        0.09317104015175466,
        0.013499841922225735,
        0.0605121719886184,
        0.009737590894720202,
        0.05150173885551691,
        0.010622826430603857,
        0.09418273790705027,
    ],
    "cod_pd_20_aug": [
        0.0,
        0.0,
        0.021524990879241153,
        0.02050346588836191,
        0.15268150310105802,
        0.048084640642101426,
        0.08529733673841663,
        0.02265596497628603,
        0.07927763589930682,
        0.0357168916453849,
        0.10058372856621671,
        0.013608172199927033,
        0.09492885808099234,
        0.022874863188617294,
        0.0646844217438891,
        0.014775629332360452,
        0.06490331995622035,
        0.010178766873403867,
        0.05257205399489238,
        0.010835461510397664,
        0.08431229478292594,
    ],
    "cod_spd_20_aug": [
        0.0,
        0.0,
        0.03253049734125743,
        0.020018767594619957,
        0.13478260869565217,
        0.04885830466061933,
        0.07854238348451674,
        0.022646230841413824,
        0.07238035658429778,
        0.033437597747888646,
        0.09843603378167032,
        0.012730685017203628,
        0.08920863309352518,
        0.022333437597747887,
        0.09302471066624961,
        0.013856740694401001,
        0.0613700344072568,
        0.01010322177040976,
        0.05173600250234595,
        0.010822646230841414,
        0.09318110728808257,
    ],
    "cod_spd_aug": [
        0.0,
        0.0,
        0.0365890618542499,
        0.020829315332690453,
        0.0429811268769803,
        0.044772007163521144,
        0.056784681085548974,
        0.01672406667585067,
        0.04380768700922992,
        0.024493731918997105,
        0.04108003857280617,
        0.007163521146163383,
        0.06144096983055517,
        0.014630114340818295,
        0.0633420581347293,
        0.007879873260779722,
        0.04355971896955504,
        0.0066951370712219314,
        0.031078660972585756,
        0.00548284887725582,
        0.05259677641548423,
        0.005593056894889103,
        0.026449924231987876,
        0.004463424714147954,
        0.04289847086375534,
        0.0015704642512742802,
        0.01727510676401708,
        0.0031409285025485604,
        0.0463149194103871,
        0.00821049731367957,
        0.015126050420168067,
        0.0008816641410662626,
        0.026752996280479405,
        0.002149056343849015,
        0.007990081278413004,
        0.001873536299765808,
        0.03771869403499105,
        0.0018459842953574873,
        0.014189282270285163,
        0.0025623364099738254,
        0.029893924783027964,
        0.002038848326215732,
        0.015594434495109518,
        0.0011847361895577903,
        0.020884419341507095,
        0.0018459842953574873,
        0.009450337512054001,
        0.0011847361895577903,
        0.019589475134316022,
        0.0012949442071910732,
        0.008100289296046287,
    ],
    "cod_sd_aug": [
        0.0,
        0.0,
        0.03885881492318947,
        0.021945866861741038,
        0.045471836137527435,
        0.047344550109729336,
        0.060073152889539135,
        0.016766642282370153,
        0.045910753474762256,
        0.02536942209217264,
        0.04298463789319678,
        0.0073152889539136795,
        0.06457937088514996,
        0.014220921726408193,
        0.06627651792245794,
        0.007929773226042429,
        0.04459400146305779,
        0.006642282370153621,
        0.031836137527432334,
        0.005120702267739576,
        0.05404535479151427,
        0.005091441111923921,
        0.026217995610826626,
        0.003365032918800293,
        0.04412582297000731,
        0.0012289685442574982,
        0.017147037307973664,
        0.002516459400146306,
        0.04661302121433797,
        0.008163862472567666,
        0.013957571324067301,
        0.0006144842721287491,
        0.026335040234089245,
        0.0017849305047549378,
        0.006876371616678859,
        0.0012874908558888076,
        0.03698610095098757,
        0.0013167520117044623,
        0.012201901975128018,
        0.001872713972201902,
        0.028558888076079005,
        0.0015508412582297,
        0.013167520117044623,
        0.0006730065837600585,
        0.01811265544989027,
        0.0009656181419166058,
        0.007637161667885881,
        0.0004974396488661302,
        0.017293343087051938,
        0.0010534016093635698,
        0.005501097293343087,
    ],
}


class GenDataset(Dataset):
    def __init__(self, dataset, total_num):
        super().__init__()
        self.total_num = total_num
        self.distribution = train_dist[dataset]
        self.num_atoms = np.random.choice(
            len(self.distribution), total_num, p=self.distribution
        )
        self.is_carbon = dataset == "carbon_24"

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):

        num_atom = self.num_atoms[index]
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
        )
        if self.is_carbon:
            data.atom_types = torch.LongTensor([6] * num_atom)
        return data


class DisorderedCrystDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        prop: ValueNode,
        niggli: ValueNode,
        primitive: ValueNode,
        graph_method: ValueNode,
        preprocess_workers: ValueNode,
        lattice_scale_method: ValueNode,
        save_path: ValueNode,
        tolerance: ValueNode,
        use_space_group: ValueNode,
        use_pos_index: ValueNode,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance

        self.preprocess(save_path, preprocess_workers, prop)

        ## unused
        # self.lattice_scale_method = lattice_scale_method
        # add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.scaler = None

    def preprocess(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess_disorder(
                self.path,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=[prop],
                use_space_group=self.use_space_group,
                tol=self.tolerance,
            )
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
        ) = data_dict["graph_arrays"]

        # info is a triplet of (atom_index, sd_info, pd_info)
        disorder_recs = [DisorderRecorder(*info) for info in data_dict["disorder_info"]]

        ## (num_atoms, MAX_ATOM_NUM), (num_atoms, MAX_ATOM_NUM), (num_atoms, 3), (num_atoms, 2)
        sd_onehot, sd_weights, pd_weights, pd_fracs, pd_onehot = recorders_to_vecs(
            disorder_recs
        )

        disorder_types = torch.LongTensor(
            [rec.get_disorder_type() for rec in disorder_recs]
        )

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T
            ).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
            sd_weights=sd_weights,
            pd_weights=pd_weights,
            pd_frac_coords=pd_fracs,
            sd_onehot=sd_onehot,
            pd_onehot=pd_onehot,
            disorder_types=disorder_types.view(-1, 1),
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])
            data.ops = torch.Tensor(data_dict["wyckoff_ops"])
            data.anchor_index = torch.LongTensor(data_dict["anchors"])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=}), size={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from diffcsp.common.data_utils import get_scaler_from_data_list

    dataset: DisorderedCrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    scaler = get_scaler_from_data_list(dataset.cached_data, key=dataset.prop)

    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
