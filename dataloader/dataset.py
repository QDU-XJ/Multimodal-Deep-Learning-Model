import glob
import os
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from utils.print_utils import print_info_message, print_log_message
from dataloader.utils import *


class BLYXDataset(Dataset):
    """
    病理影像数据集
    """
    def __init__(self, opts, split: str, split_file: str, printer):
        super().__init__()
        self.opts = opts
        self.split = split
        self.is_training = (split == "train")
        self.printer = printer

        # 病理图参数
        self.bl_img_dir = opts.bl_img_dir
        self.bl_num_bags = opts.bl_num_bags
        self.bl_bag_size = opts.bl_bag_size
        self.bl_img_extn = opts.bl_img_extn
        self.bl_word_size = opts.bl_word_size

        # 影像参数
        self.yx_img_dir = opts.yx_img_dir
        self.yx_rad_dir = opts.yx_rad_dir
        self.yx_img_extn = opts.yx_img_extn
        self.yx_num_lesions = opts.yx_num_lesions
        self.yx_lesion_size = opts.yx_lesion_size

        # 读取样本表
        df = pd.read_excel(split_file)

        # 初始化所有列表
        self.id_lst = []
        self.name_lst = []
        self.sex_lst = []
        self.age_lst = []
        self.yx_pid_lst = []
        self.bl_pid_lst = []
        self.pos_lesion_lst = []
        self.form_lst = []
        self.size_lst = []
        self.thickness_lst = []
        self.ca_lst = []
        self.num_lst = []
        self.stalk_lst = []
        self.border_lst = []
        self.necrosis_lst = []
        self.extra_in_lst = []
        self.lymph_node_lst = []
        self.CT_A_lst = []
        self.CT_V_lst = []
        self.CT_D_lst = []
        self.R_lst = []
        self.RFS_lst = []
        self.RFSCensor_lst = []
        self.bl_flag_lst = []
        self.yx_flag_lst = []
        self.label_lst = []

        self.qfjc_lst, self.grade_lst, self.AJCC_lst = [], [], []
        self.T_lst, self.M_lst, self.N_lst = [], [], []
        self.Tis_lst = []

        self.Urine_PH_lst, self.NLR_lst, self.Urinary_infection_lst = [], [], []
        self.Hematuria_lst, self.Proteinuria_lst = [], []

        # 计数
        self.bl_num, self.yx_num, self.blyx_num = 0, 0, 0

        # 遍历每行，筛选并填充
        for _, row in df.iterrows():
            pid = row["编号"]
            name = row["姓名"]
            sex = row["性别"]
            age = row["年龄"]
            yx_pid = row["影像号"]
            bl_pid = row["病理号"]
            pos_lesion = row["病灶位置"]
            form = row["形态"]
            size = row["大小（轴位最大径）"]
            thickness = row["膀胱壁厚度"]
            ca = row["钙化"]
            num = row["数目"]
            stalk = row["蒂"]
            border = row["边界"]
            necrosis = row["囊变坏死"]
            extra_in = row["壁外浸润"]
            lymph_node = row["区域淋巴结转移"]
            CT_A = row["CT_A"]
            CT_V = row["CT_V"]
            CT_D = row["CT_D"]

            qfjc = row["是否侵犯肌层"]
            grade = row["组织学分级"]
            AJCC = row["AJCC"]
            T = row["T分期"]
            M = row["M分期"]
            N = row["N分期"]
            Tis = row["是否存在原位癌Tis"]

            Urine_PH = row["尿pH"]
            NLR = row["NLR"]
            Urinary_infection = row["尿路感染Urinary infection"]
            Hematuria = row["血尿Hematuria"]
            Proteinuria = row["蛋白尿Proteinuria"]

            R = row["有无复发"]
            RFS = row["RFS"]
            RFSCensor = row["RFSCensor"]
            yx_flag = row["影像标志"]
            bl_flag = row["病理标志"]

            # 转换标签
            label = convert_label(R)

            # 过滤：无效 PID、标志、模型需求
            if label < -1 or yx_flag == -1 or bl_flag == -1:
                continue

            # 添加基础信息
            self.id_lst.append(pid)
            self.name_lst.append(name)
            self.sex_lst.append(sex)
            self.age_lst.append(age)
            self.yx_pid_lst.append(str(int(yx_pid)) if not pd.isna(yx_pid) else str(np.nan))
            self.bl_pid_lst.append(str(bl_pid).replace(".0", "") if bl_pid != str(np.nan) else [str(np.nan),])

            self.pos_lesion_lst.append(pos_lesion)
            self.form_lst.append(form)
            self.size_lst.append(size)
            self.thickness_lst.append(thickness)
            self.ca_lst.append(ca)
            self.num_lst.append(num)
            self.stalk_lst.append(stalk)
            self.border_lst.append(border)
            self.necrosis_lst.append(necrosis)
            self.extra_in_lst.append(extra_in)
            self.lymph_node_lst.append(lymph_node)
            self.CT_A_lst.append(CT_A)
            self.CT_V_lst.append(CT_V)
            self.CT_D_lst.append(CT_D)

            self.qfjc_lst.append(qfjc)
            self.grade_lst.append(grade)
            self.AJCC_lst.append(AJCC)
            self.T_lst.append(T)
            self.M_lst.append(M)
            self.N_lst.append(N)
            self.Tis_lst.append(Tis)
            self.R_lst.append(R)
            self.RFS_lst.append(RFS)
            self.RFSCensor_lst.append(RFSCensor)
            self.yx_flag_lst.append(yx_flag)
            self.bl_flag_lst.append(bl_flag)
            self.label_lst.append(label)

            self.Urine_PH_lst.append(Urine_PH)
            self.NLR_lst.append(NLR)
            self.Urinary_infection_lst.append(Urinary_infection)
            self.Hematuria_lst.append(Hematuria)
            self.Proteinuria_lst.append(Proteinuria)

            # 更新计数
            self.yx_num += int(not pd.isna(yx_pid))
            self.bl_num += int(not pd.isna(bl_pid))
            self.blyx_num += int(str(bl_pid) != str(np.nan) and (str(yx_pid) != str(np.nan)))

        # 标签去重统计
        self.diag_labels = deepcopy(self.label_lst)
        self.n_classes = len(np.unique(self.diag_labels))

        # 打印数据集信息
        self._print_summary()

    def _print_summary(self):
        total = len(self)
        print_info_message(
            f"Samples in {self.split}: {total} ( yx={self.yx_num})",
            self.printer
        )
        cnt0 = sum(np.array(self.label_lst) == 0)
        cnt1 = sum(np.array(self.label_lst) == 1)
        print_info_message(
            f"-- 0 (non-relapse): {cnt0} ({cnt0 / total:.2%}), "
            f"1 (relapse): {cnt1} ({cnt1 / total:.2%})",
            self.printer
        )
        # # 随机选一个索引，快速看下加载 shapes
        # sample_idx = min(5, total - 1)
        # lesions, yx_rad, lesions_label, yx_flag, roi_masks = self._load_yx_data(sample_idx)
        # print_info_message(
        #     f"[DEBUG] sample {sample_idx} → lesions.shape={tuple(lesions.shape)}, "
        #     f"roi_masks.shape={tuple(roi_masks.shape)}, "
        #     f"rad.shape={tuple(yx_rad.shape)}, labels.shape={tuple(lesions_label.shape)}",
        #     self.printer
        # )

    def __len__(self):
        return len(self.id_lst)

    def __getitem__(self, index: int) -> dict:
        # 加载影像和ROI掩码
        lesions, yx_rad, lesions_label, yx_flag, roi_masks = self._load_yx_data(index)
        feat_words, mask_bags_label, mask_words_label, bl_flag, has_path_img,roi_masks_bl = self._load_bl_data(index)

        assert bl_flag or yx_flag, (self.id_lst[index], self.yx_pid_lst[index], self.bl_pid_lst[index])

        def debug_shapes(sample, idx=None):
            """
            打印单个样本中每个 key 的 shape / type，方便定位哪个字段尺寸不一致。
            sample: dict，来自 Dataset.__getitem__ 返回的 dict
            idx: 可选，样本索引（用来定位是哪一个样本）
            """
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    print(f"idx={idx} key={k} shape={tuple(v.shape)} dtype={v.dtype}")
                else:
                    # 尝试把非 tensor 转为 tensor 再看 shape（若不可行就打印类型）
                    try:
                        a = torch.as_tensor(v)
                        print(f"idx={idx} key={k} (converted) shape={tuple(a.shape)} dtype={a.dtype}")
                    except Exception as e:
                        print(f"idx={idx} key={k} type={type(v)} (not tensor) -> {e}")

        sample = {
            "id": self.id_lst[index],
            "name": self.name_lst[index],
            "label": self.label_lst[index],
            "R": self.R_lst[index],
            "RFS": self.RFS_lst[index],
            "RFSCensor": self.RFSCensor_lst[index],

            # 影像
            "lesions": lesions,
            "roi_mask": roi_masks,  # 新增：ROI掩码
            "lesions_label": lesions_label,
            "yx_radiomics_feat": yx_rad,
            "yx_pid": self.yx_pid_lst[index],
            "yx_flag": yx_flag,

            #病理
            "feat_words": feat_words,
            "roi_mask_bl": roi_masks_bl,  # 新增：ROI掩码
            "mask_bags_label": mask_bags_label,
            "mask_words_label": mask_words_label,
            "bl_pid": self.bl_pid_lst[index],
            "bl_flag": bl_flag,

            # 临床 one-hot
            "clinical_sex": ccd_sex(self.sex_lst[index]),
            "clinical_age": ccd_age(self.age_lst[index]),
            "clinical_yx_flag": ccd_yx_flag(self.yx_flag_lst[index]),
            "clinical_yx_pos_lesion": ccd_yx_pos_lesion(self.pos_lesion_lst[index]),
            "clinical_yx_form": ccd_yx_form(self.form_lst[index]),
            "clinical_yx_ca": ccd_yx_ca(self.ca_lst[index]),
            "clinical_yx_stalk": ccd_yx_stalk(self.stalk_lst[index]),
            "clinical_yx_num": ccd_yx_num(self.num_lst[index]),
            "clinical_yx_size": ccd_yx_size(self.size_lst[index]),
            "clinical_yx_thickness": ccd_yx_thickness(self.thickness_lst[index]),
            "clinical_yx_border": ccd_yx_border(self.border_lst[index]),
            "clinical_yx_necrosis": ccd_yx_necrosis(self.necrosis_lst[index]),
            "clinical_yx_extra_in": ccd_yx_extra_in(self.extra_in_lst[index]),
            "clinical_yx_lymph_node": ccd_yx_lymph_node(self.lymph_node_lst[index]),
            "clinical_yx_CT_A": ccd_yx_CT_A(self.CT_A_lst[index]),
            "clinical_yx_CT_V": ccd_yx_CT_V(self.CT_V_lst[index]),
            "clinical_yx_CT_D": ccd_yx_CT_D(self.CT_D_lst[index]),

            "clinical_bl_qfjc": ccd_bl_qfjc(self.qfjc_lst[index]),  # (2,)
            "clinical_bl_grade": ccd_bl_grade(self.grade_lst[index]),  # (3,)
            "clinical_bl_AJCC": ccd_bl_AJCC(self.AJCC_lst[index]),  # (6,)
            "clinical_bl_Tis": ccd_bl_Tis(self.Tis_lst[index]),  # (2,)
            "clinical_bl_T": ccd_bl_T(self.T_lst[index]),  # (6,)
            "clinical_bl_M": ccd_bl_M(self.M_lst[index]),  # (2,)
            "clinical_bl_N": ccd_bl_N(self.N_lst[index]),  # (3,)
            "clinical_bl_flag": ccd_bl_flag(self.bl_flag_lst[index]),  # (3,)

            "clinical_Urine_PH": ccd_Urine_PH(self.Urine_PH_lst[index]),  # (1,)
            "clinical_NLR": ccd_NLR(self.NLR_lst[index]),  # (1,)
            "clinical_Urinary_infection": ccd_Urinary_infection(self.Urinary_infection_lst[index]),  # (2,)
            "clinical_Hematuria": ccd_Hematuria(self.Hematuria_lst[index]),  # (7,)
            "clinical_Proteinuria": ccd_Proteinuria(self.Proteinuria_lst[index]),  # (5,)

        }

        # 调试：打印 shapes（开发完成后记得删掉或注释）
        # debug_shapes(sample, id)

        return {
            "id": self.id_lst[index],
            "name": self.name_lst[index],
            "label": self.label_lst[index],
            "R": self.R_lst[index],
            "RFS": self.RFS_lst[index],
            "RFSCensor": self.RFSCensor_lst[index],

            # 影像
            "lesions": lesions,
            "roi_mask": roi_masks,  # 新增：ROI掩码
            "lesions_label": lesions_label,
            "yx_radiomics_feat": yx_rad,
            "yx_pid": self.yx_pid_lst[index],
            "yx_flag": yx_flag,

            #病理
            "feat_words": feat_words,
            "roi_masks_bl": roi_masks_bl,  # 新增：ROI掩码
            "mask_bags_label": mask_bags_label,
            "mask_words_label": mask_words_label,
            "bl_pid": self.bl_pid_lst[index],
            "bl_flag": bl_flag,

            # 临床 one-hot
            "clinical_sex": ccd_sex(self.sex_lst[index]),
            "clinical_age": ccd_age(self.age_lst[index]),
            "clinical_yx_flag": ccd_yx_flag(self.yx_flag_lst[index]),
            "clinical_yx_pos_lesion": ccd_yx_pos_lesion(self.pos_lesion_lst[index]),
            "clinical_yx_form": ccd_yx_form(self.form_lst[index]),
            "clinical_yx_ca": ccd_yx_ca(self.ca_lst[index]),
            "clinical_yx_stalk": ccd_yx_stalk(self.stalk_lst[index]),
            "clinical_yx_num": ccd_yx_num(self.num_lst[index]),
            "clinical_yx_size": ccd_yx_size(self.size_lst[index]),
            "clinical_yx_thickness": ccd_yx_thickness(self.thickness_lst[index]),
            "clinical_yx_border": ccd_yx_border(self.border_lst[index]),
            "clinical_yx_necrosis": ccd_yx_necrosis(self.necrosis_lst[index]),
            "clinical_yx_extra_in": ccd_yx_extra_in(self.extra_in_lst[index]),
            "clinical_yx_lymph_node": ccd_yx_lymph_node(self.lymph_node_lst[index]),
            "clinical_yx_CT_A": ccd_yx_CT_A(self.CT_A_lst[index]),
            "clinical_yx_CT_V": ccd_yx_CT_V(self.CT_V_lst[index]),
            "clinical_yx_CT_D": ccd_yx_CT_D(self.CT_D_lst[index]),

            "clinical_bl_qfjc": ccd_bl_qfjc(self.qfjc_lst[index]),  # (2,)
            "clinical_bl_grade": ccd_bl_grade(self.grade_lst[index]),  # (3,)
            "clinical_bl_AJCC": ccd_bl_AJCC(self.AJCC_lst[index]),  # (6,)
            "clinical_bl_Tis": ccd_bl_Tis(self.Tis_lst[index]),  # (2,)
            "clinical_bl_T": ccd_bl_T(self.T_lst[index]),  # (6,)
            "clinical_bl_M": ccd_bl_M(self.M_lst[index]),  # (2,)
            "clinical_bl_N": ccd_bl_N(self.N_lst[index]),  # (3,)
            "clinical_bl_flag": ccd_bl_flag(self.bl_flag_lst[index]),  # (3,)

            "clinical_Urine_PH": ccd_Urine_PH(self.Urine_PH_lst[index]),  # (1,)
            "clinical_NLR": ccd_NLR(self.NLR_lst[index]),  # (1,)
            "clinical_Urinary_infection": ccd_Urinary_infection(self.Urinary_infection_lst[index]),  # (2,)
            "clinical_Hematuria": ccd_Hematuria(self.Hematuria_lst[index]),  # (7,)
            "clinical_Proteinuria": ccd_Proteinuria(self.Proteinuria_lst[index]),  # (5,)

        }

    def _load_yx_data(self, index: int):
        pid = self.yx_pid_lst[index]
        try:
            # 调用修改后的 load_lesions（默认 mask_placeholder=255；如需 0，可传入）
            lesions, keys, lesions_label, roi_masks = load_lesions(
                self.yx_img_dir, pid,
                self.yx_lesion_size, self.yx_lesion_size,
                self.yx_img_extn,
                is_training=self.is_training,
                num_lesions=self.yx_num_lesions,
                split=self.split
            )

            # 加载影像组学特征
            rad_file = os.path.join(self.yx_rad_dir, f"radiomics_{pid}_norm.csv")
            rad_df = pd.read_csv(rad_file, encoding_errors='ignore').values
            rad_dict = {str(line[0]): torch.FloatTensor(np.nan_to_num(line[1:].astype(np.float32), 0.0)) for line in
                        rad_df}
            yx_rad = torch.stack([rad_dict[k] for k in keys], dim=0)

            # 返回所有数据，包括ROI掩码
            return lesions, yx_rad, lesions_label, 1, roi_masks
        except Exception as e:
            print(f"Error loading data for patient {pid}: {str(e)}")
            B = max(1, self.yx_num_lesions)
            lesions = torch.zeros(B, 3, self.yx_lesion_size, self.yx_lesion_size)
            roi_masks = torch.zeros(B, 3, self.yx_lesion_size, self.yx_lesion_size)  # 全零ROI掩码
            yx_rad = torch.zeros(B, 3, 24)
            lesions_label = torch.full((B,), -1, dtype=torch.long)
            return lesions, yx_rad, lesions_label, 0, roi_masks

    def _generate_mask_bags_label(self, mask_bags):
        # mask_bags: (N_B, B_H, B_W)
        mask_bags_label = []
        mask_bags = mask_bags.reshape((mask_bags.shape[0], -1))
        for nb in range(mask_bags.shape[0]):
            mask_bags_label.append(np.argmax(np.bincount(mask_bags[nb].astype(int))))

        mask_bags_label = torch.LongTensor(mask_bags_label) # (N_B,)
        return mask_bags_label

    def _generate_mask_words_label(self, mask_words):
        # mask_words: (N_B, N_W, W_H, W_W)
        mask_words_label = []
        mask_words = mask_words.reshape((mask_words.shape[0], mask_words.shape[1], -1))

        for nb in range(mask_words.shape[0]):
            mask_words_label_tmp = []
            for nw in range(mask_words.shape[1]):
                mask_words_label_tmp.append(np.argmax(np.bincount(mask_words[nb, nw])))
            mask_words_label.append(mask_words_label_tmp)
        mask_words_label = torch.LongTensor(mask_words_label) # (N_B, N_W)
        return mask_words_label

    def _load_bl_data(self, index):
        num_words_per_bag = (self.opts.bl_bag_size // self.opts.bl_word_size) ** 2
        bl_pid_val = self.bl_pid_lst[index]

        # 检查 PID 是否有效
        if (bl_pid_val
                and str(bl_pid_val).lower() not in ['nan', 'none', '']
                and not pd.isna(bl_pid_val)
                and isinstance(bl_pid_val, str)
                and bl_pid_val.strip()):
            bl_pid = sorted(bl_pid_val.strip().split("+"))[-1]
            path = os.path.join(self.bl_img_dir, bl_pid)

            if os.path.exists(path):
                try:
                    # 1) 加载病理图像 bags + masks
                    bags, masks, keys = load_all_bags_with_masks(
                        path,
                        self.opts.bl_bag_size, self.opts.bl_bag_size,
                        self.opts.bl_img_extn,
                        self.opts.bl_num_bags,
                        split=self.split,
                        mask_placeholder=0,  # 缺失掩码使用 0 表示无标注；需要 255 则改为 255
                        debug=True
                    )

                    if masks.max() > 1:
                        masks = (masks > 0).astype(np.uint8)  # 或根据需要 threshold

                    # 将 numpy masks 转为 torch，与空返回保持一致类型
                    if isinstance(masks, np.ndarray):
                        roi_masks_bl = torch.from_numpy(masks).float()
                    else:
                        # 如果 load_all_bags_with_masks 返回了占位 torch.Tensor(1)，直接沿用
                        roi_masks_bl = masks

                    if isinstance(bags, np.ndarray):
                        feat_words = bags_to_words(bags, self.opts.bl_word_size, self.opts.bl_word_size, bl_pid).float()
                    else:
                        # 如果没有有效图像，则创建空数据
                        feat_words = torch.randn(
                            max(1, self.opts.bl_num_bags),
                            num_words_per_bag,
                            3,
                            self.opts.bl_word_size,
                            self.opts.bl_word_size
                        ).float()

                    mask_bags_label = self._generate_mask_bags_label(
                        masks if isinstance(masks, np.ndarray) else roi_masks_bl.detach().cpu().numpy())
                    roi_masks_bl = masks_to_words(masks if isinstance(masks, np.ndarray) else roi_masks_bl.detach().cpu().numpy(),
                                                self.opts.bl_word_size, self.opts.bl_word_size, bl_pid).long()
                    mask_words_label = self._generate_mask_words_label(roi_masks_bl)
                    has_path_img = True

                    flag = 1
                    # if True:
                    #     print(
                    #         f"[DEBUG _load_bl_data] bl_pid={bl_pid} | feat_words.shape={feat_words.shape} | roi_masks_bl.shape={roi_masks_bl.shape} | mask_bags_label.shape={mask_bags_label.shape}")
                    return (
                        feat_words,
                        mask_bags_label,
                        mask_words_label,
                        flag,
                        has_path_img,
                        roi_masks_bl
                    )
                except Exception as e:
                    print(f"[Error] 加载 BL 病理数据异常：{e}")
                    traceback.print_exc()

        # 如果无效 PID 或加载异常，返回空数据
        feat_words, mask_bags_label, mask_words_label, flag0, roi_masks_bl = self._generate_empty_bl_data(
            num_words_per_bag)
        return (
            feat_words,
            mask_bags_label,
            mask_words_label,
            flag0,
            False,
            roi_masks_bl
        )

    def _generate_empty_bl_data(self, num_words_per_bag):
        seed = 42
        torch.manual_seed(seed)

        feat_words = torch.randn(
            max(1, self.opts.bl_num_bags),
            num_words_per_bag,
            3,
            self.opts.bl_word_size,
            self.opts.bl_word_size
        ).float()

        mask_bags_label = torch.full(
            (max(1, self.opts.bl_num_bags),),
            -1,
            dtype=torch.long
        )
        mask_words_label = torch.full(
            (max(1, self.opts.bl_num_bags), num_words_per_bag),
            -1,
            dtype=torch.long
        )
        # 创建空ROI掩码 (全零)
        roi_masks_bl = torch.zeros(
            (max(1, self.opts.bl_num_bags),4, self.opts.bl_word_size, self.opts.bl_word_size),
            dtype=torch.long
        )
        return feat_words, mask_bags_label, mask_words_label, 0, roi_masks_bl
